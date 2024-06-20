import time
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from omegaconf import DictConfig
from tqdm import tqdm

from osuT5.dataset.ors_dataset import STEPS_PER_MILLISECOND, LABEL_IGNORE_ID
from osuT5.tokenizer import EventType, Tokenizer
from osuT5.utils import (
    setup_args,
    get_model,
    get_dataloaders, Averager, add_prefix, acc_range, fuzzy_acc_range,
    get_shared_training_state,
)

logger = get_logger(__name__)


def test(args: DictConfig, accelerator: Accelerator, model, tokenizer, prefix: str):
    setup_args(args)

    shared = get_shared_training_state()
    shared.current_train_step = args.optim.total_steps

    _, test_dataloader = get_dataloaders(tokenizer, args, shared)

    # noinspection PyTypeChecker
    test_dataloader = accelerator.prepare(test_dataloader)

    with torch.no_grad():
        start_time = time.time()
        averager = Averager()

        max_time = 1000 * args.data.src_seq_len * args.model.spectrogram.hop_length / args.model.spectrogram.sample_rate
        n_bins = 100
        bins = np.linspace(0, max_time, n_bins + 1)[1:]
        bin_totals = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        max_rhythm_complexity = 4
        rhythm_complexity_n_bins = 20
        rhythm_complexity_bins = np.linspace(0, max_rhythm_complexity, rhythm_complexity_n_bins + 1)[1:]
        rhythm_complexity_bin_totals = np.zeros(rhythm_complexity_n_bins)
        rhythm_complexity_bin_counts = np.zeros(rhythm_complexity_n_bins)
        fuzzy_rhythm_complexity_bin_totals = np.zeros(rhythm_complexity_n_bins)
        precision_bin_range = 3
        precision_bins = np.arange(-precision_bin_range, precision_bin_range + 1)
        precision_bin_totals = np.zeros(2 * precision_bin_range + 1)
        precision_bin_counts = np.zeros(2 * precision_bin_range + 1)

        for batch_id, batch in enumerate(tqdm(test_dataloader), start=1):
            if batch_id == args.eval.steps * args.optim.grad_acc:
                break

            rhythm_complexity: Optional[np.ndarray] = None
            if "sample_weights" in batch:
                rhythm_complexity = batch["sample_weights"].cpu().numpy()
                del batch["sample_weights"]

            # We can't use the beatmap idx of the test set because these are not known by the model
            del batch["beatmap_idx"]

            outputs = model(**batch)
            loss = outputs.loss

            stats = {"loss": loss.detach().float().item()}

            # Calculate accuracy metrics
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            stats["timing_acc"] = acc_range(preds, labels, tokenizer.event_start[EventType.TIME_SHIFT],
                                            tokenizer.event_end[EventType.TIME_SHIFT])
            stats["spacing_acc"] = acc_range(preds, labels, tokenizer.event_start[EventType.DISTANCE],
                                             tokenizer.event_end[EventType.DISTANCE])
            stats["other_acc"] = acc_range(preds, labels, tokenizer.event_end[EventType.DISTANCE],
                                           tokenizer.event_end[EventType.DISTANCE] + tokenizer.vocab_size_out)

            # Calculate timing precision histogram
            index = (tokenizer.event_start[EventType.TIME_SHIFT] <= labels) & (labels < tokenizer.event_end[EventType.TIME_SHIFT])
            range_labels = labels[index]
            range_preds = preds[index]
            timing_diffs = (range_preds - range_labels).detach().cpu().numpy()
            for i, n in enumerate(precision_bins):
                accs = timing_diffs == n
                precision_bin_totals[i] += np.sum(accs)
                precision_bin_counts[i] += len(accs)

            # Bin labels by time and calculate accuracy
            preds = preds.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            label_times = np.empty(labels.shape, dtype=np.float32)
            for j in range(len(labels)):
                t = 0
                for i, l in enumerate(labels[j]):
                    if tokenizer.event_start[EventType.TIME_SHIFT] <= l < tokenizer.event_end[EventType.TIME_SHIFT]:
                        time_event = tokenizer.decode(l)
                        t = time_event.value / STEPS_PER_MILLISECOND
                    label_times[j, i] = t

            binned_labels = np.digitize(label_times, bins)
            for i in range(n_bins):
                bin_preds = preds[binned_labels == i]
                bin_labels = labels[binned_labels == i]
                index = (bin_labels != LABEL_IGNORE_ID) & (bin_labels != tokenizer.eos_id)
                bin_preds = bin_preds[index]
                bin_labels = bin_labels[index]
                bin_totals[i] += np.sum(bin_preds == bin_labels)
                bin_counts[i] += len(bin_preds)

            # Bin timing accuracy by rhythm complexity
            if rhythm_complexity is not None:
                binned_rhythm_complexity = np.digitize(rhythm_complexity, rhythm_complexity_bins)
                for i in range(len(rhythm_complexity)):
                    sample_bin = np.clip(binned_rhythm_complexity[i], 0, n_bins - 1)
                    sample = acc_range(preds[i], labels[i], tokenizer.event_start[EventType.TIME_SHIFT],
                                       tokenizer.event_end[EventType.TIME_SHIFT])
                    fuzzy_sample = fuzzy_acc_range(preds[i], labels[i], tokenizer.event_start[EventType.TIME_SHIFT],
                                                   tokenizer.event_end[EventType.TIME_SHIFT], 2)
                    rhythm_complexity_bin_totals[sample_bin] += np.sum(sample)
                    rhythm_complexity_bin_counts[sample_bin] += len(sample)
                    fuzzy_rhythm_complexity_bin_totals[sample_bin] += np.sum(fuzzy_sample)

            averager.update(stats)

        def plot_bins(bin_totals, bin_counts, bins, y_name, x_name):
            bin_accs = bin_totals / bin_counts
            wandb.define_metric(y_name, step_metric=x_name)

            # Log the plot
            for (bin_x, bin_acc) in zip(bins, bin_accs):
                if not np.isnan(bin_acc):
                    wandb.log({y_name: bin_acc, x_name: bin_x})

        # Plot timing over rhythm complexity
        if rhythm_complexity_bin_counts.sum() > 0:
            plot_bins(rhythm_complexity_bin_totals, rhythm_complexity_bin_counts, rhythm_complexity_bins,
                      prefix + "/timing_acc_over_rhythm_complexity", "rhythm_complexity")
            plot_bins(fuzzy_rhythm_complexity_bin_totals, rhythm_complexity_bin_counts, rhythm_complexity_bins,
                      prefix + "/fuzzy_timing_acc_over_rhythm_complexity", "rhythm_complexity")

        # Plot timing precision
        plot_bins(precision_bin_totals, precision_bin_counts, precision_bins, prefix + "/timing_precision", "offset")

        # Plot bin accuracies
        plot_bins(bin_totals, bin_counts, bins, prefix + "/acc_over_time", "bin_time")

        averager.update({"time": time.time() - start_time})
        averaged_stats = averager.average()
        averaged_stats = add_prefix(prefix, averaged_stats)
        accelerator.log(averaged_stats)
        logger.info(averaged_stats)


@hydra.main(config_path="../configs/osuT5", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir="", logging_dir="tensorboard_logs"
        ),
    )
    accelerator.init_trackers(
        "osuT5",
        init_kwargs={
            "wandb": {
                "entity": "mappingtools",
                "job_type": "testing",
                "config": dict(args),
            }
        }
    )

    ckpt_path = Path(args.checkpoint_path)
    model_state = torch.load(ckpt_path / "pytorch_model.bin")
    tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl")

    tokenizer = Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    model = get_model(args, tokenizer)
    model.load_state_dict(model_state)
    model.eval()

    # noinspection PyTypeChecker
    model = accelerator.prepare(model)

    args.data.sample_weights_path = "../../../rcomplexion/rhythm_complexities.csv"
    args.data.timing_random_offset = 0
    test(args, accelerator, model, tokenizer, "test")

    args.data.timing_random_offset = 2
    test(args, accelerator, model, tokenizer, "test_noise")


if __name__ == "__main__":
    main()
