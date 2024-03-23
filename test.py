import time

import hydra
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from omegaconf import DictConfig

from osuT5.dataset.ors_dataset import STEPS_PER_MILLISECOND, LABEL_IGNORE_ID
from osuT5.tokenizer import EventType
from osuT5.utils import (
    setup_args,
    get_config,
    get_model,
    get_tokenizer,
    get_dataloaders, eval_model, Averager, forward, add_prefix, get_optimizer, get_scheduler, acc_range,
)

logger = get_logger(__name__)

@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(args: DictConfig):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir=".", logging_dir="tensorboard_logs"
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

    setup_args(args)

    tokenizer = get_tokenizer(args)
    config = get_config(args, tokenizer)
    model = get_model(config)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args)

    # noinspection PyTypeChecker
    (
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    accelerator.register_for_checkpointing(tokenizer)

    accelerator.load_state(args.checkpoint_path)

    if args.compile:
        model = torch.compile(model)

    model.eval()

    with torch.no_grad():
        start_time = time.time()
        averager = Averager()

        max_time = 1000 * args.model.max_seq_len * args.model.spectrogram.hop_length / args.model.spectrogram.sample_rate
        n_bins = 100
        bins = np.linspace(0, max_time, n_bins + 1)[1:]
        bin_totals = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for batch_id, batch in enumerate(test_dataloader, start=1):
            if batch_id == args.eval.steps * args.optim.grad_acc:
                break

            # We can't use the beatmap idx of the test set because these are not known by the model
            del batch["beatmap_idx"]

            outputs = model(**batch)
            loss = outputs.loss

            stats = {"loss": loss.detach().float().item()}

            # Calculate accuracy metrics
            preds = torch.argmax(outputs.logits, dim=-1)
            stats["timing_acc"] = acc_range(preds, batch["labels"], tokenizer.event_start[EventType.TIME_SHIFT],
                                            tokenizer.event_end[EventType.TIME_SHIFT])
            stats["spacing_acc"] = acc_range(preds, batch["labels"], tokenizer.event_start[EventType.DISTANCE],
                                             tokenizer.event_end[EventType.DISTANCE])
            stats["other_acc"] = acc_range(preds, batch["labels"], tokenizer.event_end[EventType.DISTANCE],
                                           tokenizer.event_end[EventType.DISTANCE] + tokenizer.vocab_size_out)

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

            averager.update(stats)

        # Plot bin accuracies
        bin_accs = bin_totals / bin_counts
        bin_accs = np.nan_to_num(bin_accs)

        # Log the plot
        for i, (bin_t, bin_acc) in enumerate(zip(bins, bin_accs)):
            wandb.log({"acc_over_time": bin_acc}, step=round(bin_t))

        averager.update({"time": time.time() - start_time})
        averaged_stats = averager.average()
        averaged_stats = add_prefix("test", averaged_stats)
        accelerator.log(averaged_stats)
        logger.info(averaged_stats)


if __name__ == "__main__":
    main()
