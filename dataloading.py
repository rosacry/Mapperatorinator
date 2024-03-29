import multiprocessing

import hydra
import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from osuT5.dataset import OsuParser, OrsDataset
from osuT5.dataset.ors_dataset import STEPS_PER_MILLISECOND
from osuT5.model.spectrogram import MelSpectrogram
from osuT5.tokenizer import EventType
from osuT5.utils import (
    setup_args,
    get_tokenizer,
    worker_init_fn,
)


@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    setup_args(args)

    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    tokenizer = get_tokenizer(args)
    parser = OsuParser(tokenizer)
    dataset = OrsDataset(
        args.data,
        parser,
        tokenizer,
        shared=shared,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.optim.batch_size,
        num_workers=args.dataloader.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.dataloader.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    transform = MelSpectrogram(
        args.model.spectrogram.sample_rate,
        args.model.spectrogram.n_fft,
        args.model.spectrogram.n_mels,
        args.model.spectrogram.hop_length,
    )

    if args.mode == 'benchmark':
        # Make histogram of the lengths of the sequences
        lengths = []
        diff_unks = 0
        style_unks = 0
        for b in tqdm.tqdm(dataloader, smoothing=0.01):
            # for i in range(len(b["frames"])):  # batch size
            #     length = b['decoder_attention_mask'][i].sum().item()
            #     lengths.append(length)
            #     if b['decoder_input_ids'][i][0] == tokenizer.diff_unk:
            #         diff_unks += 1
            #     if b['decoder_input_ids'][i][1] == tokenizer.style_unk:
            #         style_unks += 1
            shared.current_train_step += 1
            # if len(lengths) > 1000:
            #     break

        plt.hist(lengths, bins=100)
        plt.show()

        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {sum(lengths) / len(lengths)}")
        print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
        print(f"75th percentile: {sorted(lengths)[len(lengths) * 3 // 4]}")
        print(f"90th percentile: {sorted(lengths)[len(lengths) * 9 // 10]}")
        print(f"95th percentile: {sorted(lengths)[len(lengths) * 19 // 20]}")
        print(f"99th percentile: {sorted(lengths)[len(lengths) * 99 // 100]}")
        print(f"99.9th percentile: {sorted(lengths)[len(lengths) * 999 // 1000]}")
        print(f"99.99th percentile: {sorted(lengths)[len(lengths) * 9999 // 10000]}")
        print(f"99.999th percentile: {sorted(lengths)[len(lengths) * 99999 // 100000]}")

        print(f"Total number of sequences: {len(lengths)}")
        print(f"Total number of tokens: {sum(lengths)}")
        print(f"Total number of sequences with length 0: {lengths.count(2)}")

        print(f"Total number of diff unks: {diff_unks}")
        print(f"Total number of style unks: {style_unks}")

    if args.mode == 'plot':
        for b in tqdm.tqdm(dataloader, smoothing=0.01):
            mels = transform(b["frames"])
            # plot the melspectrogram
            for i in range(len(mels)):
                plt.imshow(mels[i].numpy().T, aspect="auto", origin="lower", norm="log")

                # plot the timing of events as vertical lines
                timings = b["decoder_input_ids"][i]
                for t in timings:
                    if tokenizer.event_start[EventType.TIME_SHIFT] <= t < tokenizer.event_end[EventType.TIME_SHIFT]:
                        time_event = tokenizer.decode(t.item())
                        x = time_event.value / STEPS_PER_MILLISECOND / 1000 * args.model.spectrogram.sample_rate / args.model.spectrogram.hop_length
                        plt.vlines(x=x, ymin=0, ymax=mels[i].shape[0], color='b')

                plt.show()
            break

if __name__ == "__main__":
    main()
