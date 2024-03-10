import hydra
import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from osuT5.dataset import OsuParser, OrsDataset
from osuT5.model.spectrogram import MelSpectrogram
from osuT5.utils import (
    setup_args,
    get_tokenizer,
    worker_init_fn,
)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(args: DictConfig):
    setup_args(args)

    tokenizer = get_tokenizer(args)
    parser = OsuParser(tokenizer)
    dataset = OrsDataset(
        args.train_dataset_path,
        args.train_dataset_start,
        args.train_dataset_end,
        args.model.spectrogram.sample_rate,
        args.model.spectrogram.hop_length,
        args.model.max_seq_len,
        args.model.max_target_len,
        parser,
        tokenizer,
        args.optim.cycle_length,
        True,
        args.optim.per_track,
        True,
        args.control.class_dropout_prob,
        args.control.diff_dropout_prob,
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

    # Make histogram of the lengths of the sequences
    lengths = []
    for b in tqdm.tqdm(dataloader, smoothing=0.01):
        for i in range(len(b["frames"])):  # batch size
            length = b['decoder_attention_mask'][i].sum().item()
            lengths.append(length)
        if len(lengths) > 10000:
            break

        # mels = transform(b["frames"])
        # plot the melspectrogram
        # for i in range(len(mels)):
        #     plt.imshow(mels[i].numpy().T, aspect="auto", origin="lower", norm="log")
        #     plt.show()
        # break

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

    print("learnt beatmap idx", tokenizer.beatmap_idx)

if __name__ == "__main__":
    main()
