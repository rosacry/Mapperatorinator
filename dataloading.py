from pathlib import Path

import hydra
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from osuT5.dataset import OsuParser, OrsDataset
from osuT5.utils import (
    setup_args,
    get_tokenizer,
    worker_init_fn,
)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(args: DictConfig):
    setup_args(args)

    tokenizer = get_tokenizer()
    parser = OsuParser()
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

    for _ in tqdm.tqdm(dataloader, smoothing=0.01):
        pass


if __name__ == "__main__":
    main()
