import torch
import numpy as np
from transformers import T5Config, Adafactor
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)

from osuT5.dataset import OrsDataset, OsuParser
from osuT5.model.osu_t import OsuT
from osuT5.tokenizer import Tokenizer


def get_config(args: DictConfig) -> T5Config:
    config = T5Config.from_pretrained(args.model.name)

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    tokenizer = Tokenizer(args)
    setattr(config, "vocab_size", tokenizer.vocab_size_out)
    setattr(config, "vocab_size_in", tokenizer.vocab_size_in)
    return config


def get_model(config: T5Config) -> OsuT:
    model = OsuT(config)
    return model


def get_tokenizer(args: DictConfig) -> Tokenizer:
    return Tokenizer(args)


def get_optimizer(model: OsuT, args: DictConfig) -> Optimizer:
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        from transformers import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer: Optimizer, args: DictConfig) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps - args.optim.warmup_steps,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps],
    )

    return scheduler


def get_dataloaders(tokenizer: Tokenizer, args: DictConfig) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(tokenizer)
    dataset = {
        "train": OrsDataset(
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
        ),
        "test": OrsDataset(
            args.test_dataset_path,
            args.test_dataset_start,
            args.test_dataset_end,
            args.model.spectrogram.sample_rate,
            args.model.spectrogram.hop_length,
            args.model.max_seq_len,
            args.model.max_target_len,
            parser,
            tokenizer,
            per_track=args.optim.per_track,
            class_dropout_prob=1.0,
            diff_dropout_prob=0.0,
        ),
    }

    dataloaders = {}
    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.dataloader.num_workers > 0,
            worker_init_fn=worker_init_fn,
        )

    return dataloaders["train"], dataloaders["test"]


def worker_init_fn(worker_id: int) -> None:
    """
    Give each dataloader a unique slice of the full dataset.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        np.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
