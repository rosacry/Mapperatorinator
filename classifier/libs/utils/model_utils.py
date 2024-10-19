import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)
from torch.utils.data import DataLoader

from ..dataset import OrsDataset, OsuParser
from ..model.model import OsuClassifier
from ..tokenizer import Tokenizer


def get_model(args: DictConfig, tokenizer: Tokenizer) -> OsuClassifier:
    model = OsuClassifier(args, tokenizer)
    return model


def get_tokenizer(args: DictConfig) -> Tokenizer:
    return Tokenizer(args)


def get_optimizer(parameters, args: DictConfig) -> Optimizer:
    if args.optim.name == 'adamw':
        optimizer = AdamW(
            parameters,
            lr=args.optim.base_lr,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer: Optimizer, args: DictConfig, num_processes=1) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps * num_processes,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps * num_processes - args.optim.warmup_steps * num_processes,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps * num_processes],
    )

    return scheduler


def get_dataloaders(tokenizer: Tokenizer, args: DictConfig) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(args, tokenizer)
    dataset = {
        "train": OrsDataset(
            args.data,
            parser,
            tokenizer,
        ),
        "test": OrsDataset(
            args.data,
            parser,
            tokenizer,
            test=True,
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
