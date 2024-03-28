import multiprocessing

import numpy as np
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import open_dict, DictConfig
import hydra
import torch
import time

from osuT5.utils import (
    setup_args,
    train,
    get_model,
    get_tokenizer,
    get_scheduler,
    get_optimizer,
    get_dataloaders,
)


@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
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
                "job_type": "training",
                "config": dict(args),
            }
        }
    )

    setup_args(args)

    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    shared.current_epoch = 1
    shared.last_log = time.time()
    shared.current_loss = np.Infinity
    shared.best_loss = np.Infinity

    tokenizer = get_tokenizer(args)
    model = get_model(args.model, tokenizer)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args, shared)

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

    if args.checkpoint_path:
        accelerator.load_state(args.checkpoint_path)

    if args.compile:
        model = torch.compile(model)

    train(
        model,
        train_dataloader,
        test_dataloader,
        accelerator,
        scheduler,
        optimizer,
        tokenizer,
        args,
        shared,
    )


if __name__ == "__main__":
    main()
