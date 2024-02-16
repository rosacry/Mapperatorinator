import time

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from osuT5.model import OsuT
from .log_utils import Averager


def forward(model: OsuT, batch):
    outputs = model(**batch)
    loss = outputs.loss

    stats = {"loss": loss.detach().float().item()}

    return loss, stats


def add_prefix(prefix: str, stats: dict[str, float]):
    return {f"{prefix}/{k}": v for k, v in stats.items()}


def maybe_save_checkpoint(accelerator: Accelerator, args: DictConfig):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.checkpoint.every_steps == 0
    ):
        output_dir = f"checkpoint-{args.current_train_step}"
        accelerator.save_state(output_dir=output_dir)


def maybe_eval(
    model: OsuT,
    accelerator: Accelerator,
    dataloader: DataLoader,
    args: DictConfig,
):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.eval.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            eval(model, accelerator, dataloader, args)

        args.last_log = time.time()
        model.train()


def maybe_logging(
    model: OsuT,
    accelerator: Accelerator,
    optimizer: Optimizer,
    averager: Averager,
    args: DictConfig,
):
    def extra_stats(args, model, optimizer):
        stats = {}

        if args.logging.weights_l2:
            weights_l2 = (
                sum(p.detach().norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
            )
            stats["weights_l2"] = weights_l2

        stats["lr"] = optimizer.param_groups[0]["lr"]
        stats["seconds_per_step"] = (
            time.time() - args.last_log
        ) / args.logging.every_steps

        return stats

    if args.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, model, optimizer)

        averager.update(stats)
        averaged_stats = averager.average()
        averaged_stats = add_prefix("train", averaged_stats)
        accelerator.log(averaged_stats, step=args.current_train_step)
        averaged_stats["step"] = args.current_train_step
        print(averaged_stats)

        args.last_log = time.time()


def maybe_grad_clip_and_grad_calc(
    model: OsuT,
    accelerator: Accelerator,
    args: DictConfig,
):
    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        )
    else:
        grad_l2 = None

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                sum(
                    p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()
                )
                ** 0.5
            )

        return {"grad_l2": grad_l2}
    else:
        return {}


def eval(
    model: OsuT,
    accelerator: Accelerator,
    dataloader: DataLoader,
    args: DictConfig,
):
    args.last_log = time.time()
    averager = Averager()

    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.eval.steps * args.optim.grad_acc:
            break

        # We can't use the beatmap idx of the test set because these are not known by the model
        del batch["beatmap_idx"]

        _, stats = forward(model, batch)
        averager.update(stats)

    averager.update({"time": time.time() - args.last_log})
    averaged_stats = averager.average()
    averaged_stats = add_prefix("test", averaged_stats)
    accelerator.log(averaged_stats, step=args.current_train_step)


def train(
    model: OsuT,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    accelerator: Accelerator,
    lr_scheduler: LRScheduler,
    optimizer: Optimizer,
    args: DictConfig,
):
    model.train()

    train_averager = Averager()

    while args.current_train_step <= args.optim.total_steps:
        # In case there is a remainder from previous epoch, we need to reset the optimizer
        optimizer.zero_grad(set_to_none=True)

        print(f"Epoch {args.current_epoch}")

        for batch_id, batch in enumerate(train_dataloader, start=1):
            if args.current_train_step > args.optim.total_steps:
                break

            loss, stats = forward(model, batch)

            accelerator.backward(loss / args.optim.grad_acc)
            train_averager.update(stats)

            if batch_id % args.optim.grad_acc == 0:
                stats = maybe_grad_clip_and_grad_calc(model, accelerator, args)
                train_averager.update(stats)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                maybe_logging(model, accelerator, optimizer, train_averager, args)
                maybe_eval(model, accelerator, test_dataloader, args)
                maybe_save_checkpoint(accelerator, args)

                args.current_train_step += 1

        args.current_epoch += 1

    maybe_eval(model, accelerator, test_dataloader, args)
    maybe_save_checkpoint(accelerator, args)
