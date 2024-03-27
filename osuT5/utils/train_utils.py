import os.path
import time

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from osuT5.tokenizer import Tokenizer, EventType
from osuT5.model import OsuT
from .log_utils import Averager

logger = get_logger(__name__)


def forward(model: OsuT, batch, tokenizer: Tokenizer = None):
    outputs = model(**batch)
    loss = outputs.loss

    stats = {"loss": loss.detach().float().item()}

    if tokenizer is not None:
        # Calculate accuracy metrics
        preds = torch.argmax(outputs.logits, dim=-1)
        stats["timing_acc"] = acc_range(preds, batch["labels"], tokenizer.event_start[EventType.TIME_SHIFT],
                                        tokenizer.event_end[EventType.TIME_SHIFT])
        stats["spacing_acc"] = acc_range(preds, batch["labels"], tokenizer.event_start[EventType.DISTANCE],
                                         tokenizer.event_end[EventType.DISTANCE])
        stats["other_acc"] = acc_range(preds, batch["labels"], tokenizer.event_end[EventType.DISTANCE],
                                       tokenizer.event_end[EventType.DISTANCE] + tokenizer.vocab_size_out)

    return loss, stats


def add_prefix(prefix: str, stats: dict[str, float]):
    return {f"{prefix}/{k}": v for k, v in stats.items()}


def maybe_save_checkpoint(accelerator: Accelerator, args: DictConfig):
    if (
            args.current_train_step > args.optim.total_steps
            or args.current_train_step % args.checkpoint.every_steps == 0
    ):
        if args.current_loss < args.best_loss:
            args.best_loss = args.current_loss
            is_best = True
        else:
            is_best = False

        output_dir = f"checkpoint-{args.current_train_step}"
        # Saving T5 has an issue that safe serialization removes shared tensors and then the model can't be loaded.
        accelerator.save_state(output_dir=output_dir, safe_serialization=False)

        wandb_tracker = accelerator.get_tracker("wandb")
        if wandb_tracker is not None:
            art = wandb.Artifact(
                f"osuT5-{wandb.run.id}",
                type="model",
                metadata={
                    "format": "accelerate",
                    "max_seq_len": args.model.max_seq_len,
                    "max_target_len": args.model.max_target_len,
                    "num_classes": args.data.num_classes,
                    "num_diff_classes": args.data.num_diff_classes,
                    "max_difficulty": args.data.max_diff,
                    "class_dropout_prob": args.data.class_dropout_prob,
                    "diff_dropout_prob": args.data.diff_dropout_prob,
                    "spectrogram": args.model.spectrogram,
                    "current_train_step": args.current_train_step,
                    "current_epoch": args.current_epoch,
                    "current_loss": args.current_loss,
                },
            )

            for file in os.listdir(output_dir):
                art.add_file(os.path.join(output_dir, file))

            wandb.log_artifact(art, aliases=["best"] if is_best else None)
            logger.info(f"Logged checkpoint to wandb: {art.name}")


def maybe_eval(
        model: OsuT,
        accelerator: Accelerator,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        args: DictConfig,
):
    if (
            args.current_train_step > args.optim.total_steps
            or args.current_train_step % args.eval.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            eval_model(model, accelerator, dataloader, tokenizer, args)

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
        averaged_stats["epoch"] = args.current_epoch
        averaged_stats = add_prefix("train", averaged_stats)
        accelerator.log(averaged_stats, step=args.current_train_step)
        averaged_stats["step"] = args.current_train_step
        logger.info(averaged_stats)

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
        ).item()
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


def eval_model(
        model: OsuT,
        accelerator: Accelerator,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        args: DictConfig,
):
    args.last_log = time.time()
    averager = Averager()

    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.eval.steps * args.optim.grad_acc:
            break

        # We can't use the beatmap idx of the test set because these are not known by the model
        del batch["beatmap_idx"]

        _, stats = forward(model, batch, tokenizer)
        averager.update(stats)

    averager.update({"time": time.time() - args.last_log})
    averaged_stats = averager.average()
    averaged_stats = add_prefix("test", averaged_stats)
    accelerator.log(averaged_stats, step=args.current_train_step)
    logger.info(averaged_stats)

    args.current_loss = averaged_stats["test/loss"]


def acc_range(preds, labels, start_index, end_index):
    index = (start_index <= labels) & (labels < end_index)
    range_labels = labels[index]
    range_preds = preds[index]
    return (range_preds == range_labels).detach().float().cpu().numpy()


def train(
        model: OsuT,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        accelerator: Accelerator,
        lr_scheduler: LRScheduler,
        optimizer: Optimizer,
        tokenizer: Tokenizer,
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
                maybe_eval(model, accelerator, test_dataloader, tokenizer, args)
                maybe_save_checkpoint(accelerator, args)

                args.current_train_step += 1

        args.current_epoch += 1

    maybe_eval(model, accelerator, test_dataloader, tokenizer, args)
    maybe_save_checkpoint(accelerator, args)

    accelerator.end_training()
