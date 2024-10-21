from pathlib import Path

import hydra
import torch
import torchmetrics.functional
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import lightning
from omegaconf import DictConfig
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput

from libs.model.model import OsuClassifier, OsuClassifierOutput
from libs import (
    get_model,
    get_tokenizer,
    get_scheduler,
    get_optimizer,
    get_dataloaders,
)

torch.set_float32_matmul_precision('high')


class LitOsuClassifier(lightning.LightningModule):
    def __init__(self, args: DictConfig, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model: OsuClassifier = get_model(args, tokenizer)

    def forward(self, **kwargs) -> OsuClassifierOutput:
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        output: Seq2SeqSequenceClassifierOutput = self.model(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output: Seq2SeqSequenceClassifierOutput = self.model(**batch)
        loss = output.loss
        preds = output.logits.argmax(dim=1)
        labels = batch["labels"]
        accuracy = torchmetrics.functional.accuracy(preds, labels, "multiclass", num_classes=self.args.data.num_classes)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.args)
        scheduler = get_scheduler(optimizer, self.args)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }}


def load_old_model(path: str, model: OsuClassifier):
    ckpt_path = Path(path)
    model_state = torch.load(ckpt_path / "pytorch_model.bin", weights_only=True)

    ignore_list = [
        "transformer.model.decoder.embed_tokens.weight",
        "transformer.model.decoder.embed_positions.weight",
        "decoder_embedder.weight",
        "transformer.proj_out.weight",
        "loss_fn.weight",
    ]
    fixed_model_state = {}

    for k, v in model_state.items():
        if k in ignore_list:
            continue
        if k.startswith("transformer.model."):
            fixed_model_state["transformer." + k[18:]] = v
        else:
            fixed_model_state[k] = v

    model.load_state_dict(fixed_model_state, strict=False)


@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    wandb_logger = WandbLogger(
        project="osu-classifier",
        entity="mappingtools",
        job_type="training",
        offline=args.logging.mode == "offline",
        log_model="all" if args.logging.mode == "online" else False,
    )

    tokenizer = get_tokenizer(args)
    train_dataloader, val_dataloader = get_dataloaders(tokenizer, args)

    model = LitOsuClassifier(args, tokenizer)

    if args.pretrained_path:
        load_old_model(args.pretrained_path, model.model)

    if args.compile:
        model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.checkpoint.every_steps, save_top_k=2, monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = lightning.Trainer(
        accelerator=args.device,
        precision=args.precision,
        logger=wandb_logger,
        max_steps=args.optim.total_steps,
        accumulate_grad_batches=args.optim.grad_acc,
        gradient_clip_val=args.optim.grad_clip,
        val_check_interval=args.eval.every_steps,
        log_every_n_steps=args.logging.every_steps,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("final.ckpt")


if __name__ == "__main__":
    main()
