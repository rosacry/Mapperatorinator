import hydra
import torch
import torchmetrics.functional
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import lightning
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput

from libs.model.model import OsuClassifier
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

    def forward(self, x):
        return self.model(x)

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

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader


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


if __name__ == "__main__":
    main()
