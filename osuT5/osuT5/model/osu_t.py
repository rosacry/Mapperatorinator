from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import T5Config, T5ForConditionalGeneration, WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

from ..model.spectrogram import MelSpectrogram
from ..tokenizer import Tokenizer, EventType


LABEL_IGNORE_ID = -100


def get_backbone_model(args, tokenizer: Tokenizer):
    if args.model.name.startswith("google/t5"):
        config = T5Config.from_pretrained(args.model.name)
    elif args.model.name.startswith("openai/whisper"):
        config = WhisperConfig.from_pretrained(args.model.name)
    else:
        raise NotImplementedError

    config.vocab_size = tokenizer.vocab_size_out

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    if args.model.name.startswith("google/t5"):
        model = T5ForConditionalGeneration(config)
    elif args.model.name.startswith("openai/whisper"):
        config.num_mel_bins = config.d_model
        config.pad_token_id = tokenizer.pad_id
        config.bos_token_id = tokenizer.sos_id
        config.eos_token_id = tokenizer.eos_id
        config.max_source_positions = args.data.src_seq_len // 2
        config.max_target_positions = args.data.tgt_seq_len
        model = WhisperForConditionalGeneration(config)
    else:
        raise NotImplementedError

    return model, config.d_model


class OsuT(nn.Module):
    __slots__ = ["spectrogram", "decoder_embedder", "encoder_embedder", "transformer", "style_embedder", "num_classes"]

    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        super().__init__()

        self.transformer, d_model = get_backbone_model(args, tokenizer)
        self.num_classes = tokenizer.num_classes
        self.input_features = args.model.input_features

        self.decoder_embedder = nn.Embedding(tokenizer.vocab_size_in, d_model)
        self.decoder_embedder.weight.data.normal_(mean=0.0, std=1.0)
        # self.class_ids = Parameter(torch.full([self.num_classes + 1], -1, dtype=torch.long), requires_grad=False)

        self.spectrogram = MelSpectrogram(
            args.model.spectrogram.sample_rate, args.model.spectrogram.n_fft,
            args.model.spectrogram.n_mels, args.model.spectrogram.hop_length
        )

        self.do_style_embed = args.model.do_style_embed

        if self.do_style_embed:
            self.style_embedder = LabelEmbedder(self.num_classes, d_model)
            self.encoder_embedder = nn.Linear(args.model.spectrogram.n_mels + d_model, d_model)
            nn.init.normal_(self.style_embedder.embedding_table.weight, std=0.02)
        else:
            self.encoder_embedder = nn.Linear(args.model.spectrogram.n_mels, d_model)

        self.vocab_size_out = tokenizer.vocab_size_out
        class_weights = torch.ones(self.vocab_size_out)
        class_weights[tokenizer.event_start[EventType.TIME_SHIFT]:tokenizer.event_end[EventType.TIME_SHIFT]] = args.data.rhythm_weight
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=LABEL_IGNORE_ID)

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            beatmap_idx: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            sample_weights: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> Seq2SeqLMOutput:
        """
        frames: B x L_encoder x mel_bins, float32
        decoder_input_ids: B x L_decoder, int64
        beatmap_idx: B, int64
        beatmap_id: B, int64
        encoder_outputs: B x L_encoder x D, float32
        """
        if beatmap_idx is None:
            batch_size = frames.shape[0] if frames is not None else decoder_input_ids.shape[0]
            device = frames.device if frames is not None else decoder_input_ids.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        inputs_embeds = None
        if encoder_outputs is None:
            frames = self.spectrogram(frames)  # (N, L, M)
            if self.do_style_embed:
                style_embeds = self.style_embedder(beatmap_idx)  # (N, D)
                frames_concat = torch.concatenate((frames, style_embeds.unsqueeze(1).expand((-1, frames.shape[1], -1))), -1)
                inputs_embeds = self.encoder_embedder(frames_concat)
            else:
                inputs_embeds = self.encoder_embedder(frames)

        decoder_inputs_embeds = self.decoder_embedder(decoder_input_ids)
        if self.input_features:
            input_features = torch.swapaxes(inputs_embeds, 1, 2)
            # noinspection PyTypeChecker
            output = self.transformer.forward(input_features=input_features,
                                              decoder_inputs_embeds=decoder_inputs_embeds,
                                              encoder_outputs=encoder_outputs, **kwargs)
        else:
            output = self.transformer.forward(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
                                              encoder_outputs=encoder_outputs, labels=labels, **kwargs)
        # output = self.transformer.forward(inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids,encoder_outputs=encoder_outputs, **kwargs)

        if labels is not None:
            unreduced_loss = self.loss_fn(torch.swapaxes(output.logits, 1, -1), labels)
            if sample_weights is not None:
                unreduced_loss *= sample_weights.unsqueeze(1)
            output.loss = unreduced_loss.sum() / (labels != LABEL_IGNORE_ID).sum()

        return output


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(
            num_classes + 1,
            hidden_size,
        )

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
