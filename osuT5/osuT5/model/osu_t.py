from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import T5Config, T5ForConditionalGeneration, WhisperForConditionalGeneration, WhisperConfig, \
    PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .configuration_nwhisper import NWhisperConfig
from .modeling_nwhisper import NWhisperForConditionalGeneration
from ..model.spectrogram import MelSpectrogram
from ..tokenizer import Tokenizer, EventType

LABEL_IGNORE_ID = -100


def get_backbone_config(args, tokenizer: Tokenizer):
    if args.model.name.startswith("google/t5"):
        config = T5Config.from_pretrained(args.model.name)
    elif args.model.name.startswith("openai/whisper"):
        config = WhisperConfig.from_pretrained(args.model.name)
    elif args.model.name.startswith("olibomby/nwhisper"):
        config = NWhisperConfig.from_pretrained(args.model.config_base)
    else:
        raise NotImplementedError

    config.vocab_size = tokenizer.vocab_size_out
    config.use_cache = False

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    if isinstance(config, WhisperConfig):
        config.num_mel_bins = config.d_model
        config.pad_token_id = tokenizer.pad_id
        config.bos_token_id = tokenizer.sos_id
        config.eos_token_id = tokenizer.eos_id
        config.max_source_positions = args.data.src_seq_len // 2
        config.max_target_positions = args.data.tgt_seq_len
        config.begin_suppress_tokens = None
        config.decoder_start_token_id = tokenizer.sos_id
        config.do_sample = True
        config.forced_decoder_ids = None
        config.max_length = args.data.tgt_seq_len
        config.suppress_tokens = None
        config.top_k = 0
        if args.flash_attention:
            config._attn_implementation = "flash_attention_2"
    if isinstance(config, NWhisperConfig):
        config.input_vocab_size = tokenizer.vocab_size_in

    return config


def get_backbone_model(config):
    if isinstance(config, T5Config):
        model = T5ForConditionalGeneration(config)
    elif isinstance(config, NWhisperConfig):
        model = NWhisperForConditionalGeneration(config)
    elif isinstance(config, WhisperConfig):
        model = WhisperForConditionalGeneration(config)
    else:
        raise NotImplementedError

    return model


class OsuT(PreTrainedModel):
    __slots__ = ["spectrogram", "decoder_embedder", "encoder_embedder", "transformer", "style_embedder", "num_classes"]
    config_class = WhisperConfig
    base_model_prefix = "model"
    main_input_name = "frames"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        config = get_backbone_config(args, tokenizer)
        d_model = config.d_model

        super().__init__(config)

        self.transformer: WhisperForConditionalGeneration = get_backbone_model(config)

        self.num_classes = tokenizer.num_classes
        self.input_features = args.model.input_features
        self.embed_decoder_input = args.model.embed_decoder_input

        if self.embed_decoder_input:
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
            decoder_attention_mask: Optional[torch.Tensor] = None,
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
        if beatmap_idx is None and self.do_style_embed:
            batch_size = frames.shape[0] if frames is not None else decoder_input_ids.shape[0]
            device = frames.device if frames is not None else decoder_input_ids.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        inputs_embeds = None
        if encoder_outputs is None and frames is not None:
            frames = self.spectrogram(frames)  # (N, L, M)
            if self.do_style_embed:
                style_embeds = self.style_embedder(beatmap_idx)  # (N, D)
                frames_concat = torch.concatenate((frames, style_embeds.unsqueeze(1).expand((-1, frames.shape[1], -1))), -1)
                inputs_embeds = self.encoder_embedder(frames_concat)
            else:
                inputs_embeds = self.encoder_embedder(frames)

        inputs = dict(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, labels=labels, **kwargs
        )

        if self.embed_decoder_input:
            inputs["decoder_inputs_embeds"] = self.decoder_embedder(decoder_input_ids)
            del inputs["decoder_input_ids"]

        if self.input_features:
            inputs["input_features"] = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None
            del inputs["inputs_embeds"]

        output = self.transformer.forward(**inputs)

        if labels is not None:
            unreduced_loss = self.loss_fn(torch.swapaxes(output.logits, 1, -1), labels)
            if sample_weights is not None:
                unreduced_loss *= sample_weights.unsqueeze(1)
            output.loss = unreduced_loss.sum() / (labels != LABEL_IGNORE_ID).sum()

        return output

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        beatmap_idx=None,
        decoder_attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        inputs = self.transformer.prepare_inputs_for_generation(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        inputs["beatmap_idx"] = beatmap_idx
        return inputs

    def can_generate(self) -> bool:
        return True

    def get_encoder(self):
        return OsuTEncoder(
            self.transformer.get_encoder(),
            self.spectrogram,
            self.style_embedder if self.do_style_embed else None,
            self.encoder_embedder,
            self.num_classes,
            self.input_features,
            self.do_style_embed
        )

    def get_decoder(self):
        return self.transformer.get_decoder()


class OsuTEncoder(nn.Module):
    def __init__(
            self,
            base_encoder: WhisperEncoder,
            spectrogram: MelSpectrogram,
            style_embedder: LabelEmbedder,
            encoder_embedder: nn.Linear,
            num_classes: int,
            input_features: bool,
            do_style_embed: bool
    ):
        super().__init__()
        self.base = base_encoder
        self.spectrogram = spectrogram
        self.style_embedder = style_embedder
        self.encoder_embedder = encoder_embedder
        self.num_classes = num_classes
        self.input_features = input_features
        self.do_style_embed = do_style_embed

    def forward(
            self,
            frames: torch.FloatTensor,
            beatmap_idx: torch.Tensor,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False
    ):
        if beatmap_idx is None and self.do_style_embed:
            batch_size = frames.shape[0]
            device = frames.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        frames = self.spectrogram(frames)  # (N, L, M)
        if self.do_style_embed:
            style_embeds = self.style_embedder(beatmap_idx)  # (N, D)
            frames_concat = torch.concatenate((frames, style_embeds.unsqueeze(1).expand((-1, frames.shape[1], -1))),
                                              -1)
            inputs_embeds = self.encoder_embedder(frames_concat)
        else:
            inputs_embeds = self.encoder_embedder(frames)

        if self.input_features:
            inputs_embeds = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None

        return self.base.forward(
            inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


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
