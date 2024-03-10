from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from osuT5.model.spectrogram import MelSpectrogram


class OsuT(nn.Module):
    __slots__ = ["spectrogram", "decoder_embedder", "encoder_embedder", "transformer"]

    def __init__(self, config: T5Config):
        super().__init__()

        self.decoder_embedder = nn.Embedding(config.vocab_size_in, config.d_model)
        self.decoder_embedder.weight.data.normal_(mean=0.0, std=1.0)

        self.spectrogram = MelSpectrogram(
            config.sample_rate, config.n_fft, config.n_mels, config.hop_length
        )
        self.encoder_embedder = nn.Linear(config.n_mels, config.d_model)

        self.transformer = T5ForConditionalGeneration(config)

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> Seq2SeqLMOutput:
        """
        frames: B x L_encoder x mel_bins, float32
        decoder_input_ids: B x L_decoder, int64
        beatmap_idx: B, int64
        beatmap_id: B, int64
        encoder_outputs: B x L_encoder x D, float32
        """

        inputs_embeds = None
        if encoder_outputs is None:
            frames = self.spectrogram(frames)  # (N, L, M)
            inputs_embeds = self.encoder_embedder(frames)

        decoder_inputs_embeds = self.decoder_embedder(decoder_input_ids)

        output = self.transformer.forward(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, encoder_outputs=encoder_outputs, **kwargs)

        return output

