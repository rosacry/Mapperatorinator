from __future__ import annotations

from typing import Optional, Mapping, Any, OrderedDict

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from osuT5.model.spectrogram import MelSpectrogram
from osuT5.model.t5 import T5


class OsuT(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()

        self.spectrogram = MelSpectrogram(
            config.sample_rate, config.n_fft, config.n_mels, config.hop_length
        )
        self.encoder_embedder = nn.Linear(config.n_mels, config.d_model)
        self.transformer = T5ForConditionalGeneration(config)

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            encoder_outputs=None,
            **kwargs
    ) -> Seq2SeqLMOutput:
        """
        frames: B x L_encoder x mel_bins, float32
        attention_mask: B x L_encoder, int64
            1 for tokens to attend to, 0 for tokens to ignore
        tokens: B x L_decoder, int64
        """
        inputs_embeds = None

        if encoder_outputs is None:
            frames = self.spectrogram(frames)
            inputs_embeds = self.encoder_embedder(frames)

        output = self.transformer.forward(inputs_embeds=inputs_embeds, encoder_outputs=encoder_outputs, **kwargs)

        if not hasattr(output, "encoder_outputs"):
            output.encoder_outputs = (
                output.encoder_last_hidden_state, output.encoder_hidden_states, output.encoder_attentions)

        return output

    def load_state_dict_old(self, state_dict: OrderedDict[str, Any], strict: bool = True, assign: bool = False):
        if "shared.weight" not in state_dict and "decoder_embedder.weight" in state_dict:
            state_dict["shared.weight"] = state_dict["decoder_embedder.weight"]

            if self.transformer is not T5:
                state_dict["encoder.embed_tokens.weight"] = state_dict["decoder_embedder.weight"]

            self.transformer.load_state_dict(state_dict, False, assign)
            self.encoder_embedder.load_state_dict(get_state_dict_part(state_dict, "encoder_embedder"), strict, assign)
            self.spectrogram.load_state_dict(get_state_dict_part(state_dict, "spectrogram"), strict, assign)
        else:
            self.load_state_dict(state_dict, strict, assign)


def get_state_dict_part(state_dict: Mapping[str, Any], part: str):
    sub_dist = {}

    for k in state_dict:
        i = k.index('.')
        if k[:i] == part:
            sub_dist[k[i + 1:]] = state_dict[k]

    return sub_dist
