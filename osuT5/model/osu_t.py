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
        self.style_embedder = LabelEmbedder(config.num_classes, config.d_model, config.class_dropout_prob)
        self.encoder_embedder = nn.Linear(config.n_mels + config.d_model, config.d_model)

        # Initialize label embedding table:
        nn.init.normal_(self.style_embedder.embedding_table.weight, std=0.02)

        self.transformer = T5(config)

    def forward(
            self,
            samples: Optional[torch.FloatTensor] = None,
            styles: Optional[torch.LongTensor] = None,
            encoder_outputs=None,
            **kwargs
    ) -> Seq2SeqLMOutput:
        """
        frames: B x samples_per_sequence, float32
        attention_mask: B x L_encoder, int64
            1 for tokens to attend to, 0 for tokens to ignore
        tokens: B x L_decoder, int64
        """
        inputs_embeds = None

        if encoder_outputs is None:
            frames = self.spectrogram(samples)  # (N, L, M)
            style_embeds = self.style_embedder(styles, self.training)  # (N, D)
            frames_concat = torch.concatenate((frames, style_embeds.unsqueeze(1)), -1)
            inputs_embeds = self.encoder_embedder(frames_concat)

        output = self.transformer.forward(inputs_embeds=inputs_embeds, encoder_outputs=encoder_outputs, **kwargs)

        if isinstance(output, Seq2SeqLMOutput) and not hasattr(output, "encoder_outputs"):
            # noinspection PyUnresolvedReferences
            output.encoder_outputs = (
                output.encoder_last_hidden_state, output.encoder_hidden_states, output.encoder_attentions)

        return output

    def load_state_dict_old(self, state_dict: OrderedDict[str, Any], strict: bool = True, assign: bool = False):
        if "shared.weight" not in state_dict and "decoder_embedder.weight" in state_dict:
            state_dict["shared.weight"] = state_dict["decoder_embedder.weight"]

            if not isinstance(self.transformer, T5):
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


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding,
            hidden_size,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
