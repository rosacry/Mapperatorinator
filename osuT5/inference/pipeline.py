from __future__ import annotations
import torch
import torch.nn.functional as F
from tqdm import tqdm

from omegaconf import DictConfig
from osuT5.tokenizer import Event, EventType, Tokenizer
from osuT5.model import OsuT

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


class Pipeline(object):
    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        """Model inference stage that processes sequences."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.tgt_seq_len = args.model.max_target_len
        self.frame_seq_len = args.model.max_seq_len - 1
        self.frame_size = args.model.spectrogram.hop_length
        self.sample_rate = args.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.miliseconds_per_sequence = (
            self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        )
        self.beatmap_id = args.beatmap_id
        self.difficulty = args.difficulty

    def generate( self, model: OsuT, sequences: torch.Tensor) -> list[Event]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            model: Trained model to use for inference.
            sequences: A list of batched source sequences.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """
        events = []
        prev_targets = torch.full((1, 0), self.tokenizer.pad_id, dtype=torch.long, device=self.device)

        idx_dict = self.tokenizer.beatmap_idx
        if self.beatmap_id in idx_dict:
            style_token = self.tokenizer.encode_style(self.beatmap_id)
        else:
            print(f"Beatmap ID {self.beatmap_id} not found in dataset, using default beatmap.")
            style_token = self.tokenizer.style_unk

        diff_token = self.tokenizer.encode_diff(self.difficulty) if self.difficulty is not None else self.tokenizer.diff_unk

        for sequence_index, frames in enumerate(tqdm(sequences)):
            m = prev_targets.shape[1]
            targets = torch.tensor([[diff_token, style_token]], dtype=torch.long, device=self.device)
            targets = torch.concatenate([targets, prev_targets], dim=-1)
            frames = frames.to(self.device)
            encoder_outputs = None

            for _ in range(self.tgt_seq_len // 2):
                out = model.forward(
                    frames=frames.unsqueeze(0),
                    decoder_input_ids=targets,
                    decoder_attention_mask=targets.ne(self.tokenizer.pad_id),
                    encoder_outputs=encoder_outputs,
                )
                encoder_outputs = (out.encoder_last_hidden_state, out.encoder_hidden_states, out.encoder_attentions)
                logits = out.logits
                logits = logits[:, -1, :]
                logits = self._filter(logits, 0.9)
                probabilities = F.softmax(logits, dim=-1)
                tokens = torch.multinomial(probabilities, 1)

                targets = torch.cat([targets, tokens.to(self.device)], dim=-1)

                # check if any sentence in batch has reached EOS, mark as finished
                eos_in_sentence = tokens == self.tokenizer.eos_id

                # stop preemptively when all sentences have finished
                if eos_in_sentence.all():
                    break

            predicted_targets = targets[:, m + 2:-1]
            result = self._decode(predicted_targets[0], sequence_index)
            events += result

            prev_targets = predicted_targets
            if prev_targets.shape[1] > 0:
                prev_targets = self._timeshift_tokens(prev_targets, -self.miliseconds_per_sequence)

        return events

    def _timeshift_tokens(self, tokens: torch.Tensor, time_offset: float) -> torch.Tensor:
        """Changes the time offset of TIME_SHIFT tokens.

        Args:
            tokens: Long tensor of tokens shaped (batch size, sequence length).
            time_offset: Time offset in miliseconds.

        Returns:
            tokens: List of tokens with updated time values.
        """
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                token = tokens[i, j]
                if self.tokenizer.event_start[EventType.TIME_SHIFT] <= token < self.tokenizer.event_end[EventType.TIME_SHIFT]:
                    event = self.tokenizer.decode(token.item())
                    event.value += int(time_offset / MILISECONDS_PER_STEP)
                    tokens[i, j] = self.tokenizer.encode(event)
        return tokens

    def _decode(self, tokens: torch.Tensor, index: int) -> list[Event]:
        """Converts a list of tokens into Event objects and converts to absolute time values.

        Args:
            tokens: List of tokens.
            index: Index of current source sequence.

        Returns:
            events: List of Event objects.
        """
        events = []
        for token in tokens:
            if token == self.tokenizer.eos_id:
                break

            try:
                event = self.tokenizer.decode(token.item())
            except:
                continue

            if event.type == EventType.TIME_SHIFT:
                current_time = (
                    index * self.miliseconds_per_sequence
                    + event.value * MILISECONDS_PER_STEP
                )
                event.value = current_time

            events.append(event)

        return events

    def _filter(
        self, logits: torch.Tensor, top_p: float, filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """Filter a distribution of logits using nucleus (top-p) filtering.

        Source: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        Args:
            logits: logits distribution of shape (batch size, vocabulary size).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

        Returns:
            logits of top tokens.
        """
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        return logits
