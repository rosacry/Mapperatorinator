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
        self.sequence_stride = int(self.samples_per_sequence * args.data.sequence_stride)
        self.miliseconds_per_sequence = self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        self.miliseconds_per_stride = self.sequence_stride * MILISECONDS_PER_SECOND / self.sample_rate
        self.beatmap_id = args.beatmap_id
        self.difficulty = args.difficulty
        self.center_pad_decoder = args.data.center_pad_decoder
        self.special_token_len = args.data.special_token_len
        self.diff_token_index = args.data.diff_token_index
        self.style_token_index = args.data.style_token_index
        self.max_pre_token_len = args.data.max_pre_token_len

    def generate(self, model: OsuT, sequences: torch.Tensor) -> list[Event]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            model: Trained model to use for inference.
            sequences: A list of batched source sequences.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """
        events = []
        event_times = []

        idx_dict = self.tokenizer.beatmap_idx
        if self.beatmap_id in idx_dict:
            beatmap_idx = torch.tensor([idx_dict[self.beatmap_id]], dtype=torch.long, device=self.device)
            style_token = self.tokenizer.encode_style(self.beatmap_id)
        else:
            print(f"Beatmap ID {self.beatmap_id} not found in dataset, using default beatmap.")
            beatmap_idx = torch.tensor([self.tokenizer.num_classes], dtype=torch.long, device=self.device)
            style_token = self.tokenizer.style_unk

        diff_token = self.tokenizer.encode_diff(self.difficulty) if self.difficulty != -1 else self.tokenizer.diff_unk

        special_tokens = torch.empty((1, self.special_token_len), dtype=torch.long, device=self.device)
        special_tokens[:, self.diff_token_index] = diff_token
        special_tokens[:, self.style_token_index] = style_token

        for sequence_index, frames in enumerate(tqdm(sequences)):
            # Get tokens of previous frame
            frame_time = sequence_index * self.miliseconds_per_stride
            prev_events = self._get_events_time_range(
                events, event_times, frame_time - self.miliseconds_per_sequence, frame_time)
            post_events = self._get_events_time_range(
                events, event_times, frame_time, frame_time + self.miliseconds_per_sequence)
            prev_tokens = self._encode(prev_events, frame_time)
            post_tokens = self._encode(post_events, frame_time)
            post_token_length = post_tokens.shape[1]

            if 0 <= self.max_pre_token_len < prev_tokens.shape[1]:
                prev_tokens = prev_tokens[:, -self.max_pre_token_len:]

            # Get prefix tokens
            prefix = torch.concatenate([special_tokens, prev_tokens], dim=-1)
            if self.center_pad_decoder:
                prefix = F.pad(prefix, (self.tgt_seq_len // 2 - prefix.shape[1], 0), value=self.tokenizer.pad_id)
            prefix_length = prefix.shape[1]

            # Get tokens
            tokens = torch.tensor([[self.tokenizer.sos_id]], dtype=torch.long, device=self.device)
            tokens = torch.concatenate([prefix, tokens, post_tokens], dim=-1)

            frames = frames.to(self.device).unsqueeze(0)
            encoder_outputs = None

            for _ in range(self.tgt_seq_len // 2):
                out = model.forward(
                    frames=frames,
                    decoder_input_ids=tokens,
                    decoder_attention_mask=tokens.ne(self.tokenizer.pad_id),
                    encoder_outputs=encoder_outputs,
                    beatmap_idx=beatmap_idx,
                )
                encoder_outputs = (out.encoder_last_hidden_state, out.encoder_hidden_states, out.encoder_attentions)
                logits = out.logits
                logits = logits[:, -1, :]
                logits = self._filter(logits, 0.9)
                probabilities = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probabilities, 1)

                tokens = torch.cat([tokens, next_tokens], dim=-1)

                # check if any sentence in batch has reached EOS, mark as finished
                eos_in_sentence = next_tokens == self.tokenizer.eos_id

                # stop preemptively when all sentences have finished
                if eos_in_sentence.all():
                    break

            # Trim prefix, SOS, post-tokens, and EOS tokens
            predicted_tokens = tokens[:, prefix_length + 1 + post_token_length:-1]
            result = self._decode(predicted_tokens[0], frame_time)
            events += result
            self._update_event_times(events, event_times, frame_time)

        return events

    def _get_events_time_range(self, events: list[Event], event_times: list[float], start_time: float, end_time: float):
        # Look from the end of the list
        s = 0
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] < start_time:
                s = i + 1
                break
        e = 0
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] < end_time:
                e = i + 1
                break
        return events[s:e]

    def _update_event_times(self, events: list[Event], event_times: list[float], frame_time: float):
        non_timed_events = [
            EventType.BEZIER_ANCHOR,
            EventType.PERFECT_ANCHOR,
            EventType.CATMULL_ANCHOR,
            EventType.RED_ANCHOR,
        ]
        timed_events = [
            EventType.CIRCLE,
            EventType.SPINNER,
            EventType.SPINNER_END,
            EventType.SLIDER_HEAD,
            EventType.LAST_ANCHOR,
            EventType.SLIDER_END,
        ]

        end_time = frame_time + self.miliseconds_per_sequence
        start_index = len(event_times)
        end_index = len(events)
        ct = 0 if len(event_times) == 0 else event_times[-1]
        for i in range(start_index, end_index):
            event = events[i]
            if event.type == EventType.TIME_SHIFT:
                ct = event.value
            event_times.append(ct)

        # Interpolate time for control point events
        # T-D-Start-D-CP-D-CP-T-D-LCP-T-D-End
        # 1-1-1-----1-1--1-1--7-7--7--9-9-9--
        # 1-1-1-----3-3--5-5--7-7--7--9-9-9--
        ct = end_time
        interpolate = False
        for i in range(end_index - 1, start_index - 1, -1):
            event = events[i]

            if event.type in timed_events:
                interpolate = False

            if event.type in non_timed_events:
                interpolate = True

            if not interpolate:
                ct = event_times[i]
                continue

            if event.type not in non_timed_events:
                event_times[i] = ct
                continue

            # Find the time of the first timed event and the number of control points between
            j = i
            count = 0
            t = ct
            while j >= 0:
                event2 = events[j]
                if event2.type == EventType.TIME_SHIFT:
                    t = event_times[j]
                    break
                if event2.type in non_timed_events:
                    count += 1
                j -= 1
            if i < 0:
                t = 0

            # Interpolate the time
            ct = (ct - t) / (count + 1) * count + t
            event_times[i] = ct

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
                if self.tokenizer.event_start[EventType.TIME_SHIFT] <= token < self.tokenizer.event_end[
                    EventType.TIME_SHIFT]:
                    event = self.tokenizer.decode(token.item())
                    event.value += int(time_offset / MILISECONDS_PER_STEP)
                    tokens[i, j] = self.tokenizer.encode(event)
        return tokens

    def _encode(self, events: list[Event], frame_time: float) -> torch.Tensor:
        tokens = torch.empty((1, len(events)), dtype=torch.long)
        for i, event in enumerate(events):
            if event.type == EventType.TIME_SHIFT:
                event = Event(type=event.type, value=int((event.value - frame_time) / MILISECONDS_PER_STEP))
            tokens[0, i] = self.tokenizer.encode(event)
        return tokens.to(self.device)

    def _decode(self, tokens: torch.Tensor, frame_time: float) -> list[Event]:
        """Converts a list of tokens into Event objects and converts to absolute time values.

        Args:
            tokens: List of tokens.
            frame time: Start time of current source sequence.

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
                event.value = frame_time + event.value * MILISECONDS_PER_STEP

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
