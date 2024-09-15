from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from slider import Beatmap
from tqdm import tqdm
from transformers.generation import ClassifierFreeGuidanceLogitsProcessor
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

from omegaconf import DictConfig

from ..dataset import OsuParser
from ..dataset.data_utils import update_event_times, remove_events_of_type
from ..tokenizer import Event, EventType, Tokenizer, ContextType
from ..model import OsuT

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


class Pipeline(object):
    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        """Model inference stage that processes sequences."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.tgt_seq_len = args.osut5.data.tgt_seq_len
        self.frame_seq_len = args.osut5.data.src_seq_len - 1
        self.frame_size = args.osut5.model.spectrogram.hop_length
        self.sample_rate = args.osut5.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.sequence_stride = int(self.samples_per_sequence * (1 - args.lookback - args.lookahead))
        self.miliseconds_per_sequence = self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        self.miliseconds_per_stride = self.sequence_stride * MILISECONDS_PER_SECOND / self.sample_rate
        self.lookahead_max_time = (1 - args.lookahead) * self.miliseconds_per_sequence
        self.eos_time = (1 - args.osut5.data.lookahead) * self.miliseconds_per_sequence
        self.center_pad_decoder = args.osut5.data.center_pad_decoder
        self.special_token_len = args.osut5.data.special_token_len
        self.diff_token_index = args.osut5.data.diff_token_index
        self.style_token_index = args.osut5.data.style_token_index
        self.mapper_token_index = args.osut5.data.mapper_token_index
        self.cs_token_index = args.osut5.data.cs_token_index
        self.add_descriptors = args.osut5.data.add_descriptors
        self.max_pre_token_len = args.osut5.data.max_pre_token_len
        self.add_pre_tokens = args.osut5.data.add_pre_tokens
        self.add_gd_context = args.osut5.data.add_gd_context
        self.parser = OsuParser(args.osut5, self.tokenizer)
        self.need_beatmap_idx = args.osut5.model.do_style_embed
        self.add_positions = args.osut5.data.add_positions

        if self.add_positions:
            self.position_precision = args.osut5.data.position_precision
            x_min, x_max, y_min, y_max = args.osut5.data.position_range
            self.x_min = x_min / self.position_precision
            self.x_max = x_max / self.position_precision
            self.y_min = y_min / self.position_precision
            self.y_max = y_max / self.position_precision
            self.x_count = self.x_max - self.x_min + 1

        self.cfg_scale = args.cfg_scale
        processor_list = [] if self.cfg_scale <= 1 else [ClassifierFreeGuidanceLogitsProcessor(self.cfg_scale)]
        self.logits_processor = LogitsProcessorList(processor_list + [
            TemperatureLogitsWarper(args.temperature),
            TopPLogitsWarper(args.top_p),
        ])

        self.timeshift_bias = args.timeshift_bias
        self.time_range = range(tokenizer.event_start[EventType.TIME_SHIFT], tokenizer.event_end[EventType.TIME_SHIFT])
        self.types_first = args.osut5.data.types_first

    def get_class_vector(
            self,
            beatmap_id: int = -1,
            difficulty: float = -1,
            mapper_id: int = -1,
            descriptors: list[str] | list[int] = None,
            circle_size: float = -1,
            verbose: bool = False
    ):
        descriptor_tokens = []
        if self.add_descriptors:
            if descriptors is not None and len(descriptors) > 0:
                for descriptor in descriptors:
                    if isinstance(descriptor, str):
                        if descriptor not in self.tokenizer.descriptor_idx:
                            if verbose:
                                print(f"Descriptor class {descriptor} not found. Skipping.")
                            continue
                        descriptor_tokens.append(self.tokenizer.encode_descriptor_name(descriptor))
                    elif isinstance(descriptor, int):
                        if descriptor < self.tokenizer.event_range[EventType.DESCRIPTOR].min_value or \
                                descriptor > self.tokenizer.event_range[EventType.DESCRIPTOR].max_value:
                            if verbose:
                                print(f"Descriptor idx {descriptor} out of range. Skipping.")
                            continue
                        descriptor_tokens.append(self.tokenizer.encode_descriptor_idx(descriptor))
            if descriptors is None or len(descriptors) == 0:
                descriptor_tokens = [self.tokenizer.descriptor_unk]

        cond_tokens = torch.empty((1, self.special_token_len + len(descriptors)), dtype=torch.long, device=self.device)

        if self.style_token_index >= 0:
            style_token = self.tokenizer.encode_style(beatmap_id) if beatmap_id != -1 else self.tokenizer.style_unk
            cond_tokens[:, self.style_token_index] = style_token
            if beatmap_id != -1 and beatmap_id not in self.tokenizer.beatmap_idx and verbose:
                print(f"Beatmap class {beatmap_id} not found. Using default.")
        if self.diff_token_index >= 0:
            diff_token = self.tokenizer.encode_diff(difficulty) if difficulty != -1 else self.tokenizer.diff_unk
            cond_tokens[:, self.diff_token_index] = diff_token
        if self.mapper_token_index >= 0:
            mapper_token = self.tokenizer.encode_mapper_id(mapper_id) if mapper_id != -1 else self.tokenizer.mapper_unk
            cond_tokens[:, self.mapper_token_index] = mapper_token
            if mapper_id != -1 and mapper_id not in self.tokenizer.mapper_idx and verbose:
                print(f"Mapper class {mapper_id} not found. Using default.")
        if self.cs_token_index >= 0:
            cs_token = self.tokenizer.encode_cs(circle_size) if circle_size != -1 else self.tokenizer.cs_unk
            cond_tokens[:, self.cs_token_index] = cs_token
        for i, descriptor in enumerate(descriptors):
            cond_tokens[:, self.special_token_len + i] = descriptor_tokens[i]

        return cond_tokens

    def generate(
            self,
            model: OsuT,
            sequences: torch.Tensor,
            beatmap_id: int = -1,
            difficulty: float = -1,
            mapper_id: int = -1,
            descriptors: list[str] = None,
            circle_size: float = -1,
            other_beatmap_path: str = '',
            context_type: ContextType = None,
            negative_descriptors: list[str] = None,
    ) -> list[Event]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            model: Trained model to use for inference.
            sequences: A list of batched source sequences.
            beatmap_id: Beatmap ID of the desired style.
            difficulty: The desired difficulty in star rating.
            mapper_id: Mapper ID for the style of beatmap.
            descriptors: List of descriptors for the style of beatmap.
            circle_size: Circle size of the beatmap.
            other_beatmap_path: Path to the beatmap file to use as context.
            context_type: Type of context to use for inference.
            negative_descriptors: List of descriptors for negative class vector.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """

        events = []
        event_times = []

        # Prepare special tokens
        beatmap_idx = torch.tensor([self.tokenizer.num_classes], dtype=torch.long, device=self.device)
        if self.need_beatmap_idx:
            beatmap_idx = torch.tensor([self.tokenizer.beatmap_idx[beatmap_id]], dtype=torch.long, device=self.device)

        # Prepare unconditional prompt
        cond_tokens = self.get_class_vector(beatmap_id, difficulty, mapper_id, descriptors, circle_size, verbose=True)
        uncond_tokens = self.get_class_vector(-1, difficulty, -1, negative_descriptors, circle_size)

        # Prepare other beatmap context
        other_events, other_event_times = [], []
        other_cond_tokens = torch.empty((1, 0), dtype=torch.long, device=self.device)
        if self.add_gd_context or context_type == ContextType.GD or context_type == ContextType.TIMING or context_type == ContextType.NO_HS:
            other_beatmap_path = Path(other_beatmap_path)

            if not other_beatmap_path.is_file():
                raise FileNotFoundError(f"Beatmap file {other_beatmap_path} not found.")

            other_beatmap = Beatmap.from_path(other_beatmap_path)

            if self.add_gd_context or context_type == ContextType.GD:
                other_events, other_event_times = self.parser.parse(other_beatmap)
                other_beatmap_id = other_beatmap.beatmap_id

                other_cond_tokens = self.get_class_vector(
                    other_beatmap_id,
                    float(other_beatmap.stars()),
                    self.tokenizer.beatmap_mapper.get(other_beatmap_id, -1),
                    self.tokenizer.beatmap_descriptors.get(other_beatmap_id, []),
                    other_beatmap.circle_size,
                )
            elif context_type == ContextType.NO_HS:
                other_events, other_event_times = self.parser.parse(other_beatmap)
                other_events, other_event_times = remove_events_of_type(other_events, other_event_times, [EventType.HITSOUND, EventType.VOLUME])
            elif context_type == ContextType.TIMING:
                other_events, other_event_times = self.parser.parse_timing(other_beatmap)

        # Prepare context type indicator tokens
        context_sos = torch.tensor([[self.tokenizer.context_sos[context_type]]], dtype=torch.long, device=self.device)\
            if context_type is not None else torch.empty((1, 0), dtype=torch.long, device=self.device)
        context_eos = torch.tensor([[self.tokenizer.context_eos[context_type]]], dtype=torch.long, device=self.device)\
            if context_type is not None else torch.empty((1, 0), dtype=torch.long, device=self.device)

        def get_prompt(user_prompt, context_tokens, prev_tokens, post_tokens):
            prefix = torch.concatenate([context_sos, other_cond_tokens, context_tokens, context_eos, user_prompt, prev_tokens], dim=-1)
            if self.center_pad_decoder:
                prefix = F.pad(prefix, (self.tgt_seq_len // 2 - prefix.shape[1], 0), value=self.tokenizer.pad_id)

            prompt = torch.tensor([[self.tokenizer.sos_id]], dtype=torch.long, device=self.device)
            prompt = torch.concatenate([prefix, prompt, post_tokens], dim=-1)
            return prompt

        # Start generation
        for sequence_index, frames in enumerate(tqdm(sequences)):
            frames = frames.to(self.device).unsqueeze(0)

            # Get tokens of previous frame
            frame_time = sequence_index * self.miliseconds_per_stride

            prev_events = self._get_events_time_range(
                events, event_times, frame_time - self.miliseconds_per_sequence, frame_time) \
                if self.add_pre_tokens else []
            prev_tokens = self._encode(prev_events, frame_time)
            if 0 <= self.max_pre_token_len < prev_tokens.shape[1]:
                prev_tokens = prev_tokens[:, -self.max_pre_token_len:]

            post_events = self._get_events_time_range(
                events, event_times, frame_time, frame_time + self.miliseconds_per_sequence)
            post_tokens = self._encode(post_events, frame_time)

            context_events = self._get_events_time_range(
                other_events, other_event_times, frame_time,
                frame_time + self.miliseconds_per_sequence)
            context_tokens = self._encode(context_events, frame_time)

            # Prepare classifier-free guidance
            cond_prompt = get_prompt(cond_tokens, context_tokens, prev_tokens, post_tokens)
            prompt = cond_prompt
            prompt_length = prompt.shape[1]

            if self.cfg_scale > 1:
                uncond_prompt = get_prompt(uncond_tokens, context_tokens, prev_tokens, post_tokens)
                # Left-pad unconditional prompt to match the length of conditional prompt
                uncond_prompt = F.pad(uncond_prompt, (prompt_length - uncond_prompt.shape[1], 0), value=self.tokenizer.pad_id)
                prompt = torch.concatenate([cond_prompt, uncond_prompt], dim=0)

                # Repeat frames to match the batch size
                frames = frames.repeat(prompt.shape[0], 1)

            # Prepare cache for autoregressive decoding
            encoder_outputs = None
            past_key_values = None

            input_ids = cond_prompt

            while input_ids.shape[1] < self.tgt_seq_len:
                if past_key_values is not None:
                    out = model.forward(
                        decoder_input_ids=input_ids[:, -1:].repeat(2, 1) if self.cfg_scale > 1 else input_ids[:, -1:],
                        encoder_outputs=encoder_outputs,
                        beatmap_idx=beatmap_idx,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                else:
                    out = model.forward(
                        frames=frames,
                        decoder_input_ids=prompt,
                        decoder_attention_mask=prompt.ne(self.tokenizer.pad_id),
                        beatmap_idx=beatmap_idx,
                    )

                past_key_values = out.past_key_values
                encoder_outputs = (out.encoder_last_hidden_state, out.encoder_hidden_states, out.encoder_attentions)
                logits = out.logits[:, -1, :]

                if self.timeshift_bias != 0:
                    logits[:, self.time_range] += self.timeshift_bias

                # noinspection PyTypeChecker
                logits = self.logits_processor(input_ids, logits)
                probabilities = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probabilities, 1)

                input_ids = torch.cat([input_ids, next_tokens], dim=-1)

                # check if any sentence in batch has reached EOS, mark as finished
                eos_in_sentence = next_tokens == self.tokenizer.eos_id

                # stop preemptively when all sentences have finished
                if eos_in_sentence.all():
                    break

                next_event = self.tokenizer.decode(next_tokens[0].item())
                if (sequence_index != len(sequences) - 1 and
                        next_event.type == EventType.TIME_SHIFT and
                        next_event.value * MILISECONDS_PER_STEP > self.lookahead_max_time):
                    # If the type token comes before the timeshift token we should remove the type token too
                    if self.types_first:
                        input_ids = input_ids[:, :-1]
                    break
                if (next_event.type == EventType.BEAT or next_event.type == EventType.MEASURE) and context_tokens.shape[1] > 1 and input_ids.shape[1] > prompt_length + 1:
                    # Ensure the beat or measure token matches the timing of the context
                    input_ids[0, -2] = self._get_beat_time_token_from_context(context_tokens[0], input_ids[0, prompt_length - post_tokens.shape[1]:-2])

            # Trim prompt and EOS tokens
            predicted_tokens = input_ids[:, prompt_length:-1]
            result = self._decode(predicted_tokens[0], frame_time)
            events += result
            self._update_event_times(events, event_times, frame_time + self.eos_time)

            # Trim events which are in the lookahead window
            if sequence_index != len(sequences) - 1:
                lookahead_time = frame_time + self.lookahead_max_time
                self._trim_events_after_time(events, event_times, lookahead_time)

        # Rescale and unpack position events
        if self.add_positions:
            events = self._rescale_positions(events)

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

    def _trim_events_after_time(self, events, event_times, lookahead_time):
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] > lookahead_time:
                del events[i]
                del event_times[i]
            else:
                break

    def _update_event_times(self, events: list[Event], event_times: list[float], end_time: float):
        update_event_times(events, event_times, end_time)

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

    def _rescale_positions(self, events: list[Event]) -> list[Event]:
        new_events = []
        offset = self.position_precision // 2 if self.position_precision > 1 else 0
        for event in events:
            if event.type == EventType.POS_X or event.type == EventType.POS_Y:
                new_events.append(Event(type=event.type, value=event.value * self.position_precision))
            elif event.type == EventType.POS:
                new_events.append(Event(type=EventType.POS_X, value=((event.value % self.x_count) + self.x_min) * self.position_precision + offset))
                new_events.append(Event(type=EventType.POS_Y, value=((event.value // self.x_count) + self.y_min) * self.position_precision + offset))
            else:
                new_events.append(event)

        return new_events

    def _get_beat_time_token_from_context(self, context_tokens, generated_tokens):
        beat_tokens = [self.tokenizer.event_start[EventType.BEAT], self.tokenizer.event_start[EventType.MEASURE]]
        context_tokens = context_tokens.cpu()
        generated_tokens = generated_tokens.cpu()

        # Search generated tokens in reverse order for the latest time shift token followed by a beat or measure token
        latest_time = -1000
        beat_offset = -1 if self.types_first else 1
        for i in range(len(generated_tokens) - 1, -1, -1):
            token = generated_tokens[i]
            if (token in self.time_range and
                    0 <= i + beat_offset < len(generated_tokens) and generated_tokens[i + beat_offset] in beat_tokens):
                latest_time = token
                break

        # Search context tokens in order for the first time shift token after latest_time which is followed by a beat or measure token
        for i, token in enumerate(context_tokens):
            if (token in self.time_range and token > latest_time + 1 and
                    0 <= i + beat_offset < len(context_tokens) and context_tokens[i + beat_offset] in beat_tokens):
                return token
