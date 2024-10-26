from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class GenerationConfig:
    beatmap_id: int = -1
    difficulty: float = -1
    mapper_id: int = -1
    descriptors: list[str] = None
    negative_descriptors: list[str] = None
    circle_size: float = -1


def generation_config_from_beatmap(beatmap: Beatmap, tokenizer: Tokenizer) -> GenerationConfig:
    return GenerationConfig(
        beatmap_id=beatmap.beatmap_id,
        difficulty=float(beatmap.stars()),
        mapper_id=tokenizer.mapper_idx.get(beatmap.beatmap_id, -1),
        descriptors=[tokenizer.descriptor_name(idx) for idx in tokenizer.beatmap_descriptors.get(beatmap.beatmap_id, [])],
        circle_size=beatmap.circle_size,
    )


class Processor(object):
    def __init__(self, args: DictConfig, model: OsuT, tokenizer: Tokenizer):
        """Model inference stage that processes sequences."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
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
        self.lookahead_time_range = range(tokenizer.encode(Event(EventType.TIME_SHIFT, int(self.lookahead_max_time / MILISECONDS_PER_STEP) + 1)), tokenizer.event_end[EventType.TIME_SHIFT])
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
        self.add_timing = args.osut5.data.add_timing
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
        self.beat_range = [self.tokenizer.event_start[EventType.BEAT], self.tokenizer.event_start[EventType.MEASURE]]
        self.types_first = args.osut5.data.types_first

    def get_context(self, context: ContextType, beatmap_path, add_type=True):
        beatmap_path = Path(beatmap_path)
        if context != ContextType.NONE and not beatmap_path.is_file():
            raise FileNotFoundError(f"Beatmap file {beatmap_path} not found.")

        data = {"context_type": ContextType(context), "add_type": add_type}

        if context == ContextType.NONE:
            data["events"], data["event_times"] = [], []
        elif context == ContextType.TIMING:
            beatmap = Beatmap.from_path(beatmap_path)
            data["events"], data["event_times"] = self.parser.parse_timing(beatmap)
        elif context == ContextType.NO_HS:
            beatmap = Beatmap.from_path(beatmap_path)
            hs_events, hs_event_times = self.parser.parse(beatmap)
            data["events"], data["event_times"] = remove_events_of_type(hs_events, hs_event_times,
                                                                        [EventType.HITSOUND, EventType.VOLUME])
        elif context == ContextType.GD:
            beatmap = Beatmap.from_path(beatmap_path)
            data["events"], data["event_times"] = self.parser.parse(beatmap)
            data["class"] = self.get_class_vector(generation_config_from_beatmap(beatmap, self.tokenizer))
        else:
            raise ValueError(f"Invalid context type {context}")
        return data

    def get_in_context(
            self,
            in_context: list[ContextType],
            beatmap_path: Path
    ) -> list[dict[str, Any]]:
        in_context = [self.get_context(context, beatmap_path) for context in in_context]
        if self.add_gd_context:
            in_context.append(self.get_context(ContextType.GD, beatmap_path, add_type=False))
        return in_context

    def get_class_vector(
            self,
            config: GenerationConfig,
            verbose: bool = False,
    ):
        descriptors = config.descriptors if config.descriptors is not None else []
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

        cond_tokens = torch.empty((1, self.special_token_len + len(descriptor_tokens)), dtype=torch.long, device=self.device)

        if self.style_token_index >= 0:
            style_token = self.tokenizer.encode_style(config.beatmap_id) if config.beatmap_id != -1 else self.tokenizer.style_unk
            cond_tokens[:, self.style_token_index] = style_token
            if config.beatmap_id != -1 and config.beatmap_id not in self.tokenizer.beatmap_idx and verbose:
                print(f"Beatmap class {config.beatmap_id} not found. Using default.")
        if self.diff_token_index >= 0:
            diff_token = self.tokenizer.encode_diff(config.difficulty) if config.difficulty != -1 else self.tokenizer.diff_unk
            cond_tokens[:, self.diff_token_index] = diff_token
        if self.mapper_token_index >= 0:
            mapper_token = self.tokenizer.encode_mapper_id(config.mapper_id) if config.mapper_id != -1 else self.tokenizer.mapper_unk
            cond_tokens[:, self.mapper_token_index] = mapper_token
            if config.mapper_id != -1 and config.mapper_id not in self.tokenizer.mapper_idx and verbose:
                print(f"Mapper class {config.mapper_id} not found. Using default.")
        if self.cs_token_index >= 0:
            cs_token = self.tokenizer.encode_cs(config.circle_size) if config.circle_size != -1 else self.tokenizer.cs_unk
            cond_tokens[:, self.cs_token_index] = cs_token
        for i, descriptor_token in enumerate(descriptor_tokens):
            cond_tokens[:, self.special_token_len + i] = descriptor_token

        return cond_tokens

    def generate(
            self,
            *,
            sequences: torch.Tensor,
            generation_config: GenerationConfig,
            in_context: list[dict[str, Any]] = None,
            **kwargs,
    ) -> list[Event]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            sequences: A list of batched source sequences.
            generation_config: Generation configuration.
            in_context: List of context information.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """

        events = []
        event_times = []

        # Find the timing context if any
        if self.add_timing:
            # Timing tokens are in all the non-empty contexts
            timing_context = next((context for context in in_context if context["context_type"] != ContextType.NONE), None)
        else:
            timing_context = next((context for context in in_context if context["context_type"] == ContextType.TIMING), None)

        # Prepare special tokens
        beatmap_idx = torch.tensor([self.tokenizer.num_classes], dtype=torch.long, device=self.device)
        if self.need_beatmap_idx:
            beatmap_idx = torch.tensor([self.tokenizer.beatmap_idx[generation_config.beatmap_id]], dtype=torch.long, device=self.device)

        # Prepare unconditional prompt
        cond_tokens = self.get_class_vector(generation_config, verbose=True)
        uncond_tokens = self.get_class_vector(GenerationConfig(
            difficulty=generation_config.difficulty,
            descriptors=generation_config.negative_descriptors
        ))

        # Prepare context type indicator tokens
        def get_context_tokens(context):
            context_type = context["context_type"]
            tokens = context["tokens"]
            if "class" in context:
                tokens = torch.concatenate([context["class"], tokens], dim=-1)
            if context["add_type"]:
                tokens = torch.concatenate([
                    torch.tensor([[self.tokenizer.context_sos[context_type]]], dtype=torch.long, device=self.device),
                    tokens,
                    torch.tensor([[self.tokenizer.context_eos[context_type]]], dtype=torch.long, device=self.device)
                ], dim=-1)
            return tokens

        def get_prompt(user_prompt, prev_tokens, post_tokens):
            prefix = torch.concatenate([get_context_tokens(context) for context in in_context] + [user_prompt, prev_tokens], dim=-1)
            if self.center_pad_decoder:
                prefix = F.pad(prefix, (self.tgt_seq_len // 2 - prefix.shape[1], 0), value=self.tokenizer.pad_id)

            prompt = torch.tensor([[self.tokenizer.sos_id]], dtype=torch.long, device=self.device)
            prompt = torch.concatenate([prefix, prompt, post_tokens], dim=-1)
            return prompt

        # Start generation
        for sequence_index, frames in enumerate(tqdm(sequences)):
            # noinspection PyUnresolvedReferences
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

            # Get context tokens
            for context in in_context:
                context_events = self._get_events_time_range(
                    context["events"], context["event_times"], frame_time,
                    frame_time + self.miliseconds_per_sequence)
                context["tokens"] = self._encode(context_events, frame_time)

            # Prepare classifier-free guidance
            cond_prompt = get_prompt(cond_tokens, prev_tokens, post_tokens)
            prompt = cond_prompt
            prompt_length = prompt.shape[1]

            if self.cfg_scale > 1:
                uncond_prompt = get_prompt(uncond_tokens, prev_tokens, post_tokens)
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
                    out = self.model.forward(
                        decoder_input_ids=input_ids[:, -1:].repeat(2, 1) if self.cfg_scale > 1 else input_ids[:, -1:],
                        encoder_outputs=encoder_outputs,
                        beatmap_idx=beatmap_idx,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                else:
                    out = self.model.forward(
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

                last_tokens = input_ids[0, -2:].cpu()
                if sequence_index != len(sequences) - 1 and last_tokens[-1] in self.lookahead_time_range:
                    # If the type token comes before the timeshift token we should remove the type token too
                    if self.types_first:
                        input_ids = input_ids[:, :-1]
                    break
                # Ensure the beat or measure token matches the timing of the context
                if timing_context is not None and timing_context["tokens"].shape[1] > 1 and input_ids.shape[1] > prompt_length + 1:
                    beat_offset = -2 if self.types_first else -1
                    time_offset = -1 if self.types_first else -2
                    if last_tokens[beat_offset] in self.beat_range and last_tokens[time_offset] in self.time_range:
                        context_time = self._get_beat_time_token_from_context(timing_context["tokens"][0], input_ids[0, prompt_length - post_tokens.shape[1]:-2])
                        if context_time is not None:
                            input_ids[0, time_offset] = context_time

            # Trim prompt and EOS tokens
            predicted_tokens = input_ids[:, prompt_length:-1]
            result = self._decode(predicted_tokens[0], frame_time)
            events += result
            update_event_times(events, event_times, frame_time + self.eos_time, self.types_first)

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
        context_tokens = context_tokens.cpu()
        generated_tokens = generated_tokens.cpu()

        # Search generated tokens in reverse order for the latest time shift token followed by a beat or measure token
        latest_time = -1000
        beat_offset = -1 if self.types_first else 1
        for i in range(len(generated_tokens) - 1, -1, -1):
            token = generated_tokens[i]
            if (token in self.time_range and
                    0 <= i + beat_offset < len(generated_tokens) and generated_tokens[i + beat_offset] in self.beat_range):
                latest_time = token
                break

        # Search context tokens in order for the first time shift token after latest_time which is followed by a beat or measure token
        for i, token in enumerate(context_tokens):
            if (token in self.time_range and token > latest_time + 1 and
                    0 <= i + beat_offset < len(context_tokens) and context_tokens[i + beat_offset] in self.beat_range):
                return token
