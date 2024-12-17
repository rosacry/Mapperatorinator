from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from slider import Beatmap
from tqdm import tqdm
from transformers import LogitsProcessorList, LogitsProcessor

from ..dataset import OsuParser
from ..dataset.data_utils import update_event_times, remove_events_of_type, get_hold_note_ratio, get_scroll_speed_ratio, \
    events_of_type, get_hitsounded_status
from ..model import OsuT
from ..tokenizer import Event, EventType, Tokenizer, ContextType

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


@dataclass
class GenerationConfig:
    gamemode: int = -1
    beatmap_id: int = -1
    difficulty: float = -1
    mapper_id: int = -1
    year: int = -1
    hitsounded: bool = True
    slider_multiplier: float = 1.4
    circle_size: float = -1
    keycount: int = -1
    hold_note_ratio: float = -1
    scroll_speed_ratio: float = -1
    descriptors: list[str] = None
    negative_descriptors: list[str] = None


def generation_config_from_beatmap(beatmap: Beatmap, tokenizer: Tokenizer) -> GenerationConfig:
    gamemode = int(beatmap.mode)
    return GenerationConfig(
        gamemode=gamemode,
        beatmap_id=beatmap.beatmap_id,
        difficulty=float(beatmap.stars()) if gamemode == 0 else -1,  # We don't have diffcalc for other gamemodes
        mapper_id=tokenizer.beatmap_mapper.get(beatmap.beatmap_id, -1),
        slider_multiplier=beatmap.slider_multiplier,
        circle_size=beatmap.circle_size,
        hitsounded=get_hitsounded_status(beatmap),
        keycount=int(beatmap.circle_size),
        hold_note_ratio=get_hold_note_ratio(beatmap) if gamemode == 3 else -1,
        scroll_speed_ratio=get_scroll_speed_ratio(beatmap) if gamemode in [1, 3] else -1,
        descriptors=[tokenizer.descriptor_name(idx) for idx in tokenizer.beatmap_descriptors.get(beatmap.beatmap_id, [])],
    )


def get_beat_type_tokens(tokenizer: Tokenizer) -> list[int]:
    beat_range = [
        tokenizer.event_start[EventType.BEAT],
        tokenizer.event_start[EventType.MEASURE],
    ]
    if EventType.TIMING_POINT in tokenizer.event_start:
        beat_range.append(tokenizer.event_start[EventType.TIMING_POINT])
    return beat_range


def get_mania_type_tokens(tokenizer: Tokenizer) -> list[int]:
    return [
        tokenizer.event_start[EventType.CIRCLE],
        tokenizer.event_start[EventType.HOLD_NOTE],
        tokenizer.event_start[EventType.HOLD_NOTE_END],
    ] if EventType.HOLD_NOTE_END in tokenizer.event_start else []


def get_scroll_speed_tokens(tokenizer: Tokenizer) -> range:
    return range(tokenizer.event_start[EventType.SCROLL_SPEED], tokenizer.event_end[EventType.SCROLL_SPEED])


class TimeshiftBias(LogitsProcessor):
    def __init__(self, timeshift_bias: float, time_range: range):
        self.timeshift_bias = timeshift_bias
        self.time_range = time_range

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        scores_processed = scores.clone()
        scores_processed[:, self.time_range] += self.timeshift_bias
        return scores_processed


class ConditionalTemperatureLogitsWarper(LogitsProcessor):
    def __init__(
            self,
            args: DictConfig,
            tokenizer: Tokenizer,
            gamemode: int,
    ):
        self.gamemode = gamemode
        self.temperature = args.temperature
        self.conditionals = []

        if args.osut5.data.add_timing:
            self.conditionals.append((args.timing_temperature, get_beat_type_tokens(tokenizer), 1))
        if gamemode == 3:
            self.conditionals.append((args.mania_column_temperature, get_mania_type_tokens(tokenizer), 3))
        if gamemode == 1:
            self.conditionals.append((args.taiko_hit_temperature, get_scroll_speed_tokens(tokenizer), 1))

        if not args.osut5.data.types_first:
            print("WARNING: Conditional temperature is not supported for types_first=False. Ignoring.")
            self.conditionals = []

        self.max_offset = max([offset for _, _, offset in self.conditionals], default=0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if len(self.conditionals) > 0:
            lookback = input_ids[0, -self.max_offset:].cpu()
            for temperature, tokens, offset in self.conditionals:
                if len(lookback) >= offset and lookback[-offset] in tokens:
                    return scores / temperature

        return scores / self.temperature


class Processor(object):
    def __init__(self, args: DictConfig, model: OsuT, tokenizer: Tokenizer):
        """Model inference stage that processes sequences."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
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
        self.lookahead_time_range = range(tokenizer.encode(Event(EventType.TIME_SHIFT, int(self.lookahead_max_time / MILISECONDS_PER_STEP))), tokenizer.event_end[EventType.TIME_SHIFT])
        self.eos_time = (1 - args.osut5.data.lookahead) * self.miliseconds_per_sequence
        self.center_pad_decoder = args.osut5.data.center_pad_decoder
        self.add_gamemode_token = args.osut5.data.add_gamemode_token
        self.add_style_token = args.osut5.data.add_style_token
        self.add_diff_token = args.osut5.data.add_diff_token
        self.add_mapper_token = args.osut5.data.add_mapper_token
        self.add_year_token = args.osut5.data.add_year_token
        self.add_hitsounded_token = args.osut5.data.add_hitsounded_token
        self.add_song_length_token = args.osut5.data.add_song_length_token
        self.add_song_position_token = args.osut5.data.add_song_position_token
        self.add_cs_token = args.osut5.data.add_cs_token
        self.add_descriptors = args.osut5.data.add_descriptors
        self.add_kiai = args.osut5.data.add_kiai
        self.max_pre_token_len = args.osut5.data.max_pre_token_len
        self.add_pre_tokens = args.osut5.data.add_pre_tokens
        self.add_gd_context = args.osut5.data.add_gd_context
        self.add_timing = args.osut5.data.add_timing
        self.parser = OsuParser(args.osut5, self.tokenizer)
        self.need_beatmap_idx = args.osut5.model.do_style_embed
        self.add_positions = args.osut5.data.add_positions
        self.add_sv_special_token = args.osut5.data.add_sv_special_token
        self.add_sv = args.osut5.data.add_sv

        if self.add_positions:
            self.position_precision = args.osut5.data.position_precision
            x_min, x_max, y_min, y_max = args.osut5.data.position_range
            self.x_min = x_min / self.position_precision
            self.x_max = x_max / self.position_precision
            self.y_min = y_min / self.position_precision
            self.y_max = y_max / self.position_precision
            self.x_count = self.x_max - self.x_min + 1

        self.cfg_scale = args.cfg_scale
        self.top_p = args.top_p

        self.timeshift_bias = args.timeshift_bias
        self.time_range = range(tokenizer.event_start[EventType.TIME_SHIFT], tokenizer.event_end[EventType.TIME_SHIFT])
        self.beat_type_tokens = get_beat_type_tokens(tokenizer)
        self.types_first = args.osut5.data.types_first

        self.logit_processor = LogitsProcessorList()
        if self.timeshift_bias != 0:
            self.logit_processor.append(TimeshiftBias(self.timeshift_bias, self.time_range))

    def get_context(self, context: ContextType, beatmap_path, song_length: float, add_type=True):
        beatmap_path = Path(beatmap_path)
        if context != ContextType.NONE and not beatmap_path.is_file():
            raise FileNotFoundError(f"Beatmap file {beatmap_path} not found.")

        data = {"context_type": ContextType(context), "add_type": add_type}

        if context == ContextType.NONE:
            data["events"], data["event_times"] = [], []
        elif context == ContextType.TIMING:
            beatmap = Beatmap.from_path(beatmap_path)
            data["events"], data["event_times"] = self.parser. parse_timing(beatmap)
        elif context == ContextType.NO_HS:
            beatmap = Beatmap.from_path(beatmap_path)
            hs_events, hs_event_times = self.parser.parse(beatmap)
            data["events"], data["event_times"] = remove_events_of_type(hs_events, hs_event_times,
                                                                        [EventType.HITSOUND, EventType.VOLUME])
        elif context == ContextType.GD:
            beatmap = Beatmap.from_path(beatmap_path)
            data["events"], data["event_times"] = self.parser.parse(beatmap)
            data["class"] = self.get_class_vector(generation_config_from_beatmap(beatmap, self.tokenizer), song_length)
            if self.add_kiai:
                data["kiai_events"], data["kiai_event_times"] = events_of_type(data["events"], data["event_times"], EventType.KIAI)
            if self.add_sv_special_token:
                data["sv_events"], data["sv_event_times"] = events_of_type(data["events"], data["event_times"], EventType.SCROLL_SPEED)
        else:
            raise ValueError(f"Invalid context type {context}")
        return data

    def get_in_context(
            self,
            in_context: list[ContextType],
            beatmap_path: Path,
            song_length: float,
    ) -> list[dict[str, Any]]:
        in_context = [self.get_context(context, beatmap_path, song_length) for context in in_context]
        if self.add_gd_context:
            in_context.append(self.get_context(ContextType.GD, beatmap_path, song_length, add_type=False))
        return in_context

    def get_class_vector(
            self,
            config: GenerationConfig,
            song_length: float,
            verbose: bool = False,
    ):
        cond_tokens = []

        if self.add_gamemode_token:
            gamemode_token = self.tokenizer.encode_gamemode(config.gamemode)
            cond_tokens.append(gamemode_token)
        if self.add_style_token:
            style_token = self.tokenizer.encode_style(config.beatmap_id) if config.beatmap_id != -1 else self.tokenizer.style_unk
            cond_tokens.append(style_token)
            if config.beatmap_id != -1 and config.beatmap_id not in self.tokenizer.beatmap_idx and verbose:
                print(f"Beatmap class {config.beatmap_id} not found. Using default.")
        if self.add_diff_token:
            diff_token = self.tokenizer.encode_diff(config.difficulty) if config.difficulty != -1 else self.tokenizer.diff_unk
            cond_tokens.append(diff_token)
        if self.add_mapper_token:
            mapper_token = self.tokenizer.encode_mapper_id(config.mapper_id) if config.mapper_id != -1 else self.tokenizer.mapper_unk
            cond_tokens.append(mapper_token)
            if config.mapper_id != -1 and config.mapper_id not in self.tokenizer.beatmap_mapper and verbose:
                print(f"Mapper class {config.mapper_id} not found. Using default.")
        if self.add_year_token:
            year_token = self.tokenizer.encode_year(config.year) if config.year != -1 else self.tokenizer.year_unk
            cond_tokens.append(year_token)
        if self.add_hitsounded_token:
            hitsounded_token = self.tokenizer.encode(Event(EventType.HITSOUNDED, int(config.hitsounded)))
            cond_tokens.append(hitsounded_token)
        if self.add_song_length_token:
            song_length_token = self.tokenizer.encode_song_length(song_length)
            cond_tokens.append(song_length_token)
        if self.add_sv and config.gamemode in [0, 2]:
            global_sv_token = self.tokenizer.encode_global_sv(config.slider_multiplier)
            cond_tokens.append(global_sv_token)
        if self.add_cs_token and config.gamemode in [0, 2]:
            cs_token = self.tokenizer.encode_cs(config.circle_size) if config.circle_size != -1 else self.tokenizer.cs_unk
            cond_tokens.append(cs_token)
        if config.gamemode == 3:
            keycount_token = self.tokenizer.encode(Event(EventType.MANIA_KEYCOUNT, config.keycount))
            cond_tokens.append(keycount_token)
            hold_note_ratio_token = self.tokenizer.encode_hold_note_ratio(config.hold_note_ratio) if config.hold_note_ratio != -1 else self.tokenizer.hold_note_ratio_unk
            cond_tokens.append(hold_note_ratio_token)
        if config.gamemode in [1, 3]:
            scroll_speed_ratio_token = self.tokenizer.encode_scroll_speed_ratio(config.scroll_speed_ratio) if config.scroll_speed_ratio != -1 else self.tokenizer.scroll_speed_ratio_unk
            cond_tokens.append(scroll_speed_ratio_token)

        descriptors = config.descriptors if config.descriptors is not None else []
        descriptors_added = 0
        if self.add_descriptors:
            if descriptors is not None and len(descriptors) > 0:
                for descriptor in descriptors:
                    if isinstance(descriptor, str):
                        if descriptor not in self.tokenizer.descriptor_idx:
                            if verbose:
                                print(f"Descriptor class {descriptor} not found. Skipping.")
                            continue
                        cond_tokens.append(self.tokenizer.encode_descriptor_name(descriptor))
                        descriptors_added += 1
                    elif isinstance(descriptor, int):
                        if descriptor < self.tokenizer.event_range[EventType.DESCRIPTOR].min_value or \
                                descriptor > self.tokenizer.event_range[EventType.DESCRIPTOR].max_value:
                            if verbose:
                                print(f"Descriptor idx {descriptor} out of range. Skipping.")
                            continue
                        cond_tokens.append(self.tokenizer.encode_descriptor_idx(descriptor))
                        descriptors_added += 1
            if descriptors is None or descriptors_added == 0:
                cond_tokens.append(self.tokenizer.descriptor_unk)

        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        return cond_tokens

    def generate(
            self,
            *,
            sequences: torch.Tensor,
            generation_config: GenerationConfig,
            in_context: list[dict[str, Any]] = None,
            verbose: bool = True,
    ) -> list[Event]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            sequences: A list of batched source sequences.
            generation_config: Generation configuration.
            in_context: List of context information.
            verbose: Whether to show progress bar.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """

        events = []
        event_times = []
        song_length = (len(sequences) - 1) * self.miliseconds_per_stride + self.miliseconds_per_sequence

        # Prepare logit processors
        logit_processor = LogitsProcessorList(self.logit_processor + [ConditionalTemperatureLogitsWarper(
            self.args,
            self.tokenizer,
            generation_config.gamemode,
        )])

        # Prepare special tokens
        beatmap_idx = torch.tensor([self.tokenizer.num_classes], dtype=torch.long, device=self.device)
        if self.need_beatmap_idx:
            beatmap_idx = torch.tensor([self.tokenizer.beatmap_idx[generation_config.beatmap_id]], dtype=torch.long, device=self.device)

        # Prepare unconditional prompt
        cond_tokens = self.get_class_vector(generation_config, song_length, verbose=verbose)
        uncond_tokens = self.get_class_vector(GenerationConfig(
            gamemode=generation_config.gamemode,
            difficulty=generation_config.difficulty,
            circle_size=generation_config.circle_size,
            hitsounded=generation_config.hitsounded,
            slider_multiplier=generation_config.slider_multiplier,
            keycount=generation_config.keycount,
            hold_note_ratio=generation_config.hold_note_ratio,
            scroll_speed_ratio=generation_config.scroll_speed_ratio,
            descriptors=generation_config.negative_descriptors,
        ), song_length)

        # Prepare context type indicator tokens
        def get_context_tokens(context):
            context_type = context["context_type"]
            tokens = context["tokens"]

            # Trim tokens if they are too long
            max_context_length = self.tgt_seq_len // 2
            if tokens.shape[1] > max_context_length:
                tokens = tokens[:, :max_context_length]

            to_concat = []
            if context["add_type"]:
                to_concat.append(torch.tensor([[self.tokenizer.context_sos[context_type]]], dtype=torch.long, device=self.device))

            if "class" in context:
                to_concat.append(context["class"])

            to_concat.append(context["extra_special_tokens"])
            to_concat.append(tokens)

            if context["add_type"]:
                to_concat.append(torch.tensor([[self.tokenizer.context_eos[context_type]]], dtype=torch.long, device=self.device))

            return torch.concatenate(to_concat, dim=-1)

        def get_prompt(user_prompt, extra_special_tokens, prev_tokens, post_tokens):
            to_concat = [get_context_tokens(context) for context in in_context] + [user_prompt, extra_special_tokens, prev_tokens]
            prefix = torch.concatenate(to_concat, dim=-1)

            if self.center_pad_decoder:
                prefix = F.pad(prefix, (self.tgt_seq_len // 2 - prefix.shape[1], 0), value=self.tokenizer.pad_id)

            prompt = torch.tensor([[self.tokenizer.sos_id]], dtype=torch.long, device=self.device)
            prompt = torch.concatenate([prefix, prompt, post_tokens], dim=-1)
            return prompt

        # Start generation
        iterator = tqdm(sequences) if verbose else sequences
        for sequence_index, frames in enumerate(iterator):
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

                # Prepare extra special tokens
                context_extra_special_events = []
                if self.add_kiai and "kiai_events" in context:
                    last_kiai = self._kiai_before_time(context["kiai_events"], context["kiai_event_times"], frame_time)
                    context_extra_special_events.append(last_kiai)
                if self.add_sv_special_token and "sv_events" in context:
                    last_sv = self._sv_before_time(context["sv_events"], context["sv_event_times"], frame_time)
                    context_extra_special_events.append(last_sv)
                if self.add_song_position_token and "class" in context:
                    context_extra_special_events.append(self.tokenizer.encode_song_position_event(frame_time, song_length))
                context["extra_special_tokens"] = self._encode(context_extra_special_events, frame_time)

            # Prepare extra special tokens
            extra_special_events = []
            if self.add_kiai:
                last_kiai = self._kiai_before_time(events, event_times, frame_time)
                extra_special_events.append(last_kiai)
            if self.add_sv_special_token:
                last_sv = self._sv_before_time(events, event_times, frame_time)
                extra_special_events.append(last_sv)
            if self.add_song_position_token:
                extra_special_events.append(self.tokenizer.encode_song_position_event(frame_time, song_length))
            extra_special_tokens = self._encode(extra_special_events, frame_time)

            # Prepare classifier-free guidance
            cond_prompt = get_prompt(cond_tokens, extra_special_tokens, prev_tokens, post_tokens)
            uncond_prompt = get_prompt(uncond_tokens, extra_special_tokens, prev_tokens, post_tokens) if self.cfg_scale > 1 else None

            # Make sure the prompt is not too long
            i = 0
            while cond_prompt.shape[1] >= self.tgt_seq_len:
                i += 1
                if i > 10:
                    raise ValueError("Prompt is too long.")
                prev_tokens = prev_tokens[:, -(prev_tokens.shape[1] // 2):]
                post_tokens = post_tokens[:, -(post_tokens.shape[1] // 2):]
                cond_prompt = get_prompt(cond_tokens, extra_special_tokens, prev_tokens, post_tokens)
                uncond_prompt = get_prompt(uncond_tokens, extra_special_tokens, prev_tokens, post_tokens) if self.cfg_scale > 1 else None

            eos_token_id = [self.tokenizer.eos_id]
            if sequence_index != len(sequences) - 1:
                eos_token_id += self.lookahead_time_range

            predicted_tokens = self.model.generate(
                frames,
                decoder_input_ids=cond_prompt,
                beatmap_idx=beatmap_idx,
                logits_processor=logit_processor,
                top_p=self.top_p,
                guidance_scale=self.cfg_scale,
                negative_prompt_ids=uncond_prompt,
                eos_token_id=eos_token_id,
                use_cache=True,
                cache_implementation="static",
            )
            # Only support batch size 1 for now
            predicted_tokens = predicted_tokens[0].cpu()

            # Trim prompt and eos tokens
            predicted_tokens = predicted_tokens[cond_prompt.shape[1]:]
            if predicted_tokens[-1] == self.tokenizer.eos_id:
                predicted_tokens = predicted_tokens[:-1]
            elif sequence_index != len(sequences) - 1 and predicted_tokens[-1] in self.lookahead_time_range:
                # If the type token comes before the timeshift token we should remove the type token too
                if self.types_first:
                    predicted_tokens = predicted_tokens[:-2]
                else:
                    predicted_tokens = predicted_tokens[:-1]

            result = self._decode(predicted_tokens, frame_time)
            events += result
            update_event_times(events, event_times, frame_time + self.eos_time, self.types_first)

            # Trim events which are in the lookahead window
            if sequence_index != len(sequences) - 1:
                lookahead_time = frame_time + self.lookahead_max_time
                self._trim_events_after_time(events, event_times, lookahead_time)

        # Post-process events
        # Rescale and unpack position events
        if self.add_positions:
            events = self._rescale_positions(events)

        # Turn mania key column into X position
        if generation_config.gamemode == 3:
            events = self._convert_column_to_position(events, generation_config.keycount)

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
        timeshift_range = self.tokenizer.event_range[EventType.TIME_SHIFT]
        for i, event in enumerate(events):
            if event.type == EventType.TIME_SHIFT:
                value = int((event.value - frame_time) / MILISECONDS_PER_STEP)
                value = np.clip(value, timeshift_range.min_value, timeshift_range.max_value)
                event = Event(type=event.type, value=value)
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
                    0 <= i + beat_offset < len(generated_tokens) and generated_tokens[i + beat_offset] in self.beat_type_tokens):
                latest_time = token
                break

        # Search context tokens in order for the first time shift token after latest_time which is followed by a beat or measure token
        for i, token in enumerate(context_tokens):
            if (token in self.time_range and token > latest_time + 1 and
                    0 <= i + beat_offset < len(context_tokens) and context_tokens[i + beat_offset] in self.beat_type_tokens):
                return token

    def _kiai_before_time(self, events, event_times, time) -> Event:
        for i in range(len(events) - 1, -1, -1):
            if events[i].type == EventType.KIAI and event_times[i] < time:
                return events[i]
        return Event(EventType.KIAI, 0)

    def _sv_before_time(self, events, event_times, time) -> Event:
        for i in range(len(events) - 1, -1, -1):
            if events[i].type == EventType.SCROLL_SPEED and event_times[i] < time:
                return events[i]
        return Event(EventType.SCROLL_SPEED, 100)

    def _convert_column_to_position(self, events, key_count) -> list[Event]:
        new_events = []
        for event in events:
            if event.type == EventType.MANIA_COLUMN:
                x = int((event.value + 0.5) * 512 / key_count)
                new_events.append(Event(EventType.POS_X, x))
                new_events.append(Event(EventType.POS_Y, 192))
            else:
                new_events.append(event)
        return new_events
