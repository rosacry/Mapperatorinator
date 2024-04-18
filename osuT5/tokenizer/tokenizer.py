import json
import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from .event import Event, EventType, EventRange

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


class Tokenizer:
    __slots__ = [
        "_offset",
        "event_ranges",
        "input_event_ranges",
        "num_classes",
        "num_diff_classes",
        "max_difficulty",
        "event_range",
        "event_start",
        "event_end",
        "vocab_size_out",
        "vocab_size_in",
        "beatmap_idx",
    ]

    def __init__(self, args: DictConfig = None):
        """Fixed vocabulary tokenizer."""
        self._offset = 3
        self.beatmap_idx: dict[int, int] = {}

        if args is not None:
            miliseconds_per_sequence = ((args.data.src_seq_len - 1) * args.model.spectrogram.hop_length *
                                        MILISECONDS_PER_SECOND / args.model.spectrogram.sample_rate)
            max_time_shift = int(miliseconds_per_sequence / MILISECONDS_PER_STEP)
            min_time_shift = -max_time_shift if args.data.add_pre_tokens or args.data.add_pre_tokens_at_step >= 0 else 0
            self.event_ranges = [EventRange(EventType.TIME_SHIFT, min_time_shift, max_time_shift)]

            self.input_event_ranges: list[EventRange] = []
            if args.data.style_token_index >= 0:
                self.input_event_ranges.append(EventRange(EventType.STYLE, 0, args.data.num_classes))
            if args.data.diff_token_index >= 0:
                self.input_event_ranges.append(EventRange(EventType.DIFFICULTY, 0, args.data.num_diff_classes))

            self.num_classes = args.data.num_classes
            self.num_diff_classes = args.data.num_diff_classes
            self.max_difficulty = args.data.max_diff

            self._init_beatmap_idx(args)
        else:
            self.event_ranges = [EventRange(EventType.TIME_SHIFT, -512, 512)]
            self.input_event_ranges = []
            self.num_classes = 0
            self.num_diff_classes = 0
            self.max_difficulty = 0

        self.event_ranges: list[EventRange] = self.event_ranges + [
            EventRange(EventType.DISTANCE, 0, 640),
            EventRange(EventType.NEW_COMBO, 0, 0),
            EventRange(EventType.CIRCLE, 0, 0),
            EventRange(EventType.SPINNER, 0, 0),
            EventRange(EventType.SPINNER_END, 0, 0),
            EventRange(EventType.SLIDER_HEAD, 0, 0),
            EventRange(EventType.BEZIER_ANCHOR, 0, 0),
            EventRange(EventType.PERFECT_ANCHOR, 0, 0),
            EventRange(EventType.CATMULL_ANCHOR, 0, 0),
            EventRange(EventType.RED_ANCHOR, 0, 0),
            EventRange(EventType.LAST_ANCHOR, 0, 0),
            EventRange(EventType.SLIDER_END, 0, 0),
        ]

        self.event_range: dict[EventType, EventRange] = {er.type: er for er in self.event_ranges} | {er.type: er for er in self.input_event_ranges}

        self.event_start: dict[EventType, int] = {}
        self.event_end: dict[EventType, int] = {}
        offset = self._offset
        for er in self.event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset
        for er in self.input_event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset

        self.vocab_size_out: int = self._offset + sum(
            er.max_value - er.min_value + 1 for er in self.event_ranges
        )
        self.vocab_size_in: int = self.vocab_size_out + sum(
            er.max_value - er.min_value + 1 for er in self.input_event_ranges
        )

    @property
    def pad_id(self) -> int:
        """[PAD] token for padding."""
        return 0

    @property
    def sos_id(self) -> int:
        """[SOS] token for start-of-sequence."""
        return 1

    @property
    def eos_id(self) -> int:
        """[EOS] token for end-of-sequence."""
        return 2

    def decode(self, token_id: int) -> Event:
        """Converts token ids into Event objects."""
        offset = self._offset
        for er in self.event_ranges:
            if offset <= token_id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + token_id - offset)
            offset += er.max_value - er.min_value + 1
        for er in self.input_event_ranges:
            if offset <= token_id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + token_id - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"id {token_id} is not mapped to any event")

    def encode(self, event: Event) -> int:
        """Converts Event objects into token ids."""
        if event.type not in self.event_range:
            raise ValueError(f"unknown event type: {event.type}")

        er = self.event_range[event.type]
        offset = self.event_start[event.type]

        if not er.min_value <= event.value <= er.max_value:
            raise ValueError(
                f"event value {event.value} is not within range "
                f"[{er.min_value}, {er.max_value}] for event type {event.type}"
            )

        return offset + event.value - er.min_value

    def event_type_range(self, event_type: EventType) -> tuple[int, int]:
        """Get the token id range of each Event type."""
        if event_type not in self.event_range:
            raise ValueError(f"unknown event type: {event_type}")

        er = self.event_range[event_type]
        offset = self.event_start[event_type]
        return offset, offset + (er.max_value - er.min_value)

    def encode_diff_event(self, diff: float) -> Event:
        """Converts difficulty value into event."""
        return Event(type=EventType.DIFFICULTY, value=np.clip(
            int(diff * self.num_diff_classes / self.max_difficulty), 0, self.num_diff_classes - 1))

    def encode_diff(self, diff: float) -> int:
        """Converts difficulty value into token id."""
        return self.encode(self.encode_diff_event(diff))

    @property
    def diff_unk(self) -> int:
        """Gets the unknown difficulty value token id."""
        return self.encode(Event(type=EventType.DIFFICULTY, value=self.num_diff_classes))

    def encode_style_event(self, beatmap_id: int) -> Event:
        """Converts beatmap id into style event."""
        style_idx = self.beatmap_idx.get(beatmap_id, self.num_classes)
        return Event(type=EventType.STYLE, value=style_idx)

    def encode_style(self, beatmap_id: int) -> int:
        """Converts beatmap id into token id."""
        return self.encode(self.encode_style_event(beatmap_id))

    def encode_style_idx(self, beatmap_idx: int) -> int:
        """Converts beatmap idx into token id."""
        return self.encode(Event(type=EventType.STYLE, value=beatmap_idx))

    @property
    def style_unk(self) -> int:
        """Gets the unknown style value token id."""
        return self.encode(Event(type=EventType.STYLE, value=self.num_classes))

    def _init_beatmap_idx(self, args: DictConfig) -> None:
        """Initializes and caches the beatmap index."""
        if args is None or "train_dataset_path" not in args.data:
            return

        path = Path(args.data.train_dataset_path)
        cache_path = path / "beatmap_idx.pickle"

        if cache_path.exists():
            with open(path / "beatmap_idx.pickle", "rb") as f:
                self.beatmap_idx = pickle.load(f)
            return

        print("Caching beatmap index...")

        for track in tqdm(path.iterdir()):
            if not track.is_dir():
                continue
            metadata_file = track / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            for beatmap_name in metadata["Beatmaps"]:
                beatmap_metadata = metadata["Beatmaps"][beatmap_name]
                self.beatmap_idx[beatmap_metadata["BeatmapId"]] = beatmap_metadata["Index"]

        with open(cache_path, "wb") as f:
            pickle.dump(self.beatmap_idx, f)

    def state_dict(self):
        return {
            "event_ranges": self.event_ranges,
            "input_event_ranges": self.input_event_ranges,
            "num_classes": self.num_classes,
            "num_diff_classes": self.num_diff_classes,
            "max_difficulty": self.max_difficulty,
            "event_range": self.event_range,
            "event_start": self.event_start,
            "event_end": self.event_end,
            "vocab_size_out": self.vocab_size_out,
            "vocab_size_in": self.vocab_size_in,
            "beatmap_idx": self.beatmap_idx,
        }

    def load_state_dict(self, state_dict):
        self.event_ranges = state_dict["event_ranges"]
        self.input_event_ranges = state_dict["input_event_ranges"]
        self.num_classes = state_dict["num_classes"]
        self.num_diff_classes = state_dict["num_diff_classes"]
        self.max_difficulty = state_dict["max_difficulty"]
        self.event_range = state_dict["event_range"]
        self.event_start = state_dict["event_start"]
        self.event_end = state_dict["event_end"]
        self.vocab_size_out = state_dict["vocab_size_out"]
        self.vocab_size_in = state_dict["vocab_size_in"]
        self.beatmap_idx = state_dict["beatmap_idx"]
