import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pandas import DataFrame
from tqdm import tqdm

from .event import Event, EventType, EventRange, ContextType

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


class Tokenizer:
    __slots__ = [
        "offset",
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
        "context_sos",
        "context_eos",
        "mapper_idx",
        "beatmap_mapper",
        "num_mapper_classes",
        "beatmap_descriptors",
        "descriptor_idx",
        "num_descriptor_classes",
        "num_cs_classes",
        "metadata",
    ]

    def __init__(self, args: DictConfig = None):
        """Fixed vocabulary tokenizer."""
        self.offset = 3
        self.beatmap_idx: dict[int, int] = {}  # beatmap_id -> beatmap_idx
        self.context_sos: dict[ContextType, int] = {}
        self.context_eos: dict[ContextType, int] = {}
        self.event_ranges: list[EventRange] = [
            EventRange(EventType.TIME_SHIFT, -512, 512),
            EventRange(EventType.SNAPPING, 0, 16),
            EventRange(EventType.DISTANCE, 0, 640),
        ]
        self.input_event_ranges: list[EventRange] = []
        self.num_classes = 0
        self.num_diff_classes = 0
        self.max_difficulty = 0
        self.beatmap_mapper: dict[int, int] = {}  # beatmap_id -> mapper_id
        self.mapper_idx: dict[int, int] = {}  # mapper_id -> mapper_idx
        self.num_mapper_classes = 0
        self.beatmap_descriptors: dict[int, list[int]] = {}  # beatmap_id -> [descriptor_idx]
        self.descriptor_idx: dict[str, int] = {}  # descriptor_name -> descriptor_idx
        self.num_descriptor_classes = 0
        self.num_cs_classes = 0

        if args is not None:
            def add_context_type(cts):
                ct = ContextType(cts)
                if ct not in self.context_sos:
                    self.context_sos[ct] = self.offset
                    self.offset += 1
                    self.context_eos[ct] = self.offset
                    self.offset += 1

            for cts in args.data.context_types:
                if isinstance(cts, str):
                    add_context_type(cts)
                else:
                    for ctss in cts["in"] + [cts["out"]]:
                        add_context_type(ctss)

            miliseconds_per_sequence = ((args.data.src_seq_len - 1) * args.model.spectrogram.hop_length *
                                        MILISECONDS_PER_SECOND / args.model.spectrogram.sample_rate)
            max_time_shift = int(miliseconds_per_sequence / MILISECONDS_PER_STEP)
            min_time_shift = -max_time_shift if args.data.add_pre_tokens or args.data.add_pre_tokens_at_step >= 0 else 0

            if args.data.dataset_type == "mmrs":
                self.metadata = self._get_metadata(args)

            self.event_ranges = [
                EventRange(EventType.TIME_SHIFT, min_time_shift, max_time_shift),
                EventRange(EventType.SNAPPING, 0, 16),
            ]
            self.input_event_ranges: list[EventRange] = []

            if args.data.gamemodes != [0]:
                self.input_event_ranges.append(EventRange(EventType.GAMEMODE, 0, 3))

            if args.model.do_style_embed or args.data.add_style_token:
                self._init_beatmap_idx(args)
                self.num_classes = args.data.num_classes
                if args.data.add_style_token:
                    self.input_event_ranges.append(EventRange(EventType.STYLE, 0, self.num_classes))

            if args.data.add_diff_token:
                self.num_diff_classes = args.data.num_diff_classes
                self.max_difficulty = args.data.max_diff
                self.input_event_ranges.append(EventRange(EventType.DIFFICULTY, 0, self.num_diff_classes))

            if args.data.add_mapper_token:
                self._init_mapper_idx(args)
                self.input_event_ranges.append(EventRange(EventType.MAPPER, 0, self.num_mapper_classes))

            if args.data.add_cs_token:
                self.num_cs_classes = args.data.num_cs_classes
                self.input_event_ranges.append(EventRange(EventType.CS, 0, self.num_cs_classes))

            if args.data.add_year_token:
                self.input_event_ranges.append(EventRange(EventType.YEAR, 2007, 2077))

            if args.data.add_hitsounded_token:
                self.input_event_ranges.append(EventRange(EventType.HITSOUNDED, 0, 1))

            if args.data.add_descriptors:
                self._init_descriptor_idx(args)
                self.input_event_ranges.append(EventRange(EventType.DESCRIPTOR, 0, self.num_descriptor_classes))

            if args.data.add_distances:
                self.event_ranges.append(EventRange(EventType.DISTANCE, 0, 640))

            if args.data.add_positions:
                p = args.data.position_precision
                x_min, x_max, y_min, y_max = args.data.position_range
                x_min, x_max, y_min, y_max = x_min // p, x_max // p, y_min // p, y_max // p

                if args.data.position_split_axes:
                    self.event_ranges.append(EventRange(EventType.POS_X, x_min, x_max))
                    self.event_ranges.append(EventRange(EventType.POS_Y, y_min, y_max))
                else:
                    x_count = x_max - x_min + 1
                    y_count = y_max - y_min + 1
                    self.event_ranges.append(EventRange(EventType.POS, 0, x_count * y_count - 1))

            if 3 in args.data.gamemodes:
                self.input_event_ranges.append(EventRange(EventType.MANIA_KEYCOUNT, 1, 18))
                self.input_event_ranges.append(EventRange(EventType.HOLD_NOTE_RATIO, -1, 12))
                self.event_ranges.append(EventRange(EventType.MANIA_COLUMN, 0, 17))

            if 1 in args.data.gamemodes or 3 in args.data.gamemodes:
                self.input_event_ranges.append(EventRange(EventType.SCROLL_SPEED_RATIO, -1, 12))
                self.event_ranges.append(EventRange(EventType.SCROLL_SPEED, 0, 1000))

        self.event_ranges: list[EventRange] = self.event_ranges + [
            EventRange(EventType.NEW_COMBO, 0, 0),
            EventRange(EventType.HITSOUND, 0, 2 ** 3 * 3 * 3),
            EventRange(EventType.VOLUME, 0, 100),
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
            EventRange(EventType.BEAT, 0, 0),
            EventRange(EventType.MEASURE, 0, 0),
        ]

        # These are placed after the event ranges so they count as 'other' when calculating validation accuracy
        if args is not None:
            if args.data.add_timing_points:
                self.event_ranges.append(EventRange(EventType.TIMING_POINT, 0, 0))

            if args.data.add_kiai:
                self.event_ranges.append(EventRange(EventType.KIAI, 0, 1))

            if 3 in args.data.gamemodes:
                self.event_ranges.append(EventRange(EventType.HOLD_NOTE, 0, 0))
                self.event_ranges.append(EventRange(EventType.HOLD_NOTE_END, 0, 0))
                self.event_ranges.append(EventRange(EventType.SCROLL_SPEED_CHANGE, 0, 0))

            if 1 in args.data.gamemodes:
                self.event_ranges.append(EventRange(EventType.DRUMROLL, 0, 0))
                self.event_ranges.append(EventRange(EventType.DRUMROLL_END, 0, 0))
                self.event_ranges.append(EventRange(EventType.DENDEN, 0, 0))
                self.event_ranges.append(EventRange(EventType.DENDEN_END, 0, 0))

        self.event_range: dict[EventType, EventRange] = {er.type: er for er in self.event_ranges} | {er.type: er for er in self.input_event_ranges}

        self.event_start: dict[EventType, int] = {}
        self.event_end: dict[EventType, int] = {}
        offset = self.offset
        for er in self.event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset
        for er in self.input_event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset

        self.vocab_size_out: int = self.offset + sum(
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
        offset = self.offset
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

    def decode_diff(self, token_id: int) -> float:
        """Converts token id into difficulty value."""
        if token_id == self.diff_unk:
            return -1
        elif not (self.event_start[EventType.DIFFICULTY] <= token_id < self.event_end[EventType.DIFFICULTY]):
            raise ValueError(f"token id {token_id} is not a difficulty token")
        return self.decode(token_id).value * self.max_difficulty / self.num_diff_classes

    def encode_gamemode(self, gamemode: int) -> int:
        """Converts gamemode into token id."""
        return self.encode(Event(type=EventType.GAMEMODE, value=gamemode))

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

    def encode_mapper_event(self, beatmap_id: int) -> Event:
        """Converts beatmap id into mapper event."""
        user_id = self.beatmap_mapper.get(beatmap_id, -1)
        mapper_idx = self.mapper_idx.get(user_id, self.num_mapper_classes)
        return Event(type=EventType.MAPPER, value=mapper_idx)

    def encode_mapper(self, beatmap_id: int) -> int:
        """Converts beatmap id into token id."""
        return self.encode(self.encode_mapper_event(beatmap_id))

    def encode_mapper_id(self, user_id: int) -> int:
        """Converts user id into token id."""
        mapper_idx = self.mapper_idx.get(user_id, self.num_mapper_classes)
        return self.encode(Event(type=EventType.MAPPER, value=mapper_idx))

    @property
    def mapper_unk(self) -> int:
        """Gets the unknown mapper value token id."""
        return self.encode(Event(type=EventType.MAPPER, value=self.num_mapper_classes))

    def encode_cs_event(self, cs: float) -> Event:
        """Converts circle size value into event."""
        return Event(type=EventType.CS, value=np.clip(int(cs * (self.num_cs_classes - 1) / 10), 0, self.num_cs_classes - 1))

    def encode_cs(self, cs: float) -> int:
        """Converts circle size value into token id."""
        return self.encode(self.encode_cs_event(cs))

    @property
    def cs_unk(self) -> int:
        """Gets the unknown circle size value token id."""
        return self.encode(Event(type=EventType.CS, value=self.num_cs_classes))

    def encode_descriptor_events(self, beatmap_id: int) -> list[Event]:
        """Converts beatmap id into descriptor events."""
        return [Event(type=EventType.DESCRIPTOR, value=descriptor_idx) for descriptor_idx in self.beatmap_descriptors.get(beatmap_id, [self.num_descriptor_classes])]

    def encode_descriptor(self, beatmap_id: int) -> list[int]:
        """Converts beatmap id into token ids."""
        return [self.encode(event) for event in self.encode_descriptor_events(beatmap_id)]

    def encode_descriptor_name(self, descriptor: str) -> int:
        """Converts descriptor into token id."""
        descriptor_idx = self.descriptor_idx.get(descriptor, self.num_descriptor_classes)
        return self.encode_descriptor_idx(descriptor_idx)

    def encode_descriptor_idx(self, descriptor_idx: int) -> int:
        """Converts descriptor idx into token id."""
        return self.encode(Event(type=EventType.DESCRIPTOR, value=descriptor_idx))

    @property
    def descriptor_unk(self) -> int:
        """Gets the unknown descriptor value token id."""
        return self.encode(Event(type=EventType.DESCRIPTOR, value=self.num_descriptor_classes))

    def descriptor_name(self, descriptor_idx: int) -> str:
        """Converts descriptor idx into descriptor."""
        for descriptor_name, idx in self.descriptor_idx.items():
            if idx == descriptor_idx:
                return descriptor_name
        return "unknown"

    @property
    def year_unk(self) -> int:
        """Gets the unknown year value token id."""
        return self.encode(Event(type=EventType.YEAR, value=2077))

    def encode_year(self, year: int) -> int:
        """Converts year into token id."""
        return self.encode(Event(type=EventType.YEAR, value=year))

    @property
    def hold_note_ratio_unk(self) -> int:
        """Gets the unknown hold note ratio value token id."""
        return self.encode(Event(type=EventType.HOLD_NOTE_RATIO, value=12))

    def encode_hold_note_ratio(self, hold_note_ratio: float) -> int:
        """Converts hold note ratio into token id."""
        value = self.ratio_to_value(hold_note_ratio, 10)
        return self.encode(Event(type=EventType.HOLD_NOTE_RATIO, value=value))

    @property
    def scroll_speed_ratio_unk(self) -> int:
        """Gets the unknown scroll speed ratio value token id."""
        return self.encode(Event(type=EventType.SCROLL_SPEED_RATIO, value=12))

    def encode_scroll_speed_ratio(self, scroll_speed_ratio: float) -> int:
        """Converts scroll speed ratio into token id."""
        value = self.ratio_to_value(scroll_speed_ratio, 10)
        return self.encode(Event(type=EventType.SCROLL_SPEED_RATIO, value=value))

    def ratio_to_value(self, ratio: float, resolution: int) -> int:
        value = np.clip(round(ratio * resolution), 0, resolution)
        if ratio <= 0:
            value = -1
        elif ratio >= 1:
            value = resolution + 1
        return value

    def _init_beatmap_idx(self, args: DictConfig) -> None:
        """Initializes and caches the beatmap index."""
        if args.data.dataset_type == "ors":
            self._init_beatmap_idx_ors(args)
        elif args.data.dataset_type == "mmrs":
            self._init_beatmap_idx_mmrs(args)

    def _init_beatmap_idx_ors(self, args: DictConfig) -> None:
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
            pickle.dump(self.beatmap_idx, f)  # type: ignore

    def _init_beatmap_idx_mmrs(self, args: DictConfig) -> None:
        self.beatmap_idx = self.metadata.reset_index().set_index(["Id"])["BeatmapIdx"].to_dict()

    def _get_metadata(self, args: DictConfig) -> DataFrame:
        df = pd.read_parquet(Path(args.data.train_dataset_path) / "metadata.parquet")
        df["BeatmapIdx"] = df.index
        df.set_index(["BeatmapSetId", "Id"], inplace=True)
        df.sort_index(inplace=True)

        sets = df.index.to_frame()["BeatmapSetId"].unique().tolist()
        sets = sets[args.data.train_dataset_start:args.data.train_dataset_end]

        return df.loc[sets]

    def _init_mapper_idx(self, args):
        """Indexes beatmap mappers and mapper idx."""
        if args.data.dataset_type == "ors":
            self._init_mapper_idx_ors(args)
        elif args.data.dataset_type == "mmrs":
            self._init_mapper_idx_mmrs(args)

    def _init_mapper_idx_ors(self, args):
        if args is None or "mappers_path" not in args.data:
            raise ValueError("mappers_path not found in args")

        path = Path(args.data.mappers_path)

        if not path.exists():
            raise ValueError(f"mappers_path {path} not found")

        # Load JSON data from file
        with open(path, 'r') as file:
            data = json.load(file)

        # Populate beatmap_mapper
        for item in data:
            self.beatmap_mapper[item['id']] = item['user_id']

        # Get unique user_ids from beatmap_mapper values
        unique_user_ids = list(set(self.beatmap_mapper.values()))

        # Create mapper_idx
        self.mapper_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.num_mapper_classes = len(unique_user_ids)

    def _init_mapper_idx_mmrs(self, args):
        self.beatmap_mapper = self.metadata.reset_index().set_index(["Id"])["UserId"].to_dict()

        # Get unique user_ids from beatmap_mapper values
        unique_user_ids = self.metadata["UserId"].unique().tolist()

        # Create mapper_idx
        self.mapper_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.num_mapper_classes = len(unique_user_ids)

    def _init_descriptor_idx(self, args):
        """"Indexes beatmap descriptors and descriptor idx."""
        if args.data.dataset_type == "ors":
            self._init_descriptor_idx_ors(args)
        elif args.data.dataset_type == "mmrs":
            self._init_descriptor_idx_mmrs(args)

    def _init_descriptor_idx_ors(self, args):
        if args is None or "descriptors_path" not in args.data:
            raise ValueError("descriptors_path not found in args")

        path = Path(args.data.descriptors_path)

        if not path.exists():
            raise ValueError(f"descriptors_path {path} not found")

        # The descriptors file is a CSV file with the following format:
        # beatmap_id,descriptor_name
        with open(path, 'r') as file:
            data = file.readlines()

        # Populate descriptor_idx
        for line in data:
            _, descriptor_name = line.strip().split(',')
            if descriptor_name not in self.descriptor_idx:
                self.descriptor_idx[descriptor_name] = len(self.descriptor_idx)

        # Populate beatmap_descriptors
        for line in data:
            beatmap_id_str, descriptor_name = line.strip().split(',')
            beatmap_id = int(beatmap_id_str)
            descriptor_idx = self.descriptor_idx[descriptor_name]
            if beatmap_id not in self.beatmap_descriptors:
                self.beatmap_descriptors[beatmap_id] = []
            self.beatmap_descriptors[beatmap_id].append(descriptor_idx)

        self.num_descriptor_classes = len(self.descriptor_idx)

    def _init_descriptor_idx_mmrs(self, args):
        # Populate descriptor_idx
        descriptors = self.metadata["OmdbTags"].explode().dropna().unique()
        for descriptor_name in descriptors:
            self.descriptor_idx[descriptor_name] = len(self.descriptor_idx)

        # Populate beatmap_descriptors
        self.beatmap_descriptors = (self.metadata.reset_index().set_index(["Id"])["OmdbTags"]
                                    .apply(lambda x: None if np.count_nonzero(x) == 0 else [self.descriptor_idx[y] for y in x]).dropna().to_dict())

        self.num_descriptor_classes = len(self.descriptor_idx)

    def state_dict(self):
        return {
            "offset": self.offset,
            "context_sos": self.context_sos,
            "context_eos": self.context_eos,
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
            "beatmap_mapper": self.beatmap_mapper,
            "mapper_idx": self.mapper_idx,
            "num_mapper_classes": self.num_mapper_classes,
            "beatmap_descriptors": self.beatmap_descriptors,
            "descriptor_idx": self.descriptor_idx,
            "num_descriptor_classes": self.num_descriptor_classes,
            "num_cs_classes": self.num_cs_classes,
        }

    def load_state_dict(self, state_dict):
        if "offset" in state_dict:
            self.offset = state_dict["offset"]
        else:
            # Backward compatibility. Old models use offset 3.
            self.offset = 3
        if "context_sos" in state_dict:
            self.context_sos = state_dict["context_sos"]
        if "context_eos" in state_dict:
            self.context_eos = state_dict["context_eos"]
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
        if "beatmap_mapper" in state_dict:
            self.beatmap_mapper = state_dict["beatmap_mapper"]
        if "mapper_idx" in state_dict:
            self.mapper_idx = state_dict["mapper_idx"]
        if "num_mapper_classes" in state_dict:
            self.num_mapper_classes = state_dict["num_mapper_classes"]
        if "beatmap_descriptors" in state_dict:
            self.beatmap_descriptors = state_dict["beatmap_descriptors"]
        if "descriptor_idx" in state_dict:
            self.descriptor_idx = state_dict["descriptor_idx"]
        if "num_descriptor_classes" in state_dict:
            self.num_descriptor_classes = state_dict["num_descriptor_classes"]
        if "num_cs_classes" in state_dict:
            self.num_cs_classes = state_dict["num_cs_classes"]
