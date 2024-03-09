import numpy as np
from omegaconf import DictConfig

from .event import event_ranges, Event, EventType, EventRange


class Tokenizer:
    def __init__(self, args: DictConfig):
        """Fixed vocabulary tokenizer."""
        self._offset = 2
        self.input_event_ranges: list[EventRange] = [
            EventRange(EventType.STYLE, 0, args.control.num_classes),
            EventRange(EventType.DIFFICULTY, 0, args.control.num_diff_classes),
        ]
        self.num_classes = args.control.num_classes
        self.num_diff_classes = args.control.num_diff_classes
        self.max_difficulty = args.control.max_diff
        self.beatmap_idx: dict[int, int] = {}

        self.event_range: dict[EventType, EventRange] = {er.type: er for er in event_ranges} | {er.type: er for er in self.input_event_ranges}

        self.event_start: dict[EventType, int] = {}
        self.event_end: dict[EventType, int] = {}
        offset = self._offset
        for er in event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset
        for er in self.input_event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset

        self.vocab_size_out: int = self._offset + sum(
            er.max_value - er.min_value + 1 for er in event_ranges
        )
        self.vocab_size_in: int = self.vocab_size_out + sum(
            er.max_value - er.min_value + 1 for er in self.input_event_ranges
        )

    @property
    def pad_id(self) -> int:
        """[PAD] token for padding."""
        return 0

    @property
    def eos_id(self) -> int:
        """[EOS] token for end-of-sequence."""
        return 1

    def decode(self, token_id: int) -> Event:
        """Converts token ids into Event objects."""
        offset = self._offset
        for er in event_ranges:
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

    @property
    def style_unk(self) -> int:
        """Gets the unknown style value token id."""
        return self.encode(Event(type=EventType.STYLE, value=self.num_diff_classes))


