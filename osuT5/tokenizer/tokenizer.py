from __future__ import annotations

from .event import event_ranges, Event, EventType, event_range


class Tokenizer:
    def __init__(self):
        """Fixed vocabulary tokenizer."""
        self._offset = 3
        self.event_start: dict[EventType, int] = {}
        self.event_end: dict[EventType, int] = {}
        offset = self._offset
        for er in event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset
        self.vocab_size: int = self._offset + sum(
            er.max_value - er.min_value + 1 for er in event_ranges
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
        for er in event_ranges:
            if offset <= token_id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + token_id - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"id {token_id} is not mapped to any event")

    def encode(self, event: Event) -> int:
        """Converts Event objects into token ids."""
        if event.type not in event_range:
            raise ValueError(f"unknown event type: {event.type}")

        er = event_range[event.type]
        offset = self.event_start[event.type]

        if not er.min_value <= event.value <= er.max_value:
            raise ValueError(
                f"event value {event.value} is not within range "
                f"[{er.min_value}, {er.max_value}] for event type {event.type}"
            )

        return offset + event.value - er.min_value

    def event_type_range(self, event_type: EventType) -> tuple[int, int]:
        """Get the token id range of each Event type."""
        if event_type not in event_range:
            raise ValueError(f"unknown event type: {event_type}")

        er = event_range[event_type]
        offset = self.event_start[event_type]
        return offset, offset + (er.max_value - er.min_value)

