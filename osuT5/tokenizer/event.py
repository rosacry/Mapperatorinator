from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    DISTANCE = "dist"
    NEW_COMBO = "new_combo"
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    BEZIER_ANCHOR = "bezier_anchor"
    PERFECT_ANCHOR = "perfect_anchor"
    CATMULL_ANCHOR = "catmull_anchor"
    RED_ANCHOR = "red_anchor"
    LAST_ANCHOR = "last_anchor"
    SLIDER_END = "slider_end"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"

    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"


event_ranges: list[EventRange] = [
    EventRange(EventType.TIME_SHIFT, -512, 512),
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

event_range: dict[EventType, EventRange] = {er.type: er for er in event_ranges}
