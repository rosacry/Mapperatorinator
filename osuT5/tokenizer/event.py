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
    STYLE = "style"
    DIFFICULTY = "difficulty"
    POS_X = "pos_x"
    POS_Y = "pos_y"


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
