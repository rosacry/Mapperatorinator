from __future__ import annotations

from datetime import timedelta

import numpy as np
import numpy.typing as npt
from slider import Beatmap, Circle, Slider, Spinner
from slider.curve import Linear, Catmull, Perfect, MultiBezier

from osuT5.tokenizer import Event, EventType


class OsuParser:
    def __init__(self) -> None:
        pass

    def parse(self, beatmap: Beatmap) -> list[Event]:
        # noinspection PyUnresolvedReferences
        """Parse an .osu beatmap.

        Each hit object is parsed into a list of Event objects, in order of its
        appearance in the beatmap. In other words, in ascending order of time.

        Args:
            beatmap: Beatmap object parsed from an .osu file.

        Returns:
            events: List of Event object lists.

        Example::
            >>> beatmap = [
                "64,80,11000,1,0",
                "100,100,16000,2,0,B|200:200|250:200|250:200|300:150,2"
            ]
            >>> events = parse(beatmap)
            >>> print(events)
            [
                Event(EventType.TIME_SHIFT, 11000), Event(EventType.DISTANCE, 36), Event(EventType.CIRCLE),
                Event(EventType.TIME_SHIFT, 16000), Event(EventType.DISTANCE, 42), Event(EventType.SLIDER_HEAD),
                Event(EventType.TIME_SHIFT, 16500), Event(EventType.DISTANCE, 141), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.TIME_SHIFT, 17000), Event(EventType.DISTANCE, 50), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.TIME_SHIFT, 17500), Event(EventType.DISTANCE, 10), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.TIME_SHIFT, 18000), Event(EventType.DISTANCE, 64), Event(EventType.LAST _ANCHOR),
                Event(EventType.TIME_SHIFT, 20000), Event(EventType.DISTANCE, 11), Event(EventType.SLIDER_END)
            ]
        """
        hit_objects = beatmap.hit_objects(stacking=False)
        last_pos = np.array((256, 192))
        events = []

        for hit_object in hit_objects:
            if isinstance(hit_object, Circle):
                last_pos = self._parse_circle(hit_object, events, last_pos)
            elif isinstance(hit_object, Slider):
                last_pos = self._parse_slider(hit_object, events, last_pos)
            elif isinstance(hit_object, Spinner):
                last_pos = self._parse_spinner(hit_object, events)

        return events

    def _parse_circle(self, circle: Circle, events: list[Event], last_pos: npt.NDArray) -> npt.NDArray:
        """Parse a circle hit object.

        Args:
            circle: Circle object.
            events: List of events to add to.
            last_pos: Last position of the hit objects.

        Returns:
            pos: Position of the circle.
        """
        time = int(circle.time.total_seconds() * 1000)
        pos = np.array(circle.position)
        dist = int(np.linalg.norm(pos - last_pos))

        events.append(Event(EventType.TIME_SHIFT, time))
        events.append(Event(EventType.DISTANCE, dist))
        if circle.new_combo:
            events.append(Event(EventType.NEW_COMBO))
        events.append(Event(EventType.CIRCLE))

        return pos

    def _parse_slider(self, slider: Slider, events: list[Event], last_pos: npt.NDArray) -> npt.NDArray:
        """Parse a slider hit object.

        Args:
            slider: Slider object.
            events: List of events to add to.
            last_pos: Last position of the hit objects.

        Returns:
            pos: Last position of the slider.
        """
        # Ignore sliders which are too big
        if len(slider.curve.points) >= 100:
            return last_pos

        time = int(slider.time.total_seconds() * 1000)
        pos = np.array(slider.position)
        dist = int(np.linalg.norm(pos - last_pos))
        last_pos = pos

        events.append(Event(EventType.TIME_SHIFT, time))
        events.append(Event(EventType.DISTANCE, dist))
        if slider.new_combo:
            events.append(Event(EventType.NEW_COMBO))
        events.append(Event(EventType.SLIDER_HEAD))

        duration: timedelta = (slider.end_time - slider.time) / slider.repeat
        control_point_count = len(slider.curve.points)

        def append_control_points(event_type: EventType, last_pos: npt.NDArray = last_pos) -> npt.NDArray:
            for i in range(1, control_point_count - 1):
                last_pos = add_anchor_time_dist(i, last_pos)
                events.append(Event(event_type))

            return last_pos

        def add_anchor_time_dist(i: int, last_pos: npt.NDArray) -> npt.NDArray:
            time = int((slider.time + i / (control_point_count - 1) * duration).total_seconds() * 1000)
            pos = np.array(slider.curve.points[i])
            dist = int(np.linalg.norm(pos - last_pos))
            last_pos = pos

            events.append(Event(EventType.TIME_SHIFT, time))
            events.append(Event(EventType.DISTANCE, dist))

            return last_pos

        if isinstance(slider.curve, Linear):
            last_pos = append_control_points(EventType.RED_ANCHOR, last_pos)
        elif isinstance(slider.curve, Catmull):
            last_pos = append_control_points(EventType.CATMULL_ANCHOR, last_pos)
        elif isinstance(slider.curve, Perfect):
            last_pos = append_control_points(EventType.PERFECT_ANCHOR, last_pos)
        elif isinstance(slider.curve, MultiBezier):
            for i in range(1, control_point_count - 1):
                if slider.curve.points[i] == slider.curve.points[i + 1]:
                    last_pos = add_anchor_time_dist(i, last_pos)
                    events.append(Event(EventType.RED_ANCHOR))
                elif slider.curve.points[i] != slider.curve.points[i - 1]:
                    last_pos = add_anchor_time_dist(i, last_pos)
                    events.append(Event(EventType.BEZIER_ANCHOR))

        last_pos = add_anchor_time_dist(control_point_count - 1, last_pos)
        events.append(Event(EventType.LAST_ANCHOR))

        time = int(slider.end_time.total_seconds() * 1000)
        pos = np.array(slider.curve(1))
        dist = int(np.linalg.norm(pos - last_pos))
        last_pos = pos

        events.append(Event(EventType.TIME_SHIFT, time))
        events.append(Event(EventType.DISTANCE, dist))
        events.append(Event(EventType.SLIDER_END))

        return last_pos

    def _parse_spinner(self, spinner: Spinner, events: list[Event]) -> npt.NDArray:
        """Parse a spinner hit object.

        Args:
            spinner: Spinner object.
            events: List of events to add to.

        Returns:
            pos: Last position of the spinner.
        """
        time = int(spinner.time.total_seconds() * 1000)
        events.append(Event(EventType.TIME_SHIFT, time))
        events.append(Event(EventType.SPINNER))

        time = int(spinner.end_time.total_seconds() * 1000)
        events.append(Event(EventType.TIME_SHIFT, time))
        events.append(Event(EventType.SPINNER_END))

        return np.array((256, 192))
