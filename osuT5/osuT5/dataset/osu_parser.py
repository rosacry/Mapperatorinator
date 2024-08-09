from __future__ import annotations

from datetime import timedelta

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from slider import Beatmap, Circle, Slider, Spinner
from slider.curve import Linear, Catmull, Perfect, MultiBezier

from ..tokenizer import Event, EventType, Tokenizer
from .data_utils import merge_events


class OsuParser:
    def __init__(self, args: DictConfig, tokenizer: Tokenizer) -> None:
        self.add_timing = args.data.add_timing
        self.add_hitsounds = args.data.add_hitsounds
        self.add_positions = args.data.add_positions
        if self.add_positions:
            x_range = tokenizer.event_range[EventType.POS_X]
            y_range = tokenizer.event_range[EventType.POS_Y]
            self.x_min = x_range.min_value
            self.x_max = x_range.max_value
            self.y_min = y_range.min_value
            self.y_max = y_range.max_value
        else:
            dist_range = tokenizer.event_range[EventType.DISTANCE]
            self.dist_min = dist_range.min_value
            self.dist_max = dist_range.max_value

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
                last_pos = self._parse_circle(hit_object, events, last_pos, beatmap)
            elif isinstance(hit_object, Slider):
                last_pos = self._parse_slider(hit_object, events, last_pos, beatmap)
            elif isinstance(hit_object, Spinner):
                last_pos = self._parse_spinner(hit_object, events, beatmap)

        if self.add_timing:
            events = merge_events(self.parse_timing(beatmap), events)

        return events

    def parse_timing(self, beatmap: Beatmap) -> list[Event]:
        """Extract all timing information from a beatmap."""
        events = []
        last_ho = beatmap.hit_objects(stacking=False)[-1]
        last_time = last_ho.end_time if hasattr(last_ho, "end_time") else last_ho.time

        # Get all timing points with BPM changes
        timing_points = [tp for tp in beatmap.timing_points if tp.bpm]

        for i, tp in enumerate(timing_points):
            # Generate beat and measure events until the next timing point
            next_tp = timing_points[i + 1] if i + 1 < len(timing_points) else None
            next_time = next_tp.offset - timedelta(milliseconds=10) if next_tp else last_time
            time = tp.offset
            measure_counter = 0
            while time <= next_time:
                self._add_time_event(time, beatmap, events, add_snap=False)

                if measure_counter % tp.meter == 0:
                    events.append(Event(EventType.MEASURE))
                else:
                    events.append(Event(EventType.BEAT))

                measure_counter += 1
                time += timedelta(milliseconds=tp.ms_per_beat)

        return events

    def _clip_dist(self, dist: int) -> int:
        """Clip distance to valid range."""
        return int(np.clip(dist, self.dist_min, self.dist_max))

    @staticmethod
    def uninherited_point_at(time: timedelta, beatmap: Beatmap):
        tp = beatmap.timing_point_at(time)
        return tp if tp.parent is None else tp.parent

    @staticmethod
    def hitsound_point_at(time: timedelta, beatmap: Beatmap):
        hs_query = time + timedelta(milliseconds=5)
        return beatmap.timing_point_at(hs_query)

    def _add_time_event(self, time: timedelta, beatmap: Beatmap, events: list[Event], add_snap: bool = True) -> None:
        """Add a snapping event to the event list.

        Args:
            time: Time of the snapping event.
            beatmap: Beatmap object.
            events: List of events to add to.
            add_snap: Whether to add a snapping event.
        """
        time_ms = int(time.total_seconds() * 1000)
        events.append(Event(EventType.TIME_SHIFT, time_ms))

        if not add_snap or not self.add_timing:
            return

        tp = self.uninherited_point_at(time, beatmap)
        beats = (time - tp.offset).total_seconds() * 1000 / tp.ms_per_beat
        snapping = 0
        for i in range(1, 17):
            # If the difference between the time and the snapped time is less than 2 ms, that is the correct snapping
            if abs(beats - round(beats * i) / i) * tp.ms_per_beat < 2:
                snapping = i
                break

        events.append(Event(EventType.SNAPPING, snapping))

    def _add_hitsound_event(self, time: timedelta, hitsound: int, addition: str, beatmap: Beatmap, events: list[Event]) -> None:
        if not self.add_hitsounds:
            return

        tp = self.hitsound_point_at(time, beatmap)
        tp_sample_set = tp.sample_type if tp.sample_type != 0 else 2  # Inherit to soft sample set
        addition_split = addition.split(":")
        sample_set = int(addition_split[0]) if addition_split[0] != "0" else tp_sample_set
        addition_set = int(addition_split[1]) if addition_split[1] != "0" else sample_set

        sample_set = sample_set if 0 < sample_set < 4 else 1  # Overflow default to normal sample set
        addition_set = addition_set if 0 < addition_set < 4 else 1  # Overflow default to normal sample set
        hitsound = hitsound & 14  # Only take the bits for normal, whistle, and finish

        hitsound_idx = hitsound // 2 + 8 * (sample_set - 1) + 24 * (addition_set - 1)

        events.append(Event(EventType.HITSOUND, hitsound_idx))
        events.append(Event(EventType.VOLUME, tp.volume))

    def _add_position_event(self, pos: npt.NDArray, last_pos: npt.NDArray, events: list[Event]) -> npt.NDArray:
        if self.add_positions:
            events.append(Event(EventType.POS_X, int(np.clip(pos[0], self.x_min, self.x_max))))
            events.append(Event(EventType.POS_Y, int(np.clip(pos[1], self.y_min, self.y_max))))
        else:
            dist = self._clip_dist(np.linalg.norm(pos - last_pos))
            events.append(Event(EventType.DISTANCE, dist))
        return pos


    def _parse_circle(self, circle: Circle, events: list[Event], last_pos: npt.NDArray, beatmap: Beatmap) -> npt.NDArray:
        """Parse a circle hit object.

        Args:
            circle: Circle object.
            events: List of events to add to.
            last_pos: Last position of the hit objects.

        Returns:
            pos: Position of the circle.
        """
        pos = np.array(circle.position)

        self._add_time_event(circle.time, beatmap, events)
        last_pos = self._add_position_event(pos, last_pos, events)
        if circle.new_combo:
            events.append(Event(EventType.NEW_COMBO))
        self._add_hitsound_event(circle.time, circle.hitsound, circle.addition, beatmap, events)
        events.append(Event(EventType.CIRCLE))

        return last_pos

    def _parse_slider(self, slider: Slider, events: list[Event], last_pos: npt.NDArray, beatmap: Beatmap) -> npt.NDArray:
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

        pos = np.array(slider.position)

        self._add_time_event(slider.time, beatmap, events)
        last_pos = self._add_position_event(pos, last_pos, events)
        if slider.new_combo:
            events.append(Event(EventType.NEW_COMBO))
        self._add_hitsound_event(slider.time, slider.edge_sounds[0] if len(slider.edge_sounds) > 0 else 0,
                                 slider.edge_additions[0] if len(slider.edge_additions) > 0 else '0:0', beatmap, events)
        events.append(Event(EventType.SLIDER_HEAD))

        duration: timedelta = (slider.end_time - slider.time) / slider.repeat
        control_point_count = len(slider.curve.points)

        def append_control_points(event_type: EventType, last_pos: npt.NDArray = last_pos) -> npt.NDArray:
            for i in range(1, control_point_count - 1):
                last_pos = add_anchor_time_dist(i, last_pos)
                events.append(Event(event_type))

            return last_pos

        def add_anchor_time_dist(i: int, last_pos: npt.NDArray, add_snap: bool = False) -> npt.NDArray:
            time = slider.time + i / (control_point_count - 1) * duration
            pos = np.array(slider.curve.points[i])

            self._add_time_event(time, beatmap, events, add_snap)
            last_pos = self._add_position_event(pos, last_pos, events)

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

        last_pos = add_anchor_time_dist(control_point_count - 1, last_pos, True)

        # Add body hitsounds and remaining edge hitsounds
        self._add_hitsound_event(slider.time + timedelta(milliseconds=1), slider.hitsound, slider.addition, beatmap, events)
        for i in range(1, slider.repeat):
            self._add_hitsound_event(slider.time + i * duration, slider.edge_sounds[i] if len(slider.edge_sounds) > i else 0,
                                     slider.edge_additions[i] if len(slider.edge_additions) > i else '0:0', beatmap, events)

        events.append(Event(EventType.LAST_ANCHOR))

        pos = np.array(slider.curve(1))

        self._add_time_event(slider.end_time, beatmap, events)
        last_pos = self._add_position_event(pos, last_pos, events)
        self._add_hitsound_event(slider.end_time, slider.edge_sounds[-1] if len(slider.edge_sounds) > 0 else 0,
                                 slider.edge_additions[-1] if len(slider.edge_additions) > 0 else '0:0', beatmap, events)
        events.append(Event(EventType.SLIDER_END))

        return last_pos

    def _parse_spinner(self, spinner: Spinner, events: list[Event], beatmap: Beatmap) -> npt.NDArray:
        """Parse a spinner hit object.

        Args:
            spinner: Spinner object.
            events: List of events to add to.

        Returns:
            pos: Last position of the spinner.
        """
        self._add_time_event(spinner.time, beatmap, events)
        events.append(Event(EventType.SPINNER))

        self._add_time_event(spinner.end_time, beatmap, events)
        self._add_hitsound_event(spinner.end_time, spinner.hitsound, spinner.addition, beatmap, events)
        events.append(Event(EventType.SPINNER_END))

        return np.array((256, 192))
