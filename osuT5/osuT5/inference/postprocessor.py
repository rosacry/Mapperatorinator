from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import uuid
from datetime import timedelta
from string import Template
from typing import Optional

import numpy as np
from omegaconf import DictConfig
from slider import TimingPoint

from .slider_path import SliderPath
from .timing_points_change import TimingPointsChange, sort_timing_points
from ..tokenizer import Event, EventType

OSU_FILE_EXTENSION = ".osu"
OSU_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.osu")
STEPS_PER_MILLISECOND = 0.1


@dataclasses.dataclass
class BeatmapConfig:
    # General
    audio_filename: str = ""
    preview_time: int = -1

    # Metadata
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version: str = ""

    # Difficulty
    hp_drain_rate: float = 5
    circle_size: float = 4
    overall_difficulty: float = 8
    approach_rate: float = 9
    slider_multiplier: float = 1.8

    # Events
    background_line: str = ""


@dataclasses.dataclass
class Group:
    event_type: EventType = None
    time: int = 0
    x: float = 256
    y: float = 192
    new_combo: bool = False
    hitsounds: list[int] = dataclasses.field(default_factory=list)
    samplesets: list[int] = dataclasses.field(default_factory=list)
    additions: list[int] = dataclasses.field(default_factory=list)
    volumes: list[int] = dataclasses.field(default_factory=list)


def calculate_coordinates(last_pos, dist, num_samples, playfield_size):
    # Generate a set of angles
    angles = np.linspace(0, 2*np.pi, num_samples)

    # Calculate the x and y coordinates for each angle
    x_coords = last_pos[0] + dist * np.cos(angles)
    y_coords = last_pos[1] + dist * np.sin(angles)

    # Combine the x and y coordinates into a list of tuples
    coordinates = list(zip(x_coords, y_coords))

    # Filter out coordinates that are outside the playfield
    coordinates = [(x, y) for x, y in coordinates if 0 <= x <= playfield_size[0] and 0 <= y <= playfield_size[1]]

    if len(coordinates) == 0:
        return [playfield_size] if last_pos[0] + last_pos[1] > (playfield_size[0] + playfield_size[1]) / 2 else [(0, 0)]

    return coordinates


def position_to_progress(slider_path: SliderPath, pos: np.ndarray) -> np.ndarray:
    eps = 1e-4
    lr = 1
    t = 1
    for i in range(100):
        grad = np.linalg.norm(slider_path.position_at(t) - pos) - np.linalg.norm(
            slider_path.position_at(t - eps) - pos,
        )
        t -= lr * grad

        if grad == 0 or t < 0 or t > 1:
            break

    return np.clip(t, 0, 1)


class Postprocessor(object):
    def __init__(self, args: DictConfig):
        """Postprocessing stage that converts a list of Event objects to a beatmap file."""
        self.curve_type_shorthand = {
            "B": "Bezier",
            "P": "PerfectCurve",
            "C": "Catmull",
        }

        self.output_path = args.output_path
        self.audio_path = args.audio_path
        self.beatmap_config = BeatmapConfig(
            title=str(args.title),
            artist=str(args.artist),
            title_unicode=str(args.title),
            artist_unicode=str(args.artist),
            audio_filename=pathlib.Path(args.audio_path).name,
            slider_multiplier=float(args.slider_multiplier),
            creator=str(args.creator),
            version=str(args.version),
            background_line=f"0,0,\"{args.background}\",0,0\n" if args.background else "",
            preview_time=args.preview_time,
        )
        self.offset = args.offset
        self.beat_length = 60000 / args.bpm
        self.slider_multiplier = self.beatmap_config.slider_multiplier
        self.timing_leniency = args.timing_leniency
        self.types_first = args.osut5.data.types_first
        self.has_pos = args.osut5.data.add_positions

    def get_groups(self, events: list[Event]) -> list[Group]:
        type_events = [
            EventType.CIRCLE,
            EventType.SPINNER,
            EventType.SPINNER_END,
            EventType.SLIDER_HEAD,
            EventType.BEZIER_ANCHOR,
            EventType.PERFECT_ANCHOR,
            EventType.CATMULL_ANCHOR,
            EventType.RED_ANCHOR,
            EventType.LAST_ANCHOR,
            EventType.SLIDER_END,
            EventType.BEAT,
            EventType.MEASURE,
        ]

        groups = []
        group = Group()
        last_x, last_y = 256, 192
        for event in events:
            if event.type == EventType.TIME_SHIFT:
                group.time = event.value
            elif event.type == EventType.DISTANCE:
                # Find a point which is dist away from the last point but still within the playfield
                coordinates = calculate_coordinates((last_x, last_y), event.value, 500, (512, 384))
                pos = coordinates[np.random.randint(len(coordinates))]
                group.x, group.y = pos
                last_x, last_y = pos
            elif event.type == EventType.POS_X:
                group.x = event.value
                last_x = event.value
            elif event.type == EventType.POS_Y:
                group.y = event.value
                last_y = event.value
            elif event.type == EventType.NEW_COMBO:
                group.new_combo = True
            elif event.type == EventType.HITSOUND:
                group.hitsounds.append((event.value % 8) * 2)
                group.samplesets.append(((event.value // 8) % 3) + 1)
                group.additions.append(((event.value // 24) % 3) + 1)
            elif event.type == EventType.VOLUME:
                group.volumes.append(event.value)
            elif event.type in type_events:
                if self.types_first:
                    if group.event_type is not None:
                        groups.append(group)
                        group = Group()
                    group.event_type = event.type
                else:
                    group.event_type = event.type
                    groups.append(group)
                    group = Group()

        if group.event_type is not None:
            groups.append(group)

        return groups

    def generate(self, events: list[Event], timing: list[TimingPoint] = None):
        """Generate a beatmap file.

        Args:
            events: List of Event objects.
            timing: List of TimingPoint objects.

        Returns:
            None. An .osu file will be generated.
        """

        hit_object_strings = []
        spinner_start = None
        slider_head = None
        anchor_info = []
        last_anchor = None

        if timing is None:
            timing = [TimingPoint(
                timedelta(milliseconds=self.offset), self.beat_length, 4, 2, 0, 100, None, False
            )]

        groups = self.get_groups(events)

        # Convert to .osu format
        for group in groups:
            hit_type = group.event_type

            if hit_type == EventType.CIRCLE:
                hit_object_strings.append(f"{int(round(group.x))},{int(round(group.y))},{int(round(group.time))},{5 if group.new_combo else 1},{group.hitsounds[0]},{group.samplesets[0]}:{group.additions[0]}:0:0:")
                timing = self.set_volume(timedelta(milliseconds=int(round(group.time))), group.volumes[0], timing)

            elif hit_type == EventType.SPINNER:
                spinner_start = group

            elif hit_type == EventType.SPINNER_END and spinner_start is not None:
                hit_object_strings.append(
                    f"{256},{192},{int(round(spinner_start.time))},{12},{group.hitsounds[0]},{int(round(group.time))},{group.samplesets[0]}:{group.additions[0]}:0:0:"
                )
                timing = self.set_volume(timedelta(milliseconds=int(round(group.time))), group.volumes[0], timing)
                spinner_start = None

            elif hit_type == EventType.SLIDER_HEAD:
                slider_head = group

            elif hit_type == EventType.BEZIER_ANCHOR:
                anchor_info.append(('B', group.x, group.y))

            elif hit_type == EventType.PERFECT_ANCHOR:
                anchor_info.append(('P', group.x, group.y))

            elif hit_type == EventType.CATMULL_ANCHOR:
                anchor_info.append(('C', group.x, group.y))

            elif hit_type == EventType.RED_ANCHOR:
                anchor_info.append(('B', group.x, group.y))
                anchor_info.append(('B', group.x, group.y))

            elif hit_type == EventType.LAST_ANCHOR:
                anchor_info.append(('B', group.x, group.y))
                last_anchor = group

            elif hit_type == EventType.SLIDER_END and slider_head is not None and last_anchor is not None:
                slider_start_time = int(round(slider_head.time))
                curve_type = anchor_info[0][0]
                span_duration = last_anchor.time - slider_head.time
                total_duration = group.time - slider_head.time

                if total_duration == 0 or span_duration == 0:
                    continue

                slides = max(int(round(total_duration / span_duration)), 1)
                span_duration = total_duration / slides
                slider_path = SliderPath(self.curve_type_shorthand[curve_type], np.array([(slider_head.x, slider_head.y)] + [(cp[1], cp[2]) for cp in anchor_info], dtype=float))
                length = slider_path.get_distance()

                req_length = length * position_to_progress(
                    slider_path,
                    np.array((group.x, group.y)),
                ) if self.has_pos else length - np.linalg.norm(np.array((group.x, group.y)) - np.array((last_anchor.x, last_anchor.y)))

                if req_length < 1e-4:
                    continue

                tp = self.timing_point_at(timedelta(milliseconds=slider_start_time), timing)
                redline = tp if tp.parent is None else tp.parent
                last_sv = 1 if tp.parent is None else -100 / tp.ms_per_beat

                sv, adjusted_length = self.get_human_sv_and_length(req_length, length, span_duration, last_sv, redline, slider_head.new_combo)

                # If the adjusted length is too long, scale the control points to fit the length
                if adjusted_length > length + 1e-4:
                    scale = adjusted_length / length
                    anchor_info = [(cp[0], (cp[1] - slider_head.x) * scale + slider_head.x, (cp[2] - slider_head.y) * scale + slider_head.y) for cp in anchor_info]

                if sv != last_sv:
                    timing = self.set_sv(timedelta(milliseconds=slider_start_time), sv, timing)

                node_hitsounds = slider_head.hitsounds + last_anchor.hitsounds[1:] + group.hitsounds
                node_samplesets = slider_head.samplesets + last_anchor.samplesets[1:] + group.samplesets
                node_additions = slider_head.additions + last_anchor.additions[1:] + group.additions
                node_volumes = slider_head.volumes + last_anchor.volumes[1:] + group.volumes

                control_points = "|".join(f"{int(round(cp[1]))}:{int(round(cp[2]))}" for cp in anchor_info)
                node_hitsounds = "|".join(map(str, node_hitsounds))
                node_sampleset = "|".join(f"{s}:{a}" for s, a in zip(node_samplesets, node_additions))

                hit_object_strings.append(
                    f"{int(round(slider_head.x))},{int(round(slider_head.y))},{slider_start_time},{6 if slider_head.new_combo else 2},{last_anchor.hitsounds[0]},{curve_type}|{control_points},{slides},{adjusted_length},{node_hitsounds},{node_sampleset},{last_anchor.samplesets[0]}:{last_anchor.additions[0]}:0:0:"
                )

                # Set volume for each node sample
                for i in range(min(slides + 1, len(node_volumes))):
                    t = int(round(slider_head.time + span_duration * i))
                    node_volume = node_volumes[i]
                    timing = self.set_volume(timedelta(milliseconds=t), node_volume, timing)

                    if last_anchor.volumes[0] != node_volume and i < slides and span_duration > 6:
                        # Add a volume change after each node sample to make sure the body volume is maintained
                        timing = self.set_volume(timedelta(milliseconds=t + 6), last_anchor.volumes[0], timing)

                slider_head = None
                last_anchor = None
                anchor_info = []

        # Write .osu file
        with open(OSU_TEMPLATE_PATH, "r") as tf:
            template = Template(tf.read())
            hit_objects = {"hit_objects": "\n".join(hit_object_strings)}
            timing_points = {"timing_points": "\n".join(tp.pack() for tp in timing)}
            beatmap_config = dataclasses.asdict(self.beatmap_config)
            result = template.safe_substitute({**beatmap_config, **hit_objects, **timing_points})

            # Write .osu file to directory
            osu_path = os.path.join(self.output_path, f"beatmap{str(uuid.uuid4().hex)}{OSU_FILE_EXTENSION}")
            with open(osu_path, "w") as osu_file:
                osu_file.write(result)

    @staticmethod
    def set_volume(time: timedelta, volume: int, timing: list[TimingPoint]) -> list[TimingPoint]:
        """Set the volume of the hitsounds at a specific time."""
        tp = TimingPoint(time, -100, 4, 2, 0, volume, None, False)
        tp_change = TimingPointsChange(tp, volume=True)
        return tp_change.add_change(timing, True)

    @staticmethod
    def set_sv(time: timedelta, sv: float, timing: list[TimingPoint]) -> list[TimingPoint]:
        """Set the slider velocity at a specific time."""
        tp = TimingPoint(time, -100 / sv, 4, 2, 0, 100, None, False)
        tp_change = TimingPointsChange(tp, mpb=True)
        return tp_change.add_change(timing, True)

    def get_human_sv_and_length(self, req_length, length, span_duration, last_sv, redline, new_combo):
        # Only change sv if the difference is more than 10%
        sv = req_length / 100 / span_duration * redline.ms_per_beat / self.slider_multiplier
        leniency = 0.05 if new_combo else 0.15

        if abs(sv - last_sv) / last_sv <= leniency:
            sv = last_sv
        else:
            # Quantize the sv to multiples of 1/20 to 'humanize' the beatmap
            rounded_sv = round(sv * 20) / 20
            if rounded_sv < 0.1:  # If the sv is low, try higher precision
                rounded_sv = round(sv * 100) / 100
            sv = rounded_sv if rounded_sv > 1E-5 else sv

        # Recalculate the required length to align with the actual sv
        adjusted_length = sv * span_duration * 100 / redline.ms_per_beat * self.slider_multiplier

        return sv, adjusted_length

    def resnap_events(self, events: list[Event], timing: list[TimingPoint]) -> list[Event]:
        """Resnap events to the designated beat snap divisors."""
        timing = sort_timing_points(timing)
        resnapped_events = []
        for i, event in enumerate(events):
            if event.type != EventType.TIME_SHIFT:
                resnapped_events.append(event)
                continue

            time = event.value
            snap_divisor = 0

            if i + 1 < len(events) and events[i + 1].type == EventType.SNAPPING:
                snap_divisor = events[i + 1].value

            if snap_divisor > 0:
                time = int(self.resnap(time, timing, snap_divisor))

            resnapped_events.append(Event(EventType.TIME_SHIFT, time))

        return resnapped_events

    @dataclasses.dataclass
    class Marker:
        time: float
        is_measure: bool
        beats_from_last_marker: int = 1

    @staticmethod
    def timing_point_at(time: timedelta, timing_points: list[TimingPoint]) -> TimingPoint:
        for tp in reversed(timing_points):
            if tp.offset <= time:
                return tp

        return timing_points[0]

    @staticmethod
    def uninherited_timing_point_after(time: timedelta, timing_points: list[TimingPoint]) -> Optional[TimingPoint]:
        for tp in timing_points:
            if tp.offset > time and tp.parent is None:
                return tp

        return None

    def generate_timing(self, events: list[Event]) -> list[TimingPoint]:
        """Generate timing points from a list of Event objects."""

        markers: list[Postprocessor.Marker] = []
        step = 1 if self.types_first else -1
        for i, event in enumerate(events):
            if ((event.type == EventType.BEAT or event.type == EventType.MEASURE) and
                    i + step < len(events) and events[i + step].type == EventType.TIME_SHIFT):
                markers.append(self.Marker(int(events[i + step].value), event.type == EventType.MEASURE))

        if len(markers) == 0:
            return []

        markers.sort(key=lambda x: x.time)

        timing: list[TimingPoint] = [
            TimingPoint(timedelta(milliseconds=markers[0].time), 1000, 4, 2, 0, 100, None, False)
        ]

        counter = 0
        last_measure_time = markers[0].time

        for marker in markers:
            time = marker.time
            redline = self.timing_point_at(timedelta(milliseconds=time - 1), timing)
            redline = redline if redline.parent is None else redline.parent
            redline_offset = redline.offset.total_seconds() * 1000

            if redline_offset == time:
                continue

            counter += 1

            if not marker.is_measure:
                continue

            if redline.meter != counter:
                if last_measure_time <= redline_offset:
                    # We can edit the meter of the redline
                    redline.meter = counter
                else:
                    # We need to create a new redline
                    tp = TimingPoint(timedelta(milliseconds=last_measure_time), 100, counter, 2, 0, 100, None, False)
                    tp_change = TimingPointsChange(tp, meter=True, uninherited=True)
                    timing = tp_change.add_change(timing, True)

            counter = 0
            last_measure_time = time

        counter = 0

        for marker in markers:
            time = marker.time
            redline = self.timing_point_at(timedelta(milliseconds=time - 1), timing)
            redline = redline if redline.parent is None else redline.parent
            redline_offset = redline.offset.total_seconds() * 1000
            beats_from_last_marker = marker.beats_from_last_marker

            if beats_from_last_marker == 0 or redline_offset == time:
                continue

            markers_before = [o for o in markers if time > o.time > redline_offset] + [marker]

            mpb = 0
            beats_from_redline = 0
            for marker_b in markers_before:
                beats_from_redline += marker_b.beats_from_last_marker
                mpb += self.get_ms_per_beat(marker_b.time - redline_offset, beats_from_redline, 0)
            mpb /= len(markers_before)

            can_change_redline = self.check_ms_per_beat(mpb, markers_before, redline)

            if can_change_redline:
                mpb = self.human_round_ms_per_beat(mpb, markers_before, redline)
                redline.ms_per_beat = mpb
            elif len(markers_before) > 1:
                last_time = markers_before[-2].time
                tp = TimingPoint(
                    timedelta(milliseconds=last_time),
                    self.get_ms_per_beat(time - last_time, beats_from_last_marker, self.timing_leniency),
                    4, 2, 0, 100, None, False)
                tp_change = TimingPointsChange(tp, mpb=True, uninherited=True)
                timing = tp_change.add_change(timing, True)
                counter = 0

            counter += 1
            if marker.is_measure:
                # Add a redline in case the measure counter is out of sync
                if redline.meter != counter:
                    tp = TimingPoint(timedelta(milliseconds=time), redline.ms_per_beat, redline.meter, 2, 0, 100, None, False)
                    tp_change = TimingPointsChange(tp, mpb=True, uninherited=True)
                    timing = tp_change.add_change(timing, True)
                counter = 0

        return timing

    def resnap(self, time: float, timing: list[TimingPoint], snap_divisor: int, floor: bool = True) -> float:
        """Resnap a time to the nearest beat divisor."""
        before_tp = self.timing_point_at(timedelta(milliseconds=time), timing)
        before_tp = before_tp if before_tp.parent is None else before_tp.parent
        before_time = before_tp.offset.total_seconds() * 1000
        after_tp = self.uninherited_timing_point_after(timedelta(milliseconds=time), timing)
        after_time = after_tp.offset.total_seconds() * 1000 if after_tp is not None else None

        d = before_tp.ms_per_beat / snap_divisor
        remainder = (time - before_time) % d

        if remainder < d / 2:
            new_time = time - remainder
        else:
            new_time = time + d - remainder

        if after_time is not None and new_time > before_time + 10 and new_time >= after_time - 10:
            new_time = after_time

        return math.floor(new_time + 1e-4) if floor else new_time

    def check_ms_per_beat(self, mpb_new: float, markers: list[Postprocessor.Marker], redline: TimingPoint):
        mpb_old = redline.ms_per_beat
        redline_offset = redline.offset.total_seconds() * 1000
        beats_from_redline = 0
        can_change_redline = True
        for marker_b in markers:
            time_b = marker_b.time
            beats_from_redline += marker_b.beats_from_last_marker
            redline.ms_per_beat = mpb_new
            resnapped_time_ba = redline_offset + redline.ms_per_beat * beats_from_redline
            beats_from_redline_ba = (resnapped_time_ba - redline_offset) / redline.ms_per_beat
            redline.ms_per_beat = mpb_old

            if (abs(beats_from_redline_ba - beats_from_redline) < 0.1 and
                    self.is_snapped(time_b, resnapped_time_ba, self.timing_leniency)):
                continue
            can_change_redline = False
        return can_change_redline

    def human_round_ms_per_beat(self, mpb: float, markers: list[Postprocessor.Marker], redline: TimingPoint):
        bpm = 60000 / mpb
        mpb_integer = 60000 / round(bpm)
        if self.check_ms_per_beat(mpb_integer, markers, redline):
            return mpb_integer

        mpb_halves = 60000 / (round(bpm * 2) / 2)
        if self.check_ms_per_beat(mpb_halves, markers, redline):
            return mpb_halves

        mpb_tenths = 60000 / (round(bpm * 10) / 10)
        if self.check_ms_per_beat(mpb_tenths, markers, redline):
            return mpb_tenths

        mpb_hundredths = 60000 / (round(bpm * 100) / 100)
        if self.check_ms_per_beat(mpb_hundredths, markers, redline):
            return mpb_hundredths

        mpb_thousandths = 60000 / (round(bpm * 1000) / 1000)
        if self.check_ms_per_beat(mpb_thousandths, markers, redline):
            return mpb_thousandths

        return mpb

    def get_ms_per_beat(self, time_from_redline: float, beats_from_redline: float, leniency: float):
        mpb = time_from_redline / beats_from_redline
        bpm = 60000 / mpb

        mpb_integer = 60000 / round(bpm)
        if self.is_snapped(time_from_redline, mpb_integer * beats_from_redline, leniency):
            return mpb_integer

        mpb_halves = 60000 / (round(bpm * 2) / 2)
        if self.is_snapped(time_from_redline, mpb_halves * beats_from_redline, leniency):
            return mpb_halves

        mpb_tenths = 60000 / (round(bpm * 10) / 10)
        if self.is_snapped(time_from_redline, mpb_tenths * beats_from_redline, leniency):
            return mpb_tenths

        mpb_hundredths = 60000 / (round(bpm * 100) / 100)
        if self.is_snapped(time_from_redline, mpb_hundredths * beats_from_redline, leniency):
            return mpb_hundredths

        mpb_thousandths = 60000 / (round(bpm * 1000) / 1000)
        if self.is_snapped(time_from_redline, mpb_thousandths * beats_from_redline, leniency):
            return mpb_thousandths

        return mpb

    @staticmethod
    def is_snapped(time: float, resnapped_time: float, leniency: float):
        return abs(time - resnapped_time) <= leniency
