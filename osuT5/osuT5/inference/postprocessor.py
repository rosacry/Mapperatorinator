from __future__ import annotations

import dataclasses
import os
import uuid
from datetime import timedelta
from os import PathLike
from string import Template
from typing import Optional

import numpy as np
from omegaconf import DictConfig
from slider import TimingPoint, Beatmap

from .slider_path import SliderPath
from .timing_points_change import TimingPointsChange, sort_timing_points
from ..dataset.data_utils import get_groups, Group
from ..tokenizer import Event, EventType

OSU_FILE_EXTENSION = ".osu"
OSU_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.osu")
STEPS_PER_MILLISECOND = 0.1


@dataclasses.dataclass
class BeatmapConfig:
    # General
    audio_filename: str = ""
    preview_time: int = -1
    mode: int = 0

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

    # Timing
    bpm: float = 120
    offset: int = 0

    # Events
    background_line: str = ""


def background_line(background: str) -> str:
    return f"0,0,\"{background}\",0,0\n" if background else ""


def beatmap_config_from_beatmap(beatmap: Beatmap) -> BeatmapConfig:
    return BeatmapConfig(
        title=beatmap.title,
        artist=beatmap.artist,
        title_unicode=beatmap.title,
        artist_unicode=beatmap.artist,
        audio_filename=beatmap.audio_filename,
        circle_size=beatmap.circle_size,
        slider_multiplier=beatmap.slider_multiplier,
        creator=beatmap.creator,
        version=beatmap.version,
        background_line=background_line(beatmap.background),
        preview_time=int(beatmap.preview_time.total_seconds() * 1000),
        bpm=beatmap.bpm_max(),
        offset=int(min(tp.offset.total_seconds() * 1000 for tp in beatmap.timing_points)),
    )


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

        self.offset = args.offset
        self.beat_length = 60000 / args.bpm
        self.timing_leniency = args.timing_leniency
        self.types_first = args.osut5.data.types_first
        self.has_pos = args.osut5.data.add_positions

    def generate(
            self,
            events: list[Event],
            beatmap_config: BeatmapConfig,
            timing: list[TimingPoint] = None,
    ):
        """Generate a beatmap file.

        Args:
            events: List of Event objects.
            output_path: Path to the output directory.
            beatmap_config: BeatmapConfig object.
            timing: List of TimingPoint objects.

        Returns:
            None. An .osu file will be generated.
        """

        hit_object_strings = []
        spinner_start = None
        slider_head = None
        anchor_info = []
        last_anchor = None
        hold_note_start = None
        drumroll_start = None
        denden_start = None

        if timing is None:
            timing = [TimingPoint(
                timedelta(milliseconds=self.offset), self.beat_length, 4, 2, 0, 100, None, False
            )]

        groups = get_groups(events, types_first=self.types_first)
        last_x, last_y = 256, 192

        self.snap_near_perfect_overlaps(groups)

        # Convert to .osu format
        for group in groups:
            hit_type = group.event_type

            if group.distance is not None and group.x is None and group.y is None:
                # Find a point which is dist away from the last point but still within the playfield
                coordinates = calculate_coordinates((last_x, last_y), group.distance, 500, (512, 384))
                group.x, group.y = coordinates[np.random.randint(len(coordinates))]

            if group.x is None or group.y is None:
                # Maybe the model forgot to add any distance or position tokens, so let's just assume the last position
                group.x, group.y = last_x, last_y

            if hit_type in [EventType.CIRCLE, EventType.SLIDER_HEAD, EventType.BEZIER_ANCHOR, EventType.PERFECT_ANCHOR, EventType.CATMULL_ANCHOR, EventType.RED_ANCHOR, EventType.LAST_ANCHOR, EventType.SLIDER_END]:
                last_x, last_y = group.x, group.y

            if beatmap_config.mode == 1:
                group.x, group.y = 256, 192

            if beatmap_config.mode == 3:
                group.y = 192

            if hit_type == EventType.CIRCLE:
                hitsound = group.hitsounds[0] if len(group.hitsounds) > 0 else 0
                sampleset = group.samplesets[0] if len(group.samplesets) > 0 else 0
                addition = group.additions[0] if len(group.additions) > 0 else 0
                volume = group.volumes[0] if len(group.volumes) > 0 and beatmap_config.mode == 3 else 0
                hit_object_strings.append(f"{int(round(group.x))},{int(round(group.y))},{int(round(group.time))},{5 if group.new_combo else 1},{hitsound},{sampleset}:{addition}:{volume}:0:")
                if len(group.volumes) > 0 and beatmap_config.mode != 3:
                    timing = self.set_volume(timedelta(milliseconds=int(round(group.time))), group.volumes[0], timing)
                if beatmap_config.mode == 1 and group.scroll_speed is not None:
                    timing = self.set_sv(timedelta(milliseconds=int(round(group.time))), group.scroll_speed, timing)

            elif hit_type == EventType.HOLD_NOTE:
                hold_note_start = group

            elif hit_type == EventType.HOLD_NOTE_END and hold_note_start is not None:
                hitsound = hold_note_start.hitsounds[0] if len(hold_note_start.hitsounds) > 0 else 0
                sampleset = hold_note_start.samplesets[0] if len(hold_note_start.samplesets) > 0 else 0
                addition = hold_note_start.additions[0] if len(hold_note_start.additions) > 0 else 0
                volume = hold_note_start.volumes[0] if len(hold_note_start.volumes) > 0 and beatmap_config.mode == 3 else 0
                hit_object_strings.append(
                    f"{int(round(hold_note_start.x))},{192},{int(round(hold_note_start.time))},{128},{hitsound},{int(round(group.time))}:{sampleset}:{addition}:{volume}:0:"
                )
                if len(hold_note_start.volumes) > 0 and beatmap_config.mode != 3:
                    timing = self.set_volume(timedelta(milliseconds=int(round(hold_note_start.time))), hold_note_start.volumes[0], timing)
                hold_note_start = None

            elif hit_type == EventType.DRUMROLL:
                drumroll_start = group

            elif hit_type == EventType.DRUMROLL_END and drumroll_start is not None:
                drumroll_start_time = int(round(drumroll_start.time))
                duration = int(round(group.time)) - drumroll_start_time

                if duration < 1:
                    drumroll_start = None
                    continue

                hitsound = drumroll_start.hitsounds[0] if len(drumroll_start.hitsounds) > 0 else 0
                sampleset = drumroll_start.samplesets[0] if len(drumroll_start.samplesets) > 0 else 0
                addition = drumroll_start.additions[0] if len(drumroll_start.additions) > 0 else 0
                if len(drumroll_start.volumes) > 0:
                    timing = self.set_volume(timedelta(milliseconds=int(round(drumroll_start.time))), drumroll_start.volumes[0], timing)
                if beatmap_config.mode == 1 and drumroll_start.scroll_speed is not None:
                    timing = self.set_sv(timedelta(milliseconds=int(round(drumroll_start.time))), drumroll_start.scroll_speed, timing)

                tp = self.timing_point_at(timedelta(milliseconds=drumroll_start_time), timing)
                redline = tp if tp.parent is None else tp.parent
                sv = 1 if tp.parent is None else -100 / tp.ms_per_beat
                length = sv * duration * 100 / redline.ms_per_beat * beatmap_config.slider_multiplier

                start_pos, *anchor_info = self.get_control_points_for_length(length)
                control_points = "|".join(f"{cp[0]}:{cp[1]}" for cp in anchor_info)

                hit_object_strings.append(
                    f"{start_pos[0]},{start_pos[1]},{drumroll_start_time},{2},{hitsound},L|{control_points},{1},{length},0:0,0:0|0:0,{sampleset}:{addition}:0:0:"
                )

                drumroll_start = None

            elif hit_type == EventType.DENDEN:
                denden_start = group

            elif hit_type == EventType.DENDEN_END and denden_start is not None:
                hitsound = denden_start.hitsounds[0] if len(denden_start.hitsounds) > 0 else 0
                sampleset = denden_start.samplesets[0] if len(denden_start.samplesets) > 0 else 0
                addition = denden_start.additions[0] if len(denden_start.additions) > 0 else 0
                hit_object_strings.append(
                    f"{256},{192},{int(round(denden_start.time))},{12},{hitsound},{int(round(group.time))},{sampleset}:{addition}:0:0:"
                )
                if len(denden_start.volumes) > 0:
                    timing = self.set_volume(timedelta(milliseconds=int(round(denden_start.time))), denden_start.volumes[0], timing)
                if beatmap_config.mode == 1 and denden_start.scroll_speed is not None:
                    timing = self.set_sv(timedelta(milliseconds=int(round(denden_start.time))), denden_start.scroll_speed, timing)
                denden_start = None

            elif hit_type == EventType.SPINNER:
                spinner_start = group

            elif hit_type == EventType.SPINNER_END and spinner_start is not None:
                hitsound = group.hitsounds[0] if len(group.hitsounds) > 0 else 0
                sampleset = group.samplesets[0] if len(group.samplesets) > 0 else 0
                addition = group.additions[0] if len(group.additions) > 0 else 0
                hit_object_strings.append(
                    f"{256},{192},{int(round(spinner_start.time))},{12},{hitsound},{int(round(group.time))},{sampleset}:{addition}:0:0:"
                )
                if len(group.volumes) > 0:
                    timing = self.set_volume(timedelta(milliseconds=int(round(group.time))), group.volumes[0], timing)
                spinner_start = None
                last_x, last_y = 256, 192

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

                sv, length = self.get_human_sv_and_length(req_length, length, span_duration, last_sv, redline, slider_head.new_combo, beatmap_config.slider_multiplier)

                # If the adjusted length is too long, scale the control points to fit the length
                if length > length + 1e-4:
                    scale = length / length
                    anchor_info = [(cp[0], (cp[1] - slider_head.x) * scale + slider_head.x, (cp[2] - slider_head.y) * scale + slider_head.y) for cp in anchor_info]

                if sv != last_sv:
                    timing = self.set_sv(timedelta(milliseconds=slider_start_time), sv, timing)

                node_hitsounds = slider_head.hitsounds + last_anchor.hitsounds[1:] + group.hitsounds
                node_samplesets = slider_head.samplesets + last_anchor.samplesets[1:] + group.samplesets
                node_additions = slider_head.additions + last_anchor.additions[1:] + group.additions
                node_volumes = slider_head.volumes + last_anchor.volumes[1:] + group.volumes

                body_hitsound = last_anchor.hitsounds[0] if len(last_anchor.hitsounds) > 0 else 0
                body_sampleset = last_anchor.samplesets[0] if len(last_anchor.samplesets) > 0 else 0
                body_addition = last_anchor.additions[0] if len(last_anchor.additions) > 0 else 0

                control_points = "|".join(f"{int(round(cp[1]))}:{int(round(cp[2]))}" for cp in anchor_info)
                node_hitsounds = "|".join(map(str, node_hitsounds))
                node_sampleset = "|".join(f"{s}:{a}" for s, a in zip(node_samplesets, node_additions))

                hit_object_strings.append(
                    f"{int(round(slider_head.x))},{int(round(slider_head.y))},{slider_start_time},{6 if slider_head.new_combo else 2},{body_hitsound},{curve_type}|{control_points},{slides},{length},{node_hitsounds},{node_sampleset},{body_sampleset}:{body_addition}:0:0:"
                )

                # Set volume for each node sample
                for i in range(min(slides + 1, len(node_volumes))):
                    t = int(round(slider_head.time + span_duration * i))
                    node_volume = node_volumes[i]
                    timing = self.set_volume(timedelta(milliseconds=t), node_volume, timing)

                    if len(last_anchor.volumes) > 0 and last_anchor.volumes[0] != node_volume and i < slides and span_duration > 6:
                        # Add a volume change after each node sample to make sure the body volume is maintained
                        timing = self.set_volume(timedelta(milliseconds=t + 6), last_anchor.volumes[0], timing)

                slider_head = None
                last_anchor = None
                anchor_info = []

            elif hit_type == EventType.KIAI:
                timing = self.set_kiai(timedelta(milliseconds=group.time), bool(group.value), timing)

            elif hit_type == EventType.SCROLL_SPEED_CHANGE and group.scroll_speed is not None:
                timing = self.set_sv(timedelta(milliseconds=group.time), group.scroll_speed, timing)

        # Write .osu file
        with open(OSU_TEMPLATE_PATH, "r") as tf:
            template = Template(tf.read())
            hit_objects = {"hit_objects": "\n".join(hit_object_strings)}
            timing_points = {"timing_points": "\n".join(tp.pack() for tp in timing)}
            # noinspection PyTypeChecker
            beatmap_config = dataclasses.asdict(beatmap_config)
            result = template.safe_substitute({**beatmap_config, **hit_objects, **timing_points})
            return result

    def write_result(self, result: str, output_path: PathLike):
        # Write .osu file to directory
        osu_path = os.path.join(output_path, f"beatmap{str(uuid.uuid4().hex)}{OSU_FILE_EXTENSION}")
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

    @staticmethod
    def set_kiai(time: timedelta, kiai: bool, timing: list[TimingPoint]) -> list[TimingPoint]:
        """Set the kiai mode at a specific time."""
        tp = TimingPoint(time, -100, 4, 2, 0, 100, None, kiai)
        tp_change = TimingPointsChange(tp, kiai=True)
        return tp_change.add_change(timing, True)

    def get_control_points_for_length(self, length: float) -> list[tuple[int, int]]:
        # Constructs a slider that zigzags back and forth to cover the required length
        control_points = [(0, 192)]
        y = 192
        for i in range(int(np.ceil(length / 512))):
            x = 512 if i % 2 == 0 else 0
            control_points.append((x, y))
        return control_points

    def get_human_sv_and_length(self, req_length, length, span_duration, last_sv, redline, new_combo, slider_multiplier):
        # Only change sv if the difference is more than 10%
        sv = req_length / 100 / span_duration * redline.ms_per_beat / slider_multiplier
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
        adjusted_length = sv * span_duration * 100 / redline.ms_per_beat * slider_multiplier

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

    def resnap(self, time: float, timing: list[TimingPoint], snap_divisor: int) -> float:
        """Resnap a time to the nearest beat divisor."""
        ignore_ticks = {
            1: [],
            4: [2],
            6: [2, 3],
            8: [4],
            9: [3],
            10: [2, 5],
            12: [4, 6],
            14: [2, 7],
            15: [3, 5],
            16: [8],
        }

        before_tp = self.timing_point_at(timedelta(milliseconds=time), timing)
        before_tp = before_tp if before_tp.parent is None else before_tp.parent
        before_time = before_tp.offset.total_seconds() * 1000
        after_tp = self.uninherited_timing_point_after(timedelta(milliseconds=time), timing)
        after_time = after_tp.offset.total_seconds() * 1000 if after_tp is not None else None

        def local_ticks(divisor: int) -> set[int]:
            ms_per_tick = before_tp.ms_per_beat / divisor
            remainder = (time - before_time) % ms_per_tick
            return {
                int(time - remainder - ms_per_tick),
                int(time - remainder),
                int(time - remainder + ms_per_tick),
                int(time - remainder + 2 * ms_per_tick)
            }

        ticks = local_ticks(snap_divisor)

        # Remove ticks that are from bigger snap divisors because we specifically want to snap to the snap_divisor
        ignore_divisors = ignore_ticks.get(snap_divisor, [1])
        for ignore_divisor in ignore_divisors:
            ticks -= local_ticks(ignore_divisor)

        # Find the closest tick to the original time
        new_time = min(ticks, key=lambda x: abs(x - time))

        # If the new time is too close to the next timing point, snap to the next timing point
        if after_time is not None and new_time > before_time + 10 and new_time >= after_time - 10:
            new_time = after_time

        return new_time

    @dataclasses.dataclass
    class Marker:
        time: float
        is_measure: bool
        is_redline: bool
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
                markers.append(self.Marker(
                    int(events[i + step].value),
                    event.type == EventType.MEASURE,
                    event.type == EventType.TIMING_POINT,
                    0 if event.type == EventType.TIMING_POINT else 1
                ))

        if len(markers) == 0:
            return []

        markers.sort(key=lambda x: x.time)

        timing: list[TimingPoint] = []

        # Add redlines for each redline marker
        for marker in markers:
            if not marker.is_redline:
                continue

            time = marker.time
            tp = TimingPoint(timedelta(milliseconds=time), 100, 4, 2, 0, 100, None, False)
            tp_change = TimingPointsChange(tp, uninherited=True)
            timing = tp_change.add_change(timing, True)

        if len(timing) == 0:
            timing = [
                TimingPoint(timedelta(milliseconds=markers[0].time), 1000, 4, 2, 0, 100, None, False)
            ]

        counter = 0
        last_measure_time = markers[0].time

        # Add redlines to make sure the measure counter is correct
        for marker in markers:
            time = marker.time

            if marker.is_redline:
                counter = 0
                last_measure_time = time
                continue

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

        # Add redlines to make sure each beat is snapped correctly
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

    def snap_near_perfect_overlaps(self, groups: list[Group]):
        snappable_types = {
            EventType.CIRCLE,
            EventType.SLIDER_HEAD,
            EventType.RED_ANCHOR,
            EventType.LAST_ANCHOR,
            EventType.SLIDER_END,
        }
        space_leniency = 3.8
        time_leniency = 1000
        prev_groups: list[Group] = []

        for i, group in enumerate(groups):
            if group.event_type not in snappable_types:
                continue

            if group.x is None or group.y is None:
                continue

            # Filter previous groups to only include groups that are close in time
            prev_groups = [prev_group for prev_group in prev_groups if abs(group.time - prev_group.time) <= time_leniency]

            for prev_group in prev_groups:
                if np.linalg.norm(np.array([group.x, group.y]) - np.array([prev_group.x, prev_group.y])) < space_leniency:
                    group.x = prev_group.x
                    group.y = prev_group.y
                    break

            prev_groups.append(group)
