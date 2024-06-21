from __future__ import annotations

import dataclasses
import os
import pathlib
import uuid
from string import Template

import numpy as np
from omegaconf import DictConfig

from .slider_path import SliderPath
from ..tokenizer import Event, EventType

OSU_FILE_EXTENSION = ".osu"
OSU_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.osu")
STEPS_PER_MILLISECOND = 0.1


@dataclasses.dataclass
class BeatmapConfig:
    # General
    audio_filename: str = ""

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
        )
        self.offset = args.offset
        self.beat_length = 60000 / args.bpm
        self.slider_multiplier = self.beatmap_config.slider_multiplier

    def generate(self, events: list[Event]):
        """Generate a beatmap file.

        Args:
            events: List of Event objects.

        Returns:
            None. An .osu file will be generated.
        """

        hit_object_strings = []
        time = 0
        dist = 0
        x = 256
        y = 192
        has_pos = False
        new_combo = 0
        ho_info = []
        anchor_info = []
        last_sv = 1

        timing_point_strings = [
            f"{self.offset},{self.beat_length},4,2,0,100,1,0"
        ]

        # Convert to .osu format
        for event in events:
            hit_type = event.type

            if hit_type == EventType.TIME_SHIFT:
                time = event.value
                continue
            elif hit_type == EventType.DISTANCE:
                # Find a point which is dist away from the last point but still within the playfield
                dist = event.value
                coordinates = calculate_coordinates((x, y), dist, 500, (512, 384))
                pos = coordinates[np.random.randint(len(coordinates))]
                x, y = pos
                continue
            elif hit_type == EventType.POS_X:
                x = event.value
                has_pos = True
                continue
            elif hit_type == EventType.POS_Y:
                y = event.value
                has_pos = True
                continue
            elif hit_type == EventType.NEW_COMBO:
                new_combo = 4
                continue

            if hit_type == EventType.CIRCLE:
                hit_object_strings.append(f"{int(round(x))},{int(round(y))},{int(round(time))},{1 | new_combo},0")
                ho_info = []

            elif hit_type == EventType.SPINNER:
                ho_info = [time, new_combo]

            elif hit_type == EventType.SPINNER_END and len(ho_info) == 2:
                hit_object_strings.append(
                    f"{256},{192},{int(round(ho_info[0]))},{8 | ho_info[1]},0,{int(round(time))}"
                )
                ho_info = []

            elif hit_type == EventType.SLIDER_HEAD:
                ho_info = [x, y, time, new_combo]
                anchor_info = []

            elif hit_type == EventType.BEZIER_ANCHOR:
                anchor_info.append(('B', x, y))

            elif hit_type == EventType.PERFECT_ANCHOR:
                anchor_info.append(('P', x, y))

            elif hit_type == EventType.CATMULL_ANCHOR:
                anchor_info.append(('C', x, y))

            elif hit_type == EventType.RED_ANCHOR:
                anchor_info.append(('B', x, y))
                anchor_info.append(('B', x, y))

            elif hit_type == EventType.LAST_ANCHOR:
                ho_info.append(time)
                anchor_info.append(('B', x, y))

            elif hit_type == EventType.SLIDER_END and len(ho_info) == 5 and len(anchor_info) > 0:
                curve_type = anchor_info[0][0]
                span_duration = ho_info[4] - ho_info[2]
                total_duration = time - ho_info[2]

                if total_duration == 0 or span_duration == 0:
                    continue

                slides = max(int(round(total_duration / span_duration)), 1)
                slider_path = SliderPath(self.curve_type_shorthand[curve_type], np.array([(ho_info[0], ho_info[1])] + [(cp[1], cp[2]) for cp in anchor_info], dtype=float))
                length = slider_path.get_distance()

                req_length = length * position_to_progress(
                    slider_path,
                    np.array((x, y)),
                ) if has_pos else length - dist

                if req_length < 1e-4:
                    continue

                sv, adjusted_length = self.get_human_sv_and_length(req_length, length, span_duration, last_sv)

                # If the adjusted length is too long, scale the control points to fit the length
                if adjusted_length > length + 1e-4:
                    scale = adjusted_length / length
                    anchor_info = [(cp[0], (cp[1] - ho_info[0]) * scale + ho_info[0], (cp[2] - ho_info[1]) * scale + ho_info[1]) for cp in anchor_info]

                if sv != last_sv:
                    timing_point_strings.append(
                        f"{int(round(ho_info[2]))},{(-100 / sv)},4,2,0,100,0,0"
                    )
                    last_sv = sv

                control_points = "|".join(f"{int(round(cp[1]))}:{int(round(cp[2]))}" for cp in anchor_info)
                hit_object_strings.append(
                    f"{int(round(ho_info[0]))},{int(round(ho_info[1]))},{int(round(ho_info[2]))},{2 | ho_info[3]},0,{curve_type}|{control_points},{slides},{adjusted_length}"
                )

            new_combo = 0

        # Write .osu file
        with open(OSU_TEMPLATE_PATH, "r") as tf:
            template = Template(tf.read())
            hit_objects = {"hit_objects": "\n".join(hit_object_strings)}
            timing_points = {"timing_points": "\n".join(timing_point_strings)}
            beatmap_config = dataclasses.asdict(self.beatmap_config)
            result = template.safe_substitute({**beatmap_config, **hit_objects, **timing_points})

            # Write .osu file to directory
            osu_path = os.path.join(self.output_path, f"beatmap{str(uuid.uuid4().hex)}{OSU_FILE_EXTENSION}")
            with open(osu_path, "w") as osu_file:
                osu_file.write(result)

    def get_human_sv_and_length(self, req_length, length, span_duration, last_sv):
        # Only change sv if the difference is more than 10%
        sv = req_length / 100 / span_duration * self.beat_length / self.slider_multiplier
        if abs(sv - last_sv) / sv <= 0.1:
            sv = last_sv
        else:
            # Quantize the sv to multiples of 1/20 to 'humanize' the beatmap
            sv = round(sv * 20) / 20

        # Recalculate the required length to align with the actual sv
        adjusted_length = sv * span_duration * 100 / self.beat_length * self.slider_multiplier

        return sv, adjusted_length
