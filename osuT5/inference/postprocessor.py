from __future__ import annotations

import dataclasses
import os
import pathlib
import uuid
from string import Template

import numpy as np
from omegaconf import DictConfig

from osuT5.tokenizer import Event, EventType

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

    # Difficulty
    hp_drain_rate: float = 5
    circle_size: float = 4
    overall_difficulty: float = 8
    approach_rate: float = 9
    slider_multiplier: float = 1.8


class Postprocessor(object):
    def __init__(self, args: DictConfig):
        """Postprocessing stage that converts a list of Event objects to a beatmap file."""

        self.output_path = args.output_path
        self.audio_path = args.audio_path
        self.beatmap_config = BeatmapConfig(
            title=str(args.title),
            artist=str(args.artist),
            title_unicode=str(args.title),
            artist_unicode=str(args.artist),
            audio_filename=f"audio{pathlib.Path(args.audio_path).suffix}",
            slider_multiplier=float(args.slider_multiplier),
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
        new_combo = 0
        ho_info = []
        anchor_info = []

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
                dist = event.value
                continue
            elif hit_type == EventType.NEW_COMBO:
                new_combo = 4
                continue

            x = int(np.clip(dist, 0, 512))
            y = int(np.clip(dist - 512, 0, 384))

            if hit_type == EventType.CIRCLE:
                hit_object_strings.append(f"{x},{y},{time},{1 | new_combo},0")
                ho_info = []

            elif hit_type == EventType.SPINNER:
                ho_info = [time, new_combo]

            elif hit_type == EventType.SPINNER_END and len(ho_info) == 2:
                hit_object_strings.append(
                    f"{256},{192},{ho_info[0]},{8 | ho_info[1]},0,{time}"
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
                slides = int(round(span_duration / total_duration))
                control_points = "|".join(f"{cp[1]}:{cp[2]}" for cp in anchor_info)
                length = -dist
                last_pos = ho_info[:2]
                for anchor in anchor_info:
                    length += np.sqrt((anchor[1] - last_pos[0]) ** 2 + (anchor[2] - last_pos[1]) ** 2)
                    last_pos = anchor[1:]

                hit_object_strings.append(
                    f"{ho_info[0]},{ho_info[1]},{ho_info[2]},{2 | ho_info[3]},0,{curve_type}|{control_points},{slides},{length}"
                )

                sv = span_duration / length / self.beat_length * self.slider_multiplier * -10000
                timing_point_strings.append(
                    f"{ho_info[2]},{sv},4,2,0,100,0,0"
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
