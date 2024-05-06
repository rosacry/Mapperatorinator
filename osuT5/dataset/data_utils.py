from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment

import numpy.typing as npt

from osuT5.tokenizer import Event, EventType

MILISECONDS_PER_SECOND = 1000


def load_audio_file(file: Path, sample_rate: int) -> npt.NDArray:
    """Load an audio file as a numpy time-series array

    The signals are resampled, converted to mono channel, and normalized.

    Args:
        file: Path to audio file.
        sample_rate: Sample rate to resample the audio.

    Returns:
        samples: Audio time series.
    """
    audio = AudioSegment.from_file(file, format=file.suffix[1:])
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples *= 1.0 / np.max(np.abs(samples))
    return samples


def update_event_times(events: list[Event], event_times: list[float], end_time: Optional[float] = None):
    non_timed_events = [
        EventType.BEZIER_ANCHOR,
        EventType.PERFECT_ANCHOR,
        EventType.CATMULL_ANCHOR,
        EventType.RED_ANCHOR,
    ]
    timed_events = [
        EventType.CIRCLE,
        EventType.SPINNER,
        EventType.SPINNER_END,
        EventType.SLIDER_HEAD,
        EventType.LAST_ANCHOR,
        EventType.SLIDER_END,
    ]

    start_index = len(event_times)
    end_index = len(events)
    ct = 0 if len(event_times) == 0 else event_times[-1]
    for i in range(start_index, end_index):
        event = events[i]
        if event.type == EventType.TIME_SHIFT:
            ct = event.value
        event_times.append(ct)

    # Interpolate time for control point events
    # T-D-Start-D-CP-D-CP-T-D-LCP-T-D-End
    # 1-1-1-----1-1--1-1--7-7--7--9-9-9--
    # 1-1-1-----3-3--5-5--7-7--7--9-9-9--
    ct = end_time if end_time is not None else event_times[-1]
    interpolate = False
    for i in range(end_index - 1, start_index - 1, -1):
        event = events[i]

        if event.type in timed_events:
            interpolate = False

        if event.type in non_timed_events:
            interpolate = True

        if not interpolate:
            ct = event_times[i]
            continue

        if event.type not in non_timed_events:
            event_times[i] = ct
            continue

        # Find the time of the first timed event and the number of control points between
        j = i
        count = 0
        t = ct
        while j >= 0:
            event2 = events[j]
            if event2.type == EventType.TIME_SHIFT:
                t = event_times[j]
                break
            if event2.type in non_timed_events:
                count += 1
            j -= 1
        if i < 0:
            t = 0

        # Interpolate the time
        ct = (ct - t) / (count + 1) * count + t
        event_times[i] = ct
