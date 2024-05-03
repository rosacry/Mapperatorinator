from pathlib import Path

import numpy as np
from pydub import AudioSegment

import numpy.typing as npt

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
