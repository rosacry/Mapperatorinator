from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt

from config import InferenceConfig
from ..dataset.data_utils import load_audio_file, MILISECONDS_PER_SECOND


class Preprocessor(object):
    def __init__(self, args: InferenceConfig, parallel: bool = False):
        """Preprocess audio data into sequences."""
        self.frame_seq_len = args.osut5.data.src_seq_len - 1
        self.frame_size = args.osut5.data.hop_length
        self.sample_rate = args.osut5.data.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.sequence_stride = int(self.samples_per_sequence * (1 - args.lookback - args.lookahead))
        self.parallel = parallel
        if parallel:
            self.sequence_stride = self.samples_per_sequence
        self.miliseconds_per_stride = self.sequence_stride * MILISECONDS_PER_SECOND / self.sample_rate

    def load(self, path: str) -> npt.ArrayLike:
        """Load an audio file as audio frames. Convert stereo to mono, normalize.

        Args:
            path: Path to audio file.

        Returns:
            samples: Audio time-series.
        """
        return load_audio_file(path, self.sample_rate)

    def segment(
            self,
            samples: npt.ArrayLike,
            begin_pad: int = 0,
            end_pad: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment audio samples into sequences. Sequences are flattened frames.

        Args:
            samples: Audio time-series.
            begin_pad: Padding at the beginning of the audio in samples.
            end_pad: Padding at the end of the audio in samples.

        Returns:
            sequences: A list of sequences of shape (batch size, samples per sequence).
            sequence_times: A list of sequence start times in miliseconds.
        """
        samples = np.pad(samples, [begin_pad, end_pad])
        samples = np.pad(
            samples,
            [0, self.sequence_stride - (len(samples) - self.samples_per_sequence) % self.sequence_stride],
        )
        sequences = self.window(samples, self.samples_per_sequence, self.sequence_stride)
        sequences = torch.from_numpy(sequences).to(torch.float32)
        sequence_times = torch.arange(0, len(sequences) * self.miliseconds_per_stride,
                                      self.miliseconds_per_stride).to(torch.int32)
        return sequences, sequence_times

    @staticmethod
    def window(a, w, o, copy=False):
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
        if copy:
            return view.copy()
        else:
            return view
