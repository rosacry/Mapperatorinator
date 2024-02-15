from __future__ import annotations

import os
import random
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from pydub import AudioSegment
from slider import Beatmap
from torch.utils.data import IterableDataset

from .osu_parser import OsuParser
from osuT5.tokenizer import Event, EventType, Tokenizer

OSZ_FILE_EXTENSION = ".osz"
AUDIO_FILE_NAME = "audio.mp3"
MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100


class OrsDataset(IterableDataset):
    __slots__ = (
        "path",
        "start",
        "end",
        "sample_rate",
        "frame_size",
        "src_seq_len",
        "tgt_seq_len",
        "parser",
        "tokenizer",
        "cycle_length",
        "shuffle",
        "per_track",
        "beatmap_files",
    )

    def __init__(
            self,
            path: str,
            start: int,
            end: int,
            sample_rate: int,
            frame_size: int,
            src_seq_len: int,
            tgt_seq_len: int,
            parser: OsuParser,
            tokenizer: Tokenizer,
            cycle_length: int = 1,
            shuffle: bool = False,
            per_track: bool = False,
            beatmap_files: Optional[list[Path]] = None,
    ):
        """Manage and process ORS dataset.

        Attributes:
            path: Location of ORS dataset to load.
            sample_rate: Sampling rate of audio file (samples/second).
            frame_size: Samples per audio frame (samples/frame).
            src_seq_len: Maximum length of source sequence.
            tgt_seq_len: Maximum length of target sequence.
            parser: Instance of OsuParser class.
            tokenizer: Instance of Tokenizer class.
        """
        super().__init__()
        self.path = path
        self.start = start
        self.end = end
        self.parser = parser
        self.tokenizer = tokenizer
        self.cycle_length = cycle_length
        self.shuffle = shuffle
        self.per_track = per_track and beatmap_files is None
        self.beatmap_files = beatmap_files
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

    def _get_beatmap_files(self) -> list[Path]:
        if self.beatmap_files is not None:
            return self.beatmap_files

        # Get a list of all beatmap files in the dataset path in the track index range between start and end
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
            for beatmap_file in os.listdir(
                    os.path.join(self.path, track_name, "beatmaps"),
            ):
                beatmap_files.append(
                    Path(
                        os.path.join(
                            self.path,
                            track_name,
                            "beatmaps",
                            beatmap_file,
                        )
                    ),
                )

        return beatmap_files

    def _get_track_paths(self) -> list[Path]:
        track_paths = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
            track_paths.append(Path(os.path.join(self.path, track_name)))
        return track_paths

    def __iter__(self):
        beatmap_files = self._get_track_paths() if self.per_track else self._get_beatmap_files()

        if self.shuffle:
            random.shuffle(beatmap_files)

        if self.cycle_length > 1:
            return InterleavingBeatmapDatasetIterable(
                beatmap_files,
                self._iterable_factory,
                self.cycle_length,
            )

        return self._iterable_factory(beatmap_files).__iter__()

    def _iterable_factory(self, beatmap_files: list[Path]):
        return BeatmapDatasetIterable(
            beatmap_files,
            self.sample_rate,
            self.frame_size,
            self.src_seq_len,
            self.tgt_seq_len,
            self.parser,
            self.tokenizer,
            self.per_track,
        )


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
            self,
            beatmap_files: list[Path],
            iterable_factory: Callable,
            cycle_length: int,
    ):
        per_worker = int(np.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                beatmap_files[
                    i * per_worker: min(len(beatmap_files), (i + 1) * per_worker)
                ]
            ).__iter__()
            for i in range(cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                self.workers.remove(self.workers[self.index])
        raise StopIteration


class BeatmapDatasetIterable:
    __slots__ = (
        "beatmap_files",
        "parser",
        "tokenizer",
        "sample_rate",
        "frame_size",
        "src_seq_len",
        "tgt_seq_len",
        "frame_seq_len",
        "token_seq_len",
        "per_track",
    )

    def __init__(
            self,
            beatmap_files: list[Path],
            sample_rate: int,
            frame_size: int,
            src_seq_len: int,
            tgt_seq_len: int,
            parser: OsuParser,
            tokenizer: Tokenizer,
            per_track: bool,
    ):
        self.beatmap_files = beatmap_files
        self.parser = parser
        self.tokenizer = tokenizer
        self.per_track = per_track
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        # let N = |src_seq_len|
        # N-1 frames creates N mel-spectrogram frames
        self.frame_seq_len = src_seq_len - 1
        # let N = |tgt_seq_len|
        # [SOS] token + event_tokens + [EOS] token creates N+1 tokens
        # [SOS] token + event_tokens[:-1] creates N target sequence
        # event_tokens[1:] + [EOS] token creates N label sequence
        self.token_seq_len = tgt_seq_len + 1

    def _load_audio_file(self, file: Path) -> npt.NDArray:
        """Load an audio file as a numpy time-series array

        The signals are resampled, converted to mono channel, and normalized.

        Args:
            file: Path to audio file.
        Returns:
            samples: Audio time series.
        """
        audio = AudioSegment.from_file(file, format=file.suffix[1:])
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples *= 1.0 / np.max(np.abs(samples))
        return samples

    def _get_frames(self, samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Segment audio samples into frames.

        Each frame has `frame_size` audio samples.
        It will also calculate and return the time of each audio frame, in miliseconds.

        Args:
            samples: Audio time-series.

        Returns:
            frames: Audio frames.
            frame_times: Audio frame times.
        """
        samples = np.pad(samples, [0, self.frame_size - len(samples) % self.frame_size])
        frames = np.reshape(samples, (-1, self.frame_size))
        frames_per_milisecond = (
                self.sample_rate / self.frame_size / MILISECONDS_PER_SECOND
        )
        frame_times = np.arange(len(frames)) / frames_per_milisecond
        return frames, frame_times

    def _create_sequences(
            self,
            events: list[Event],
            frames: npt.NDArray,
            frame_times: npt.NDArray,
            beatmap_idx: int,
            beatmap_id: int,
    ) -> list[dict[str, int | npt.NDArray | list[Event]]]:
        """Create frame and token sequences for training/testing.

        Args:
            events: Events and time shifts.
            frames: Audio frames.

        Returns:
            A list of source and target sequences.
        """
        # Corresponding start event index for every audio frame.
        event_start_indices = []
        event_index = 0
        event_time = -np.inf

        for current_time in frame_times:
            while event_time < current_time and event_index < len(events):
                if events[event_index].type == EventType.TIME_SHIFT:
                    event_time = events[event_index].value
                event_index += 1
            event_start_indices.append(event_index - 1)

        # Corresponding end event index for every audio frame.
        event_end_indices = event_start_indices[1:] + [len(events)]

        sequences = []
        n_frames = len(frames)
        offset = random.randint(0, self.frame_seq_len)
        # Divide audio frames into splits
        for split_start_idx in range(offset, n_frames, self.frame_seq_len):
            split_end_idx = min(split_start_idx + self.frame_seq_len, n_frames)
            target_start_idx = event_start_indices[split_start_idx]
            target_end_idx = event_end_indices[split_end_idx - 1]

            # Create the sequence
            sequence = {
                "time": frame_times[split_start_idx],
                "frames": frames[split_start_idx:split_end_idx],
                "events": events[target_start_idx:target_end_idx],
                "beatmap_idx": beatmap_idx,
                "beatmap_id": beatmap_id,
            }
            sequences.append(sequence)

        return sequences

    @staticmethod
    def _trim_time_shifts(sequence: dict) -> dict:
        """Make all time shifts in the sequence relative to the start time of the sequence,
        and normalize time values,
        and remove any time shifts for anchor events.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with trimmed time shifts.
        """
        start_time = sequence["time"]
        for event in sequence["events"]:
            if event.type == EventType.TIME_SHIFT:
                event.value = int((event.value - start_time) * STEPS_PER_MILLISECOND)

        # Loop through the events in reverse to remove any time shifts that occur before anchor events
        events = sequence["events"]
        delete_next_time_shift = False
        for i in range(len(events) - 1, -1, -1):
            if events[i].type == EventType.TIME_SHIFT and delete_next_time_shift:
                delete_next_time_shift = False
                del events[i]
                continue
            elif events[i].type in [EventType.BEZIER_ANCHOR, EventType.PERFECT_ANCHOR, EventType.CATMULL_ANCHOR,
                                    EventType.RED_ANCHOR]:
                delete_next_time_shift = True

        sequence["events"] = events
        del sequence["time"]

        return sequence

    def _tokenize_sequence(self, sequence: dict) -> dict:
        """Tokenize the event sequence.

        Begin token sequence with `[SOS]` token (start-of-sequence).
        End token sequence with `[EOS]` token (end-of-sequence).

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with tokenized events.
        """
        tokens = torch.empty(len(sequence["events"]) + 2, dtype=torch.long)
        tokens[0] = self.tokenizer.sos_id
        for i, event in enumerate(sequence["events"]):
            tokens[i + 1] = self.tokenizer.encode(event)
        tokens[-1] = self.tokenizer.eos_id
        sequence["tokens"] = tokens
        del sequence["events"]
        return sequence

    def _pad_and_split_token_sequence(self, sequence: dict) -> dict:
        """Pad token sequence to a fixed length.

        Pad with `[PAD]` tokens until `token_seq_len`.

        Token sequence (w/o last token) is the input to the transformer decoder,
        token sequence (w/o first token) is the label, a.k.a. decoder ground truth.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded tokens.
        """
        tokens = sequence["tokens"]
        n = min(self.token_seq_len, len(tokens))
        padded_tokens = (
                torch.ones(self.token_seq_len, dtype=tokens.dtype, device=tokens.device)
                * self.tokenizer.pad_id
        )
        padded_tokens[:n] = tokens[:n]
        sequence["decoder_input_ids"] = padded_tokens[:-1]
        # noinspection PyTypeChecker
        sequence["labels"] = torch.where(padded_tokens[1:] == self.tokenizer.pad_id, LABEL_IGNORE_ID, padded_tokens[1:])
        sequence["decoder_attention_mask"] = padded_tokens[:-1] != self.tokenizer.pad_id
        del sequence["tokens"]
        return sequence

    def _pad_frame_sequence(self, sequence: dict) -> dict:
        """Pad frame sequence with zeros until `frame_seq_len`.

        Frame sequence can be further processed into Mel spectrogram frames,
        which is the input to the transformer encoder.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded frames.
        """
        frames = torch.from_numpy(sequence["frames"]).to(torch.float32)

        if frames.shape[0] != self.frame_seq_len:
            n = min(self.frame_seq_len, len(frames))
            padded_frames = torch.zeros(
                self.frame_seq_len,
                frames.shape[-1],
                dtype=frames.dtype,
                device=frames.device,
            )
            padded_frames[:n] = frames[:n]
            sequence["frames"] = torch.flatten(padded_frames)
        else:
            sequence["frames"] = torch.flatten(frames)

        return sequence

    def __iter__(self):
        return self._get_next_tracks() if self.per_track else self._get_next_beatmaps()

    def _get_next_beatmaps(self) -> dict:
        for beatmap_path in self.beatmap_files:
            audio_path = beatmap_path.parents[1] / list(beatmap_path.parents[1].glob('audio.*'))[0]
            audio_samples = self._load_audio_file(audio_path)

            for sample in self._get_next_beatmap(audio_samples, beatmap_path):
                yield sample

    def _get_next_tracks(self) -> dict:
        for track_path in self.beatmap_files:
            beatmap_files = track_path.glob("beatmaps/*")
            audio_path = track_path / list(track_path.glob('audio.*'))[0]
            audio_samples = self._load_audio_file(audio_path)

            for beatmap_path in beatmap_files:
                for sample in self._get_next_beatmap(audio_samples, beatmap_path):
                    yield sample

    def _get_next_beatmap(self, audio_samples, beatmap_path) -> dict:
        osu_beatmap = Beatmap.from_path(beatmap_path)
        current_idx = int(os.path.basename(beatmap_path)[:6])
        current_id = osu_beatmap.beatmap_id

        frames, frame_times = self._get_frames(audio_samples)
        events = self.parser.parse(osu_beatmap)

        sequences = self._create_sequences(
            events,
            frames,
            frame_times,
            current_idx,
            current_id,
        )

        for sequence in sequences:
            sequence = self._trim_time_shifts(sequence)
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            # if sequence["tokens"][1] == self.tokenizer.eos_id:
            #    continue
            yield sequence
