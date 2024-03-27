from __future__ import annotations

import json
import os
import random
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig
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
        "args",
        "parser",
        "tokenizer",
        "beatmap_files",
        "test",
    )

    def __init__(
            self,
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            beatmap_files: Optional[list[Path]] = None,
            test: bool = False,
    ):
        """Manage and process ORS dataset.

        Attributes:
            args: Data loading arguments.
            parser: Instance of OsuParser class.
            tokenizer: Instance of Tokenizer class.
            beatmap_files: List of beatmap files to process. Overrides track index range.
            test: Whether to load the test dataset.
        """
        super().__init__()
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.beatmap_files = beatmap_files
        self.test = test

    def _get_beatmap_files(self) -> list[Path]:
        if self.beatmap_files is not None:
            return self.beatmap_files

        # Get a list of all beatmap files in the dataset path in the track index range between start and end
        path, start, end = self._get_dataset_params()
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(start, end)]
        for track_name in track_names:
            for beatmap_file in os.listdir(
                    os.path.join(path, track_name, "beatmaps"),
            ):
                beatmap_files.append(
                    Path(
                        os.path.join(
                            path,
                            track_name,
                            "beatmaps",
                            beatmap_file,
                        )
                    ),
                )

        return beatmap_files

    def _get_track_paths(self) -> list[Path]:
        path, start, end = self._get_dataset_params()
        track_paths = []
        track_names = ["Track" + str(i).zfill(5) for i in range(start, end)]
        for track_name in track_names:
            track_paths.append(Path(os.path.join(path, track_name)))
        return track_paths

    def _get_dataset_params(self):
        path = self.args.test_dataset_path if self.test else self.args.train_dataset_path
        start = self.args.test_dataset_start if self.test else self.args.train_dataset_start
        end = self.args.test_dataset_end if self.test else self.args.train_dataset_end
        return path, start, end

    def __iter__(self):
        beatmap_files = self._get_track_paths() if self.args.per_track else self._get_beatmap_files()

        if not self.test:
            random.shuffle(beatmap_files)

        if self.args.cycle_length > 1:
            return InterleavingBeatmapDatasetIterable(
                beatmap_files,
                self._iterable_factory,
                self.args.cycle_length,
            )

        return self._iterable_factory(beatmap_files).__iter__()

    def _iterable_factory(self, beatmap_files: list[Path]):
        return BeatmapDatasetIterable(
            beatmap_files,
            self.args,
            self.parser,
            self.tokenizer,
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
        "args",
        "parser",
        "tokenizer",
        "frame_seq_len",
        "min_pre_token_len",
        "pre_token_len",
    )

    def __init__(
            self,
            beatmap_files: list[Path],
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
    ):
        self.beatmap_files = beatmap_files
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        # let N = |src_seq_len|
        # N-1 frames creates N mel-spectrogram frames
        self.frame_seq_len = args.src_seq_len - 1
        # let N = |tgt_seq_len|
        # [SOS] token + event_tokens + [EOS] token creates N+1 tokens
        # [SOS] token + event_tokens[:-1] creates N target sequence
        # event_tokens[1:] + [EOS] token creates N label sequence
        self.min_pre_token_len = 64
        self.pre_token_len = args.tgt_seq_len // 2

    def _load_audio_file(self, file: Path) -> npt.NDArray:
        """Load an audio file as a numpy time-series array

        The signals are resampled, converted to mono channel, and normalized.

        Args:
            file: Path to audio file.
        Returns:
            samples: Audio time series.
        """
        audio = AudioSegment.from_file(file, format=file.suffix[1:])
        audio = audio.set_frame_rate(self.args.sample_rate)
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
        samples = np.pad(samples, [0, self.args.hop_length - len(samples) % self.args.hop_length])
        frames = np.reshape(samples, (-1, self.args.hop_length))
        frames_per_milisecond = (
                self.args.sample_rate / self.args.hop_length / MILISECONDS_PER_SECOND
        )
        frame_times = np.arange(len(frames)) / frames_per_milisecond
        return frames, frame_times

    def _create_sequences(
            self,
            events: list[Event],
            frames: npt.NDArray,
            frame_times: npt.NDArray,
            beatmap_idx: int,
            difficulty: float,
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

            split_pre_idx = max(split_start_idx - self.frame_seq_len, 0)
            target_pre_idx = event_start_indices[split_pre_idx]

            # Create the sequence
            sequence = {
                "time": frame_times[split_start_idx],
                "frames": frames[split_start_idx:split_end_idx],
                "events": events[target_start_idx:target_end_idx],
                "pre_events": events[target_pre_idx:target_start_idx],
                "beatmap_idx": beatmap_idx,
                "difficulty": difficulty,
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
        def process(events: list[Event]) -> list[Event]:
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    # We cant modify the event objects themselves because that will affect subsequent sequences
                    events[i] = Event(EventType.TIME_SHIFT, int((event.value - start_time) * STEPS_PER_MILLISECOND))

            # Loop through the events in reverse to remove any time shifts that occur before anchor events
            delete_next_time_shift = False
            for i in range(len(events) - 1, -1, -1):
                if events[i].type == EventType.TIME_SHIFT and delete_next_time_shift:
                    delete_next_time_shift = False
                    del events[i]
                    continue
                elif events[i].type in [EventType.BEZIER_ANCHOR, EventType.PERFECT_ANCHOR, EventType.CATMULL_ANCHOR,
                                        EventType.RED_ANCHOR]:
                    delete_next_time_shift = True

            return events

        start_time = sequence["time"]
        sequence["events"] = process(sequence["events"])
        sequence["pre_events"] = process(sequence["pre_events"])
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

        pre_tokens = torch.empty(len(sequence["pre_events"]), dtype=torch.long)
        for i, event in enumerate(sequence["pre_events"]):
            pre_tokens[i] = self.tokenizer.encode(event)
        sequence["pre_tokens"] = pre_tokens
        del sequence["pre_events"]

        sequence["beatmap_idx_token"] = self.tokenizer.encode_style_idx(sequence["beatmap_idx"])\
            if random.random() >= self.args.class_dropout_prob else self.tokenizer.style_unk

        sequence["difficulty_token"] = self.tokenizer.encode_diff(sequence["difficulty"])\
            if random.random() >= self.args.diff_dropout_prob else self.tokenizer.diff_unk

        sequence["beatmap_idx"] = sequence["beatmap_idx"]\
            if random.random() >= self.args.class_dropout_prob else self.tokenizer.num_classes

        return sequence

    def _pad_and_split_token_sequence(self, sequence: dict) -> dict:
        """Pad token sequence to a fixed length and split decoder input and labels.

        Pad with `[PAD]` tokens until `tgt_seq_len`.

        Token sequence (w/o last token) is the input to the transformer decoder,
        token sequence (w/o first token) is the label, a.k.a. decoder ground truth.

        Prefix the token sequence with the pre_tokens sequence.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded tokens.
        """
        special_token_length = self.args.special_token_len

        tokens = sequence["tokens"]
        pre_tokens = sequence["pre_tokens"]

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=tokens.dtype, device=tokens.device)
        label_tokens = torch.full((self.args.tgt_seq_len,), LABEL_IGNORE_ID, dtype=tokens.dtype, device=tokens.device)

        if self.args.center_pad_decoder:
            n = min(self.args.tgt_seq_len - self.pre_token_len, len(tokens) - 1)
            m = min(self.pre_token_len - special_token_length, len(pre_tokens))
            start_index = self.pre_token_len - m - special_token_length
        else:
            # n + m + special_token_length + padding = tgt_seq_len
            n = min(self.args.tgt_seq_len - special_token_length - min(self.min_pre_token_len, len(pre_tokens)), len(tokens) - 1)
            m = min(self.args.tgt_seq_len - n - special_token_length, len(pre_tokens))
            start_index = 0

        if self.args.diff_token_index >= 0:
            input_tokens[start_index + self.args.diff_token_index] = sequence["difficulty_token"]
        if self.args.style_token_index >= 0:
            input_tokens[start_index + self.args.style_token_index] = sequence["beatmap_idx_token"]
        input_tokens[start_index + special_token_length:start_index + m + special_token_length] = pre_tokens[-m:]
        input_tokens[start_index + m + special_token_length:start_index + m + special_token_length + n] = tokens[:n]
        label_tokens[start_index + m + special_token_length:start_index + m + special_token_length + n] = tokens[1:n + 1]

        # Randomize some input tokens
        # input_tokens = torch.where((self.tokenizer.event_start[EventType.TIME_SHIFT] <= input_tokens) & (input_tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
        #                            torch.clamp(input_tokens + torch.randint_like(input_tokens, -10, 10), self.tokenizer.event_start[EventType.TIME_SHIFT], self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
        #                            input_tokens)
        # input_tokens = torch.where((self.tokenizer.event_start[EventType.DISTANCE] <= input_tokens) & (input_tokens < self.tokenizer.event_end[EventType.DISTANCE]),
        #                               torch.clamp(input_tokens + torch.randint_like(input_tokens, -10, 10), self.tokenizer.event_start[EventType.DISTANCE], self.tokenizer.event_end[EventType.DISTANCE] - 1),
        #                               input_tokens)

        sequence["decoder_input_ids"] = input_tokens
        sequence["decoder_attention_mask"] = input_tokens != self.tokenizer.pad_id
        sequence["labels"] = label_tokens

        del sequence["tokens"]
        del sequence["pre_tokens"]
        del sequence["difficulty_token"]
        del sequence["beatmap_idx_token"]
        del sequence["difficulty"]

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
        return self._get_next_tracks() if self.args.per_track else self._get_next_beatmaps()

    @staticmethod
    def _load_metadata(track_path: Path) -> dict:
        metadata_file = track_path / "metadata.json"
        with open(metadata_file) as f:
            return json.load(f)

    def _get_next_beatmaps(self) -> dict:
        for beatmap_path in self.beatmap_files:
            metadata = self._load_metadata(beatmap_path.parents[1])
            difficulty = metadata["Beatmaps"][beatmap_path.stem]["StandardStarRating"]["0"]

            audio_path = beatmap_path.parents[1] / list(beatmap_path.parents[1].glob('audio.*'))[0]
            audio_samples = self._load_audio_file(audio_path)

            for sample in self._get_next_beatmap(audio_samples, beatmap_path, difficulty):
                yield sample

    def _get_next_tracks(self) -> dict:
        for track_path in self.beatmap_files:
            metadata = self._load_metadata(track_path)

            audio_path = track_path / list(track_path.glob('audio.*'))[0]
            audio_samples = self._load_audio_file(audio_path)

            for beatmap_name in metadata["Beatmaps"]:
                beatmap_path = (track_path / "beatmaps" / beatmap_name).with_suffix(".osu")
                difficulty = metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["0"]

                for sample in self._get_next_beatmap(audio_samples, beatmap_path, difficulty):
                    yield sample

    def _get_next_beatmap(self, audio_samples, beatmap_path, difficulty: float) -> dict:
        osu_beatmap = Beatmap.from_path(beatmap_path)
        current_idx = int(os.path.basename(beatmap_path)[:6])

        frames, frame_times = self._get_frames(audio_samples)
        events = self.parser.parse(osu_beatmap)

        sequences = self._create_sequences(
            events,
            frames,
            frame_times,
            current_idx,
            difficulty,
        )

        for sequence in sequences:
            sequence = self._trim_time_shifts(sequence)
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            if self.args.remove_empty_sequences and ((sequence["labels"] == self.tokenizer.eos_id) | (sequence["labels"] == self.tokenizer.pad_id)).all():
                continue
            # if sequence["decoder_input_ids"][self.pre_token_len - 1] != self.tokenizer.pad_id:
            #     continue
            yield sequence
