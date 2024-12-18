import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from scipy.signal import find_peaks, find_peaks_cwt
from tqdm import tqdm

from ..dataset.data_utils import get_song_length, get_groups
from ..tokenizer import ContextType, EventType, Event
from .preprocessor import Preprocessor
from .processor import Processor, GenerationConfig, MILISECONDS_PER_SECOND


class SuperTimingGenerator:
    def __init__(
            self,
            args: DictConfig,
            model,
            tokenizer,
    ):
        self.args = args
        self.model = model
        self.preprocessor = Preprocessor(args, parallel=True)
        self.processor = Processor(args, model, tokenizer, parallel=True)
        self.processor.do_sample = False
        self.processor.num_beams = args.num_beams
        self.processor.top_p = 1
        self.processor.top_k = 50

        self.frame_seq_len = args.osut5.data.src_seq_len - 1
        self.frame_size = args.osut5.model.spectrogram.hop_length
        self.sample_rate = args.osut5.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.miliseconds_per_sequence = self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate

    def generate(
            self,
            audio: npt.ArrayLike,
            generation_config: GenerationConfig,
            iterations: int = 20,
            verbose: bool = False,
    ):
        in_context = self.processor.get_in_context([ContextType.NONE], None,
                                                   get_song_length(audio, self.sample_rate))

        # Prepare beat histograms
        num_miliseconds = len(audio) * MILISECONDS_PER_SECOND // self.sample_rate
        beats = np.zeros([num_miliseconds], dtype=int)
        measures = np.zeros([num_miliseconds], dtype=int)
        timing_points = np.zeros([num_miliseconds], dtype=int)

        for i in tqdm(range(iterations)):
            audio_offset = np.random.randint(0, self.miliseconds_per_sequence)
            begin_pad = audio_offset * self.sample_rate // MILISECONDS_PER_SECOND
            end_pad = self.samples_per_sequence - begin_pad
            sequences = self.preprocessor.segment(audio, begin_pad, end_pad)
            events = self.processor.generate(
                sequences=sequences,
                generation_config=generation_config,
                in_context=in_context,
                verbose=False,
            )
            groups = get_groups(events, types_first=self.args.osut5.data.types_first)
            for group in groups:
                time = group.time - audio_offset
                if time < 0 or time >= num_miliseconds:
                    continue
                if group.event_type == EventType.BEAT:
                    beats[time] += 1
                elif group.event_type == EventType.MEASURE:
                    measures[time] += 1
                elif group.event_type == EventType.TIMING_POINT:
                    timing_points[time] += 1

        signal = beats + measures + timing_points * 2
        peakind, _ = find_peaks(signal, distance=150, prominence=2, rel_height=1, width=2, wlen=30)

        events = []
        w = 10
        for peak in peakind:
            # Classify the peak as a beat, measure, or timing point
            beat = beats[peak - w:peak + w].sum()
            measure = measures[peak - w:peak + w].sum()
            timing_point = timing_points[peak - w:peak + w].sum()
            if beat > measure and beat > timing_point:
                event_type = EventType.BEAT
            elif measure > beat and measure > timing_point:
                event_type = EventType.MEASURE
            else:
                event_type = EventType.TIMING_POINT

            events.append(Event(event_type))
            events.append(Event(EventType.TIME_SHIFT, peak))

        # # plot beats+measures+timing_points histograms
        # import matplotlib as mpl
        # mpl.use('TkAgg')
        # plt.ion()
        # plt.figure(figsize=(10, 6))
        # plt.show()
        #
        # def plot():
        #     plt.cla()
        #     plt.plot(signal, label='beats')
        #     plt.plot(measures, label='measures')
        #     plt.plot(timing_points, label='timing_points')
        #     plt.vlines(x=peakind, ymin=-1, ymax=0, color='b')
        #     # plt.legend()
        #
        # plot()
        # plt.pause(0.001)

        return events

