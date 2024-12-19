import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from scipy.signal import find_peaks
from tqdm import tqdm

from ..dataset.data_utils import get_song_length, get_groups, BEAT_TYPES
from ..tokenizer import ContextType, EventType, Event
from .preprocessor import Preprocessor
from .processor import Processor, GenerationConfig, MILISECONDS_PER_SECOND
from ..tokenizer.tokenizer import MILISECONDS_PER_STEP


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
        self.processor.num_beams = args.timer_num_beams
        self.processor.top_p = 1
        self.processor.top_k = 50
        self.bpm_change_threshold = args.timer_bpm_threshold

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
        tpbs = []

        iterator = tqdm(range(iterations)) if verbose else range(iterations)
        for _ in iterator:
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
            last_beat_time = None
            last_group_type = None
            for group in groups:
                time = group.time - audio_offset
                if time < 0 or time >= num_miliseconds:
                    continue
                if group.event_type not in BEAT_TYPES:
                    continue
                if group.event_type == EventType.BEAT:
                    beats[time] += 1
                elif group.event_type == EventType.MEASURE:
                    measures[time] += 1
                elif group.event_type == EventType.TIMING_POINT:
                    timing_points[time] += 1

                if (last_beat_time is not None and last_beat_time != time and
                        not (group.event_type == EventType.TIMING_POINT and last_group_type != EventType.TIMING_POINT)):
                    tpb = (time - last_beat_time) // MILISECONDS_PER_STEP
                    if 20 < tpb < 100:
                        tpbs.append((last_beat_time, tpb))

                last_beat_time = time
                last_group_type = group.event_type

        # # plot beats+measures+timing_points histograms
        # import matplotlib as mpl
        # import pywt
        # mpl.use('TkAgg')
        # plt.ion()
        # plt.figure(figsize=(10, 6))
        # plt.show()
        #
        # def plot2(s):
        #     plt.cla()
        #     plt.plot(s)
        #
        # def plot3(x, y, **kwargs):
        #     plt.cla()
        #     plt.scatter(x, y, **kwargs)

        # Sort the ticks per beats points
        tpbs = sorted(tpbs, key=lambda x: x[0])

        signal = beats + measures + timing_points * 2
        peakind, properties = find_peaks(signal, distance=50, prominence=1, rel_height=1, width=2, wlen=30)
        prominences = properties["prominences"]
        # TODO: Fit Gaussians to interpolate peak positions

        # For each peak determine the BPM by taking nearby BPMs and get the interpolated most common BPM
        # Use peak finding and take the highest peak
        # If there is no clear peak, we don't assign a BPM value and instead infer it from the surrounding BPMs
        def get_peak_bpms(w=300, thresh=0.6):
            peak_bpms = []
            for peak in peakind:
                nearby_tpbs = [tpb for time, tpb in tpbs if peak - w < time < peak + w]
                hist, bins = np.histogram(nearby_tpbs, bins=range(20, 100))
                if hist.max() > thresh * hist.sum():
                    peak_bpms.append(60_000 / (bins[np.argmax(hist)] * 10))
                else:
                    peak_bpms.append(None)
            # plot3(peakind, peak_bpms)
            return peak_bpms

        peak_bpms = get_peak_bpms(100, self.bpm_change_threshold)

        # Fill in the missing BPM values by finding the nearest BPM value
        for i, bpm in enumerate(peak_bpms):
            if bpm is not None:
                continue
            left = i - 1
            while left >= 0 and peak_bpms[left] is None:
                left -= 1
            right = i + 1
            while right < len(peak_bpms) and peak_bpms[right] is None:
                right += 1
            if left >= 0 and (right >= len(peak_bpms) or i - left <= right - i):
                peak_bpms[i] = peak_bpms[left]
            elif right < len(peak_bpms) and (left < 0 or i - left > right - i):
                peak_bpms[i] = peak_bpms[right]
            else:
                peak_bpms[i] = 150  # Default BPM

        # Go from one peak to the next. Use the BPM to estimate where the next beat should be and find the nearest.
        # Depending on how clear the beats are, stick closer to the current BPM.
        peaks = list(zip(peakind, prominences, peak_bpms))
        beat_times = []
        to_process = sorted(peaks, key=lambda x: x[1], reverse=True)

        def remove_range(t1, t2):
            if t1 > t2:
                t1, t2 = t2, t1
            i = 0
            while i < len(to_process):
                peak = to_process[i]
                if t1 <= peak[0] <= t2:
                    to_process.pop(i)
                    i -= 1
                i += 1

        def walk(time, period_ms, direction):
            while True:
                previous_time = time
                time += direction * period_ms
                if not (0 <= time < num_miliseconds):
                    remove_range(previous_time, time)
                    break
                nearest_peak: tuple = min(peaks, key=lambda x: abs(x[0] - time) / (x[1] ** 2))
                if abs(nearest_peak[0] - time) / (nearest_peak[1] ** 2) < 30:
                    # There is a good beat nearby
                    if nearest_peak not in to_process:
                        # This beat has already been processed
                        # Remove all peaks between the previous time and the current time from to_process
                        remove_range(previous_time, time)
                        break
                    time = nearest_peak[0]
                    period_ms = 60_000 / nearest_peak[2]
                else:
                    if abs(nearest_peak[0] - time) / (nearest_peak[1] ** 2) < 300:
                        # There is a beat nearby, but it's likely on another BPM
                        # FIXME: Prevent very near beats at the seams
                        break
                    # There is no beat nearby, so make an imaginary beat
                beat_times.append(int(time))
                # Remove all peaks between the previous time and the current time from to_process
                remove_range(previous_time, time)

        while to_process:
            peak = to_process.pop(0)
            time = peak[0]
            period_ms = 60_000 / peak[2]
            beat_times.append(int(time))
            walk(time, period_ms, 1)
            walk(time, period_ms, -1)

        beat_times = sorted(beat_times)

        # intervals = np.diff(peakind)
        # bpm_values = 60_000 / intervals  # 60,000 ms per minute
        # bpm = round(np.median(bpm_values), 2)
        # # Normalize all bpm values in to the range [bpm/1.5, bpm*1.5] by integer division or multiplication
        # bpm_values = bpm_values / np.ceil(bpm_values / (bpm * 1.5))
        # bpm_values = bpm_values * np.ceil((bpm / 1.5) / bpm_values)
        #
        # def test_bpm(bpm, w=1):
        #     # Generate periodic pulse train
        #     period_ms = 60_000 / bpm
        #     pulse_train = np.zeros_like(signal)
        #     for i in range(w):
        #         pulse_train[np.arange(i, len(pulse_train), period_ms).astype(int)] = 1  # Period in samples
        #     # Cross-correlation
        #     correlation = correlate(signal, pulse_train, mode="full")
        #     offset = np.argmax(correlation) - len(signal) + w // 2
        #     offset += period_ms * np.ceil(-offset / period_ms)
        #     return bpm, offset, correlation.max() / np.sqrt(bpm)
        #
        # bpm, offset, _ = max([test_bpm(bpm, 5) for bpm in np.arange(bpm - 1, bpm + 1, 0.01)], key=lambda x: x[2])

        events = []
        w = 10
        for beat_time in beat_times:
            # Classify the peak as a beat, measure, or timing point
            beat = beats[beat_time - w:beat_time + w].sum()
            measure = measures[beat_time - w:beat_time + w].sum()
            timing_point = timing_points[beat_time - w:beat_time + w].sum()
            if beat > measure and beat > timing_point:
                event_type = EventType.BEAT
            elif measure > beat and measure > timing_point:
                # FIXME: Improve regularity of measures
                event_type = EventType.MEASURE
            else:
                # FIXME: Ignore timing points with low evidence
                event_type = EventType.TIMING_POINT

            events.append(Event(event_type))
            events.append(Event(EventType.TIME_SHIFT, beat_time))

        # def plot():
        #     plt.cla()
        #     plt.plot(signal, label='beats')
        #     plt.plot(measures, label='measures')
        #     plt.plot(timing_points, label='timing_points')
        #     plt.vlines(x=peakind, ymin=-prominences, ymax=0, color='b')
        #     plt.vlines(x=beat_times, ymin=-0.5, ymax=0, color='r')
        #     # plt.legend()

        # plot()
        # bpms = [(t, 60000 / tpb / 10) for (t, tpb) in tpbs]
        # plot3(*list(zip(*bpms)), alpha=0.1)
        # plot()
        # plt.pause(0.001)

        return events
