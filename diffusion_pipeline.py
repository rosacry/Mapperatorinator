import pickle
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from osu_diffusion import timestep_embedding
from osu_diffusion import repeat_type
from osu_diffusion import create_diffusion
from osu_diffusion import DiT
from osuT5.dataset.data_utils import update_event_times
from osuT5.tokenizer import Event, EventType


def get_beatmap_idx(path) -> dict[int, int]:
    p = Path(path)
    with p.open("rb") as f:
        beatmap_idx = pickle.load(f)
    return beatmap_idx


class DiffisionPipeline(object):
    def __init__(self, args: DictConfig):
        """Model inference stage that generates positions for distance events."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_sampling_steps = args.num_sampling_steps
        self.cfg_scale = args.cfg_scale
        self.seq_len = args.diffusion.seq_len
        self.num_classes = args.diffusion.num_classes
        self.beatmap_idx = get_beatmap_idx(args.beatmap_idx)
        self.style_id = args.style_id
        self.refine_iters = args.refine_iters

        if self.style_id in self.beatmap_idx:
            self.class_label = self.beatmap_idx[self.style_id]
        else:
            print(f"Beatmap ID {self.style_id} not found in dataset, using default style.")
            self.class_label = self.num_classes

    def generate(self, model: DiT, events: list[Event], refine_model: DiT = None) -> list[Event]:
        """Generate position events for distance events in the Event list.

        Args:
            model: Trained model to use for inference.
            events: List of Event objects with distance events.
            refine_model: Optional model to refine the generated positions.

        Returns:
            events: List of Event objects with position events.
        """

        seq_o, seq_c, seq_len, seq_indices = self.events_to_sequence(events)

        seq_o = seq_o - seq_o[0]  # Normalize to relative time
        print(f"seq len {seq_len}")

        diffusion = create_diffusion(
            str(self.num_sampling_steps),
            noise_schedule="squaredcos_cap_v2",
        )

        # Create banded matrix attention mask for increased sequence length
        attn_mask = torch.full((seq_len, seq_len), True, dtype=torch.bool, device=self.device)
        for i in range(seq_len):
            attn_mask[max(0, i - self.seq_len): min(seq_len, i + self.seq_len), i] = False

        class_labels = [self.class_label]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 2, seq_len, device=self.device)
        o = seq_o.repeat(n, 1).to(self.device)
        c = seq_c.repeat(n, 1, 1).to(self.device)
        y = torch.tensor(class_labels, device=self.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        o = torch.cat([o, o], 0)
        c = torch.cat([c, c], 0)
        y_null = torch.tensor([self.num_classes] * n, device=self.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(o=o, c=c, y=y, cfg_scale=self.cfg_scale, attn_mask=attn_mask)

        def to_positions(samples):
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples *= torch.tensor((512, 384), device=self.device).repeat(n, 1).unsqueeze(2)
            return samples.cpu()

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )

        if refine_model is not None:
            # Refine result with refine model
            for _ in tqdm(range(self.refine_iters)):
                t = torch.tensor([0] * samples.shape[0], device=self.device)
                with torch.no_grad():
                    out = diffusion.p_sample(
                        model.forward_with_cfg,
                        samples,
                        t,
                        clip_denoised=True,
                        model_kwargs=model_kwargs,
                    )
                    samples = out["sample"]

        positions = to_positions(samples)
        return self.events_with_pos(events, positions.squeeze(0), seq_indices)

    @staticmethod
    def events_to_sequence(events: list[Event]) -> tuple[torch.Tensor, torch.Tensor, int, dict[int, int]]:
        # Calculate the time of every event and interpolate time for control point events
        event_times = []
        update_event_times(events, event_times)

        # Calculate the number of repeats for each slider end event
        # Convert to vectorized form for osu-diffusion
        nc_types = [EventType.CIRCLE, EventType.SLIDER_HEAD]
        event_index = {
            EventType.CIRCLE: 0,
            EventType.SPINNER: 2,
            EventType.SPINNER_END: 3,
            EventType.SLIDER_HEAD: 4,
            EventType.BEZIER_ANCHOR: 6,
            EventType.PERFECT_ANCHOR: 7,
            EventType.CATMULL_ANCHOR: 8,
            EventType.RED_ANCHOR: 9,
            EventType.LAST_ANCHOR: 10,
            EventType.SLIDER_END: 11,
        }

        seq_indices = {}
        indices = []
        data_chunks = []
        distance = 0
        new_combo = False
        head_time = 0
        last_anchor_time = 0
        for i, event in enumerate(events):
            indices.append(i)
            if event.type == EventType.DISTANCE:
                distance = event.value
            elif event.type == EventType.NEW_COMBO:
                new_combo = True
            elif event.type in event_index:
                time = event_times[i]
                index = event_index[event.type]

                # Handle NC index offset
                if event.type in nc_types and new_combo:
                    index += 1
                    new_combo = False

                # Add slider end repeats index offset
                if event.type == EventType.SLIDER_END:
                    span_duration = last_anchor_time - head_time
                    total_duration = time - head_time
                    repeats = max(int(round(total_duration / span_duration)), 1)
                    index += repeat_type(repeats)
                elif event.type == EventType.SLIDER_HEAD:
                    head_time = time
                elif event.type == EventType.LAST_ANCHOR:
                    last_anchor_time = time

                features = torch.zeros(18)
                features[0] = time
                features[1] = distance
                features[index + 2] = 1
                data_chunks.append(features)

                for j in indices:
                    seq_indices[j] = len(data_chunks) - 1
                indices = []

        for j in indices:
            seq_indices[j] = len(data_chunks) - 1

        seq = torch.stack(data_chunks, 0)
        seq = torch.swapaxes(seq, 0, 1)
        seq_o = seq[0, :]
        seq_d = seq[1, :]
        seq_c = torch.concatenate(
            [
                timestep_embedding(seq_d, 128).T,
                seq[2:, :],
            ],
            0,
        )

        return seq_o, seq_c, seq.shape[1], seq_indices

    @staticmethod
    def events_with_pos(events: list[Event], sampled_seq: torch.Tensor, seq_indices: dict[int, int]) -> list[Event]:
        new_events = []

        for i, event in enumerate(events):
            if event.type == EventType.DISTANCE:
                index = seq_indices[i]
                pos_x = sampled_seq[0, index].item()
                pos_y = sampled_seq[1, index].item()
                new_events.append(Event(EventType.POS_X, int(round(pos_x))))
                new_events.append(Event(EventType.POS_Y, int(round(pos_y))))
            else:
                new_events.append(event)

        return new_events
