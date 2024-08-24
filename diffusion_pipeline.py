import pickle
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from osu_diffusion import timestep_embedding
from osu_diffusion import repeat_type
from osu_diffusion import create_diffusion
from osu_diffusion import DiT
from osuT5.osuT5.dataset import update_event_times
from osuT5.osuT5.tokenizer import Event, EventType


def get_beatmap_idx(path) -> dict[int, int]:
    p = Path(path)
    with p.open("rb") as f:
        beatmap_idx = pickle.load(f)
    return beatmap_idx


class DiffisionPipeline(object):
    def __init__(self, args: DictConfig):
        """Model inference stage that generates positions for distance events."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_steps = args.diffusion.diffusion_steps
        self.cfg_scale = args.cfg_scale
        self.seq_len = args.diffusion.seq_len
        self.num_classes = args.diffusion.num_classes
        self.beatmap_idx = get_beatmap_idx(args.beatmap_idx)
        self.style_id = args.style_id
        self.refine_iters = args.refine_iters
        self.random_init = args.random_init

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

        seq_x, seq_o, seq_c, seq_len, seq_indices = self.events_to_sequence(events)

        seq_o = seq_o - seq_o[0]  # Normalize to relative time
        print(f"seq len {seq_len}")

        diffusion = create_diffusion(
            [100,0,0,0,0,0,0,0,0,0],
            diffusion_steps=self.diffusion_steps,
            noise_schedule="squaredcos_cap_v2",
        )

        # Create banded matrix attention mask for increased sequence length
        attn_mask = torch.full((seq_len, seq_len), True, dtype=torch.bool, device=self.device)
        for i in range(seq_len):
            attn_mask[max(0, i - self.seq_len): min(seq_len, i + self.seq_len), i] = False

        class_labels = [self.class_label]

        # Create sampling noise:
        n = len(class_labels)
        z = seq_x.repeat(n, 1, 1).to(self.device)
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

        if self.random_init:
            z = torch.randn(*z.shape, device=z.device)

        def to_positions(samples):
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples *= torch.tensor((512, 384), device=self.device).repeat(n, 1).unsqueeze(2)
            return samples.cpu()

        # Sample positions:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )

        # Refine result with refine model
        if refine_model is not None:
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
    def events_to_sequence(events: list[Event]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[int, int]]:
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
        last_pos = (256, 192)
        pos = (256, 192)
        distance_defined = False
        for i, event in enumerate(events):
            indices.append(i)
            if event.type == EventType.DISTANCE:
                distance = event.value
                distance_defined = True
            elif event.type == EventType.POS_X:
                pos = (event.value, pos[1])
            elif event.type == EventType.POS_Y:
                pos = (pos[0], event.value)
                if not distance_defined:
                    distance = ((pos[0] - last_pos[0]) ** 2 + (pos[1] - last_pos[1]) ** 2) ** 0.5
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
                    repeats = max(int(round(total_duration / span_duration)), 1) if span_duration > 0 else 1
                    index += repeat_type(repeats)
                elif event.type == EventType.SLIDER_HEAD:
                    head_time = time
                elif event.type == EventType.LAST_ANCHOR:
                    last_anchor_time = time

                features = torch.zeros(20)
                features[0] = pos[0]
                features[1] = pos[1]
                features[2] = time
                features[3] = distance
                features[index + 4] = 1
                data_chunks.append(features)

                for j in indices:
                    seq_indices[j] = len(data_chunks) - 1
                indices = []

                last_pos = pos

        for j in indices:
            seq_indices[j] = len(data_chunks) - 1

        seq = torch.stack(data_chunks, 0)
        seq = torch.swapaxes(seq, 0, 1)
        seq_x = seq[:2, :] / torch.tensor((512, 384)).unsqueeze(1)
        seq_o = seq[2, :]
        seq_d = seq[3, :]
        seq_c = torch.concatenate(
            [
                timestep_embedding(seq_d, 128).T,
                seq[4:, :],
            ],
            0,
        )

        return seq_x, seq_o, seq_c, seq.shape[1], seq_indices

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
            elif event.type == EventType.POS_X:
                index = seq_indices[i]
                pos_x = sampled_seq[0, index].item()
                new_events.append(Event(EventType.POS_X, int(round(pos_x))))
            elif event.type == EventType.POS_Y:
                index = seq_indices[i]
                pos_y = sampled_seq[1, index].item()
                new_events.append(Event(EventType.POS_Y, int(round(pos_y))))
            else:
                new_events.append(event)

        return new_events
