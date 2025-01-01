import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from config import InferenceConfig
from osu_diffusion import timestep_embedding, Tokenizer
from osu_diffusion import repeat_type
from osu_diffusion import create_diffusion
from osu_diffusion import DiT
from osuT5.osuT5.inference import GenerationConfig
from osuT5.osuT5.dataset import update_event_times
from osuT5.osuT5.tokenizer import Event, EventType
from osuT5.osuT5.dataset.data_utils import get_groups, get_group_indices


def get_beatmap_idx(path) -> dict[int, int]:
    p = Path(path)
    with p.open("rb") as f:
        beatmap_idx = pickle.load(f)
    return beatmap_idx


class DiffisionPipeline(object):
    def __init__(
            self,
            args: InferenceConfig,
            model: DiT,
            tokenizer: Tokenizer,
            refine_model: DiT = None,
    ):
        """Model inference stage that generates positions for distance events."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.refine_model = refine_model
        self.diffusion_steps = args.diffusion.model.diffusion_steps
        self.noise_schedule = args.diffusion.model.noise_schedule
        self.seq_len = args.diffusion.data.seq_len
        self.max_seq_len = args.max_seq_len
        self.overlap_buffer = args.overlap_buffer
        self.timesteps = args.timesteps
        self.cfg_scale = args.diff_cfg_scale
        self.refine_iters = args.refine_iters
        self.random_init = args.random_init
        self.types_first = args.osut5.data.types_first
        self.pad_sequence = args.pad_sequence

    def get_class_vector(
            self,
            config: GenerationConfig,
    ) -> torch.Tensor:
        """Get class vector for the given beatmap."""
        class_vector = torch.zeros(self.tokenizer.num_tokens)
        if self.tokenizer.num_classes > 0:
            if config.beatmap_id is not None:
                class_vector[self.tokenizer.encode_style(config.beatmap_id)] = 1
                if config.beatmap_id not in self.tokenizer.beatmap_idx:
                    print(f"Beatmap class {config.beatmap_id} not found. Using default.")
            else:
                class_vector[self.tokenizer.style_unk] = 1
        if self.tokenizer.num_diff_classes > 0:
            if config.difficulty is not None:
                class_vector[self.tokenizer.encode_diff(config.difficulty)] = 1
            else:
                class_vector[self.tokenizer.diff_unk] = 1
        if self.tokenizer.num_mapper_classes > 0:
            if config.mapper_id is not None:
                class_vector[self.tokenizer.encode_mapper(config.mapper_id)] = 1
                if config.mapper_id not in self.tokenizer.mapper_idx:
                    print(f"Mapper class {config.mapper_id} not found. Using default.")
            else:
                class_vector[self.tokenizer.mapper_unk] = 1
        if self.tokenizer.num_descriptor_classes > 0:
            if config.descriptors is not None and len(config.descriptors) > 0:
                if all(descriptor not in self.tokenizer.descriptor_idx for descriptor in config.descriptors):
                    print("Descriptor classes not found. Using default.")
                    class_vector[self.tokenizer.descriptor_unk] = 1
                else:
                    for descriptor in config.descriptors:
                        if descriptor in self.tokenizer.descriptor_idx:
                            class_vector[self.tokenizer.encode_descriptor_name(descriptor)] = 1
                        else:
                            print(f"Descriptor class {descriptor} not found. Skipping.")
            else:
                class_vector[self.tokenizer.descriptor_unk] = 1
        if self.tokenizer.num_cs_classes > 0:
            if config.circle_size is not None:
                class_vector[self.tokenizer.encode_cs(config.circle_size)] = 1
            else:
                class_vector[self.tokenizer.cs_unk] = 1
        return class_vector

    def generate(
            self,
            events: list[Event],
            generation_config: GenerationConfig,
            verbose: bool = False,
    ) -> list[Event]:
        """Generate position events for distance events in the Event list.

        Args:
            events: List of Event objects with distance events.
            generation_config: GenerationConfig object with beatmap metadata.
            verbose: Whether to print debug information.

        Returns:
            events: List of Event objects with position events.
        """

        seq_x, seq_c, seq_len, seq_indices = self.events_to_sequence(events)
        if verbose:
            print(f"seq len {seq_len}")

        if seq_len == 0:
            return events

        diffusion = create_diffusion(
            timestep_respacing=self.timesteps,
            diffusion_steps=self.diffusion_steps,
            noise_schedule=self.noise_schedule,
        )

        # Create banded matrix attention mask for increased sequence length
        attn_mask = torch.full((seq_len, seq_len), True, dtype=torch.bool, device=self.device)
        for i in range(seq_len):
            attn_mask[max(0, i - self.seq_len): min(seq_len, i + self.seq_len), i] = False

        class_vector = self.get_class_vector(generation_config)
        unk_class_vector = self.get_class_vector(GenerationConfig(
            difficulty=generation_config.difficulty,
            descriptors=generation_config.negative_descriptors,
            circle_size=generation_config.circle_size,
        ))

        # Create sampling noise:
        n = 1
        z = seq_x.repeat(n, 1, 1).to(self.device)
        c = seq_c.repeat(n, 1, 1).to(self.device)
        y = class_vector.repeat(n, 1).to(self.device)
        y_null = unk_class_vector.repeat(n, 1).to(self.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        c = torch.cat([c, c], 0)
        y = torch.cat([y, y_null], 0)

        if self.random_init:
            z = torch.randn(*z.shape, device=z.device)

        def sample_part(z, start, end, start_mask_size=0):
            z_part = z[:, :, start:end]
            c_part = c[:, :, start:end]
            attn_mask_part = attn_mask[start:end, start:end]
            key_padding_mask = None

            # Pad to max seq len
            pad_amount = self.max_seq_len - z_part.shape[2] if self.pad_sequence else 0
            if pad_amount > 0:
                z_part = torch.nn.functional.pad(z_part, (0, pad_amount))
                c_part = torch.nn.functional.pad(c_part, (0, pad_amount))
                attn_mask_part = torch.nn.functional.pad(attn_mask_part, (0, pad_amount, 0, pad_amount), value=False)
                key_padding_mask = torch.full((z_part.shape[0], self.max_seq_len), False, dtype=torch.bool, device=self.device)
                key_padding_mask[:, -pad_amount:] = True

            model_kwargs = dict(
                c=c_part,
                y=y,
                cfg_scale=self.cfg_scale,
                attn_mask=attn_mask_part,
                key_padding_mask=key_padding_mask,
            )

            # Make in-paint mask
            def in_paint_mask(x):
                return torch.where(mask, x, z_part)

            # True means it will be generated
            mask = torch.full_like(z_part, False, dtype=torch.bool)
            mask[:, :, start_mask_size:] = True
            z_part = in_paint_mask(z_part)

            # Sample positions:
            samples = diffusion.p_sample_loop(
                self.model.forward_with_cfg,
                z_part.shape,
                z_part,
                denoised_fn=in_paint_mask,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=verbose,
                device=self.device,
            )

            # Refine result with refine model
            if self.refine_model is not None:
                refine_iters = tqdm(range(self.refine_iters)) if verbose else range(self.refine_iters)
                for _ in refine_iters:
                    t = torch.tensor([0] * samples.shape[0], device=self.device)
                    with torch.no_grad():
                        out = diffusion.p_sample(
                            self.model.forward_with_cfg,
                            samples,
                            t,
                            denoised_fn=in_paint_mask,
                            clip_denoised=True,
                            model_kwargs=model_kwargs,
                        )
                        samples = out["sample"]

            # Remove the padding
            if pad_amount > 0:
                samples = samples[:, :, :-pad_amount]

            return samples

        full_samples = z.clone()
        for i in range(0, seq_len - self.overlap_buffer * 2, self.max_seq_len - self.overlap_buffer * 2):
            end = min(i + self.max_seq_len, seq_len)
            if i > 0:
                # The first buffer is already done
                # The second buffer is generated but should be regenerated, so we reset it to the random noise
                full_samples[:, :, i + self.overlap_buffer:i + self.overlap_buffer * 2] = z[:, :, i + self.overlap_buffer:i + self.overlap_buffer * 2]
            samples = sample_part(full_samples, i, end, start_mask_size=self.overlap_buffer if i > 0 else 0)
            full_samples[:, :, i:end] = samples

        def to_positions(samples):
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples += 1
            samples /= 2
            samples *= torch.tensor((512, 384), device=self.device).repeat(n, 1).unsqueeze(2)
            return samples.cpu()

        positions = to_positions(full_samples)
        return self.events_with_pos(events, positions.squeeze(0), seq_indices)

    def events_to_sequence(self, events: list[Event]) -> tuple[torch.Tensor, torch.Tensor, int, dict[int, int]]:
        # Calculate the time of every event and interpolate time for control point events
        event_times = []
        update_event_times(events, event_times, types_first=self.types_first)

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

        groups = get_groups(events, event_times=event_times, types_first=self.types_first)
        group_indices = get_group_indices(events, self.types_first)
        assert len(groups) == len(group_indices)

        seq_indices = {}
        indices = []
        data_chunks = []
        head_time = 0
        last_anchor_time = 0
        last_pos = (256, 192)
        for i, group in enumerate(groups):
            indices.extend(group_indices[i])

            if group.event_type not in event_index:
                continue

            time = group.time
            index = event_index[group.event_type]

            # Handle NC index offset
            if group.event_type in nc_types and group.new_combo:
                index += 1

            # Add slider end repeats index offset
            if group.event_type == EventType.SLIDER_END:
                span_duration = last_anchor_time - head_time
                total_duration = time - head_time
                repeats = max(int(round(total_duration / span_duration)), 1) if span_duration > 0 else 1
                index += repeat_type(repeats)
            elif group.event_type == EventType.SLIDER_HEAD:
                head_time = time
            elif group.event_type == EventType.LAST_ANCHOR:
                last_anchor_time = time

            if not group.x or not group.y:
                group.x, group.y = 256, 192

            pos = (group.x, group.y)

            if not group.distance:
                group.distance = ((pos[0] - last_pos[0]) ** 2 + (pos[1] - last_pos[1]) ** 2) ** 0.5

            features = torch.zeros(20)
            features[0] = pos[0]
            features[1] = pos[1]
            features[2] = time
            features[3] = group.distance
            features[index + 4] = 1
            data_chunks.append(features)

            for j in indices:
                seq_indices[j] = len(data_chunks) - 1
            indices = []

            last_pos = pos

        for j in indices:
            seq_indices[j] = len(data_chunks) - 1

        if len(data_chunks) == 0:
            return torch.zeros(2, 0), torch.zeros(1, 0), 0, {}

        seq = torch.stack(data_chunks, 0)
        seq = torch.swapaxes(seq, 0, 1)
        seq_x = seq[:2, :] / torch.tensor((512, 384)).unsqueeze(1) * 2 - 1
        seq_o = seq[2, :]
        seq_d = seq[3, :]
        seq_c = torch.concatenate(
            [
                timestep_embedding(seq_o * 0.1, 128).T,
                timestep_embedding(seq_d, 128).T,
                seq[4:, :],
            ],
            0,
        )

        return seq_x, seq_c, seq.shape[1], seq_indices

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
