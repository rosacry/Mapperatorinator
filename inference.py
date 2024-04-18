from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from slider import Beatmap

from osuT5.inference import Preprocessor, Pipeline, Postprocessor
from osuT5.tokenizer import Tokenizer
from osuT5.utils import get_model


def get_args_from_beatmap(args: DictConfig):
    if args.beatmap_path is None or args.beatmap_path == "":
        return

    beatmap_path = Path(args.beatmap_path)

    if not beatmap_path.is_file():
        raise FileNotFoundError(f"Beatmap file {beatmap_path} not found.")

    beatmap = Beatmap.from_path(beatmap_path)
    args.audio_path = beatmap_path.parent / beatmap.audio_filename
    args.output_path = beatmap_path.parent
    args.bpm = beatmap.bpm_max()
    args.offset = min(tp.offset.total_seconds() * 1000 for tp in beatmap.timing_points)
    args.slider_multiplier = beatmap.slider_multiplier
    args.title = beatmap.title
    args.artist = beatmap.artist
    args.beatmap_id = beatmap.beatmap_id if args.beatmap_id == -1 else args.beatmap_id
    args.difficulty = float(beatmap.stars()) if args.difficulty == -1 else args.difficulty


@hydra.main(config_path="configs", config_name="inference_v1", version_base="1.1")
def main(args: DictConfig):
    get_args_from_beatmap(args)

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.model_path)
    model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device)
    tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl")

    tokenizer = Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    model = get_model(args, tokenizer)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)

    preprocessor = Preprocessor(args)
    pipeline = Pipeline(args, tokenizer)
    postprocessor = Postprocessor(args)

    audio = preprocessor.load(args.audio_path)
    sequences = preprocessor.segment(audio)
    events = pipeline.generate(model, sequences)
    postprocessor.generate(events)


if __name__ == "__main__":
    main()
