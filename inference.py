from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from slider import Beatmap

from osuT5.inference import Preprocessor, Pipeline, Postprocessor
from osuT5.tokenizer import Tokenizer
from osuT5.utils import get_config, get_model


def get_args_from_beatmap(args: DictConfig):
    if args.beatmap_path is None:
        return

    beatmap_path = Path(args.beatmap_path)
    beatmap = Beatmap.from_path(beatmap_path)
    args.audio_path = beatmap_path.parent / beatmap.audio_filename
    args.output_path = beatmap_path.parent
    args.bpm = beatmap.bpm_max()
    args.offset = min(tp.offset.total_seconds() * 1000 for tp in beatmap.timing_points)
    args.slider_multiplier = beatmap.slider_multiplier
    args.title = beatmap.title
    args.artist = beatmap.artist
    args.beatmap_id = beatmap.beatmap_id if args.beatmap_id == -1 else args.beatmap_id


@hydra.main(config_path="configs", config_name="inference", version_base="1.1")
def main(args: DictConfig):
    get_args_from_beatmap(args)

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    t5_config = get_config(args)

    model = get_model(t5_config)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    tokenizer = Tokenizer()
    preprocessor = Preprocessor(args)
    pipeline = Pipeline(args, tokenizer)
    postprocessor = Postprocessor(args)

    audio = preprocessor.load(args.audio_path)
    sequences = preprocessor.segment(audio)
    events = pipeline.generate(model, sequences)
    postprocessor.generate(events)


if __name__ == "__main__":
    main()
