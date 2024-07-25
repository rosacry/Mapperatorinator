from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from slider import Beatmap

import routed_pickle
from diffusion_pipeline import DiffisionPipeline
from osuT5.osuT5.inference import Preprocessor, Pipeline, Postprocessor
from osuT5.osuT5.tokenizer import Tokenizer, ContextType
from osuT5.osuT5.utils import get_model
from osu_diffusion import DiT_models


def prepare_args(args: DictConfig):
    if not isinstance(args.context_type, str):
        return
    args.context_type = ContextType(args.context_type) if args.context_type != "" else None


def get_args_from_beatmap(args: DictConfig, tokenizer: Tokenizer):
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
    args.style_id = beatmap.beatmap_id if args.style_id == -1 else args.style_id
    args.difficulty = float(beatmap.stars()) if args.difficulty == -1 else args.difficulty
    args.mapper_id = tokenizer.mapper_idx.get(beatmap.beatmap_id, -1) if args.mapper_id == -1 else args.mapper_id
    args.descriptors = tokenizer.beatmap_descriptors.get(beatmap.beatmap_id, []) if len(args.descriptors) == 0 else args.descriptors
    args.other_beatmap_path = args.beatmap_path


def find_model(ckpt_path, args: DictConfig, device):
    assert Path(ckpt_path).exists(), f"Could not find DiT checkpoint at {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]

    model = DiT_models[args.diffusion.model](
        num_classes=args.diffusion.num_classes,
        context_size=19 - 3 + 128,
    ).to(device)
    model.load_state_dict(checkpoint)
    model.eval()  # important!
    return model


@hydra.main(config_path="configs", config_name="inference_v1", version_base="1.1")
def main(args: DictConfig):
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\584787 Yuiko Ohara - Hoshi o Tadoreba\\Yuiko Ohara - Hoshi o Tadoreba (Yumeno Himiko) [015's Hard].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\859916 DJ Noriken (Remixed _ Covered by Camellia) - Jingle (Metal Arrange _ Cover)\\DJ Noriken (Remixed  Covered by Camellia) - Jingle (Metal Arrange  Cover) (StunterLetsPlay) [Extra].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\989342 Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis\\Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\qyoh for upload\\Camellia - Qyoh (Nine Stars) (OliBomby) [Yoalteuctin (Rabbit Hole Collab)].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\1903968 Kisumi Reika - Sekai wa Futari no Tame ni\\Kisumi Reika - Sekai wa Futari no Tame ni (Ayesha Altugle) [Normal].osu"

    prepare_args(args)

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.model_path)
    model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device)
    tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=routed_pickle)

    tokenizer = Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    get_args_from_beatmap(args, tokenizer)

    model = get_model(args.osut5, tokenizer)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)

    preprocessor = Preprocessor(args)
    pipeline = Pipeline(args, tokenizer)
    postprocessor = Postprocessor(args)

    audio = preprocessor.load(args.audio_path)
    sequences = preprocessor.segment(audio)
    events = pipeline.generate(
        model=model,
        sequences=sequences,
        beatmap_id=args.beatmap_id,
        difficulty=args.difficulty,
        mapper_id=args.mapper_id,
        descriptors=args.descriptors,
        other_beatmap_path=args.other_beatmap_path,
        context_type=args.context_type,
    )

    # Generate timing and resnap timing events
    timing = None
    if args.context_type == ContextType.TIMING or args.context_type == ContextType.NO_HS or args.context_type == ContextType.GD:
        # Exact timing is provided in the other beatmap, so we don't need to generate it
        other_beatmap_path = Path(args.other_beatmap_path)
        timing = Beatmap.from_path(other_beatmap_path).timing_points
        events = postprocessor.resnap_events(events, timing)
    elif args.osut5.data.add_timing:
        timing = postprocessor.generate_timing(events)
        events = postprocessor.resnap_events(events, timing)

    if args.generate_positions:
        model = find_model(args.diff_ckpt, args, device)
        refine_model = find_model(args.diff_refine_ckpt, args, device) if len(args.diff_refine_ckpt) > 0 else None
        diffusion_pipeline = DiffisionPipeline(args)
        events = diffusion_pipeline.generate(model, events, refine_model)

    postprocessor.generate(events, timing)


if __name__ == "__main__":
    main()
