from pathlib import Path

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import DictConfig
from slider import Beatmap

import osu_diffusion
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
    set_seed(args.seed)


def get_args_from_beatmap(args: DictConfig, tokenizer: Tokenizer):
    if args.beatmap_path is None or args.beatmap_path == "":
        return

    beatmap_path = Path(args.beatmap_path)

    if not beatmap_path.is_file():
        raise FileNotFoundError(f"Beatmap file {beatmap_path} not found.")

    beatmap = Beatmap.from_path(beatmap_path)
    print(f"Using metadata from beatmap: {beatmap.display_name}")
    args.audio_path = beatmap_path.parent / beatmap.audio_filename
    args.output_path = beatmap_path.parent
    args.bpm = beatmap.bpm_max()
    args.offset = min(tp.offset.total_seconds() * 1000 for tp in beatmap.timing_points)
    args.slider_multiplier = beatmap.slider_multiplier
    args.title = beatmap.title
    args.artist = beatmap.artist
    if args.beatmap_id == -1 and (args.osut5.data.style_token_index >= 0 or args.diffusion.data.beatmap_class):
        args.beatmap_id = beatmap.beatmap_id
        print(f"Using beatmap ID {args.beatmap_id}")
    if args.difficulty == -1 and (args.osut5.data.diff_token_index >= 0 or args.diffusion.data.difficulty_class):
        args.difficulty = float(beatmap.stars())
        print(f"Using difficulty {args.difficulty}")
    if args.mapper_id == -1 and beatmap.beatmap_id in tokenizer.mapper_idx:
        args.mapper_id = tokenizer.mapper_idx[beatmap.beatmap_id]
        print(f"Using mapper ID {args.mapper_id}")
    if len(args.descriptors) == 0 and beatmap.beatmap_id in tokenizer.beatmap_descriptors:
        args.descriptors = tokenizer.beatmap_descriptors[beatmap.beatmap_id]
        print(f"Using descriptors {args.descriptors}")
    if args.circle_size == -1 and (args.osut5.data.cs_token_index >= 0 or args.diffusion.data.circle_size_class):
        args.circle_size = beatmap.circle_size
        print(f"Using circle size {args.circle_size}")
    args.other_beatmap_path = args.beatmap_path


def find_model(ckpt_path, args: DictConfig, device):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Could not find DiT checkpoint at {ckpt_path}"

    tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_1.pkl", pickle_module=routed_pickle, weights_only=False)
    tokenizer = osu_diffusion.utils.tokenizer.Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    ema_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=routed_pickle, weights_only=False)
    model = DiT_models[args.diffusion.model.model](
        context_size=args.diffusion.model.context_size,
        class_size=tokenizer.num_tokens,
    ).to(device)
    model.load_state_dict(ema_state)
    model.eval()  # important!
    return model, tokenizer


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
    model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device, weights_only=True)
    tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=routed_pickle, weights_only=False)

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
        circle_size=args.circle_size,
        other_beatmap_path=args.other_beatmap_path,
        context_type=args.context_type,
        negative_descriptors=args.negative_descriptors,
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
        model, diff_tokenizer = find_model(args.diff_ckpt, args, device)
        refine_model = find_model(args.diff_refine_ckpt, args, device)[0] if len(args.diff_refine_ckpt) > 0 else None

        if args.compile:
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

        diffusion_pipeline = DiffisionPipeline(args, diff_tokenizer)
        events = diffusion_pipeline.generate(
            model=model,
            events=events,
            beatmap_id=args.beatmap_id,
            difficulty=args.difficulty,
            mapper_id=args.mapper_id,
            descriptors=args.descriptors,
            circle_size=args.circle_size,
            refine_model=refine_model,
            negative_descriptors=args.negative_descriptors,
        )

    postprocessor.generate(events, timing)


if __name__ == "__main__":
    main()
