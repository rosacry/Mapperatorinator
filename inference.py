from os import PathLike
from pathlib import Path

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from slider import Beatmap

import osu_diffusion
import routed_pickle
from diffusion_pipeline import DiffisionPipeline
from osuT5.osuT5.dataset.data_utils import get_song_length, events_of_type, TIMING_TYPES
from osuT5.osuT5.inference import Preprocessor, Processor, Postprocessor, BeatmapConfig, GenerationConfig, \
    generation_config_from_beatmap, beatmap_config_from_beatmap, background_line
from osuT5.osuT5.inference.super_timing_generator import SuperTimingGenerator
from osuT5.osuT5.tokenizer import Tokenizer, ContextType
from osuT5.osuT5.utils import get_model
from osu_diffusion import DiT_models


def prepare_args(args: DictConfig):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')
    set_seed(args.seed)
    if isinstance(args.output_type, str):
        args.output_type = ContextType(args.output_type) if args.output_type != "" else None
    args.in_context = [ContextType(ctx) for ctx in args.in_context]


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
    args.other_beatmap_path = args.beatmap_path

    generation_config = generation_config_from_beatmap(beatmap, tokenizer)

    if args.gamemode == -1:
        args.gamemode = generation_config.gamemode
        print(f"Using game mode {args.gamemode}")
    if args.beatmap_id == -1 and generation_config.beatmap_id:
        args.beatmap_id = generation_config.beatmap_id
        print(f"Using beatmap ID {args.beatmap_id}")
    if args.difficulty == -1 and generation_config.difficulty != -1 and len(beatmap.hit_objects(stacking=False)) > 0:
        args.difficulty = generation_config.difficulty
        print(f"Using difficulty {args.difficulty}")
    if args.mapper_id == -1 and beatmap.beatmap_id in tokenizer.beatmap_mapper:
        args.mapper_id = generation_config.mapper_id
        print(f"Using mapper ID {args.mapper_id}")
    if len(args.descriptors) == 0 and beatmap.beatmap_id in tokenizer.beatmap_descriptors:
        args.descriptors = generation_config.descriptors
        print(f"Using descriptors {args.descriptors}")
    if args.circle_size == -1:
        args.circle_size = generation_config.circle_size
        print(f"Using circle size {args.circle_size}")
    if args.slider_multiplier == -1:
        args.slider_multiplier = generation_config.slider_multiplier
        print(f"Using slider multiplier {args.slider_multiplier}")
    if args.keycount == -1 and args.gamemode == 3:
        args.keycount = int(generation_config.keycount)
        print(f"Using keycount {args.keycount}")
    if args.hold_note_ratio == -1 and args.gamemode == 3:
        args.hold_note_ratio = generation_config.hold_note_ratio
        print(f"Using hold note ratio {args.hold_note_ratio}")
    if args.scroll_speed_ratio == -1 and args.gamemode == 3:
        args.scroll_speed_ratio = generation_config.scroll_speed_ratio
        print(f"Using scroll speed ratio {args.scroll_speed_ratio}")

    beatmap_config = beatmap_config_from_beatmap(beatmap)

    args.title = beatmap_config.title
    args.artist = beatmap_config.artist
    args.bpm = beatmap_config.bpm
    args.offset = beatmap_config.offset
    args.background = beatmap.background
    args.preview_time = beatmap_config.preview_time


def get_config(args: DictConfig):
    # Create tags that describes args
    tags = dict(
        lookback=args.lookback,
        lookahead=args.lookahead,
        beatmap_id=args.beatmap_id,
        difficulty=args.difficulty,
        mapper_id=args.mapper_id,
        year=args.year,
        hitsounded=args.hitsounded,
        hold_note_ratio=args.hold_note_ratio,
        scroll_speed_ratio=args.scroll_speed_ratio,
        descriptors=args.descriptors,
        negative_descriptors=args.negative_descriptors,
        timing_leniency=args.timing_leniency,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        timing_temperature=args.timing_temperature,
        mania_column_temperature=args.mania_column_temperature,
        taiko_hit_temperature=args.taiko_hit_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        parallel=args.parallel,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        super_timing=args.super_timing,
        timer_num_beams=args.timer_num_beams,
        timer_bpm_threshold=args.timer_bpm_threshold,
        generate_positions=args.generate_positions,
        diff_cfg_scale=args.diff_cfg_scale,
        max_seq_len=args.max_seq_len,
        overlap_buffer=args.overlap_buffer,
    )
    # Filter to all non-default values
    defaults = OmegaConf.load("configs/inference.yaml")
    tags = {k: v for k, v in tags.items() if v != defaults[k]}
    # To string separated by spaces
    tags = " ".join(f"{k}={v}" for k, v in tags.items())
    return GenerationConfig(
        gamemode=args.gamemode,
        beatmap_id=args.beatmap_id,
        difficulty=args.difficulty,
        mapper_id=args.mapper_id,
        year=args.year,
        hitsounded=args.hitsounded,
        slider_multiplier=args.slider_multiplier,
        circle_size=args.circle_size,
        keycount=args.keycount,
        hold_note_ratio=args.hold_note_ratio,
        scroll_speed_ratio=args.scroll_speed_ratio,
        descriptors=args.descriptors,
        negative_descriptors=args.negative_descriptors,
    ), BeatmapConfig(
        title=args.title,
        artist=args.artist,
        title_unicode=args.title,
        artist_unicode=args.artist,
        audio_filename=Path(args.audio_path).name,
        circle_size=args.keycount if args.gamemode == 3 else args.circle_size,
        slider_multiplier=args.slider_multiplier,
        creator=args.creator,
        version=args.version,
        tags=tags,
        background_line=background_line(args.background),
        preview_time=args.preview_time,
        bpm=args.bpm,
        offset=args.offset,
        mode=args.gamemode,
    )


def add_timing_to_context(in_context, timing_events, timing_times):
    for ctx in in_context:
        if ctx["context_type"] != ContextType.NONE:
            continue
        ctx["context_type"] = ContextType.TIMING
        ctx["events"] = timing_events
        ctx["event_times"] = timing_times


def generate(
        args: DictConfig,
        *,
        audio_path: PathLike = None,
        other_beatmap_path: PathLike = None,
        generation_config: GenerationConfig,
        beatmap_config: BeatmapConfig,
        model,
        tokenizer,
        diff_model=None,
        diff_tokenizer=None,
        refine_model=None,
        verbose=True,
):
    audio_path = args.audio_path if audio_path is None else audio_path
    other_beatmap_path = args.other_beatmap_path if other_beatmap_path is None else other_beatmap_path

    preprocessor = Preprocessor(args, parallel=args.parallel)
    processor = Processor(args, model, tokenizer, parallel=args.parallel)
    postprocessor = Postprocessor(args)

    audio = preprocessor.load(audio_path)
    in_context = processor.get_in_context(args.in_context, other_beatmap_path, get_song_length(audio, args.osut5.data.sample_rate))
    sequences = preprocessor.segment(audio)

    # Auto generate timing if not provided in in_context and required for the model and this output_type
    timing_events, timing_times, timing = None, None, None
    if args.super_timing and ContextType.NONE in args.in_context:
        super_timing_generator = SuperTimingGenerator(args, model, tokenizer)
        timing_events, timing_times = super_timing_generator.generate(audio, generation_config, verbose=verbose)
        add_timing_to_context(in_context, timing_events, timing_times)
        timing = postprocessor.generate_timing(timing_events)
    elif (args.output_type == ContextType.TIMING or
          (ContextType.NONE in args.in_context and args.output_type == ContextType.MAP and
           not any("none" in ctx["in"] and ctx["out"] == "map" for ctx in args.osut5.data.context_types))):
        # Generate timing and convert in_context to timing context
        timing_events, timing_times = processor.generate(
            sequences=sequences,
            generation_config=generation_config,
            in_context=in_context,
            verbose=verbose,
        )
        timing_events, timing_times = events_of_type(timing_events, timing_times, TIMING_TYPES)
        add_timing_to_context(in_context, timing_events, timing_times)
        timing = postprocessor.generate_timing(timing_events)
    elif ContextType.TIMING in args.in_context or (
            args.osut5.data.add_timing and any(t in args.in_context for t in [ContextType.GD, ContextType.NO_HS])):
        # Exact timing is provided in the other beatmap, so we don't need to generate it
        timing = [tp for tp in Beatmap.from_path(other_beatmap_path).timing_points if tp.parent is None]

    # Generate beatmap
    if args.output_type == ContextType.MAP:
        events, _ = processor.generate(
            sequences=sequences,
            generation_config=generation_config,
            in_context=in_context,
            verbose=verbose,
        )

        # Resnap timing events
        if timing is not None:
            events = postprocessor.resnap_events(events, timing)
    else:
        events = timing_events

    # Generate positions with diffusion
    if args.generate_positions and args.gamemode in [0, 2] and args.output_type == ContextType.MAP:
        diffusion_pipeline = DiffisionPipeline(args, diff_model, diff_tokenizer, refine_model)
        events = diffusion_pipeline.generate(
            events=events,
            generation_config=generation_config,
            verbose=verbose,
        )

    result = postprocessor.generate(
        events=events,
        beatmap_config=beatmap_config,
        timing=timing,
    )

    if args.output_path is not None and args.output_path != "":
        if verbose:
            print(f"Generated beatmap saved to {args.output_path}")
        postprocessor.write_result(result, args.output_path)

    return result


def load_model(
        ckpt_path: PathLike,
        t5_args: DictConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(ckpt_path)
    model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device, weights_only=True)
    tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=routed_pickle, weights_only=False)

    tokenizer = Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    model = get_model(t5_args, tokenizer)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)

    return model, tokenizer


def load_diff_model(ckpt_path, diff_args: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Could not find DiT checkpoint at {ckpt_path}"

    tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_1.pkl", pickle_module=routed_pickle, weights_only=False)
    tokenizer = osu_diffusion.utils.tokenizer.Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    ema_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=routed_pickle, weights_only=False)
    model = DiT_models[diff_args.model.model](
        context_size=diff_args.model.context_size,
        class_size=tokenizer.num_tokens,
    ).to(device)
    model.load_state_dict(ema_state)
    model.eval()  # important!
    return model, tokenizer


@hydra.main(config_path="configs", config_name="inference_v28", version_base="1.1")
def main(args: DictConfig):
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\584787 Yuiko Ohara - Hoshi o Tadoreba\\Yuiko Ohara - Hoshi o Tadoreba (Yumeno Himiko) [015's Hard].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\859916 DJ Noriken (Remixed _ Covered by Camellia) - Jingle (Metal Arrange _ Cover)\\DJ Noriken (Remixed  Covered by Camellia) - Jingle (Metal Arrange  Cover) (StunterLetsPlay) [Extra].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\989342 Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis\\Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\qyoh for upload\\Camellia - Qyoh (Nine Stars) (OliBomby) [Yoalteuctin (Rabbit Hole Collab)].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\1903968 Kisumi Reika - Sekai wa Futari no Tame ni\\Kisumi Reika - Sekai wa Futari no Tame ni (Ayesha Altugle) [Normal].osu"

    prepare_args(args)

    model, tokenizer = load_model(args.model_path, args.osut5)

    diff_model, diff_tokenizer, refine_model = None, None, None
    if args.generate_positions:
        diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion)

        if len(args.diff_refine_ckpt) > 0:
            refine_model = load_diff_model(args.diff_refine_ckpt, args.diffusion)[0]

        if args.compile:
            diff_model.forward = torch.compile(diff_model.forward, mode="reduce-overhead", fullgraph=True)

    get_args_from_beatmap(args, tokenizer)
    generation_config, beatmap_config = get_config(args)

    generate(
        args,
        generation_config=generation_config,
        beatmap_config=beatmap_config,
        model=model,
        tokenizer=tokenizer,
        diff_model=diff_model,
        diff_tokenizer=diff_tokenizer,
        refine_model=refine_model,
    )


if __name__ == "__main__":
    main()
