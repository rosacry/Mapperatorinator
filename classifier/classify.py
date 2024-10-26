import json
from pathlib import Path
import numpy.typing as npt

import hydra
import torch
from omegaconf import DictConfig
from slider import Beatmap

from classifier.libs.dataset import OsuParser
from classifier.libs.dataset.data_utils import load_audio_file
from classifier.libs.dataset.ors_dataset import STEPS_PER_MILLISECOND
from classifier.libs.model.model import OsuClassifierOutput
from classifier.libs.tokenizer import Tokenizer, Event, EventType
from classifier.libs.utils import load_ckpt


def iterate_examples(
        beatmap: Beatmap,
        audio: npt.NDArray,
        model_args: DictConfig,
        tokenizer: Tokenizer,
        device: torch.device
):
    frame_seq_len = model_args.data.src_seq_len - 1
    frame_size = model_args.data.hop_length
    sample_rate = model_args.data.sample_rate
    samples_per_sequence = frame_seq_len * frame_size

    for sample in range(0, len(audio) - samples_per_sequence, samples_per_sequence):
        example = create_example(beatmap, audio, sample / sample_rate, model_args, tokenizer, device)
        yield example


def create_example(
        beatmap: Beatmap,
        audio: npt.NDArray,
        time: float,
        model_args: DictConfig,
        tokenizer: Tokenizer,
        device: torch.device
):
    frame_seq_len = model_args.data.src_seq_len - 1
    frame_size = model_args.data.hop_length
    sample_rate = model_args.data.sample_rate
    samples_per_sequence = frame_seq_len * frame_size
    sequence_duration = samples_per_sequence / sample_rate

    # Get audio frames
    frame_start = int(time * sample_rate)
    frames = audio[frame_start:frame_start + samples_per_sequence]
    frames = torch.from_numpy(frames).to(torch.float32).unsqueeze(0).to(device)

    parser = OsuParser(model_args, tokenizer)
    events, event_times = parser.parse(beatmap)
    # Get the events between time and time + sequence_duration
    events = [event for event, event_time in zip(events, event_times) if
              time <= event_time / 1000 < time + sequence_duration]
    # Normalize time shifts
    for i, event in enumerate(events):
        if event.type == EventType.TIME_SHIFT:
            events[i] = Event(EventType.TIME_SHIFT, int((event.value - time * 1000) * STEPS_PER_MILLISECOND))

    # Tokenize the events
    tokens = torch.full((model_args.data.tgt_seq_len,), tokenizer.pad_id, dtype=torch.long)
    for i, event in enumerate(events):
        tokens[i] = tokenizer.encode(event)
    tokens = tokens.unsqueeze(0).to(device)

    return {
        "decoder_input_ids": tokens,
        "decoder_attention_mask": tokens != tokenizer.pad_id,
        "frames": frames,
    }


def create_example_from_path(
        beatmap_path: str,
        time: float,
        model_args: DictConfig,
        tokenizer: Tokenizer,
        device: torch.device
):
    sample_rate = model_args.data.sample_rate

    beatmap_path = Path(beatmap_path)
    beatmap = Beatmap.from_path(beatmap_path)

    # Get audio frames
    audio_path = beatmap_path.parent / beatmap.audio_filename
    audio = load_audio_file(audio_path, sample_rate)

    return create_example(beatmap, audio, time, model_args, tokenizer, device)


def get_mapper_names():
    path = Path(r"C:\Users\Olivier\Documents\GitHub\Mapperatorinator\datasets\beatmap_users.json")

    # Load JSON data from file
    with open(path, 'r') as file:
        data = json.load(file)

    # Populate beatmap_mapper
    mapper_names = {}
    for item in data:
        if len(item['username']) == 0:
            mapper_name = "Unknown"
        else:
            mapper_name = item['username'][0]
        mapper_names[item['user_id']] = mapper_name

    return mapper_names


@hydra.main(config_path="configs", config_name="inference", version_base="1.1")
def main(args: DictConfig):
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\584787 Yuiko Ohara - Hoshi o Tadoreba\\Yuiko Ohara - Hoshi o Tadoreba (Yumeno Himiko) [015's Hard].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\859916 DJ Noriken (Remixed _ Covered by Camellia) - Jingle (Metal Arrange _ Cover)\\DJ Noriken (Remixed  Covered by Camellia) - Jingle (Metal Arrange  Cover) (StunterLetsPlay) [Extra].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\989342 Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis\\Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\Kou_ - RE_generate_fractal (OliBomby)\\Kou! - RE_generatefractal (OliBomby) [I love lazer hexgrid].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\the answer\\MIMI - Answer (feat. Wanko) (OliBomby) [AI's Insane].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\518426 Bernd Krueger - Sonata No14 in cis-Moll, Op 27-2 - 3 Satz\\Bernd Krueger - Sonata No.14 in cis-Moll, Op. 272 - 3. Satz (Fenza) [Presto Agitato].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\2036508 Sydosys - Lunar Gateway\\Sydosys - Lunar Gateway (Gamelan4) [Hivie's Oni].osu"
    args.beatmap_path = r"C:\Users\Olivier\AppData\Local\osu!\Songs\1790119 THE ORAL CIGARETTES - ReI\THE ORAL CIGARETTES - ReI (Sotarks) [Cataclysm.].osu"
    # args.beatmap_path = r"C:\Users\Olivier\AppData\Local\osu!\Songs\1790119 THE ORAL CIGARETTES - ReI\THE ORAL CIGARETTES - ReI (OliBomby) [osuT5 V21 timing cheri cfg 4].osu"
    # args.beatmap_path = r"C:\Users\Olivier\AppData\Local\osu!\Songs\1790119 THE ORAL CIGARETTES - ReI\THE ORAL CIGARETTES - ReI (Sotarks) [banter's Insane].osu"
    # args.beatmap_path = r"C:\Users\Olivier\AppData\Local\osu!\Songs\1790119 THE ORAL CIGARETTES - ReI\THE ORAL CIGARETTES - ReI (Sotarks) [Pepekcz's Hard].osu"
    # args.beatmap_path = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs\\634147 Kaneko Chiharu - iLLness LiLin\\Kaneko Chiharu - iLLness LiLin (Kroytz) [TERMiNALLY iLL].osu"
    # args.beatmap_path = r"C:\Users\Olivier\AppData\Local\osu!\Songs\glass beach - neon glow\glass beach - neon glow (OliBomby) [e].osu"
    args.checkpoint_path = r"C:\Users\Olivier\Documents\GitHub\Mapperatorinator\test\classifier_v3\model.ckpt"
    args.time = 60

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_args, tokenizer = load_ckpt(args.checkpoint_path)
    model.eval().to(device)

    example = create_example_from_path(args.beatmap_path, args.time, model_args, tokenizer, device)
    result: OsuClassifierOutput = model(**example)
    logits = result.logits

    # Print the top 100 mappers with confidences
    top_k = 100
    top_k_indices = logits[0].topk(top_k).indices
    top_k_confidences = logits[0].topk(top_k).values

    mapper_idx_id = {idx: ids for ids, idx in tokenizer.mapper_idx.items()}
    mapper_names = get_mapper_names()

    for idx, confidence in zip(top_k_indices, top_k_confidences):
        mapper_id = mapper_idx_id[idx.item()]
        mapper_name = mapper_names.get(mapper_id, "Unknown")
        print(f"Mapper: {mapper_name} ({mapper_id}) with confidence: {confidence.item()}")


if __name__ == "__main__":
    main()
