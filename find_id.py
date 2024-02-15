from omegaconf import DictConfig
import hydra

import hydra
from omegaconf import DictConfig
from slider import Beatmap

from osuT5.dataset import OrsDataset, OsuParser
from osuT5.utils import (
    setup_args,
    get_tokenizer,
)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(args: DictConfig):
    setup_args(args)

    tokenizer = get_tokenizer()
    parser = OsuParser()
    dataset = OrsDataset(
            args.train_dataset_path,
            args.train_dataset_start,
            args.train_dataset_end,
            args.model.spectrogram.sample_rate,
            args.model.spectrogram.hop_length,
            args.model.max_seq_len,
            args.model.max_target_len,
            parser,
            tokenizer,
            args.optim.cycle_length,
            True,
        )

    beatmaps = dataset._get_beatmap_files()
    for beatmap_path in beatmaps:
        beatmap = Beatmap.from_path(beatmap_path)
        if beatmap.beatmap_id == 1473252:
            print(beatmap_path)


if __name__ == "__main__":
    main()
