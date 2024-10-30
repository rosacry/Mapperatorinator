import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from scipy import linalg
from slider import Beatmap
from tqdm import tqdm

from classifier.classify import iterate_examples
from classifier.libs.model.model import OsuClassifierOutput
from classifier.libs.utils import load_ckpt
from inference import prepare_args, load_diff_model, generate, load_model
from osuT5.osuT5.dataset.data_utils import load_audio_file
from osuT5.osuT5.inference import generation_config_from_beatmap, beatmap_config_from_beatmap
from multiprocessing import Manager, Process


def get_beatmap_paths(args) -> list[Path]:
    beatmap_files = []
    track_names = ["Track" + str(i).zfill(5) for i in range(args.dataset_start, args.dataset_end)]
    for track_name in track_names:
        for beatmap_file in os.listdir(
                os.path.join(args.dataset_path, track_name, "beatmaps"),
        ):
            beatmap_files.append(
                Path(
                    os.path.join(
                        args.dataset_path,
                        track_name,
                        "beatmaps",
                        beatmap_file,
                    )
                ),
            )

    return beatmap_files


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def worker(beatmap_paths, args, return_dict, idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_args(args)

    model, tokenizer = load_model(args.model_path, args.osut5)

    if args.compile:
        model.transformer.forward = torch.compile(model.transformer.forward, mode="reduce-overhead", fullgraph=True)

    diff_model, diff_tokenizer, refine_model = None, None, None
    if args.generate_positions:
        diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion)

        if len(args.diff_refine_ckpt) > 0:
            refine_model = load_diff_model(args.diff_refine_ckpt, args.diffusion)[0]

        # Compiling is disabled for the diffision model because it uses various input sizes and that slows down the process with recompilations
        # if args.compile:
        #     diff_model.forward = torch.compile(diff_model.forward, mode="reduce-overhead", fullgraph=False)

    classifier_model, classifier_args, classifier_tokenizer = load_ckpt(args.classifier_ckpt)

    if args.compile:
        classifier_model.model.transformer.forward = torch.compile(classifier_model.model.transformer.forward, mode="reduce-overhead", fullgraph=False)

    real_features = []
    generated_features = []

    for beatmap_path in tqdm(beatmap_paths):
        audio_path = beatmap_path.parents[1] / list(beatmap_path.parents[1].glob('audio.*'))[0]
        beatmap = Beatmap.from_path(beatmap_path)

        generation_config = generation_config_from_beatmap(beatmap, tokenizer)
        beatmap_config = beatmap_config_from_beatmap(beatmap)

        result = generate(
            args,
            audio_path=audio_path,
            other_beatmap_path=beatmap_path,
            generation_config=generation_config,
            beatmap_config=beatmap_config,
            model=model,
            tokenizer=tokenizer,
            diff_model=diff_model,
            diff_tokenizer=diff_tokenizer,
            refine_model=refine_model,
            verbose=False,
        )
        generated_beatmap = Beatmap.parse(result)

        # Calculate feature vectors for real and generated beatmaps
        sample_rate = classifier_args.data.sample_rate
        audio = load_audio_file(audio_path, sample_rate)

        for example in iterate_examples(beatmap, audio, classifier_args, classifier_tokenizer, device):
            classifier_result: OsuClassifierOutput = classifier_model(**example)
            features = classifier_result.feature_vector
            real_features.append(features.squeeze(0).cpu().numpy())

        for example in iterate_examples(generated_beatmap, audio, classifier_args, classifier_tokenizer, device):
            classifier_result: OsuClassifierOutput = classifier_model(**example)
            features = classifier_result.feature_vector
            generated_features.append(features.squeeze(0).cpu().numpy())

        torch.cuda.empty_cache()  # Clear any cached memory

    return_dict[idx] = (real_features, generated_features)


@hydra.main(config_path="configs", config_name="inference_v1", version_base="1.1")
def main(args: DictConfig):
    beatmap_paths = get_beatmap_paths(args)
    num_processes = 4
    chunk_size = len(beatmap_paths) // num_processes
    chunks = [beatmap_paths[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    # TODO Add support for uneven chunks
    manager = Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(num_processes):
        p = Process(target=worker, args=(chunks[i], args, return_dict, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    real_features = []
    generated_features = []
    for i in range(num_processes):
        real_features.extend(return_dict[i][0])
        generated_features.extend(return_dict[i][1])

    # Calculate FID
    real_features = np.stack(real_features)
    generated_features = np.stack(generated_features)
    m1, s1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    m2, s2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    fid = calculate_frechet_distance(m1, s1, m2, s2)

    print(f"FID: {fid}")


if __name__ == "__main__":
    main()
