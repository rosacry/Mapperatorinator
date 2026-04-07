"""
Build an MMRS-format training dataset from downloaded .osz files.

This script handles the parts that DON'T require the Mapperator .NET tool:
- Extracts .osz files (they're just .zip archives)
- Builds the metadata.parquet from .osu file headers + osu! API data
- Organizes files into MMRS directory structure

For the full pipeline with Mapperator, use the Mapperator.ConsoleApp.exe dataset2 command.
This script is an alternative if you can't run .NET on your system.

Usage:
  python training/build_dataset.py --input ./training/osz_2024_2026 --output ./training/dataset_2024_2026

Requirements:
  pip install requests tqdm pandas pyarrow
"""

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_osu_metadata(osu_path: Path) -> dict:
    """Parse metadata from a .osu file header."""
    metadata = {}
    section = None

    try:
        with open(osu_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()

                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1]
                    continue

                if section in ("General", "Metadata", "Difficulty") and ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()

                # Stop after we have what we need
                if section == "HitObjects":
                    break
    except (UnicodeDecodeError, OSError):
        pass

    return metadata


def extract_osz(osz_path: Path, output_dir: Path) -> list[dict]:
    """Extract an .osz file and return metadata for each difficulty."""
    beatmapset_id = osz_path.stem
    set_dir = output_dir / "data" / str(beatmapset_id)

    if set_dir.exists():
        # Already extracted
        return []

    set_dir.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        with zipfile.ZipFile(osz_path, "r") as z:
            # Only extract .osu files and audio — skip images, videos, storyboards
            osu_files = [n for n in z.namelist() if n.endswith(".osu")]
            audio_files = [n for n in z.namelist()
                           if n.lower().endswith((".mp3", ".ogg", ".wav"))]

            # Extract audio (just the first one found)
            if audio_files:
                z.extract(audio_files[0], set_dir)

            # Extract .osu files
            for name in osu_files:
                z.extract(name, set_dir)

            # Parse each .osu file
            for name in osu_files:
                    osu_path = set_dir / name
                    meta = parse_osu_metadata(osu_path)

                    beatmap_id = meta.get("BeatmapID", "0")
                    mode = int(meta.get("Mode", "0"))
                    difficulty_name = meta.get("Version", "Unknown")

                    results.append({
                        "BeatmapSetId": int(beatmapset_id),
                        "Id": int(beatmap_id),
                        "BeatmapFile": name,
                        "ModeInt": mode,
                        "DifficultyName": difficulty_name,
                    })

    except (zipfile.BadZipFile, OSError) as e:
        print(f"  Failed to extract {osz_path}: {e}")
        if set_dir.exists():
            shutil.rmtree(set_dir, ignore_errors=True)
        return []

    return results


def enrich_with_api_metadata(records: list[dict], api_metadata_path: Path) -> list[dict]:
    """Merge extracted .osu metadata with API metadata (ranked date, star rating)."""
    if not api_metadata_path.exists():
        print("Warning: No API metadata file found. Star ratings and ranked dates will be estimated.")
        return records

    with open(api_metadata_path, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    # Build lookup: beatmapset_id -> {beatmap_id -> beatmap_data}
    api_lookup = {}
    for bset in api_data:
        bset_id = bset["id"]
        api_lookup[bset_id] = {
            "ranked_date": bset.get("ranked_date") or bset.get("submitted_date"),
            "beatmaps": {}
        }
        for bm in bset.get("beatmaps", []):
            api_lookup[bset_id]["beatmaps"][bm["id"]] = {
                "difficulty_rating": bm.get("difficulty_rating", 0),
                "mode_int": bm.get("mode_int", 0),
            }

    enriched = []
    for rec in records:
        bset_id = rec["BeatmapSetId"]
        bm_id = rec["Id"]

        api_bset = api_lookup.get(bset_id, {})
        api_bm = api_bset.get("beatmaps", {}).get(bm_id, {})

        # Parse ranked date
        ranked_str = api_bset.get("ranked_date")
        if ranked_str:
            try:
                ranked_date = datetime.fromisoformat(ranked_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ranked_date = datetime(2024, 1, 1)
        else:
            ranked_date = datetime(2024, 1, 1)

        rec["DifficultyRating"] = api_bm.get("difficulty_rating", 0.0)
        rec["RankedDate"] = ranked_date
        rec["ModeInt"] = api_bm.get("mode_int", rec.get("ModeInt", 0))
        enriched.append(rec)

    return enriched


def build_metadata_parquet(records: list[dict], output_dir: Path):
    """Build the metadata.parquet file in MMRS format."""
    if not records:
        print("No records to write!")
        return

    df = pd.DataFrame(records)

    # Add BeatmapIdx (sequential index)
    df["BeatmapIdx"] = range(len(df))

    # Ensure correct types
    df["BeatmapSetId"] = df["BeatmapSetId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["ModeInt"] = df["ModeInt"].astype(int)
    df["DifficultyRating"] = df["DifficultyRating"].astype(float)

    # Sort by BeatmapSetId
    df = df.sort_values("BeatmapSetId")

    # Set multi-index as expected by data_utils.py
    df = df.set_index(["BeatmapSetId", "Id"])

    parquet_path = output_dir / "metadata.parquet"
    df.to_parquet(parquet_path)

    # Print stats
    n_sets = df.index.get_level_values(0).nunique()
    n_maps = len(df)
    print(f"\nDataset built:")
    print(f"  Beatmap sets: {n_sets}")
    print(f"  Total difficulties: {n_maps}")
    print(f"  Output: {parquet_path}")

    # Year distribution
    if "RankedDate" in df.columns:
        year_counts = df["RankedDate"].dt.year.value_counts().sort_index()
        print(f"\n  Year distribution:")
        for year, count in year_counts.items():
            print(f"    {year}: {count} beatmaps")

    # Gamemode distribution
    mode_names = {0: "Standard", 1: "Taiko", 2: "Catch", 3: "Mania"}
    mode_counts = df["ModeInt"].value_counts().sort_index()
    print(f"\n  Gamemode distribution:")
    for mode, count in mode_counts.items():
        print(f"    {mode_names.get(mode, f'Mode {mode}')}: {count}")

    return n_sets


def main():
    parser = argparse.ArgumentParser(description="Build MMRS dataset from .osz files")
    parser.add_argument("--input", required=True,
                        help="Directory containing .osz files (from download_beatmaps.py)")
    parser.add_argument("--output", required=True,
                        help="Output directory for MMRS dataset")
    parser.add_argument("--delete-osz", action="store_true",
                        help="Delete each .osz file after successful extraction to save disk space")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    osz_files = sorted(input_dir.glob("*.osz"))
    if not osz_files:
        print(f"No .osz files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(osz_files)} .osz files")

    # Extract all .osz files
    all_records = []
    for osz_path in tqdm(osz_files, desc="Extracting .osz files"):
        records = extract_osz(osz_path, output_dir)
        all_records.extend(records)
        # Delete .osz after successful extraction to free disk space
        if args.delete_osz and records:
            osz_path.unlink()

    # Also check already-extracted directories for records
    if not all_records:
        print("Checking already-extracted directories...")
        data_dir = output_dir / "data"
        if data_dir.exists():
            for set_dir in sorted(data_dir.iterdir()):
                if set_dir.is_dir():
                    for osu_file in set_dir.glob("*.osu"):
                        meta = parse_osu_metadata(osu_file)
                        all_records.append({
                            "BeatmapSetId": int(set_dir.name),
                            "Id": int(meta.get("BeatmapID", "0")),
                            "BeatmapFile": osu_file.name,
                            "ModeInt": int(meta.get("Mode", "0")),
                            "DifficultyName": meta.get("Version", "Unknown"),
                        })

    print(f"\nTotal beatmap difficulties found: {len(all_records)}")

    # Enrich with API metadata
    api_metadata_path = input_dir / "beatmapset_metadata.json"
    all_records = enrich_with_api_metadata(all_records, api_metadata_path)

    # Build parquet
    n_sets = build_metadata_parquet(all_records, output_dir)

    if n_sets:
        train_end = int(n_sets * 0.9)
        print(f"\n--- Training Config Values ---")
        print(f"  train_dataset_path: \"{output_dir.resolve()}\"")
        print(f"  train_dataset_start: 0")
        print(f"  train_dataset_end: {train_end}")
        print(f"  test_dataset_start: {train_end}")
        print(f"  test_dataset_end: {n_sets}")
        print(f"\nUpdate configs/train/lora_2024_2026.yaml with these values.")


if __name__ == "__main__":
    main()
