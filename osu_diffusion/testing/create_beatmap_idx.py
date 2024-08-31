import json
import os
import pickle

beatmap_idx = {}
dataset_path = r"C:\Users\Olivier\Documents\Collections\Beatmap ML Datasets\ORS16291"
idx = 0

for i in range(0, 16291):
    track_name = "Track" + str(i).zfill(5)
    metadata_File = os.path.join(dataset_path, track_name, "metadata.json")
    with open(metadata_File) as f:
        metadata = json.load(f)
    for j in metadata["Beatmaps"]:
        beatmap_metadata = metadata["Beatmaps"][j]
        beatmap_idx[beatmap_metadata["BeatmapId"]] = beatmap_metadata["Index"]
        idx += 1
        print(f"\r{idx}", end="")

with open("../beatmap_idx.pickle", "wb") as f:
    pickle.dump(beatmap_idx, f)
