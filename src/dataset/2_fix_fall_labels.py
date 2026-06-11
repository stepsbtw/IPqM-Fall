import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import src.config as config

TRIALS_ROOT = config.DATASET_ROOT / "raw"

FALL_LABELS = {
    "FRONTAL-FALL-SUPINE","FRONTAL-FALL-PRONE","BACKWARD-FALL-SUPINE",
    "LATERAL-FALL-RIGHT","LATERAL-FALL-LEFT",
    "FRONTAL-FALL-SUPINE-RIFLE","FRONTAL-FALL-PRONE-RIFLE",
    "BACKWARD-FALL-SUPINE-RIFLE","LATERAL-FALL-RIGHT-RIFLE",
    "LATERAL-FALL-LEFT-RIFLE"
}

PRE_FALL_LABEL = "STANDING"

FALL_CATEGORY_MAPPING = {
    "FRONTAL-FALL-SUPINE": "FRONTAL-FALL",
    "FRONTAL-FALL-PRONE": "FRONTAL-FALL",
    "FRONTAL-FALL-SUPINE-RIFLE": "FRONTAL-FALL",
    "FRONTAL-FALL-PRONE-RIFLE": "FRONTAL-FALL",

    "BACKWARD-FALL-SUPINE": "BACKWARD-FALL",
    "BACKWARD-FALL-SUPINE-RIFLE": "BACKWARD-FALL",

    "LATERAL-FALL-RIGHT": "LATERAL-FALL-RIGHT",
    "LATERAL-FALL-RIGHT-RIFLE": "LATERAL-FALL-RIGHT",

    "LATERAL-FALL-LEFT": "LATERAL-FALL-LEFT",
    "LATERAL-FALL-LEFT-RIFLE": "LATERAL-FALL-LEFT",
}

COARSE_FALL_LABELS = set(FALL_CATEGORY_MAPPING.values())

POST_FALL_MAPPING = {
    "FRONTAL-FALL-SUPINE": "DOWN-SUPINE",
    "FRONTAL-FALL-PRONE": "DOWN-PRONE",
    "BACKWARD-FALL-SUPINE": "DOWN-SUPINE",
    "LATERAL-FALL-RIGHT": "DOWN-RIGHT",
    "LATERAL-FALL-LEFT": "DOWN-LEFT",

    "FRONTAL-FALL-SUPINE-RIFLE": "DOWN-SUPINE",
    "FRONTAL-FALL-PRONE-RIFLE": "DOWN-PRONE",
    "BACKWARD-FALL-SUPINE-RIFLE": "DOWN-SUPINE",
    "LATERAL-FALL-RIGHT-RIFLE": "DOWN-RIGHT",
    "LATERAL-FALL-LEFT-RIFLE": "DOWN-LEFT",
}

meta = pd.read_parquet(config.WINDOWS_FILE)

def load_trial(path):
    return pd.read_parquet(path).reset_index(drop=True)

def compute_combined_signal(df):
    acc_mag = df["amag"].values
    gyro_mag = df["wmag"].values

    signal = acc_mag + gyro_mag

    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

updated_trials = []
trial_files = sorted(meta["file"].unique())

for trial_file in tqdm(trial_files, desc="Processing trials"):

    trial_meta = meta[meta["file"] == trial_file].copy()

    has_fall = trial_meta["label"].isin(FALL_LABELS).any()

    if not has_fall:
        updated_trials.append(trial_meta)
        continue

    trial_path = TRIALS_ROOT / trial_file

    try:
        trial_df = load_trial(trial_path)
    except Exception as e:
        print(f"ERROR loading {trial_file}: {e}")
        updated_trials.append(trial_meta)
        continue

    try:
        signal = compute_combined_signal(trial_df)
    except Exception as e:
        print(f"ERROR computing signal in {trial_file}: {e}")
        updated_trials.append(trial_meta)
        continue

    if len(signal) == 0 or np.all(signal == 0):
        updated_trials.append(trial_meta)
        continue

    peak_idx = int(np.argmax(signal))

    peak_start = max(0, int(peak_idx - config.PRE_FALL_SECONDS * config.FS))
    peak_end = min(len(signal), int(peak_idx + config.POST_FALL_SECONDS * config.FS))

    trial_fall_labels = trial_meta[
        trial_meta["label"].isin(FALL_LABELS)
    ]["label"].unique()

    if len(trial_fall_labels) == 0:
        updated_trials.append(trial_meta)
        continue

    fall_type = trial_fall_labels[0]
    post_fall_label = POST_FALL_MAPPING.get(fall_type, "UNKNOWN")

    for idx, row in trial_meta.iterrows():

        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        original_label = row["label"]

        if original_label in FALL_LABELS:

            overlap = not (end_idx < peak_start or start_idx > peak_end)

            if overlap:
                trial_meta.loc[idx, "label"] = FALL_CATEGORY_MAPPING.get(
                    original_label,
                    "UNKNOWN_FALL"
                )

            elif end_idx < peak_start:
                trial_meta.loc[idx, "label"] = PRE_FALL_LABEL

            elif start_idx > peak_end:
                trial_meta.loc[idx, "label"] = post_fall_label

            else:
                trial_meta.loc[idx, "label"] = "UNKNOWN"

            trial_meta.loc[idx, "reviewed"] = True

    updated_trials.append(trial_meta)

final_meta = pd.concat(updated_trials, ignore_index=True)

final_meta = final_meta.sort_values(
    by=["subject_id", "sensor_pos", "file", "start_idx"]
)

final_meta.to_parquet(config.WINDOWS_FILE, index=False)

before_fall = meta["label"].isin(FALL_LABELS).sum()

coarse_fall_count = final_meta["label"].isin(COARSE_FALL_LABELS).sum()

standing_count = (final_meta["label"] == "STANDING").sum()
supine_count = (final_meta["label"] == "DOWN-SUPINE").sum()
prone_count = (final_meta["label"] == "DOWN-PRONE").sum()
left_count = (final_meta["label"] == "DOWN-LEFT").sum()
right_count = (final_meta["label"] == "DOWN-RIGHT").sum()
unknown_count = (final_meta["label"] == "UNKNOWN").sum()

print("\n==================== SUMMARY ====================")
print(f"Original fall windows : {before_fall}")
print(f"Coarse fall windows   : {coarse_fall_count}")
print(f"Standing windows      : {standing_count}")
print(f"Supine windows        : {supine_count}")
print(f"Prone windows         : {prone_count}")
print(f"Lateral-left windows  : {left_count}")
print(f"Lateral-right windows : {right_count}")
print(f"Unknown windows       : {unknown_count}")
print("=================================================\n")

print(f"Saved refined metadata to:\n{config.WINDOWS_FILE}")