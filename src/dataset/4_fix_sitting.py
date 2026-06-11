import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import src.config as config

TRIALS_ROOT = Path("IPqM-Fall/raw")

ADL_TRANSITIONS = {
    # Stand-to-Sit
    "ADL5":  {"pre": "STANDING",       "trans": "STANDING-SITTING",       "post": "SITTING",       "duration": 2.0},
    "ADL5R": {"pre": "STANDING-RIFLE", "trans": "STANDING-SITTING-RIFLE", "post": "SITTING-RIFLE", "duration": 2.0},
    
    # Sit-to-Stand
    "ADL6":  {"pre": "SITTING",       "trans": "SITTING-STANDING",       "post": "STANDING",       "duration": 2.0},
    "ADL6R": {"pre": "SITTING-RIFLE", "trans": "SITTING-STANDING-RIFLE", "post": "STANDING-RIFLE", "duration": 2.0},
}

def load_trial(path):
    return pd.read_parquet(path).reset_index(drop=True)

def find_settle_point(df):
    wmag = df["wmag"].values

    kernel = np.ones(config.TRANSITION_SMOOTHING_WINDOW) / config.TRANSITION_SMOOTHING_WINDOW
    smooth = np.convolve(wmag, kernel, mode="same")

    above = np.where(smooth > config.TRANSITION_STATIC_THRESHOLD)[0]

    if len(above) == 0:
        return len(df) - 1

    return int(above[-1])

meta = pd.read_parquet(config.WINDOWS_FILE)
updated_trials = []

trial_files = sorted(meta["file"].unique())

for trial_file in tqdm(trial_files, desc="Processing Chair Transitions"):

    trial_meta = meta[meta["file"] == trial_file].copy()

    trial_codes = trial_meta["activity_code"].dropna().unique()
    adl_codes = [c for c in trial_codes if c in ADL_TRANSITIONS]

    if len(adl_codes) == 0:
        updated_trials.append(trial_meta)
        continue

    trial_path = TRIALS_ROOT / trial_file

    try:
        trial_df = load_trial(trial_path)
    except Exception as e:
        print(f"ERROR loading {trial_file}: {e}")
        updated_trials.append(trial_meta)
        continue

    settle_idx = find_settle_point(trial_df)

    for adl_code in adl_codes:

        cfg = ADL_TRANSITIONS[adl_code]
        duration = int(cfg["duration"] * config.FS)

        trans_end = settle_idx
        trans_start = max(0, trans_end - duration)

        pre_label = cfg["pre"]
        trans_label = cfg["trans"]
        post_label = cfg["post"]

        for idx, row in trial_meta.iterrows():

            if row["activity_code"] != adl_code:
                continue  

            start = int(row["start_idx"])
            end = int(row["end_idx"])

            mid_idx = (start + end) / 2.0

            if mid_idx < trans_start:
                trial_meta.loc[idx, "label"] = pre_label

            elif mid_idx > trans_end:
                trial_meta.loc[idx, "label"] = post_label

            else:
                trial_meta.loc[idx, "label"] = trans_label

            trial_meta.loc[idx, "reviewed"] = True

    updated_trials.append(trial_meta)

final_meta = pd.concat(updated_trials, ignore_index=True)

final_meta = final_meta.sort_values(
    by=["subject_id", "sensor_pos", "file", "start_idx"]
)

final_meta.to_parquet(config.WINDOWS_FILE, index=False)

print("\n==================== SUMMARY ====================")
print("Chair Transitions Extracted:\n")

trans_labels = {v["trans"] for v in ADL_TRANSITIONS.values()}

for label in sorted(trans_labels):
    c = (final_meta["label"] == label).sum()
    if c > 0:
        print(f"{label}: {c}")

print("=================================================\n")
print(f"Final Dataset saved to: {config.WINDOWS_FILE}")