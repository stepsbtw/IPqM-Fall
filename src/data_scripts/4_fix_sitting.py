import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================

TRIALS_ROOT = Path("IPqM-Fall/raw")
# Point this to the output of your previous tactical (MO) script
# This creates a perfect daisy-chain processing pipeline
META_FILE = "IPqM-Fall/windows_mo_fixed.parquet" 
OUTPUT_FILE = "IPqM-Fall/windows_final.parquet"

FS = 90
STATIC_THRESHOLD = 0.3
SMOOTHING_WINDOW = int(FS * 0.5)

# =========================================================
# ADL TRANSITION TAXONOMY
# =========================================================
# Note: I am introducing the "SITTING" and "SITTING-RIFLE" labels 
# for the post/pre states, assuming the soldier remains seated.

ADL_TRANSITIONS = {
    # Stand-to-Sit (2.0 seconds)
    "ADL5":  {"pre": "STANDING",       "trans": "STANDING-SITTING",       "post": "SITTING",       "duration": 2.0},
    "ADL5R": {"pre": "STANDING-RIFLE", "trans": "STANDING-SITTING-RIFLE", "post": "SITTING-RIFLE", "duration": 2.0},
    
    # Sit-to-Stand (2.0 seconds)
    "ADL6":  {"pre": "SITTING",       "trans": "SITTING-STANDING",       "post": "STANDING",       "duration": 2.0},
    "ADL6R": {"pre": "SITTING-RIFLE", "trans": "SITTING-STANDING-RIFLE", "post": "STANDING-RIFLE", "duration": 2.0},
}

# =========================================================
# HELPERS
# =========================================================

def load_trial(path):
    return pd.read_parquet(path).reset_index(drop=True)

def find_settle_point(df):
    """
    Vectorized scan to find the exact moment the torso stops rotating
    (e.g., resting against the chair backrest, or locking into a vertical stand).
    """
    wmag = df["wmag"].values

    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smooth = np.convolve(wmag, kernel, mode="same")

    above = np.where(smooth > STATIC_THRESHOLD)[0]

    if len(above) == 0:
        return len(df) - 1

    return int(above[-1])

# =========================================================
# MAIN PROCESSING
# =========================================================

meta = pd.read_parquet(META_FILE)
updated_trials = []

trial_files = sorted(meta["file"].unique())

for trial_file in tqdm(trial_files, desc="Processing Chair Transitions"):

    trial_meta = meta[meta["file"] == trial_file].copy()

    trial_codes = trial_meta["activity_code"].dropna().unique()
    adl_codes = [c for c in trial_codes if c in ADL_TRANSITIONS]

    # If this file doesn't contain chair transitions, skip processing
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
        duration = int(cfg["duration"] * FS)

        trans_end = settle_idx
        trans_start = max(0, trans_end - duration)

        pre_label = cfg["pre"]
        trans_label = cfg["trans"]
        post_label = cfg["post"]

        # =====================================================
        # MIDPOINT LABELING RULE
        # =====================================================

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

# =========================================================
# FINAL OUTPUT
# =========================================================

final_meta = pd.concat(updated_trials, ignore_index=True)

final_meta = final_meta.sort_values(
    by=["subject_id", "sensor_pos", "file", "start_idx"]
)

final_meta.to_parquet(OUTPUT_FILE, index=False)

# =========================================================
# SUMMARY
# =========================================================

print("\n==================== SUMMARY ====================")
print("Chair Transitions Extracted:\n")

trans_labels = {v["trans"] for v in ADL_TRANSITIONS.values()}

for label in sorted(trans_labels):
    c = (final_meta["label"] == label).sum()
    if c > 0:
        print(f"{label}: {c}")

print("=================================================\n")
print(f"Final Dataset saved to: {OUTPUT_FILE}")