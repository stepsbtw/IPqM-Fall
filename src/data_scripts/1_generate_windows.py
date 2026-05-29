import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import re

# =========================================================
# CONFIG
# =========================================================

TRIALS_ROOT = Path("IPqM-Fall/raw")

WINDOW_SEC = 2
STRIDE_SEC = 1
FS = 90

WINDOW_SIZE = WINDOW_SEC * FS
STRIDE = STRIDE_SEC * FS

# =========================================================
# LABEL TRANSLATION
# =========================================================

LABEL_MAPPING = {
    "ADL1": "STANDING",
    "ADL1R": "STANDING-RIFLE",
    "ADL2": "WALKING",
    "ADL3": "RUNNING",
    "ADL4": "JUMPING",
    "ADL4R": "JUMPING-RIFLE",

    "ADL5": "STANDING-SITTING",
    "ADL5R": "STANDING-SITTING-RIFLE",
    "ADL6": "SITTING-STANDING",
    "ADL6R": "SITTING-STANDING-RIFLE",

    "ADL7": "WALKING-UPHILL",
    "ADL8": "WALKING-DOWNHILL",

    "ADL9": "RUNNING-UPHILL",
    "ADL9R": "RUNNING-UPHILL-RIFLE",
    "ADL10": "RUNNING-DOWNHILL",
    "ADL10R": "RUNNING-DOWNHILL-RIFLE",

    "ADL11": "WALKING-UPSTAIRS",
    "ADL11R": "WALKING-UPSTAIRS-RIFLE",
    "ADL12": "WALKING-DOWNSTAIRS",
    "ADL12R": "WALKING-DOWNSTAIRS-RIFLE",

    "ADL13": "JUMPING-UPSTAIRS",
    "ADL13R": "JUMPING-UPSTAIRS-RIFLE",

    "MO1R": "WALKING-SWEEPING",
    "MO2R": "ENGAGING-SWEEPING",

    "MO3R": "STANDING-KNEELING-SHOOTING",
    "MO4R": "WALKING-KNEELING-SHOOTING",
    "MO5R": "RUNNING-KNEELING-SHOOTING",

    "MO6R": "STANDING-PRONE-SHOOTING",
    "MO7R": "WALKING-PRONE-SHOOTING",
    "MO8R": "RUNNING-PRONE-SHOOTING",

    "MO9": "CRAWLING",

    "FALL1": "FRONTAL-FALL-SUPINE",
    "FALL2": "FRONTAL-FALL-PRONE",
    "FALL3": "BACKWARD-FALL-SUPINE",
    "FALL4": "LATERAL-FALL-RIGHT",
    "FALL5": "LATERAL-FALL-LEFT",

    "FALL1R": "FRONTAL-FALL-SUPINE-RIFLE",
    "FALL2R": "FRONTAL-FALL-PRONE-RIFLE",
    "FALL3R": "BACKWARD-FALL-SUPINE-RIFLE",
    "FALL4R": "LATERAL-FALL-RIGHT-RIFLE",
    "FALL5R": "LATERAL-FALL-LEFT-RIFLE"
}

# =========================================================
# SAFE REGEX (FIXED)
# =========================================================

CODE_REGEX = re.compile(r"\b(ADL\d+R?|MO\d+R?|FALL\d+R?)(?:_|$)")

# =========================================================
# PROCESS
# =========================================================

rows = []
trial_files = sorted(TRIALS_ROOT.rglob("*.parquet"))

start_time = time.time()

for parquet_file in tqdm(trial_files, desc="Processing trials", unit="file"):

    # -------------------------
    # SAFE LOAD
    # -------------------------
    try:
        df = pd.read_parquet(parquet_file)
    except Exception:
        continue

    if len(df) < WINDOW_SIZE:
        continue

    # -------------------------
    # METADATA
    # -------------------------
    subject_id = parquet_file.parent.parent.name
    sensor_pos = parquet_file.parent.name
    rel_path = parquet_file.relative_to(TRIALS_ROOT).as_posix()

    stem = parquet_file.stem

    match = CODE_REGEX.search(stem)

    if match:
        activity_code = match.group(1)
        initial_label = LABEL_MAPPING.get(activity_code, "UNKNOWN")
    else:
        activity_code = "UNKNOWN"
        initial_label = "UNKNOWN"

    # -------------------------
    # WINDOWING
    # -------------------------
    n_windows = ((len(df) - WINDOW_SIZE) // STRIDE) + 1

    for i in range(n_windows):

        start = i * STRIDE
        end = start + WINDOW_SIZE

        # UNIQUE SAFE ID (FIXED)
        window_id = (
            f"{subject_id}_"
            f"{sensor_pos}_"
            f"{parquet_file.stem}_"
            f"win_{i:06d}"
        )

        rows.append({
            "file": rel_path,
            "subject_id": subject_id,
            "sensor_pos": sensor_pos,
            "window_id": window_id,
            "start_idx": start,
            "end_idx": end,
            "activity_code": activity_code,
            "label": initial_label,
            "reviewed": False
        })

# =========================================================
# FINAL DATASET
# =========================================================

elapsed = time.time() - start_time

meta = pd.DataFrame(rows)

meta = meta.sort_values(
    by=["subject_id", "sensor_pos", "file", "start_idx"]
)

meta.to_parquet("IPqM-Fall/windows.parquet", index=False)

# =========================================================
# SUMMARY
# =========================================================

print(f"\nTrials processed : {len(trial_files)}")
print(f"Windows created   : {len(meta)}")
print(f"Elapsed time      : {elapsed:.2f} sec\n")

print("Label distribution:\n")
print(meta["label"].value_counts().sort_index())

print("\nSample:\n")
print(meta.head())