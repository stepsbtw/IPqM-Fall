from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("/home/caio.torkst/Projetos/tcc/IPqM-Fall/organized")

WINDOWS_CSV = "IPqM-Fall/windows.csv"

OUTPUT_DIR = Path("IPqM-Fall/windowed_synchronized")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "ax", "ay", "az", "amag",
    "wx", "wy", "wz", "wmag"
]

FALL_CLASSES = {
    "FALL_1",
    "FALL_2",
    "FALL_3",
    "FALL_5",
    "FALL_6"
}

SENSORS = ["CHEST", "LEFT", "RIGHT"]

EXPECTED_WINDOW_SAMPLES = 180

# ============================================================
# LOAD CSV
# ============================================================

print("Reading metadata CSV...")

windows_df = pd.read_csv(WINDOWS_CSV)

# ============================================================
# CREATE SYNCHRONIZED WINDOW ID
# ============================================================

# Remove sensor name from synchronization key
# Example:
#
# BEFORE:
# ID1_CHEST_ADL_1_trial1_win_0
#
# AFTER:
# ID1_ADL_1_trial1_win_0

windows_df["sync_id"] = (
    windows_df["subject_id"].astype(str)
    + "_"
    + windows_df["file"]
        .apply(lambda x: Path(x).stem)
        .str.replace("_combined", "", regex=False)
    + "_win_"
    + windows_df["start_idx"].astype(str)
)

# ============================================================
# GROUP WINDOWS
# ============================================================

grouped = windows_df.groupby("sync_id")

print(f"Total groups found: {len(grouped)}")

# ============================================================
# CACHE PARQUET FILES
# ============================================================

parquet_cache = {}

# ============================================================
# FINAL ARRAYS
# ============================================================

X_chest_list = []
X_left_list = []
X_right_list = []

y_list = []
groups_list = []
sync_ids_list = []

# ============================================================
# PROCESS GROUPS
# ============================================================

valid_groups = 0
invalid_groups = 0

for sync_id, group in tqdm(grouped, desc="Synchronizing windows"):

    # Must contain exactly 3 sensors
    sensors_present = set(group["sensor_pos"])

    if sensors_present != set(SENSORS):
        invalid_groups += 1
        continue

    try:

        sensor_windows = {}

        # ====================================================
        # LOAD EACH SENSOR WINDOW
        # ====================================================

        for _, row in group.iterrows():

            sensor = row["sensor_pos"]

            parquet_path = DATASET_ROOT / row["file"]

            if parquet_path not in parquet_cache:
                parquet_cache[parquet_path] = pd.read_parquet(
                    parquet_path
                )

            df = parquet_cache[parquet_path]

            window = df.iloc[
                row["start_idx"]:row["end_idx"]
            ]

            features = window[
                FEATURE_COLUMNS
            ].to_numpy(dtype=np.float32)

            # Safety check
            if len(features) != EXPECTED_WINDOW_SAMPLES:
                raise ValueError(
                    f"Invalid window size: {len(features)}"
                )

            sensor_windows[sensor] = features

        # ====================================================
        # LABEL
        # ====================================================

        label_name = group.iloc[0]["label"]

        label = (
            1
            if label_name in FALL_CLASSES
            else 0
        )

        subject_id = group.iloc[0]["subject_id"]

        # ====================================================
        # APPEND SYNCHRONIZED SAMPLE
        # ====================================================

        X_chest_list.append(sensor_windows["CHEST"])
        X_left_list.append(sensor_windows["LEFT"])
        X_right_list.append(sensor_windows["RIGHT"])

        y_list.append(label)

        groups_list.append(subject_id)

        sync_ids_list.append(sync_id)

        valid_groups += 1

    except Exception as e:

        print(f"\nSkipping {sync_id}")
        print(e)

        invalid_groups += 1

# ============================================================
# CONVERT TO NUMPY
# ============================================================

print("\nConverting to NumPy arrays...")

X_chest = np.array(X_chest_list, dtype=np.float32)
X_left = np.array(X_left_list, dtype=np.float32)
X_right = np.array(X_right_list, dtype=np.float32)

y = np.array(y_list, dtype=np.int64)

groups = np.array(groups_list)

sync_ids = np.array(sync_ids_list)

# ============================================================
# VALIDATION
# ============================================================

print("\nValidation...")

assert len(X_chest) == len(X_left)
assert len(X_chest) == len(X_right)

assert len(X_chest) == len(y)
assert len(X_chest) == len(groups)

print("Synchronization OK.")

# ============================================================
# SAVE DATASETS
# ============================================================

print("\nSaving datasets...")

np.save(OUTPUT_DIR / "X_chest.npy", X_chest)
np.save(OUTPUT_DIR / "X_left.npy", X_left)
np.save(OUTPUT_DIR / "X_right.npy", X_right)

np.save(OUTPUT_DIR / "y.npy", y)

np.save(OUTPUT_DIR / "groups.npy", groups)

np.save(OUTPUT_DIR / "sync_ids.npy", sync_ids)

# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("SYNCHRONIZED DATASET CREATED")
print("==============================")

print(f"Valid groups   : {valid_groups}")
print(f"Invalid groups : {invalid_groups}")

print(f"\nCHEST shape : {X_chest.shape}")
print(f"LEFT shape  : {X_left.shape}")
print(f"RIGHT shape : {X_right.shape}")

print(f"\ny shape      : {y.shape}")
print(f"groups shape : {groups.shape}")

print("\nFiles saved to:")
print(OUTPUT_DIR)

print("\nDone.")