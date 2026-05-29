from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = Path("IPqM-Fall")
WINDOWS_FILE = "IPqM-Fall/windows_final.parquet" # Updated to Parquet
OUTPUT_DIR = Path("IPqM-Fall/windowed")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["ax", "ay", "az", "amag", "wx", "wy", "wz", "wmag"]
SENSORS = ["CHEST", "LEFT", "RIGHT"]
EXPECTED_WINDOW_SAMPLES = 180

# ============================================================
# 1. NEW STRING-BASED TAXONOMY (Padronizado com Hifens)
# ============================================================

RAW_MAPPINGS = {
    "y_fall_map": {
        "0": ["CRAWLING", "ENGAGING-SWEEPING", "JUMPING", "JUMPING-RIFLE", "JUMPING-UPSTAIRS", "KNEELING-SHOOTING", "PRONE-SHOOTING", "RUNNING", "RUNNING-DOWNHILL", "RUNNING-DOWNHILL-RIFLE", "RUNNING-KNEELING-SHOOTING", "RUNNING-PRONE-SHOOTING", "RUNNING-RIFLE", "RUNNING-UPHILL", "RUNNING-UPHILL-RIFLE", "SITTING", "SITTING-RIFLE", "SITTING-STANDING", "SITTING-STANDING-RIFLE", "STANDING", "STANDING-KNEELING-SHOOTING", "STANDING-PRONE-SHOOTING", "STANDING-RIFLE", "STANDING-SITTING", "STANDING-SITTING-RIFLE", "WALKING", "WALKING-DOWNHILL", "WALKING-DOWNSTAIRS", "WALKING-KNEELING-SHOOTING", "WALKING-PRONE-SHOOTING", "WALKING-SWEEPING", "WALKING-UPHILL", "WALKING-UPSTAIRS"],
        "1": ["BACKWARD-FALL", "DOWN-LEFT", "DOWN-PRONE", "DOWN-RIGHT", "DOWN-SUPINE", "FRONTAL-FALL", "LATERAL-FALL-LEFT", "LATERAL-FALL-RIGHT"]
    },
    "y_static_map": {
        "0": ["STANDING", "STANDING-RIFLE"],
        "1": ["SITTING", "SITTING-RIFLE"],
        "2": ["KNEELING-SHOOTING"],
        "3": ["PRONE-SHOOTING"],
        "-1": ["BACKWARD-FALL", "DOWN-LEFT", "DOWN-PRONE", "DOWN-RIGHT", "DOWN-SUPINE", "FRONTAL-FALL", "LATERAL-FALL-LEFT", "LATERAL-FALL-RIGHT", "CRAWLING", "ENGAGING-SWEEPING", "JUMPING", "JUMPING-RIFLE", "JUMPING-UPSTAIRS", "RUNNING", "RUNNING-DOWNHILL", "RUNNING-DOWNHILL-RIFLE", "RUNNING-KNEELING-SHOOTING", "RUNNING-PRONE-SHOOTING", "RUNNING-RIFLE", "RUNNING-UPHILL", "RUNNING-UPHILL-RIFLE", "SITTING-STANDING", "SITTING-STANDING-RIFLE", "STANDING-KNEELING-SHOOTING", "STANDING-PRONE-SHOOTING", "STANDING-SITTING", "STANDING-SITTING-RIFLE", "WALKING", "WALKING-DOWNHILL", "WALKING-DOWNSTAIRS", "WALKING-KNEELING-SHOOTING", "WALKING-PRONE-SHOOTING", "WALKING-SWEEPING", "WALKING-UPHILL", "WALKING-UPSTAIRS"]
    },
    "y_dynamic_map": {
        "0": ["WALKING", "WALKING-DOWNHILL", "WALKING-DOWNSTAIRS", "WALKING-UPHILL", "WALKING-UPSTAIRS"],
        "1": ["ENGAGING-SWEEPING", "WALKING-SWEEPING"],
        "2": ["RUNNING", "RUNNING-DOWNHILL", "RUNNING-DOWNHILL-RIFLE", "RUNNING-RIFLE", "RUNNING-UPHILL", "RUNNING-UPHILL-RIFLE"],
        "3": ["JUMPING", "JUMPING-RIFLE", "JUMPING-UPSTAIRS"],
        "4": ["CRAWLING"],
        "-1": ["BACKWARD-FALL", "DOWN-LEFT", "DOWN-PRONE", "DOWN-RIGHT", "DOWN-SUPINE", "FRONTAL-FALL", "LATERAL-FALL-LEFT", "LATERAL-FALL-RIGHT", "KNEELING-SHOOTING", "PRONE-SHOOTING", "SITTING", "SITTING-RIFLE", "SITTING-STANDING", "SITTING-STANDING-RIFLE", "STANDING", "STANDING-KNEELING-SHOOTING", "STANDING-PRONE-SHOOTING", "STANDING-RIFLE", "STANDING-SITTING", "STANDING-SITTING-RIFLE", "WALKING-KNEELING-SHOOTING", "WALKING-PRONE-SHOOTING", "RUNNING-KNEELING-SHOOTING", "RUNNING-PRONE-SHOOTING"]
    },
    "y_transition_map": {
        "0": ["RUNNING-KNEELING-SHOOTING", "STANDING-KNEELING-SHOOTING", "WALKING-KNEELING-SHOOTING"],
        "1": ["RUNNING-PRONE-SHOOTING", "STANDING-PRONE-SHOOTING", "WALKING-PRONE-SHOOTING"],
        "2": ["SITTING-STANDING", "SITTING-STANDING-RIFLE", "STANDING-SITTING", "STANDING-SITTING-RIFLE"],
        "-1": ["BACKWARD-FALL", "DOWN-LEFT", "DOWN-PRONE", "DOWN-RIGHT", "DOWN-SUPINE", "FRONTAL-FALL", "LATERAL-FALL-LEFT", "LATERAL-FALL-RIGHT", "CRAWLING", "ENGAGING-SWEEPING", "JUMPING", "JUMPING-RIFLE", "JUMPING-UPSTAIRS", "KNEELING-SHOOTING", "PRONE-SHOOTING", "RUNNING", "RUNNING-DOWNHILL", "RUNNING-DOWNHILL-RIFLE", "RUNNING-RIFLE", "RUNNING-UPHILL", "RUNNING-UPHILL-RIFLE", "SITTING", "SITTING-RIFLE", "STANDING", "STANDING-RIFLE", "WALKING", "WALKING-DOWNHILL", "WALKING-DOWNSTAIRS", "WALKING-SWEEPING", "WALKING-UPHILL", "WALKING-UPSTAIRS"]
    },
    "y_weapon_map": {
        "0": ["CRAWLING", "JUMPING", "JUMPING-UPSTAIRS", "RUNNING", "RUNNING-DOWNHILL", "RUNNING-UPHILL", "SITTING", "SITTING-STANDING", "STANDING", "STANDING-SITTING", "WALKING", "WALKING-DOWNHILL", "WALKING-DOWNSTAIRS", "WALKING-UPHILL", "WALKING-UPSTAIRS"],
        "1": ["ENGAGING-SWEEPING", "JUMPING-RIFLE", "KNEELING-SHOOTING", "PRONE-SHOOTING", "RUNNING-DOWNHILL-RIFLE", "RUNNING-KNEELING-SHOOTING", "RUNNING-PRONE-SHOOTING", "RUNNING-RIFLE", "RUNNING-UPHILL-RIFLE", "SITTING-RIFLE", "SITTING-STANDING-RIFLE", "STANDING-KNEELING-SHOOTING", "STANDING-PRONE-SHOOTING", "STANDING-RIFLE", "STANDING-SITTING-RIFLE", "WALKING-KNEELING-SHOOTING", "WALKING-PRONE-SHOOTING", "WALKING-SWEEPING"],
        "-1": ["BACKWARD-FALL", "DOWN-LEFT", "DOWN-PRONE", "DOWN-RIGHT", "DOWN-SUPINE", "FRONTAL-FALL", "LATERAL-FALL-LEFT", "LATERAL-FALL-RIGHT"] 
    }
}

def build_dict(json_map):
    """Flattens the arrays into direct String -> Integer mappings and handles the -RIFLE suffix automatically."""
    mapping = {}
    for target_int_str, string_list in json_map.items():
        target_int = int(target_int_str)
        for label_string in string_list:
            mapping[label_string] = target_int
            # Automatically map the Armed equivalent to the same movement class
            if "-RIFLE" not in label_string:
                mapping[f"{label_string}-RIFLE"] = target_int
    return mapping

# Generate the flattened dictionaries
DICT_FALL = build_dict(RAW_MAPPINGS["y_fall_map"])
DICT_STATIC = build_dict(RAW_MAPPINGS["y_static_map"])
DICT_DYNAMIC = build_dict(RAW_MAPPINGS["y_dynamic_map"])
DICT_TRANSITION = build_dict(RAW_MAPPINGS["y_transition_map"])
DICT_WEAPON = build_dict(RAW_MAPPINGS["y_weapon_map"])

# ============================================================
# DATA PROCESSING
# ============================================================
print("Lendo metadata Parquet...")
windows_df = pd.read_parquet(WINDOWS_FILE)

# CRITICAL FIX: Drop the unknown/quarantined windows before exporting
initial_len = len(windows_df)
windows_df = windows_df[windows_df["label"] != "UNKNOWN"]
print(f"Descartadas {initial_len - len(windows_df)} janelas UNKNOWN.")

# Create y_complete dynamically based on unique remaining strings
unique_labels = sorted(windows_df["label"].unique())
DICT_COMPLETE = {label: idx for idx, label in enumerate(unique_labels)}

# Build the sync_id
windows_df["sync_id"] = (
    windows_df["subject_id"].astype(str) + "_" +
    windows_df["file"].apply(lambda x: Path(x).stem).str.replace("_combined", "", regex=False) +
    "_win_" + windows_df["start_idx"].astype(str)
)

grouped = windows_df.groupby("sync_id")
print(f"Total de grupos sincronizados encontrados: {len(grouped)}")

parquet_cache = {}

X_chest_list, X_left_list, X_right_list = [], [], []
y_fall_list, y_static_list, y_dynamic_list, y_transition_list, y_weapon_list, y_complete_list = [], [], [], [], [], []
groups_list, sync_ids_list = [], []

valid_groups, invalid_groups = 0, 0

for sync_id, group in tqdm(grouped, desc="Extraindo e Rotulando Janelas"):
    if set(group["sensor_pos"]) != set(SENSORS):
        invalid_groups += 1
        continue

    try:
        sensor_windows = {}
        for _, row in group.iterrows():
            sensor = row["sensor_pos"]
            parquet_path = DATASET_ROOT / "raw" / row["file"] # Ensure path matches your structure

            if parquet_path not in parquet_cache:
                parquet_cache[parquet_path] = pd.read_parquet(parquet_path)

            df = parquet_cache[parquet_path]
            features = df.iloc[row["start_idx"]:row["end_idx"]][FEATURE_COLUMNS].to_numpy(dtype=np.float32)

            if len(features) != EXPECTED_WINDOW_SAMPLES:
                raise ValueError(f"Tamanho de janela inválido: {len(features)}")

            sensor_windows[sensor] = features

        # Extract features
        X_chest_list.append(sensor_windows["CHEST"])
        X_left_list.append(sensor_windows["LEFT"])
        X_right_list.append(sensor_windows["RIGHT"])

        # ====================================================
        # MULTI-HEAD LABELING
        # ====================================================
        label_str = group.iloc[0]["label"]

        # Default to -1 (Ignored) if the specific label isn't found in a subset dictionary
        y_fall_list.append(DICT_FALL.get(label_str, -1))
        y_static_list.append(DICT_STATIC.get(label_str, -1))
        y_dynamic_list.append(DICT_DYNAMIC.get(label_str, -1))
        y_transition_list.append(DICT_TRANSITION.get(label_str, -1))
        y_weapon_list.append(DICT_WEAPON.get(label_str, -1))
        y_complete_list.append(DICT_COMPLETE[label_str])

        groups_list.append(group.iloc[0]["subject_id"])
        sync_ids_list.append(sync_id)

        valid_groups += 1

    except Exception as e:
        invalid_groups += 1

# ============================================================
# EXPORT
# ============================================================
print("\nConvertendo para NumPy arrays (Isto pode levar alguns segundos)...")
np.save(OUTPUT_DIR / "X_chest.npy", np.array(X_chest_list, dtype=np.float32))
np.save(OUTPUT_DIR / "X_left.npy", np.array(X_left_list, dtype=np.float32))
np.save(OUTPUT_DIR / "X_right.npy", np.array(X_right_list, dtype=np.float32))

np.save(OUTPUT_DIR / "y_fall.npy", np.array(y_fall_list, dtype=np.int64))
np.save(OUTPUT_DIR / "y_static.npy", np.array(y_static_list, dtype=np.int64))
np.save(OUTPUT_DIR / "y_dynamic.npy", np.array(y_dynamic_list, dtype=np.int64))
np.save(OUTPUT_DIR / "y_transition.npy", np.array(y_transition_list, dtype=np.int64))
np.save(OUTPUT_DIR / "y_weapon.npy", np.array(y_weapon_list, dtype=np.int64))
np.save(OUTPUT_DIR / "y_complete.npy", np.array(y_complete_list, dtype=np.int64))

np.save(OUTPUT_DIR / "groups.npy", np.array(groups_list))
np.save(OUTPUT_DIR / "sync_ids.npy", np.array(sync_ids_list))

# Save documentation
mapa_documentacao = {
    "y_fall_map": RAW_MAPPINGS["y_fall_map"],
    "y_static_map": RAW_MAPPINGS["y_static_map"],
    "y_dynamic_map": RAW_MAPPINGS["y_dynamic_map"],
    "y_transition_map": RAW_MAPPINGS["y_transition_map"],
    "y_weapon_map": RAW_MAPPINGS["y_weapon_map"],
    "y_complete_map": {str(v): k for k, v in DICT_COMPLETE.items()}
}

with open(OUTPUT_DIR / "mapping.json", "w") as f:
    json.dump(mapa_documentacao, f, indent=4)

print("\n=======================================================")
print("DATASET OMNI-CASCATA CRIADO (ATUALIZADO PARA STRINGS)!")
print("=======================================================")
print(f"Janelas Sincronizadas Válidas : {valid_groups}")
print(f"Janelas Inválidas/Perdidas    : {invalid_groups}")
print(f"Ficheiros gravados em         : {OUTPUT_DIR}")