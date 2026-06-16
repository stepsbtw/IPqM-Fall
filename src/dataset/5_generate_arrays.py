from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import src.config as config

OUTPUT_DIR = Path(f"IPqM-Fall/windowed/{config.WINDOW_SEC}-sec_{config.STRIDE_SEC}-step")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["ax", "ay", "az", "amag", "wx", "wy", "wz", "wmag"]
SENSORS = ["CHEST", "LEFT", "RIGHT"]

RAW_MAPPINGS = {
    "y_detect_fall": {
        "0": ["BACKWARD-FALL", "FRONTAL-FALL", "LATERAL-FALL-LEFT", "LATERAL-FALL-RIGHT"]
    },
    "y_detect_movement": {
        "0": ["STANDING", "STANDING-RIFLE", "SITTING", "SITTING-RIFLE", "KNEELING-SHOOTING", "PRONE-SHOOTING", "DOWN-SUPINE", "DOWN-PRONE", "DOWN-LEFT", "DOWN-RIGHT"]
    },
    "y_classify_fall": {
        "0": ["BACKWARD-FALL"],
        "1": ["FRONTAL-FALL"],
        "2": ["LATERAL-FALL-LEFT"],
        "3": ["LATERAL-FALL-RIGHT"]
    },
    "y_classify_posture": {
        "0": ["STANDING", "STANDING-RIFLE"],
        "1": ["SITTING", "SITTING-RIFLE"],
        "2": ["KNEELING-SHOOTING"],
        "3": ["PRONE-SHOOTING", "DOWN-SUPINE", "DOWN-PRONE", "DOWN-LEFT", "DOWN-RIGHT"]
    },
    "y_classify_movement": {
        "0": ["WALKING", "WALKING-DOWNHILL", "WALKING-DOWNSTAIRS", "WALKING-UPHILL", "WALKING-UPSTAIRS"],
        "1": ["ENGAGING-SWEEPING", "WALKING-SWEEPING"],
        "2": ["RUNNING", "RUNNING-DOWNHILL", "RUNNING-DOWNHILL-RIFLE", "RUNNING-RIFLE", "RUNNING-UPHILL", "RUNNING-UPHILL-RIFLE"],
        "3": ["JUMPING", "JUMPING-RIFLE", "JUMPING-UPSTAIRS"],
        "4": ["CRAWLING"]
    },
    "y_classify_transition": {
        "0": ["RUNNING-KNEELING-SHOOTING", "STANDING-KNEELING-SHOOTING", "WALKING-KNEELING-SHOOTING"],
        "1": ["RUNNING-PRONE-SHOOTING", "STANDING-PRONE-SHOOTING", "WALKING-PRONE-SHOOTING"],
        "2": ["SITTING-STANDING", "SITTING-STANDING-RIFLE", "STANDING-SITTING", "STANDING-SITTING-RIFLE"],
    },
    "y_unified": {
        "0": ["BACKWARD-FALL"],
        "1": ["FRONTAL-FALL"],
        "2": ["LATERAL-FALL-LEFT"],
        "3": ["LATERAL-FALL-RIGHT"],
        "4": ["STANDING", "STANDING-RIFLE"],
        "5": ["SITTING","SITTING-RIFLE"],
        "6": ["KNEELING-SHOOTING"],
        "7": ["PRONE-SHOOTING", "DOWN-SUPINE", "DOWN-PRONE", "DOWN-LEFT", "DOWN-RIGHT"],
        "8": ["WALKING", "WALKING-DOWNHILL", "WALKING-DOWNSTAIRS", "WALKING-UPHILL", "WALKING-UPSTAIRS"],
        "9": ["ENGAGING-SWEEPING", "WALKING-SWEEPING"],
        "10": ["RUNNING", "RUNNING-DOWNHILL", "RUNNING-DOWNHILL-RIFLE", "RUNNING-RIFLE", "RUNNING-UPHILL", "RUNNING-UPHILL-RIFLE"],
        "11": ["JUMPING", "JUMPING-RIFLE", "JUMPING-UPSTAIRS"],
        "12": ["CRAWLING"],
    }
}

def build_dict(json_map):
    mapping = {}
    for target_int_str, string_list in json_map.items():
        target_int = int(target_int_str)
        for label_string in string_list:
            mapping[label_string] = target_int
            if "-RIFLE" not in label_string:
                mapping[f"{label_string}-RIFLE"] = target_int
    return mapping

DICT_DETECT_FALL = build_dict(RAW_MAPPINGS["y_detect_fall"])
DICT_DETECT_MOVEMENT = build_dict(RAW_MAPPINGS["y_detect_movement"])
DICT_CLASSIFY_FALL = build_dict(RAW_MAPPINGS["y_classify_fall"])
DICT_CLASSIFY_POSTURE = build_dict(RAW_MAPPINGS["y_classify_posture"])
DICT_CLASSIFY_MOVEMENT = build_dict(RAW_MAPPINGS["y_classify_movement"])
DICT_CLASSIFY_TRANSITION = build_dict(RAW_MAPPINGS["y_classify_transition"])
DICT_UNIFIED = build_dict(RAW_MAPPINGS["y_unified"])
print("Lendo metadata Parquet...")
windows_df = pd.read_parquet(config.WINDOWS_FILE)

initial_len = len(windows_df)
windows_df = windows_df[windows_df["label"] != "UNKNOWN"]
print(f"Descartadas {initial_len - len(windows_df)} janelas UNKNOWN.")

unique_labels = sorted(windows_df["label"].unique())
DICT_COMPLETE = {label: idx for idx, label in enumerate(unique_labels)}

trial_num = windows_df["file"].apply(
    lambda x: Path(x).stem.rsplit("_TRIAL", 1)[1]
)

windows_df["sync_id"] = (
    windows_df["subject_id"].astype(str)
    + "_"
    + windows_df["activity_code"].astype(str)
    + "_TRIAL"
    + trial_num
    + "_win_"
    + windows_df["start_idx"].astype(str)
)

grouped = windows_df.groupby("sync_id")
print(f"Total de grupos sincronizados encontrados: {len(grouped)}")

parquet_cache = {}

X_chest_list, X_left_list, X_right_list = [], [], []

y_detect_fall_list, y_detect_movement_list = [], []
y_classify_fall_list, y_classify_posture_list, y_classify_movement_list, y_classify_transition_list, y_unified_list = [], [], [], [], []
y_complete_list = []
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
            parquet_path = config.DATASET_ROOT / "raw" / row["file"] 

            if parquet_path not in parquet_cache:
                parquet_cache[parquet_path] = pd.read_parquet(parquet_path)

            df = parquet_cache[parquet_path]
            features = df.iloc[row["start_idx"]:row["end_idx"]][FEATURE_COLUMNS].to_numpy(dtype=np.float32)

            if len(features) != config.WINDOW_SAMPLES:
                raise ValueError(f"Tamanho de janela inválido: {len(features)}")

            sensor_windows[sensor] = features

        X_chest_list.append(sensor_windows["CHEST"])
        X_left_list.append(sensor_windows["LEFT"])
        X_right_list.append(sensor_windows["RIGHT"])

        label_str = group.iloc[0]["label"]

        # Detectors default to 1 (No Event / No Movement) if not found in the "0" target lists
        y_detect_fall_list.append(DICT_DETECT_FALL.get(label_str, 1))
        y_detect_movement_list.append(DICT_DETECT_MOVEMENT.get(label_str, 1))

        # Classifiers default to -1 (Ignore class) if the specific label isn't part of that sub-task
        y_classify_fall_list.append(DICT_CLASSIFY_FALL.get(label_str, -1))
        y_classify_posture_list.append(DICT_CLASSIFY_POSTURE.get(label_str, -1))
        y_classify_movement_list.append(DICT_CLASSIFY_MOVEMENT.get(label_str, -1))
        y_classify_transition_list.append(DICT_CLASSIFY_TRANSITION.get(label_str, -1))
        y_complete_list.append(DICT_COMPLETE[label_str])
        y_unified_list.append(DICT_UNIFIED.get(label_str, -1))
        

        groups_list.append(group.iloc[0]["subject_id"])
        sync_ids_list.append(sync_id)

        valid_groups += 1

    except Exception as e:
        invalid_groups += 1

print("\nConvertendo para NumPy arrays (Isto pode levar alguns segundos)...")
np.save(config.WINDOWED_DATASET_DIR / "X_chest.npy", np.array(X_chest_list, dtype=np.float32))
np.save(config.WINDOWED_DATASET_DIR / "X_left.npy", np.array(X_left_list, dtype=np.float32))
np.save(config.WINDOWED_DATASET_DIR / "X_right.npy", np.array(X_right_list, dtype=np.float32))

np.save(config.WINDOWED_DATASET_DIR / "y_detect_fall.npy", np.array(y_detect_fall_list, dtype=np.int64))
np.save(config.WINDOWED_DATASET_DIR / "y_detect_movement.npy", np.array(y_detect_movement_list, dtype=np.int64))
np.save(config.WINDOWED_DATASET_DIR / "y_classify_fall.npy", np.array(y_classify_fall_list, dtype=np.int64))
np.save(config.WINDOWED_DATASET_DIR / "y_classify_posture.npy", np.array(y_classify_posture_list, dtype=np.int64))
np.save(config.WINDOWED_DATASET_DIR / "y_classify_movement.npy", np.array(y_classify_movement_list, dtype=np.int64))
np.save(config.WINDOWED_DATASET_DIR / "y_classify_transition.npy", np.array(y_classify_transition_list, dtype=np.int64))
np.save(config.WINDOWED_DATASET_DIR / "y_complete.npy", np.array(y_complete_list, dtype=np.int64))
np.save(config.WINDOWED_DATASET_DIR / "y_unified.npy", np.array(y_unified_list, dtype=np.int64))

np.save(config.WINDOWED_DATASET_DIR / "groups.npy", np.array(groups_list))
np.save(config.WINDOWED_DATASET_DIR / "sync_ids.npy", np.array(sync_ids_list))

mapa_documentacao = {
    "y_detect_fall": RAW_MAPPINGS["y_detect_fall"],
    "y_detect_movement": RAW_MAPPINGS["y_detect_movement"],
    "y_classify_fall": RAW_MAPPINGS["y_classify_fall"],
    "y_classify_posture": RAW_MAPPINGS["y_classify_posture"],
    "y_classify_movement": RAW_MAPPINGS["y_classify_movement"],
    "y_classify_transition": RAW_MAPPINGS["y_classify_transition"],
    "y_unified": RAW_MAPPINGS["y_unified"],
    "y_complete_map": {str(v): k for k, v in DICT_COMPLETE.items()}
}

with open(config.DATASET_ROOT / "mapping.json", "w") as f:
    json.dump(mapa_documentacao, f, indent=4)

print(f"Janelas Sincronizadas Válidas : {valid_groups}")
print(f"Janelas Inválidas/Perdidas    : {invalid_groups}")
print(f"Ficheiros gravados em         : {config.WINDOWED_DATASET_DIR}")