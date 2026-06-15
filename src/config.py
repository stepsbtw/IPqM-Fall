from pathlib import Path
import torch

DATASET_ROOT = Path("IPqM-Fall")
RAW_DATASET_DIR = DATASET_ROOT / "raw"

FS = 90
WINDOW_SEC = 5
STRIDE_SEC = 1

WINDOW_SAMPLES = FS * WINDOW_SEC
STRIDE_SAMPLES = FS * STRIDE_SEC

WINDOW_TAG = f"{WINDOW_SEC}-sec_{STRIDE_SEC}-step"

WINDOWS_FILE = DATASET_ROOT / f"windows_{WINDOW_SEC}_{STRIDE_SEC}.parquet"

WINDOWED_DATASET_DIR = DATASET_ROOT / "windowed" / WINDOW_TAG
WINDOWED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

PRE_FALL_SECONDS = 0.5
POST_FALL_SECONDS = 0.5

PRE_TRANSITION_SECONDS = 0.5
POST_TRANSITION_SECONDS = 0.5

TRANSITION_STATIC_THRESHOLD = 0.3
TRANSITION_SMOOTHING_WINDOW = int(FS * 0.5)


EXPERIMENTS_TO_RUN = [
    # "SINGLE",
    "UNIFIED",
    # "FALL_DETECT_POSTURE",
    # "FALL_DETECT_MOVEMENT",
    # "POSTURE_MOVEMENT",
    # "FALL_DETECT_POSTURE_MOVEMENT",
    # "FALL_CLASSIFY_POSTURE",
    # "FALL_CLASSIFY_POSTURE_MOVEMENT",
]

MULTI_TASK_MODEL = "CNN1Conv"

# Flat Task-vs-Non-Task baseline (13 macro-classes).
UNIFIED_MODEL = "CNN1Conv"
UNIFIED_TARGET = "y_unified.npy"
UNIFIED_NUM_CLASSES = 13

SINGLE_TASK_MODELS = {
    "y_detect_fall": "CNN1Conv", # CNN3B3Conv, LSTM, DeepConvLSTM
    #"y_detect_movement": "CNN1Conv",
    "y_classify_fall": "CNN1Conv",
    "y_classify_posture": "CNN1Conv",
    "y_classify_movement": "CNN1Conv",
    #"y_classify_transition": "CNN1Conv",
}

CLASSICAL_MODEL = "RF"

MULTI_TASK_WEIGHTS = {
    "fall": 1.0,
    #"movement_detect": 1.0,
    "fall_classify": 1.0,
    "posture": 1.0,
    "movement": 1.0,
    #"transition": 1.0,
}

EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 0.01 # 0.001
DROPOUT = 0.35

NUM_WORKERS = 4
PIN_MEMORY = True

RESULTS_DIR = Path("results") / WINDOW_TAG
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints") / WINDOW_TAG
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")