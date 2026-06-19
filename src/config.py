from pathlib import Path
import torch

DATASET_ROOT = Path("IPqM-Fall")
RAW_DATASET_DIR = DATASET_ROOT / "raw"

FS = 90
WINDOW_SEC = 2
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
    "TASK_MODEL_MATRIX",
    "UNIFIED_MODEL_MATRIX",
]

TASK_MODELS_TO_RUN = [
#    "CNN1Conv",
    #"MLP",
    #"LOGREG",
    #"RF",
    #"SVM",
    #"KNN",
    "LGBM",
]

TASK_MODALITIES_TO_RUN = [
    "FULL_IMU",
#    "ACCELEROMETER",
#    "GYROSCOPE",
]

MODALITY_ABLATIONS = {
#    "ACCELEROMETER": [0, 1, 2, 3],
#   "GYROSCOPE": [4, 5, 6, 7],
    "FULL_IMU": [0, 1, 2, 3, 4, 5, 6, 7],
}

MULTI_TASK_MODEL = "CNN1Conv"

UNIFIED_MODELS_TO_RUN = [
 # "CNN1Conv",
   #"MLP",
   #"LOGREG",
   # "RF",
   # "SVM",
   # "KNN",
   "LGBM",
]

UNIFIED_MODALITIES_TO_RUN = [
    "FULL_IMU",
#    "ACCELEROMETER",
#    "GYROSCOPE",
]
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

CLASSICAL_FEATURE_SET = "COMPACT_7_TIME_DOMAIN"

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

LOGREG_PARAMS = {
    "penalty": "l2",
    "C": 1.0e8,
    "solver": "lbfgs",
    "max_iter": 5000,
    "tol": 1.0e-4,
    "class_weight": None,
    "random_state": 42,
}

MLP_HIDDEN_UNITS = 100
MLP_EPOCHS = 50
MLP_FULL_BATCH = True
MLP_SOURCE = (
    "Georgakopoulos et al., Change detection and "
    "convolution neural networks for fall recognition"
)

MULTI_TASK_WEIGHTS = {
    "fall": 1.0,
    #"movement_detect": 1.0,
    "fall_classify": 1.0,
    "posture": 1.0,
    "movement": 1.0,
    #"transition": 1.0,
}

RESUME_COMPLETED = True
FORCE_RERUN = False

EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 0.01 # 0.001
DROPOUT = 0.35

NUM_WORKERS = 4
PIN_MEMORY = True

RESULTS_DIR = Path("results") / WINDOW_TAG
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_DIRS = {
    "CNN1Conv": "cnn",
    "CNN3B3Conv": "cnn",
    "DeepConvLSTM": "cnn",
    "LSTM": "cnn",
    "MLP": "mlp",
    "LOGREG": "logreg",
    "RF": "rf",
    "SVM": "svm",
    "KNN": "knn",
    "LGBM": "lightgbm",
    "LIGHTGBM": "lightgbm",
}


def model_results_dir(model_type):
    key = model_type.upper()
    normalized = {
        "CNN1CONV": "CNN1Conv",
        "CNN3B3CONV": "CNN3B3Conv",
        "DEEPCONVLSTM": "DeepConvLSTM",
        "LSTM": "LSTM",
        "MLP": "MLP",
        "LOGREG": "LOGREG",
        "RF": "RF",
        "SVM": "SVM",
        "KNN": "KNN",
        "LGBM": "LGBM",
        "LIGHTGBM": "LIGHTGBM",
    }.get(key, model_type)

    folder = MODEL_OUTPUT_DIRS.get(
        normalized,
        str(model_type).lower(),
    )
    path = RESULTS_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    return path

CHECKPOINT_DIR = Path("checkpoints") / WINDOW_TAG
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")


def experiment_signature():
    """Return the settings that define checkpoint/cache compatibility."""
    return {
        "fs": int(FS),
        "window_sec": float(WINDOW_SEC),
        "stride_sec": float(STRIDE_SEC),
        "window_samples": int(WINDOW_SAMPLES),
        "stride_samples": int(STRIDE_SAMPLES),
        "window_tag": str(WINDOW_TAG),
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "dropout": float(DROPOUT),
        "classical_feature_set": str(CLASSICAL_FEATURE_SET),
    }


def print_experiment_summary():
    print("=" * 72)
    print("IPqM-Fall experiment matrix")
    print("=" * 72)
    print(f"Window: {WINDOW_SEC} s")
    print(f"Stride: {STRIDE_SEC} s")
    print(f"Sampling frequency: {FS} Hz")
    print(f"Device: {DEVICE}")

    print(f"Windowed dataset directory: {WINDOWED_DATASET_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Experiments: {EXPERIMENTS_TO_RUN}")

    if "TASK_MODEL_MATRIX" in EXPERIMENTS_TO_RUN:
        print(f"Tasks: {list(SINGLE_TASK_MODELS)}")
        print(f"Models: {TASK_MODELS_TO_RUN}")
        print(f"Modalities: {TASK_MODALITIES_TO_RUN}")

        print("Result directories:")
        for model_name in TASK_MODELS_TO_RUN:
            folder = MODEL_OUTPUT_DIRS.get(
                model_name,
                model_name.lower(),
            )
            print(f"  {model_name}: {RESULTS_DIR / folder}")
        print(
            "Sensor outputs per model/modality/task: "
            "CHEST, LEFT, RIGHT, CHEST_LEFT, CHEST_RIGHT, "
            "LEFT_RIGHT, CHEST_LEFT_RIGHT, "
            "ENSEMBLE_CHEST_LEFT, ENSEMBLE_CHEST_RIGHT, "
            "ENSEMBLE_LEFT_RIGHT, ENSEMBLE_CHEST_LEFT_RIGHT"
        )

    if "UNIFIED_MODEL_MATRIX" in EXPERIMENTS_TO_RUN:
        print(f"Unified models: {UNIFIED_MODELS_TO_RUN}")
        print(f"Unified modalities: {UNIFIED_MODALITIES_TO_RUN}")
        print(f"Unified classes: {UNIFIED_NUM_CLASSES}")

    print(f"Resume completed work: {RESUME_COMPLETED}")
    print(f"Force rerun: {FORCE_RERUN}")
    print("=" * 72)

