from pathlib import Path
import torch

EPOCHS = 80
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 2
PIN_MEMORY = True

TASK_MODELS = {
    "y_detect_fall": "CNN1Conv",
    #"y_classify_fall": "CNN1Conv",
    "y_classify_posture": "CNN1Conv", #LSTM
    "y_classify_movement": "CNN1Conv", #LSTM
    #"y_detect_movement": "DeepConvLSTM",
    #"y_complete": "CNN1Conv" 
}

DATASET_DIR = Path("IPqM-Fall/windowed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SAMPLES = 180
LEARNING_RATE = 3e-4
DROPOUT = 0.5
DROPOUT_HYBRID = 0.35