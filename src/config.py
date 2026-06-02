from pathlib import Path
import torch

EXPERIMENTS_TO_RUN = [
    #"SINGLE",      # Redes isoladas
    #"TRIPLE",      # Multitarefa (Queda + Postura + Movimento)
    "DOUBLE_FP",   # Multitarefa (Queda + Postura)
    #"DOUBLE_PM"    # Multitarefa (Postura + Movimento)
]

MULTI_TASK_MODEL = "CNN1Conv" # LSTM # DeepConvLSTM

SINGLE_TASK_MODELS = {
    "y_detect_fall": "CNN1Conv", # LSTM # DeepConvLSTM
    "y_classify_posture": "CNN1Conv", 
    "y_classify_movement": "CNN1Conv", 
}

CLASSICAL_MODEL = "RF"

EPOCHS = 80
NUM_WORKERS = 4
PIN_MEMORY = True

DATASET_DIR = Path("IPqM-Fall/windowed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu")

WINDOW_SAMPLES = 180
LEARNING_RATE = 0.01  
DROPOUT = 0.35