from pathlib import Path
import torch

EXPERIMENTS_TO_RUN = [
    "SINGLE",      # Redes isoladas
    #"FALL_DETECT_POSTURE_MOVEMENT",      # Multitarefa (Queda + Postura + Movimento)
    #"FALL_DETECT_POSTURE",   # Multitarefa (Queda + Postura)
    #"POSTURE_MOVEMENT"    # Multitarefa (Postura + Movimento)
    #"FALL_CLASSIFY_POSTURE_MOVEMENT"    # Multitarefa (Classificar Queda + Postura + Movimento)
    #"FALL_CLASSIFY_POSTURE"    # Multitarefa (Classificar Queda + Postura)
]

MULTI_TASK_MODEL = "CNN3B3Conv" # LSTM # DeepConvLSTM

SINGLE_TASK_MODELS = {
    "y_detect_fall": "CNN3B3Conv", # LSTM # DeepConvLSTM
    "y_classify_posture": "CNN3B3Conv", # LSTM # DeepConvLSTM
    "y_classify_fall": "CNN3B3Conv", # LSTM # DeepConvLSTM
    "y_classify_movement": "CNN3B3Conv" # LSTM # DeepConvLSTM, 
}

MULTI_TASK_WEIGHTS = { # hiperparametro!
    "fall": 1.0, # 0.5
    "posture": 1.0, # 1.0
    "movement": 1.0, # 1.0
    "fall_classify": 1.0 # 1.0
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