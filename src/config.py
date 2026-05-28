from pathlib import Path
import torch

EPOCHS = 80
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 2
PIN_MEMORY = True

MODEL_TYPE = "CNN"  

DATASET_DIR = Path("IPqM-Fall/windowed_synchronized")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / f"results_{MODEL_TYPE.lower()}.json"
CHECKPOINT_DIR = Path(f"checkpoints/checkpoints_{MODEL_TYPE.lower()}")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SAMPLES = 180
LEARNING_RATE = 3e-4
DROPOUT = 0.5
DROPOUT_HYBRID = 0.35
NUM_CLASSES = 2