from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True

# --- Configuration Paths ---
PREPARED_DATA_DIR = Path("IPqM-Fall/fall_dataset_2_1")
RESULTS_FILE = "results/results_cnn1d_2_1.json"
CHECKPOINT_DIR = Path("checkpoints_cnn1d_2_1")
CHECKPOINT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["ax", "ay", "az", "amag", "wx", "wy", "wz", "wmag"]
SENSORS = ["CHEST", "LEFT", "RIGHT"]

# --- Global Window Parameters ---
SAMPLING_RATE_HZ = 90
WINDOW_SIZE_MS = 2000
WINDOW_SAMPLES = 180 

# # --- Global Window Parameters ---
# SAMPLING_RATE_HZ = 90
# WINDOW_SIZE_MS = 750
# WINDOW_SAMPLES = 68

# --- Hyperparameters ---
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 3e-4
DROPOUT = 0.35 # Updated to 35%
NUM_CLASSES = 2
EARLY_STOPPING_PATIENCE = 10

NUM_WORKERS = 2  # Set to 0 if you still see MemoryErrors
PIN_MEMORY = True
USE_CHANNELS_LAST = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FallDataset(Dataset):
    def __init__(self, X, y):
        # Convert shape from (N, Time, Channels) to PyTorch Conv1d format (N, Channels, Time)
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1Conv(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        # Single Convolutional Block (CNN-1Conv architecture)
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(DROPOUT) 
        )
        
        # Dynamically calculate the output size of the MaxPool1d layer
        # Formula for Conv1d length: L_out = L_in + 2*padding - kernel_size + 1
        conv_out_length = WINDOW_SAMPLES + 2*2 - 4 + 1
        # Formula for MaxPool1d length: L_out = L_in // kernel_size
        pool_out_length = conv_out_length // 3
        flattened_size = 64 * pool_out_length
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        if USE_CHANNELS_LAST and DEVICE.type == "cuda":
            X_batch = X_batch.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        else:
            X_batch = X_batch.to(DEVICE, non_blocking=True)

        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            if USE_CHANNELS_LAST and DEVICE.type == "cuda":
                X_batch = X_batch.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
            else:
                X_batch = X_batch.to(DEVICE, non_blocking=True)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


# --- CRITICAL FIX: ALL EXECUTABLE LOGIC INSIDE MAIN ---
if __name__ == '__main__':
    print(f"DEVICE: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"CUDA GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load or initialize tracking results
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = {}

    logo = LeaveOneGroupOut()

    for sensor in SENSORS:
        sensor_lower = sensor.lower()
        
        # Define paths to prepared data files
        x_path = PREPARED_DATA_DIR / f"X_{sensor_lower}.npy"
        y_path = PREPARED_DATA_DIR / f"y_{sensor_lower}.npy"
        groups_path = PREPARED_DATA_DIR / f"groups_{sensor_lower}.npy"
        
        if not (x_path.exists() and y_path.exists() and groups_path.exists()):
            print(f"Prepared files missing for sensor {sensor} at {PREPARED_DATA_DIR}. Skipping.")
            continue
            
        print(f"\nLoading prepared NumPy arrays for: {sensor}")
        X_sensor = np.load(x_path)
        y_sensor = np.load(y_path)
        groups_sensor = np.load(groups_path)

        folds = list(logo.split(X_sensor, y_sensor, groups_sensor))

        if sensor not in results:
            results[sensor] = {"folds": []}

        completed_folds = set(fold["fold"] for fold in results[sensor]["folds"])
        pbar = tqdm(folds, desc=sensor, total=len(folds))

        for fold_number, (train_idx, test_idx) in enumerate(pbar, start=1):
            if fold_number in completed_folds:
                pbar.set_postfix_str(f"fold={fold_number} skipped")
                continue

            X_train, X_test = X_sensor[train_idx], X_sensor[test_idx]
            y_train, y_test = y_sensor[train_idx], y_sensor[test_idx]

            # Inner-fold standard scaling to avoid data leakage
            train_mean = X_train.mean(axis=(0, 1), keepdims=True)
            train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

            # Perform operations in-place where possible to conserve memory
            X_train = (X_train - train_mean) / train_std
            X_test = (X_test - train_mean) / train_std

            test_subject = groups_sensor[test_idx][0]
            checkpoint_path = CHECKPOINT_DIR / f"{sensor}_fold_{fold_number}_subject_{test_subject}.pth"

            train_dataset = FallDataset(X_train, y_train)
            test_dataset = FallDataset(X_test, y_test)

            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0
            )

            model = CNN1Conv(num_features=len(FEATURE_COLUMNS))

            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
                model = nn.DataParallel(model)

            model = model.to(DEVICE)

            if USE_CHANNELS_LAST and DEVICE.type == "cuda":
                model = model.to(memory_format=torch.channels_last)

            fall_count = np.sum(y_train == 1)
            nonfall_count = np.sum(y_train == 0)

            class_weights = torch.tensor([1.0 / nonfall_count, 1.0 / fall_count], dtype=torch.float32).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            if checkpoint_path.exists():
                model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            else:
                best_loss = float("inf")
                patience_counter = 0

                for epoch in range(EPOCHS):
                    loss = train_epoch(model, train_loader, criterion, optimizer)
                    pbar.set_postfix_str(f"fold={fold_number} epoch={epoch+1}/{EPOCHS} loss={loss:.4f}")

                    if loss < best_loss:
                        best_loss = loss
                        patience_counter = 0
                        torch.save(
                            model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                            checkpoint_path
                        )
                    else:
                        patience_counter += 1

                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        break

                model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

            y_true, y_pred = evaluate(model, test_loader)

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            pbar.set_postfix_str(f"fold={fold_number} acc={acc:.4f} f1={f1:.4f}")

            fold_result = {
                "fold": fold_number,
                "test_subject": str(test_subject),
                "checkpoint": str(checkpoint_path),
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn)
            }

            results[sensor]["folds"].append(fold_result)
            folds_results = results[sensor]["folds"]

            for metric in ["accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]:
                values = [x[metric] for x in folds_results]
                results[sensor][f"{metric}_mean"] = float(np.mean(values))
                results[sensor][f"{metric}_std"] = float(np.std(values))

            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))