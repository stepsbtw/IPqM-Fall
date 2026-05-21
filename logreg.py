from pathlib import Path
import json
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Configuration Paths ---
PREPARED_DATA_DIR = Path("IPqM-Fall/fall_dataset")
RESULTS_FILE = "results.json"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

SENSORS = ["CHEST", "LEFT", "RIGHT"]

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
    X_sensor_raw = np.load(x_path)
    y_sensor_binary = np.load(y_path)
    groups_sensor = np.load(groups_path)

    # Flatten the windows from 3D (samples, window_length, features) 
    # to 2D (samples, window_length * features) for Logistic Regression
    X_sensor = X_sensor_raw.reshape(X_sensor_raw.shape[0], -1)

    # Dynamically convert binary 1/0 to "FALL"/"NON_FALL" text labels
    y_sensor = np.where(y_sensor_binary == 1, "FALL", "NON_FALL")

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

        test_subject = groups_sensor[test_idx][0]
        checkpoint_path = CHECKPOINT_DIR / f"{sensor}_fold_{fold_number}_subject_{test_subject}.joblib"

        pbar.set_postfix_str(f"fold={fold_number} subject={test_subject}")

        if checkpoint_path.exists():
            model = joblib.load(checkpoint_path)
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000
                )
            )
            model.fit(X_train, y_train)
            joblib.dump(model, checkpoint_path)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="FALL", zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label="FALL", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="FALL", zero_division=0)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=["NON_FALL", "FALL"]
        ).ravel()

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

        # Vectorized metrics updates for tracking
        for metric in ["accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]:
            values = [x[metric] for x in folds_results]
            results[sensor][f"{metric}_mean"] = float(np.mean(values))
            results[sensor][f"{metric}_std"] = float(np.std(values))

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)

print(json.dumps(results, indent=4))