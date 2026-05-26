import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Classical Models ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import warnings
# Suppress scipy kurtosis warnings for perfectly flat arrays if they exist
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration Paths ---
DATASET_DIR = Path("IPqM-Fall/windowed_synchronized")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---> CHOOSE YOUR MODEL HERE: "RF", "SVM", or "KNN" <---
MODEL_TYPE = "RF" 

RESULTS_FILE = RESULTS_DIR / f"results_{MODEL_TYPE.lower()}.json"
CHECKPOINT_DIR = Path(f"checkpoints_{MODEL_TYPE.lower()}")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHEST_DIR = CHECKPOINT_DIR / "CHEST"
LEFT_DIR = CHECKPOINT_DIR / "LEFT"
RIGHT_DIR = CHECKPOINT_DIR / "RIGHT"

CHEST_DIR.mkdir(exist_ok=True)
LEFT_DIR.mkdir(exist_ok=True)
RIGHT_DIR.mkdir(exist_ok=True)

def get_model():
    """Returns the requested model with parameters optimized for imbalanced fall data."""
    if MODEL_TYPE == "SVM":
        return SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    elif MODEL_TYPE == "RF":
        return RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    elif MODEL_TYPE == "KNN":
        return KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose 'SVM', 'RF', or 'KNN'.")

def extract_handcrafted_features(X):
    """
    Compresses raw windowed data (N, 180, 8) into statistical features (N, 56).
    This exactly replicates the methodology of the UP-FALL and AHMET TURAN papers.
    """
    # Calculate statistics across axis 1 (the 180 time steps in the window)
    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    max_vals = np.max(X, axis=1)
    min_vals = np.min(X, axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    
    # Add a tiny epsilon to variance to prevent division by zero in skew/kurtosis
    X_safe = X + np.random.normal(0, 1e-8, X.shape)
    skewness = skew(X_safe, axis=1, bias=False)
    kurt = kurtosis(X_safe, axis=1, bias=False)
    
    # Concatenate all features: 8 channels * 7 statistics = 56 features per window
    X_features = np.concatenate([means, stds, max_vals, min_vals, rms, skewness, kurt], axis=1)
    
    # Replace any NaNs with 0 (in case of perfectly flat sensor readings)
    return np.nan_to_num(X_features)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

def train_single_sensor_classical(sensor_name, X_train, X_test, y_train, y_test, test_subject, fold_number):
    """Trains a classical ML model on the extracted features."""
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and Train
    model = get_model()
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] 

    metrics = compute_metrics(y_test, y_pred)
    metrics["fold"] = fold_number
    metrics["test_subject"] = str(test_subject)

    return {"metrics": metrics, "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}

if __name__ == "__main__":
    print(f"--- Running Classical Model: {MODEL_TYPE} ---")

    # 1. Load Raw Data
    print("Loading raw datasets...")
    X_chest_raw = np.load(DATASET_DIR / "X_chest.npy")
    X_left_raw = np.load(DATASET_DIR / "X_left.npy")
    X_right_raw = np.load(DATASET_DIR / "X_right.npy")
    
    y = np.load(DATASET_DIR / "y.npy")
    groups = np.load(DATASET_DIR / "groups.npy")

    # 2. Extract Features
    print(f"Extracting features from {X_chest_raw.shape[0]} windows... this takes a few seconds.")
    X_chest = extract_handcrafted_features(X_chest_raw)
    X_left = extract_handcrafted_features(X_left_raw)
    X_right = extract_handcrafted_features(X_right_raw)
    
    print(f"Dimensionality Reduction: {X_chest_raw.shape[1] * X_chest_raw.shape[2]} raw data points -> {X_chest.shape[1]} statistical features.")

    # 3. Cross-Validation Setup
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_chest, y, groups))

    results = {
        "CHEST": {"folds": []},
        "LEFT": {"folds": []},
        "RIGHT": {"folds": []},
        "ENSEMBLE": {"folds": []}
    }

    pbar = tqdm(folds, total=len(folds), desc="LOSO CV")

    # 4. Training Loop
    for fold_number, (train_idx, test_idx) in enumerate(pbar, start=1):
        test_subject = groups[test_idx][0]

        chest_results = train_single_sensor_classical("CHEST", X_chest[train_idx], X_chest[test_idx], y[train_idx], y[test_idx], test_subject, fold_number)
        left_results = train_single_sensor_classical("LEFT", X_left[train_idx], X_left[test_idx], y[train_idx], y[test_idx], test_subject, fold_number)
        right_results = train_single_sensor_classical("RIGHT", X_right[train_idx], X_right[test_idx], y[train_idx], y[test_idx], test_subject, fold_number)

        # Simple average ensemble across all 3 sensors
        ensemble_probs = (chest_results["y_prob"] + left_results["y_prob"] + right_results["y_prob"]) / 3.0
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)

        ensemble_metrics = compute_metrics(chest_results["y_true"], ensemble_preds)
        ensemble_metrics["fold"] = fold_number
        ensemble_metrics["test_subject"] = str(test_subject)

        results["CHEST"]["folds"].append(chest_results["metrics"])
        results["LEFT"]["folds"].append(left_results["metrics"])
        results["RIGHT"]["folds"].append(right_results["metrics"])
        results["ENSEMBLE"]["folds"].append(ensemble_metrics)

        pbar.set_postfix(ensemble_f1=f"{ensemble_metrics['f1']:.4f}")

    # 5. Calculate final summary statistics
    for model_name in ["CHEST", "LEFT", "RIGHT", "ENSEMBLE"]:
        folds_results = results[model_name]["folds"]
        for metric in ["accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]:
            values = [x[metric] for x in folds_results]
            results[model_name][f"{metric}_mean"] = float(np.mean(values))
            results[model_name][f"{metric}_std"] = float(np.std(values))

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nTraining Complete! Final Results saved to {RESULTS_FILE}")