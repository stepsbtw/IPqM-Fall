from sklearn.model_selection import LeaveOneGroupOut
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils import compute_metrics, train_dl_model, train_classical_model

import src.config as config

if __name__ == "__main__":
    print(f"=== INICIANDO PIPELINE UNIFICADO: {config.MODEL_TYPE} ===")
    torch.backends.cudnn.benchmark = True if config.MODEL_TYPE in ["CNN", "CNN_LSTM", "LSTM", "MLP"] else False

    X_chest, X_left, X_right = np.load(config.DATASET_DIR/"X_chest.npy"), np.load(config.DATASET_DIR/"X_left.npy"), np.load(config.DATASET_DIR/"X_right.npy")
    y, groups = np.load(config.DATASET_DIR/"y.npy"), np.load(config.DATASET_DIR/"groups.npy")

    models_data = {
        "CHEST": X_chest, "LEFT": X_left, "RIGHT": X_right,
        "CHEST_LEFT": np.concatenate((X_chest, X_left), axis=2),
        "CHEST_RIGHT": np.concatenate((X_chest, X_right), axis=2),
        "LEFT_RIGHT": np.concatenate((X_left, X_right), axis=2),
        "CHEST_LEFT_RIGHT": np.concatenate((X_chest, X_left, X_right), axis=2)
    }

    results = {name: {"folds": []} for name in models_data.keys()}
    
    late_fusions_map = {
        "ENSEMBLE_CHEST_LEFT": ["CHEST", "LEFT"],
        "ENSEMBLE_CHEST_RIGHT": ["CHEST", "RIGHT"],
        "ENSEMBLE_LEFT_RIGHT": ["LEFT", "RIGHT"],
        "ENSEMBLE_CHEST_LEFT_RIGHT": ["CHEST", "LEFT", "RIGHT"]
    }
    for fusion_name in late_fusions_map.keys():
        results[fusion_name] = {"folds": []}
    
    folds = list(LeaveOneGroupOut().split(X_chest, y, groups))
    pbar = tqdm(folds, total=len(folds), desc="LOSO CV")

    for fold_number, (train_idx, test_idx) in enumerate(pbar, start=1):
        test_subject = groups[test_idx][0]
        fold_probs = {}

        for model_name, X_data in models_data.items():
            save_dir = config.CHECKPOINT_DIR / model_name
            save_dir.mkdir(exist_ok=True)

            if config.MODEL_TYPE in ["RF", "SVM", "KNN"]:
                metrics, y_prob = train_classical_model(model_name, X_data[train_idx], X_data[test_idx], y[train_idx], y[test_idx], test_subject, fold_number)
            else:
                metrics, y_prob = train_dl_model(model_name, X_data[train_idx], X_data[test_idx], y[train_idx], y[test_idx], test_subject, fold_number, save_dir)
            
            fold_probs[model_name] = y_prob
            metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
            results[model_name]["folds"].append(metrics)

        for fusion_name, sensors in late_fusions_map.items():
            late_probs = sum(fold_probs[s] for s in sensors) / len(sensors)
            late_preds = (late_probs >= 0.5).astype(int)
            
            late_metrics = compute_metrics(y[test_idx], late_preds)
            late_metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
            results[fusion_name]["folds"].append(late_metrics)

        with open(config.RESULTS_FILE, "w") as f: json.dump(results, f, indent=4)

    for model_name in results.keys():
        for metric in ["accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]:
            values = [x[metric] for x in results[model_name]["folds"]]
            results[model_name][f"{metric}_mean"], results[model_name][f"{metric}_std"] = float(np.mean(values)), float(np.std(values))

    with open(config.RESULTS_FILE, "w") as f: json.dump(results, f, indent=4)
    print(f"\n Pipeline concluído com sucesso! Resultados salvos em: {config.RESULTS_FILE}")