from sklearn.model_selection import LeaveOneGroupOut
import json
import torch
import numpy as np
from tqdm import tqdm
from utils import compute_metrics, train_dl_model, train_classical_model

import config

def train_task(task_name, model_type, X_chest_full, X_left_full, X_right_full, groups_full):
    print(f"\n{'='*50}")
    print(f"=== STARTING TASK: {task_name.upper()} using {model_type} ===")
    print(f"{'='*50}")
    
    torch.backends.cudnn.benchmark = True if model_type in ["CNN1Conv", "DeepConvLSTM", "LSTM", "MLP"] else False

    y_full = np.load(config.DATASET_DIR / f"{task_name}.npy")
    valid_idx = y_full != -1
    y = y_full[valid_idx]
    
    if len(y) == 0:
        print(f"No valid instances found for {task_name}. Skipping...")
        return

    groups = groups_full[valid_idx]
    X_chest, X_left, X_right = X_chest_full[valid_idx], X_left_full[valid_idx], X_right_full[valid_idx]

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
    pbar = tqdm(folds, total=len(folds), desc=f"LOSO CV - {task_name} ({model_type})")

    task_results_file = config.RESULTS_DIR / f"results_{task_name}_{model_type.lower()}.json"

    for fold_number, (train_idx, test_idx) in enumerate(pbar, start=1):
        test_subject = groups[test_idx][0]
        fold_probs = {}

        for model_name, X_data in models_data.items():
            save_dir = config.CHECKPOINT_DIR / f"{task_name}_{model_type.lower()}" / model_name
            save_dir.mkdir(parents=True, exist_ok=True)

            if model_type in ["RF", "SVM", "KNN"]:
                metrics, y_prob = train_classical_model(
                    model_name, X_data[train_idx], X_data[test_idx], 
                    y[train_idx], y[test_idx], test_subject, fold_number, model_type
                )
            else:
                metrics, y_prob = train_dl_model(
                    model_name, X_data[train_idx], X_data[test_idx], 
                    y[train_idx], y[test_idx], test_subject, fold_number, save_dir, model_type
                )
            
            fold_probs[model_name] = y_prob
            metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
            results[model_name]["folds"].append(metrics)

        for fusion_name, sensors in late_fusions_map.items():
            late_probs = sum(fold_probs[s] for s in sensors) / len(sensors)
            
            if len(late_probs.shape) > 1 and late_probs.shape[1] > 1:
                late_preds = np.argmax(late_probs, axis=1)
            else:
                late_preds = (late_probs >= 0.5).astype(int)
            
            late_metrics = compute_metrics(y[test_idx], late_preds)
            late_metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
            results[fusion_name]["folds"].append(late_metrics)

        with open(task_results_file, "w") as f: 
            json.dump(results, f, indent=4)

    for model_name in results.keys():
        available_metrics = [k for k in results[model_name]["folds"][0].keys() if k not in ["fold", "test_subject"]]
        for metric in available_metrics:
            values = [x[metric] for x in results[model_name]["folds"]]
            results[model_name][f"{metric}_mean"] = float(np.mean(values))
            results[model_name][f"{metric}_std"] = float(np.std(values))

    with open(task_results_file, "w") as f: 
        json.dump(results, f, indent=4)
        
    print(f"\nPipeline concluído! Resultados salvos em: {task_results_file}")

if __name__ == "__main__":
    print("=== INICIANDO PIPELINE MULTI-TAREFA/MULTI-MODELO ===")
    
    print("Carregando Dados Sensores Base...")
    X_chest_full = np.load(config.DATASET_DIR / "X_chest.npy")
    X_left_full  = np.load(config.DATASET_DIR / "X_left.npy")
    X_right_full = np.load(config.DATASET_DIR / "X_right.npy")
    groups_full  = np.load(config.DATASET_DIR / "groups.npy")

    for schema, target_model in config.TASK_MODELS.items():
        train_task(
            task_name=schema, 
            model_type=target_model, # Pass specific model down
            X_chest_full=X_chest_full, 
            X_left_full=X_left_full, 
            X_right_full=X_right_full, 
            groups_full=groups_full
        )

    print("\nTODOS OS EXPERIMENTOS FORAM CONCLUÍDOS.")