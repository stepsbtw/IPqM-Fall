import json
import numpy as np
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import skew, kurtosis
import joblib
import config
import models

warnings.filterwarnings("ignore", category=RuntimeWarning)

def extract_handcrafted_features(X):
    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    max_vals = np.max(X, axis=1)
    min_vals = np.min(X, axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))

    X_safe = X + np.random.normal(0, 1e-8, X.shape)
    skewness = skew(X_safe, axis=1, bias=False)
    kurt = kurtosis(X_safe, axis=1, bias=False)

    return np.nan_to_num(
        np.concatenate([means, stds, max_vals, min_vals, rms, skewness, kurt], axis=1)
    )

def compute_metrics(y_true, y_pred):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    is_binary = len(unique_classes) <= 2 and set(unique_classes).issubset({0, 1})

    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}

    if is_binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        })
    else:
        metrics.update({
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        })
    return metrics

class DynamicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_dl_model(sensor_name, X_train, X_test, y_train, y_test, test_subject, fold_number, save_dir, model_type):
    ckpt_path = save_dir / f"{sensor_name}_fold_{fold_number}_subject_{test_subject}.pth"
    
    train_mean = X_train.mean(axis=(0, 1), keepdims=True)
    train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    np.save(save_dir / f"{sensor_name}_fold_{fold_number}_mean.npy", train_mean)
    np.save(save_dir / f"{sensor_name}_fold_{fold_number}_std.npy", train_std)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    num_classes = int(np.max(y_train)) + 1

    if model_type in ["CNN1Conv", "DeepConvLSTM"]:
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)
        bs = 256
        if model_type == "CNN1Conv":
            model = models.CNN1Conv(X_train.shape[1], num_classes).to(config.DEVICE)
        else:
            model = models.DeepConvLSTM(X_train.shape[1], num_classes).to(config.DEVICE)
    elif model_type == "LSTM":
        bs = 512
        model = models.LSTMModel(X_train.shape[2], num_classes).to(config.DEVICE)
    elif model_type == "MLP":
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        bs = len(X_train)
        model = models.MLP(X_train.shape[1], num_classes).to(config.DEVICE)

    if torch.cuda.device_count() > 1 and config.DEVICE.type == "cuda":
        model = nn.DataParallel(model)

    train_loader = DataLoader(DynamicDataset(X_train, y_train), batch_size=bs, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(DynamicDataset(X_test, y_test), batch_size=bs, shuffle=False)

    class_counts = np.array([np.sum(y_train == c) for c in range(num_classes)])
    class_weights = torch.tensor(len(y_train) / (num_classes * (class_counts + 1e-6)),
                                 dtype=torch.float32, device=config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    start_epoch = 0
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

    epoch_pbar = tqdm(range(start_epoch, config.EPOCHS), desc=f"{sensor_name} Fold {fold_number}", leave=False)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(config.DEVICE), y_b.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_pbar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}")

        torch.save({
            "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "sensor_name": sensor_name, "fold": fold_number, "model_type": model_type
        }, ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(config.DEVICE)
            outputs = model(X_b)
            probs = F.softmax(outputs, dim=1)
            y_true.extend(y_b.numpy())
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return compute_metrics(y_true, y_pred), np.array(y_prob)


def train_classical_model(sensor_name, X_train, X_test, y_train, y_test, test_subject, fold_number, model_type):
    X_train_feat = extract_handcrafted_features(X_train)
    X_test_feat = extract_handcrafted_features(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    model = models.get_classical_model()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    return compute_metrics(y_test, y_pred), y_prob


def train_single_task(task_name, model_type, X_chest_full, X_left_full, X_right_full, groups_full):
    print(f"\n{'='*50}")
    print(f"=== INICIANDO TAREFA ISOLADA: {task_name.upper()} ({model_type}) ===")
    print(f"{'='*50}")
    
    torch.backends.cudnn.benchmark = True if model_type in ["CNN1Conv", "DeepConvLSTM", "LSTM", "MLP"] else False

    y_full = np.load(config.DATASET_DIR / f"{task_name}.npy")
    valid_idx = y_full != -1
    y = y_full[valid_idx]
    
    if len(y) == 0:
        print(f"Nenhuma instância válida para {task_name}. Pulando...")
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
        "ENSEMBLE_CHEST_LEFT": ["CHEST", "LEFT"], "ENSEMBLE_CHEST_RIGHT": ["CHEST", "RIGHT"],
        "ENSEMBLE_LEFT_RIGHT": ["LEFT", "RIGHT"], "ENSEMBLE_CHEST_LEFT_RIGHT": ["CHEST", "LEFT", "RIGHT"]
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
                metrics, y_prob = train_classical_model(model_name, X_data[train_idx], X_data[test_idx], y[train_idx], y[test_idx], test_subject, fold_number, model_type)
            else:
                metrics, y_prob = train_dl_model(model_name, X_data[train_idx], X_data[test_idx], y[train_idx], y[test_idx], test_subject, fold_number, save_dir, model_type)
            
            fold_probs[model_name] = y_prob
            metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
            results[model_name]["folds"].append(metrics)

        for fusion_name, sensors in late_fusions_map.items():
            late_probs = sum(fold_probs[s] for s in sensors) / len(sensors)
            late_preds = np.argmax(late_probs, axis=1) if len(late_probs.shape) > 1 and late_probs.shape[1] > 1 else (late_probs >= 0.5).astype(int)
            late_metrics = compute_metrics(y[test_idx], late_preds)
            late_metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
            results[fusion_name]["folds"].append(late_metrics)

        with open(task_results_file, "w") as f: json.dump(results, f, indent=4)

    # Save summary results
    for model_name in results.keys():
        available_metrics = [k for k in results[model_name]["folds"][0].keys() if k not in ["fold", "test_subject"]]
        for metric in available_metrics:
            values = [x[metric] for x in results[model_name]["folds"]]
            results[model_name][f"{metric}_mean"] = float(np.mean(values))
            results[model_name][f"{metric}_std"] = float(np.std(values))
    with open(task_results_file, "w") as f: json.dump(results, f, indent=4)

    # Final Training
    print("\nTraining FINAL models on all 15 subjects...\n")
    for model_name, X_data in models_data.items():
        final_dir = config.CHECKPOINT_DIR / f"{task_name}_{model_type.lower()}" / "FINAL" / model_name
        if model_type in ["RF", "SVM", "KNN"]:
            X_feat = extract_handcrafted_features(X_data)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_feat)
            model = models.get_classical_model()
            model.fit(X_scaled, y)
            joblib.dump(model, final_dir / "final_model.pkl")
            joblib.dump(scaler, final_dir / "scaler.pkl")
        else:
            train_final_single_model(model_name, X_data, y, model_type, final_dir)
    print("FINAL models saved.")

def custom_multitask_loss(predictions, targets, criteria, active_tasks):
    total_loss = 0.0
    for task_name in active_tasks:
        pred = predictions[task_name]
        targ = targets[task_name]
        valid_indices = (targ >= 0)
        if valid_indices.sum() > 0:
            task_loss = criteria[task_name](pred[valid_indices], targ[valid_indices])
            lambda_weight = config.MULTI_TASK_WEIGHTS[task_name]
            total_loss += lambda_weight * task_loss
    return total_loss

def run_multitask(mode, X_chest_full, X_left_full, X_right_full, groups_full):
    print(f"\n{'='*60}")
    print(f"=== INICIANDO MULTI-TASK & FUSÃO DE SENSORES ({mode}) ===")
    print(f"Arquitetura Base: {config.MULTI_TASK_MODEL}")
    print(f"{'='*60}")

    if mode == "TRIPLE":
        active_tasks = ["fall", "posture", "movement"]
        num_classes = {"fall": 2, "posture": 4, "movement": 5}
    elif mode == "DOUBLE_FP":
        active_tasks = ["fall", "posture"]
        num_classes = {"fall": 2, "posture": 4}
    elif mode == "DOUBLE_PM":
        active_tasks = ["posture", "movement"]
        num_classes = {"posture": 4, "movement": 5}
    else:
        raise ValueError("Modo Multitarefa Inválido.")

    y_targets = {}
    if "fall" in active_tasks: y_targets["fall"] = np.load(config.DATASET_DIR / "y_detect_fall.npy")
    if "posture" in active_tasks: y_targets["posture"] = np.load(config.DATASET_DIR / "y_classify_posture.npy")
    if "movement" in active_tasks: y_targets["movement"] = np.load(config.DATASET_DIR / "y_classify_movement.npy")

    models_data = {
        "CHEST": X_chest_full, "LEFT": X_left_full, "RIGHT": X_right_full,
        "CHEST_LEFT": np.concatenate((X_chest_full, X_left_full), axis=2),
        "CHEST_RIGHT": np.concatenate((X_chest_full, X_right_full), axis=2),
        "LEFT_RIGHT": np.concatenate((X_left_full, X_right_full), axis=2),
        "CHEST_LEFT_RIGHT": np.concatenate((X_chest_full, X_left_full, X_right_full), axis=2)
    }

    late_fusions_map = {
        "ENSEMBLE_CHEST_LEFT": ["CHEST", "LEFT"], "ENSEMBLE_CHEST_RIGHT": ["CHEST", "RIGHT"],
        "ENSEMBLE_LEFT_RIGHT": ["LEFT", "RIGHT"], "ENSEMBLE_CHEST_LEFT_RIGHT": ["CHEST", "LEFT", "RIGHT"]
    }

    results = {m_name: {task: {"folds": []} for task in active_tasks} for m_name in list(models_data.keys()) + list(late_fusions_map.keys())}
    task_results_file = config.RESULTS_DIR / f"results_multitask_{mode}_{config.MULTI_TASK_MODEL.lower()}.json"

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_chest_full, groups_full, groups_full))
    pbar = tqdm(folds, total=len(folds), desc=f"LOSO CV - Multitask {mode}")

    for fold_number, (train_idx, test_idx) in enumerate(pbar, start=1):
        test_subject = groups_full[test_idx][0]
        fold_probs = {m_name: {task: [] for task in active_tasks} for m_name in models_data.keys()}
        y_test_true = {task: y_targets[task][test_idx] for task in active_tasks}

        for model_name, X_data in models_data.items():
            save_dir = config.CHECKPOINT_DIR / f"multitask_{mode}_{config.MULTI_TASK_MODEL.lower()}" / model_name
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / f"{model_name}_fold_{fold_number}_subject_{test_subject}.pth"
            
            X_train_raw, X_test_raw = X_data[train_idx], X_data[test_idx]

            # Scaling
            train_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
            train_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8
            np.save(save_dir / f"{model_name}_fold_{fold_number}_mean.npy", train_mean)
            np.save(save_dir / f"{model_name}_fold_{fold_number}_std.npy", train_std)

            X_train_scaled = (X_train_raw - train_mean) / train_std
            X_test_scaled = (X_test_raw - train_mean) / train_std

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

            y_train_tensors = {t: torch.tensor(y_targets[t][train_idx], dtype=torch.long) for t in active_tasks}
            y_test_tensors = {t: torch.tensor(y_targets[t][test_idx], dtype=torch.long) for t in active_tasks}

            train_loader = DataLoader(TensorDataset(X_train_tensor, *[y_train_tensors[t] for t in active_tasks]), batch_size=256, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test_tensor, *[y_test_tensors[t] for t in active_tasks]), batch_size=256, shuffle=False)

            num_features = X_train_tensor.shape[1]
            
            if config.MULTI_TASK_MODEL == "CNN1Conv":
                model = models.CNN1Conv_MultiTask(num_features, num_classes).to(config.DEVICE)
            elif config.MULTI_TASK_MODEL == "DeepConvLSTM":
                model = models.DeepConvLSTM_MultiTask(num_features, num_classes).to(config.DEVICE)
            elif config.MULTI_TASK_MODEL == "LSTM":
                model = models.LSTMModel_MultiTask(num_features, num_classes).to(config.DEVICE)
            else:
                raise ValueError(f"Modelo Multitarefa {config.MULTI_TASK_MODEL} não reconhecido.")

            criteria = {}
            for task in active_tasks:
                valid_labels = y_targets[task][train_idx]
                valid_labels = valid_labels[valid_labels >= 0]
                n_classes = num_classes[task]
                class_counts = np.array([np.sum(valid_labels == c) for c in range(n_classes)])
                class_weights = torch.tensor(len(valid_labels) / (n_classes * (class_counts + 1e-6)), dtype=torch.float32, device=config.DEVICE)
                criteria[task] = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

            model.train()
            epoch_pbar = tqdm(range(config.EPOCHS), desc=f"Training {model_name}", leave=False)
            for epoch in epoch_pbar:
                for data in train_loader:
                    b_x = data[0].to(config.DEVICE)
                    b_y = {task: data[i + 1].to(config.DEVICE) for i, task in enumerate(active_tasks)}
                    
                    optimizer.zero_grad()
                    predictions = model(b_x)
                    loss = custom_multitask_loss(predictions, b_y, criteria, active_tasks)
                    if loss > 0:
                        loss.backward()
                        optimizer.step()
                epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")

            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "fold": fold_number, "sensor_name": model_name}, ckpt_path)

            model.eval()
            y_prob_collector = {task: [] for task in active_tasks}
            with torch.no_grad():
                for data in test_loader:
                    b_x = data[0].to(config.DEVICE)
                    outputs = model(b_x)
                    for task in active_tasks:
                        probs = F.softmax(outputs[task], dim=1).cpu().numpy()
                        y_prob_collector[task].extend(probs)
            
            for task in active_tasks:
                probs_array = np.array(y_prob_collector[task])
                fold_probs[model_name][task] = probs_array 
                preds = np.argmax(probs_array, axis=1)
                true_labels = y_test_true[task]
                
                valid_idx = (true_labels >= 0)
                if valid_idx.sum() > 0:
                    metrics = compute_metrics(true_labels[valid_idx], preds[valid_idx])
                else:
                    metrics = {"accuracy": 0.0, "f1_macro": 0.0}
                    
                metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
                results[model_name][task]["folds"].append(metrics)

        for fusion_name, sensors in late_fusions_map.items():
            for task in active_tasks:
                late_probs = sum(fold_probs[s][task] for s in sensors) / len(sensors)
                late_preds = np.argmax(late_probs, axis=1)
                true_labels = y_test_true[task]
                valid_idx = (true_labels >= 0)
                
                if valid_idx.sum() > 0:
                    late_metrics = compute_metrics(true_labels[valid_idx], late_preds[valid_idx])
                else:
                    late_metrics = {"accuracy": 0.0, "f1_macro": 0.0}
                    
                late_metrics.update({"fold": fold_number, "test_subject": str(test_subject)})
                results[fusion_name][task]["folds"].append(late_metrics)

        with open(task_results_file, "w") as f: json.dump(results, f, indent=4)

    for model_name in results.keys():
        for task in active_tasks:
            folds_data = results[model_name][task]["folds"]
            if len(folds_data) > 0:
                acc_values = [x.get("accuracy", 0.0) for x in folds_data]
                f1_key = "f1" if "f1" in folds_data[0] else "f1_macro"
                f1_values = [x.get(f1_key, 0.0) for x in folds_data]
                results[model_name][task]["accuracy_mean"] = float(np.mean(acc_values))
                results[model_name][task]["accuracy_std"] = float(np.std(acc_values))
                results[model_name][task][f"{f1_key}_mean"] = float(np.mean(f1_values))
                results[model_name][task][f"{f1_key}_std"] = float(np.std(f1_values))

    with open(task_results_file, "w") as f: json.dump(results, f, indent=4)
    print("\nTraining FINAL multitask models on all 15 subjects...\n")

    for model_name, X_data in models_data.items():
        final_dir = config.CHECKPOINT_DIR / f"multitask_{mode}_{config.MULTI_TASK_MODEL.lower()}" / "FINAL" / model_name
        train_final_multitask_model(
            sensor_name=model_name,
            X=X_data,
            y_targets=y_targets,
            active_tasks=active_tasks,
            num_classes=num_classes,
            save_dir=final_dir
        )

    print("FINAL multitask models saved.")

def train_final_single_model(sensor_name, X, y, model_type, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    train_mean = X.mean(axis=(0, 1), keepdims=True)
    train_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    np.save(save_dir / "mean.npy", train_mean)
    np.save(save_dir / "std.npy", train_std)

    X = (X - train_mean) / train_std
    num_classes = int(np.max(y)) + 1

    # Initialize model
    if model_type in ["CNN1Conv", "DeepConvLSTM"]:
        X = X.transpose(0, 2, 1)
        model = models.CNN1Conv(X.shape[1], num_classes).to(config.DEVICE) if model_type == "CNN1Conv" else models.DeepConvLSTM(X.shape[1], num_classes).to(config.DEVICE)
        bs = 256
    elif model_type == "LSTM":
        model = models.LSTMModel(X.shape[2], num_classes).to(config.DEVICE)
        bs = 512
    elif model_type == "MLP":
        X = X.reshape(X.shape[0], -1)
        model = models.MLP(X.shape[1], num_classes).to(config.DEVICE)
        bs = len(X)
    else:
        raise ValueError(model_type)

    loader = DataLoader(DynamicDataset(X, y), batch_size=bs, shuffle=True, 
                        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # Setup Criteria
    class_counts = np.array([np.sum(y == c) for c in range(num_classes)])
    class_weights = torch.tensor(len(y) / (num_classes * (class_counts + 1e-6)), 
                                 dtype=torch.float32, device=config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for epoch in tqdm(range(config.EPOCHS), desc=f"FINAL {sensor_name}"):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(config.DEVICE), y_b.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

    torch.save({"model_state": model.state_dict(), "model_type": model_type, "num_classes": num_classes}, 
               save_dir / "final_model.pth")

def train_final_multitask_model(sensor_name, X, y_targets, active_tasks, num_classes, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    train_mean = X.mean(axis=(0, 1), keepdims=True)
    train_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    np.save(save_dir / "mean.npy", train_mean)
    np.save(save_dir / "std.npy", train_std)

    X = (X - train_mean) / train_std
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)

    y_tensors = {task: torch.tensor(y_targets[task], dtype=torch.long) for task in active_tasks}
    loader = DataLoader(TensorDataset(X_tensor, *[y_tensors[t] for t in active_tasks]), batch_size=256, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    criteria = {}
    for task in active_tasks:
        n_c = num_classes[task]
        labels = y_targets[task]
        class_counts = np.array([np.sum(labels == c) for c in range(n_c)])
        weights = torch.tensor(len(labels) / (n_c * (class_counts + 1e-6)), dtype=torch.float32, device=config.DEVICE)
        criteria[task] = nn.CrossEntropyLoss(weight=weights)

    # Select Model
    num_features = X_tensor.shape[1]
    if config.MULTI_TASK_MODEL == "CNN1Conv":
        model = models.CNN1Conv_MultiTask(num_features, num_classes).to(config.DEVICE)
    elif config.MULTI_TASK_MODEL == "DeepConvLSTM":
        model = models.DeepConvLSTM_MultiTask(num_features, num_classes).to(config.DEVICE)
    elif config.MULTI_TASK_MODEL == "LSTM":
        model = models.LSTMModel_MultiTask(num_features, num_classes).to(config.DEVICE)
    else:
        raise ValueError(config.MULTI_TASK_MODEL)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for epoch in tqdm(range(config.EPOCHS), desc=f"FINAL {sensor_name}"):
        for data in loader:
            b_x = data[0].to(config.DEVICE)
            b_y = {task: data[i + 1].to(config.DEVICE) for i, task in enumerate(active_tasks)}
            
            optimizer.zero_grad()
            predictions = model(b_x)
            loss = custom_multitask_loss(predictions, b_y, criteria, active_tasks)
            loss.backward()
            optimizer.step()

    torch.save({"model_state": model.state_dict(), "active_tasks": active_tasks, "num_classes": num_classes}, 
               save_dir / "final_model.pth")