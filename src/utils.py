import json
import warnings

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
import models

warnings.filterwarnings("ignore", category=RuntimeWarning)

CLASSICAL_MODELS = {"RF", "SVM", "KNN"}
CONV_MODELS = {"CNN1Conv", "CNN3B3Conv", "DeepConvLSTM"}
DL_MODELS = CONV_MODELS | {"LSTM", "MLP"}
LATE_FUSIONS = {
    "ENSEMBLE_CHEST_LEFT": ["CHEST", "LEFT"],
    "ENSEMBLE_CHEST_RIGHT": ["CHEST", "RIGHT"],
    "ENSEMBLE_LEFT_RIGHT": ["LEFT", "RIGHT"],
    "ENSEMBLE_CHEST_LEFT_RIGHT": ["CHEST", "LEFT", "RIGHT"],
}
MULTITASK_CONFIGS = {
    "FALL_DETECT_POSTURE_MOVEMENT": {"fall": 2, "posture": 4, "movement": 5},
    "FALL_DETECT_POSTURE": {"fall": 2, "posture": 4},
    "POSTURE_MOVEMENT": {"posture": 4, "movement": 5},
    "FALL_CLASSIFY_POSTURE_MOVEMENT": {"fall_classify": 4, "posture": 4, "movement": 5},
    "FALL_CLASSIFY_POSTURE": {"fall_classify": 4, "posture": 4},
}
TARGET_FILES = {
    "fall": "y_detect_fall.npy",
    "fall_classify": "y_classify_fall.npy",
    "posture": "y_classify_posture.npy",
    "movement": "y_classify_movement.npy",
}


def extract_handcrafted_features(X):
    stats = [
        np.mean(X, axis=1), np.std(X, axis=1), np.max(X, axis=1),
        np.min(X, axis=1), np.sqrt(np.mean(X**2, axis=1)),
    ]
    X_safe = X + np.random.normal(0, 1e-8, X.shape)
    stats += [skew(X_safe, axis=1, bias=False), kurtosis(X_safe, axis=1, bias=False)]
    return np.nan_to_num(np.concatenate(stats, axis=1))


def compute_metrics(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    binary = len(classes) <= 2 and set(classes).issubset({0, 1})
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        result.update({
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    else:
        result.update({
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        })
    return result




def compute_fixed_multiclass_metrics(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def compute_unified_mapped_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results = {}

    # Preserve the existing task coding: 0 = Fall, 1 = Non-Fall.
    fd_true = np.where(y_true <= 3, 0, 1)
    fd_pred = np.where(y_pred <= 3, 0, 1)
    results["fall_detection"] = compute_metrics(fd_true, fd_pred)

    specs = {
        "fall_type": ([0, 1, 2, 3], 0, 4),
        "posture": ([4, 5, 6, 7], 4, 4),
        "movement": ([8, 9, 10, 11, 12], 8, 5),
    }
    for task, (ids, offset, n_classes) in specs.items():
        mask = np.isin(y_true, ids)
        task_true = y_true[mask] - offset
        raw_pred = y_pred[mask]
        task_pred = np.where(np.isin(raw_pred, ids), raw_pred - offset, -1)
        metrics = compute_fixed_multiclass_metrics(task_true, task_pred, range(n_classes))
        metrics["valid_samples"] = int(mask.sum())
        results[task] = metrics
    return results


def sensor_sets(chest, left, right):
    return {
        "CHEST": chest,
        "LEFT": left,
        "RIGHT": right,
        "CHEST_LEFT": np.concatenate((chest, left), axis=2),
        "CHEST_RIGHT": np.concatenate((chest, right), axis=2),
        "LEFT_RIGHT": np.concatenate((left, right), axis=2),
        "CHEST_LEFT_RIGHT": np.concatenate((chest, left, right), axis=2),
    }


def prepare_data(X_train, model_type, save_dir, X_test=None, prefix=""):
    save_dir.mkdir(parents=True, exist_ok=True)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    np.save(save_dir / f"{prefix}mean.npy", mean)
    np.save(save_dir / f"{prefix}std.npy", std)

    X_train = (X_train - mean) / std
    X_test = None if X_test is None else (X_test - mean) / std

    if model_type in CONV_MODELS:
        X_train = X_train.transpose(0, 2, 1)
        X_test = None if X_test is None else X_test.transpose(0, 2, 1)
    elif model_type == "MLP":
        X_train = X_train.reshape(len(X_train), -1)
        X_test = None if X_test is None else X_test.reshape(len(X_test), -1)
    elif model_type != "LSTM":
        raise ValueError(f"Modelo {model_type} não reconhecido.")
    return X_train, X_test


def create_model(model_type, X, num_classes, multitask=False):
    suffix = "_MultiTask" if multitask else ""
    class_name = "LSTMModel" if model_type == "LSTM" else model_type
    model_class = getattr(models, class_name + suffix, None)
    if model_class is None:
        raise ValueError(f"Modelo {'multitarefa ' if multitask else ''}{model_type} não reconhecido.")
    num_features = X.shape[2] if model_type == "LSTM" else X.shape[1]
    return model_class(num_features, num_classes).to(config.DEVICE)


def class_weights(labels, num_classes):
    labels = np.asarray(labels)
    labels = labels[labels >= 0]
    if len(labels) == 0:
        raise ValueError("Não há rótulos válidos para calcular os pesos das classes.")
    counts = np.array([np.sum(labels == c) for c in range(num_classes)])
    weights = len(labels) / (num_classes * (counts + 1e-6))
    return torch.tensor(weights, dtype=torch.float32, device=config.DEVICE)


def single_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate_single(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in loader:
            probs = F.softmax(model(X.to(config.DEVICE)), dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(probs.argmax(dim=1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return compute_metrics(y_true, y_pred), np.asarray(y_prob)


def multitask_loss(predictions, targets, criteria, tasks):
    total = None
    for task in tasks:
        valid = targets[task] >= 0
        if valid.sum() > 0:
            loss = config.MULTI_TASK_WEIGHTS[task] * criteria[task](predictions[task][valid], targets[task][valid])
            total = loss if total is None else total + loss
    if total is None:
        return torch.tensor(0.0, device=next(iter(predictions.values())).device)
    return total


def train_multitask(model, loader, criteria, tasks, optimizer, description, leave=False):
    progress = tqdm(range(config.EPOCHS), desc=description, leave=leave)
    for _ in progress:
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            X = batch[0].to(config.DEVICE)
            targets = {task: batch[i + 1].to(config.DEVICE) for i, task in enumerate(tasks)}
            optimizer.zero_grad()
            loss = multitask_loss(model(X), targets, criteria, tasks)
            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
        progress.set_postfix(loss=f"{total / batches if batches else 0.0:.4f}")


def evaluate_multitask(model, loader, tasks):
    model.eval()
    probabilities = {task: [] for task in tasks}
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch[0].to(config.DEVICE))
            for task in tasks:
                probabilities[task].extend(F.softmax(outputs[task], dim=1).cpu().numpy())
    return {task: np.asarray(values) for task, values in probabilities.items()}


def make_loader(X, targets, batch_size, shuffle, workers=False):
    tensors = [torch.tensor(X, dtype=torch.float32)]
    if isinstance(targets, dict):
        tensors += [torch.tensor(y, dtype=torch.long) for y in targets.values()]
    else:
        tensors.append(torch.tensor(targets, dtype=torch.long))
    kwargs = {}
    if workers:
        kwargs = {"num_workers": config.NUM_WORKERS, "pin_memory": config.PIN_MEMORY}
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, **kwargs)


def summarize(entry):
    if not entry["folds"]:
        return
    for metric in (k for k in entry["folds"][0] if k not in {"fold", "test_subject"}):
        values = [fold[metric] for fold in entry["folds"]]
        entry[f"{metric}_mean"] = float(np.mean(values))
        entry[f"{metric}_std"] = float(np.std(values))


def train_classical(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(extract_handcrafted_features(X_train))
    X_test = scaler.transform(extract_handcrafted_features(X_test))
    model = models.get_classical_model()
    model.fit(X_train, y_train)
    return compute_metrics(y_test, model.predict(X_test)), model.predict_proba(X_test)


def train_single_task(task_name, model_type, X_chest_full, X_left_full, X_right_full, groups_full):
    print(f"\n{'=' * 50}\n=== INICIANDO TAREFA ISOLADA: {task_name.upper()} ({model_type}) ===\n{'=' * 50}")
    torch.backends.cudnn.benchmark = model_type in DL_MODELS

    y_full = np.load(config.WINDOWED_DATASET_DIR / f"{task_name}.npy")
    valid = y_full != -1
    y = y_full[valid]
    if len(y) == 0:
        print(f"Nenhuma instância válida para {task_name}. Pulando...")
        return

    groups = groups_full[valid]
    datasets = sensor_sets(X_chest_full[valid], X_left_full[valid], X_right_full[valid])
    results = {name: {"folds": []} for name in [*datasets, *LATE_FUSIONS]}
    result_file = config.RESULTS_DIR / f"results_{task_name}_{model_type.lower()}.json"
    folds = LeaveOneGroupOut().split(datasets["CHEST"], y, groups)

    for fold, (train_idx, test_idx) in enumerate(tqdm(list(folds), desc=f"LOSO CV - {task_name} ({model_type})"), 1):
        subject = groups[test_idx][0]
        fold_probs = {}
        for name, X in datasets.items():
            save_dir = config.CHECKPOINT_DIR / f"{task_name}_{model_type.lower()}" / name
            save_dir.mkdir(parents=True, exist_ok=True)

            if model_type in CLASSICAL_MODELS:
                metrics, probs = train_classical(X[train_idx], X[test_idx], y[train_idx], y[test_idx])
            else:
                ckpt = save_dir / f"{name}_fold_{fold}_subject_{subject}.pth"
                X_train, X_test = prepare_data(X[train_idx], model_type, save_dir, X[test_idx], f"{name}_fold_{fold}_")
                n_classes = int(np.max(y[train_idx])) + 1
                model = create_model(model_type, X_train, n_classes)
                if torch.cuda.device_count() > 1 and config.DEVICE.type == "cuda":
                    model = nn.DataParallel(model)
                train_loader = make_loader(X_train, y[train_idx], config.BATCH_SIZE, True, workers=True)
                test_loader = make_loader(X_test, y[test_idx], config.BATCH_SIZE, False)
                criterion = nn.CrossEntropyLoss(weight=class_weights(y[train_idx], n_classes))
                optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

                start = 0
                if ckpt.exists():
                    state = torch.load(ckpt, map_location=config.DEVICE)
                    (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state["model_state"])
                    optimizer.load_state_dict(state["optimizer_state"])
                    start = state["epoch"] + 1

                progress = tqdm(range(start, config.EPOCHS), desc=f"{name} Fold {fold}", leave=False)
                for epoch in progress:
                    progress.set_postfix(loss=f"{single_epoch(model, train_loader, criterion, optimizer):.4f}")
                    torch.save({
                        "model_state": (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                        "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                        "sensor_name": name, "fold": fold, "model_type": model_type,
                    }, ckpt)
                state = torch.load(ckpt, map_location=config.DEVICE)
                (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state["model_state"])
                metrics, probs = evaluate_single(model, test_loader)

            fold_probs[name] = probs
            metrics.update({"fold": fold, "test_subject": str(subject)})
            results[name]["folds"].append(metrics)

        for fusion, sensors in LATE_FUSIONS.items():
            probs = sum(fold_probs[s] for s in sensors) / len(sensors)
            preds = np.argmax(probs, axis=1) if probs.ndim > 1 and probs.shape[1] > 1 else (probs >= 0.5).astype(int)
            metrics = compute_metrics(y[test_idx], preds)
            metrics.update({"fold": fold, "test_subject": str(subject)})
            results[fusion]["folds"].append(metrics)
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)

    for entry in results.values():
        summarize(entry)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nTraining FINAL models on all 15 subjects...\n")
    for name, X in datasets.items():
        final_dir = config.CHECKPOINT_DIR / f"{task_name}_{model_type.lower()}" / "FINAL" / name
        final_dir.mkdir(parents=True, exist_ok=True)
        if model_type in CLASSICAL_MODELS:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(extract_handcrafted_features(X))
            model = models.get_classical_model()
            model.fit(X_scaled, y)
            joblib.dump(model, final_dir / "final_model.pkl")
            joblib.dump(scaler, final_dir / "scaler.pkl")
            continue

        X_train, _ = prepare_data(X, model_type, final_dir)
        n_classes = int(np.max(y)) + 1
        model = create_model(model_type, X_train, n_classes)
        loader = make_loader(X_train, y, config.BATCH_SIZE, True, workers=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights(y, n_classes))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        for _ in tqdm(range(config.EPOCHS), desc=f"FINAL {name}"):
            single_epoch(model, loader, criterion, optimizer)
        torch.save({"model_state": model.state_dict(), "model_type": model_type, "num_classes": n_classes}, final_dir / "final_model.pth")
    print("FINAL models saved.")

def train_unified_model(model_type, X_chest_full, X_left_full, X_right_full, groups_full):
    print(f"\n{'=' * 70}\n=== UNIFIED NON-TASK BASELINE: 13 CLASSES ({model_type}) ===\n{'=' * 70}")
    torch.backends.cudnn.benchmark = model_type in DL_MODELS

    target_path = config.WINDOWED_DATASET_DIR / config.UNIFIED_TARGET
    if not target_path.exists():
        raise FileNotFoundError(f"Unified target not found: {target_path}")

    y_full = np.load(target_path)
    if not (len(y_full) == len(groups_full) == len(X_chest_full) == len(X_left_full) == len(X_right_full)):
        raise ValueError("X, groups, and y_unified must have the same length.")

    valid = y_full >= 0
    y = y_full[valid].astype(np.int64)
    groups = groups_full[valid]
    datasets = sensor_sets(X_chest_full[valid], X_left_full[valid], X_right_full[valid])

    observed = set(np.unique(y).tolist())
    expected = set(range(config.UNIFIED_NUM_CLASSES))
    if observed != expected:
        raise ValueError(f"Expected unified classes 0..12, observed {sorted(observed)}")

    print(f"Valid unified samples: {len(y)}")
    print(f"Ignored transition samples: {int((~valid).sum())}")

    names = [*datasets, *LATE_FUSIONS]
    results = {
        name: {
            "native": {"folds": []},
            "mapped": {
                "fall_detection": {"folds": []},
                "fall_type": {"folds": []},
                "posture": {"folds": []},
                "movement": {"folds": []},
            },
        }
        for name in names
    }
    result_file = config.RESULTS_DIR / f"results_y_unified_{model_type.lower()}.json"
    folds = list(LeaveOneGroupOut().split(datasets["CHEST"], y, groups))

    for fold, (train_idx, test_idx) in enumerate(tqdm(folds, desc=f"LOSO CV - y_unified ({model_type})"), 1):
        subject = groups[test_idx][0]
        fold_probs = {}
        train_classes = set(np.unique(y[train_idx]).tolist())
        if train_classes != expected:
            raise ValueError(f"Fold {fold} training partition missing classes: {sorted(expected - train_classes)}")

        for name, X in datasets.items():
            save_dir = config.CHECKPOINT_DIR / f"y_unified_{model_type.lower()}" / name
            save_dir.mkdir(parents=True, exist_ok=True)

            if model_type in CLASSICAL_MODELS:
                native, probs = train_classical(X[train_idx], X[test_idx], y[train_idx], y[test_idx])
            else:
                ckpt = save_dir / f"{name}_fold_{fold}_subject_{subject}.pth"
                X_train, X_test = prepare_data(X[train_idx], model_type, save_dir, X[test_idx], f"{name}_fold_{fold}_")
                model = create_model(model_type, X_train, config.UNIFIED_NUM_CLASSES)
                if torch.cuda.device_count() > 1 and config.DEVICE.type == "cuda":
                    model = nn.DataParallel(model)
                train_loader = make_loader(X_train, y[train_idx], config.BATCH_SIZE, True, workers=True)
                test_loader = make_loader(X_test, y[test_idx], config.BATCH_SIZE, False)
                criterion = nn.CrossEntropyLoss(weight=class_weights(y[train_idx], config.UNIFIED_NUM_CLASSES))
                optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

                start = 0
                if ckpt.exists():
                    state = torch.load(ckpt, map_location=config.DEVICE)
                    base = model.module if isinstance(model, nn.DataParallel) else model
                    base.load_state_dict(state["model_state"])
                    optimizer.load_state_dict(state["optimizer_state"])
                    start = state["epoch"] + 1

                progress = tqdm(range(start, config.EPOCHS), desc=f"{name} Fold {fold}", leave=False)
                for epoch in progress:
                    loss = single_epoch(model, train_loader, criterion, optimizer)
                    progress.set_postfix(loss=f"{loss:.4f}")
                    base = model.module if isinstance(model, nn.DataParallel) else model
                    torch.save({
                        "model_state": base.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "sensor_name": name,
                        "fold": fold,
                        "model_type": model_type,
                        "num_classes": config.UNIFIED_NUM_CLASSES,
                        "target": "y_unified",
                    }, ckpt)

                state = torch.load(ckpt, map_location=config.DEVICE)
                base = model.module if isinstance(model, nn.DataParallel) else model
                base.load_state_dict(state["model_state"])
                native, probs = evaluate_single(model, test_loader)

            preds = np.argmax(probs, axis=1)
            fold_probs[name] = probs
            native.update({"fold": fold, "test_subject": str(subject), "valid_samples": int(len(test_idx))})
            results[name]["native"]["folds"].append(native)
            mapped = compute_unified_mapped_metrics(y[test_idx], preds)
            for task, metrics in mapped.items():
                metrics.update({"fold": fold, "test_subject": str(subject)})
                results[name]["mapped"][task]["folds"].append(metrics)

        for fusion, sensors in LATE_FUSIONS.items():
            probs = sum(fold_probs[s] for s in sensors) / len(sensors)
            preds = np.argmax(probs, axis=1)
            native = compute_metrics(y[test_idx], preds)
            native.update({"fold": fold, "test_subject": str(subject), "valid_samples": int(len(test_idx))})
            results[fusion]["native"]["folds"].append(native)
            mapped = compute_unified_mapped_metrics(y[test_idx], preds)
            for task, metrics in mapped.items():
                metrics.update({"fold": fold, "test_subject": str(subject)})
                results[fusion]["mapped"][task]["folds"].append(metrics)

        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)

    for cfg_results in results.values():
        summarize(cfg_results["native"])
        for task_results in cfg_results["mapped"].values():
            summarize(task_results)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nTraining FINAL unified models on all valid samples...\n")
    for name, X in datasets.items():
        final_dir = config.CHECKPOINT_DIR / f"y_unified_{model_type.lower()}" / "FINAL" / name
        final_dir.mkdir(parents=True, exist_ok=True)
        X_train, _ = prepare_data(X, model_type, final_dir)
        model = create_model(model_type, X_train, config.UNIFIED_NUM_CLASSES)
        loader = make_loader(X_train, y, config.BATCH_SIZE, True, workers=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights(y, config.UNIFIED_NUM_CLASSES))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        for _ in tqdm(range(config.EPOCHS), desc=f"FINAL UNIFIED {name}"):
            single_epoch(model, loader, criterion, optimizer)
        torch.save({
            "model_state": model.state_dict(),
            "model_type": model_type,
            "num_classes": config.UNIFIED_NUM_CLASSES,
            "target": "y_unified",
        }, final_dir / "final_model.pth")

    print(f"Unified results saved to: {result_file}")


def run_multitask(mode, X_chest_full, X_left_full, X_right_full, groups_full):
    print(f"\n{'=' * 60}\n=== INICIANDO MULTI-TASK & FUSÃO DE SENSORES ({mode}) ===\nArquitetura Base: {config.MULTI_TASK_MODEL}\n{'=' * 60}")
    if mode not in MULTITASK_CONFIGS:
        raise ValueError(f"Modo Multitarefa Inválido: {mode}")

    n_classes = MULTITASK_CONFIGS[mode]
    tasks = list(n_classes)
    targets = {task: np.load(config.WINDOWED_DATASET_DIR / TARGET_FILES[task]) for task in tasks}
    datasets = sensor_sets(X_chest_full, X_left_full, X_right_full)
    results = {name: {task: {"folds": []} for task in tasks} for name in [*datasets, *LATE_FUSIONS]}
    result_file = config.RESULTS_DIR / f"results_multitask_{mode}_{config.MULTI_TASK_MODEL.lower()}.json"
    folds = LeaveOneGroupOut().split(X_chest_full, groups_full, groups_full)

    for fold, (train_idx, test_idx) in enumerate(tqdm(list(folds), desc=f"LOSO CV - Multitask {mode}"), 1):
        subject = groups_full[test_idx][0]
        fold_probs = {}
        for name, X in datasets.items():
            save_dir = config.CHECKPOINT_DIR / f"multitask_{mode}_{config.MULTI_TASK_MODEL.lower()}" / name
            X_train, X_test = prepare_data(X[train_idx], config.MULTI_TASK_MODEL, save_dir, X[test_idx], f"{name}_fold_{fold}_")
            y_train = {task: targets[task][train_idx] for task in tasks}
            y_test = {task: targets[task][test_idx] for task in tasks}
            train_loader = make_loader(X_train, y_train, 256, True)
            test_loader = make_loader(X_test, y_test, 256, False)
            model = create_model(config.MULTI_TASK_MODEL, X_train, n_classes, multitask=True)
            criteria = {task: nn.CrossEntropyLoss(weight=class_weights(y_train[task], n_classes[task])) for task in tasks}
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
            train_multitask(model, train_loader, criteria, tasks, optimizer, f"Training {name}")
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "fold": fold, "sensor_name": name,
            }, save_dir / f"{name}_fold_{fold}_subject_{subject}.pth")

            probs = evaluate_multitask(model, test_loader, tasks)
            fold_probs[name] = probs
            for task in tasks:
                valid = y_test[task] >= 0
                metrics = compute_metrics(y_test[task][valid], np.argmax(probs[task], axis=1)[valid]) if valid.sum() else {"accuracy": 0.0, "f1_macro": 0.0}
                metrics.update({"fold": fold, "test_subject": str(subject)})
                results[name][task]["folds"].append(metrics)

        for fusion, sensors in LATE_FUSIONS.items():
            for task in tasks:
                probs = sum(fold_probs[s][task] for s in sensors) / len(sensors)
                true = targets[task][test_idx]
                valid = true >= 0
                metrics = compute_metrics(true[valid], np.argmax(probs, axis=1)[valid]) if valid.sum() else {"accuracy": 0.0, "f1_macro": 0.0}
                metrics.update({"fold": fold, "test_subject": str(subject)})
                results[fusion][task]["folds"].append(metrics)
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)

    for model_results in results.values():
        for task_results in model_results.values():
            folds_data = task_results["folds"]
            if folds_data:
                acc = [x.get("accuracy", 0.0) for x in folds_data]
                f1_key = "f1" if "f1" in folds_data[0] else "f1_macro"
                f1 = [x.get(f1_key, 0.0) for x in folds_data]
                task_results.update({
                    "accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc)),
                    f"{f1_key}_mean": float(np.mean(f1)), f"{f1_key}_std": float(np.std(f1)),
                })
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nTraining FINAL multitask models on all 15 subjects...\n")
    for name, X in datasets.items():
        final_dir = config.CHECKPOINT_DIR / f"multitask_{mode}_{config.MULTI_TASK_MODEL.lower()}" / "FINAL" / name
        X_train, _ = prepare_data(X, config.MULTI_TASK_MODEL, final_dir)
        loader = make_loader(X_train, targets, 256, True, workers=True)
        model = create_model(config.MULTI_TASK_MODEL, X_train, n_classes, multitask=True)
        criteria = {task: nn.CrossEntropyLoss(weight=class_weights(targets[task], n_classes[task])) for task in tasks}
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        train_multitask(model, loader, criteria, tasks, optimizer, f"FINAL {name}", leave=True)
        torch.save({"model_state": model.state_dict(), "active_tasks": tasks, "num_classes": n_classes}, final_dir / "final_model.pth")
    print("FINAL multitask models saved.")