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


def _experiment_signature():
    if hasattr(config, "experiment_signature"):
        return config.experiment_signature()

    return {
        "fs": int(config.FS),
        "window_sec": float(config.WINDOW_SEC),
        "stride_sec": float(config.STRIDE_SEC),
        "window_samples": int(config.WINDOW_SAMPLES),
        "stride_samples": int(config.STRIDE_SAMPLES),
        "window_tag": str(config.WINDOW_TAG),
        "epochs": int(config.EPOCHS),
        "batch_size": int(config.BATCH_SIZE),
        "learning_rate": float(config.LEARNING_RATE),
        "dropout": float(config.DROPOUT),
        "classical_feature_set": str(
            config.CLASSICAL_FEATURE_SET
        ),
    }


def _signatures_match(saved, current=None):
    if current is None:
        current = _experiment_signature()
    return saved == current


def _checkpoint_compatible(state):
    saved = state.get("experiment_signature")
    if saved is None:
        return False
    return _signatures_match(saved)


def _metadata_file_compatible(path):
    if not path.exists():
        return False
    try:
        metadata = _read_json(path)
    except Exception:
        return False
    return _signatures_match(
        metadata.get("experiment_signature")
    )


CLASSICAL_MODELS = {"LOGREG", "RF", "SVM", "KNN", "LGBM"}
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


def epochs_for(model_type):
    if model_type == "MLP":
        return int(config.MLP_EPOCHS)
    return int(config.EPOCHS)


def batch_size_for(model_type, n_samples):
    if model_type == "MLP" and config.MLP_FULL_BATCH:
        return int(n_samples)
    return int(config.BATCH_SIZE)


def optimizer_for(model_type, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )


def extract_handcrafted_features(X):
    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape (samples, time, channels), got {X.shape}."
        )

    features = [
        np.mean(X, axis=1),
        np.std(X, axis=1),
        np.max(X, axis=1),
        np.min(X, axis=1),
        np.sqrt(np.mean(np.square(X), axis=1)),
        skew(X, axis=1, bias=False, nan_policy="omit"),
        kurtosis(X, axis=1, bias=False, nan_policy="omit"),
    ]

    return np.nan_to_num(
        np.concatenate(features, axis=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def compute_metrics(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    binary = len(classes) <= 2 and set(classes).issubset({0, 1})
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if binary:
        # In y_detect_fall: 0 = Fall and 1 = Non-Fall.
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tp = matrix[0, 0]
        fn = matrix[0, 1]
        fp = matrix[1, 0]
        tn = matrix[1, 1]

        result.update({
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                )
            ),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })
    else:
        labels = sorted(
            int(value)
            for value in classes.tolist()
        )
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        total = int(matrix.sum())
        per_class_counts = {}

        for index, class_id in enumerate(labels):
            tp = int(matrix[index, index])
            fn = int(matrix[index, :].sum() - tp)
            fp = int(matrix[:, index].sum() - tp)
            tn = int(total - tp - fn - fp)

            per_class_counts[str(class_id)] = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }

        result.update({
            "precision_macro": float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_macro": float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_macro": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_labels": labels,
            "confusion_matrix": (
                matrix.astype(int).tolist()
            ),
            "per_class_counts": per_class_counts,
        })
    return result


def compute_fixed_multiclass_metrics(
    y_true,
    y_pred,
    labels,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [int(label) for label in labels]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    total = int(matrix.sum())
    per_class_counts = {}

    for index, class_id in enumerate(labels):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(total - tp - fn - fp)

        per_class_counts[str(class_id)] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_labels": labels,
        "confusion_matrix": (
            matrix.astype(int).tolist()
        ),
        "per_class_counts": per_class_counts,
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


def train_multitask(
    model,
    loader,
    criteria,
    tasks,
    optimizer,
    description,
    leave=False,
    start_epoch=0,
    checkpoint_path=None,
    checkpoint_metadata=None,
):
    progress = tqdm(
        range(start_epoch, config.EPOCHS),
        desc=description,
        leave=leave,
    )

    for epoch in progress:
        model.train()
        total, batches = 0.0, 0

        for batch in loader:
            X = batch[0].to(config.DEVICE)
            targets = {
                task: batch[i + 1].to(config.DEVICE)
                for i, task in enumerate(tasks)
            }

            optimizer.zero_grad()
            loss = multitask_loss(
                model(X),
                targets,
                criteria,
                tasks,
            )

            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1

        progress.set_postfix(
            loss=(
                f"{total / batches if batches else 0.0:.4f}"
            )
        )

        if checkpoint_path is not None:
            state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "experiment_signature": (
                    _experiment_signature()
                ),
            }
            if checkpoint_metadata:
                state.update(checkpoint_metadata)
            torch.save(state, checkpoint_path)


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
    folds = entry.get("folds", [])
    if not folds:
        return

    ignored = {
        "fold",
        "test_subject",
        "confusion_labels",
        "confusion_matrix",
        "per_class_counts",
    }

    candidate_metrics = set()
    for fold in folds:
        candidate_metrics.update(fold.keys())

    for metric in sorted(candidate_metrics - ignored):
        values = [
            fold[metric]
            for fold in folds
            if metric in fold
        ]

        if not values:
            continue

        # Only aggregate true scalar numeric values.
        if not all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            continue

        numeric = np.asarray(values, dtype=np.float64)
        entry[f"{metric}_mean"] = float(np.mean(numeric))
        entry[f"{metric}_std"] = float(np.std(numeric))



def _resume_enabled():
    return bool(
        getattr(config, "RESUME_COMPLETED", True)
        and not getattr(config, "FORCE_RERUN", False)
    )


def _read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def _fold_cache_paths(save_dir, fold, subject):
    stem = f"fold_{fold}_subject_{subject}"
    return {
        "metrics": save_dir / f"{stem}_metrics.json",
        "probabilities": save_dir / f"{stem}_probabilities.npy",
        "model": save_dir / f"{stem}_model.pkl",
        "scaler": save_dir / f"{stem}_scaler.pkl",
        "metadata": save_dir / f"{stem}_metadata.json",
    }


def _load_fold_cache(save_dir, fold, subject, expected_samples):
    if not _resume_enabled():
        return None

    paths = _fold_cache_paths(save_dir, fold, subject)
    if not (
        paths["metrics"].exists()
        and paths["probabilities"].exists()
        and paths["metadata"].exists()
    ):
        return None

    if not _metadata_file_compatible(
        paths["metadata"]
    ):
        print(
            f"[CACHE INVALID] {save_dir.name} fold {fold}: "
            "experiment signature changed."
        )
        return None

    metrics = _read_json(paths["metrics"])
    probabilities = np.load(paths["probabilities"])

    if len(probabilities) != expected_samples:
        print(
            f"[CACHE INVALID] {save_dir.name} fold {fold}: "
            f"{len(probabilities)} probabilities for "
            f"{expected_samples} test samples."
        )
        return None

    print(
        f"[SKIP] Loaded completed fold {fold} "
        f"for {save_dir.name}."
    )
    return metrics, probabilities


def _save_fold_cache(
    save_dir,
    fold,
    subject,
    metrics,
    probabilities,
):
    paths = _fold_cache_paths(save_dir, fold, subject)
    _write_json(paths["metrics"], metrics)
    np.save(paths["probabilities"], probabilities)
    _write_json(
        paths["metadata"],
        {
            "experiment_signature": (
                _experiment_signature()
            ),
            "fold": int(fold),
            "test_subject": str(subject),
            "samples": int(len(probabilities)),
        },
    )


def _multitask_probability_cache(save_dir, fold, subject):
    return save_dir / (
        f"fold_{fold}_subject_{subject}_probabilities.npz"
    )


def _load_multitask_probability_cache(
    save_dir,
    fold,
    subject,
    tasks,
    expected_samples,
):
    if not _resume_enabled():
        return None

    path = _multitask_probability_cache(
        save_dir,
        fold,
        subject,
    )
    if not path.exists():
        return None

    with np.load(path) as cached:
        if not all(task in cached for task in tasks):
            return None
        probabilities = {
            task: cached[task]
            for task in tasks
        }

    if any(
        len(values) != expected_samples
        for values in probabilities.values()
    ):
        return None

    print(
        f"[SKIP] Loaded completed multitask fold {fold} "
        f"for {save_dir.name}."
    )
    return probabilities


def _save_multitask_probability_cache(
    save_dir,
    fold,
    subject,
    probabilities,
):
    path = _multitask_probability_cache(
        save_dir,
        fold,
        subject,
    )
    np.savez(path, **probabilities)

def train_classical(
    X_train,
    X_test,
    y_train,
    y_test,
    model_type,
    model_path=None,
    scaler_path=None,
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        extract_handcrafted_features(X_train)
    )
    X_test = scaler.transform(
        extract_handcrafted_features(X_test)
    )

    observed_classes = np.unique(y_train)
    num_classes = len(observed_classes)

    if num_classes < 2:
        raise ValueError(
            f"{model_type}: training partition contains only "
            f"{num_classes} class: {observed_classes.tolist()}."
        )

    model = models.get_classical_model(
        model_type=model_type,
        num_classes=num_classes,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return compute_metrics(y_test, predictions), probabilities


def train_single_task(
    task_name,
    model_type,
    X_chest_full,
    X_left_full,
    X_right_full,
    groups_full,
    experiment_tag="FULL_IMU",
):
    print(
        f"\n{'=' * 60}\n"
        f"=== TAREFA: {task_name.upper()} | MODELO: {model_type} "
        f"| MODALIDADE: {experiment_tag} ===\n"
        f"{'=' * 60}"
    )
    torch.backends.cudnn.benchmark = model_type in DL_MODELS

    y_full = np.load(
        config.WINDOWED_DATASET_DIR / f"{task_name}.npy"
    )
    valid = y_full != -1
    y = y_full[valid]

    if len(y) == 0:
        print(
            f"Nenhuma instância válida para {task_name}. "
            "Pulando..."
        )
        return

    groups = groups_full[valid]
    datasets = sensor_sets(
        X_chest_full[valid],
        X_left_full[valid],
        X_right_full[valid],
    )

    results = {
        name: {
            "folds": [],
            "model_type": model_type,
            "modality": experiment_tag,
            "feature_set": (
                config.CLASSICAL_FEATURE_SET
                if model_type in CLASSICAL_MODELS
                else "RAW_SEQUENCE"
            ),
            "input_channels_per_sensor": int(
                X_chest_full.shape[2]
            ),
        }
        for name in [*datasets, *LATE_FUSIONS]
    }

    result_file = (
        config.model_results_dir(model_type)
        / (
            f"results_{task_name}_"
            f"{model_type.lower()}_"
            f"{experiment_tag.lower()}.json"
        )
    )

    folds = list(
        LeaveOneGroupOut().split(
            datasets["CHEST"],
            y,
            groups,
        )
    )

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(
            folds,
            desc=f"LOSO CV - {task_name} ({model_type})",
        ),
        1,
    ):
        subject = groups[test_idx][0]
        fold_probs = {}

        for name, X in datasets.items():
            save_dir = (
                config.CHECKPOINT_DIR
                / (
                    f"{task_name}_"
                    f"{model_type.lower()}_"
                    f"{experiment_tag.lower()}"
                )
                / name
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            cached = _load_fold_cache(
                save_dir,
                fold,
                subject,
                expected_samples=len(test_idx),
            )

            if cached is not None:
                metrics, probs = cached
            elif model_type in CLASSICAL_MODELS:
                cache_paths = _fold_cache_paths(
                    save_dir,
                    fold,
                    subject,
                )
                metrics, probs = train_classical(
                    X[train_idx],
                    X[test_idx],
                    y[train_idx],
                    y[test_idx],
                    model_type,
                    model_path=cache_paths["model"],
                    scaler_path=cache_paths["scaler"],
                )
                _save_fold_cache(
                    save_dir,
                    fold,
                    subject,
                    metrics,
                    probs,
                )
            else:
                ckpt = save_dir / (
                    f"{name}_fold_{fold}_"
                    f"subject_{subject}.pth"
                )

                X_train, X_test = prepare_data(
                    X[train_idx],
                    model_type,
                    save_dir,
                    X[test_idx],
                    f"{name}_fold_{fold}_",
                )
                n_classes = int(np.max(y[train_idx])) + 1
                model = create_model(
                    model_type,
                    X_train,
                    n_classes,
                )

                if (
                    torch.cuda.device_count() > 1
                    and config.DEVICE.type == "cuda"
                ):
                    model = nn.DataParallel(model)

                train_loader = make_loader(
                    X_train,
                    y[train_idx],
                    batch_size_for(
                        model_type,
                        len(X_train),
                    ),
                    True,
                    workers=True,
                )
                test_loader = make_loader(
                    X_test,
                    y[test_idx],
                    batch_size_for(
                        model_type,
                        len(X_test),
                    ),
                    False,
                )
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights(
                        y[train_idx],
                        n_classes,
                    )
                )
                optimizer = optimizer_for(
                    model_type,
                    model,
                )

                start = 0
                if _resume_enabled() and ckpt.exists():
                    state = torch.load(
                        ckpt,
                        map_location=config.DEVICE,
                    )
                    if not _checkpoint_compatible(state):
                        print(
                            f"[CHECKPOINT INVALID] {ckpt}: "
                            "experiment signature changed; "
                            "training from epoch 0."
                        )
                        state = None

                    base = (
                        model.module
                        if isinstance(model, nn.DataParallel)
                        else model
                    )
                    if state is not None:
                        base.load_state_dict(
                            state["model_state"]
                        )
                        optimizer.load_state_dict(
                            state["optimizer_state"]
                        )
                        start = int(state["epoch"]) + 1
                        print(
                            f"[RESUME] {name} fold {fold} "
                            f"from epoch {start}."
                        )

                progress = tqdm(
                    range(start, epochs_for(model_type)),
                    desc=f"{name} Fold {fold}",
                    leave=False,
                )

                for epoch in progress:
                    loss = single_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                    )
                    progress.set_postfix(
                        loss=f"{loss:.4f}"
                    )
                    base = (
                        model.module
                        if isinstance(model, nn.DataParallel)
                        else model
                    )
                    torch.save(
                        {
                            "model_state": (
                                base.state_dict()
                            ),
                            "experiment_signature": (
                                _experiment_signature()
                            ),
                            "optimizer_state": (
                                optimizer.state_dict()
                            ),
                            "epoch": epoch,
                            "sensor_name": name,
                            "fold": fold,
                            "model_type": model_type,
                            "experiment_tag": experiment_tag,
                            "input_channels_per_sensor": int(
                                X_chest_full.shape[2]
                            ),
                        },
                        ckpt,
                    )

                state = torch.load(
                    ckpt,
                    map_location=config.DEVICE,
                )
                base = (
                    model.module
                    if isinstance(model, nn.DataParallel)
                    else model
                )
                base.load_state_dict(
                    state["model_state"]
                )
                metrics, probs = evaluate_single(
                    model,
                    test_loader,
                )
                _save_fold_cache(
                    save_dir,
                    fold,
                    subject,
                    metrics,
                    probs,
                )

            fold_probs[name] = probs
            metrics = dict(metrics)
            metrics.update(
                {
                    "fold": fold,
                    "test_subject": str(subject),
                }
            )
            results[name]["folds"].append(metrics)

        for fusion, sensors in LATE_FUSIONS.items():
            probs = (
                sum(fold_probs[s] for s in sensors)
                / len(sensors)
            )
            preds = np.argmax(probs, axis=1)
            metrics = compute_metrics(
                y[test_idx],
                preds,
            )
            metrics.update(
                {
                    "fold": fold,
                    "test_subject": str(subject),
                }
            )
            results[fusion]["folds"].append(metrics)

        _write_json(result_file, results)

    for entry in results.values():
        summarize(entry)

    _write_json(result_file, results)

    print(
        f"\nTraining FINAL models on all "
        f"{len(np.unique(groups))} available subjects...\n"
    )

    for name, X in datasets.items():
        final_dir = (
            config.CHECKPOINT_DIR
            / (
                f"{task_name}_"
                f"{model_type.lower()}_"
                f"{experiment_tag.lower()}"
            )
            / "FINAL"
            / name
        )
        final_dir.mkdir(parents=True, exist_ok=True)

        if model_type in CLASSICAL_MODELS:
            model_path = final_dir / "final_model.pkl"
            scaler_path = final_dir / "scaler.pkl"
            metadata_path = final_dir / "metadata.json"

            if (
                _resume_enabled()
                and model_path.exists()
                and scaler_path.exists()
                and metadata_path.exists()
                and _metadata_file_compatible(
                    metadata_path
                )
            ):
                print(
                    f"[SKIP] Final {model_type} model "
                    f"already exists for {name}."
                )
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(
                extract_handcrafted_features(X)
            )
            model = models.get_classical_model(
                model_type=model_type,
                num_classes=len(np.unique(y)),
            )
            model.fit(X_scaled, y)
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            _write_json(
                metadata_path,
                {
                    "task_name": task_name,
                    "model_type": model_type,
                    "experiment_signature": (
                        _experiment_signature()
                    ),
                    "modality": experiment_tag,
                    "feature_set": (
                        config.CLASSICAL_FEATURE_SET
                    ),
                    "input_channels_per_sensor": int(
                        X_chest_full.shape[2]
                    ),
                    "features_per_channel": 7,
                    "total_features": int(
                        extract_handcrafted_features(
                            X[:1]
                        ).shape[1]
                    ),
                    "reference": (
                        "MobiAct WEKA-default Logistic"
                        if model_type == "LOGREG"
                        else None
                    ),
                },
            )
            continue

        final_model_path = final_dir / "final_model.pth"
        training_state_path = (
            final_dir / "final_training_state.pth"
        )

        if (
            _resume_enabled()
            and final_model_path.exists()
        ):
            final_state = torch.load(
                final_model_path,
                map_location="cpu",
            )
            if _checkpoint_compatible(final_state):
                print(
                    f"[SKIP] Final neural model already "
                    f"exists for {name}."
                )
                continue
            print(
                f"[FINAL MODEL INVALID] {final_model_path}: "
                "experiment signature changed; retraining."
            )

        X_train, _ = prepare_data(
            X,
            model_type,
            final_dir,
        )
        n_classes = int(np.max(y)) + 1
        model = create_model(
            model_type,
            X_train,
            n_classes,
        )
        loader = make_loader(
            X_train,
            y,
            batch_size_for(
                model_type,
                len(X_train),
            ),
            True,
            workers=True,
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights(
                y,
                n_classes,
            )
        )
        optimizer = optimizer_for(
            config.MULTI_TASK_MODEL,
            model,
        )

        start = 0
        if (
            _resume_enabled()
            and training_state_path.exists()
        ):
            state = torch.load(
                training_state_path,
                map_location=config.DEVICE,
            )
            if _checkpoint_compatible(state):
                model.load_state_dict(
                    state["model_state"]
                )
                optimizer.load_state_dict(
                    state["optimizer_state"]
                )
                start = int(state["epoch"]) + 1
                print(
                    f"[RESUME] FINAL {name} "
                    f"from epoch {start}."
                )
            else:
                print(
                    f"[TRAINING STATE INVALID] "
                    f"{training_state_path}; "
                    "training from epoch 0."
                )

        for epoch in tqdm(
            range(start, epochs_for(model_type)),
            desc=f"FINAL {name}",
        ):
            single_epoch(
                model,
                loader,
                criterion,
                optimizer,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "experiment_signature": (
                        _experiment_signature()
                    ),
                    "optimizer_state": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                },
                training_state_path,
            )

        torch.save(
            {
                "model_state": model.state_dict(),
                "experiment_signature": (
                    _experiment_signature()
                ),
                "model_type": model_type,
                "num_classes": n_classes,
                "experiment_tag": experiment_tag,
                "input_channels_per_sensor": int(
                    X_chest_full.shape[2]
                ),
                "epochs": epochs_for(model_type),
                "batch_size": batch_size_for(
                    model_type,
                    len(X_train),
                ),
                "reference": (
                    config.MLP_SOURCE
                    if model_type == "MLP"
                    else None
                ),
            },
            final_model_path,
        )

    print("FINAL models saved or reused.")


def train_unified_model(
    model_type,
    X_chest_full,
    X_left_full,
    X_right_full,
    groups_full,
    experiment_tag="FULL_IMU",
):
    print(
        f"\n{'=' * 70}\n"
        f"=== UNIFIED 13-CLASS BASELINE: "
        f"{model_type} | {experiment_tag} ===\n"
        f"{'=' * 70}"
    )
    torch.backends.cudnn.benchmark = model_type in DL_MODELS

    target_path = (
        config.WINDOWED_DATASET_DIR
        / config.UNIFIED_TARGET
    )
    if not target_path.exists():
        raise FileNotFoundError(
            f"Unified target not found: {target_path}"
        )

    y_full = np.load(target_path)
    if not (
        len(y_full)
        == len(groups_full)
        == len(X_chest_full)
        == len(X_left_full)
        == len(X_right_full)
    ):
        raise ValueError(
            "X, groups, and y_unified must have "
            "the same length."
        )

    valid = y_full >= 0
    y = y_full[valid].astype(np.int64)
    groups = groups_full[valid]
    datasets = sensor_sets(
        X_chest_full[valid],
        X_left_full[valid],
        X_right_full[valid],
    )

    expected = set(
        range(config.UNIFIED_NUM_CLASSES)
    )
    observed = set(np.unique(y).tolist())
    if observed != expected:
        raise ValueError(
            f"Expected unified classes 0..12, "
            f"observed {sorted(observed)}"
        )

    print(f"Valid unified samples: {len(y)}")
    print(
        f"Ignored transition samples: "
        f"{int((~valid).sum())}"
    )

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

    result_file = (
        config.model_results_dir(model_type)
        / (
            f"results_y_unified_{model_type.lower()}_"
            f"{experiment_tag.lower()}.json"
        )
    )
    folds = list(
        LeaveOneGroupOut().split(
            datasets["CHEST"],
            y,
            groups,
        )
    )

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(
            folds,
            desc=f"LOSO CV - y_unified ({model_type})",
        ),
        1,
    ):
        subject = groups[test_idx][0]
        fold_probs = {}
        train_classes = set(
            np.unique(y[train_idx]).tolist()
        )

        if train_classes != expected:
            raise ValueError(
                f"Fold {fold} training partition "
                f"missing classes: "
                f"{sorted(expected - train_classes)}"
            )

        for name, X in datasets.items():
            save_dir = (
                config.CHECKPOINT_DIR
                / (
                    f"y_unified_{model_type.lower()}_"
                    f"{experiment_tag.lower()}"
                )
                / name
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            cached = _load_fold_cache(
                save_dir,
                fold,
                subject,
                expected_samples=len(test_idx),
            )

            if cached is not None:
                native, probs = cached
            elif model_type in CLASSICAL_MODELS:
                cache_paths = _fold_cache_paths(
                    save_dir,
                    fold,
                    subject,
                )
                native, probs = train_classical(
                    X[train_idx],
                    X[test_idx],
                    y[train_idx],
                    y[test_idx],
                    model_type,
                    model_path=cache_paths["model"],
                    scaler_path=cache_paths["scaler"],
                )
                _save_fold_cache(
                    save_dir,
                    fold,
                    subject,
                    native,
                    probs,
                )
            else:
                ckpt = save_dir / (
                    f"{name}_fold_{fold}_"
                    f"subject_{subject}.pth"
                )

                X_train, X_test = prepare_data(
                    X[train_idx],
                    model_type,
                    save_dir,
                    X[test_idx],
                    f"{name}_fold_{fold}_",
                )
                model = create_model(
                    model_type,
                    X_train,
                    config.UNIFIED_NUM_CLASSES,
                )

                if (
                    torch.cuda.device_count() > 1
                    and config.DEVICE.type == "cuda"
                ):
                    model = nn.DataParallel(model)

                train_loader = make_loader(
                    X_train,
                    y[train_idx],
                    batch_size_for(
                        model_type,
                        len(X_train),
                    ),
                    True,
                    workers=True,
                )
                test_loader = make_loader(
                    X_test,
                    y[test_idx],
                    batch_size_for(
                        model_type,
                        len(X_test),
                    ),
                    False,
                )
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights(
                        y[train_idx],
                        config.UNIFIED_NUM_CLASSES,
                    )
                )
                optimizer = optimizer_for(
                    model_type,
                    model,
                )

                start = 0
                if _resume_enabled() and ckpt.exists():
                    state = torch.load(
                        ckpt,
                        map_location=config.DEVICE,
                    )
                    if not _checkpoint_compatible(state):
                        print(
                            f"[CHECKPOINT INVALID] {ckpt}: "
                            "experiment signature changed; "
                            "training from epoch 0."
                        )
                        state = None

                    base = (
                        model.module
                        if isinstance(model, nn.DataParallel)
                        else model
                    )
                    if state is not None:
                        base.load_state_dict(
                            state["model_state"]
                        )
                        optimizer.load_state_dict(
                            state["optimizer_state"]
                        )
                        start = int(state["epoch"]) + 1
                        print(
                            f"[RESUME] {name} fold {fold} "
                            f"from epoch {start}."
                        )

                progress = tqdm(
                    range(start, epochs_for(model_type)),
                    desc=f"{name} Fold {fold}",
                    leave=False,
                )

                for epoch in progress:
                    loss = single_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                    )
                    progress.set_postfix(
                        loss=f"{loss:.4f}"
                    )
                    base = (
                        model.module
                        if isinstance(model, nn.DataParallel)
                        else model
                    )
                    torch.save(
                        {
                            "model_state": (
                                base.state_dict()
                            ),
                            "experiment_signature": (
                                _experiment_signature()
                            ),
                            "optimizer_state": (
                                optimizer.state_dict()
                            ),
                            "epoch": epoch,
                            "sensor_name": name,
                            "fold": fold,
                            "model_type": model_type,
                            "num_classes": (
                                config.UNIFIED_NUM_CLASSES
                            ),
                            "target": "y_unified",
                            "experiment_tag": experiment_tag,
                            "input_channels_per_sensor": int(
                                X_chest_full.shape[2]
                            ),
                        },
                        ckpt,
                    )

                state = torch.load(
                    ckpt,
                    map_location=config.DEVICE,
                )
                base = (
                    model.module
                    if isinstance(model, nn.DataParallel)
                    else model
                )
                base.load_state_dict(
                    state["model_state"]
                )
                native, probs = evaluate_single(
                    model,
                    test_loader,
                )
                _save_fold_cache(
                    save_dir,
                    fold,
                    subject,
                    native,
                    probs,
                )

            preds = np.argmax(probs, axis=1)
            fold_probs[name] = probs

            native = dict(native)
            native.update(
                {
                    "fold": fold,
                    "test_subject": str(subject),
                    "valid_samples": int(len(test_idx)),
                }
            )
            results[name]["native"]["folds"].append(
                native
            )

            mapped = compute_unified_mapped_metrics(
                y[test_idx],
                preds,
            )
            for task, metrics in mapped.items():
                metrics.update(
                    {
                        "fold": fold,
                        "test_subject": str(subject),
                    }
                )
                results[name]["mapped"][task][
                    "folds"
                ].append(metrics)

        for fusion, sensors in LATE_FUSIONS.items():
            probs = (
                sum(fold_probs[s] for s in sensors)
                / len(sensors)
            )
            preds = np.argmax(probs, axis=1)

            native = compute_metrics(
                y[test_idx],
                preds,
            )
            native.update(
                {
                    "fold": fold,
                    "test_subject": str(subject),
                    "valid_samples": int(len(test_idx)),
                }
            )
            results[fusion]["native"]["folds"].append(
                native
            )

            mapped = compute_unified_mapped_metrics(
                y[test_idx],
                preds,
            )
            for task, metrics in mapped.items():
                metrics.update(
                    {
                        "fold": fold,
                        "test_subject": str(subject),
                    }
                )
                results[fusion]["mapped"][task][
                    "folds"
                ].append(metrics)

        _write_json(result_file, results)

    for cfg_results in results.values():
        summarize(cfg_results["native"])
        for task_results in (
            cfg_results["mapped"].values()
        ):
            summarize(task_results)

    _write_json(result_file, results)

    print(
        "\nTraining FINAL unified models "
        "on all valid samples...\n"
    )

    for name, X in datasets.items():
        final_dir = (
            config.CHECKPOINT_DIR
            / (
                f"y_unified_{model_type.lower()}_"
                f"{experiment_tag.lower()}"
            )
            / "FINAL"
            / name
        )
        final_dir.mkdir(parents=True, exist_ok=True)

        if model_type in CLASSICAL_MODELS:
            model_path = final_dir / "final_model.pkl"
            scaler_path = final_dir / "scaler.pkl"

            metadata_path = final_dir / "metadata.json"

            if (
                _resume_enabled()
                and model_path.exists()
                and scaler_path.exists()
                and metadata_path.exists()
                and _metadata_file_compatible(
                    metadata_path
                )
            ):
                print(
                    f"[SKIP] Final unified {model_type} "
                    f"model already exists for {name}."
                )
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(
                extract_handcrafted_features(X)
            )
            model = models.get_classical_model(
                model_type=model_type,
                num_classes=config.UNIFIED_NUM_CLASSES,
            )
            model.fit(X_scaled, y)
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            _write_json(
                final_dir / "metadata.json",
                {
                    "target": "y_unified",
                    "model_type": model_type,
                    "experiment_signature": (
                        _experiment_signature()
                    ),
                    "experiment_signature": (
                        _experiment_signature()
                    ),
                    "modality": experiment_tag,
                    "num_classes": config.UNIFIED_NUM_CLASSES,
                    "feature_set": config.CLASSICAL_FEATURE_SET,
                    "input_channels_per_sensor": int(
                        X_chest_full.shape[2]
                    ),
                    "features_per_channel": 7,
                    "total_features": int(
                        extract_handcrafted_features(
                            X[:1]
                        ).shape[1]
                    ),
                    "reference": (
                        "MobiAct WEKA-default Logistic"
                        if model_type == "LOGREG"
                        else None
                    ),
                },
            )
            continue

        final_model_path = final_dir / "final_model.pth"
        training_state_path = (
            final_dir / "final_training_state.pth"
        )

        if (
            _resume_enabled()
            and final_model_path.exists()
        ):
            final_state = torch.load(
                final_model_path,
                map_location="cpu",
            )
            if _checkpoint_compatible(final_state):
                print(
                    f"[SKIP] Final unified neural model "
                    f"already exists for {name}."
                )
                continue
            print(
                f"[FINAL MODEL INVALID] {final_model_path}: "
                "experiment signature changed; retraining."
            )

        X_train, _ = prepare_data(
            X,
            model_type,
            final_dir,
        )
        model = create_model(
            model_type,
            X_train,
            config.UNIFIED_NUM_CLASSES,
        )
        loader = make_loader(
            X_train,
            y,
            batch_size_for(
                model_type,
                len(X_train),
            ),
            True,
            workers=True,
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights(
                y,
                config.UNIFIED_NUM_CLASSES,
            )
        )
        optimizer = optimizer_for(
            model_type,
            model,
        )

        start = 0
        if (
            _resume_enabled()
            and training_state_path.exists()
        ):
            state = torch.load(
                training_state_path,
                map_location=config.DEVICE,
            )
            if _checkpoint_compatible(state):
                model.load_state_dict(
                    state["model_state"]
                )
                optimizer.load_state_dict(
                    state["optimizer_state"]
                )
                start = int(state["epoch"]) + 1
                print(
                    f"[RESUME] FINAL UNIFIED {name} "
                    f"from epoch {start}."
                )
            else:
                print(
                    f"[TRAINING STATE INVALID] "
                    f"{training_state_path}; "
                    "training from epoch 0."
                )

        for epoch in tqdm(
            range(start, epochs_for(model_type)),
            desc=f"FINAL UNIFIED {name}",
        ):
            single_epoch(
                model,
                loader,
                criterion,
                optimizer,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "experiment_signature": (
                        _experiment_signature()
                    ),
                    "optimizer_state": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                },
                training_state_path,
            )

        torch.save(
            {
                "model_state": model.state_dict(),
                "experiment_signature": (
                    _experiment_signature()
                ),
                "model_type": model_type,
                "num_classes": (
                    config.UNIFIED_NUM_CLASSES
                ),
                "target": "y_unified",
                "experiment_tag": experiment_tag,
                "input_channels_per_sensor": int(
                    X_chest_full.shape[2]
                ),
            },
            final_model_path,
        )

    print(
        f"Unified results saved to: {result_file}"
    )


def run_multitask(
    mode,
    X_chest_full,
    X_left_full,
    X_right_full,
    groups_full,
):
    print(
        f"\n{'=' * 60}\n"
        f"=== INICIANDO MULTI-TASK & "
        f"FUSÃO DE SENSORES ({mode}) ===\n"
        f"Arquitetura Base: "
        f"{config.MULTI_TASK_MODEL}\n"
        f"{'=' * 60}"
    )

    if mode not in MULTITASK_CONFIGS:
        raise ValueError(
            f"Modo Multitarefa Inválido: {mode}"
        )

    n_classes = MULTITASK_CONFIGS[mode]
    tasks = list(n_classes)
    targets = {
        task: np.load(
            config.WINDOWED_DATASET_DIR
            / TARGET_FILES[task]
        )
        for task in tasks
    }
    datasets = sensor_sets(
        X_chest_full,
        X_left_full,
        X_right_full,
    )
    results = {
        name: {
            task: {"folds": []}
            for task in tasks
        }
        for name in [*datasets, *LATE_FUSIONS]
    }
    result_file = (
        config.model_results_dir(config.MULTI_TASK_MODEL)
        / (
            f"results_multitask_{mode}_"
            f"{config.MULTI_TASK_MODEL.lower()}.json"
        )
    )
    folds = list(
        LeaveOneGroupOut().split(
            X_chest_full,
            groups_full,
            groups_full,
        )
    )

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(
            folds,
            desc=f"LOSO CV - Multitask {mode}",
        ),
        1,
    ):
        subject = groups_full[test_idx][0]
        fold_probs = {}

        for name, X in datasets.items():
            save_dir = (
                config.CHECKPOINT_DIR
                / (
                    f"multitask_{mode}_"
                    f"{config.MULTI_TASK_MODEL.lower()}"
                )
                / name
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            cached_probs = (
                _load_multitask_probability_cache(
                    save_dir,
                    fold,
                    subject,
                    tasks,
                    expected_samples=len(test_idx),
                )
            )

            y_test = {
                task: targets[task][test_idx]
                for task in tasks
            }

            if cached_probs is not None:
                probs = cached_probs
            else:
                X_train, X_test = prepare_data(
                    X[train_idx],
                    config.MULTI_TASK_MODEL,
                    save_dir,
                    X[test_idx],
                    f"{name}_fold_{fold}_",
                )
                y_train = {
                    task: targets[task][train_idx]
                    for task in tasks
                }

                train_loader = make_loader(
                    X_train,
                    y_train,
                    config.BATCH_SIZE,
                    True,
                    workers=True,
                )
                test_loader = make_loader(
                    X_test,
                    y_test,
                    config.BATCH_SIZE,
                    False,
                )
                model = create_model(
                    config.MULTI_TASK_MODEL,
                    X_train,
                    n_classes,
                    multitask=True,
                )
                criteria = {
                    task: nn.CrossEntropyLoss(
                        weight=class_weights(
                            y_train[task],
                            n_classes[task],
                        )
                    )
                    for task in tasks
                }
                optimizer = optimizer_for(
                    config.MULTI_TASK_MODEL,
                    model,
                )

                ckpt = save_dir / (
                    f"{name}_fold_{fold}_"
                    f"subject_{subject}.pth"
                )
                start = 0

                if _resume_enabled() and ckpt.exists():
                    state = torch.load(
                        ckpt,
                        map_location=config.DEVICE,
                    )
                    if _checkpoint_compatible(state):
                        model.load_state_dict(
                            state["model_state"]
                        )
                        optimizer.load_state_dict(
                            state["optimizer_state"]
                        )
                        start = int(state["epoch"]) + 1
                        print(
                            f"[RESUME] Multitask {name} "
                            f"fold {fold} from epoch {start}."
                        )
                    else:
                        print(
                            f"[CHECKPOINT INVALID] {ckpt}: "
                            "experiment signature changed; "
                            "training from epoch 0."
                        )

                train_multitask(
                    model,
                    train_loader,
                    criteria,
                    tasks,
                    optimizer,
                    f"Training {name} Fold {fold}",
                    start_epoch=start,
                    checkpoint_path=ckpt,
                    checkpoint_metadata={
                        "fold": fold,
                        "sensor_name": name,
                        "active_tasks": tasks,
                    },
                )

                state = torch.load(
                    ckpt,
                    map_location=config.DEVICE,
                )
                model.load_state_dict(
                    state["model_state"]
                )
                probs = evaluate_multitask(
                    model,
                    test_loader,
                    tasks,
                )
                _save_multitask_probability_cache(
                    save_dir,
                    fold,
                    subject,
                    probs,
                )

            fold_probs[name] = probs

            for task in tasks:
                valid = y_test[task] >= 0
                if valid.sum():
                    metrics = compute_metrics(
                        y_test[task][valid],
                        np.argmax(
                            probs[task],
                            axis=1,
                        )[valid],
                    )
                else:
                    metrics = {
                        "accuracy": 0.0,
                        "f1_macro": 0.0,
                    }

                metrics.update(
                    {
                        "fold": fold,
                        "test_subject": str(subject),
                    }
                )
                results[name][task]["folds"].append(
                    metrics
                )

        for fusion, sensors in LATE_FUSIONS.items():
            for task in tasks:
                probs = (
                    sum(
                        fold_probs[s][task]
                        for s in sensors
                    )
                    / len(sensors)
                )
                true = targets[task][test_idx]
                valid = true >= 0

                if valid.sum():
                    metrics = compute_metrics(
                        true[valid],
                        np.argmax(
                            probs,
                            axis=1,
                        )[valid],
                    )
                else:
                    metrics = {
                        "accuracy": 0.0,
                        "f1_macro": 0.0,
                    }

                metrics.update(
                    {
                        "fold": fold,
                        "test_subject": str(subject),
                    }
                )
                results[fusion][task]["folds"].append(
                    metrics
                )

        _write_json(result_file, results)

    for model_results in results.values():
        for task_results in model_results.values():
            summarize(task_results)

    _write_json(result_file, results)

    print(
        f"\nTraining FINAL multitask models on all "
        f"{len(np.unique(groups_full))} "
        f"available subjects...\n"
    )

    for name, X in datasets.items():
        final_dir = (
            config.CHECKPOINT_DIR
            / (
                f"multitask_{mode}_"
                f"{config.MULTI_TASK_MODEL.lower()}"
            )
            / "FINAL"
            / name
        )
        final_dir.mkdir(parents=True, exist_ok=True)

        final_model_path = final_dir / "final_model.pth"
        training_state_path = (
            final_dir / "final_training_state.pth"
        )

        if (
            _resume_enabled()
            and final_model_path.exists()
        ):
            final_state = torch.load(
                final_model_path,
                map_location="cpu",
            )
            if _checkpoint_compatible(final_state):
                print(
                    f"[SKIP] Final multitask model "
                    f"already exists for {name}."
                )
                continue
            print(
                f"[FINAL MODEL INVALID] {final_model_path}: "
                "experiment signature changed; retraining."
            )

        X_train, _ = prepare_data(
            X,
            config.MULTI_TASK_MODEL,
            final_dir,
        )
        loader = make_loader(
            X_train,
            targets,
            config.BATCH_SIZE,
            True,
            workers=True,
        )
        model = create_model(
            config.MULTI_TASK_MODEL,
            X_train,
            n_classes,
            multitask=True,
        )
        criteria = {
            task: nn.CrossEntropyLoss(
                weight=class_weights(
                    targets[task],
                    n_classes[task],
                )
            )
            for task in tasks
        }
        optimizer = optimizer_for(
            model_type,
            model,
        )

        start = 0
        if (
            _resume_enabled()
            and training_state_path.exists()
        ):
            state = torch.load(
                training_state_path,
                map_location=config.DEVICE,
            )
            if _checkpoint_compatible(state):
                model.load_state_dict(
                    state["model_state"]
                )
                optimizer.load_state_dict(
                    state["optimizer_state"]
                )
                start = int(state["epoch"]) + 1
                print(
                    f"[RESUME] FINAL multitask {name} "
                    f"from epoch {start}."
                )
            else:
                print(
                    f"[TRAINING STATE INVALID] "
                    f"{training_state_path}; "
                    "training from epoch 0."
                )

        train_multitask(
            model,
            loader,
            criteria,
            tasks,
            optimizer,
            f"FINAL {name}",
            leave=True,
            start_epoch=start,
            checkpoint_path=training_state_path,
            checkpoint_metadata={
                "active_tasks": tasks,
                "num_classes": n_classes,
            },
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "experiment_signature": (
                    _experiment_signature()
                ),
                "active_tasks": tasks,
                "num_classes": n_classes,
            },
            final_model_path,
        )

    print(
        "FINAL multitask models saved or reused."
    )
