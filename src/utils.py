import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import skew, kurtosis
import warnings

import config
import models

warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_handcrafted_features(X):
    means, stds = np.mean(X, axis=1), np.std(X, axis=1)
    max_vals, min_vals = np.max(X, axis=1), np.min(X, axis=1)
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


def train_dl_model(sensor_name, X_train, X_test, y_train, y_test,
                   test_subject, fold_number, save_dir, model_type):

    ckpt_path = save_dir / f"{sensor_name}_fold_{fold_number}_subject_{test_subject}.pth"
    mean_path = save_dir / f"{sensor_name}_fold_{fold_number}_mean.npy"
    std_path = save_dir / f"{sensor_name}_fold_{fold_number}_std.npy"

    train_mean = X_train.mean(axis=(0, 1), keepdims=True)
    train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    np.save(mean_path, train_mean)
    np.save(std_path, train_std)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    num_classes = int(np.max(y_train)) + 1

    if model_type in ["CNN1Conv", "DeepConvLSTM"]:
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)
        bs = 256

        model = (
            models.CNN1Conv(X_train.shape[1], num_classes)
            if model_type == "CNN1Conv"
            else models.DeepConvLSTM(X_train.shape[1], num_classes)
        ).to(config.DEVICE)

    elif model_type == "LSTM":
        bs = 256
        model = models.LSTMModel(X_train.shape[2], num_classes).to(config.DEVICE)

    elif model_type == "MLP":
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        bs = len(X_train)
        model = models.MLP(X_train.shape[1], num_classes).to(config.DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_loader = DataLoader(
        DynamicDataset(X_train, y_train),
        batch_size=bs,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        DynamicDataset(X_test, y_test),
        batch_size=len(X_test) if model_type == "MLP" else bs,
        shuffle=False
    )

    class_counts = [np.sum(y_train == c) for c in range(num_classes)]
    class_weights = torch.tensor(
        [1.0 / (c + 1e-6) for c in class_counts],
        dtype=torch.float32
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    start_epoch = 0

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"[RESUME] {sensor_name} fold {fold_number} from epoch {start_epoch}")

    epoch_pbar = tqdm(
        range(start_epoch, config.EPOCHS),
        desc=f"{sensor_name} Fold {fold_number} ({model_type})",
        leave=False
    )

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

        avg_loss = total_loss / len(train_loader)
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")

        torch.save({
            "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "sensor_name": sensor_name,
            "fold": fold_number,
            "model_type": model_type
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


def train_classical_model(sensor_name, X_train, X_test, y_train, y_test,
                          test_subject, fold_number, model_type):

    X_train_feat = extract_handcrafted_features(X_train)
    X_test_feat = extract_handcrafted_features(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    model = models.get_classical_model(model_type)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    return compute_metrics(y_test, y_pred), y_prob