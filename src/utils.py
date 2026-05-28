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

import src.config as config
import src.models as models

warnings.filterwarnings("ignore", category=RuntimeWarning)

def extract_handcrafted_features(X):
    """Extrai features estatísticas do eixo do tempo. X shape: (N, 180, Features)"""
    means, stds = np.mean(X, axis=1), np.std(X, axis=1)
    max_vals, min_vals = np.max(X, axis=1), np.min(X, axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    
    X_safe = X + np.random.normal(0, 1e-8, X.shape)
    skewness, kurt = skew(X_safe, axis=1, bias=False), kurtosis(X_safe, axis=1, bias=False)
    
    return np.nan_to_num(np.concatenate([means, stds, max_vals, min_vals, rms, skewness, kurt], axis=1))

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

class DynamicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_dl_model(sensor_name, X_train, X_test, y_train, y_test, test_subject, fold_number, save_dir):
    ckpt_path = save_dir / f"{sensor_name}_fold_{fold_number}_subject_{test_subject}.pth"
    mean_path = save_dir / f"{sensor_name}_fold_{fold_number}_mean.npy"
    std_path = save_dir / f"{sensor_name}_fold_{fold_number}_std.npy"

    # Z-Score Normalization
    train_mean, train_std = X_train.mean(axis=(0, 1), keepdims=True), X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    np.save(mean_path, train_mean)
    np.save(std_path, train_std)
    X_train, X_test = (X_train - train_mean) / train_std, (X_test - train_mean) / train_std

    # Roteamento de Particularidades das Arquiteturas
    if config.MODEL_TYPE in ["CNN", "CNN_LSTM"]:
        X_train, X_test = X_train.transpose(0, 2, 1), X_test.transpose(0, 2, 1)
        bs = 256
        model = models.CNN1Conv(X_train.shape[1]).to(config.DEVICE) if config.MODEL_TYPE == "CNN" else models.DeepConvLSTM(X_train.shape[1]).to(config.DEVICE)
    elif config.MODEL_TYPE == "LSTM":
        bs = 256
        model = models.LSTMModel(X_train.shape[2]).to(config.DEVICE)
    elif config.MODEL_TYPE == "MLP":
        X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
        bs = len(X_train) # Full Batch obrigatório
        model = models.MLP(X_train.shape[1]).to(config.DEVICE)

    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    train_loader = DataLoader(DynamicDataset(X_train, y_train), batch_size=bs, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(DynamicDataset(X_test, y_test), batch_size=len(X_test) if config.MODEL_TYPE == "MLP" else bs, shuffle=False)

    class_weights = torch.tensor([1.0 / np.sum(y_train == 0), 1.0 / np.sum(y_train == 1)], dtype=torch.float32).to(config.DEVICE)
    criterion, optimizer = nn.CrossEntropyLoss(weight=class_weights), torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    else:
        best_loss, patience_counter = float("inf"), 0
        epoch_pbar = tqdm(range(config.EPOCHS), desc=f"{sensor_name} Fold {fold_number}", leave=False)
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
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}", best=f"{best_loss:.4f}")
            if avg_loss < best_loss:
                best_loss, patience_counter = avg_loss, 0
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE: break
        
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))

    # Previsão
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            outputs = model(X_b.to(config.DEVICE))
            probs = F.softmax(outputs, dim=1)
            y_true.extend(y_b.numpy())
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    return compute_metrics(y_true, y_pred), np.array(y_prob)

def train_classical_model(sensor_name, X_train, X_test, y_train, y_test, test_subject, fold_number):
    X_train_feat = extract_handcrafted_features(X_train)
    X_test_feat = extract_handcrafted_features(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    model = models.get_classical_model()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] 

    return compute_metrics(y_test, y_pred), y_prob