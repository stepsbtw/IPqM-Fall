import torch.nn as nn
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

import config

def get_classical_model(model_type, num_classes=None):
    model_type = model_type.upper()

    if model_type == "LOGREG":
        return LogisticRegression(
            **config.LOGREG_PARAMS
        )

    if model_type == "RF":
        return RandomForestClassifier(
            n_estimators=10,
            criterion="gini",
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "SVM":
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="auto",
            shrinking=True,
            tol=0.001,
            probability=True,
            class_weight=None,
            random_state=42,
        )

    if model_type == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            algorithm="auto",
            leaf_size=30,
            metric="euclidean",
            n_jobs=-1,
        )

    if model_type == "LGBM":
        if num_classes is None:
            raise ValueError(
                "num_classes must be provided when creating LightGBM."
            )

        params = dict(config.LIGHTGBM_PARAMS)

        if num_classes == 2:
            params["objective"] = "binary"
            params.pop("num_class", None)
        elif num_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = int(num_classes)
        else:
            raise ValueError(
                f"LightGBM requires at least two classes, got {num_classes}."
            )

        return LGBMClassifier(**params)

    raise ValueError(
        f"Modelo clássico {model_type} não reconhecido."
    )


class CNN1Conv(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_features, config.WINDOW_SAMPLES)
            flattened_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNN3B3Conv(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_features, config.WINDOW_SAMPLES)
            flattened_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
        
class DeepConvLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_block(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(self.dropout(lstm_out[:, -1, :]))

class LSTMModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=200, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        self.classifier = nn.Sequential(nn.Linear(200, 200), nn.ReLU(), nn.Dropout(config.DROPOUT), nn.Linear(200, num_classes))
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])

class MLP(nn.Module):
    """
    Raw-window ANN baseline based on Georgakopoulos et al.

    Architecture reported in the paper:
      flattened raw window -> 100 hidden neurons -> output layer

    The paper reports 50 epochs and full-batch training. Those settings
    are enforced by utils.py.
    """

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, config.MLP_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(config.MLP_HIDDEN_UNITS, num_classes),
        )

    def forward(self, x):
        return self.network(x)

class CNN1Conv_MultiTask(nn.Module):
    def __init__(self, num_features, num_classes_dict):
        super().__init__()

        self.shared_features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_features, config.WINDOW_SAMPLES)
            shared_size = self.shared_features(dummy).view(1, -1).shape[1]

        self.heads = nn.ModuleDict()
        for task_name, out_features in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(shared_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, out_features)
            )

    def forward(self, x):
        shared = self.shared_features(x)
        return {task_name: head(shared) for task_name, head in self.heads.items()}

class CNN3B3Conv_MultiTask(nn.Module):
    def __init__(self, num_features, num_classes_dict):
        super().__init__()

        self.shared_features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(config.DROPOUT)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_features, config.WINDOW_SAMPLES)
            shared_size = self.shared_features(dummy).view(1, -1).shape[1]

        self.heads = nn.ModuleDict()
        for task_name, out_features in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(shared_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, out_features)
            )

    def forward(self, x):
        shared = self.shared_features(x)
        return {task_name: head(shared) for task_name, head in self.heads.items()}

class DeepConvLSTM_MultiTask(nn.Module):
    def __init__(self, num_features, num_classes_dict):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        self.heads = nn.ModuleDict()
        for task_name, out_features in num_classes_dict.items():
            self.heads[task_name] = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.conv_block(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        
        shared = self.dropout(lstm_out[:, -1, :])
        
        return {task_name: head(shared) for task_name, head in self.heads.items()}


class LSTMModel_MultiTask(nn.Module):
    def __init__(self, num_features, num_classes_dict):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=200, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        
        self.heads = nn.ModuleDict()
        for task_name, out_features in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Linear(200, 200), nn.ReLU(), 
                nn.Dropout(config.DROPOUT), 
                nn.Linear(200, out_features)
            )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        
        shared = hidden[-1]
        
        return {task_name: head(shared) for task_name, head in self.heads.items()}

# class MLP_MultiTask(nn.Module):
#    def __init__(self, input_size, num_classes_dict):
#        super().__init__()
#        self.shared_features = nn.Sequential(nn.Linear(input_size, 100), nn.ReLU())
#        self.heads = nn.ModuleDict()
#        for task_name, out_features in num_classes_dict.items():
#            self.heads[task_name] = nn.Linear(100, out_features)
#    def forward(self, x):
#        shared = self.shared_features(x)
#        return {task_name: head(shared) for task_name, head in self.heads.items()}