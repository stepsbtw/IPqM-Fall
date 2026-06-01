# IPqM-Fall: Real-Time Military Activity Recognition and Fall Detection with Wearable Inertial Sensors

This repository contains the official machine learning pipeline for the **IPqM-Fall** dataset. Designed for tactical and military environments, the pipeline supports Human Activity Recognition (HAR) and Fall Detection using wearable IMU sensors positioned on the chest and wrists.

To simplify reproduction, this repository includes a curated `windows.parquet` file containing all window boundaries and validated labels. Users only need to download the raw dataset and run the final dataset generation step before training.

---

# Dataset Download

The raw continuous IMU recordings are hosted on Zenodo.

## Download Instructions

1. Download the raw dataset from Zenodo:

   **Zenodo:** `doi.org/10.5281/zenodo.20431609`

2. Extract the dataset so that the project structure becomes:

```text
project_root/
├── IPqM-Fall/
│   ├── raw/
│   │   ├── ID1/
│   │   ├── ID2/
│   │   └── ...
│   └── windows.parquet
├── src/
└── README.md
```

The repository already includes:

```text
IPqM-Fall/windows.parquet
```

which contains:

* Window start indices
* Window end indices
* Corrected activity labels
* Fall refinements
* Transition refinements

Therefore, users do **not** need to run the original preprocessing pipeline.

---

# Generate Training Arrays

Generate the machine learning datasets using:

```bash
python src/generate_dataset.py
```

The script uses:

* `IPqM-Fall/raw/`
* `IPqM-Fall/windows.parquet`

to generate:

```text
X_chest.npy
X_left.npy
X_right.npy

y_detect_fall.npy
y_detect_movement.npy
y_classify_fall.npy
y_classify_posture.npy
y_classify_movement.npy
y_complete.npy

groups.npy
```

## Multi-Head Labeling Schema

| Task                | Description                       |
| ------------------- | --------------------------------- |
| `detect_fall`       | Binary fall detection             |
| `detect_movement`   | Movement vs static classification |
| `classify_fall`     | Fall type classification          |
| `classify_posture`  | Static posture classification     |
| `classify_movement` | Dynamic activity classification   |
| `complete`          | Complete activity taxonomy        |

The file `groups.npy` contains subject identifiers used for Leave-One-Subject-Out (LOSO) evaluation.

---

# Machine Learning Pipeline

The repository provides a unified framework for training and evaluating both classical machine learning and deep learning models.

## Configuration

Experiment settings are centralized in:

```text
src/config.py
```

including model selection, training hyperparameters, and dataset paths.

## Supported Models

### Classical Machine Learning

* Support Vector Machine (SVM)
* Random Forest (RF)
* K-Nearest Neighbors (KNN)

### Deep Learning

* CNN1Conv
* DeepConvLSTM
* LSTM
* MLP

## Training

Run:

```bash
python src/train.py
```

The framework automatically:

* Performs Leave-One-Subject-Out (LOSO) cross-validation
* Supports single-sensor, early-fusion, and late-fusion configurations
* Applies class weighting and early stopping for deep learning models
* Extracts statistical features automatically for classical models

## Evaluation Metrics

The pipeline reports:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Results are automatically saved for later analysis.

---

# Edge Deployment

Convert trained PyTorch models to TensorFlow Lite:

```bash
python src/extra/tflite_converter.py
```

Conversion pipeline:

```text
PyTorch → ONNX → TensorFlow → INT8 TFLite
```

for deployment on smartphones and wearable devices.

---

# Reproducibility

The repository also includes the original preprocessing scripts used to generate `windows.parquet`:

```text
src/data_scripts/
├── 1_generate_windows.py
├── 2_fix_fall_labels.py
├── 3_fix_transitions.py
├── 4_fix_sitting.py
└── 5_generate_dataset.py
```

Scripts 1–4 were used to create the curated `windows.parquet` file and are provided for transparency and reproducibility. They are **not required** for standard usage.

---

# Citation

If you use the IPqM-Fall dataset or this repository in academic work, please cite the accompanying publication and Zenodo dataset release.
