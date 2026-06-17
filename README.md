# IPqM-Fall: Military Fall Detection and Activity Classification Using Wearable Inertial Sensors

This repository contains the official machine learning pipeline for the **IPqM-Fall** dataset. Designed for tactical and military environments, it supports Human Activity Recognition (HAR) and Fall Detection using synchronized wearable inertial measurement units (IMUs) positioned on the chest and both wrists.

The framework supports configurable window sizes, strides, and sampling rates through `src/config.py`, allowing researchers to reproduce experiments under different temporal segmentation settings. It includes task-specific and unified activity formulations, sensor-modality ablations, individual sensors, Early Fusion, Late Fusion, classical machine-learning baselines, and neural-network models evaluated through Leave-One-Subject-Out (LOSO) cross-validation.

## 1. Getting Started

The raw continuous IMU recordings are hosted on Zenodo. To reproduce the experiments, first download the dataset and place the files in the expected directory structure.

### 1.1 Installation

Create a Python environment and install the project dependencies:

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch installation may depend on the available CPU, CUDA, or Intel XPU environment. All commands below should be executed from the repository root.

### 1.2 Download and Organize the Dataset

1. Download the raw dataset from Zenodo: `https://doi.org/10.5281/zenodo.20431609`.
2. Extract the downloaded archive.
3. Organize the files as follows:

```text
project_root/
├── IPqM-Fall/
│   └── raw/
│       ├── ID1/
│       │   ├── CHEST/
│       │   ├── LEFT/
│       │   └── RIGHT/
│       ├── ID2/
│       └── ...
├── src/
├── requirements.txt
└── README.md
```

### 1.3 Generate Training Arrays

After the dataset has been placed in the correct location, generate the NumPy arrays used for training and evaluation:

```bash
python src/generate_dataset.py
```

This script executes the complete preprocessing pipeline:

```text
1_generate_windows.py
        ↓
2_fix_fall_labels.py
        ↓
3_fix_transitions.py
        ↓
4_fix_sitting.py
        ↓
5_generate_arrays.py
```

The pipeline creates the synchronized sensor tensors:

```text
X_chest.npy
X_left.npy
X_right.npy
```

Each sensor tensor contains eight channels:

```text
ax, ay, az, amag, wx, wy, wz, wmag
```

The pipeline also creates subject identifiers for LOSO cross-validation, synchronized window identifiers, and the label arrays required by the supported tasks:

```text
groups.npy
sync_ids.npy
y_detect_fall.npy
y_classify_fall.npy
y_classify_posture.npy
y_classify_movement.npy
y_unified.npy
```

Additional auxiliary targets are generated for movement detection, transition classification, and the complete raw-label taxonomy. The exact label mappings are stored in `IPqM-Fall/mapping.json`.

Generated datasets are stored under:

```text
IPqM-Fall/windowed/<window-configuration>/
```

For example:

```text
IPqM-Fall/windowed/5-sec_1-step/
```

This allows multiple window configurations to coexist simultaneously.

### 1.4 Dataset Configuration

Dataset generation is fully parameterized through:

```text
src/config.py
```

The current default configuration is:

```python
FS = 90
WINDOW_SEC = 5
STRIDE_SEC = 1
```

The paper compares 2-second and 5-second windows using a common 1-second prediction step. Change `WINDOW_SEC`, regenerate the arrays, and rerun the experiments for each temporal configuration.

---

## 2. Training and Evaluation

The repository provides a unified framework for evaluating classical machine-learning and neural-network approaches using LOSO cross-validation. The implemented architectures and their initial hyperparameters are provided as baseline configurations and have not necessarily been systematically optimized for every IPqM-Fall task.

### Supported Architectures

#### Classical Machine Learning

* Logistic Regression (`LOGREG`)
* Random Forest (`RF`)
* Support Vector Machine (`SVM`)
* K-Nearest Neighbors (`KNN`)
* LightGBM (`LGBM`)

The classical models operate on handcrafted time-domain features extracted from each inertial channel.

#### Deep Learning

* **MLP**: A shallow fully connected baseline applied to flattened raw windows.
* **CNN1Conv**: A lightweight one-block convolutional architecture for temporal inertial classification.
* **CNN3B3Conv**: A deeper convolutional architecture composed of three convolutional blocks.
* **LSTM**: A recurrent model for temporal sequence classification.
* **DeepConvLSTM**: A hybrid architecture that combines convolutional feature extraction with recurrent temporal modeling.

### Task-Specific and Unified Formulations

The main experiment entry point supports two learning formulations configured through `EXPERIMENTS_TO_RUN`:

```python
EXPERIMENTS_TO_RUN = [
    "TASK_MODEL_MATRIX",
    "UNIFIED_MODEL_MATRIX",
]
```

#### Task-Specific Models

`TASK_MODEL_MATRIX` trains independent models for the selected tasks:

* Fall Detection: 2 classes;
* Fall-Type Classification: 4 classes;
* Posture Classification: 4 classes;
* Movement Classification: 5 classes.

Task-specific classification arrays use `-1` for samples that do not belong to the corresponding task. These samples are excluded before training and evaluation.

#### Unified Model

`UNIFIED_MODEL_MATRIX` trains one flat 13-class classifier containing:

```text
Backward Fall, Frontal Fall, Lateral Fall Left, Lateral Fall Right,
Standing, Sitting, Kneeling, Prone and Down,
Walking, Sweeping, Running, Jumping, Crawling
```

The unified predictions are also mapped back to Fall Detection, Fall-Type Classification, Posture Classification, and Movement Classification, allowing direct comparison with the task-specific models.

### Sensor Modalities

The framework supports:

* accelerometer: `ax`, `ay`, `az`, and `amag`;
* gyroscope: `wx`, `wy`, `wz`, and `wmag`;
* full IMU: all eight channels.

The active modalities are selected through `TASK_MODALITIES_TO_RUN`, `UNIFIED_MODALITIES_TO_RUN`, and `MODALITY_ABLATIONS` in `src/config.py`.

### Sensor Fusion

The framework evaluates individual sensors and Early-Fusion combinations:

```text
CHEST
LEFT
RIGHT
CHEST_LEFT
CHEST_RIGHT
LEFT_RIGHT
CHEST_LEFT_RIGHT
```

In Early Fusion, channels from multiple body positions are concatenated before model training, allowing cross-sensor relationships to be learned directly from the data.

The framework also evaluates Late-Fusion ensembles:

```text
ENSEMBLE_CHEST_LEFT
ENSEMBLE_CHEST_RIGHT
ENSEMBLE_LEFT_RIGHT
ENSEMBLE_CHEST_LEFT_RIGHT
```

Late Fusion averages the class-probability vectors produced by independently trained sensor models.

### Training

Model training, evaluation, class weighting, metric reporting, checkpointing, and fusion handling are managed through:

```bash
python src/train.py
```

Experiment settings are configured in:

```text
src/config.py
```

The configuration file controls:

* dataset generation parameters;
* experiment formulation;
* task selection;
* model selection;
* sensor modalities;
* learning rate;
* dropout;
* number of epochs;
* batch size;
* checkpoint resumption;
* output directories.

The training pipeline reports Accuracy, macro Precision, macro Recall, macro F1-score, class-wise metrics, and Confusion Matrices under LOSO cross-validation.

All experiments use Leave-One-Subject-Out cross-validation to assess subject-independent generalization and prevent subject-specific information leakage between the training and test sets.

### Results and Checkpoints

Experiment results are stored under:

```text
results/<window-configuration>/<model-family>/
```

Fold checkpoints, normalization statistics, trained models, and probability caches are stored under:

```text
checkpoints/<window-configuration>/
```

Compatible completed folds can be resumed or skipped using:

```python
RESUME_COMPLETED = True
FORCE_RERUN = False
```

### LaTeX Table Generation

After generating the JSON results, create the paper and repository tables with:

```bash
python src/output/latex_tables.py
```

The generated tables are organized under:

```text
results/tables/main/
results/tables/appendix/
results/tables/repository/
```

---

## 3. Experimental Extensions

### Multi-Task Learning

Multi-task model classes and training utilities are included in `src/models.py` and `src/utils.py`. They use a shared feature extractor with task-specific output heads and ignore invalid `-1` labels separately for each task.

However, Multi-Task Learning is not currently exposed through the default `src/train.py` experiment matrix and is not part of the main paper experiments. It should therefore be treated as an experimental extension until it is reintegrated and validated against the current task-specific and unified pipelines.

### TensorFlow Lite Conversion

A legacy conversion prototype is available at:

```text
src/output/tflite_converter.py
```

Its intended workflow is:

```text
PyTorch → ONNX → TensorFlow → INT8 TensorFlow Lite
```

This utility is not part of the validated reproducibility pipeline. It assumes an earlier checkpoint organization, requires additional TensorFlow and ONNX dependencies, and may require adaptation before being used with models produced by the current training matrix.

---

## 4. Reproducibility and Citation

The preprocessing scripts are included for transparency and reproducibility. Researchers may regenerate datasets with different window lengths, strides, and sampling rates through `src/config.py`.

For consistent experiments:

* run all scripts from the repository root;
* use the same temporal configuration during dataset generation and training;
* regenerate the arrays whenever the segmentation or label-refinement settings change;
* do not reuse checkpoints created with incompatible experiment settings;
* keep results from different window configurations in their automatically generated directories.

The public dataset is available at:

> **IPqM-Fall: Multi-Sensor Wearable Dataset for Military Activities and Fall Detection.** Zenodo. https://doi.org/10.5281/zenodo.20431609

The accompanying paper is currently titled:

> **Military Fall Detection and Activity Classification Using Wearable Inertial Sensors.**

The complete paper citation will be added after publication. If you use the IPqM-Fall dataset or this repository in academic work, please cite both the Zenodo dataset release and the accompanying publication.

### References for Baseline Architectures

#### CNN1Conv

Santos, G. L., Endo, P. T., Monteiro, K. H. d. C., Rocha, E. S., Silva, I., and Lynn, T. (2019). *Accelerometer-Based Human Fall Detection Using Convolutional Neural Networks*. *Sensors*, 19(7), 1644.

DOI: https://doi.org/10.3390/s19071644

#### DeepConvLSTM

Ordóñez, F. J., and Roggen, D. (2016). *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*. *Sensors*, 16(1), 115.

DOI: https://doi.org/10.3390/s16010115

#### Classical and Sensor-Placement Baselines

Özdemir, A. T. (2016). *An Analysis on Sensor Locations of the Human Body for Wearable Fall Detection Devices: Principles and Practice*. *Sensors*, 16(8), 1161.

DOI: https://doi.org/10.3390/s16081161

#### Multimodal Fall-Detection Baseline

Martínez-Villaseñor, L., Ponce, H., Brieva, J., Moya-Albor, E., Núñez-Martínez, J., and Peñafort-Asturiano, C. (2019). *UP-Fall Detection Dataset: A Multimodal Approach*. *Sensors*, 19(9), 1988.

DOI: https://doi.org/10.3390/s19091988
