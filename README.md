# IPqM-Fall: Real-Time Military Activity Recognition and Fall Detection with Wearable Inertial Sensors

This repository contains the official machine learning pipeline for the **IPqM-Fall** dataset. Designed for tactical and military environments, it supports Human Activity Recognition (HAR) and Fall Detection using wearable inertial measurement units (IMUs) positioned on the chest and both wrists.

The framework supports configurable window sizes, strides, and sampling rates through `src/config.py`, allowing researchers to reproduce experiments under different temporal segmentation settings.

## 1. Getting Started

The raw continuous IMU recordings are hosted on Zenodo. To reproduce the experiments, first download the dataset and place the files in the expected directory structure.

### 1.1 Download and Organize the Dataset

1. Download the raw dataset from Zenodo: `doi.org/10.5281/zenodo.20431609`.
2. Extract the downloaded archive.
3. Organize the files as follows:

```text
project_root/
├── IPqM-Fall/
│   └── raw/
│       ├── ID1/
│       ├── ID2/
│       └── ...
├── src/
└── README.md
```

### 1.2 Generate Training Arrays

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

The pipeline creates the sensor tensors (`X_chest.npy`, `X_left.npy`, and `X_right.npy`), subject identifiers for Leave-One-Subject-Out (LOSO) cross-validation, and the label arrays required for the supported recognition tasks.

Generated datasets are stored under:

```text
IPqM-Fall/windowed/<window-configuration>/
```

allowing multiple window configurations to coexist simultaneously.

### 1.3 Dataset Configuration

Dataset generation is fully parameterized through:

```text
src/config.py
```

Key parameters include:

```python
FS = 90

WINDOW_SEC = 2
STRIDE_SEC = 1
```

allowing experiments with different window lengths, overlap ratios, and sampling configurations without modifying the preprocessing scripts.

---

## 2. Training and Evaluation

The repository provides a unified framework for evaluating both classical machine learning and deep learning approaches using LOSO cross-validation. To facilitate comparison with previous work, the implemented architectures and their initial hyperparameters were adopted from the original publications and are provided as baseline configurations. These settings have not yet been systematically optimized for the IPqM-Fall dataset.

The framework includes both classical machine learning baselines and neural-network-based approaches commonly used in wearable sensing research.

### Supported Architectures

#### Classical Machine Learning

* Support Vector Machines (SVM)
* Random Forests (RF)
* K-Nearest Neighbors (KNN)

These models serve as established baselines for fall detection and activity recognition using wearable sensor data.

#### Deep Learning

* **CNN1Conv**: A lightweight convolutional architecture designed for accelerometer-based fall detection.
* **CNN3B3Conv**: A deeper convolutional architecture composed of three convolutional blocks for enhanced temporal feature extraction.
* **DeepConvLSTM**: A hybrid architecture that combines convolutional feature extraction with recurrent temporal modeling.
* **LSTM**

### Sensor Fusion

The framework supports both early and late sensor fusion strategies.

In the early-fusion configuration, signals from the chest and wrist sensors are combined before being presented to the model, allowing cross-sensor relationships to be learned directly from the data.

In the late-fusion configuration, independent models are trained on each sensor stream and their predictions are aggregated through ensemble methods such as majority voting. This approach can improve robustness when individual sensors become unreliable or unavailable.

### Multi-Task and Multi-Model Learning

The framework can be configured either as a collection of task-specific models or as a unified multi-task learning (MTL) system.

In the multi-model configuration, separate networks are trained for each prediction task, such as binary fall detection and posture classification.

In the MTL configuration, a shared feature-extraction backbone feeds multiple task-specific output heads, enabling joint learning of fall detection, fall classification, posture recognition, movement recognition, and transition recognition tasks. Task-specific loss weights can be configured through `src/config.py`.

### Training

Model training, evaluation, class weighting, metric reporting, and fusion handling are managed through a centralized training script:

```bash
python src/train.py
```

Experiment settings, model selection, paths, and training parameters can be configured in:

```text
src/config.py
```

The configuration file controls:

* Dataset generation parameters
* Model selection
* STL and MTL settings
* Learning rate
* Dropout
* Number of epochs
* Multi-task loss weights
* Sensor-fusion configurations

The training pipeline reports standard evaluation metrics including Accuracy, Precision, Recall, F1-score, and Confusion Matrices under LOSO cross-validation.

All experiments are evaluated using Leave-One-Subject-Out (LOSO) cross-validation to assess subject-independent generalization and avoid subject-specific information leakage between training and testing sets.

---

## 3. Deployment

Trained PyTorch models can be converted to optimized INT8 TensorFlow Lite models for deployment on smartphones, embedded devices, and wearable platforms.

```bash
python src/extra/tflite_converter.py
```

The conversion workflow follows:

```text
PyTorch → ONNX → TensorFlow → INT8 TensorFlow Lite
```

---

## 4. Reproducibility and Citation

The original preprocessing scripts are included for transparency and reproducibility purposes. Researchers may regenerate datasets with different window lengths, strides, and sampling rates through `src/config.py`, allowing systematic evaluation of segmentation strategies and temporal modeling choices.

If you use the IPqM-Fall dataset or this repository in academic work, please cite the accompanying publication and Zenodo dataset release.

### References for Baseline Architectures

#### [CNN1Conv]

Santos, G. L., Endo, P. T., Monteiro, K. H. d. C., Rocha, E. S., Silva, I., & Lynn, T. (2019). *Accelerometer-Based Human Fall Detection Using Convolutional Neural Networks*. *Sensors*, 19(7), 1644.

DOI: https://doi.org/10.3390/s19071644

#### [DeepConvLSTM]

Ordóñez, F., & Roggen, D. (2016). *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*. *Sensors*, 16(1), 115.

DOI: https://doi.org/10.3390/s16010115

#### [Classical/Sensor Placement]

Özdemir, A. T. (2016). *An Analysis on Sensor Locations of the Human Body for Wearable Fall Detection Devices: Principles and Practice*. *Sensors*, 16(8), 1161.

DOI: https://doi.org/10.3390/s16081161

#### [Classical/Multimodal Baseline]

Martínez-Villaseñor, L., Ponce, H., Brieva, J., Moya-Albor, E., Núñez-Martínez, J., & Peñafort-Asturiano, C. (2019). *UP-Fall Detection Dataset: A Multimodal Approach*. *Sensors*, 19(9), 1988.

DOI: https://doi.org/10.3390/s19091988
