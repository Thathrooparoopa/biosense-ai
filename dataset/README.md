# BioSense AI Dataset

## Primary Dataset

BioSense AI uses the **Scent of Health (S-O-H)** public electronic-nose
dataset as the primary research dataset for development and evaluation.

The dataset contains clinical breath measurements collected using an
electronic-nose sensor array.

## Dataset Overview

- Dataset: Scent of Health (S-O-H)
- Data type: Exhaled-breath electronic-nose measurements
- Patients: 1,234
- Diagnostic groups: 9
- eNose channels: 17
- Sampling rate: approximately 0.4 Hz
- Measurement duration: approximately 895 seconds per sample
- Clinical sites: 2
- Collection period: 13 consecutive weeks

The dataset includes a lung-cancer class identified by ICD-10 code C34.

## Initial BioSense AI Research Task

The first machine-learning task will investigate binary classification
between:

1. Healthy control
2. Lung cancer

The exact preprocessing, feature extraction, model architecture,
train/test strategy, and evaluation metrics will be defined in later
phases.

## Dataset Source

Dataset:
Scent of Health (S-O-H)

Repository:
https://huggingface.co/datasets/ivanpodd/S-OH

Dataset DOI:
10.57967/hf/9752

License:
MIT according to the dataset repository.

## Data Storage

The original dataset should not be committed directly to the Git
repository.

Recommended local structure:

```text
dataset/
├── raw/
│   └── downloaded S-O-H files
├── processed/
│   └── generated feature datasets
└── README.md