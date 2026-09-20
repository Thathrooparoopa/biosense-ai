# BioSense AI - S-O-H Data Preprocessing

This directory contains preprocessing scripts for the Scent of Health
(S-O-H) electronic-nose research dataset.

## Current Research Task

The first experimental task is binary classification:

- Z00 → Healthy Control → 0
- C34 → Lung Cancer → 1

This is a research classification task and is not a clinical diagnostic system.

## Sensor Data

The preprocessing pipeline uses the 17 primary eNose channels:

R1, R2, R3, R4, R5, R6, R7, R8, R9,
R10, R11, R12, R13, R14, R15, R16, R17

Auxiliary sensors are not included in this first baseline dataset.

## Feature Extraction

Each patient's sensor time series is converted into patient-level
statistical features.

For each sensor channel, the pipeline calculates:

- mean
- standard deviation
- minimum
- maximum
- median
- range
- first value
- last value
- mean absolute difference
- standard deviation of first differences

The resulting dataset contains one row per patient.

## Leakage Prevention

The following fields are not used as model features:

- Patient_id
- Diagnosis
- D_class
- D_bin_class
- Datetime
- Week
- Site

Patient metadata is retained separately for later analysis and
drift-aware evaluation.

## Output

The preprocessing pipeline generates:

dataset/processed/soh_binary_features.csv

and:

dataset/processed/preprocessing_report.txt

## Medical Disclaimer

This application is an experimental research prototype and is not a
medical diagnostic device. Its results must not be used to diagnose,
treat, or rule out cancer. Medical decisions should be made only by
qualified healthcare professionals.