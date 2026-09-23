# TTL Assignment 05 — Predictive Maintenance (Equipment Failure Classification)

## Overview
This notebook builds a binary classifier to predict equipment failure from sensor readings, simulating a typical industrial predictive-maintenance use case.

## What the notebook does
1. **Synthetic sensor data generation** — Creates 800 simulated equipment readings with `temperature_C`, `vibration_mm_s`, `pressure_psi`, `rpm`, and `run_hours`. A `failure_score` is computed from a weighted combination of deviation in temperature/vibration and run hours (plus noise), and thresholded to create a binary `failure` label. Reports the overall failure rate.
2. **Train/test split & scaling** — Splits data 75/25 with stratification on the `failure` label (to preserve class balance), then standard-scales the features.
3. **Model training** — Trains a `RandomForestClassifier` (300 trees, `class_weight="balanced"` to handle class imbalance) on the scaled training data.
4. **Evaluation** — Prints a classification report (precision/recall/F1 for "Healthy" vs "Failure") and plots a confusion matrix to visualize prediction performance.

## Key libraries
`numpy`, `pandas`, `matplotlib`, `scikit-learn` (`train_test_split`, `StandardScaler`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`)

## Outcome
A trained classifier with reported precision/recall/F1 for detecting equipment failures, and a confusion matrix showing the breakdown of correct vs. incorrect predictions on the test set.

## How to run
Run all cells top to bottom. `np.random.seed(42)` ensures reproducible synthetic data and results. Note the class imbalance in the data is explicitly handled via stratified splitting and balanced class weights.
