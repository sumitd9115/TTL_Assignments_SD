# TTL Assignment 06 — Traffic Sign Classification using CNN

## Overview
This notebook trains a Convolutional Neural Network (CNN) to classify traffic signs into 8 categories, using a simulated GTSRB-style (German Traffic Sign Recognition Benchmark) image dataset.

## What the notebook does
1. **Synthetic image dataset generation** — Creates 32x32 RGB images for 8 sign classes (Speed 30/60/90, Stop, No Entry, Yield, Pedestrian Crossing, No Overtaking), each with a distinct base color and a shape marker (circle region) plus noise, so the CNN has learnable visual structure. Displays one sample image per class.
2. **Train/test split** — Splits the dataset 80/20 with stratification and one-hot encodes the labels.
3. **CNN architecture** — Builds a Keras `Sequential` model: two Conv2D+MaxPooling blocks (32 then 64 filters), followed by Flatten → Dense(128) → Dropout(0.3) → Dense(8, softmax).
4. **Training** — Trains for 10 epochs with a 15% validation split.
5. **Evaluation & visualization** — Reports test accuracy/loss, plots training/validation accuracy and loss curves, and shows a full classification report + confusion matrix over the 8 sign classes.

## Key libraries
`numpy`, `matplotlib`, `tensorflow`/`keras`, `scikit-learn` (train/test split, classification metrics)

## Outcome
A trained CNN with reported test accuracy and a confusion matrix showing per-class classification performance on the simulated sign dataset.

## Important note
The dataset here is **synthetically generated** (not the real GTSRB dataset) since it wasn't available in this environment. The CNN architecture and training/evaluation code are fully correct and directly reusable — swap in the real GTSRB dataset (resized to 32x32 RGB) to get real-world results before presenting this as a finished project.

## How to run
Run all cells top to bottom in a Jupyter environment with `tensorflow`, `numpy`, `matplotlib`, and `scikit-learn` installed. Seeds are fixed (`np.random.seed(42)`, `tf.random.set_seed(42)`) for reproducibility.
