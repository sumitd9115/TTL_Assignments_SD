# TTL Assignment 08 — Sentiment Analysis using LSTM

## Overview
This notebook builds an LSTM-based deep learning model to classify vehicle owner feedback as positive or negative sentiment.

## What the notebook does
1. **Simulated feedback dataset generation** — Generates 1,000 vehicle feedback reviews by sampling from curated positive phrases (e.g. "smooth ride and great mileage") and negative phrases (e.g. "poor fuel economy"), with light random word-order noise added for variety. Labels are binary (1 = positive, 0 = negative).
2. **Text preprocessing** — Tokenizes the review text with Keras `Tokenizer` (vocabulary size 500) and pads all sequences to a fixed length of 12 tokens.
3. **Train/test split** — Splits the padded sequences 80/20 with stratification on sentiment label.
4. **LSTM model** — Builds a Keras model: `Embedding` layer → `LSTM(32)` → `Dense(16, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)` for binary classification.
5. **Training & evaluation** — Trains for 8 epochs with a 15% validation split, then reports test accuracy/loss, a classification report, and a confusion matrix.
6. **Manual inference demo** — Runs the trained model on 3 brand-new, unseen feedback sentences and prints the predicted sentiment with confidence score for each.

## Key libraries
`numpy`, `pandas`, `matplotlib`, `tensorflow`/`keras` (`Tokenizer`, `pad_sequences`, `Embedding`, `LSTM`), `scikit-learn` (split, classification metrics)

## Outcome
A trained LSTM sentiment classifier with reported accuracy/precision/recall, a confusion matrix, and live predictions on new example reviews demonstrating the model generalizes to unseen text.

## Important note
The feedback dataset is **synthetically generated** from a small set of template phrases (not scraped real customer reviews). This keeps the demo self-contained and reproducible, but for a real-world evaluation, swap in an actual vehicle review dataset (e.g. from a dealership or review site) — the tokenization/LSTM/evaluation pipeline will work unchanged.

## How to run
Run all cells top to bottom in a Jupyter environment with `tensorflow`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn` installed. Seeds are fixed (`np.random.seed(42)`, `random.seed(42)`) for reproducibility.
