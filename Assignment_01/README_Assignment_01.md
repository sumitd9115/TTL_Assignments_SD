# TTL Assignment 01 — Car Fuel Efficiency (MPG) Prediction

## Overview
This notebook builds a regression model to predict a car's fuel efficiency (MPG) from basic vehicle specs, using a synthetic dataset generated with NumPy.

## What the notebook does
1. **Synthetic data generation** — Creates 500 simulated cars with `engine_size_L`, `horsepower`, `weight_kg`, and `cylinders`, then derives `mpg` from a formula with added noise (so relationships are realistic but not perfectly linear).
2. **Train/test split & model training** — Splits data 80/20 and trains two models:
   - Linear Regression
   - Random Forest Regressor (200 trees)
3. **Evaluation** — Compares both models using MAE, RMSE, and R² on the test set.
4. **Visualization** — Plots Actual vs. Predicted MPG for the best-performing model (Random Forest) with a "perfect prediction" reference line.

## Key libraries
`numpy`, `pandas`, `matplotlib`, `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, metrics)

## Outcome
Random Forest is manually selected as the better-performing model based on test metrics, and its predictions are visualized against actual values.

## How to run
Run all cells top to bottom in a Jupyter environment with `numpy`, `pandas`, `matplotlib`, and `scikit-learn` installed. `np.random.seed(42)` ensures reproducible results.
