# TTL Assignment 09 — Feature Importance Visualization

## Overview
This notebook trains a regression model on automotive data and visualizes/interprets which features drive vehicle price predictions, using three complementary interpretability techniques.

## What the notebook does
1. **Synthetic automotive dataset generation** — Creates 700 simulated vehicles with `engine_size_L`, `horsepower`, `weight_kg`, `age_years`, `mileage_km`, `n_doors`, and `brand_score`. `price` is derived from a realistic formula combining these factors plus noise.
2. **Model training** — Splits the data 80/20 and trains a `RandomForestRegressor` (300 trees) to predict price, reporting MAE and R² on the test set.
3. **Built-in feature importance** — Plots the Random Forest's impurity-based `feature_importances_` as a horizontal bar chart, ranking features by importance.
4. **Permutation importance** — Computes and plots permutation importance (20 repeats), which is more robust than impurity-based importance since it measures the actual drop in test-set R² when each feature is shuffled, with error bars for variability.
5. **SHAP analysis** — Uses `shap.TreeExplainer` to compute SHAP values and produces a SHAP summary plot, showing both the magnitude and direction (positive/negative impact) of each feature on individual price predictions.

## Key libraries
`numpy`, `pandas`, `matplotlib`, `scikit-learn` (`RandomForestRegressor`, `permutation_importance`), `shap`

## Outcome
Three different views of feature importance (impurity-based, permutation-based, and SHAP) for the same trained model, letting you compare and validate which features (e.g. horsepower, age, brand_score) most strongly influence predicted vehicle price — and in which direction.

## Important note
The dataset is **synthetically generated** with a known underlying price formula, which is ideal for teaching/demoing interpretability methods (you can sanity-check that the "discovered" important features match the formula). Swap in a real automotive dataset for an actual business use case.

## How to run
Run all cells top to bottom in a Jupyter environment with `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `shap` installed (`pip install shap` if not already available). `np.random.seed(42)` ensures reproducible results.
