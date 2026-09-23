# TTL Assignment 04 — Vehicle Price Prediction with a Preprocessing Pipeline

## Overview
This notebook predicts used-vehicle prices from mixed numeric and categorical features, using a full `scikit-learn` `Pipeline` (preprocessing + model) and comparing three regression algorithms.

## What the notebook does
1. **Synthetic data generation** — Creates 600 simulated vehicles with `brand` (Toyota/Honda/Ford/BMW/Hyundai), `age_years`, `mileage_km`, `engine_size_L`, and `fuel_type` (Petrol/Diesel/Electric). `price` is derived from a formula incorporating brand premiums, fuel-type premiums, depreciation by age/mileage, and noise.
2. **Preprocessing pipeline** — Builds a `ColumnTransformer` that standard-scales numeric features (`age_years`, `mileage_km`, `engine_size_L`) and one-hot encodes categorical features (`brand`, `fuel_type`).
3. **Model comparison** — Wraps the preprocessor and each model in a `Pipeline`, training and evaluating three regressors:
   - Linear Regression
   - Random Forest Regressor (300 trees)
   - Gradient Boosting Regressor
   
   Compares them via MAE, RMSE, and R² in a results table.
4. **Automatic best-model selection & visualization** — Selects the model with the lowest RMSE automatically and plots its Actual vs. Predicted price scatter plot with a reference diagonal.

## Key libraries
`numpy`, `pandas`, `matplotlib`, `scikit-learn` (`ColumnTransformer`, `Pipeline`, `StandardScaler`, `OneHotEncoder`, `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`)

## Outcome
A results table ranking all three models by MAE/RMSE/R², with the best model (by RMSE) automatically identified and its predictions visualized.

## How to run
Run all cells top to bottom. `np.random.seed(42)` ensures reproducible synthetic data and results.
