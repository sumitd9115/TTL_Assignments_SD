# TTL Assignment 03 — Data Cleaning & Preprocessing Pipeline (Used Car Data)

## Overview
This notebook demonstrates a full data-cleaning workflow — handling missing values, capping outliers, and encoding/scaling features — on a deliberately "messy" synthetic used-car dataset.

## What the notebook does
1. **Synthetic messy data generation** — Creates 400 simulated used cars (`mileage_km`, `age_years`, `engine_size_L`, `fuel_type`, `price`), then deliberately injects:
   - Missing values (3–5% per column) in `mileage_km`, `age_years`, `price`, `fuel_type`
   - Extreme outliers in `price` (8 rows, 3–5x inflated) and `mileage_km` (6 rows, 3–6x inflated)
2. **Imputation** — Fills missing numeric values with the median (`SimpleImputer`) and missing categorical values with the most frequent category.
3. **Outlier capping** — Uses the IQR method (`cap_outliers_iqr`) to clip extreme values in `price` and `mileage_km` to acceptable bounds, printing how many outliers were capped per column.
4. **Before/after visualization** — Side-by-side boxplots comparing `price` and `mileage_km` distributions before and after outlier capping.
5. **Encoding & scaling** — One-hot encodes `fuel_type` (dropping the first category) and standard-scales all numeric columns, producing a final `processed` DataFrame ready for modeling.

## Key libraries
`numpy`, `pandas`, `matplotlib`, `scikit-learn` (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`)

## Outcome
A fully cleaned, encoded, and scaled dataset (`processed`) with no missing values and outliers controlled, ready to feed into a downstream model.

## How to run
Run all cells top to bottom. `np.random.seed(42)` ensures the same messy dataset (and same missing/outlier positions) is generated each run.
