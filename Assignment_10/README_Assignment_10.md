# TTL Assignment 10 — MLOps Workflow Simulation (MLflow + Docker + CI/CD)

## Overview
This notebook simulates a complete MLOps workflow for a vehicle price prediction model: experiment tracking, model registration, containerization, and CI/CD deployment automation.

## What the notebook does
1. **Data & baseline setup** — Generates a small simulated vehicle price dataset (`engine_size_L`, `horsepower`, `age_years` → `price`) and splits it 80/20 for training/testing.
2. **Experiment tracking with MLflow** — Sets up a local MLflow tracking store and experiment (`vehicle_price_ci_cd`). Trains two models (Linear Regression and Random Forest) inside `mlflow.start_run()` blocks, logging parameters, metrics (MAE, RMSE, R²), and the trained model artifact for each run.
3. **Model selection & registration** — Queries the MLflow tracking server for all runs in the experiment, automatically selects the best one by lowest RMSE, and registers it to the MLflow Model Registry as `vehicle_price_predictor`.
4. **Containerization** — Programmatically generates a `Dockerfile` that installs dependencies and serves the registered MLflow model via `mlflow models serve` on port 5001.
5. **CI/CD pipeline definition** — Generates a `ci_cd_pipeline.yaml` (GitHub Actions workflow) that automates: checkout → install dependencies → run training/MLflow logging → build Docker image → push to a registry → deploy the container.

## Key libraries
`numpy`, `pandas`, `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, metrics), `mlflow` (tracking, model registry), plus generated `Dockerfile` and GitHub Actions YAML

## Outcome
A full, runnable MLOps loop: trained models are tracked and compared in MLflow, the best model is registered, and both a `Dockerfile` and a CI/CD YAML file are generated on disk — mirroring a real production ML deployment pipeline end to end.

## Important note
The Dockerfile and CI/CD YAML are **generated as files** by the notebook but not actually built/executed here (no Docker daemon or CI runner in this environment). To demo this fully:
- Run `docker build -t vehicle-price-model .` locally using the generated `Dockerfile`.
- Push `ci_cd_pipeline.yaml` to `.github/workflows/` in a GitHub repo to activate the pipeline (update the registry name/credentials first).

## How to run
Run all cells top to bottom in a Jupyter environment with `mlflow`, `scikit-learn`, `numpy`, and `pandas` installed (`pip install mlflow`). This creates an `mlruns/` folder locally to store experiment tracking data, plus `Dockerfile` and `ci_cd_pipeline.yaml` in the working directory. `np.random.seed(42)` ensures reproducible data.
