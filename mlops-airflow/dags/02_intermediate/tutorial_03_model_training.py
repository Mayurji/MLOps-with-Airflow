#!/usr/bin/env python3
"""
Tutorial 03 - Model Training Pipeline
=======================================
Level    : BASIC → INTERMEDIATE
Topic    : Training, evaluating, and saving ML models with Airflow.

Concepts Covered:
  - Training multiple models in parallel (fan-out)
  - Model comparison and selection (best model)
  - Saving models with joblib / pickle
  - Metrics logging to JSON (foundation for MLflow in later tutorials)

Run:
  airflow dags test tutorial_03_model_training 2024-01-01
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from airflow import DAG
from airflow.decorators import task
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────
PROJECT_ROOT  = Path(os.getenv("PROJECT_ROOT", "/home/mayur/Desktop/MLOps-Airflow-Tutorials/mlops-airflow"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
METRICS_DIR   = PROJECT_ROOT / "logs" / "metrics"

FEATURE_COLS = ["area_sqft", "bedrooms", "bathrooms", "house_age",
                "distance_km", "garage_spots", "price_per_sqft",
                "bed_bath_ratio", "is_new_house"]
TARGET_COL   = "price_usd"

default_args = {
    "owner": "mlops-learner",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


# SHARED HELPER
def _load_xy():
    """Load features and target from processed Parquet."""
    df = pd.read_parquet(PROCESSED_DIR / "housing_clean.parquet")
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    return X, y

def _compute_metrics(y_true, y_pred, model_name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {
        "model": model_name,
        "mae":   round(mae, 2),
        "mse":   round(mse, 2),
        "rmse":  round(rmse, 2),
        "r2":    round(r2, 4),
    }

# TASKS
@task(task_id="prepare_data_splits")
def prepare_data_splits():
    """Split data into train/val/test and save splits."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = _load_xy()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    splits_path = MODELS_DIR / "data_splits.pkl"
    joblib.dump({
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }, splits_path)

    # Fit and save scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    log.info("📂 Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))
    return str(splits_path)


@task(task_id="train_ridge_regression")
def train_ridge_regression(splits_path: str):
    """Train Ridge Regression (baseline model)."""
    splits = joblib.load(splits_path)
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    X_train_s = scaler.transform(splits["X_train"])
    X_val_s   = scaler.transform(splits["X_val"])

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, splits["y_train"])

    val_pred = model.predict(X_val_s)
    metrics  = _compute_metrics(splits["y_val"], val_pred, "ridge_regression")

    # Cross-validation R²
    cv_scores = cross_val_score(model, X_train_s, splits["y_train"], cv=5, scoring="r2")
    metrics["cv_r2_mean"]  = round(cv_scores.mean(), 4)
    metrics["cv_r2_std"]   = round(cv_scores.std(), 4)

    # Save
    joblib.dump(model, MODELS_DIR / "ridge_model.pkl")
    (METRICS_DIR / "ridge_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("🔵 Ridge  → R²=%.4f | RMSE=$%.0f", metrics["r2"], metrics["rmse"])
    return metrics


@task(task_id="train_random_forest")
def train_random_forest(splits_path: str):
    """Train Random Forest Regressor."""
    splits = joblib.load(splits_path)

    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=5,
        n_jobs=-1, random_state=42
    )
    model.fit(splits["X_train"], splits["y_train"])

    val_pred = model.predict(splits["X_val"])
    metrics  = _compute_metrics(splits["y_val"], val_pred, "random_forest")

    # Feature importances
    fi = dict(zip(FEATURE_COLS, model.feature_importances_.round(4)))
    metrics["feature_importances"] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    joblib.dump(model, MODELS_DIR / "rf_model.pkl")
    (METRICS_DIR / "rf_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("🌲 RF     → R²=%.4f | RMSE=$%.0f", metrics["r2"], metrics["rmse"])
    return metrics


@task(task_id="train_gradient_boosting")
def train_gradient_boosting(splits_path: str):
    """Train Gradient Boosting Regressor."""
    splits = joblib.load(splits_path)

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42
    )
    model.fit(splits["X_train"], splits["y_train"])

    val_pred = model.predict(splits["X_val"])
    metrics  = _compute_metrics(splits["y_val"], val_pred, "gradient_boosting")

    joblib.dump(model, MODELS_DIR / "gb_model.pkl")
    (METRICS_DIR / "gb_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("⚡ GBM    → R²=%.4f | RMSE=$%.0f", metrics["r2"], metrics["rmse"])
    return metrics


@task(task_id="select_best_model", trigger_rule=TriggerRule.ALL_SUCCESS)
def select_best_model(ridge_m: dict, rf_m: dict, gb_m: dict):
    """Compare all models and select the winner based on validation R²."""
    candidates = [ridge_m, rf_m, gb_m]
    best = max(candidates, key=lambda m: m["r2"])

    model_file_map = {
        "ridge_regression":  "ridge_model.pkl",
        "random_forest":     "rf_model.pkl",
        "gradient_boosting": "gb_model.pkl",
    }

    # Copy best model to a canonical name
    import shutil
    src = MODELS_DIR / model_file_map[best["model"]]
    dst = MODELS_DIR / "best_model.pkl"
    shutil.copy(src, dst)

    summary = {
        "winner":     best["model"],
        "val_r2":     best["r2"],
        "val_rmse":   best["rmse"],
        "all_models": {m["model"]: {"r2": m["r2"], "rmse": m["rmse"]} for m in candidates},
        "selected_at": datetime.utcnow().isoformat(),
    }
    (MODELS_DIR / "selection_summary.json").write_text(json.dumps(summary, indent=2))

    log.info("🏆 WINNER: %s  (R²=%.4f, RMSE=$%.0f)",
             best["model"], best["r2"], best["rmse"])
    return summary


@task(task_id="evaluate_on_test_set")
def evaluate_on_test_set(selection: dict):
    """Run final evaluation on held-out test set."""
    splits = joblib.load(MODELS_DIR / "data_splits.pkl")
    model  = joblib.load(MODELS_DIR / "best_model.pkl")

    winner = selection["winner"]
    if winner == "ridge_regression":
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        X_test = scaler.transform(splits["X_test"])
    else:
        X_test = splits["X_test"]

    test_pred = model.predict(X_test)
    test_metrics = _compute_metrics(splits["y_test"], test_pred, f"{winner}_test")

    log.info("🎯 TEST SET RESULTS")
    log.info("   Model : %s", winner)
    log.info("   R²    : %.4f", test_metrics["r2"])
    log.info("   MAE   : $%.0f", test_metrics["mae"])
    log.info("   RMSE  : $%.0f", test_metrics["rmse"])

    (METRICS_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    return test_metrics


# ──────────────────────────────────────────────
# DAG DEFINITION
# ──────────────────────────────────────────────
with DAG(
    dag_id="tutorial_03_model_training",
    description="Intermediate: parallel model training, comparison, and selection",
    default_args=default_args,
    schedule_interval=None,          # Triggered manually / by upstream DAG
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "intermediate", "level-03", "training"],
) as dag:

    splits = prepare_data_splits()

    # Fan-out: train 3 models in parallel
    ridge = train_ridge_regression(splits)
    rf    = train_random_forest(splits)
    gb    = train_gradient_boosting(splits)

    # Fan-in: compare and select
    selection = select_best_model(ridge, rf, gb)

    # Final evaluation
    evaluate_on_test_set(selection)
