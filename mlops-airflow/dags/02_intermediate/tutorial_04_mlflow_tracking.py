#!/usr/bin/env python3
"""
Tutorial 04 - MLflow Experiment Tracking
==========================================
Level    : INTERMEDIATE
Topic    : Integrating MLflow for experiment tracking, model registry, and artifact storage.

Concepts Covered:
  - MLflow Tracking: logging params, metrics, artifacts
  - MLflow Model Registry: registering and staging models
  - Using Airflow + MLflow together
  - Comparing runs via MLflow UI

Prerequisites:
  Start MLflow server: mlflow server --host 0.0.0.0 --port 5000

Run:
  airflow dags test tutorial_04_mlflow_tracking 2024-01-01
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from airflow import DAG
from airflow.decorators import task

log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
PROJECT_ROOT  = Path(os.getenv("PROJECT_ROOT", "/home/mayur/Desktop/mlops-airflow"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT    = "housing_price_prediction"

FEATURE_COLS = ["area_sqft", "bedrooms", "bathrooms", "house_age",
                "distance_km", "garage_spots", "price_per_sqft",
                "bed_bath_ratio", "is_new_house"]
TARGET_COL   = "price_usd"

default_args = {
    "owner": "mlops-learner",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)


def _get_data():
    df = pd.read_parquet(PROCESSED_DIR / "housing_clean.parquet")
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ──────────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────────

@task(task_id="run_ridge_experiment")
def run_ridge_experiment():
    """Log Ridge Regression run to MLflow."""
    _setup_mlflow()
    X_train, X_test, y_train, y_test = _get_data()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    params = {"alpha": 1.0, "model_type": "ridge"}
    model  = Ridge(**{k: v for k, v in params.items() if k != "model_type"})

    with mlflow.start_run(run_name="Ridge_Regression") as run:
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        metrics = {
            "mae":  round(mean_absolute_error(y_test, preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "r2":   round(r2_score(y_test, preds), 4),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        log.info("🔵 Ridge run_id=%s  R²=%.4f", run_id, metrics["r2"])

    return {"run_id": run_id, "model": "ridge", **metrics}


@task(task_id="run_rf_experiment")
def run_rf_experiment():
    """Log Random Forest run to MLflow with param grid."""
    _setup_mlflow()
    X_train, X_test, y_train, y_test = _get_data()

    params = {
        "n_estimators": 150,
        "max_depth":    12,
        "min_samples_split": 4,
        "model_type": "random_forest",
    }
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        n_jobs=-1, random_state=42
    )

    with mlflow.start_run(run_name="Random_Forest") as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "mae":  round(mean_absolute_error(y_test, preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "r2":   round(r2_score(y_test, preds), 4),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model",
                                 registered_model_name="HousingPricePredictor")

        # Log feature importance as artifact
        fi = dict(zip(FEATURE_COLS, model.feature_importances_.round(4)))
        fi_path = MODELS_DIR / "feature_importance.json"
        fi_path.write_text(json.dumps(dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)), indent=2))
        mlflow.log_artifact(str(fi_path))

        run_id = run.info.run_id
        log.info("🌲 RF     run_id=%s  R²=%.4f", run_id, metrics["r2"])

    return {"run_id": run_id, "model": "random_forest", **metrics}


@task(task_id="run_gb_experiment")
def run_gb_experiment():
    """Log Gradient Boosting with nested params."""
    _setup_mlflow()
    X_train, X_test, y_train, y_test = _get_data()

    params = {
        "n_estimators":  200,
        "learning_rate": 0.08,
        "max_depth":     5,
        "subsample":     0.85,
        "model_type":    "gradient_boosting",
    }
    model = GradientBoostingRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        random_state=42,
    )

    with mlflow.start_run(run_name="GradientBoosting") as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "mae":  round(mean_absolute_error(y_test, preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "r2":   round(r2_score(y_test, preds), 4),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model",
                                 registered_model_name="HousingPricePredictor")

        run_id = run.info.run_id
        log.info("⚡ GBM    run_id=%s  R²=%.4f", run_id, metrics["r2"])

    return {"run_id": run_id, "model": "gradient_boosting", **metrics}


@task(task_id="compare_and_promote")
def compare_and_promote(ridge: dict, rf: dict, gb: dict):
    """
    Compare all runs, promote the best model to 'Staging' in MLflow Registry.
    """
    _setup_mlflow()

    candidates = [ridge, rf, gb]
    best = max(candidates, key=lambda x: x["r2"])

    log.info("📊 Experiment Summary:")
    for c in candidates:
        marker = "🏆" if c["model"] == best["model"] else "  "
        log.info("  %s %-20s  R²=%.4f  RMSE=$%.0f",
                 marker, c["model"], c["r2"], c["rmse"])

    # Promote best model in registry (if it was registered)
    if best["model"] != "ridge":
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.get_latest_versions(
                "HousingPricePredictor", stages=["None"]
            )
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                client.transition_model_version_stage(
                    name="HousingPricePredictor",
                    version=latest.version,
                    stage="Staging",
                )
                log.info("🚀 Promoted '%s' v%s → Staging",
                         "HousingPricePredictor", latest.version)
        except Exception as e:
            log.warning("Could not promote model: %s", e)

    return best


# ──────────────────────────────────────────────
# DAG DEFINITION
# ──────────────────────────────────────────────
with DAG(
    dag_id="tutorial_04_mlflow_tracking",
    description="Intermediate: MLflow experiment tracking & model registry",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "intermediate", "level-04", "mlflow"],
) as dag:

    ridge_result = run_ridge_experiment()
    rf_result    = run_rf_experiment()
    gb_result    = run_gb_experiment()

    compare_and_promote(ridge_result, rf_result, gb_result)
