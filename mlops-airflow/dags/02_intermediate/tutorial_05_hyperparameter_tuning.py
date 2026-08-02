#!/usr/bin/env python3
"""
Tutorial 05 - Hyperparameter Tuning with Optuna
=================================================
Level    : INTERMEDIATE
Topic    : Automated hyperparameter search integrated into Airflow pipelines.

Concepts Covered:
  - Optuna TPE sampler for intelligent search
  - Parallel Airflow tasks for different model families
  - Logging study results to MLflow
  - Pruning unpromising trials (early stopping)

Run:
  airflow dags test tutorial_05_hyperparameter_tuning 2024-01-01
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from airflow import DAG
from airflow.decorators import task

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ──────────────────────────────────────────────────
PROJECT_ROOT  = Path(os.getenv("PROJECT_ROOT", "/home/mayur/Desktop/MLOps-Airflow-Tutorials/mlops-airflow"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

FEATURE_COLS = ["area_sqft", "bedrooms", "bathrooms", "house_age",
                "distance_km", "garage_spots", "price_per_sqft",
                "bed_bath_ratio", "is_new_house"]
TARGET_COL = "price_usd"
N_TRIALS   = 30   # Increase to 100+ for production

default_args = {
    "owner": "mlops-learner",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _load_data():
    df = pd.read_parquet(PROCESSED_DIR / "housing_clean.parquet")
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# OPTUNA OBJECTIVE FUNCTIONS

def _rf_objective(trial, X_train, y_train):
    """Optuna objective: maximize CV R² for Random Forest."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    model  = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
    return scores.mean()


def _gb_objective(trial, X_train, y_train):
    """Optuna objective: maximize CV R² for Gradient Boosting."""
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":     trial.suggest_int("max_depth", 2, 8),
        "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }
    model  = GradientBoostingRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
    return scores.mean()


# ──────────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────────

@task(task_id="tune_random_forest")
def tune_random_forest():
    """Run Optuna study for Random Forest and log results to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("housing_hyperparameter_tuning")

    X_train, X_test, y_train, y_test = _load_data()

    study = optuna.create_study(
        study_name="rf_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _rf_objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_params
    log.info("🌲 RF Best params: %s  (CV R²=%.4f)", best, study.best_value)

    # Retrain on full training data with best params
    final_model = RandomForestRegressor(**best, n_jobs=-1, random_state=42)
    final_model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, final_model.predict(X_test))

    with mlflow.start_run(run_name="RF_Optuna_Best") as run:
        mlflow.log_params(best)
        mlflow.log_metrics({
            "cv_r2":   round(study.best_value, 4),
            "test_r2": round(test_r2, 4),
            "n_trials": N_TRIALS,
        })
        mlflow.sklearn.log_model(final_model, "model")
        run_id = run.info.run_id

    joblib.dump(final_model, MODELS_DIR / "rf_tuned.pkl")
    return {"model": "rf_tuned", "test_r2": round(test_r2, 4), "run_id": run_id, "best_params": best}


@task(task_id="tune_gradient_boosting")
def tune_gradient_boosting():
    """ Run Optuna study for Gradient Boosting and log results to MLflow.
        Pruning helps stop unpromising trials early, saving time and resources.
        MedianPruner prunes a trial if its median across all-time best trials
        is worse than the current best trial.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("housing_hyperparameter_tuning")

    X_train, X_test, y_train, y_test = _load_data()

    study = optuna.create_study(
        study_name="gb_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        lambda trial: _gb_objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_params
    log.info("⚡ GB Best params: %s  (CV R²=%.4f)", best, study.best_value)

    final_model = GradientBoostingRegressor(**best, random_state=42)
    final_model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, final_model.predict(X_test))

    with mlflow.start_run(run_name="GB_Optuna_Best") as run:
        mlflow.log_params(best)
        mlflow.log_metrics({
            "cv_r2":   round(study.best_value, 4),
            "test_r2": round(test_r2, 4),
            "n_trials": N_TRIALS,
        })
        mlflow.sklearn.log_model(final_model, "model")
        run_id = run.info.run_id

    joblib.dump(final_model, MODELS_DIR / "gb_tuned.pkl")
    return {"model": "gb_tuned", "test_r2": round(test_r2, 4), "run_id": run_id, "best_params": best}


@task(task_id="finalize_best_tuned_model")
def finalize_best_tuned_model(rf_result: dict, gb_result: dict):
    """Select the best tuned model and save summary."""
    winner = rf_result if rf_result["test_r2"] >= gb_result["test_r2"] else gb_result
    loser  = gb_result if winner == rf_result else rf_result

    log.info("🏆 Winner after tuning: %s (R²=%.4f)", winner["model"], winner["test_r2"])
    log.info("   Runner-up:           %s (R²=%.4f)", loser["model"],  loser["test_r2"])
    log.info("   Best params: %s", winner["best_params"])

    import shutil
    src = MODELS_DIR / f"{winner['model']}.pkl"
    shutil.copy(src, MODELS_DIR / "best_tuned_model.pkl")

    summary = {
        "winner":   winner,
        "runner_up": loser,
        "finalized_at": datetime.utcnow().isoformat(),
    }
    (MODELS_DIR / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ──────────────────────────────────────────────
# DAG DEFINITION
# ──────────────────────────────────────────────
with DAG(
    dag_id="tutorial_05_hyperparameter_tuning",
    description="Intermediate: Optuna hyperparameter tuning with MLflow",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "intermediate", "level-05", "optuna", "tuning"],
) as dag:

    rf_result = tune_random_forest()
    gb_result = tune_gradient_boosting()
    finalize_best_tuned_model(rf_result, gb_result)
