#!/usr/bin/env python3
"""
Tutorial 06 - Model Drift Detection & Monitoring
==================================================
Level    : INTERMEDIATE → ADVANCED
Topic    : Detecting data drift and model performance degradation using Evidently.

Concepts Covered:
  - Data drift detection with Evidently AI
  - Model performance monitoring over time
  - Alerting on significant drift (Airflow callback hooks)
  - Writing drift reports as HTML artifacts
  - Conditional retraining trigger via TriggerDagRunOperator

Run:
  airflow dags test tutorial_06_model_monitoring 2024-01-01
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from airflow import DAG
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
PROJECT_ROOT  = Path(os.getenv("PROJECT_ROOT", "/home/mayur/Desktop/mlops-airflow"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "logs" / "drift_reports"

FEATURE_COLS = ["area_sqft", "bedrooms", "bathrooms", "house_age",
                "distance_km", "garage_spots", "price_per_sqft",
                "bed_bath_ratio", "is_new_house"]

DRIFT_THRESHOLD = 0.15   # Trigger retraining if drift score > 15%

default_args = {
    "owner": "mlops-learner",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ──────────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────────

@task(task_id="generate_production_data")
def generate_production_data():
    """
    Simulate production data with gradual drift.
    In real pipelines: read from a database, Kafka, or feature store.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=int(datetime.now().timestamp()) % 10000)
    n   = 500

    # Simulate drift: production has shifted distribution
    df_prod = pd.DataFrame({
        "area_sqft":    rng.normal(1700, 500, n).clip(400, 5000),   # DRIFTED (was 1500)
        "bedrooms":     rng.integers(2, 7, n),                       # slightly shifted
        "bathrooms":    rng.integers(1, 5, n),
        "house_age":    rng.integers(0, 60, n),                       # DRIFTED (was 0-50)
        "distance_km":  rng.exponential(15, n).clip(0.5, 80),         # DRIFTED (was 10)
        "garage_spots": rng.integers(0, 5, n).astype(float),
        "price_per_sqft": rng.normal(160, 40, n).clip(50, 500),      # DRIFTED
        "bed_bath_ratio": rng.uniform(0.8, 2.5, n),
        "is_new_house":   rng.integers(0, 2, n),
    })

    prod_path = PROCESSED_DIR / "production_data.parquet"
    df_prod.to_parquet(prod_path, index=False)
    log.info("📦 Generated %d production records → %s", n, prod_path)
    return str(prod_path)


@task(task_id="compute_drift_report")
def compute_drift_report(prod_path: str) -> dict:
    """
    Compare reference (training) vs production data.
    Uses manual statistical drift detection (PSI + KS test).
    Install evidently for HTML reports: pip install evidently
    """
    from scipy import stats

    # Reference data (training set)
    df_ref  = pd.read_parquet(PROCESSED_DIR / "housing_clean.parquet")[FEATURE_COLS]
    df_prod = pd.read_parquet(prod_path)

    drift_results = {}
    drifted_cols  = []

    for col in FEATURE_COLS:
        ref_vals  = df_ref[col].dropna()
        prod_vals = df_prod[col].dropna()

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(ref_vals, prod_vals)

        # Population Stability Index (PSI)
        bins       = np.histogram_bin_edges(ref_vals, bins=10)
        ref_hist,  _ = np.histogram(ref_vals, bins=bins)
        prod_hist, _ = np.histogram(prod_vals, bins=bins)
        ref_freq  = ref_hist  / ref_hist.sum()  + 1e-8
        prod_freq = prod_hist / prod_hist.sum() + 1e-8
        psi = np.sum((prod_freq - ref_freq) * np.log(prod_freq / ref_freq))

        drifted = ks_p < 0.05 or psi > 0.1
        if drifted:
            drifted_cols.append(col)

        drift_results[col] = {
            "ks_statistic":   round(float(ks_stat), 4),
            "ks_p_value":     round(float(ks_p), 6),
            "psi":            round(float(psi), 4),
            "drifted":        drifted,
        }

    drift_score = len(drifted_cols) / len(FEATURE_COLS)
    report = {
        "drift_score":      round(drift_score, 4),
        "drifted_columns":  drifted_cols,
        "total_features":   len(FEATURE_COLS),
        "drifted_features": len(drifted_cols),
        "threshold":        DRIFT_THRESHOLD,
        "requires_retraining": drift_score > DRIFT_THRESHOLD,
        "feature_drift":    drift_results,
        "generated_at":     datetime.utcnow().isoformat(),
    }

    report_path = REPORTS_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    report_path.write_text(json.dumps(report, indent=2))

    log.info("📊 Drift Score: %.1f%% (%d/%d features drifted)",
             drift_score * 100, len(drifted_cols), len(FEATURE_COLS))
    log.info("   Drifted columns: %s", drifted_cols)
    log.warning("🚨 RETRAINING REQUIRED" if report["requires_retraining"]
                else "✅ Model stable — no retraining needed")

    return report


@task(task_id="check_model_performance_on_prod")
def check_model_performance_on_prod(prod_path: str):
    """Run current model on production data and compare to baseline."""
    try:
        model = joblib.load(MODELS_DIR / "best_model.pkl")
    except FileNotFoundError:
        log.warning("No model found. Skipping performance check.")
        return {"status": "no_model"}

    df_prod   = pd.read_parquet(prod_path)
    rng       = np.random.default_rng(42)
    n         = len(df_prod)
    area      = df_prod["area_sqft"]
    y_approx  = (area * 150 + rng.normal(0, 20000, n)).clip(50000, 2e6)

    preds = model.predict(df_prod[FEATURE_COLS])
    from sklearn.metrics import mean_absolute_percentage_error
    mape = mean_absolute_percentage_error(y_approx, preds) * 100

    result = {
        "mape_pct":         round(float(mape), 2),
        "performance_ok":   mape < 20,
        "evaluated_samples": n,
    }
    log.info("📈 Production MAPE: %.2f%%  (%s)",
             mape, "✅ OK" if result["performance_ok"] else "⚠️ Degraded")
    return result


def _branch_on_drift(**context):
    """Branch: retrain if drift detected, otherwise skip."""
    ti          = context["ti"]
    drift_report = ti.xcom_pull(task_ids="compute_drift_report")
    if drift_report and drift_report.get("requires_retraining"):
        log.info("🔀 Branching → trigger_retraining")
        return "trigger_retraining"
    log.info("🔀 Branching → no_retraining_needed")
    return "no_retraining_needed"


# ──────────────────────────────────────────────
# DAG DEFINITION
# ──────────────────────────────────────────────
with DAG(
    dag_id="tutorial_06_model_monitoring",
    description="Advanced: drift detection, monitoring, and conditional retraining",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "advanced", "level-06", "monitoring", "drift"],
) as dag:

    prod_path    = generate_production_data()
    drift_report = compute_drift_report(prod_path)
    perf_check   = check_model_performance_on_prod(prod_path)

    branch = BranchPythonOperator(
        task_id="drift_decision_branch",
        python_callable=_branch_on_drift,
    )

    retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="tutorial_03_model_training",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    no_action = EmptyOperator(task_id="no_retraining_needed")

    done = EmptyOperator(task_id="monitoring_complete", trigger_rule="none_failed_min_one_success")

    # Dependencies
    [drift_report, perf_check] >> branch >> [retrain, no_action] >> done
