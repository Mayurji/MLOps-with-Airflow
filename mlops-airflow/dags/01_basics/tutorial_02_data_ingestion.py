#!/usr/bin/env python3
"""
Tutorial 02 - Data Ingestion Pipeline
=======================================
Level    : BASIC
Topic    : Downloading, validating, and storing datasets for ML.

Concepts Covered:
  - TaskFlow API (@task decorator) — modern Airflow style
  - ShortCircuitOperator: skip downstream tasks on condition
  - File sensors and data validation
  - Storing processed data as Parquet

Run:
  airflow dags test tutorial_02_data_ingestion 2024-01-01
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.bash import BashOperator

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/home/mayur/Desktop/MLOps-Airflow-Tutorials/mlops-airflow"))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

default_args = {
    "owner": "mlops-learner",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# HELPER: Generate synthetic dataset
def _make_housing_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic California-style housing dataset.
    Real pipelines would call an API or read from a database here.
    """
    rng = np.random.default_rng(seed)
    area        = rng.normal(1500, 400, n_samples).clip(400, 5000)
    bedrooms    = rng.integers(1, 7, n_samples)
    bathrooms   = (bedrooms * rng.uniform(0.5, 1.0, n_samples)).astype(int).clip(1, 5)
    age         = rng.integers(0, 50, n_samples)
    distance_km = rng.exponential(10, n_samples).clip(0.5, 80)
    noise       = rng.normal(0, 15_000, n_samples)
    price       = (
        area * 150
        + bedrooms * 8_000
        + bathrooms * 5_000
        - age * 1_200
        - distance_km * 3_000
        + noise
    ).clip(50_000, 2_000_000)

    return pd.DataFrame({
        "area_sqft":    area.round(1),
        "bedrooms":     bedrooms,
        "bathrooms":    bathrooms,
        "house_age":    age,
        "distance_km":  distance_km.round(2),
        "price_usd":    price.round(2),
        # Inject ~5% nulls for validation practice
        "garage_spots": np.where(rng.random(n_samples) < 0.05, np.nan,
                                 rng.integers(0, 4, n_samples)),
    })

# TASKFLOW TASKS

@task(task_id="create_directories")
def create_directories():
    """Ensure all required directories exist."""
    for d in [RAW_DIR, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        log.info("✅ Directory ready: %s", d)
    return str(RAW_DIR)


@task(task_id="ingest_raw_data")
def ingest_raw_data():
    """Simulate data ingestion: generate or download raw dataset."""
    log.info("📥 Ingesting housing dataset (2000 samples)…")
    df = _make_housing_dataset(n_samples=2000)
    raw_path = RAW_DIR / "housing_raw.csv"
    df.to_csv(raw_path, index=False)
    log.info("💾 Raw data saved → %s  (%d rows)", raw_path, len(df))

    # Write metadata sidecar
    meta = {
        "rows": len(df),
        "columns": list(df.columns),
        "ingested_at": datetime.utcnow().isoformat(),
        "source": "synthetic_generator_v1",
    }
    (RAW_DIR / "housing_raw_meta.json").write_text(json.dumps(meta, indent=2))
    return str(raw_path)


@task(task_id="validate_raw_data")
def validate_raw_data(raw_path: str):
    """
    Basic data validation before processing.
    Returns True if data passes quality checks.
    """
    df = pd.read_csv(raw_path)
    issues = []

    # Check 1: Row count
    if len(df) < 100:
        issues.append(f"Too few rows: {len(df)}")

    # Check 2: Required columns
    required_cols = ["area_sqft", "bedrooms", "bathrooms", "price_usd"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check 3: Null rate in critical columns
    for col in required_cols:
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 10:
            issues.append(f"{col} has {null_pct:.1f}% nulls (threshold: 10%)")

    # Check 4: Price sanity
    if df["price_usd"].min() <= 0:
        issues.append("Negative or zero prices detected")

    report = {
        "total_rows": len(df),
        "null_rates": df.isnull().mean().round(4).to_dict(),
        "issues": issues,
        "passed": len(issues) == 0,
    }
    report_path = DATA_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    if issues:
        log.warning("⚠️  Validation issues: %s", issues)
    else:
        log.info("✅ All validation checks passed!")

    return report["passed"]


def _check_validation_passed(**context):
    """ShortCircuit: skip downstream if validation failed."""
    ti = context["ti"]
    passed = ti.xcom_pull(task_ids="validate_raw_data")
    log.info("Validation result: %s", passed)
    return bool(passed)


@task(task_id="process_and_clean")
def process_and_clean(raw_path: str):
    """Clean raw data: handle nulls, fix types, engineer basic features."""
    df = pd.read_csv(raw_path)
    log.info("🔧 Processing %d rows…", len(df))

    # Fill missing garage spots with median
    df["garage_spots"] = df["garage_spots"].fillna(df["garage_spots"].median())

    # Feature engineering
    df["price_per_sqft"] = (df["price_usd"] / df["area_sqft"]).round(2)
    df["bed_bath_ratio"]  = (df["bedrooms"] / df["bathrooms"].clip(lower=1)).round(2)
    df["is_new_house"]    = (df["house_age"] < 5).astype(int)
    df["price_tier"]      = pd.qcut(
        df["price_usd"], q=4, labels=["budget", "mid", "premium", "luxury"]
    )

    out_path = PROCESSED_DIR / "housing_clean.parquet"
    df.to_parquet(out_path, index=False)
    log.info("✅ Processed data saved → %s", out_path)
    return str(out_path)


@task(task_id="generate_eda_stats")
def generate_eda_stats(processed_path: str):
    """Compute and save basic EDA statistics."""
    df = pd.read_parquet(processed_path)

    stats = {
        "shape": list(df.shape),
        "numeric_summary": json.loads(
            df.describe(include="number").round(2).to_json()
        ),
        "price_tier_counts": df["price_tier"].value_counts().to_dict(),
        "generated_at": datetime.utcnow().isoformat(),
    }
    stats_path = DATA_DIR / "eda_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    log.info("📊 EDA stats saved → %s", stats_path)
    log.info("Shape: %s | Price range: $%.0f – $%.0f",
             df.shape, df["price_usd"].min(), df["price_usd"].max())


# ──────────────────────────────────────────────
# DAG DEFINITION
# ──────────────────────────────────────────────
with DAG(
    dag_id="tutorial_02_data_ingestion",
    description="Beginner: data ingestion, validation, and preprocessing",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "basic", "level-02", "data"],
) as dag:

    dirs     = create_directories()
    raw_path = ingest_raw_data()
    valid    = validate_raw_data(raw_path)

    gate = ShortCircuitOperator(
        task_id="validation_gate",
        python_callable=_check_validation_passed,
    )

    proc  = process_and_clean(raw_path)
    stats = generate_eda_stats(proc)

    # Dependency chain
    dirs >> raw_path >> valid >> gate >> proc >> stats
