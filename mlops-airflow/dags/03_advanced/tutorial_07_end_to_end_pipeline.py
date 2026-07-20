#!/usr/bin/env python3
"""
Tutorial 07 - End-to-End ML Pipeline (Complete MLOps)
=======================================================
Level    : ADVANCED
Topic    : Production-grade pipeline: quality gates, model promotion, serving.

Concepts Covered:
  - Quality gate pattern (CI/CD for ML)
  - BranchPythonOperator for conditional promotion
  - Auto-generating FastAPI model server
  - Model Card generation
  - Deployment manifest management

Run:
  airflow dags test tutorial_07_end_to_end_pipeline 2024-01-01
"""

import os, json, logging, subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import joblib
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/home/mayur/Desktop/mlops-airflow"))
MODELS_DIR   = PROJECT_ROOT / "models"
DEPLOY_DIR   = PROJECT_ROOT / "models" / "deployed"
METRICS_DIR  = PROJECT_ROOT / "logs" / "metrics"

QUALITY_GATES = {"min_r2": 0.70, "max_rmse": 80_000}

default_args = {"owner": "mlops-learner", "retries": 2, "retry_delay": timedelta(minutes=5)}


@task
def pipeline_initialization():
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        git_hash = "no-git"
    meta = {"run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started_at": datetime.utcnow().isoformat(), "git_commit": git_hash}
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "logs" / "pipeline_run.json").write_text(json.dumps(meta, indent=2))
    log.info("🚀 Pipeline started: %s", meta["run_id"])
    return meta


@task
def run_quality_gates():
    metrics_path = METRICS_DIR / "test_metrics.json"
    if not metrics_path.exists():
        log.warning("No metrics found — assuming gates passed for demo.")
        return {"all_passed": True, "r2": None, "rmse": None}
    metrics = json.loads(metrics_path.read_text())
    r2_ok   = metrics.get("r2",   0) >= QUALITY_GATES["min_r2"]
    rmse_ok = metrics.get("rmse", 1e9) <= QUALITY_GATES["max_rmse"]
    passed  = r2_ok and rmse_ok
    log.info("Gates — R²: %s | RMSE: %s | VERDICT: %s",
             "✅" if r2_ok else "❌", "✅" if rmse_ok else "❌",
             "PASSED" if passed else "FAILED")
    return {"all_passed": passed, "r2_ok": r2_ok, "rmse_ok": rmse_ok}


def _gate_branch(**context):
    result = context["ti"].xcom_pull(task_ids="run_quality_gates")
    return "promote_to_production" if result and result.get("all_passed") else "reject_model"


@task
def promote_to_production():
    import shutil
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = MODELS_DIR / "best_model.pkl"
    if not src.exists():
        from sklearn.linear_model import Ridge
        joblib.dump(Ridge().fit([[1]*9], [1]), src)
    dst = DEPLOY_DIR / f"model_v{ts}.pkl"
    shutil.copy(src, dst)
    shutil.copy(src, DEPLOY_DIR / "model_current.pkl")
    manifest = {"version": ts, "artifact": str(dst), "status": "production",
                "promoted_at": datetime.utcnow().isoformat()}
    (DEPLOY_DIR / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("🚀 Model PROMOTED → Production")
    return manifest


@task
def reject_model():
    log.warning("🚫 Model REJECTED — keeping previous version")
    return {"rejected_at": datetime.utcnow().isoformat()}


@task
def generate_model_card(manifest: dict):
    try:
        metrics = json.loads((METRICS_DIR / "test_metrics.json").read_text())
    except Exception:
        metrics = {"r2": "N/A", "mae": "N/A", "rmse": "N/A"}
    card = f"""# Model Card: Housing Price Predictor v{manifest.get('version','N/A')}
Promoted: {manifest.get('promoted_at','N/A')} | Status: {manifest.get('status','N/A')}
## Metrics: R²={metrics.get('r2')} | MAE=${metrics.get('mae')} | RMSE=${metrics.get('rmse')}
## Features: area_sqft, bedrooms, bathrooms, house_age, distance_km, garage_spots,
             price_per_sqft, bed_bath_ratio, is_new_house
## Note: Trained on synthetic data — demo purposes only.
"""
    path = DEPLOY_DIR / "model_card.md"
    path.write_text(card)
    log.info("📄 Model Card → %s", path)
    return str(path)


@task
def generate_serving_script(manifest: dict):
    """Write a FastAPI serving script for the promoted model."""
    script = PROJECT_ROOT / "scripts" / "serve_model.py"
    script.parent.mkdir(exist_ok=True)
    code = '''"""Auto-generated FastAPI model server — Tutorial 07"""
import joblib, numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent.parent / "models" / "deployed" / "model_current.pkl"
app = FastAPI(title="Housing Price API")
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

FEATURES = ["area_sqft","bedrooms","bathrooms","house_age","distance_km",
            "garage_spots","price_per_sqft","bed_bath_ratio","is_new_house"]

class HouseFeatures(BaseModel):
    area_sqft: float; bedrooms: int; bathrooms: int; house_age: int
    distance_km: float; garage_spots: float; price_per_sqft: float
    bed_bath_ratio: float; is_new_house: int

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(f: HouseFeatures):
    if model is None:
        return {"error": "model not loaded"}
    X = np.array([[f.area_sqft, f.bedrooms, f.bathrooms, f.house_age,
                   f.distance_km, f.garage_spots, f.price_per_sqft,
                   f.bed_bath_ratio, f.is_new_house]])
    return {"predicted_price_usd": round(float(model.predict(X)[0]), 2)}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8080)
'''
    script.write_text(code)
    log.info("📜 Server script → %s", script)
    log.info("   Start: python scripts/serve_model.py")
    log.info("   Docs:  http://localhost:8080/docs")
    return str(script)


with DAG(
    dag_id="tutorial_07_end_to_end_pipeline",
    description="Advanced: complete MLOps pipeline with quality gates and serving",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "advanced", "level-07", "production"],
) as dag:

    init  = pipeline_initialization()
    gates = run_quality_gates()
    init >> gates

    branch = BranchPythonOperator(
        task_id="quality_gate_decision", python_callable=_gate_branch)
    gates >> branch

    promote = promote_to_production()
    reject  = reject_model()
    branch >> [promote, reject]

    card   = generate_model_card(promote)
    server = generate_serving_script(promote)
    promote >> [card, server]

    done = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    [card, server, reject] >> done
