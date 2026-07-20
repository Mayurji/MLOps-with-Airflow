#!/usr/bin/env python3
"""
Tutorial 08 - GPU-Accelerated Deep Learning with PyTorch + Airflow
===================================================================
Level    : ADVANCED (GPU)
Topic    : Training a neural network on GPU (RTX 2070 SUPER 8GB) via Airflow.

Concepts Covered:
  - GPU detection and memory management in Airflow tasks
  - PyTorch training loop with CUDA
  - Mixed precision training (torch.cuda.amp) for 8GB VRAM
  - Saving/loading PyTorch checkpoints
  - Logging GPU metrics to MLflow

Prerequisites:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Run:
  airflow dags test tutorial_08_gpu_training 2024-01-01
"""

import os, json, logging, time
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/home/mayur/Desktop/mlops-airflow"))
MODELS_DIR   = PROJECT_ROOT / "models" / "pytorch"
DATA_DIR     = PROJECT_ROOT / "data"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

default_args = {"owner": "mlops-learner", "retries": 1, "retry_delay": timedelta(minutes=5)}


# ──────────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────────

@task(task_id="check_gpu_availability")
def check_gpu_availability():
    """Detect GPU, log specs, set memory limits for 8GB VRAM."""
    try:
        import torch
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count":   torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            gpu_info.update({
                "device_name":    torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "cuda_version":   torch.version.cuda,
                "pytorch_version": torch.__version__,
            })
            log.info("🟢 GPU FOUND: %s (%.1f GB VRAM)", gpu_info["device_name"],
                     gpu_info["total_memory_gb"])
        else:
            log.warning("🟡 No GPU — will run on CPU (slower)")
    except ImportError:
        gpu_info = {"cuda_available": False, "error": "PyTorch not installed"}
        log.warning("PyTorch not installed. Install: pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu118")
    return gpu_info


@task(task_id="prepare_tabular_dataset")
def prepare_tabular_dataset():
    """Prepare housing data as PyTorch tensors."""
    import numpy as np
    import pandas as pd

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = DATA_DIR / "processed" / "housing_clean.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        FEATURE_COLS = ["area_sqft", "bedrooms", "bathrooms", "house_age",
                        "distance_km", "garage_spots", "price_per_sqft",
                        "bed_bath_ratio", "is_new_house"]
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df["price_usd"].values.astype(np.float32)
    else:
        log.info("No parquet found — generating synthetic data")
        rng = np.random.default_rng(42)
        n   = 2000
        X   = rng.normal(0, 1, (n, 9)).astype(np.float32)
        y   = (X.sum(axis=1) * 50000 + 200000 + rng.normal(0, 10000, n)).astype(np.float32)

    # Normalize
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = y.mean(),   y.std()  + 1e-8
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    # Train/val split
    n       = len(X_norm)
    n_train = int(0.8 * n)
    idx     = np.random.permutation(n)

    data = {
        "X_train": X_norm[idx[:n_train]],
        "X_val":   X_norm[idx[n_train:]],
        "y_train": y_norm[idx[:n_train]],
        "y_val":   y_norm[idx[n_train:]],
        "y_mean":  float(y_mean),
        "y_std":   float(y_std),
        "n_features": X.shape[1],
    }
    import pickle
    (MODELS_DIR / "dataset.pkl").write_bytes(pickle.dumps(data))
    log.info("📊 Dataset prepared: %d train | %d val | %d features",
             n_train, n - n_train, X.shape[1])
    return str(MODELS_DIR / "dataset.pkl")


@task(task_id="train_neural_network")
def train_neural_network(gpu_info: dict, dataset_path: str):
    """
    Train a feed-forward neural network for housing price regression.
    Uses mixed precision (AMP) to fit within 8GB VRAM.
    """
    import pickle
    import numpy as np

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.warning("PyTorch not installed — simulating training results")
        return _simulate_training()

    # ── Load data ───────────────────────────────────────────
    data = pickle.loads(Path(dataset_path).read_bytes())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("🖥️  Training on: %s", device)

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    X_train = to_tensor(data["X_train"]).to(device)
    y_train = to_tensor(data["y_train"]).unsqueeze(1).to(device)
    X_val   = to_tensor(data["X_val"]).to(device)
    y_val   = to_tensor(data["y_val"]).unsqueeze(1).to(device)

    train_ds     = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=False)

    # ── Model Architecture ──────────────────────────────────
    class HousingNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),    nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x)

    model = HousingNet(data["n_features"]).to(device)
    log.info("🧠 Model params: %d", sum(p.numel() for p in model.parameters()))

    # ── Training ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history    = {"train_loss": [], "val_loss": [], "epoch": []}
    best_val   = float("inf")
    EPOCHS     = 50
    patience   = 10
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(Xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        avg_train = train_loss / len(train_loader)
        history["train_loss"].append(round(avg_train, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["epoch"].append(epoch)

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODELS_DIR / "best_nn.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0:
            log.info("Epoch %3d | train=%.4f | val=%.4f | best=%.4f",
                     epoch, avg_train, val_loss, best_val)

        if no_improve >= patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    # ── Save artifacts ───────────────────────────────────────
    (MODELS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    torch.save(model, MODELS_DIR / "full_nn_model.pt")

    # ── VRAM report ──────────────────────────────────────────
    if device.type == "cuda":
        used_gb = torch.cuda.max_memory_allocated(0) / 1e9
        log.info("🎮 Peak VRAM used: %.2f GB / %.1f GB", used_gb, gpu_info.get("total_memory_gb", 8))
        torch.cuda.reset_peak_memory_stats()

    return {"best_val_loss": round(best_val, 6), "epochs_trained": epoch, "device": str(device)}


def _simulate_training():
    """Fallback when PyTorch is not installed."""
    import random
    log.info("⚠️  Simulating training (PyTorch not installed)")
    history = {
        "epoch":      list(range(1, 51)),
        "train_loss": [round(1.0 / (i + 1) + random.uniform(0, 0.05), 4) for i in range(50)],
        "val_loss":   [round(1.0 / (i + 1) + random.uniform(0, 0.08), 4) for i in range(50)],
    }
    (MODELS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    return {"best_val_loss": 0.0234, "epochs_trained": 47, "device": "simulated"}


@task(task_id="log_to_mlflow")
def log_to_mlflow(gpu_info: dict, training_result: dict):
    """Log GPU training results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("gpu_neural_network")

        with mlflow.start_run(run_name="PyTorch_HousingNet"):
            mlflow.log_params({
                "model_type":    "FeedForwardNN",
                "device":        training_result.get("device", "unknown"),
                "gpu_name":      gpu_info.get("device_name", "CPU"),
                "vram_gb":       gpu_info.get("total_memory_gb", 0),
                "mixed_precision": True,
            })
            mlflow.log_metrics({
                "best_val_loss":   training_result.get("best_val_loss", 0),
                "epochs_trained":  training_result.get("epochs_trained", 0),
            })
            history_path = MODELS_DIR / "training_history.json"
            if history_path.exists():
                mlflow.log_artifact(str(history_path))
            log.info("📊 Results logged to MLflow")
    except Exception as e:
        log.warning("MLflow logging failed (is server running?): %s", e)
    return training_result


# ──────────────────────────────────────────────
# DAG DEFINITION
# ──────────────────────────────────────────────
with DAG(
    dag_id="tutorial_08_gpu_training",
    description="Advanced GPU: PyTorch neural net with AMP on RTX 2070 SUPER",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "advanced", "level-08", "gpu", "pytorch"],
) as dag:

    gpu_info  = check_gpu_availability()
    dataset   = prepare_tabular_dataset()
    training  = train_neural_network(gpu_info, dataset)
    log_to_mlflow(gpu_info, training)
