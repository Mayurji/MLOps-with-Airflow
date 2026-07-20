# MLOps with Apache Airflow
### A Complete MLOps Tutorial Series — Basic to Advanced
> Optimized for: **NVIDIA RTX 2070 SUPER (8GB VRAM)** | Python 3.10 | Ubuntu Linux

---

## 📚 Tutorial Overview

| # | Tutorial | Level | Key Concepts |
|---|----------|-------|--------------|
| 01 | Hello Airflow | 🟢 Basic | DAG, PythonOperator, BashOperator, XCom |
| 02 | Data Ingestion | 🟢 Basic | TaskFlow API, ShortCircuitOperator, Parquet |
| 03 | Model Training | 🟡 Intermediate | Parallel tasks, fan-out/fan-in, joblib |
| 04 | MLflow Tracking | 🟡 Intermediate | Experiment tracking, model registry |
| 05 | Hyperparameter Tuning | 🟡 Intermediate | Optuna TPE, pruning, MLflow integration |
| 06 | Model Monitoring | 🟠 Advanced | Drift detection, KS test, PSI, auto-retrain |
| 07 | E2E Pipeline | 🟠 Advanced | Quality gates, model promotion, FastAPI serving |
| 08 | GPU Training | 🔴 Advanced (GPU) | PyTorch, CUDA AMP, mixed precision, VRAM mgmt |
| 09 | Master Pipeline | 🔴 Advanced | DAG of DAGs, full orchestration |

---

## 🚀 Quick Start

### Option A: Automated Setup (Recommended)
```bash
cd /home/mayur/Desktop/mlops-airflow
chmod +x setup.sh && ./setup.sh
```

### Option B: Manual Setup

**1. Activate virtual environment**
```bash
source mlops-airflow/bin/activate
```

**2. Install dependencies**
```bash
# Airflow (with constraints for reproducibility)
pip install "apache-airflow==2.9.3" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.10.txt"

# Data science stack
pip install numpy pandas scikit-learn matplotlib seaborn joblib \
            mlflow optuna fastapi uvicorn scipy pydantic python-dotenv

# PyTorch GPU (RTX 2070 SUPER — CUDA 11.8)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

**3. Initialize Airflow**
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__EXECUTOR="SequentialExecutor"
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:///$(pwd)/airflow_home/airflow.db

airflow db migrate
airflow users create --username admin --password admin \
    --firstname MLOps --lastname Learner --role Admin --email admin@mlops.local
```

**4. Start services (3 separate terminals)**

*Terminal 1 — Airflow Webserver:*
```bash
source mlops-airflow/bin/activate && export AIRFLOW_HOME=$(pwd)/airflow_home
airflow webserver --port 8080
```

*Terminal 2 — Airflow Scheduler:*
```bash
source mlops-airflow/bin/activate && export AIRFLOW_HOME=$(pwd)/airflow_home
airflow scheduler
```

*Terminal 3 — MLflow Tracking Server (for tutorials 04, 05, 08):*
```bash
source mlops-airflow/bin/activate
mlflow server --host 0.0.0.0 --port 5000
```

**5. Open Airflow UI**
- URL: http://localhost:8080
- Login: `admin` / `admin`

---

## 📋 Tutorial Execution Order

Run tutorials **sequentially** — each builds on the previous:

```bash
# Test a DAG without triggering (dry run)
airflow dags test <dag_id> 2024-01-01

# Trigger a DAG run
airflow dags trigger <dag_id>

# View logs
airflow tasks logs <dag_id> <task_id> <execution_date>
```

### Recommended order:
```bash
# 1. Hello Airflow (10 seconds)
airflow dags test tutorial_01_hello_airflow 2024-01-01

# 2. Data Ingestion — generates housing dataset (20 seconds)
airflow dags test tutorial_02_data_ingestion 2024-01-01

# 3. Model Training — trains 3 models in parallel (1-2 min)
airflow dags test tutorial_03_model_training 2024-01-01

# 4. MLflow Tracking — requires: mlflow server running on :5000
airflow dags test tutorial_04_mlflow_tracking 2024-01-01

# 5. Hyperparameter Tuning — 30 Optuna trials per model (3-5 min)
airflow dags test tutorial_05_hyperparameter_tuning 2024-01-01

# 6. Model Monitoring — drift detection (30 seconds)
airflow dags test tutorial_06_model_monitoring 2024-01-01

# 7. E2E Pipeline — quality gates + serving script
airflow dags test tutorial_07_end_to_end_pipeline 2024-01-01

# 8. GPU Training — PyTorch neural net on RTX 2070 SUPER (2-5 min)
airflow dags test tutorial_08_gpu_training 2024-01-01

# 9. Master Pipeline — orchestrates all of the above
airflow dags trigger tutorial_09_master_pipeline
```

---

## 📁 Project Structure

```
mlops-airflow/
├── mlops-airflow/          ← Python virtual environment
├── airflow_home/           ← Airflow database & config (auto-created)
├── dags/
│   ├── 01_basics/
│   │   ├── tutorial_01_hello_airflow.py
│   │   └── tutorial_02_data_ingestion.py
│   ├── 02_intermediate/
│   │   ├── tutorial_03_model_training.py
│   │   ├── tutorial_04_mlflow_tracking.py
│   │   └── tutorial_05_hyperparameter_tuning.py
│   ├── 03_advanced/
│   │   ├── tutorial_06_model_monitoring.py
│   │   └── tutorial_07_end_to_end_pipeline.py
│   └── 04_ml_pipeline/
│       ├── tutorial_08_gpu_training.py
│       └── tutorial_09_master_pipeline.py
├── data/
│   ├── raw/                ← Raw ingested data (CSV)
│   └── processed/          ← Cleaned Parquet datasets
├── models/
│   ├── *.pkl               ← Scikit-learn models
│   ├── pytorch/            ← PyTorch checkpoints
│   └── deployed/           ← Production-promoted models
├── logs/
│   ├── metrics/            ← Evaluation metrics JSON
│   └── drift_reports/      ← Drift detection reports
├── scripts/
│   └── serve_model.py      ← Auto-generated FastAPI server
├── .env                    ← Environment variables
├── requirements.txt        ← All Python dependencies
└── setup.sh                ← One-click setup script
```

---

## 🎮 GPU Tips for RTX 2070 SUPER (8GB VRAM)

| Technique | Why It Matters |
|-----------|---------------|
| Mixed Precision (AMP) | Halves VRAM usage — fit larger models |
| Batch size ≤ 512 | Prevents OOM errors on 8GB |
| `torch.cuda.empty_cache()` | Free VRAM between tasks |
| `pin_memory=False` | Avoid extra host memory allocation |
| Gradient checkpointing | Trade compute for memory |

---

## 🔧 Troubleshooting

**Airflow scheduler not picking up DAGs?**
```bash
# Verify DAGS_FOLDER is set correctly
airflow config get-value core dags_folder
# Force re-scan
airflow dags reserialize
```

**MLflow connection refused?**
```bash
# Start MLflow server first
mlflow server --host 0.0.0.0 --port 5000 &
```

**CUDA out of memory?**
```bash
# Reduce batch_size in tutorial_08 (default: 256 → try 64)
# Or use CPU: export CUDA_VISIBLE_DEVICES=""
```

**DAG import error?**
```bash
airflow dags list-import-errors
```

---

## 📊 Key Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI  | http://localhost:5000 | None |
| Model API  | http://localhost:8080 (after tutorial 07) | None |

---

## 🧠 Concepts Progression

```
Basic                  Intermediate              Advanced
──────                 ────────────              ────────
DAG structure    →     TaskFlow API        →     DAG of DAGs
PythonOperator   →     Parallel tasks      →     Quality gates
BashOperator     →     MLflow tracking     →     Drift detection
XCom             →     Optuna tuning       →     GPU training (PyTorch)
Scheduling       →     Model selection     →     FastAPI serving
```

---

*Built with ❤️ for MLOps learners | Apache Airflow 2.9.3 | PyTorch 2.3.1 | MLflow 2.14*
