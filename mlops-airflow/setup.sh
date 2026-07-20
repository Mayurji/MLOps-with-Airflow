#!/usr/bin/env bash
# =============================================================
# setup.sh — Bootstrap the MLOps-Airflow tutorial environment
# =============================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
# =============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/mlops-airflow"
AIRFLOW_HOME="$PROJECT_DIR/airflow_home"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      MLOps with Airflow — Environment Setup          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "📁 Project:  $PROJECT_DIR"
echo "🐍 Venv:     $VENV_DIR"
echo "✈️  Airflow:  $AIRFLOW_HOME"
echo ""

# ── Step 1: Virtual environment ──────────────────────────────
echo "──────────────────────────────────────────────"
echo "  STEP 1: Creating virtual environment"
echo "──────────────────────────────────────────────"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  ✅ Virtual env created: $VENV_DIR"
else
    echo "  ℹ️  Virtual env already exists"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── Step 2: Core dependencies ────────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo "  STEP 2: Installing Python dependencies"
echo "──────────────────────────────────────────────"
# Airflow requires constraint files for reproducible installs
AIRFLOW_VERSION=2.9.3
PYTHON_VERSION="$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

echo "  Installing Apache Airflow ${AIRFLOW_VERSION}…"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" -q

echo "  Installing data science stack…"
pip install numpy pandas scikit-learn matplotlib seaborn joblib \
            mlflow optuna fastapi uvicorn pydantic python-dotenv \
            pyyaml requests tqdm psutil scipy -q

echo "  ✅ Core dependencies installed"

# ── Step 3: PyTorch (GPU) ────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo "  STEP 3: PyTorch GPU setup"
echo "──────────────────────────────────────────────"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  🟢 GPU detected: $GPU_NAME"
    echo "  Installing PyTorch with CUDA 11.8 support…"
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 -q
    echo "  ✅ PyTorch GPU installed"
else
    echo "  🟡 No GPU detected — installing PyTorch CPU version"
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
    echo "  ✅ PyTorch CPU installed"
fi

# ── Step 4: Airflow configuration ────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo "  STEP 4: Configuring Airflow"
echo "──────────────────────────────────────────────"
mkdir -p "$AIRFLOW_HOME"
export AIRFLOW_HOME="$AIRFLOW_HOME"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_DIR/dags"
export AIRFLOW__CORE__EXECUTOR="SequentialExecutor"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///$AIRFLOW_HOME/airflow.db"
export AIRFLOW__WEBSERVER__SECRET_KEY="mlops-airflow-secret-key-2024"

airflow db migrate 2>&1 | tail -5
echo "  ✅ Airflow database initialized"

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname MLOps \
    --lastname Learner \
    --role Admin \
    --email admin@mlops.local 2>/dev/null || echo "  ℹ️  Admin user already exists"
echo "  ✅ Admin user: admin / admin"

# ── Step 5: Create project directories ──────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo "  STEP 5: Creating project structure"
echo "──────────────────────────────────────────────"
mkdir -p \
    "$PROJECT_DIR/data/raw" \
    "$PROJECT_DIR/data/processed" \
    "$PROJECT_DIR/models/deployed" \
    "$PROJECT_DIR/models/pytorch" \
    "$PROJECT_DIR/logs/metrics" \
    "$PROJECT_DIR/logs/drift_reports" \
    "$PROJECT_DIR/scripts"
echo "  ✅ Directory structure ready"

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║               🎉 Setup Complete!                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Next steps:"
echo ""
echo "  1️⃣  Activate environment:"
echo "       source mlops-airflow/bin/activate"
echo ""
echo "  2️⃣  Export Airflow settings:"
echo "       export AIRFLOW_HOME=$AIRFLOW_HOME"
echo "       export AIRFLOW__CORE__DAGS_FOLDER=$PROJECT_DIR/dags"
echo "       export AIRFLOW__CORE__LOAD_EXAMPLES=False"
echo ""
echo "  3️⃣  Start Airflow webserver (in terminal 1):"
echo "       airflow webserver --port 8080"
echo ""
echo "  4️⃣  Start Airflow scheduler (in terminal 2):"
echo "       airflow scheduler"
echo ""
echo "  5️⃣  (Optional) Start MLflow tracking server:"
echo "       mlflow server --host 0.0.0.0 --port 5000"
echo ""
echo "  6️⃣  Open Airflow UI: http://localhost:8080"
echo "       Login: admin / admin"
echo ""
echo "  📚 Tutorial order:"
echo "       tutorial_01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09"
echo ""
