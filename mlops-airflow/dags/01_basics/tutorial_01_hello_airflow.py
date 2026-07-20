#!/usr/bin/env python3
"""
Tutorial 01 - Hello Airflow
============================
Level    : BASIC
Topic    : Your first DAG — understanding tasks, operators, and task dependencies.

Concepts Covered:
  - What is a DAG (Directed Acyclic Graph)?
  - PythonOperator: running Python functions as tasks
  - BashOperator: running shell commands
  - Task dependencies with >> and <<
  - XCom: lightweight data passing between tasks

Run:
  airflow dags trigger tutorial_01_hello_airflow
  airflow dags test tutorial_01_hello_airflow 2024-01-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# DEFAULT ARGUMENTS (applied to every task)
default_args = {
    "owner": "mlops-learner",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# TASK FUNCTIONS

def greet_world(**context):
    """Task 1: A simple greeting — every journey starts here."""
    execution_date = context["execution_date"]
    print(f"🚀 Hello from Airflow! Execution date: {execution_date}")
    print("   This is your very first DAG task.")
    # Push a value to XCom so downstream tasks can consume it
    return "Hello from Task 1"


def describe_dag(**context):
    """Task 2: Pull XCom value from the upstream task and explain DAGs."""
    ti = context["ti"]
    upstream_message = ti.xcom_pull(task_ids="greet_world")
    print(f"📨 Received XCom from greet_world: '{upstream_message}'")
    print("\n📚 What is a DAG?")
    print("   A Directed Acyclic Graph (DAG) is Airflow's core abstraction.")
    print("   • DIRECTED  — tasks flow in one direction (no cycles)")
    print("   • ACYCLIC   — no task can be its own ancestor")
    print("   • GRAPH     — tasks are nodes, dependencies are edges")
    return "DAG explained!"

def log_system_info(**context):
    """Task 3: Log basic environment info — useful for debugging pipelines."""
    import platform, os
    print("🖥️  System Information")
    print(f"   OS      : {platform.system()} {platform.release()}")
    print(f"   Python  : {platform.python_version()}")
    print(f"   CWD     : {os.getcwd()}")
    print(f"   DAG Run : {context['run_id']}")

# DAG DEFINITION
with DAG(
    dag_id="tutorial_01_hello_airflow",
    description="Beginner tutorial: first DAG with PythonOperator and BashOperator",
    default_args=default_args,
    schedule_interval="@daily",         # Run once a day
    start_date=datetime(2024, 1, 1),
    catchup=False,                      # Don't backfill past runs
    tags=["tutorial", "basic", "level-01"],
) as dag:

    # ── Task 1: Greet ────────────────────────────────────────
    task_greet = PythonOperator(
        task_id="greet_world",
        python_callable=greet_world,
    )

    # ── Task 2: Describe DAG (uses XCom from task 1) ─────────
    task_describe = PythonOperator(
        task_id="describe_dag",
        python_callable=describe_dag,
    )

    # ── Task 3: Bash task ────────────────────────────────────
    task_bash = BashOperator(
        task_id="show_date_bash",
        bash_command='echo "📅 Bash task running at: $(date)" && echo "Airflow rocks!"',
    )

    # ── Task 4: Log system info ──────────────────────────────
    task_sysinfo = PythonOperator(
        task_id="log_system_info",
        python_callable=log_system_info,
    )

    # ── DEPENDENCY CHAIN ─────────────────────────────────────
    # task_greet → task_describe → task_bash → task_sysinfo
    task_greet >> task_describe >> task_bash >> task_sysinfo
