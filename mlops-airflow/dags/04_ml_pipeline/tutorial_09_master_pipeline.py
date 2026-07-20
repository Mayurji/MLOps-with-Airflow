#!/usr/bin/env python3
"""
Tutorial 09 - Master Pipeline (Orchestrating All Tutorials)
=============================================================
Level    : ADVANCED
Topic    : DAG of DAGs — orchestrate the full MLOps workflow.

This is the capstone DAG that ties together all previous tutorials
into a single orchestrated workflow with proper sequencing.

Run:
  airflow dags trigger tutorial_09_master_pipeline
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.empty import EmptyOperator

default_args = {"owner": "mlops-learner", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="tutorial_09_master_pipeline",
    description="Capstone: orchestrate all tutorials into one MLOps workflow",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial", "advanced", "capstone", "orchestration"],
) as dag:

    start = EmptyOperator(task_id="pipeline_start")

    # Step 1: Ingest data
    ingest = TriggerDagRunOperator(
        task_id="trigger_data_ingestion",
        trigger_dag_id="tutorial_02_data_ingestion",
        wait_for_completion=True,
        poke_interval=10,
        allowed_states=["success"],
    )

    # Step 2: Train models (after ingestion)
    train = TriggerDagRunOperator(
        task_id="trigger_model_training",
        trigger_dag_id="tutorial_03_model_training",
        wait_for_completion=True,
        poke_interval=10,
        allowed_states=["success"],
    )

    # Step 3: Track experiments in MLflow
    track = TriggerDagRunOperator(
        task_id="trigger_mlflow_tracking",
        trigger_dag_id="tutorial_04_mlflow_tracking",
        wait_for_completion=True,
        poke_interval=10,
        allowed_states=["success", "failed"],  # Non-blocking if MLflow down
    )

    # Step 4: Hyperparameter tuning
    tune = TriggerDagRunOperator(
        task_id="trigger_hyperparameter_tuning",
        trigger_dag_id="tutorial_05_hyperparameter_tuning",
        wait_for_completion=True,
        poke_interval=15,
        allowed_states=["success"],
    )

    # Step 5: GPU Training (if available)
    gpu_train = TriggerDagRunOperator(
        task_id="trigger_gpu_training",
        trigger_dag_id="tutorial_08_gpu_training",
        wait_for_completion=True,
        poke_interval=30,
        allowed_states=["success", "failed"],  # Non-blocking if no GPU
    )

    # Step 6: Promote best model to production
    promote = TriggerDagRunOperator(
        task_id="trigger_end_to_end_pipeline",
        trigger_dag_id="tutorial_07_end_to_end_pipeline",
        wait_for_completion=True,
        poke_interval=10,
        allowed_states=["success"],
    )

    # Step 7: Monitor deployed model
    monitor = TriggerDagRunOperator(
        task_id="trigger_model_monitoring",
        trigger_dag_id="tutorial_06_model_monitoring",
        wait_for_completion=True,
        poke_interval=10,
        allowed_states=["success"],
    )

    end = EmptyOperator(task_id="pipeline_complete")

    # Orchestration chain
    start >> ingest >> train >> track >> tune >> [gpu_train] >> promote >> monitor >> end
