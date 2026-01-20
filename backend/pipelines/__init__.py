"""
Pipeline Orchestration Module

Provides workflow orchestration using Prefect for:
- ML training pipelines
- Data processing pipelines
- Model deployment pipelines
- Scheduled jobs
"""

from .flows import (
    training_pipeline,
    data_pipeline,
    deployment_pipeline,
    monitoring_pipeline,
)
from .tasks import (
    load_data_task,
    preprocess_task,
    train_model_task,
    evaluate_model_task,
    register_model_task,
    deploy_model_task,
)
from .scheduler import PipelineScheduler

__all__ = [
    "training_pipeline",
    "data_pipeline",
    "deployment_pipeline",
    "monitoring_pipeline",
    "load_data_task",
    "preprocess_task",
    "train_model_task",
    "evaluate_model_task",
    "register_model_task",
    "deploy_model_task",
    "PipelineScheduler",
]
