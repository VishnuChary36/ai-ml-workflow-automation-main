"""
Prefect Tasks for ML Pipelines

Individual reusable tasks that can be composed into flows.
"""

import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
import joblib

try:
    from prefect import task, get_run_logger
    # Prefect 3.x moved artifacts
    try:
        from prefect.artifacts import create_table_artifact, create_markdown_artifact
    except ImportError:
        # Prefect 3.x compatibility
        create_table_artifact = None
        create_markdown_artifact = None
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    create_table_artifact = None
    create_markdown_artifact = None
    # Create mock decorators
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def get_run_logger():
        import logging
        return logging.getLogger("pipelines")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import settings


@task(
    name="load_data",
    description="Load dataset from file or database",
    retries=2,
    retry_delay_seconds=10,
    tags=["data", "io"]
)
def load_data_task(
    data_path: str,
    file_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from various sources.
    
    Args:
        data_path: Path to data file
        file_type: Override file type detection
        
    Returns:
        Loaded DataFrame
    """
    logger = get_run_logger()
    logger.info(f"Loading data from: {data_path}")
    
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Detect file type
    ext = file_type or path.suffix.lower()
    
    if ext in [".csv", "csv"]:
        df = pd.read_csv(path)
    elif ext in [".parquet", "parquet"]:
        df = pd.read_parquet(path)
    elif ext in [".json", "json"]:
        df = pd.read_json(path)
    elif ext in [".xlsx", ".xls", "excel"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


@task(
    name="validate_data",
    description="Validate data quality and schema",
    tags=["data", "validation"]
)
def validate_data_task(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    max_null_ratio: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate data quality.
    
    Args:
        df: Input DataFrame
        required_columns: Required column names
        max_null_ratio: Maximum allowed null ratio per column
        
    Returns:
        Tuple of (validated df, validation report)
    """
    logger = get_run_logger()
    logger.info("Validating data...")
    
    report = {
        "valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "issues": [],
        "warnings": [],
    }
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            report["valid"] = False
            report["issues"].append(f"Missing required columns: {missing}")
    
    # Check null ratios
    null_ratios = df.isnull().mean()
    high_null_cols = null_ratios[null_ratios > max_null_ratio].index.tolist()
    if high_null_cols:
        report["warnings"].append(
            f"High null ratio columns (>{max_null_ratio}): {high_null_cols}"
        )
    
    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        report["warnings"].append(f"Found {dup_count} duplicate rows")
    
    # Check for constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        report["warnings"].append(f"Constant columns: {constant_cols}")
    
    logger.info(f"Validation complete: {'PASSED' if report['valid'] else 'FAILED'}")
    
    return df, report


@task(
    name="preprocess",
    description="Preprocess data for training",
    tags=["data", "preprocessing"]
)
def preprocess_task(
    df: pd.DataFrame,
    target_column: str,
    preprocessing_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess data for model training.
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        preprocessing_config: Preprocessing configuration
        
    Returns:
        Tuple of (processed df, preprocessing artifacts)
    """
    logger = get_run_logger()
    logger.info(f"Preprocessing data with target: {target_column}")
    
    config = preprocessing_config or {}
    artifacts = {
        "encoders": {},
        "scalers": {},
        "target_encoder": None,
        "feature_columns": [],
        "target_column": target_column,
    }
    
    # Copy dataframe
    df = df.copy()
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        artifacts["encoders"][col] = le
    
    # Scale numeric features if requested
    if config.get("scale_features", True):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            artifacts["scalers"]["features"] = scaler
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    
    # Encode target if categorical
    if y.dtype == 'object' or y.nunique() < 20:
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=target_column)
        artifacts["target_encoder"] = le_target
        artifacts["problem_type"] = "classification"
    else:
        artifacts["problem_type"] = "regression"
    
    # Combine back
    df_processed = pd.concat([X, y], axis=1)
    artifacts["feature_columns"] = X.columns.tolist()
    
    logger.info(f"Preprocessing complete: {len(artifacts['feature_columns'])} features")
    
    return df_processed, artifacts


@task(
    name="split_data",
    description="Split data into train/test sets",
    tags=["data", "split"]
)
def split_data_task(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger = get_run_logger()
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    logger.info(f"Split data: train={len(X_train)}, test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test


@task(
    name="train_model",
    description="Train ML model",
    tags=["model", "training"]
)
def train_model_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train
        model_params: Model hyperparameters
        
    Returns:
        Trained model
    """
    logger = get_run_logger()
    logger.info(f"Training {model_type} model...")
    
    params = model_params or {}
    
    # Import models
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    
    try:
        from xgboost import XGBClassifier, XGBRegressor
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False
    
    # Detect problem type
    unique_values = y_train.nunique()
    is_classification = unique_values < 20
    
    # Select model
    models = {
        "random_forest": (
            RandomForestClassifier if is_classification else RandomForestRegressor
        ),
        "logistic_regression": LogisticRegression,
        "linear_regression": LinearRegression,
        "gradient_boosting": (
            GradientBoostingClassifier if is_classification else GradientBoostingRegressor
        ),
    }
    
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier if is_classification else XGBRegressor
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = models[model_type]
    
    # Handle XGBoost verbosity
    if model_type == "xgboost":
        params.setdefault("verbosity", 0)
        params.setdefault("use_label_encoder", False)
    
    model = model_class(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f}s")
    
    return model


@task(
    name="evaluate_model",
    description="Evaluate model performance",
    tags=["model", "evaluation"]
)
def evaluate_model_task(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str = "classification",
) -> Dict[str, float]:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        problem_type: classification or regression
        
    Returns:
        Dictionary of metrics
    """
    logger = get_run_logger()
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    if problem_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
    else:
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred),
        }
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Create artifact if Prefect available
    if PREFECT_AVAILABLE:
        try:
            create_table_artifact(
                key="evaluation-metrics",
                table=[{"metric": k, "value": round(v, 4)} for k, v in metrics.items()],
                description="Model evaluation metrics"
            )
        except Exception:
            pass
    
    return metrics


@task(
    name="save_model",
    description="Save model to disk",
    tags=["model", "io"]
)
def save_model_task(
    model: Any,
    model_id: str,
    preprocessing_artifacts: Dict[str, Any],
    metrics: Dict[str, float],
    model_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save trained model and artifacts.
    
    Args:
        model: Trained model
        model_id: Model identifier
        preprocessing_artifacts: Preprocessing info
        metrics: Evaluation metrics
        model_params: Model parameters
        
    Returns:
        Path to saved model directory
    """
    logger = get_run_logger()
    
    model_dir = Path(settings.model_storage_path) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    
    # Save encoders
    if preprocessing_artifacts.get("encoders"):
        encoders_path = model_dir / "encoders.joblib"
        joblib.dump(preprocessing_artifacts["encoders"], encoders_path)
    
    # Save target encoder
    if preprocessing_artifacts.get("target_encoder"):
        target_encoder_path = model_dir / "target_encoder.joblib"
        joblib.dump(preprocessing_artifacts["target_encoder"], target_encoder_path)
    
    # Save scalers
    if preprocessing_artifacts.get("scalers"):
        scalers_path = model_dir / "scalers.joblib"
        joblib.dump(preprocessing_artifacts["scalers"], scalers_path)
    
    # Save metadata
    metadata = {
        "model_id": model_id,
        "model_type": type(model).__name__,
        "model_params": model_params or {},
        "metrics": metrics,
        "feature_columns": preprocessing_artifacts.get("feature_columns", []),
        "target_column": preprocessing_artifacts.get("target_column"),
        "problem_type": preprocessing_artifacts.get("problem_type"),
        "created_at": datetime.utcnow().isoformat(),
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to: {model_dir}")
    
    return str(model_dir)


@task(
    name="register_model",
    description="Register model in registry",
    tags=["model", "registry"]
)
def register_model_task(
    model_path: str,
    model_name: str,
    metrics: Dict[str, float],
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Register model in the model registry.
    
    Args:
        model_path: Path to model directory
        model_name: Name for the registered model
        metrics: Model metrics
        description: Model description
        tags: Model tags
        
    Returns:
        Registration info
    """
    logger = get_run_logger()
    
    from mlops.registry import model_registry
    
    model_file = Path(model_path) / "model.joblib"
    
    registration = model_registry.register_model(
        name=model_name,
        model_path=str(model_file),
        description=description,
        tags=tags,
        metadata={"metrics": metrics},
    )
    
    logger.info(f"Registered model: {registration['version_id']}")
    
    return registration


@task(
    name="deploy_model",
    description="Deploy model to production",
    tags=["model", "deployment"]
)
def deploy_model_task(
    model_name: str,
    version: int,
    deployment_target: str = "local",
    deployment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deploy model to specified target.
    
    Args:
        model_name: Registered model name
        version: Model version to deploy
        deployment_target: Deployment target (local, docker, kubernetes)
        deployment_config: Deployment configuration
        
    Returns:
        Deployment info
    """
    logger = get_run_logger()
    
    from mlops.registry import model_registry, ModelStage
    
    # Transition model to production
    model_registry.transition_stage(
        model_name, 
        version, 
        ModelStage.PRODUCTION
    )
    
    deployment_info = {
        "model_name": model_name,
        "version": version,
        "target": deployment_target,
        "status": "deployed",
        "deployed_at": datetime.utcnow().isoformat(),
    }
    
    if deployment_target == "docker":
        # Generate docker deployment commands
        deployment_info["commands"] = [
            f"docker build -t {model_name}:v{version} .",
            f"docker run -p 8080:8080 {model_name}:v{version}",
        ]
    
    logger.info(f"Deployed {model_name} v{version} to {deployment_target}")
    
    return deployment_info


@task(
    name="check_drift",
    description="Check for data drift",
    tags=["monitoring", "drift"]
)
def check_drift_task(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Check for data drift between reference and current data.
    
    Args:
        reference_data: Reference/training data
        current_data: Current/production data
        threshold: Drift detection threshold
        
    Returns:
        Drift report
    """
    logger = get_run_logger()
    logger.info("Checking for data drift...")
    
    from scipy import stats
    
    drift_report = {
        "drift_detected": False,
        "columns_with_drift": [],
        "column_scores": {},
    }
    
    for col in reference_data.columns:
        if col not in current_data.columns:
            continue
        
        ref = reference_data[col].dropna()
        curr = current_data[col].dropna()
        
        if len(ref) == 0 or len(curr) == 0:
            continue
        
        # Use KS test for numeric columns
        if pd.api.types.is_numeric_dtype(ref):
            statistic, p_value = stats.ks_2samp(ref, curr)
            drift_score = statistic
        else:
            # Use chi-square for categorical
            ref_counts = ref.value_counts(normalize=True)
            curr_counts = curr.value_counts(normalize=True)
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            
            ref_dist = [ref_counts.get(c, 0.001) for c in all_categories]
            curr_dist = [curr_counts.get(c, 0.001) for c in all_categories]
            
            try:
                statistic, p_value = stats.chisquare(curr_dist, ref_dist)
                drift_score = min(statistic / 100, 1.0)
            except Exception:
                drift_score = 0.0
        
        drift_report["column_scores"][col] = round(drift_score, 4)
        
        if drift_score > threshold:
            drift_report["drift_detected"] = True
            drift_report["columns_with_drift"].append(col)
    
    logger.info(f"Drift check complete. Drift detected: {drift_report['drift_detected']}")
    
    return drift_report


@task(
    name="send_notification",
    description="Send pipeline notification",
    tags=["notification"]
)
def send_notification_task(
    message: str,
    notification_type: str = "info",
    channels: Optional[List[str]] = None,
) -> bool:
    """
    Send notification about pipeline events.
    
    Args:
        message: Notification message
        notification_type: Type (info, warning, error, success)
        channels: Notification channels (email, slack, etc.)
        
    Returns:
        Success status
    """
    logger = get_run_logger()
    
    # Log notification (in real implementation, send to channels)
    logger.info(f"[{notification_type.upper()}] {message}")
    
    # In production, integrate with notification services:
    # - Email via SMTP
    # - Slack via webhook
    # - PagerDuty for alerts
    
    return True
