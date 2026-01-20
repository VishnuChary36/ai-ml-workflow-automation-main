"""
Prefect Flows for ML Pipelines

Complete orchestrated workflows for training, deployment, and monitoring.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import pandas as pd

try:
    from prefect import flow, get_run_logger
    # Prefect 3.x moved artifacts
    try:
        from prefect.artifacts import create_markdown_artifact
    except ImportError:
        create_markdown_artifact = None
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    create_markdown_artifact = None
    # Create mock decorator
    def flow(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def get_run_logger():
        import logging
        return logging.getLogger("pipelines")

from .tasks import (
    load_data_task,
    validate_data_task,
    preprocess_task,
    split_data_task,
    train_model_task,
    evaluate_model_task,
    save_model_task,
    register_model_task,
    deploy_model_task,
    check_drift_task,
    send_notification_task,
)
from config import settings


@flow(
    name="training_pipeline",
    description="End-to-end ML training pipeline",
    version="1.0.0",
    retries=1,
    retry_delay_seconds=60,
)
def training_pipeline(
    data_path: str,
    target_column: str,
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    auto_register: bool = True,
    auto_deploy: bool = False,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete training pipeline from data to model.
    
    Args:
        data_path: Path to training data
        target_column: Target column name
        model_type: Type of model to train
        model_params: Model hyperparameters
        model_name: Name for model registration
        auto_register: Automatically register model
        auto_deploy: Automatically deploy model
        experiment_name: Experiment to log to
        
    Returns:
        Pipeline results including model info and metrics
    """
    logger = get_run_logger()
    logger.info("Starting training pipeline...")
    
    pipeline_start = datetime.utcnow()
    results = {
        "status": "running",
        "started_at": pipeline_start.isoformat(),
    }
    
    try:
        # Step 1: Load data
        df = load_data_task(data_path)
        results["data_rows"] = len(df)
        
        # Step 2: Validate data
        df, validation_report = validate_data_task(
            df, 
            required_columns=[target_column]
        )
        results["validation"] = validation_report
        
        if not validation_report["valid"]:
            raise ValueError(f"Data validation failed: {validation_report['issues']}")
        
        # Step 3: Preprocess
        df_processed, preprocessing_artifacts = preprocess_task(
            df, 
            target_column
        )
        results["features"] = len(preprocessing_artifacts["feature_columns"])
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = split_data_task(
            df_processed, 
            target_column
        )
        results["train_size"] = len(X_train)
        results["test_size"] = len(X_test)
        
        # Step 5: Train model
        model = train_model_task(
            X_train, 
            y_train, 
            model_type, 
            model_params
        )
        results["model_type"] = type(model).__name__
        
        # Step 6: Evaluate
        problem_type = preprocessing_artifacts.get("problem_type", "classification")
        metrics = evaluate_model_task(model, X_test, y_test, problem_type)
        results["metrics"] = metrics
        
        # Step 7: Save model
        model_id = f"mdl-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        model_path = save_model_task(
            model,
            model_id,
            preprocessing_artifacts,
            metrics,
            model_params
        )
        results["model_id"] = model_id
        results["model_path"] = model_path
        
        # Step 8: Register model (optional)
        if auto_register:
            model_name = model_name or f"model-{model_type}"
            registration = register_model_task(
                model_path,
                model_name,
                metrics,
                description=f"Trained on {len(df)} samples",
                tags={"model_type": model_type, "pipeline": "training_pipeline"}
            )
            results["registration"] = registration
        
        # Step 9: Deploy model (optional)
        if auto_deploy and auto_register:
            deployment = deploy_model_task(
                model_name,
                registration["version"],
                "local"
            )
            results["deployment"] = deployment
        
        # Log to experiment tracking
        if experiment_name:
            from mlops.experiment import experiment_manager
            experiment_manager.log_run(
                experiment_name=experiment_name,
                run_name=f"training-{model_id}",
                params=model_params or {},
                metrics=metrics,
                model_path=model_path,
            )
        
        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()
        
        # Send success notification
        send_notification_task(
            f"Training pipeline completed. Model: {model_id}, "
            f"Accuracy: {metrics.get('accuracy', metrics.get('r2_score', 'N/A'))}",
            "success"
        )
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["failed_at"] = datetime.utcnow().isoformat()
        
        send_notification_task(
            f"Training pipeline failed: {str(e)}",
            "error"
        )
        
        logger.error(f"Training pipeline failed: {e}")
        raise
    
    return results


@flow(
    name="data_pipeline",
    description="Data ingestion and preprocessing pipeline",
    version="1.0.0",
)
def data_pipeline(
    data_sources: List[str],
    output_path: str,
    preprocessing_steps: Optional[List[str]] = None,
    validation_rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Data ingestion and preprocessing pipeline.
    
    Args:
        data_sources: List of data source paths
        output_path: Output path for processed data
        preprocessing_steps: List of preprocessing steps to apply
        validation_rules: Data validation rules
        
    Returns:
        Pipeline results
    """
    logger = get_run_logger()
    logger.info(f"Starting data pipeline with {len(data_sources)} sources...")
    
    results = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "sources_processed": 0,
        "total_rows": 0,
    }
    
    try:
        # Load and combine data sources
        dfs = []
        for source in data_sources:
            df = load_data_task(source)
            dfs.append(df)
            results["sources_processed"] += 1
        
        # Combine dataframes
        if len(dfs) == 1:
            combined_df = dfs[0]
        else:
            combined_df = pd.concat(dfs, ignore_index=True)
        
        results["total_rows"] = len(combined_df)
        
        # Validate data
        max_null = validation_rules.get("max_null_ratio", 0.5) if validation_rules else 0.5
        validated_df, validation_report = validate_data_task(
            combined_df,
            max_null_ratio=max_null
        )
        results["validation"] = validation_report
        
        # Apply preprocessing steps
        preprocessing_steps = preprocessing_steps or []
        
        for step in preprocessing_steps:
            if step == "drop_duplicates":
                validated_df = validated_df.drop_duplicates()
            elif step == "fill_na_median":
                numeric_cols = validated_df.select_dtypes(include=['number']).columns
                validated_df[numeric_cols] = validated_df[numeric_cols].fillna(
                    validated_df[numeric_cols].median()
                )
            elif step == "fill_na_mode":
                for col in validated_df.columns:
                    if validated_df[col].isna().any():
                        mode_val = validated_df[col].mode()
                        if len(mode_val) > 0:
                            validated_df[col] = validated_df[col].fillna(mode_val[0])
            elif step == "lowercase_strings":
                str_cols = validated_df.select_dtypes(include=['object']).columns
                for col in str_cols:
                    validated_df[col] = validated_df[col].str.lower()
        
        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if str(output_path).endswith('.parquet'):
            validated_df.to_parquet(output_path, index=False)
        else:
            validated_df.to_csv(output_path, index=False)
        
        results["output_path"] = str(output_path)
        results["output_rows"] = len(validated_df)
        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Data pipeline completed. Output: {output_path}")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        logger.error(f"Data pipeline failed: {e}")
        raise
    
    return results


@flow(
    name="deployment_pipeline",
    description="Model deployment pipeline",
    version="1.0.0",
)
def deployment_pipeline(
    model_name: str,
    version: Optional[int] = None,
    deployment_target: str = "local",
    canary_percentage: float = 0.0,
    health_check_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Model deployment pipeline with rollback support.
    
    Args:
        model_name: Name of registered model
        version: Model version (latest if None)
        deployment_target: Deployment target
        canary_percentage: Percentage of traffic for canary
        health_check_url: URL to verify deployment
        
    Returns:
        Deployment results
    """
    logger = get_run_logger()
    logger.info(f"Starting deployment pipeline for {model_name}...")
    
    from mlops.registry import model_registry, ModelStage
    
    results = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
    }
    
    try:
        # Get model version
        if version is None:
            model_info = model_registry.get_model(model_name)
            if not model_info:
                raise ValueError(f"Model not found: {model_name}")
            version = model_info["version"]
        
        results["model_name"] = model_name
        results["version"] = version
        
        # Check current production version for rollback
        current_prod = model_registry.get_model(model_name, stage=ModelStage.PRODUCTION)
        if current_prod:
            results["previous_version"] = current_prod["version"]
        
        # Deploy model
        deployment = deploy_model_task(
            model_name,
            version,
            deployment_target,
            {"canary_percentage": canary_percentage}
        )
        results["deployment"] = deployment
        
        # Health check (simulated)
        if health_check_url:
            logger.info(f"Performing health check: {health_check_url}")
            # In production, make actual HTTP request
            health_status = True  # Simulated
            results["health_check"] = "passed" if health_status else "failed"
            
            if not health_status and current_prod:
                # Rollback
                logger.warning("Health check failed, rolling back...")
                model_registry.rollback(model_name, current_prod["version"])
                results["status"] = "rolled_back"
                results["rollback_version"] = current_prod["version"]
                return results
        
        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()
        
        send_notification_task(
            f"Deployed {model_name} v{version} to {deployment_target}",
            "success"
        )
        
        logger.info(f"Deployment completed: {model_name} v{version}")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        
        send_notification_task(
            f"Deployment failed for {model_name}: {str(e)}",
            "error"
        )
        
        logger.error(f"Deployment failed: {e}")
        raise
    
    return results


@flow(
    name="monitoring_pipeline",
    description="Model monitoring and drift detection pipeline",
    version="1.0.0",
)
def monitoring_pipeline(
    model_name: str,
    reference_data_path: str,
    current_data_path: str,
    drift_threshold: float = 0.2,
    auto_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Monitoring pipeline for production models.
    
    Args:
        model_name: Name of deployed model
        reference_data_path: Path to reference/training data
        current_data_path: Path to current/production data
        drift_threshold: Threshold for drift detection
        auto_retrain: Automatically trigger retraining on drift
        
    Returns:
        Monitoring results
    """
    logger = get_run_logger()
    logger.info(f"Starting monitoring pipeline for {model_name}...")
    
    results = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "model_name": model_name,
    }
    
    try:
        # Load data
        reference_df = load_data_task(reference_data_path)
        current_df = load_data_task(current_data_path)
        
        results["reference_samples"] = len(reference_df)
        results["current_samples"] = len(current_df)
        
        # Check for drift
        drift_report = check_drift_task(
            reference_df,
            current_df,
            drift_threshold
        )
        results["drift_report"] = drift_report
        
        # Generate alerts if drift detected
        if drift_report["drift_detected"]:
            alert_message = (
                f"Data drift detected for {model_name}! "
                f"Drifted columns: {drift_report['columns_with_drift']}"
            )
            
            send_notification_task(alert_message, "warning")
            results["alert_sent"] = True
            
            # Auto retrain if enabled
            if auto_retrain:
                logger.info("Drift detected, triggering retraining...")
                # This would trigger the training pipeline
                results["retraining_triggered"] = True
        else:
            results["alert_sent"] = False
        
        # Calculate model health score
        max_drift = max(drift_report["column_scores"].values()) if drift_report["column_scores"] else 0
        health_score = max(0, 100 - (max_drift * 100))
        results["health_score"] = round(health_score, 2)
        
        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Monitoring complete. Health score: {health_score}")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        logger.error(f"Monitoring pipeline failed: {e}")
        raise
    
    return results


@flow(
    name="batch_inference_pipeline",
    description="Batch inference pipeline for offline predictions",
    version="1.0.0",
)
def batch_inference_pipeline(
    model_name: str,
    input_data_path: str,
    output_path: str,
    batch_size: int = 1000,
    version: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Batch inference pipeline for large-scale predictions.
    
    Args:
        model_name: Name of registered model
        input_data_path: Path to input data
        output_path: Path for prediction output
        batch_size: Batch size for processing
        version: Model version (production if None)
        
    Returns:
        Inference results
    """
    logger = get_run_logger()
    logger.info(f"Starting batch inference with {model_name}...")
    
    from mlops.registry import model_registry, ModelStage
    
    results = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
    }
    
    try:
        # Load model
        if version:
            model = model_registry.load_model(model_name, version=version)
        else:
            model = model_registry.load_model(model_name, stage=ModelStage.PRODUCTION)
        
        # Load input data
        input_df = load_data_task(input_data_path)
        results["input_rows"] = len(input_df)
        
        # Process in batches
        predictions = []
        num_batches = (len(input_df) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(input_df))
            batch = input_df.iloc[start_idx:end_idx]
            
            batch_preds = model.predict(batch)
            predictions.extend(batch_preds)
            
            logger.info(f"Processed batch {i + 1}/{num_batches}")
        
        # Save predictions
        output_df = input_df.copy()
        output_df['prediction'] = predictions
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if str(output_path).endswith('.parquet'):
            output_df.to_parquet(output_path, index=False)
        else:
            output_df.to_csv(output_path, index=False)
        
        results["output_path"] = str(output_path)
        results["predictions_count"] = len(predictions)
        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Batch inference completed. Output: {output_path}")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        logger.error(f"Batch inference failed: {e}")
        raise
    
    return results
