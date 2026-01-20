"""
MLflow Model Tracking Integration

Provides centralized experiment tracking, parameter logging, metric tracking,
and artifact storage using MLflow.
"""

import os
import json
import time
import uuid
import joblib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MlflowClient = None

from config import settings


class MLflowTracker:
    """
    MLflow integration for experiment tracking and model logging.
    
    Features:
    - Experiment management
    - Run tracking with parameters, metrics, artifacts
    - Model logging with signatures
    - Automatic environment capture
    - Custom tags and metadata
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
            artifact_location: Default artifact storage location
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", 
            "sqlite:///mlflow.db"
        )
        self.default_experiment = experiment_name or "ai-ml-workflow"
        self.artifact_location = artifact_location or settings.artifact_storage_path
        
        self._client: Optional[MlflowClient] = None
        self._active_run = None
        self._initialized = False
        
        if MLFLOW_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize MLflow connection."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = MlflowClient(self.tracking_uri)
            
            # Create or get default experiment
            experiment = mlflow.get_experiment_by_name(self.default_experiment)
            if experiment is None:
                mlflow.create_experiment(
                    self.default_experiment,
                    artifact_location=self.artifact_location
                )
            
            self._initialized = True
            print(f"✓ MLflow initialized: {self.tracking_uri}")
        except Exception as e:
            print(f"⚠ MLflow initialization failed: {e}")
            self._initialized = False
    
    @property
    def client(self) -> Optional[MlflowClient]:
        """Get MLflow client."""
        return self._client
    
    @property
    def is_available(self) -> bool:
        """Check if MLflow is available and initialized."""
        return MLFLOW_AVAILABLE and self._initialized
    
    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            artifact_location: Custom artifact location
            tags: Experiment-level tags
            
        Returns:
            Experiment ID
        """
        if not self.is_available:
            return f"local-exp-{uuid.uuid4().hex[:8]}"
        
        try:
            # Check if exists
            experiment = mlflow.get_experiment_by_name(name)
            if experiment:
                return experiment.experiment_id
            
            # Create new
            experiment_id = mlflow.create_experiment(
                name,
                artifact_location=artifact_location or self.artifact_location,
                tags=tags or {}
            )
            return experiment_id
        except MlflowException as e:
            print(f"Failed to create experiment: {e}")
            return f"local-exp-{uuid.uuid4().hex[:8]}"
    
    @contextmanager
    def start_run(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """
        Context manager for MLflow runs.
        
        Args:
            experiment_name: Experiment to use (defaults to default_experiment)
            run_name: Human-readable run name
            tags: Run tags
            nested: Whether this is a nested run
            
        Yields:
            RunContext with logging methods
        """
        if not self.is_available:
            # Return mock context for when MLflow is unavailable
            yield MockRunContext()
            return
        
        experiment = experiment_name or self.default_experiment
        mlflow.set_experiment(experiment)
        
        try:
            run = mlflow.start_run(
                run_name=run_name,
                tags=tags or {},
                nested=nested
            )
            self._active_run = run
            
            # Add standard tags
            mlflow.set_tag("mlflow.source.type", "ai-ml-workflow")
            mlflow.set_tag("run.timestamp", datetime.utcnow().isoformat())
            
            yield RunContext(run)
            
        finally:
            mlflow.end_run()
            self._active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the active run."""
        if not self.is_available or not self._active_run:
            return
        
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        
        # MLflow has a limit on param value length
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            try:
                mlflow.log_param(key, str_value)
            except Exception:
                pass
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics to the active run."""
        if not self.is_available or not self._active_run:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        input_example: Optional[pd.DataFrame] = None,
        signature: Optional[Any] = None,
        conda_env: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Log a model to MLflow.
        
        Args:
            model: The trained model object
            artifact_path: Path within the run's artifacts
            registered_model_name: If provided, registers the model
            input_example: Example input for inference
            signature: Model signature (auto-inferred if not provided)
            conda_env: Conda environment specification
            metadata: Additional model metadata
            
        Returns:
            Model URI if successful
        """
        if not self.is_available or not self._active_run:
            return None
        
        try:
            # Infer signature if example provided
            if signature is None and input_example is not None:
                try:
                    predictions = model.predict(input_example)
                    signature = infer_signature(input_example, predictions)
                except Exception:
                    pass
            
            # Log the model
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
            )
            
            # Log additional metadata
            if metadata:
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', delete=False
                ) as f:
                    json.dump(metadata, f, indent=2, default=str)
                    mlflow.log_artifact(f.name, artifact_path=f"{artifact_path}/metadata")
            
            return model_info.model_uri
            
        except Exception as e:
            print(f"Failed to log model: {e}")
            return None
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if not self.is_available or not self._active_run:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Failed to log artifact: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all artifacts in a directory."""
        if not self.is_available or not self._active_run:
            return
        
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
        except Exception as e:
            print(f"Failed to log artifacts: {e}")
    
    def log_figure(self, figure: Any, artifact_file: str):
        """Log a matplotlib or plotly figure."""
        if not self.is_available or not self._active_run:
            return
        
        try:
            mlflow.log_figure(figure, artifact_file)
        except Exception as e:
            print(f"Failed to log figure: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags on the active run."""
        if not self.is_available or not self._active_run:
            return
        
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get run information by ID."""
        if not self.is_available:
            return None
        
        try:
            run = self._client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
            }
        except Exception:
            return None
    
    def search_runs(
        self,
        experiment_names: Optional[List[str]] = None,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> List[Dict]:
        """Search for runs matching criteria."""
        if not self.is_available:
            return []
        
        try:
            experiments = experiment_names or [self.default_experiment]
            experiment_ids = []
            
            for name in experiments:
                exp = mlflow.get_experiment_by_name(name)
                if exp:
                    experiment_ids.append(exp.experiment_id)
            
            if not experiment_ids:
                return []
            
            runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string or "",
                order_by=order_by or ["start_time DESC"],
                max_results=max_results,
            )
            
            return runs.to_dict(orient="records")
        except Exception:
            return []
    
    def _flatten_dict(
        self, 
        d: Dict, 
        parent_key: str = '', 
        sep: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class RunContext:
    """Context object for active MLflow run."""
    
    def __init__(self, run):
        self.run = run
        self.run_id = run.info.run_id
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        mlflow.log_params(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        mlflow.set_tag(key, value)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        mlflow.log_artifact(local_path, artifact_path)


class MockRunContext:
    """Mock context for when MLflow is unavailable."""
    
    def __init__(self):
        self.run_id = f"local-{uuid.uuid4().hex[:8]}"
        self._params = {}
        self._metrics = {}
        self._tags = {}
    
    def log_param(self, key: str, value: Any):
        self._params[key] = value
    
    def log_params(self, params: Dict[str, Any]):
        self._params.update(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        self._metrics[key] = value
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._metrics.update(metrics)
    
    def set_tag(self, key: str, value: str):
        self._tags[key] = value
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass


# Global tracker instance
mlflow_tracker = MLflowTracker()
