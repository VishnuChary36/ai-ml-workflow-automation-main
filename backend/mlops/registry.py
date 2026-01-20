"""
Model Registry

Provides model versioning, staging transitions, and lifecycle management
for ML models in production.
"""

import os
import json
import uuid
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum

import joblib

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities.model_registry import ModelVersion
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config import settings


class ModelStage(str, Enum):
    """Model lifecycle stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelStatus(str, Enum):
    """Model registration status."""
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class ModelRegistry:
    """
    Model Registry for versioning and lifecycle management.
    
    Features:
    - Model registration with versioning
    - Stage transitions (staging, production, archived)
    - Model metadata and tags
    - Model comparison and lineage
    - Rollback support
    - Local and MLflow backends
    """
    
    def __init__(
        self,
        registry_path: Optional[str] = None,
        use_mlflow: bool = True,
    ):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path for local registry storage
            use_mlflow: Whether to use MLflow registry
        """
        self.registry_path = Path(
            registry_path or 
            os.getenv("MODEL_REGISTRY_PATH", "./model_registry")
        )
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self._client: Optional[MlflowClient] = None
        
        # Local registry index
        self.index_file = self.registry_path / "registry_index.json"
        self._index = self._load_index()
        
        if self.use_mlflow:
            try:
                self._client = MlflowClient()
            except Exception:
                self.use_mlflow = False
    
    def _load_index(self) -> Dict:
        """Load local registry index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {"models": {}, "versions": {}}
    
    def _save_index(self):
        """Save local registry index."""
        with open(self.index_file, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
    
    def _compute_model_hash(self, model_path: str) -> str:
        """Compute SHA256 hash of model file."""
        sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def register_model(
        self,
        name: str,
        model_path: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new model or new version.
        
        Args:
            name: Model name (unique identifier)
            model_path: Path to the model file
            description: Model description
            tags: Model tags
            metadata: Additional metadata (metrics, params, etc.)
            run_id: MLflow run ID if applicable
            
        Returns:
            Registration info with version
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version
        version = self._get_next_version(name)
        version_id = f"{name}-v{version}"
        
        # Compute model hash
        model_hash = self._compute_model_hash(str(model_path))
        
        # Check for duplicate
        if name in self._index["models"]:
            for existing_version in self._index["versions"].get(name, []):
                if existing_version.get("hash") == model_hash:
                    print(f"Model with same hash already registered: {existing_version['version_id']}")
                    return existing_version
        
        # Create version directory
        version_dir = self.registry_path / name / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        dest_path = version_dir / "model.joblib"
        shutil.copy2(model_path, dest_path)
        
        # Copy additional artifacts if in a directory
        if model_path.parent != Path("."):
            for artifact in model_path.parent.glob("*"):
                if artifact.is_file() and artifact.name != model_path.name:
                    shutil.copy2(artifact, version_dir / artifact.name)
        
        # Create version info
        version_info = {
            "version_id": version_id,
            "name": name,
            "version": version,
            "path": str(dest_path),
            "hash": model_hash,
            "description": description,
            "tags": tags or {},
            "metadata": metadata or {},
            "stage": ModelStage.NONE.value,
            "status": ModelStatus.READY.value,
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "created_by": os.getenv("USER", "system"),
        }
        
        # Save version metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(version_info, f, indent=2, default=str)
        
        # Update index
        if name not in self._index["models"]:
            self._index["models"][name] = {
                "name": name,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "latest_version": version,
                "production_version": None,
                "staging_version": None,
            }
        else:
            self._index["models"][name]["latest_version"] = version
        
        if name not in self._index["versions"]:
            self._index["versions"][name] = []
        self._index["versions"][name].append(version_info)
        
        self._save_index()
        
        # Register with MLflow if enabled
        if self.use_mlflow and run_id:
            try:
                self._register_mlflow_model(
                    name=name,
                    run_id=run_id,
                    description=description,
                    tags=tags,
                )
            except Exception as e:
                print(f"MLflow registration failed: {e}")
        
        print(f"✓ Registered model: {version_id}")
        return version_info
    
    def _get_next_version(self, name: str) -> int:
        """Get next version number for a model."""
        if name not in self._index["models"]:
            return 1
        return self._index["models"][name].get("latest_version", 0) + 1
    
    def _register_mlflow_model(
        self,
        name: str,
        run_id: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Register model with MLflow Model Registry."""
        if not self._client:
            return
        
        try:
            # Create registered model if not exists
            try:
                self._client.create_registered_model(
                    name,
                    description=description,
                    tags=tags,
                )
            except MlflowException:
                pass  # Model already exists
            
            # Create model version
            model_uri = f"runs:/{run_id}/model"
            self._client.create_model_version(
                name=name,
                source=model_uri,
                run_id=run_id,
                description=description,
                tags=tags,
            )
        except Exception as e:
            print(f"MLflow model registration error: {e}")
    
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage,
        archive_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Transition model version to a new stage.
        
        Args:
            name: Model name
            version: Version number
            stage: Target stage
            archive_existing: Archive existing model in target stage
            
        Returns:
            Updated version info
        """
        if name not in self._index["versions"]:
            raise ValueError(f"Model not found: {name}")
        
        # Find version
        version_info = None
        for v in self._index["versions"][name]:
            if v["version"] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version not found: {name} v{version}")
        
        # Archive existing if transitioning to production/staging
        if archive_existing and stage in [ModelStage.PRODUCTION, ModelStage.STAGING]:
            for v in self._index["versions"][name]:
                if v["stage"] == stage.value and v["version"] != version:
                    v["stage"] = ModelStage.ARCHIVED.value
                    v["updated_at"] = datetime.utcnow().isoformat()
        
        # Update version stage
        version_info["stage"] = stage.value
        version_info["updated_at"] = datetime.utcnow().isoformat()
        
        # Update model info
        if stage == ModelStage.PRODUCTION:
            self._index["models"][name]["production_version"] = version
        elif stage == ModelStage.STAGING:
            self._index["models"][name]["staging_version"] = version
        
        self._save_index()
        
        # Update MLflow if available
        if self.use_mlflow and self._client:
            try:
                self._client.transition_model_version_stage(
                    name=name,
                    version=str(version),
                    stage=stage.value,
                    archive_existing_versions=archive_existing,
                )
            except Exception:
                pass
        
        print(f"✓ Transitioned {name} v{version} to {stage.value}")
        return version_info
    
    def get_model(
        self,
        name: str,
        version: Optional[int] = None,
        stage: Optional[ModelStage] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get model by name and version/stage.
        
        Args:
            name: Model name
            version: Specific version (optional)
            stage: Stage to get (optional, e.g., PRODUCTION)
            
        Returns:
            Model version info or None
        """
        if name not in self._index["versions"]:
            return None
        
        versions = self._index["versions"][name]
        
        if version:
            for v in versions:
                if v["version"] == version:
                    return v
            return None
        
        if stage:
            for v in reversed(versions):  # Latest first
                if v["stage"] == stage.value:
                    return v
            return None
        
        # Return latest version
        return versions[-1] if versions else None
    
    def load_model(
        self,
        name: str,
        version: Optional[int] = None,
        stage: Optional[ModelStage] = None,
    ) -> Any:
        """
        Load a model from registry.
        
        Args:
            name: Model name
            version: Specific version (optional)
            stage: Stage to load from (optional)
            
        Returns:
            Loaded model object
        """
        model_info = self.get_model(name, version, stage)
        if not model_info:
            raise ValueError(f"Model not found: {name}")
        
        model_path = model_info["path"]
        return joblib.load(model_path)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return list(self._index["models"].values())
    
    def list_versions(
        self,
        name: str,
        stage: Optional[ModelStage] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            name: Model name
            stage: Filter by stage (optional)
            
        Returns:
            List of version info dicts
        """
        if name not in self._index["versions"]:
            return []
        
        versions = self._index["versions"][name]
        
        if stage:
            return [v for v in versions if v["stage"] == stage.value]
        
        return versions
    
    def compare_versions(
        self,
        name: str,
        version_a: int,
        version_b: int,
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            name: Model name
            version_a: First version
            version_b: Second version
            
        Returns:
            Comparison results
        """
        v_a = self.get_model(name, version_a)
        v_b = self.get_model(name, version_b)
        
        if not v_a or not v_b:
            raise ValueError("One or both versions not found")
        
        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics_comparison": {},
            "param_differences": {},
        }
        
        # Compare metrics
        metrics_a = v_a.get("metadata", {}).get("metrics", {})
        metrics_b = v_b.get("metadata", {}).get("metrics", {})
        
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
        for metric in all_metrics:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)
            comparison["metrics_comparison"][metric] = {
                "version_a": val_a,
                "version_b": val_b,
                "diff": (val_b - val_a) if val_a and val_b else None,
            }
        
        # Compare params
        params_a = v_a.get("metadata", {}).get("params", {})
        params_b = v_b.get("metadata", {}).get("params", {})
        
        all_params = set(params_a.keys()) | set(params_b.keys())
        for param in all_params:
            if params_a.get(param) != params_b.get(param):
                comparison["param_differences"][param] = {
                    "version_a": params_a.get(param),
                    "version_b": params_b.get(param),
                }
        
        return comparison
    
    def delete_version(self, name: str, version: int) -> bool:
        """
        Delete a model version.
        
        Args:
            name: Model name
            version: Version to delete
            
        Returns:
            True if deleted
        """
        if name not in self._index["versions"]:
            return False
        
        # Find and remove version
        versions = self._index["versions"][name]
        for i, v in enumerate(versions):
            if v["version"] == version:
                # Can't delete production version
                if v["stage"] == ModelStage.PRODUCTION.value:
                    raise ValueError("Cannot delete production model")
                
                # Mark as deleted
                v["status"] = ModelStatus.DELETED.value
                v["updated_at"] = datetime.utcnow().isoformat()
                
                # Optionally remove files
                version_dir = self.registry_path / name / f"v{version}"
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                
                self._save_index()
                print(f"✓ Deleted {name} v{version}")
                return True
        
        return False
    
    def rollback(
        self,
        name: str,
        target_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Rollback production model to previous version.
        
        Args:
            name: Model name
            target_version: Specific version to rollback to (optional)
            
        Returns:
            New production version info
        """
        if name not in self._index["models"]:
            raise ValueError(f"Model not found: {name}")
        
        current_prod = self._index["models"][name].get("production_version")
        if not current_prod:
            raise ValueError("No production version to rollback from")
        
        # Find target version
        if target_version:
            target = self.get_model(name, target_version)
        else:
            # Find previous production version
            versions = sorted(
                [v for v in self._index["versions"][name] 
                 if v["version"] < current_prod and v["status"] == ModelStatus.READY.value],
                key=lambda x: x["version"],
                reverse=True
            )
            if not versions:
                raise ValueError("No previous version available for rollback")
            target = versions[0]
            target_version = target["version"]
        
        # Transition to production
        return self.transition_stage(
            name, 
            target_version, 
            ModelStage.PRODUCTION,
            archive_existing=True
        )


# Global registry instance
model_registry = ModelRegistry()
