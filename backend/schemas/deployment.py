"""Deployment schemas."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class DeploymentStatus(str, Enum):
    """Deployment status enum."""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


class DeploymentConfig(BaseModel):
    """Deployment configuration schema."""
    replicas: int = Field(default=1, ge=1, le=100, description="Number of replicas")
    cpu_limit: str = Field(default="1", description="CPU limit (e.g., '1', '500m')")
    memory_limit: str = Field(default="512Mi", description="Memory limit (e.g., '512Mi', '1Gi')")
    auto_scale: bool = Field(default=False, description="Enable auto-scaling")
    min_replicas: int = Field(default=1, ge=1, description="Minimum replicas for auto-scaling")
    max_replicas: int = Field(default=10, ge=1, description="Maximum replicas for auto-scaling")
    health_check_path: str = Field(default="/health", description="Health check endpoint")
    readiness_check_path: str = Field(default="/ready", description="Readiness check endpoint")
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "replicas": 2,
                "cpu_limit": "500m",
                "memory_limit": "1Gi",
                "auto_scale": True,
                "min_replicas": 1,
                "max_replicas": 5,
                "health_check_path": "/health",
                "readiness_check_path": "/ready",
                "environment_variables": {"LOG_LEVEL": "INFO"}
            }
        }
    }


class DeployModelRequest(BaseModel):
    """Deploy model request schema."""
    model_id: str = Field(..., description="Model ID to deploy")
    platform: str = Field(..., description="Deployment platform: local, docker, kubernetes, aws, gcp, azure")
    config: Optional[DeploymentConfig] = Field(None, description="Deployment configuration")
    image_tag: Optional[str] = Field(None, description="Custom image tag (auto-generated if not provided)")
    canary_percentage: Optional[int] = Field(None, ge=0, le=100, description="Canary deployment percentage")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "mdl-abc123",
                "platform": "kubernetes",
                "config": {
                    "replicas": 2,
                    "auto_scale": True
                },
                "canary_percentage": 10
            }
        }
    }


class DeploymentResponse(BaseModel):
    """Deployment response schema."""
    deployment_id: str = Field(..., description="Deployment ID")
    model_id: str = Field(..., description="Model ID")
    task_id: Optional[str] = Field(None, description="Background task ID")
    platform: str = Field(..., description="Deployment platform")
    status: DeploymentStatus = Field(..., description="Deployment status")
    url: Optional[str] = Field(None, description="Deployment URL")
    live_prediction_url: Optional[str] = Field(None, description="Live prediction endpoint")
    image_tag: Optional[str] = Field(None, description="Docker image tag")
    package_path: Optional[str] = Field(None, description="Path to deployment package")
    config: Optional[DeploymentConfig] = Field(None, description="Deployment configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "deployment_id": "dep-abc123",
                "model_id": "mdl-abc123",
                "task_id": "task-xyz789",
                "platform": "kubernetes",
                "status": "running",
                "url": "https://ml-model-abc123.example.com",
                "live_prediction_url": "https://ml-model-abc123.example.com/predict",
                "image_tag": "model:v1.0.0+abc123",
                "created_at": "2025-01-01T00:00:00Z"
            }
        }
    }


class DeploymentListResponse(BaseModel):
    """List deployments response schema."""
    deployments: List[DeploymentResponse] = Field(..., description="List of deployments")
    total: int = Field(..., description="Total number of deployments")
    model_id: Optional[str] = Field(None, description="Filter by model ID")
