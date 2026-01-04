"""Shared Pydantic schemas for request/response contracts."""

from .auth import (
    LoginRequest,
    LoginResponse,
    TokenResponse,
    UserResponse,
    APIKeyCreateRequest,
    APIKeyResponse,
)

from .prediction import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionResult,
    FeatureValue,
)

from .deployment import (
    DeployModelRequest,
    DeploymentResponse,
    DeploymentStatus,
    DeploymentConfig,
)

from .monitoring import (
    DriftMonitoringRequest,
    DriftMonitoringResponse,
    DriftAlert,
    DriftMetrics,
)

from .common import (
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
    PaginatedResponse,
)

__all__ = [
    # Auth
    "LoginRequest",
    "LoginResponse", 
    "TokenResponse",
    "UserResponse",
    "APIKeyCreateRequest",
    "APIKeyResponse",
    # Prediction
    "PredictRequest",
    "PredictResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "PredictionResult",
    "FeatureValue",
    # Deployment
    "DeployModelRequest",
    "DeploymentResponse",
    "DeploymentStatus",
    "DeploymentConfig",
    # Monitoring
    "DriftMonitoringRequest",
    "DriftMonitoringResponse",
    "DriftAlert",
    "DriftMetrics",
    # Common
    "HealthResponse",
    "ReadyResponse",
    "ErrorResponse",
    "PaginatedResponse",
]
