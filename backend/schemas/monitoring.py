"""Monitoring and drift detection schemas."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class DriftType(str, Enum):
    """Types of drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"


class DriftSeverity(str, Enum):
    """Drift severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftMonitoringRequest(BaseModel):
    """Start drift monitoring request schema."""
    model_id: str = Field(..., description="Model ID to monitor")
    reference_dataset_id: str = Field(..., description="Reference dataset ID for comparison")
    check_interval_seconds: int = Field(default=3600, ge=60, le=86400, description="Check interval in seconds")
    alert_threshold: float = Field(default=0.05, ge=0.01, le=1.0, description="Drift alert threshold")
    features_to_monitor: Optional[List[str]] = Field(None, description="Specific features to monitor (all if not specified)")
    enable_alerts: bool = Field(default=True, description="Enable drift alerts")
    alert_webhook_url: Optional[str] = Field(None, description="Webhook URL for alerts")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "mdl-abc123",
                "reference_dataset_id": "ds-xyz789",
                "check_interval_seconds": 3600,
                "alert_threshold": 0.05,
                "features_to_monitor": ["age", "income", "category"],
                "enable_alerts": True
            }
        }
    }


class DriftMonitoringResponse(BaseModel):
    """Drift monitoring response schema."""
    task_id: str = Field(..., description="Monitoring task ID")
    model_id: str = Field(..., description="Model ID being monitored")
    status: str = Field(..., description="Monitoring status")
    reference_dataset_id: str = Field(..., description="Reference dataset ID")
    check_interval_seconds: int = Field(..., description="Check interval")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Start timestamp")
    next_check_at: Optional[datetime] = Field(None, description="Next scheduled check")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task-monitor-001",
                "model_id": "mdl-abc123",
                "status": "running",
                "reference_dataset_id": "ds-xyz789",
                "check_interval_seconds": 3600,
                "started_at": "2025-01-01T00:00:00Z",
                "next_check_at": "2025-01-01T01:00:00Z"
            }
        }
    }


class DriftMetrics(BaseModel):
    """Drift metrics for a feature or model."""
    feature_name: Optional[str] = Field(None, description="Feature name (null for overall)")
    drift_score: float = Field(..., ge=0, le=1, description="Drift score (0-1)")
    p_value: Optional[float] = Field(None, ge=0, le=1, description="Statistical p-value")
    test_statistic: Optional[float] = Field(None, description="Test statistic value")
    test_type: str = Field(default="ks_test", description="Statistical test used")
    is_drifted: bool = Field(..., description="Whether drift is detected")
    baseline_stats: Optional[Dict[str, float]] = Field(None, description="Baseline statistics")
    current_stats: Optional[Dict[str, float]] = Field(None, description="Current statistics")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "feature_name": "age",
                "drift_score": 0.15,
                "p_value": 0.02,
                "test_statistic": 0.23,
                "test_type": "ks_test",
                "is_drifted": True,
                "baseline_stats": {"mean": 35.2, "std": 10.5},
                "current_stats": {"mean": 42.1, "std": 12.3}
            }
        }
    }


class DriftAlert(BaseModel):
    """Drift alert schema."""
    alert_id: str = Field(..., description="Alert ID")
    model_id: str = Field(..., description="Model ID")
    drift_type: DriftType = Field(..., description="Type of drift detected")
    severity: DriftSeverity = Field(..., description="Alert severity")
    feature_name: Optional[str] = Field(None, description="Affected feature")
    drift_score: float = Field(..., description="Drift score")
    threshold: float = Field(..., description="Alert threshold")
    message: str = Field(..., description="Alert message")
    metrics: DriftMetrics = Field(..., description="Drift metrics")
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "alert_id": "alert-001",
                "model_id": "mdl-abc123",
                "drift_type": "data_drift",
                "severity": "high",
                "feature_name": "age",
                "drift_score": 0.15,
                "threshold": 0.05,
                "message": "Significant drift detected in feature 'age'",
                "detected_at": "2025-01-01T12:00:00Z",
                "acknowledged": False
            }
        }
    }


class DriftReportResponse(BaseModel):
    """Drift report response schema."""
    model_id: str = Field(..., description="Model ID")
    report_id: str = Field(..., description="Report ID")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation timestamp")
    reference_period: Dict[str, str] = Field(..., description="Reference data period")
    current_period: Dict[str, str] = Field(..., description="Current data period")
    overall_drift_score: float = Field(..., description="Overall drift score")
    is_drifted: bool = Field(..., description="Whether overall drift is detected")
    feature_metrics: List[DriftMetrics] = Field(..., description="Per-feature drift metrics")
    alerts: List[DriftAlert] = Field(default_factory=list, description="Active alerts")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
