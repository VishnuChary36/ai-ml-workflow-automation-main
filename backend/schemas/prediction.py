"""Prediction request/response schemas."""
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


FeatureValue = Union[str, int, float, bool, None]


class PredictRequest(BaseModel):
    """Single prediction request schema."""
    model_id: Optional[str] = Field(None, description="Model ID (optional if using inference service)")
    data: Dict[str, FeatureValue] = Field(..., description="Feature values for prediction")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "mdl-abc123",
                "data": {
                    "age": 35,
                    "income": 75000.0,
                    "category": "A",
                    "is_active": True
                }
            }
        }
    }


class PredictionResult(BaseModel):
    """Individual prediction result."""
    prediction: Any = Field(..., description="Raw prediction value")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities for classification")
    label: Optional[str] = Field(None, description="Human-readable prediction label")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 1,
                "confidence": 0.92,
                "probabilities": {"class_a": 0.92, "class_b": 0.08},
                "label": "class_a"
            }
        }
    }


class PredictResponse(BaseModel):
    """Single prediction response schema."""
    prediction: Any = Field(..., description="Raw prediction value")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    label: Optional[str] = Field(None, description="Human-readable prediction label")
    model_id: Optional[str] = Field(None, description="Model ID used")
    model_name: Optional[str] = Field(None, description="Model name")
    model_version: Optional[str] = Field(None, description="Model version")
    latency_ms: Optional[float] = Field(None, description="Prediction latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 1,
                "confidence": 0.92,
                "probabilities": {"class_a": 0.92, "class_b": 0.08},
                "label": "class_a",
                "model_id": "mdl-abc123",
                "model_name": "RandomForest",
                "model_version": "1.0.0",
                "latency_ms": 15.3,
                "timestamp": "2025-01-01T00:00:00Z"
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """Batch prediction request schema."""
    model_id: Optional[str] = Field(None, description="Model ID")
    data: List[Dict[str, FeatureValue]] = Field(..., min_length=1, max_length=1000, description="List of feature dicts")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "mdl-abc123",
                "data": [
                    {"age": 35, "income": 75000.0, "category": "A"},
                    {"age": 28, "income": 55000.0, "category": "B"}
                ]
            }
        }
    }


class BatchPredictResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: List[PredictionResult] = Field(..., description="List of prediction results")
    count: int = Field(..., description="Number of predictions")
    model_id: Optional[str] = Field(None, description="Model ID used")
    model_name: Optional[str] = Field(None, description="Model name")
    total_latency_ms: Optional[float] = Field(None, description="Total batch latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch prediction timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {"prediction": 1, "confidence": 0.92, "label": "class_a"},
                    {"prediction": 0, "confidence": 0.85, "label": "class_b"}
                ],
                "count": 2,
                "model_id": "mdl-abc123",
                "model_name": "RandomForest",
                "total_latency_ms": 25.6,
                "timestamp": "2025-01-01T00:00:00Z"
            }
        }
    }
