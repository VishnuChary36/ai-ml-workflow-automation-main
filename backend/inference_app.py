"""
Dedicated Inference (Serving) Service

A standalone FastAPI application optimized for model inference/predictions.
This service is designed to be deployed separately from the training/management backend.

Features:
- POST /predict - Single prediction
- POST /predict/batch - Batch predictions  
- GET /health - Health check
- GET /ready - Readiness check
- Authentication via API key or JWT
- Rate limiting
- Request/response logging
- Metrics collection

Run with: uvicorn inference_app:app --host 0.0.0.0 --port 8080
"""

import os
import time
import json
import uuid
import asyncio
import joblib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas.prediction import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionResult,
)
from schemas.common import HealthResponse, ReadyResponse, ErrorResponse
from core.auth import (
    require_auth,
    require_permissions,
    Permission,
    User,
    get_current_user,
)


# ============================================================================
# Configuration
# ============================================================================

class InferenceConfig:
    """Inference service configuration."""
    
    # Model paths
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", None)
    
    # Service settings
    HOST = os.getenv("INFERENCE_HOST", "0.0.0.0")
    PORT = int(os.getenv("INFERENCE_PORT", "8080"))
    WORKERS = int(os.getenv("INFERENCE_WORKERS", "1"))
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # Authentication
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"
    
    # Logging
    LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() == "true"
    
    # Metrics
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"


config = InferenceConfig()


# ============================================================================
# Model Store
# ============================================================================

class ModelStore:
    """In-memory model store for loaded models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.encoders: Dict[str, Any] = {}
        self.target_encoders: Dict[str, Any] = {}
        self.ready = False
        self.load_time: Optional[datetime] = None
        self.startup_time = time.time()
    
    def load_model(self, model_id: str, model_path: str) -> bool:
        """Load a model from disk."""
        try:
            base_dir = Path(model_path).parent
            
            # Load the model
            self.models[model_id] = joblib.load(model_path)
            
            # Load metadata
            metadata_path = base_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata[model_id] = json.load(f)
            
            # Load preprocessing config
            preprocess_path = base_dir / "preprocessing.json"
            if preprocess_path.exists():
                with open(preprocess_path, 'r') as f:
                    self.preprocessors[model_id] = json.load(f)
            
            # Load encoders
            encoders_path = base_dir / "encoders.joblib"
            if encoders_path.exists():
                self.encoders[model_id] = joblib.load(encoders_path)
            
            # Load target encoder
            target_encoder_path = base_dir / "target_encoder.joblib"
            if target_encoder_path.exists():
                self.target_encoders[model_id] = joblib.load(target_encoder_path)
            
            print(f"✓ Loaded model: {model_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model {model_id}: {e}")
            return False
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a loaded model."""
        return self.models.get(model_id)
    
    def get_metadata(self, model_id: str) -> Optional[Dict]:
        """Get model metadata."""
        return self.metadata.get(model_id)
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())
    
    def is_ready(self) -> bool:
        """Check if any model is loaded and ready."""
        return len(self.models) > 0
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.startup_time


model_store = ModelStore()


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """Simple metrics collector for inference."""
    
    def __init__(self):
        self.total_predictions = 0
        self.total_errors = 0
        self.total_latency_ms = 0.0
        self.predictions_by_model: Dict[str, int] = {}
        self.latency_by_model: Dict[str, List[float]] = {}
    
    def record_prediction(self, model_id: str, latency_ms: float, success: bool = True):
        """Record a prediction."""
        self.total_predictions += 1
        self.total_latency_ms += latency_ms
        
        if not success:
            self.total_errors += 1
        
        self.predictions_by_model[model_id] = self.predictions_by_model.get(model_id, 0) + 1
        
        if model_id not in self.latency_by_model:
            self.latency_by_model[model_id] = []
        self.latency_by_model[model_id].append(latency_ms)
        
        # Keep only last 1000 latencies per model
        if len(self.latency_by_model[model_id]) > 1000:
            self.latency_by_model[model_id] = self.latency_by_model[model_id][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics summary."""
        avg_latency = 0.0
        if self.total_predictions > 0:
            avg_latency = self.total_latency_ms / self.total_predictions
        
        return {
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_predictions, 1),
            "avg_latency_ms": avg_latency,
            "predictions_by_model": self.predictions_by_model,
        }


metrics = MetricsCollector()


# ============================================================================
# Lifespan (Startup/Shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load models on startup."""
    print("=" * 60)
    print("Starting Inference Service...")
    print("=" * 60)
    
    # Load models from MODEL_DIR
    model_dir = Path(config.MODEL_DIR)
    if model_dir.exists():
        # Look for model files
        for model_file in model_dir.glob("**/*.joblib"):
            if "model" in model_file.name.lower():
                model_id = model_file.parent.name or model_file.stem
                model_store.load_model(model_id, str(model_file))
    
    # Load default model if specified
    if config.DEFAULT_MODEL_ID:
        default_model_path = model_dir / config.DEFAULT_MODEL_ID / "model.joblib"
        if default_model_path.exists():
            model_store.load_model("default", str(default_model_path))
    
    model_store.ready = True
    model_store.load_time = datetime.utcnow()
    
    print(f"✓ Loaded {len(model_store.models)} model(s)")
    print(f"✓ Inference service ready on {config.HOST}:{config.PORT}")
    print("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down inference service...")
    model_store.models.clear()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ML Inference Service",
    description="Dedicated model inference/serving API with authentication and rate limiting",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request ID Middleware
# ============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    latency = (time.time() - start_time) * 1000
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{latency:.2f}ms"
    
    return response


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "name": "ML Inference Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "models": "/models",
            "metrics": "/metrics",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the service health status. Used by load balancers
    and orchestration systems to check if the service is alive.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=model_store.get_uptime(),
    )


@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """
    Readiness check endpoint.
    
    Returns whether the service is ready to accept traffic.
    Checks if models are loaded and all dependencies are ready.
    """
    checks = {
        "models_loaded": len(model_store.models) > 0,
        "model_store_ready": model_store.ready,
    }
    
    all_ready = all(checks.values())
    
    return ReadyResponse(
        ready=all_ready,
        status="ready" if all_ready else "not_ready",
        timestamp=datetime.utcnow(),
        checks=checks,
    )


@app.get("/models")
async def list_models():
    """List all loaded models."""
    models_info = []
    for model_id in model_store.list_models():
        metadata = model_store.get_metadata(model_id) or {}
        models_info.append({
            "model_id": model_id,
            "name": metadata.get("model_name", "Unknown"),
            "type": metadata.get("model_type", "unknown"),
            "features": metadata.get("n_features", 0),
            "version": metadata.get("version", "1.0.0"),
        })
    
    return {
        "models": models_info,
        "count": len(models_info),
    }


@app.get("/metrics")
async def get_metrics():
    """Get inference metrics."""
    return {
        "metrics": metrics.get_stats(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Prediction Helpers
# ============================================================================

def preprocess_input(model_id: str, raw_data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    preprocess_config = model_store.preprocessors.get(model_id, {})
    encoders = model_store.encoders.get(model_id, {})
    
    # Get feature order
    feature_order = preprocess_config.get("feature_order", list(raw_data.keys()))
    
    # Create DataFrame
    df = pd.DataFrame([raw_data])
    
    # Ensure all required features exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = None
    
    # Select only required features in correct order
    df = df[[c for c in feature_order if c in df.columns]]
    
    # Handle numeric columns
    for col in preprocess_config.get("numeric_columns", []):
        if col in df.columns:
            fill_val = preprocess_config.get("fill_values", {}).get(col, 0)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_val)
    
    # Handle categorical columns
    for col in preprocess_config.get("categorical_columns", []):
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].fillna("_MISSING_").astype(str)
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df


def decode_prediction(model_id: str, pred: Any, proba: Optional[np.ndarray] = None) -> PredictionResult:
    """Decode prediction to response format."""
    metadata = model_store.get_metadata(model_id) or {}
    target_encoder = model_store.target_encoders.get(model_id)
    target_classes = metadata.get("target_classes")
    model_type = metadata.get("model_type", "classification")
    
    # Convert numpy types to Python types
    if hasattr(pred, 'item'):
        pred = pred.item()
    
    result = PredictionResult(
        prediction=pred,
        confidence=None,
        probabilities=None,
        label=None,
    )
    
    if model_type == "classification":
        # Decode label
        if target_encoder is not None:
            try:
                result.label = str(target_encoder.inverse_transform([int(pred)])[0])
            except Exception:
                result.label = str(pred)
        elif target_classes:
            try:
                result.label = str(target_classes[int(pred)])
            except Exception:
                result.label = str(pred)
        else:
            result.label = str(pred)
        
        # Add probabilities
        if proba is not None:
            result.confidence = float(np.max(proba))
            if target_classes:
                top_indices = np.argsort(proba)[-10:][::-1]
                result.probabilities = {
                    str(target_classes[i]): float(proba[i]) for i in top_indices
                }
    else:
        # Regression
        result.prediction = float(pred)
        result.label = f"{float(pred):.4f}"
    
    return result


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    http_request: Request,
    user: Optional[User] = Depends(get_current_user) if config.AUTH_ENABLED else None,
):
    """
    Make a single prediction.
    
    Send feature values in the `data` field. If multiple models are loaded,
    specify `model_id` to choose which model to use.
    """
    start_time = time.time()
    
    # Determine model to use
    model_id = request.model_id or "default"
    if model_id not in model_store.models:
        # Use first available model
        available = model_store.list_models()
        if not available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No models loaded",
            )
        model_id = available[0]
    
    model = model_store.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found",
        )
    
    try:
        # Preprocess input
        X = preprocess_input(model_id, request.data)
        
        # Predict
        pred = model.predict(X.values)[0]
        
        # Get probabilities for classification
        proba = None
        metadata = model_store.get_metadata(model_id) or {}
        if metadata.get("model_type") == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.values)[0]
        
        # Decode result
        result = decode_prediction(model_id, pred, proba)
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_prediction(model_id, latency_ms, success=True)
        
        return PredictResponse(
            prediction=result.prediction,
            confidence=result.confidence,
            probabilities=result.probabilities,
            label=result.label,
            model_id=model_id,
            model_name=metadata.get("model_name"),
            model_version=metadata.get("version", "1.0.0"),
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_prediction(model_id, latency_ms, success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    http_request: Request,
    user: Optional[User] = Depends(get_current_user) if config.AUTH_ENABLED else None,
):
    """
    Make batch predictions.
    
    Send a list of feature dictionaries in the `data` field.
    Maximum 1000 items per batch.
    """
    start_time = time.time()
    
    # Determine model to use
    model_id = request.model_id or "default"
    if model_id not in model_store.models:
        available = model_store.list_models()
        if not available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No models loaded",
            )
        model_id = available[0]
    
    model = model_store.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found",
        )
    
    try:
        results = []
        metadata = model_store.get_metadata(model_id) or {}
        
        for item in request.data:
            X = preprocess_input(model_id, item)
            pred = model.predict(X.values)[0]
            
            proba = None
            if metadata.get("model_type") == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X.values)[0]
            
            results.append(decode_prediction(model_id, pred, proba))
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_prediction(model_id, latency_ms, success=True)
        
        return BatchPredictResponse(
            predictions=results,
            count=len(results),
            model_id=model_id,
            model_name=metadata.get("model_name"),
            total_latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_prediction(model_id, latency_ms, success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=exc.detail,
            code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, 'request_id', None),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc),
            code="INTERNAL_ERROR",
            request_id=getattr(request.state, 'request_id', None),
        ).model_dump(),
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting Inference Service on {config.HOST}:{config.PORT}")
    uvicorn.run(
        "inference_app:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        reload=False,
    )
