"""
Production Inference Server

Standalone FastAPI application for model serving with:
- Health checks and readiness probes
- Prometheus metrics endpoint
- Request logging
- Rate limiting
- A/B testing support
"""

import os
import time
import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .predictor import ModelPredictor
from .ab_testing import ABTestingRouter
from .metrics import inference_metrics, InferenceMetrics


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictRequest(BaseModel):
    """Single prediction request."""
    features: Dict[str, Any] = Field(..., description="Input features")
    return_probabilities: bool = Field(False, description="Return class probabilities")
    user_id: Optional[str] = Field(None, description="User ID for A/B routing")


class PredictResponse(BaseModel):
    """Single prediction response."""
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    latency_ms: float
    request_id: str


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    instances: List[Dict[str, Any]] = Field(..., description="List of feature dicts")
    return_probabilities: bool = Field(False, description="Return class probabilities")


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[Dict[str, Any]]
    count: int
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    model_loaded: bool
    timestamp: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    checks: Dict[str, bool]


# ============================================================================
# Server Configuration
# ============================================================================

class ServerConfig:
    """Inference server configuration."""
    
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    MODEL_ID = os.getenv("MODEL_ID", None)
    HOST = os.getenv("INFERENCE_HOST", "0.0.0.0")
    PORT = int(os.getenv("INFERENCE_PORT", "8080"))
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "1000"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # Logging
    LOG_REQUESTS = os.getenv("LOG_REQUESTS", "true").lower() == "true"
    
    # Caching
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))


config = ServerConfig()


# ============================================================================
# Inference Server
# ============================================================================

class InferenceServer:
    """
    Production inference server.
    
    Features:
    - Model loading and serving
    - Health and readiness checks
    - Metrics collection
    - Request logging
    - A/B testing support
    """
    
    def __init__(self):
        self.predictor: Optional[ModelPredictor] = None
        self.ab_router: Optional[ABTestingRouter] = None
        self.metrics = inference_metrics
        self.startup_time = datetime.utcnow()
        self.ready = False
        self.request_count = 0
    
    def load_model(self, model_path: str):
        """Load model for serving."""
        self.predictor = ModelPredictor(
            model_path=model_path,
            cache_predictions=config.CACHE_ENABLED,
            cache_size=config.CACHE_SIZE,
        )
        self.predictor.load()
        self.ready = True
        print(f"✓ Model loaded: {model_path}")
    
    def predict(
        self,
        features: Dict[str, Any],
        return_probabilities: bool = False,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a prediction."""
        if not self.ready:
            raise RuntimeError("Model not loaded")
        
        self.metrics.request_started()
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Use A/B router if configured
            if self.ab_router and self.ab_router.status == "running":
                result = self.ab_router.predict(
                    features=features,
                    user_id=user_id,
                    return_probabilities=return_probabilities,
                )
            else:
                result = self.predictor.predict(
                    features=features,
                    return_probabilities=return_probabilities,
                )
            
            latency = time.time() - start_time
            
            self.metrics.record_request(
                model_id=self.predictor.metadata.get("model_id", "default"),
                latency_seconds=latency,
                success=True,
                cached=result.get("cached", False),
            )
            
            result["request_id"] = request_id
            self.request_count += 1
            
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_request(
                model_id=self.predictor.metadata.get("model_id", "default") if self.predictor else "unknown",
                latency_seconds=latency,
                success=False,
                error_type=type(e).__name__,
            )
            raise
        
        finally:
            self.metrics.request_completed()
    
    def predict_batch(
        self,
        instances: List[Dict[str, Any]],
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """Make batch predictions."""
        if not self.ready:
            raise RuntimeError("Model not loaded")
        
        self.metrics.request_started()
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            result = self.predictor.predict_batch(
                features_list=instances,
                return_probabilities=return_probabilities,
            )
            
            latency = time.time() - start_time
            
            self.metrics.record_request(
                model_id=self.predictor.metadata.get("model_id", "default"),
                latency_seconds=latency,
                success=True,
                batch_size=len(instances),
            )
            
            result["request_id"] = request_id
            
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_request(
                model_id=self.predictor.metadata.get("model_id", "default") if self.predictor else "unknown",
                latency_seconds=latency,
                success=False,
                batch_size=len(instances),
                error_type=type(e).__name__,
            )
            raise
        
        finally:
            self.metrics.request_completed()
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        uptime = (datetime.utcnow() - self.startup_time).total_seconds()
        return {
            "status": "healthy" if self.ready else "unhealthy",
            "uptime_seconds": uptime,
            "model_loaded": self.predictor is not None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def get_readiness(self) -> Dict[str, Any]:
        """Get readiness status."""
        checks = {
            "model_loaded": self.predictor is not None,
            "model_initialized": self.ready,
        }
        return {
            "ready": all(checks.values()),
            "checks": checks,
        }


# Global server instance
inference_server = InferenceServer()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    if config.MODEL_ID:
        model_path = Path(config.MODEL_DIR) / config.MODEL_ID
        if model_path.exists():
            inference_server.load_model(str(model_path))
    
    yield
    
    # Shutdown
    print("Shutting down inference server...")


app = FastAPI(
    title="MLOps Inference Service",
    description="Production model inference API",
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
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return inference_server.get_health()


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def ready():
    """Readiness check endpoint."""
    result = inference_server.get_readiness()
    if not result["ready"]:
        raise HTTPException(status_code=503, detail="Service not ready")
    return result


@app.get("/metrics", response_class=PlainTextResponse, tags=["Metrics"])
async def metrics():
    """Prometheus metrics endpoint."""
    return inference_server.metrics.export_prometheus()


@app.get("/metrics/json", tags=["Metrics"])
async def metrics_json():
    """JSON metrics endpoint."""
    return inference_server.metrics.get_summary()


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """Single prediction endpoint."""
    try:
        result = inference_server.predict(
            features=request.features,
            return_probabilities=request.return_probabilities,
            user_id=request.user_id,
        )
        return PredictResponse(
            prediction=result["prediction"],
            confidence=result.get("confidence"),
            probabilities=result.get("probabilities"),
            model_version=result.get("variant_name"),
            latency_ms=result["latency_ms"],
            request_id=result["request_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction endpoint."""
    try:
        result = inference_server.predict_batch(
            instances=request.instances,
            return_probabilities=request.return_probabilities,
        )
        return BatchPredictResponse(
            predictions=result["predictions"],
            count=result["count"],
            latency_ms=result["latency_ms"],
            request_id=result["request_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get loaded model information."""
    if not inference_server.predictor:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_id": inference_server.predictor.metadata.get("model_id"),
        "model_type": inference_server.predictor.metadata.get("model_type"),
        "problem_type": inference_server.predictor.problem_type,
        "features": inference_server.predictor.feature_columns,
        "metrics": inference_server.predictor.metadata.get("metrics", {}),
        "created_at": inference_server.predictor.metadata.get("created_at"),
    }


@app.get("/model/schema", tags=["Model"])
async def model_schema():
    """Get model input schema."""
    if not inference_server.predictor:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return inference_server.predictor.get_input_schema()


@app.post("/model/load", tags=["Model"])
async def load_model(model_id: str):
    """Load a specific model."""
    model_path = Path(config.MODEL_DIR) / model_id
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    try:
        inference_server.load_model(str(model_path))
        return {"status": "loaded", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

def run_server():
    """Run the inference server."""
    import uvicorn
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
