"""
Production Inference API - XGBoostClassifier
Generated: 2026-04-24T06:37:02.127882

Run with: uvicorn app:app --host 0.0.0.0 --port 8080
Test: curl http://localhost:8080/health
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="XGBoostClassifier Prediction API",
    description="Production ML model serving API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
print("Model loaded successfully!")

# Load preprocessing config
with open(os.path.join(BASE_DIR, "preprocessing.json"), "r") as f:
    preprocess_config = json.load(f)

with open(os.path.join(BASE_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

# Load encoders if available
encoders = {}
encoders_path = os.path.join(BASE_DIR, "encoders.joblib")
if os.path.exists(encoders_path):
    encoders = joblib.load(encoders_path)
    print(f"Loaded {len(encoders)} encoders")

# Load target encoder if available
target_encoder = None
target_encoder_path = os.path.join(BASE_DIR, "target_encoder.joblib")
if os.path.exists(target_encoder_path):
    target_encoder = joblib.load(target_encoder_path)
    print("Target encoder loaded")

# Constants
FEATURE_ORDER = ['Transaction ID', 'Customer ID', 'Item', 'Price Per Unit', 'Quantity', 'Total Spent', 'Transaction Date', 'Payment Method_Cash', 'Payment Method_Credit Card', 'Payment Method_Digital Wallet', 'Location_In-store', 'Location_Online', 'Discount Applied_False', 'Discount Applied_True']
MODEL_TYPE = "classification"
TARGET_CLASSES = ['Beverages', 'Butchers', 'Computers and electric accessories', 'Electric household essentials', 'Food', 'Furniture', 'Milk Products', 'Patisserie']


class PredictRequest(BaseModel):
    """Single prediction request."""
    data: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {"Transaction ID": "value", "Customer ID": "value", "Item": "value"}
            }
        }


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    data: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    """Prediction response."""
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    label: Optional[str] = None


def preprocess_input(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    # Create DataFrame with correct column order
    df = pd.DataFrame([raw_data])
    
    # Ensure all required features exist
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = None
    
    # Select only required features in correct order
    df = df[FEATURE_ORDER]
    
    # Handle numeric columns - fill missing
    for col in preprocess_config.get("numeric_columns", []):
        if col in df.columns:
            fill_val = preprocess_config.get("fill_values", {}).get(col, 0)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_val)
    
    # Handle categorical columns - encode
    for col in preprocess_config.get("categorical_columns", []):
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].fillna("_MISSING_").astype(str)
            # Handle unseen categories
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df


def decode_prediction(pred, proba=None):
    """Decode prediction to original label if applicable."""
    # Ensure prediction is Python native type
    if hasattr(pred, 'item'):
        pred = pred.item()  # Convert numpy to Python
    
    result = {
        "prediction": pred,
        "confidence": None,
        "probabilities": None,
        "label": None
    }
    
    if MODEL_TYPE == "classification":
        # Decode to original class label
        if target_encoder is not None:
            try:
                label = target_encoder.inverse_transform([int(pred)])[0]
                result["label"] = str(label)
            except Exception as e:
                result["label"] = str(pred)
        elif TARGET_CLASSES is not None:
            try:
                result["label"] = str(TARGET_CLASSES[int(pred)])
            except Exception as e:
                result["label"] = str(pred)
        else:
            result["label"] = str(pred)
        
        # Add probabilities
        if proba is not None:
            result["confidence"] = float(np.max(proba))
            if TARGET_CLASSES:
                # Limit to top 10 classes for efficiency
                top_indices = np.argsort(proba)[-10:][::-1]
                result["probabilities"] = {
                    str(TARGET_CLASSES[i]): float(proba[i]) for i in top_indices
                }
            else:
                result["probabilities"] = {
                    str(i): float(p) for i, p in enumerate(proba[:10])
                }
    else:
        # Regression
        result["prediction"] = float(pred)
        result["label"] = f"{float(pred):.4f}"
    
    return result


@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": "XGBoostClassifier Prediction API",
        "version": "1.0.0",
        "model_type": MODEL_TYPE,
        "status": "running",
        "endpoints": ["/health", "/predict", "/predict/batch", "/info"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "XGBoostClassifier",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/info")
async def info():
    """Get model information."""
    return {
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "target": metadata.get("target_column"),
        "n_features": metadata.get("n_features"),
        "features": FEATURE_ORDER,
        "target_classes": TARGET_CLASSES,
        "metrics": metadata.get("metrics"),
        "created_at": metadata.get("created_at")
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make a single prediction.
    
    Send your feature values in the `data` field.
    """
    try:
        # Preprocess
        X = preprocess_input(request.data)
        
        # Predict
        pred = model.predict(X.values)[0]
        
        # Get probabilities if classification
        proba = None
        if MODEL_TYPE == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.values)[0]
        
        # Decode result
        result = decode_prediction(pred, proba)
        
        return PredictResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """Make batch predictions."""
    try:
        results = []
        for item in request.data:
            X = preprocess_input(item)
            pred = model.predict(X.values)[0]
            
            proba = None
            if MODEL_TYPE == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X.values)[0]
            
            results.append(decode_prediction(pred, proba))
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting prediction server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
