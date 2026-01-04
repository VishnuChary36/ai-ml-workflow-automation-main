"""
Production Inference API for RandomForestClassifier
Generated at: 2025-12-28T17:16:39.301207

Usage:
    uvicorn inference_api:app --host 0.0.0.0 --port 8080
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Try to import SHAP for explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

app = FastAPI(
    title="RandomForestClassifier Inference API",
    description="Production model serving with explainability",
    version="1.0.0"
)

# Load artifacts at startup
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))

# Load preprocessing config
with open(os.path.join(MODEL_DIR, "preprocessing_pipeline.json"), "r") as f:
    preprocessing_config = json.load(f)

# Load encoders
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))

# Load metadata
with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

# Load SHAP background if available
shap_background = None
shap_explainer = None
if SHAP_AVAILABLE and os.path.exists(os.path.join(MODEL_DIR, "shap_background.csv")):
    try:
        shap_background = pd.read_csv(os.path.join(MODEL_DIR, "shap_background.csv"))
        if hasattr(model, 'feature_importances_'):
            shap_explainer = shap.TreeExplainer(model, shap_background)
        else:
            shap_explainer = shap.KernelExplainer(model.predict, shap_background.values[:50])
    except Exception as e:
        print(f"Warning: Could not initialize SHAP explainer: {e}")

# Feature names
FEATURE_NAMES = ['Name', 'Rank', 'Followers', 'Audience Country', 'Authentic Engagement', 'Engagement Avg.']

# Model type
MODEL_TYPE = "classification"


class PredictionInput(BaseModel):
    """Input data for prediction."""
    data: Dict[str, Any]
    explain: bool = False


class PredictionOutput(BaseModel):
    """Prediction response with optional explanation."""
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    timestamp: str


class BatchPredictionInput(BaseModel):
    """Batch prediction input."""
    data: List[Dict[str, Any]]
    explain: bool = False


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data using saved pipeline."""
    df = pd.DataFrame([data])
    
    # Ensure all features are present
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = None
    
    # Keep only known features
    df = df[FEATURE_NAMES]
    
    # Apply imputation for numeric columns
    for col in preprocessing_config.get("numeric_columns", []):
        if col in df.columns:
            fill_val = preprocessing_config.get("imputers", {}).get(col, {}).get("fill_value", 0)
            df[col] = df[col].fillna(fill_val)
    
    # Apply encoding for categorical columns
    for col in preprocessing_config.get("categorical_columns", []):
        if col in df.columns and col in encoders:
            le = encoders[col]
            # Handle unseen categories
            df[col] = df[col].fillna("_MISSING_").astype(str)
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df


def generate_explanation(X: np.ndarray, prediction: Any) -> Dict[str, Any]:
    """Generate SHAP-based explanation for prediction."""
    if shap_explainer is None:
        return {"status": "unavailable", "reason": "SHAP explainer not initialized"}
    
    try:
        shap_values = shap_explainer.shap_values(X)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            # Get SHAP values for predicted class
            if hasattr(prediction, '__iter__'):
                pred_class = int(prediction[0])
            else:
                pred_class = int(prediction)
            sv = shap_values[pred_class][0] if pred_class < len(shap_values) else shap_values[0][0]
        else:
            sv = shap_values[0]
        
        # Get top features
        feature_importance = {}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(sv):
                feature_importance[name] = round(float(sv[i]), 4)
        
        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        # Generate human-readable explanation
        explanation_parts = []
        for feat, val in sorted_features:
            direction = "↑ increases" if val > 0 else "↓ decreases"
            explanation_parts.append(f"{feat}: {direction} prediction ({val:+.3f})")
        
        return {
            "top_features": dict(sorted_features),
            "human_readable": explanation_parts,
            "status": "success"
        }
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "RandomForestClassifier",
        "model_type": MODEL_TYPE,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/info")
async def model_info():
    """Get model information and metadata."""
    return {
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "target_column": metadata.get("target_column"),
        "n_features": metadata.get("n_features"),
        "metrics": metadata.get("metrics"),
        "features": [f["name"] for f in metadata.get("features", [])],
        "version": metadata.get("version"),
        "created_at": metadata.get("created_at")
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a prediction with optional explanation.
    
    Example request:
    {
        "data": {"feature1": value1, "feature2": value2, ...},
        "explain": true
    }
    """
    try:
        # Preprocess input
        X = preprocess_input(input_data.data)
        X_array = X.values
        
        # Make prediction
        prediction = model.predict(X_array)
        
        # Get probabilities for classification
        confidence = None
        probabilities = None
        if MODEL_TYPE == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_array)[0]
            confidence = float(max(proba))
            
            # Map probabilities to class names if available
            if "target_info" in metadata and "classes" in metadata["target_info"]:
                classes = metadata["target_info"]["classes"]
                probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
            else:
                probabilities = {str(i): float(p) for i, p in enumerate(proba)}
        
        # Generate explanation if requested
        explanation = None
        if input_data.explain:
            explanation = generate_explanation(X_array, prediction)
        
        # Format prediction
        pred_value = prediction[0]
        if MODEL_TYPE == "classification":
            # Try to map back to original class name
            if "target_info" in metadata and "classes" in metadata["target_info"]:
                classes = metadata["target_info"]["classes"]
                if isinstance(pred_value, (int, np.integer)) and pred_value < len(classes):
                    pred_value = classes[pred_value]
        
        return PredictionOutput(
            prediction=pred_value,
            confidence=confidence,
            probabilities=probabilities,
            explanation=explanation,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(input_data: BatchPredictionInput):
    """Make batch predictions."""
    results = []
    for item in input_data.data:
        single_input = PredictionInput(data=item, explain=input_data.explain)
        result = await predict(single_input)
        results.append(result)
    
    return {"predictions": results, "count": len(results)}


@app.get("/features")
async def list_features():
    """List all expected features with their metadata."""
    return {
        "features": metadata.get("features", []),
        "required_features": FEATURE_NAMES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
