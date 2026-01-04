"""
Production Inference API - RandomForestClassifier
Generated: 2025-12-29T13:19:41.999731

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
    title="RandomForestClassifier Prediction API",
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
FEATURE_ORDER = ['Name', 'Rank', 'Followers', 'Audience Country', 'Authentic Engagement', 'Engagement Avg.']
MODEL_TYPE = "classification"
TARGET_CLASSES = ['Adult content', 'Animals', 'Art/Artists', 'Art/ArtistsBeauty', 'Art/ArtistsCinema & Actors/actresses', 'Art/ArtistsCinema & Actors/actressesMusic', 'Art/ArtistsCinema & Actors/actressesShows', 'Art/ArtistsFashionModeling', 'Art/ArtistsFashionMusic', 'Art/ArtistsShows', 'Beauty', 'BeautyArt/ArtistsCinema & Actors/actresses', 'BeautyCinema & Actors/actresses', 'BeautyCinema & Actors/actressesFashion', 'BeautyCinema & Actors/actressesModeling', 'BeautyFamilyLifestyle', 'BeautyHumor & Fun & HappinessCinema & Actors/actresses', 'BeautyLifestyle', 'BeautyLifestyleFashion', 'BeautyModelingAdult content', 'BeautyModelingCinema & Actors/actresses', 'Business & Careers', 'Business & CareersFinance & Economics', 'Cars & MotorbikesLuxury', 'Cinema & Actors/actresses', 'Cinema & Actors/actressesArt/Artists', 'Cinema & Actors/actressesComics & sketches', 'Cinema & Actors/actressesComics & sketchesShows', 'Cinema & Actors/actressesFamily', 'Cinema & Actors/actressesFashion', 'Cinema & Actors/actressesFitness & Gym', 'Cinema & Actors/actressesHumor & Fun & Happiness', 'Cinema & Actors/actressesHumor & Fun & HappinessModeling', 'Cinema & Actors/actressesLifestyle', 'Cinema & Actors/actressesModeling', 'Cinema & Actors/actressesModelingClothing & Outfits', 'Cinema & Actors/actressesModelingFashion', 'Cinema & Actors/actressesModelingLifestyle', 'Cinema & Actors/actressesMusic', 'Cinema & Actors/actressesMusicShows', 'Cinema & Actors/actressesShows', 'Clothing & OutfitsLifestyle', 'Clothing & OutfitsMusic', 'Computers & Gadgets', 'Computers & GadgetsBusiness & Careers', 'Computers & GadgetsMachinery & Technologies', 'Computers & GadgetsPhotography', 'EducationHumor & Fun & Happiness', 'Family', 'FamilyCinema & Actors/actresses', 'FamilyLifestyle', 'FamilyLifestyleCinema & Actors/actresses', 'FamilyLifestyleShows', 'FamilyMusic', 'FamilyShows', 'FamilySports with a ball', 'Fashion', 'FashionBeauty', 'FashionCinema & Actors/actresses', 'FashionLifestyleBeauty', 'FashionModeling', 'FashionModelingBeauty', 'Finance & Economics', 'Finance & EconomicsBusiness & Careers', 'Finance & EconomicsBusiness & CareersFamily', 'Fitness & Gym', 'Fitness & GymLifestyle', 'Fitness & GymShopping & RetailClothing & Outfits', 'Food & Cooking', 'Humor & Fun & Happiness', 'Humor & Fun & HappinessCinema & Actors/actresses', 'Humor & Fun & HappinessClothing & Outfits', 'Humor & Fun & HappinessLifestyleBeauty', 'Humor & Fun & HappinessShowsCinema & Actors/actresses', 'Lifestyle', 'LifestyleArt/Artists', 'LifestyleBeauty', 'LifestyleCinema & Actors/actresses', 'LifestyleCinema & Actors/actressesModeling', 'LifestyleCinema & Actors/actressesMusic', 'LifestyleFamily', 'LifestyleFashion', 'LifestyleFashionClothing & Outfits', 'LifestyleModeling', 'LifestyleModelingCinema & Actors/actresses', 'LifestyleModelingFashion', 'LifestyleMusic', 'LifestyleMusicArt/Artists', 'LifestyleMusicCinema & Actors/actresses', 'LifestyleMusicSports with a ball', 'LifestylePhotography', 'LifestyleShows', 'LifestyleShowsComputers & Gadgets', 'LifestyleSports with a ball', 'Literature & JournalismBusiness & CareersFinance & Economics', 'Literature & JournalismCinema & Actors/actresses', 'Literature & JournalismFashion', 'Literature & JournalismShows', 'Literature & JournalismTrainers & Coaches', 'Machinery & TechnologiesComputers & Gadgets', 'Management & MarketingMusic', 'Modeling', 'ModelingBeauty', 'ModelingCinema & Actors/actresses', 'ModelingFamily', 'ModelingFashion', 'ModelingFashionLifestyle', 'ModelingLifestyle', 'ModelingLifestyleBeauty', 'ModelingLifestyleFashion', 'ModelingMusic', 'Music', 'MusicArt/Artists', 'MusicArt/ArtistsCinema & Actors/actresses', 'MusicBeauty', 'MusicCinema & Actors/actresses', 'MusicCinema & Actors/actressesFashion', 'MusicCinema & Actors/actressesLifestyle', 'MusicCinema & Actors/actressesModeling', 'MusicClothing & Outfits', 'MusicFamily', 'MusicFashion', 'MusicKids & Toys', 'MusicLifestyle', 'MusicModeling', 'MusicModelingArt/Artists', 'MusicShows', 'Nature & landscapesSciencePhotography', 'Photography', 'PhotographyFashion', 'PhotographyTravel', 'Racing Sports', 'ScienceMachinery & Technologies', 'SciencePhotography', 'Shows', 'ShowsAdult content', 'ShowsBeautyFamily', 'ShowsCinema & Actors/actresses', 'ShowsCinema & Actors/actressesLifestyle', 'ShowsCinema & Actors/actressesModeling', 'ShowsFamily', 'ShowsHumor & Fun & Happiness', 'ShowsLifestyle', 'ShowsModeling', 'ShowsMusic', 'Sports with a ball', 'Sports with a ballClothing & Outfits', 'Sports with a ballFamily', 'Sports with a ballLifestyle', 'Sports with a ballLifestyleKids & Toys', 'Sports with a ballLiterature & Journalism', 'Sports with a ballShows', 'TravelCinema & Actors/actresses', '_MISSING_']


class PredictRequest(BaseModel):
    """Single prediction request."""
    data: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {"Name": "value", "Rank": "value", "Followers": "value"}
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
        "name": "RandomForestClassifier Prediction API",
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
        "model": "RandomForestClassifier",
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
