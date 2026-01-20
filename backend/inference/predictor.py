"""
Model Predictor

Handles model loading, preprocessing, and prediction with caching.
"""

import os
import json
import time
import uuid
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict

import numpy as np
import pandas as pd
import joblib

from config import settings


class LRUCache:
    """Simple LRU cache for model predictions."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[key] = value
    
    def clear(self):
        with self._lock:
            self.cache.clear()


class ModelPredictor:
    """
    Production model predictor with preprocessing and caching.
    
    Features:
    - Model loading with lazy initialization
    - Preprocessing pipeline preservation
    - Prediction caching
    - Batch prediction support
    - Input validation
    - Latency tracking
    """
    
    def __init__(
        self,
        model_path: str,
        cache_predictions: bool = True,
        cache_size: int = 1000,
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model directory
            cache_predictions: Whether to cache predictions
            cache_size: Maximum cache size
        """
        self.model_path = Path(model_path)
        self.cache_predictions = cache_predictions
        
        # Lazy loading
        self._model = None
        self._metadata: Optional[Dict] = None
        self._encoders: Dict = {}
        self._target_encoder = None
        self._scalers: Dict = {}
        self._loaded = False
        self._load_lock = threading.Lock()
        
        # Caching
        self._cache = LRUCache(cache_size) if cache_predictions else None
        
        # Metrics
        self.prediction_count = 0
        self.cache_hits = 0
        self.total_latency_ms = 0.0
    
    def load(self):
        """Load model and artifacts."""
        with self._load_lock:
            if self._loaded:
                return
            
            # Load model
            model_file = self.model_path / "model.joblib"
            if not model_file.exists():
                raise FileNotFoundError(f"Model not found: {model_file}")
            
            self._model = joblib.load(model_file)
            
            # Load metadata
            metadata_file = self.model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    self._metadata = json.load(f)
            
            # Load encoders
            encoders_file = self.model_path / "encoders.joblib"
            if encoders_file.exists():
                self._encoders = joblib.load(encoders_file)
            
            # Load target encoder
            target_encoder_file = self.model_path / "target_encoder.joblib"
            if target_encoder_file.exists():
                self._target_encoder = joblib.load(target_encoder_file)
            
            # Load scalers
            scalers_file = self.model_path / "scalers.joblib"
            if scalers_file.exists():
                self._scalers = joblib.load(scalers_file)
            
            self._loaded = True
            print(f"✓ Loaded model from {self.model_path}")
    
    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if not self._loaded:
            self.load()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        self._ensure_loaded()
        return self._metadata or {}
    
    @property
    def feature_columns(self) -> List[str]:
        """Get expected feature columns."""
        return self.metadata.get("feature_columns", [])
    
    @property
    def problem_type(self) -> str:
        """Get problem type (classification/regression)."""
        return self.metadata.get("problem_type", "classification")
    
    def _compute_cache_key(self, features: Dict[str, Any]) -> str:
        """Compute cache key for features."""
        sorted_items = sorted(features.items())
        key_str = json.dumps(sorted_items, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to input data."""
        df = df.copy()
        
        # Apply encoders
        for col, encoder in self._encoders.items():
            if col in df.columns:
                # Handle unseen categories
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Unseen category - use most frequent
                    df[col] = 0
        
        # Apply scalers
        for name, scaler in self._scalers.items():
            if name == "features":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    try:
                        df[numeric_cols] = scaler.transform(df[numeric_cols])
                    except Exception:
                        pass
        
        # Ensure column order matches training
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.feature_columns]
        
        # Fill NaN
        df = df.fillna(0)
        
        return df
    
    def _decode_prediction(self, prediction: Any) -> Any:
        """Decode prediction using target encoder."""
        if self._target_encoder is not None:
            try:
                if isinstance(prediction, (list, np.ndarray)):
                    return self._target_encoder.inverse_transform(prediction).tolist()
                else:
                    return self._target_encoder.inverse_transform([prediction])[0]
            except Exception:
                pass
        return prediction
    
    def predict(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
        decode_labels: bool = True,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            features: Input features as dict or DataFrame
            decode_labels: Decode predicted labels
            return_probabilities: Include class probabilities
            
        Returns:
            Prediction result dict
        """
        self._ensure_loaded()
        start_time = time.time()
        
        # Check cache
        if self.cache_predictions and isinstance(features, dict):
            cache_key = self._compute_cache_key(features)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                cached["cached"] = True
                return cached
        
        # Convert to DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.head(1)
        
        # Preprocess
        df_processed = self._preprocess(df)
        
        # Predict
        prediction = self._model.predict(df_processed)[0]
        
        result = {
            "prediction": self._decode_prediction(prediction) if decode_labels else prediction,
            "raw_prediction": int(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
        }
        
        # Add probabilities if available
        if return_probabilities and hasattr(self._model, "predict_proba"):
            try:
                probas = self._model.predict_proba(df_processed)[0]
                result["probabilities"] = {
                    str(i): float(p) for i, p in enumerate(probas)
                }
                result["confidence"] = float(max(probas))
            except Exception:
                pass
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        result["cached"] = False
        
        # Update metrics
        self.prediction_count += 1
        self.total_latency_ms += latency_ms
        
        # Cache result
        if self.cache_predictions and isinstance(features, dict):
            self._cache.set(cache_key, result.copy())
        
        return result
    
    def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        decode_labels: bool = True,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature dicts
            decode_labels: Decode predicted labels
            return_probabilities: Include class probabilities
            
        Returns:
            Batch prediction result
        """
        self._ensure_loaded()
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Preprocess
        df_processed = self._preprocess(df)
        
        # Predict
        predictions = self._model.predict(df_processed)
        
        results = {
            "predictions": [],
            "count": len(predictions),
        }
        
        for i, pred in enumerate(predictions):
            result = {
                "index": i,
                "prediction": self._decode_prediction(pred) if decode_labels else pred,
                "raw_prediction": int(pred) if isinstance(pred, (np.integer, np.floating)) else pred,
            }
            
            # Add probabilities if available
            if return_probabilities and hasattr(self._model, "predict_proba"):
                try:
                    probas = self._model.predict_proba(df_processed)[i]
                    result["probabilities"] = {
                        str(j): float(p) for j, p in enumerate(probas)
                    }
                    result["confidence"] = float(max(probas))
                except Exception:
                    pass
            
            results["predictions"].append(result)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        results["latency_ms"] = round(latency_ms, 2)
        results["avg_latency_ms"] = round(latency_ms / len(predictions), 2)
        
        # Update metrics
        self.prediction_count += len(predictions)
        self.total_latency_ms += latency_ms
        
        return results
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get expected input schema."""
        self._ensure_loaded()
        
        schema = {
            "type": "object",
            "properties": {},
            "required": self.feature_columns,
        }
        
        # Build properties from metadata
        for col in self.feature_columns:
            schema["properties"][col] = {
                "type": "number",
                "description": f"Feature: {col}"
            }
        
        return schema
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get predictor metrics."""
        return {
            "prediction_count": self.prediction_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.prediction_count, 1),
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / max(self.prediction_count, 1),
        }
    
    def clear_cache(self):
        """Clear prediction cache."""
        if self._cache:
            self._cache.clear()
