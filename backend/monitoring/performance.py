"""
Model Performance Monitoring

Track model performance metrics over time with alerting
and degradation detection.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import deque

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


class PerformanceWindow:
    """Sliding window for tracking metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add(self, prediction: Any, actual: Any, timestamp: Optional[datetime] = None):
        """Add a prediction-actual pair."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.utcnow())
    
    @property
    def count(self) -> int:
        return len(self.predictions)
    
    def get_arrays(self):
        """Get numpy arrays of predictions and actuals."""
        return np.array(list(self.predictions)), np.array(list(self.actuals))
    
    def clear(self):
        """Clear the window."""
        self.predictions.clear()
        self.actuals.clear()
        self.timestamps.clear()


class PerformanceMonitor:
    """
    Monitor model performance in production.
    
    Features:
    - Real-time metric tracking
    - Performance degradation detection
    - Baseline comparison
    - Metric history and trends
    """
    
    def __init__(
        self,
        model_id: str,
        problem_type: str = "classification",
        baseline_metrics: Optional[Dict[str, float]] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize performance monitor.
        
        Args:
            model_id: Model identifier
            problem_type: "classification" or "regression"
            baseline_metrics: Baseline metrics for comparison
            storage_path: Path for storing metrics history
        """
        self.model_id = model_id
        self.problem_type = problem_type
        self.baseline_metrics = baseline_metrics or {}
        
        self.storage_path = Path(
            storage_path or
            os.getenv("METRICS_STORAGE_PATH", "./metrics_history")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Sliding windows
        self.window = PerformanceWindow()
        self.hourly_window = PerformanceWindow(window_size=10000)
        
        # Metric history
        self.history: List[Dict[str, Any]] = []
        self._load_history()
        
        # Degradation thresholds
        self.degradation_thresholds = {
            "accuracy": 0.05,  # 5% drop
            "f1_score": 0.05,
            "precision": 0.05,
            "recall": 0.05,
            "mse": 0.1,  # 10% increase
            "rmse": 0.1,
            "mae": 0.1,
            "r2_score": 0.05,
        }
        
        # Tracking
        self.total_predictions = 0
        self.start_time = datetime.utcnow()
        self.last_metric_update = None
    
    def record_prediction(
        self,
        prediction: Any,
        actual: Optional[Any] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a prediction (and optionally the actual value).
        
        Args:
            prediction: Model prediction
            actual: Actual/ground truth value (for delayed feedback)
            timestamp: Prediction timestamp
        """
        self.total_predictions += 1
        
        if actual is not None:
            self.window.add(prediction, actual, timestamp)
            self.hourly_window.add(prediction, actual, timestamp)
    
    def add_ground_truth(
        self,
        predictions: List[Any],
        actuals: List[Any],
        timestamps: Optional[List[datetime]] = None,
    ):
        """
        Add ground truth for batch of predictions.
        
        Used for delayed feedback scenarios where actuals
        are collected after predictions.
        """
        timestamps = timestamps or [datetime.utcnow()] * len(predictions)
        
        for pred, actual, ts in zip(predictions, actuals, timestamps):
            self.window.add(pred, actual, ts)
            self.hourly_window.add(pred, actual, ts)
    
    def calculate_metrics(
        self,
        window: Optional[PerformanceWindow] = None,
    ) -> Dict[str, float]:
        """
        Calculate current performance metrics.
        
        Args:
            window: Window to use (default: main window)
            
        Returns:
            Dictionary of metrics
        """
        window = window or self.window
        
        if window.count < 10:
            return {}
        
        predictions, actuals = window.get_arrays()
        
        metrics = {}
        
        if self.problem_type == "classification":
            metrics["accuracy"] = accuracy_score(actuals, predictions)
            
            try:
                metrics["precision"] = precision_score(
                    actuals, predictions, average="weighted", zero_division=0
                )
                metrics["recall"] = recall_score(
                    actuals, predictions, average="weighted", zero_division=0
                )
                metrics["f1_score"] = f1_score(
                    actuals, predictions, average="weighted", zero_division=0
                )
            except Exception:
                pass
            
        else:  # regression
            metrics["mse"] = mean_squared_error(actuals, predictions)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(actuals, predictions)
            metrics["r2_score"] = r2_score(actuals, predictions)
        
        metrics["sample_count"] = window.count
        metrics["timestamp"] = datetime.utcnow().isoformat()
        
        self.last_metric_update = datetime.utcnow()
        
        return metrics
    
    def check_degradation(
        self,
        current_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Check for performance degradation vs baseline.
        
        Args:
            current_metrics: Current metrics (calculated if not provided)
            
        Returns:
            Degradation report
        """
        if not self.baseline_metrics:
            return {"error": "No baseline metrics set"}
        
        current = current_metrics or self.calculate_metrics()
        
        if not current:
            return {"error": "Insufficient data for metrics"}
        
        report = {
            "degradation_detected": False,
            "degraded_metrics": [],
            "details": {},
        }
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric not in current:
                continue
            
            current_value = current[metric]
            threshold = self.degradation_thresholds.get(metric, 0.1)
            
            # For metrics where higher is better
            if metric in ["accuracy", "precision", "recall", "f1_score", "r2_score"]:
                change = baseline_value - current_value
                is_degraded = change > (baseline_value * threshold)
            # For metrics where lower is better
            else:
                change = current_value - baseline_value
                is_degraded = change > (baseline_value * threshold)
            
            report["details"][metric] = {
                "baseline": baseline_value,
                "current": current_value,
                "change": change,
                "change_pct": (change / baseline_value) * 100 if baseline_value != 0 else 0,
                "threshold_pct": threshold * 100,
                "degraded": is_degraded,
            }
            
            if is_degraded:
                report["degradation_detected"] = True
                report["degraded_metrics"].append(metric)
        
        return report
    
    def get_trend(
        self,
        metric: str,
        periods: int = 10,
    ) -> Dict[str, Any]:
        """
        Get trend for a specific metric.
        
        Args:
            metric: Metric name
            periods: Number of historical periods
            
        Returns:
            Trend analysis
        """
        if len(self.history) < 2:
            return {"trend": "unknown", "insufficient_data": True}
        
        recent = self.history[-periods:]
        values = [h.get("metrics", {}).get(metric) for h in recent if h.get("metrics", {}).get(metric) is not None]
        
        if len(values) < 2:
            return {"trend": "unknown", "insufficient_data": True}
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "improving" if metric in ["accuracy", "precision", "recall", "f1_score", "r2_score"] else "degrading"
        else:
            trend = "degrading" if metric in ["accuracy", "precision", "recall", "f1_score", "r2_score"] else "improving"
        
        return {
            "metric": metric,
            "trend": trend,
            "slope": slope,
            "values": values,
            "current": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
        }
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Take a performance snapshot and save to history.
        
        Returns:
            Snapshot data
        """
        metrics = self.calculate_metrics()
        degradation = self.check_degradation(metrics)
        
        snapshot = {
            "model_id": self.model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "degradation": degradation,
            "total_predictions": self.total_predictions,
            "window_size": self.window.count,
        }
        
        self.history.append(snapshot)
        self._save_history()
        
        return snapshot
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get monitoring summary.
        
        Returns:
            Summary of monitoring state
        """
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        current_metrics = self.calculate_metrics()
        degradation = self.check_degradation(current_metrics)
        
        summary = {
            "model_id": self.model_id,
            "problem_type": self.problem_type,
            "uptime_seconds": uptime,
            "total_predictions": self.total_predictions,
            "predictions_rate": self.total_predictions / uptime if uptime > 0 else 0,
            "current_metrics": current_metrics,
            "baseline_metrics": self.baseline_metrics,
            "degradation": degradation,
            "window_size": self.window.count,
            "history_length": len(self.history),
        }
        
        # Add trends for key metrics
        summary["trends"] = {}
        key_metrics = ["accuracy", "f1_score"] if self.problem_type == "classification" else ["rmse", "r2_score"]
        for metric in key_metrics:
            if metric in current_metrics:
                summary["trends"][metric] = self.get_trend(metric)
        
        return summary
    
    def reset_window(self):
        """Reset the sliding window."""
        self.window.clear()
    
    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics."""
        self.baseline_metrics = metrics
    
    def _save_history(self):
        """Save history to disk."""
        history_file = self.storage_path / f"{self.model_id}_history.json"
        
        # Keep last 1000 entries
        history_to_save = self.history[-1000:]
        
        with open(history_file, "w") as f:
            json.dump(history_to_save, f, indent=2, default=str)
    
    def _load_history(self):
        """Load history from disk."""
        history_file = self.storage_path / f"{self.model_id}_history.json"
        
        if history_file.exists():
            with open(history_file) as f:
                self.history = json.load(f)
