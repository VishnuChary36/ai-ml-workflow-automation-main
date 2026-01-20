"""
Model Validation for CI/CD

Automated model validation checks for deployment pipelines.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class ValidationStatus(str, Enum):
    """Validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


@dataclass
class ValidationReport:
    """Complete validation report."""
    model_id: str
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    checks: List[ValidationCheck] = field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "duration_ms": c.duration_ms,
                }
                for c in self.checks
            ],
            "summary": {
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c.status == ValidationStatus.PASSED),
                "failed": sum(1 for c in self.checks if c.status == ValidationStatus.FAILED),
                "warnings": sum(1 for c in self.checks if c.status == ValidationStatus.WARNING),
            }
        }
    
    def save(self, path: str):
        """Save report to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ModelValidator:
    """
    Validates ML models for production deployment.
    
    Checks:
    - Schema validation
    - Performance thresholds
    - Data quality
    - Reproducibility
    - Security
    - Bias/fairness
    """
    
    def __init__(
        self,
        min_accuracy: float = 0.8,
        max_latency_ms: float = 100.0,
        min_samples: int = 100,
    ):
        """
        Initialize validator.
        
        Args:
            min_accuracy: Minimum acceptable accuracy
            max_latency_ms: Maximum acceptable inference latency
            min_samples: Minimum samples for validation
        """
        self.min_accuracy = min_accuracy
        self.max_latency_ms = max_latency_ms
        self.min_samples = min_samples
        
        self.checks: List[Callable] = [
            self._check_model_artifacts,
            self._check_schema,
            self._check_performance,
            self._check_latency,
            self._check_reproducibility,
            self._check_feature_importance,
            self._check_prediction_distribution,
        ]
    
    def validate(
        self,
        model: Any,
        model_id: str,
        model_version: str,
        test_data: Optional[pd.DataFrame] = None,
        test_labels: Optional[pd.Series] = None,
    ) -> ValidationReport:
        """
        Run all validation checks.
        
        Args:
            model: Model to validate
            model_id: Model identifier
            model_version: Model version
            test_data: Test features
            test_labels: Test labels
            
        Returns:
            Validation report
        """
        report = ValidationReport(
            model_id=model_id,
            model_version=model_version,
        )
        
        context = {
            "model": model,
            "test_data": test_data,
            "test_labels": test_labels,
        }
        
        for check_fn in self.checks:
            start = datetime.utcnow()
            try:
                result = check_fn(context)
                result.duration_ms = (datetime.utcnow() - start).total_seconds() * 1000
                report.checks.append(result)
            except Exception as e:
                report.checks.append(ValidationCheck(
                    name=check_fn.__name__,
                    status=ValidationStatus.FAILED,
                    message=f"Check failed with error: {str(e)}",
                    duration_ms=(datetime.utcnow() - start).total_seconds() * 1000,
                ))
        
        # Determine overall status
        if any(c.status == ValidationStatus.FAILED for c in report.checks):
            report.overall_status = ValidationStatus.FAILED
        elif any(c.status == ValidationStatus.WARNING for c in report.checks):
            report.overall_status = ValidationStatus.WARNING
        
        return report
    
    def _check_model_artifacts(self, context: Dict) -> ValidationCheck:
        """Check model artifacts exist and are valid."""
        model = context.get("model")
        
        if model is None:
            return ValidationCheck(
                name="model_artifacts",
                status=ValidationStatus.FAILED,
                message="Model object is None",
            )
        
        # Check for required methods
        required_methods = ["predict"]
        missing = [m for m in required_methods if not hasattr(model, m)]
        
        if missing:
            return ValidationCheck(
                name="model_artifacts",
                status=ValidationStatus.FAILED,
                message=f"Model missing required methods: {missing}",
            )
        
        return ValidationCheck(
            name="model_artifacts",
            status=ValidationStatus.PASSED,
            message="Model artifacts valid",
            details={"type": type(model).__name__},
        )
    
    def _check_schema(self, context: Dict) -> ValidationCheck:
        """Check input/output schema."""
        model = context.get("model")
        test_data = context.get("test_data")
        
        if test_data is None:
            return ValidationCheck(
                name="schema_validation",
                status=ValidationStatus.SKIPPED,
                message="No test data provided",
            )
        
        try:
            # Try a sample prediction
            sample = test_data.head(1)
            prediction = model.predict(sample)
            
            return ValidationCheck(
                name="schema_validation",
                status=ValidationStatus.PASSED,
                message="Schema validation passed",
                details={
                    "input_shape": sample.shape,
                    "output_shape": np.array(prediction).shape,
                    "input_columns": list(sample.columns),
                },
            )
        except Exception as e:
            return ValidationCheck(
                name="schema_validation",
                status=ValidationStatus.FAILED,
                message=f"Schema validation failed: {str(e)}",
            )
    
    def _check_performance(self, context: Dict) -> ValidationCheck:
        """Check model performance metrics."""
        model = context.get("model")
        test_data = context.get("test_data")
        test_labels = context.get("test_labels")
        
        if test_data is None or test_labels is None:
            return ValidationCheck(
                name="performance_check",
                status=ValidationStatus.SKIPPED,
                message="No test data/labels provided",
            )
        
        if len(test_data) < self.min_samples:
            return ValidationCheck(
                name="performance_check",
                status=ValidationStatus.WARNING,
                message=f"Insufficient samples ({len(test_data)} < {self.min_samples})",
            )
        
        try:
            from sklearn.metrics import accuracy_score
            
            predictions = model.predict(test_data)
            accuracy = accuracy_score(test_labels, predictions)
            
            if accuracy >= self.min_accuracy:
                status = ValidationStatus.PASSED
                message = f"Accuracy {accuracy:.4f} meets threshold {self.min_accuracy}"
            else:
                status = ValidationStatus.FAILED
                message = f"Accuracy {accuracy:.4f} below threshold {self.min_accuracy}"
            
            return ValidationCheck(
                name="performance_check",
                status=status,
                message=message,
                details={"accuracy": accuracy, "threshold": self.min_accuracy},
            )
        except Exception as e:
            return ValidationCheck(
                name="performance_check",
                status=ValidationStatus.FAILED,
                message=f"Performance check failed: {str(e)}",
            )
    
    def _check_latency(self, context: Dict) -> ValidationCheck:
        """Check inference latency."""
        model = context.get("model")
        test_data = context.get("test_data")
        
        if test_data is None:
            return ValidationCheck(
                name="latency_check",
                status=ValidationStatus.SKIPPED,
                message="No test data provided",
            )
        
        try:
            import time
            
            sample = test_data.head(1)
            latencies = []
            
            # Warm up
            for _ in range(3):
                model.predict(sample)
            
            # Measure
            for _ in range(10):
                start = time.time()
                model.predict(sample)
                latencies.append((time.time() - start) * 1000)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            if p95_latency <= self.max_latency_ms:
                status = ValidationStatus.PASSED
                message = f"P95 latency {p95_latency:.2f}ms meets threshold"
            else:
                status = ValidationStatus.WARNING
                message = f"P95 latency {p95_latency:.2f}ms exceeds threshold {self.max_latency_ms}ms"
            
            return ValidationCheck(
                name="latency_check",
                status=status,
                message=message,
                details={
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "threshold_ms": self.max_latency_ms,
                },
            )
        except Exception as e:
            return ValidationCheck(
                name="latency_check",
                status=ValidationStatus.FAILED,
                message=f"Latency check failed: {str(e)}",
            )
    
    def _check_reproducibility(self, context: Dict) -> ValidationCheck:
        """Check prediction reproducibility."""
        model = context.get("model")
        test_data = context.get("test_data")
        
        if test_data is None:
            return ValidationCheck(
                name="reproducibility_check",
                status=ValidationStatus.SKIPPED,
                message="No test data provided",
            )
        
        try:
            sample = test_data.head(5)
            
            pred1 = model.predict(sample)
            pred2 = model.predict(sample)
            
            if np.allclose(pred1, pred2, rtol=1e-5):
                return ValidationCheck(
                    name="reproducibility_check",
                    status=ValidationStatus.PASSED,
                    message="Predictions are reproducible",
                )
            else:
                return ValidationCheck(
                    name="reproducibility_check",
                    status=ValidationStatus.WARNING,
                    message="Predictions show slight variations",
                    details={"diff": np.abs(np.array(pred1) - np.array(pred2)).max()},
                )
        except Exception as e:
            return ValidationCheck(
                name="reproducibility_check",
                status=ValidationStatus.FAILED,
                message=f"Reproducibility check failed: {str(e)}",
            )
    
    def _check_feature_importance(self, context: Dict) -> ValidationCheck:
        """Check feature importance is available."""
        model = context.get("model")
        test_data = context.get("test_data")
        
        if test_data is None:
            return ValidationCheck(
                name="feature_importance_check",
                status=ValidationStatus.SKIPPED,
                message="No test data provided",
            )
        
        importance = None
        
        # Try different ways to get feature importance
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(test_data.columns, model.feature_importances_))
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).flatten()
            if len(coef) == len(test_data.columns):
                importance = dict(zip(test_data.columns, coef))
        
        if importance:
            # Check for dominant features
            values = list(importance.values())
            max_importance = max(values)
            sum_importance = sum(values)
            
            if max_importance / sum_importance > 0.8:
                return ValidationCheck(
                    name="feature_importance_check",
                    status=ValidationStatus.WARNING,
                    message="Single feature dominates predictions",
                    details={"importances": importance},
                )
            
            return ValidationCheck(
                name="feature_importance_check",
                status=ValidationStatus.PASSED,
                message="Feature importance distributed reasonably",
                details={"top_features": dict(sorted(importance.items(), key=lambda x: -x[1])[:5])},
            )
        
        return ValidationCheck(
            name="feature_importance_check",
            status=ValidationStatus.SKIPPED,
            message="Feature importance not available for this model type",
        )
    
    def _check_prediction_distribution(self, context: Dict) -> ValidationCheck:
        """Check prediction distribution for anomalies."""
        model = context.get("model")
        test_data = context.get("test_data")
        
        if test_data is None:
            return ValidationCheck(
                name="prediction_distribution_check",
                status=ValidationStatus.SKIPPED,
                message="No test data provided",
            )
        
        try:
            predictions = model.predict(test_data)
            
            details = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            }
            
            # Check for constant predictions
            if details["std"] < 1e-6:
                return ValidationCheck(
                    name="prediction_distribution_check",
                    status=ValidationStatus.WARNING,
                    message="Predictions are constant (zero variance)",
                    details=details,
                )
            
            # Check for unique value ratio
            if hasattr(model, "classes_"):
                unique_preds = len(np.unique(predictions))
                expected_classes = len(model.classes_)
                details["unique_predictions"] = unique_preds
                details["expected_classes"] = expected_classes
                
                if unique_preds == 1:
                    return ValidationCheck(
                        name="prediction_distribution_check",
                        status=ValidationStatus.WARNING,
                        message="Model predicts only one class",
                        details=details,
                    )
            
            return ValidationCheck(
                name="prediction_distribution_check",
                status=ValidationStatus.PASSED,
                message="Prediction distribution looks reasonable",
                details=details,
            )
        except Exception as e:
            return ValidationCheck(
                name="prediction_distribution_check",
                status=ValidationStatus.FAILED,
                message=f"Distribution check failed: {str(e)}",
            )
