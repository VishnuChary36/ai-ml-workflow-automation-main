"""
Input Validation and Sanitization

Security-focused input validation for ML models.
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)


class DataType(str, Enum):
    """Supported data types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class FieldSchema:
    """Schema definition for a field."""
    name: str
    data_type: DataType
    required: bool = True
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None


class InputValidator:
    """
    Validates and sanitizes input data.
    
    Features:
    - Type validation
    - Range checking
    - Pattern matching
    - SQL injection prevention
    - XSS prevention
    """
    
    # Patterns for detecting malicious input
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
    ]
    
    XSS_PATTERNS = [
        r"<script\b[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<\s*iframe",
        r"<\s*object",
    ]
    
    def __init__(self):
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
    
    def validate(
        self,
        data: Dict[str, Any],
        schema: List[FieldSchema],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate input data against schema.
        
        Args:
            data: Input data to validate
            schema: List of field schemas
            strict: If True, reject unknown fields
            
        Returns:
            Validated and sanitized data
            
        Raises:
            ValidationError: If validation fails
        """
        validated = {}
        schema_fields = {f.name: f for f in schema}
        
        # Check required fields
        for field_schema in schema:
            if field_schema.required and field_schema.name not in data:
                raise ValidationError(
                    f"Required field missing: {field_schema.name}",
                    field=field_schema.name
                )
        
        # Validate each field
        for name, value in data.items():
            if name not in schema_fields:
                if strict:
                    raise ValidationError(f"Unknown field: {name}", field=name)
                continue
            
            field_schema = schema_fields[name]
            validated[name] = self._validate_field(name, value, field_schema)
        
        return validated
    
    def _validate_field(
        self,
        name: str,
        value: Any,
        schema: FieldSchema,
    ) -> Any:
        """Validate a single field."""
        # Handle null
        if value is None:
            if schema.nullable:
                return None
            raise ValidationError(f"Field cannot be null: {name}", field=name)
        
        # Type validation
        validated_value = self._validate_type(name, value, schema.data_type)
        
        # String-specific validation
        if schema.data_type == DataType.STRING and isinstance(validated_value, str):
            # Check for malicious content
            self._check_injection(name, validated_value)
            
            # Length validation
            if schema.min_length and len(validated_value) < schema.min_length:
                raise ValidationError(
                    f"Field {name} too short (min: {schema.min_length})",
                    field=name
                )
            if schema.max_length and len(validated_value) > schema.max_length:
                raise ValidationError(
                    f"Field {name} too long (max: {schema.max_length})",
                    field=name
                )
            
            # Pattern validation
            if schema.pattern:
                if not re.match(schema.pattern, validated_value):
                    raise ValidationError(
                        f"Field {name} does not match pattern",
                        field=name
                    )
        
        # Numeric validation
        if schema.data_type in [DataType.INTEGER, DataType.FLOAT]:
            if schema.min_value is not None and validated_value < schema.min_value:
                raise ValidationError(
                    f"Field {name} below minimum ({schema.min_value})",
                    field=name
                )
            if schema.max_value is not None and validated_value > schema.max_value:
                raise ValidationError(
                    f"Field {name} above maximum ({schema.max_value})",
                    field=name
                )
        
        # Allowed values
        if schema.allowed_values and validated_value not in schema.allowed_values:
            raise ValidationError(
                f"Field {name} has invalid value (allowed: {schema.allowed_values})",
                field=name
            )
        
        return validated_value
    
    def _validate_type(
        self,
        name: str,
        value: Any,
        expected_type: DataType,
    ) -> Any:
        """Validate and convert type."""
        if expected_type == DataType.STRING:
            if not isinstance(value, str):
                return str(value)
            return value
        
        elif expected_type == DataType.INTEGER:
            if isinstance(value, bool):
                raise ValidationError(f"Field {name} must be integer", field=name)
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (ValueError, TypeError):
                raise ValidationError(f"Field {name} must be integer", field=name)
        
        elif expected_type == DataType.FLOAT:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValidationError(f"Field {name} must be numeric", field=name)
        
        elif expected_type == DataType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    return True
                if value.lower() in ("false", "0", "no"):
                    return False
            raise ValidationError(f"Field {name} must be boolean", field=name)
        
        elif expected_type == DataType.ARRAY:
            if not isinstance(value, (list, tuple)):
                raise ValidationError(f"Field {name} must be array", field=name)
            return list(value)
        
        elif expected_type == DataType.OBJECT:
            if not isinstance(value, dict):
                raise ValidationError(f"Field {name} must be object", field=name)
            return value
        
        return value
    
    def _check_injection(self, name: str, value: str):
        """Check for SQL injection and XSS attempts."""
        # SQL injection
        for pattern in self.sql_patterns:
            if pattern.search(value):
                raise ValidationError(
                    f"Potential SQL injection detected in field {name}",
                    field=name
                )
        
        # XSS
        for pattern in self.xss_patterns:
            if pattern.search(value):
                raise ValidationError(
                    f"Potential XSS detected in field {name}",
                    field=name
                )
    
    def sanitize_string(self, value: str) -> str:
        """Sanitize a string value."""
        if not isinstance(value, str):
            return str(value)
        
        # HTML escape
        value = (
            value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )
        
        return value


class ModelInputSanitizer:
    """
    Sanitizes input data for ML model predictions.
    
    Features:
    - Numeric value clamping
    - NaN/Inf handling
    - Feature range validation
    - Categorical encoding validation
    """
    
    def __init__(
        self,
        feature_ranges: Optional[Dict[str, tuple]] = None,
        categorical_mappings: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize sanitizer.
        
        Args:
            feature_ranges: Dict of feature name to (min, max) tuples
            categorical_mappings: Dict of feature name to valid categories
        """
        self.feature_ranges = feature_ranges or {}
        self.categorical_mappings = categorical_mappings or {}
    
    def sanitize(
        self,
        data: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Sanitize input data for model prediction.
        
        Args:
            data: Input data
            feature_names: Column names for array input
            
        Returns:
            Sanitized DataFrame
        """
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            if feature_names:
                df = pd.DataFrame(data, columns=feature_names)
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Sanitize each column
        for col in df.columns:
            df[col] = self._sanitize_column(col, df[col])
        
        return df
    
    def _sanitize_column(
        self,
        name: str,
        series: pd.Series,
    ) -> pd.Series:
        """Sanitize a single column."""
        # Check for categorical
        if name in self.categorical_mappings:
            valid_values = self.categorical_mappings[name]
            invalid_mask = ~series.isin(valid_values)
            if invalid_mask.any():
                # Replace with most common valid value or first
                series = series.copy()
                series.loc[invalid_mask] = valid_values[0]
            return series
        
        # Numeric sanitization
        if pd.api.types.is_numeric_dtype(series):
            series = series.copy()
            
            # Replace inf with NaN
            series = series.replace([np.inf, -np.inf], np.nan)
            
            # Clamp to feature range if defined
            if name in self.feature_ranges:
                min_val, max_val = self.feature_ranges[name]
                series = series.clip(lower=min_val, upper=max_val)
            
            # Fill NaN with median or 0
            if series.isna().any():
                fill_value = series.median()
                if pd.isna(fill_value):
                    fill_value = 0
                series = series.fillna(fill_value)
        
        return series
    
    def validate_schema(
        self,
        data: pd.DataFrame,
        expected_columns: List[str],
    ) -> Dict[str, Any]:
        """
        Validate data matches expected schema.
        
        Args:
            data: Input DataFrame
            expected_columns: Expected column names
            
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "missing_columns": [],
            "extra_columns": [],
            "type_mismatches": [],
        }
        
        # Check columns
        current_cols = set(data.columns)
        expected_cols = set(expected_columns)
        
        results["missing_columns"] = list(expected_cols - current_cols)
        results["extra_columns"] = list(current_cols - expected_cols)
        
        if results["missing_columns"]:
            results["valid"] = False
        
        return results
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        reference_stats: Optional[Dict[str, Dict[str, float]]] = None,
        z_threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Detect anomalous input values.
        
        Args:
            data: Input data
            reference_stats: Reference statistics (mean, std per column)
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            Anomaly detection results
        """
        results = {
            "has_anomalies": False,
            "anomaly_count": 0,
            "anomalous_columns": {},
        }
        
        if not reference_stats:
            return results
        
        for col in data.columns:
            if col not in reference_stats:
                continue
            
            stats = reference_stats[col]
            if "mean" not in stats or "std" not in stats:
                continue
            
            mean = stats["mean"]
            std = stats["std"]
            
            if std == 0:
                continue
            
            # Calculate z-scores
            z_scores = np.abs((data[col] - mean) / std)
            anomaly_mask = z_scores > z_threshold
            
            if anomaly_mask.any():
                results["has_anomalies"] = True
                anomaly_count = anomaly_mask.sum()
                results["anomaly_count"] += anomaly_count
                results["anomalous_columns"][col] = {
                    "count": int(anomaly_count),
                    "max_z_score": float(z_scores.max()),
                }
        
        return results
