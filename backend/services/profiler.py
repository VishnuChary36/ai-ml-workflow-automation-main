"""Dataset profiling service."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


class DataProfiler:
    """Profiles datasets and generates statistics."""
    
    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset profile."""
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "columns_info": [],
            "summary": {
                "numeric_columns": 0,
                "categorical_columns": 0,
                "datetime_columns": 0,
                "missing_values_total": int(df.isnull().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        }
        
        for col in df.columns:
            col_info = DataProfiler._profile_column(df, col)
            profile["columns_info"].append(col_info)
            
            # Update summary
            if col_info["type"] in ["int64", "float64"]:
                profile["summary"]["numeric_columns"] += 1
            elif col_info["type"] in ["object", "category"]:
                profile["summary"]["categorical_columns"] += 1
            elif col_info["type"] == "datetime64":
                profile["summary"]["datetime_columns"] += 1
        
        return profile
    
    @staticmethod
    def _profile_column(df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Profile a single column."""
        col_data = df[col]
        dtype = str(col_data.dtype)
        
        info = {
            "name": col,
            "type": dtype,
            "missing_count": int(col_data.isnull().sum()),
            "missing_percent": float(col_data.isnull().sum() / len(df) * 100),
            "unique_count": int(col_data.nunique()),
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            info.update({
                "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                "std": float(col_data.std()) if not col_data.isnull().all() else None,
                "min": float(col_data.min()) if not col_data.isnull().all() else None,
                "max": float(col_data.max()) if not col_data.isnull().all() else None,
                "median": float(col_data.median()) if not col_data.isnull().all() else None,
                "q25": float(col_data.quantile(0.25)) if not col_data.isnull().all() else None,
                "q75": float(col_data.quantile(0.75)) if not col_data.isnull().all() else None,
            })
        
        # Categorical columns
        elif dtype in ["object", "category"]:
            value_counts = col_data.value_counts()
            info.update({
                "top_values": [
                    {"value": str(val), "count": int(count)}
                    for val, count in value_counts.head(10).items()
                ],
                "cardinality": int(col_data.nunique()),
            })
        
        return info
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame, profile: Dict[str, Any]) -> str:
        """Attempt to detect the target column for ML."""
        # Simple heuristics
        candidates = []
        
        for col_info in profile["columns_info"]:
            col_name = col_info["name"]
            
            # Check for common target column names
            if any(keyword in col_name.lower() for keyword in ["target", "label", "class", "outcome", "y"]):
                candidates.append((col_name, 10))
            
            # Binary columns (good for classification)
            if col_info["unique_count"] == 2:
                candidates.append((col_name, 5))
            
            # Low cardinality categorical (good for classification)
            if col_info["type"] in ["object", "category"] and 2 <= col_info["unique_count"] <= 20:
                candidates.append((col_name, 3))
        
        if candidates:
            # Return highest scoring candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Default to last column
        return profile["columns_info"][-1]["name"] if profile["columns_info"] else None
