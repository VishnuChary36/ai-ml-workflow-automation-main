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
    def detect_target_column(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect the best target column for ML using smart heuristics.
        
        Returns dict with 'column', 'reason', and 'confidence'.
        """
        n_rows = len(df)
        candidates = []  # (col_name, score, reason)

        for col_info in profile["columns_info"]:
            col_name = col_info["name"]
            col_type = col_info["type"]
            unique_count = col_info["unique_count"]
            missing_pct = col_info["missing_percent"]
            unique_ratio = unique_count / max(n_rows, 1)
            score = 0
            reasons = []

            # ---- Penalty: skip columns that are clearly NOT targets ----

            # Skip columns with too many missing values (>30%)
            if missing_pct > 30:
                continue

            # Skip ID-like columns: nearly all unique values, numeric or string
            if unique_ratio > 0.9 and unique_count > 20:
                continue

            # Skip columns whose name strongly suggests an identifier
            name_lower = col_name.lower().strip()
            id_keywords = ["id", "index", "uuid", "key", "pk", "url", "link",
                           "path", "file", "image", "photo", "name", "username",
                           "email", "phone", "address", "timestamp", "date", "time"]
            if any(name_lower == kw or name_lower.endswith("_id") or name_lower.startswith("id_")
                   for kw in id_keywords):
                continue
            # Also skip if the name IS exactly one of the id keywords
            if name_lower in id_keywords:
                continue

            # ---- Positive signals ----

            # 1. Common target column names (strongest signal)
            target_keywords = ["target", "label", "class", "outcome", "output",
                               "result", "y", "diagnosis", "status", "category",
                               "species", "survived", "churn", "default", "fraud",
                               "sentiment", "rating", "grade", "type", "group"]
            for kw in target_keywords:
                if kw in name_lower:
                    score += 15
                    reasons.append(f"Column name contains '{kw}'")
                    break

            # 2. Binary columns — excellent classification targets
            if unique_count == 2:
                score += 12
                reasons.append("Binary column (2 unique values) — ideal for classification")

            # 3. Low-cardinality categorical — good classification targets
            elif col_type in ["object", "category"] and 2 < unique_count <= 20:
                score += 10
                reasons.append(f"Categorical with {unique_count} classes — good for classification")

            # 4. Low-cardinality integer — likely encoded classes
            elif col_type in ["int64", "int32"] and 2 <= unique_count <= 20:
                score += 9
                reasons.append(f"Integer column with {unique_count} distinct values — likely class labels")

            # 5. Moderate-cardinality categorical (20-50) — possible but less ideal
            elif col_type in ["object", "category"] and 20 < unique_count <= 50:
                score += 4
                reasons.append(f"Categorical with {unique_count} classes — high-cardinality classification")

            # 6. Continuous numeric — potential regression target
            elif col_type in ["float64", "float32"]:
                score += 3
                reasons.append("Continuous numeric — suitable for regression")

            elif col_type in ["int64", "int32"] and unique_count > 20:
                # Could be regression target (e.g., Rank, count, price)
                score += 2
                reasons.append(f"Numeric with {unique_count} values — possible regression target")

            # 7. Last column bonus — datasets often place target at the end
            if col_name == profile["columns_info"][-1]["name"]:
                score += 3
                reasons.append("Last column in dataset (common target position)")

            # 8. Penalize very high cardinality strings (names, descriptions)
            if col_type in ["object", "category"] and unique_count > 50:
                score -= 5

            # 9. Bonus for columns NOT correlated with row count (not sequential)
            if col_type in ["int64", "float64"] and unique_count < n_rows * 0.5:
                # Check if it looks sequential (like a rank or index)
                try:
                    col_data = df[col_name].dropna()
                    if len(col_data) > 0:
                        sorted_vals = col_data.sort_values().values
                        diffs = np.diff(sorted_vals)
                        if len(diffs) > 0 and np.std(diffs) < 0.01 * np.mean(np.abs(diffs) + 1e-10):
                            # Nearly perfectly sequential — likely a rank/index
                            score -= 8
                            reasons.append("Sequential values detected — likely rank/index")
                except Exception:
                    pass

            if score > 0:
                candidates.append((col_name, score, "; ".join(reasons) if reasons else "General heuristic"))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            confidence = min(best[1] / 20.0, 1.0)  # Normalize to 0-1
            return {
                "column": best[0],
                "reason": best[2],
                "confidence": round(confidence, 2)
            }

        # Fallback to last column
        last_col = profile["columns_info"][-1]["name"] if profile["columns_info"] else None
        return {
            "column": last_col,
            "reason": "Default: last column in dataset",
            "confidence": 0.2
        }
