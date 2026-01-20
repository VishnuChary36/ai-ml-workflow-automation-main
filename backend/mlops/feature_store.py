"""
Feature Store

Provides feature versioning, serving, and management for ML pipelines.
"""

import os
import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import numpy as np

from config import settings


class FeatureStore:
    """
    Simple Feature Store for feature management and serving.
    
    Features:
    - Feature set registration
    - Feature versioning
    - Online and offline serving
    - Feature statistics and metadata
    - Point-in-time retrieval
    """
    
    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize feature store.
        
        Args:
            store_path: Path to store features
        """
        self.store_path = Path(
            store_path or 
            os.getenv("FEATURE_STORE_PATH", "./feature_store")
        )
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Feature store index
        self.index_file = self.store_path / "feature_index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load feature store index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {"feature_sets": {}, "features": {}}
    
    def _save_index(self):
        """Save feature store index."""
        with open(self.index_file, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe content."""
        return hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]
    
    def register_feature_set(
        self,
        name: str,
        df: pd.DataFrame,
        entity_columns: List[str],
        timestamp_column: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new feature set.
        
        Args:
            name: Feature set name
            df: DataFrame with features
            entity_columns: Columns that identify entities (e.g., user_id)
            timestamp_column: Optional timestamp column for point-in-time queries
            description: Feature set description
            tags: Feature set tags
            
        Returns:
            Feature set info
        """
        version = self._get_next_version(name)
        feature_set_id = f"{name}-v{version}"
        
        # Create feature set directory
        fs_dir = self.store_path / name / f"v{version}"
        fs_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract feature metadata
        feature_columns = [c for c in df.columns 
                         if c not in entity_columns and c != timestamp_column]
        
        features = {}
        for col in feature_columns:
            features[col] = self._get_feature_metadata(df[col])
        
        # Compute data statistics
        stats = {
            "row_count": len(df),
            "entity_count": df[entity_columns].drop_duplicates().shape[0],
            "feature_count": len(feature_columns),
            "data_hash": self._compute_data_hash(df),
        }
        
        # Save feature data
        data_path = fs_dir / "features.parquet"
        df.to_parquet(data_path, index=False)
        
        # Feature set info
        fs_info = {
            "feature_set_id": feature_set_id,
            "name": name,
            "version": version,
            "description": description,
            "entity_columns": entity_columns,
            "timestamp_column": timestamp_column,
            "feature_columns": feature_columns,
            "features": features,
            "stats": stats,
            "tags": tags or {},
            "path": str(data_path),
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
        }
        
        # Save metadata
        with open(fs_dir / "metadata.json", "w") as f:
            json.dump(fs_info, f, indent=2)
        
        # Update index
        if name not in self._index["feature_sets"]:
            self._index["feature_sets"][name] = {
                "name": name,
                "latest_version": version,
                "versions": [],
                "created_at": datetime.utcnow().isoformat(),
            }
        
        self._index["feature_sets"][name]["latest_version"] = version
        self._index["feature_sets"][name]["versions"].append(version)
        
        # Index individual features
        for col in feature_columns:
            feature_id = f"{name}.{col}"
            self._index["features"][feature_id] = {
                "feature_id": feature_id,
                "feature_set": name,
                "column": col,
                "metadata": features[col],
                "latest_version": version,
            }
        
        self._save_index()
        
        print(f"✓ Registered feature set: {feature_set_id} with {len(feature_columns)} features")
        return fs_info
    
    def _get_next_version(self, name: str) -> int:
        """Get next version number."""
        if name not in self._index["feature_sets"]:
            return 1
        return self._index["feature_sets"][name].get("latest_version", 0) + 1
    
    def _get_feature_metadata(self, series: pd.Series) -> Dict[str, Any]:
        """Get metadata for a feature column."""
        metadata = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percentage": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique()),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            metadata.update({
                "type": "numeric",
                "min": float(series.min()) if not series.empty else None,
                "max": float(series.max()) if not series.empty else None,
                "mean": float(series.mean()) if not series.empty else None,
                "std": float(series.std()) if not series.empty else None,
                "median": float(series.median()) if not series.empty else None,
            })
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == object:
            metadata.update({
                "type": "categorical",
                "categories": series.unique().tolist()[:50],  # Limit categories
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            metadata.update({
                "type": "datetime",
                "min": str(series.min()) if not series.empty else None,
                "max": str(series.max()) if not series.empty else None,
            })
        else:
            metadata["type"] = "other"
        
        return metadata
    
    def get_feature_set(
        self,
        name: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get feature set metadata.
        
        Args:
            name: Feature set name
            version: Specific version (latest if None)
            
        Returns:
            Feature set info or None
        """
        if name not in self._index["feature_sets"]:
            return None
        
        fs_info = self._index["feature_sets"][name]
        version = version or fs_info["latest_version"]
        
        metadata_path = self.store_path / name / f"v{version}" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        
        return None
    
    def load_features(
        self,
        name: str,
        version: Optional[int] = None,
        columns: Optional[List[str]] = None,
        entity_filter: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Load features from store.
        
        Args:
            name: Feature set name
            version: Specific version (latest if None)
            columns: Specific columns to load
            entity_filter: Filter by entity values
            
        Returns:
            DataFrame with features
        """
        fs_info = self.get_feature_set(name, version)
        if not fs_info:
            raise ValueError(f"Feature set not found: {name}")
        
        df = pd.read_parquet(fs_info["path"])
        
        # Apply column filter
        if columns:
            cols = list(fs_info["entity_columns"]) + [
                c for c in columns if c in df.columns
            ]
            df = df[cols]
        
        # Apply entity filter
        if entity_filter:
            for col, value in entity_filter.items():
                if col in df.columns:
                    if isinstance(value, list):
                        df = df[df[col].isin(value)]
                    else:
                        df = df[df[col] == value]
        
        return df
    
    def get_online_features(
        self,
        name: str,
        entity_values: Dict[str, Any],
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get features for online serving.
        
        Args:
            name: Feature set name
            entity_values: Entity column values to look up
            features: Specific features to return
            
        Returns:
            Dictionary of feature values
        """
        fs_info = self.get_feature_set(name)
        if not fs_info:
            raise ValueError(f"Feature set not found: {name}")
        
        df = pd.read_parquet(fs_info["path"])
        
        # Filter by entity values
        mask = pd.Series([True] * len(df))
        for col, value in entity_values.items():
            if col in df.columns:
                mask &= df[col] == value
        
        result = df[mask]
        
        if result.empty:
            return {}
        
        # Get latest row if timestamp exists
        if fs_info.get("timestamp_column"):
            result = result.sort_values(
                fs_info["timestamp_column"], 
                ascending=False
            ).head(1)
        else:
            result = result.head(1)
        
        # Convert to dict
        row = result.iloc[0].to_dict()
        
        if features:
            row = {k: v for k, v in row.items() if k in features}
        
        return row
    
    def get_point_in_time_features(
        self,
        name: str,
        entity_df: pd.DataFrame,
        timestamp_column: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get point-in-time correct features.
        
        Args:
            name: Feature set name
            entity_df: DataFrame with entities and timestamps
            timestamp_column: Column with event timestamps
            features: Specific features to return
            
        Returns:
            DataFrame with features joined
        """
        fs_info = self.get_feature_set(name)
        if not fs_info:
            raise ValueError(f"Feature set not found: {name}")
        
        if not fs_info.get("timestamp_column"):
            # No timestamp - just do regular join
            feature_df = self.load_features(name, columns=features)
            return entity_df.merge(
                feature_df,
                on=fs_info["entity_columns"],
                how="left"
            )
        
        feature_df = pd.read_parquet(fs_info["path"])
        
        # Sort both by timestamp
        entity_df = entity_df.sort_values(timestamp_column)
        feature_df = feature_df.sort_values(fs_info["timestamp_column"])
        
        # Perform asof merge (point-in-time join)
        result = pd.merge_asof(
            entity_df,
            feature_df,
            left_on=timestamp_column,
            right_on=fs_info["timestamp_column"],
            by=fs_info["entity_columns"],
            direction="backward"
        )
        
        return result
    
    def list_feature_sets(self) -> List[Dict[str, Any]]:
        """List all feature sets."""
        return [
            {
                "name": name,
                "latest_version": info["latest_version"],
                "version_count": len(info["versions"]),
                "created_at": info["created_at"],
            }
            for name, info in self._index["feature_sets"].items()
        ]
    
    def list_features(
        self,
        feature_set: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all features.
        
        Args:
            feature_set: Filter by feature set name
            
        Returns:
            List of feature info dicts
        """
        features = list(self._index["features"].values())
        
        if feature_set:
            features = [f for f in features if f["feature_set"] == feature_set]
        
        return features
    
    def get_feature_statistics(
        self,
        name: str,
        feature_name: str,
        version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for a feature.
        
        Args:
            name: Feature set name
            feature_name: Feature column name
            version: Version (latest if None)
            
        Returns:
            Feature statistics
        """
        fs_info = self.get_feature_set(name, version)
        if not fs_info:
            raise ValueError(f"Feature set not found: {name}")
        
        if feature_name not in fs_info["features"]:
            raise ValueError(f"Feature not found: {feature_name}")
        
        # Get base metadata
        stats = fs_info["features"][feature_name].copy()
        
        # Load data for additional stats
        df = pd.read_parquet(fs_info["path"])
        
        if feature_name in df.columns:
            series = df[feature_name]
            
            if pd.api.types.is_numeric_dtype(series):
                stats.update({
                    "percentile_25": float(series.quantile(0.25)),
                    "percentile_75": float(series.quantile(0.75)),
                    "percentile_95": float(series.quantile(0.95)),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                })
        
        return stats
    
    def delete_feature_set(
        self,
        name: str,
        version: Optional[int] = None,
    ) -> bool:
        """
        Delete a feature set or version.
        
        Args:
            name: Feature set name
            version: Specific version (all versions if None)
            
        Returns:
            True if deleted
        """
        if name not in self._index["feature_sets"]:
            return False
        
        fs_info = self._index["feature_sets"][name]
        
        if version:
            # Mark version as inactive
            metadata_path = self.store_path / name / f"v{version}" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                metadata["status"] = "deleted"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
        else:
            # Mark all versions as deleted
            for v in fs_info["versions"]:
                metadata_path = self.store_path / name / f"v{v}" / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    metadata["status"] = "deleted"
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
        
        self._save_index()
        return True


# Global feature store instance
feature_store = FeatureStore()
