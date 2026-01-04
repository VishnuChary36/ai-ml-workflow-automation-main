"""Preprocessing service with live console logging."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime
import time


class PreprocessingService:
    """Executes preprocessing steps with detailed logging."""
    
    def __init__(self, task_id: str, emit_log_func):
        self.task_id = task_id
        self.emit_log = emit_log_func
        self.scalers = {}
        self.encoders = {}
    
    async def execute_step(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        """Execute a single preprocessing step."""
        step_type = step["type"]
        start_time = time.time()
        
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Starting step: {step['id']} ({step_type})",
            source=f"preprocess.{step_type}",
            meta={"step_id": step["id"], "step_type": step_type}
        )
        
        if step_type == "drop_column":
            df = await self._drop_column(df, step)
        elif step_type == "impute":
            df = await self._impute(df, step)
        elif step_type == "drop_duplicates":
            df = await self._drop_duplicates(df, step)
        elif step_type == "encode":
            df = await self._encode(df, step)
        elif step_type == "scale":
            df = await self._scale(df, step)
        else:
            await self.emit_log(
                self.task_id,
                "WARN",
                f"Unknown step type: {step_type}",
                source=f"preprocess.{step_type}"
            )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Completed step: {step['id']} in {elapsed_ms}ms",
            source=f"preprocess.{step_type}",
            meta={"step_id": step["id"], "elapsed_ms": elapsed_ms}
        )
        
        return df
    
    async def _drop_column(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        """Drop specified columns."""
        columns = step["target_columns"]
        
        for col in columns:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df) * 100
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"Analyzing column '{col}' -> {missing_pct:.1f}% missing",
                    source="preprocess.drop_column",
                    meta={"column": col, "missing_percent": missing_pct}
                )
                
                df = df.drop(columns=[col])
                
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"Dropped column '{col}'",
                    source="preprocess.drop_column",
                    meta={"column": col}
                )
        
        return df
    
    async def _impute(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        """Impute missing values."""
        columns = step["target_columns"]
        strategy = step["params"].get("strategy", "mean")
        
        for col in columns:
            if col not in df.columns:
                continue
            
            n_nulls = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            
            await self.emit_log(
                self.task_id,
                "INFO",
                f"Analyzing column '{col}' -> dtype={dtype}, nulls={n_nulls} ({n_nulls/len(df)*100:.1f}%)",
                source="preprocess.impute",
                meta={"column": col, "dtype": dtype, "nulls": n_nulls}
            )
            
            if n_nulls > 0:
                if strategy in ["mean", "median"]:
                    fill_value = df[col].mean() if strategy == "mean" else df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    
                    await self.emit_log(
                        self.task_id,
                        "INFO",
                        f"Replaced {n_nulls} nulls in column '{col}' with {strategy} {fill_value:.3f}",
                        source="preprocess.impute",
                        meta={"column": col, "rows_affected": int(n_nulls), "strategy": strategy, "fill_value": float(fill_value)}
                    )
                elif strategy == "most_frequent":
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else None
                    if fill_value is not None:
                        df[col].fillna(fill_value, inplace=True)
                        
                        await self.emit_log(
                            self.task_id,
                            "INFO",
                            f"Replaced {n_nulls} nulls in column '{col}' with most frequent value '{fill_value}'",
                            source="preprocess.impute",
                            meta={"column": col, "rows_affected": int(n_nulls), "strategy": strategy}
                        )
        
        return df
    
    async def _drop_duplicates(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        
        await self.emit_log(
            self.task_id,
            "INFO",
            "Scanning for duplicates...",
            source="preprocess.drop_duplicates",
            meta={"initial_rows": initial_rows}
        )
        
        df = df.drop_duplicates()
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Removed {removed} duplicate rows (kept {final_rows} unique rows)",
            source="preprocess.drop_duplicates",
            meta={"removed": removed, "final_rows": final_rows}
        )
        
        return df
    
    async def _encode(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical variables."""
        columns = step["target_columns"]
        method = step["params"].get("method", "onehot")
        
        for col in columns:
            if col not in df.columns:
                continue
            
            cardinality = df[col].nunique()
            
            if method == "onehot":
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"One-hot encoding column '{col}' with {cardinality} categories",
                    source="preprocess.encode",
                    meta={"column": col, "method": "onehot", "cardinality": cardinality}
                )
                
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"Created {len(dummies.columns)} binary columns",
                    source="preprocess.encode",
                    meta={"column": col, "created_columns": len(dummies.columns)}
                )
            
            elif method == "label":
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"Label encoding column '{col}' with {cardinality} categories",
                    source="preprocess.encode",
                    meta={"column": col, "method": "label", "cardinality": cardinality}
                )
                
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"Mapped categories to integers 0-{cardinality-1}",
                    source="preprocess.encode",
                    meta={"column": col, "range": [0, cardinality-1]}
                )
        
        return df
    
    async def _scale(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        """Scale numeric features."""
        columns = step["target_columns"]
        method = step["params"].get("method", "standard")
        
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Scaling {len(columns)} numeric columns",
            source="preprocess.scale",
            meta={"columns": columns, "method": method}
        )
        
        if method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers["numeric"] = scaler
        
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Applied {scaler.__class__.__name__} to {len(columns)} columns",
            source="preprocess.scale",
            meta={"scaler": scaler.__class__.__name__, "columns_count": len(columns)}
        )
        
        return df
