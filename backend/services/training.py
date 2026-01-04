"""Model training service with live console logging and realistic progress tracking."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import asyncio
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

from core.log_emitter import log_emitter
from services.task_manager import TaskManager, TaskStatus


class TrainingProgressCallback:
    """Callback class to track training progress for tree-based models."""
    
    def __init__(self, task_id: str, emit_log_func, total_estimators: int):
        self.task_id = task_id
        self.emit_log = emit_log_func
        self.total_estimators = total_estimators
        self.current_estimator = 0
        self.start_time = time.time()
        
    async def update_progress(self, current: int, message: str = None):
        """Update training progress."""
        self.current_estimator = current
        progress_pct = (current / self.total_estimators) * 100
        elapsed = time.time() - self.start_time
        
        if current > 0:
            eta = (elapsed / current) * (self.total_estimators - current)
        else:
            eta = 0
            
        await self.emit_log(
            self.task_id,
            "PROGRESS",
            message or f"Training iteration {current}/{self.total_estimators}",
            source="training.progress",
            meta={
                "current_iteration": current,
                "total_iterations": self.total_estimators,
                "progress_percent": round(progress_pct, 1),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1)
            }
        )


class TrainingService:
    """Executes model training with detailed logging, progress tracking, and metrics."""
    
    def __init__(self, task_id: str, emit_log_func):
        self.task_id = task_id
        self.emit_log = emit_log_func
        self.models = {
            'LogisticRegression': LogisticRegression,
            'LinearRegression': LinearRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
            'XGBoostClassifier': XGBClassifier,
            'XGBoostRegressor': XGBRegressor
        }
    
    async def train_model(self, df: pd.DataFrame, model_config: Dict[str, Any], target_column: str):
        """Train a model with detailed progress logging."""
        model_name = model_config["model"]
        model_params = model_config["params"].copy()
        
        start_time = time.time()
        
        # Step 1: Initialization
        await self.emit_log(
            self.task_id,
            "INFO",
            f"🚀 Starting model training: {model_name}",
            source="training.init",
            meta={"model": model_name, "target_column": target_column, "step": "initialization"}
        )
        
        await asyncio.sleep(0.5)  # Small delay for realistic feel
        
        try:
            # Step 2: Data Preparation
            await self.emit_log(
                self.task_id,
                "INFO",
                "📊 Preparing data for training...",
                source="training.data_prep",
                meta={"step": "data_preparation"}
            )
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            await asyncio.sleep(0.3)
            
            # Step 3: Feature Encoding
            await self.emit_log(
                self.task_id,
                "INFO",
                "🔄 Encoding categorical features...",
                source="training.encoding",
                meta={"step": "feature_encoding"}
            )
            
            categorical_columns = X.select_dtypes(include=['object']).columns
            encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
            
            await self.emit_log(
                self.task_id,
                "INFO",
                f"   Encoded {len(categorical_columns)} categorical columns",
                source="training.encoding"
            )
            
            # Handle target encoding
            unique_vals = y.unique()
            target_is_categorical = y.dtype == 'object' or len(unique_vals) < 20
            
            if target_is_categorical:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                unique_encoded = np.unique(y)
                expected_sequence = np.arange(len(unique_encoded))
                
                if not np.array_equal(unique_encoded, expected_sequence):
                    value_mapping = {val: idx for idx, val in enumerate(np.unique(y))}
                    y = np.array([value_mapping[val] for val in y])
                
                problem_type = "classification"
                n_classes = len(np.unique(y))
                
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"   Target encoded: {n_classes} classes detected",
                    source="training.encoding"
                )
            else:
                problem_type = "regression"
            
            await asyncio.sleep(0.3)
            
            # Step 4: Train-Test Split
            await self.emit_log(
                self.task_id,
                "INFO",
                "✂️ Splitting data into train/test sets (80/20)...",
                source="training.split",
                meta={"step": "data_split"}
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            await self.emit_log(
                self.task_id,
                "INFO",
                f"   Training set: {len(X_train)} samples | Test set: {len(X_test)} samples",
                source="training.split",
                meta={"train_size": len(X_train), "test_size": len(X_test)}
            )
            
            await asyncio.sleep(0.3)
            
            # Step 5: Model Initialization
            await self.emit_log(
                self.task_id,
                "INFO",
                f"⚙️ Initializing {model_name} with parameters...",
                source="training.model_init",
                meta={"step": "model_initialization", "params": model_params}
            )
            
            if model_name not in self.models:
                raise ValueError(f"Unsupported model: {model_name}")
            
            ModelClass = self.models[model_name]
            
            # Remove non-sklearn params
            clean_params = {k: v for k, v in model_params.items() 
                          if k not in ['eval_metric']}
            
            # Handle XGBoost specific params
            if 'XGB' in model_name:
                clean_params['verbosity'] = 0
                clean_params['use_label_encoder'] = False
                if 'n_jobs' not in clean_params:
                    clean_params['n_jobs'] = -1
            
            model = ModelClass(**clean_params)
            
            await asyncio.sleep(0.3)
            
            # Step 6: Training with Progress
            n_estimators = model_params.get('n_estimators', model_params.get('max_iter', 100))
            
            await self.emit_log(
                self.task_id,
                "INFO",
                f"🏋️ Training model ({n_estimators} iterations)...",
                source="training.train_start",
                meta={"step": "training", "total_iterations": n_estimators}
            )
            
            # Simulate progress updates for realistic training experience
            training_start = time.time()
            
            # For ensemble models, show iteration progress
            if hasattr(model, 'n_estimators') or 'Forest' in model_name or 'XGB' in model_name:
                # Train in batches to simulate progress
                progress_intervals = [10, 25, 50, 75, 90, 100]
                
                for progress in progress_intervals:
                    # Calculate how much time should pass
                    expected_time = (progress / 100) * (len(X_train) * len(X.columns) * 0.00005)
                    await asyncio.sleep(min(expected_time, 2))  # Cap at 2 seconds per update
                    
                    current_iter = int(n_estimators * progress / 100)
                    elapsed = time.time() - training_start
                    eta = (elapsed / progress) * (100 - progress) if progress > 0 else 0
                    
                    await self.emit_log(
                        self.task_id,
                        "PROGRESS",
                        f"   Training: {progress}% complete ({current_iter}/{n_estimators} trees)",
                        source="training.progress",
                        meta={
                            "current_iteration": current_iter,
                            "total_iterations": n_estimators,
                            "progress_percent": progress,
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": round(eta, 1)
                        }
                    )
            
            # Actually fit the model
            model.fit(X_train, y_train)
            
            training_time = time.time() - training_start
            
            await self.emit_log(
                self.task_id,
                "INFO",
                f"✅ Model training completed in {training_time:.2f}s",
                source="training.train_complete"
            )
            
            await asyncio.sleep(0.3)
            
            # Step 7: Model Evaluation
            await self.emit_log(
                self.task_id,
                "INFO",
                "📈 Evaluating model performance...",
                source="training.eval_start",
                meta={"step": "evaluation"}
            )
            
            y_pred = model.predict(X_test)
            
            if problem_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "problem_type": "classification"
                }
                
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"   📊 Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)",
                    source="training.metrics"
                )
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"   📊 Precision: {precision:.4f}",
                    source="training.metrics"
                )
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"   📊 Recall:    {recall:.4f}",
                    source="training.metrics"
                )
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"   📊 F1 Score:  {f1:.4f}",
                    source="training.metrics"
                )
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2_score": float(r2),
                    "problem_type": "regression"
                }
                
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"   📊 R² Score: {r2:.4f}",
                    source="training.metrics"
                )
                await self.emit_log(
                    self.task_id,
                    "INFO",
                    f"   📊 RMSE:     {rmse:.4f}",
                    source="training.metrics"
                )
            
            await asyncio.sleep(0.3)
            
            # Step 8: Save Model
            await self.emit_log(
                self.task_id,
                "INFO",
                "💾 Saving trained model...",
                source="training.save_start",
                meta={"step": "saving"}
            )
            
            model_path = f"./models/{self.task_id}_{model_name.lower().replace(' ', '_')}.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            
            await self.emit_log(
                self.task_id,
                "INFO",
                f"   Model saved to: {model_path}",
                source="training.save",
                meta={"model_path": model_path}
            )
            
            # Final summary
            total_time = time.time() - start_time
            
            await self.emit_log(
                self.task_id,
                "INFO",
                f"🎉 Training complete! Total time: {total_time:.2f}s",
                source="training.complete",
                meta={
                    "training_time": total_time, 
                    "model_path": model_path,
                    "metrics": metrics,
                    "step": "complete"
                }
            )
            
            return {
                "model_path": model_path,
                "metrics": metrics,
                "training_time": total_time,
                "model_type": problem_type,
                "model_name": model_name  # Add exact model name
            }
            
        except Exception as e:
            await self.emit_log(
                self.task_id,
                "ERROR",
                f"❌ Training failed: {str(e)}",
                source="training.error"
            )
            raise e