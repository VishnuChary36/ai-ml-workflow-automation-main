"""Model and data visualization service with interactive charts and plots."""
from services.task_manager import TaskManager, TaskStatus
from core.log_emitter import log_emitter
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import base64
import io
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


# Friendly model name mapping
MODEL_DISPLAY_NAMES = {
    'RandomForestClassifier': 'Random Forest Classifier',
    'RandomForestRegressor': 'Random Forest Regressor',
    'XGBoostClassifier': 'XGBoost Classifier',
    'XGBoostRegressor': 'XGBoost Regressor',
    'LogisticRegression': 'Logistic Regression',
    'LinearRegression': 'Linear Regression',
    'XGBClassifier': 'XGBoost Classifier',
    'XGBRegressor': 'XGBoost Regressor',
}


class VisualizationService:
    """Generates smart, data-analyst style visualizations for trained models."""

    def __init__(self, task_id: str, emit_log_func):
        self.task_id = task_id
        self.emit_log = emit_log_func
        self.max_samples_for_scatter = 1000  # Limit scatter plots for performance
        self.max_features_to_show = 15  # Top features to show in importance
        self.max_classes_for_detail = 10  # Max classes for detailed visualization

    def generate_visualizations(self, df: pd.DataFrame, model_path: str, target_column: str,
                                metrics: Dict[str, Any], model_type: str = "classification",
                                model_name: str = None) -> Dict[str, Any]:
        """Generate smart visualizations based on the trained model and dataset.
        
        This generates only the most relevant visualizations a data analyst would use:
        - For classification: Confusion Matrix, Feature Importance, Class Distribution
        - For regression: Actual vs Predicted (sampled), Residual Analysis, Feature Importance
        - Samples data for large datasets to avoid cluttered plots
        """
        visualizations = {}
        
        # Get friendly model display name
        display_model_name = MODEL_DISPLAY_NAMES.get(model_name, model_name) if model_name else model_type
        
        try:
            self.emit_log(
                self.task_id,
                "INFO",
                f"📊 Generating data analyst visualizations for {display_model_name}",
                source="visualization.init"
            )

            # Load the trained model
            model = joblib.load(model_path)

            # Prepare features and target
            X = df.drop(columns=[target_column])
            y_true = df[target_column].copy()
            
            # Store original values for display
            y_true_original = y_true.copy()
            n_samples = len(df)
            n_features = len(X.columns)
            
            # Dataset info for smart decisions
            is_large_dataset = n_samples > 5000
            is_high_dimensional = n_features > 20
            n_unique_classes = y_true.nunique() if model_type == "classification" else 0

            # Handle categorical features
            label_encoders = {}
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].apply(lambda x: isinstance(x, str)).any():
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].fillna('NaN').astype(str))
                    label_encoders[col] = le
                elif X[col].dtype in ['float64', 'float32']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            
            # Handle target encoding for classification
            target_encoder = None
            target_labels = None
            if model_type == "classification" and y_true.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                target_encoder = LabelEncoder()
                target_labels = y_true.unique().tolist()
                y_true = target_encoder.fit_transform(y_true.fillna('Unknown').astype(str))

            # Make predictions
            y_pred = model.predict(X)

            # ===== CLASSIFICATION VISUALIZATIONS =====
            if model_type == "classification":
                # 1. CONFUSION MATRIX - Most important for classification
                try:
                    cm = confusion_matrix(y_true, y_pred)
                    plt.figure(figsize=(8, 6))
                    
                    # Use labels if available and not too many classes
                    if target_labels and len(target_labels) <= self.max_classes_for_detail:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=target_labels, yticklabels=target_labels)
                    else:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    
                    plt.title(f'Confusion Matrix - {display_model_name}', fontsize=14, fontweight='bold')
                    plt.ylabel('Actual', fontsize=12)
                    plt.xlabel('Predicted', fontsize=12)
                    plt.tight_layout()

                    visualizations['confusion_matrix'] = self._plot_to_base64()
                    plt.close()
                    
                    self.emit_log(self.task_id, "INFO", "   ✓ Generated Confusion Matrix", source="visualization.cm")
                except Exception as e:
                    plt.close('all')
                    print(f"Error generating confusion matrix: {e}")

                # 2. FEATURE IMPORTANCE - Show only top features
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = X.columns.tolist()
                        
                        # Sort and take top N features
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=True)
                        
                        # Take only top features for clarity
                        top_n = min(self.max_features_to_show, len(importance_df))
                        importance_df = importance_df.tail(top_n)

                        plt.figure(figsize=(10, max(6, top_n * 0.4)))
                        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(importance_df)))
                        plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
                        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
                        plt.xlabel('Importance Score', fontsize=12)
                        plt.ylabel('Feature', fontsize=12)
                        
                        # Add value labels
                        for i, v in enumerate(importance_df['importance']):
                            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
                        
                        plt.tight_layout()
                        visualizations['feature_importance'] = self._plot_to_base64()
                        plt.close()
                        
                        self.emit_log(self.task_id, "INFO", f"   ✓ Generated Feature Importance (Top {top_n})", source="visualization.fi")
                except Exception as e:
                    plt.close('all')
                    print(f"Error generating feature importance: {e}")

                # 3. CLASS DISTRIBUTION - Simple bar chart comparing actual vs predicted
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Actual distribution
                    actual_counts = pd.Series(y_true_original).value_counts()
                    if len(actual_counts) > self.max_classes_for_detail:
                        actual_counts = actual_counts.head(self.max_classes_for_detail)
                    
                    axes[0].bar(range(len(actual_counts)), actual_counts.values, color='steelblue', alpha=0.8)
                    axes[0].set_xticks(range(len(actual_counts)))
                    axes[0].set_xticklabels(actual_counts.index, rotation=45, ha='right')
                    axes[0].set_title('Actual Class Distribution', fontsize=12, fontweight='bold')
                    axes[0].set_ylabel('Count')
                    
                    # Predicted distribution
                    if target_encoder:
                        pred_labels = target_encoder.inverse_transform(y_pred)
                    else:
                        pred_labels = y_pred
                    pred_counts = pd.Series(pred_labels).value_counts()
                    if len(pred_counts) > self.max_classes_for_detail:
                        pred_counts = pred_counts.head(self.max_classes_for_detail)
                    
                    axes[1].bar(range(len(pred_counts)), pred_counts.values, color='coral', alpha=0.8)
                    axes[1].set_xticks(range(len(pred_counts)))
                    axes[1].set_xticklabels(pred_counts.index, rotation=45, ha='right')
                    axes[1].set_title('Predicted Class Distribution', fontsize=12, fontweight='bold')
                    axes[1].set_ylabel('Count')
                    
                    plt.suptitle(f'Class Distribution Comparison - {display_model_name}', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    visualizations['prediction_distribution'] = self._plot_to_base64()
                    plt.close()
                    
                    self.emit_log(self.task_id, "INFO", "   ✓ Generated Class Distribution", source="visualization.dist")
                except Exception as e:
                    plt.close('all')
                    print(f"Error generating prediction distribution: {e}")

            # ===== REGRESSION VISUALIZATIONS =====
            else:
                # 1. ACTUAL VS PREDICTED - Sample for large datasets
                try:
                    plt.figure(figsize=(10, 8))
                    
                    # Sample if too many points
                    if len(y_true) > self.max_samples_for_scatter:
                        sample_idx = np.random.choice(len(y_true), self.max_samples_for_scatter, replace=False)
                        y_true_sample = np.array(y_true)[sample_idx]
                        y_pred_sample = np.array(y_pred)[sample_idx]
                        sample_note = f" (sampled {self.max_samples_for_scatter} of {len(y_true)} points)"
                    else:
                        y_true_sample = y_true
                        y_pred_sample = y_pred
                        sample_note = ""
                    
                    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5, color='steelblue', s=30)
                    
                    # Perfect prediction line
                    min_val = min(min(y_true_sample), min(y_pred_sample))
                    max_val = max(max(y_true_sample), max(y_pred_sample))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                    
                    plt.xlabel('Actual Values', fontsize=12)
                    plt.ylabel('Predicted Values', fontsize=12)
                    plt.title(f'Actual vs Predicted{sample_note}', fontsize=14, fontweight='bold')
                    plt.legend()
                    
                    # Add R² annotation
                    r2 = r2_score(y_true, y_pred)
                    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                                 fontsize=12, fontweight='bold', 
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    visualizations['actual_vs_predicted'] = self._plot_to_base64()
                    plt.close()
                    
                    self.emit_log(self.task_id, "INFO", "   ✓ Generated Actual vs Predicted", source="visualization.avp")
                except Exception as e:
                    plt.close('all')
                    print(f"Error generating actual vs predicted: {e}")

                # 2. RESIDUAL ANALYSIS - Key for regression diagnostics
                try:
                    residuals = np.array(y_true) - np.array(y_pred)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Residuals vs Predicted (sampled)
                    if len(residuals) > self.max_samples_for_scatter:
                        sample_idx = np.random.choice(len(residuals), self.max_samples_for_scatter, replace=False)
                        res_sample = residuals[sample_idx]
                        pred_sample = np.array(y_pred)[sample_idx]
                    else:
                        res_sample = residuals
                        pred_sample = y_pred
                    
                    axes[0].scatter(pred_sample, res_sample, alpha=0.5, color='steelblue', s=30)
                    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
                    axes[0].set_xlabel('Predicted Values', fontsize=11)
                    axes[0].set_ylabel('Residuals', fontsize=11)
                    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
                    
                    # Residual distribution histogram
                    axes[1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
                    axes[1].set_xlabel('Residual Value', fontsize=11)
                    axes[1].set_ylabel('Frequency', fontsize=11)
                    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
                    
                    # Add stats
                    mean_res = np.mean(residuals)
                    std_res = np.std(residuals)
                    axes[1].annotate(f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}', 
                                     xy=(0.95, 0.95), xycoords='axes fraction',
                                     fontsize=10, ha='right', va='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.suptitle(f'Residual Analysis - {display_model_name}', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    visualizations['residual_plot'] = self._plot_to_base64()
                    plt.close()
                    
                    self.emit_log(self.task_id, "INFO", "   ✓ Generated Residual Analysis", source="visualization.res")
                except Exception as e:
                    plt.close('all')
                    print(f"Error generating residual plot: {e}")

                # 3. FEATURE IMPORTANCE - For tree-based regressors
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = X.columns.tolist()
                        
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=True)
                        
                        top_n = min(self.max_features_to_show, len(importance_df))
                        importance_df = importance_df.tail(top_n)

                        plt.figure(figsize=(10, max(6, top_n * 0.4)))
                        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(importance_df)))
                        plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
                        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
                        plt.xlabel('Importance Score', fontsize=12)
                        plt.ylabel('Feature', fontsize=12)
                        
                        for i, v in enumerate(importance_df['importance']):
                            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
                        
                        plt.tight_layout()
                        visualizations['feature_importance'] = self._plot_to_base64()
                        plt.close()
                        
                        self.emit_log(self.task_id, "INFO", f"   ✓ Generated Feature Importance (Top {top_n})", source="visualization.fi")
                except Exception as e:
                    plt.close('all')
                    print(f"Error generating feature importance: {e}")

            self.emit_log(
                self.task_id,
                "INFO",
                f"✅ Generated {len(visualizations)} key visualizations",
                source="visualization.complete"
            )

            return {
                "visualizations": visualizations,
                "metrics": metrics,
                "model_type": model_type,
                "model_name": display_model_name,
                "target_column": target_column,
                "dataset_info": {
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "n_classes": int(n_unique_classes) if model_type == "classification" else None
                }
            }

        except Exception as e:
            self.emit_log(
                self.task_id,
                "ERROR",
                f"Visualization generation failed: {str(e)}",
                source="visualization.error"
            )
            raise e

    def _plot_to_base64(self) -> str:
        """Convert current matplotlib plot to base64 string."""
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        img_buffer.close()
        return img_str


async def generate_model_visualizations(task_id: str, dataset_path: str, model_path: str,
                                        target_column: str, metrics: Dict[str, Any],
                                        model_type: str, emit_log_func, model_id: str,
                                        model_name: str = None):
    """Generate visualizations for a trained model."""
    from core.database import SessionLocal

    db = SessionLocal()

    try:
        # Update status
        TaskManager.update_status(db, task_id, TaskStatus.RUNNING)

        await emit_log_func(
            task_id, "INFO",
            f"Starting visualization generation for model {model_path}",
            source="visualization.init"
        )

        # Load dataset
        df = pd.read_csv(dataset_path)

        # Initialize visualization service
        viz_service = VisualizationService(task_id, emit_log_func)

        # Generate visualizations with model name
        result = viz_service.generate_visualizations(
            df, model_path, target_column, metrics, model_type, model_name)

        # Save visualization result
        viz_result_path = f"./artifacts/{task_id}_{model_id}_visualizations.json"
        os.makedirs(os.path.dirname(viz_result_path), exist_ok=True)

        with open(viz_result_path, 'w') as f:
            json.dump({
                "task_id": task_id,
                "model_id": model_id,
                "generated_at": datetime.utcnow().isoformat(),
                "visualizations": result["visualizations"],
                "metrics": result["metrics"],
                "model_type": result["model_type"],
                "model_name": result.get("model_name", model_name),
                "target_column": result["target_column"],
                "dataset_info": result.get("dataset_info", {})
            }, f, indent=2)

        await emit_log_func(
            task_id, "INFO",
            f"Visualizations saved to: {viz_result_path}",
            source="visualization.save",
            meta={"result_path": viz_result_path}
        )

        # Update task
        TaskManager.update_status(
            db, task_id, TaskStatus.COMPLETED,
            result={"visualizations_path": viz_result_path,
                    "model_type": model_type}
        )

    except Exception as e:
        await emit_log_func(
            task_id, "ERROR",
            f"Visualization generation failed: {str(e)}",
            source="visualization.error"
        )
        TaskManager.update_status(db, task_id, TaskStatus.FAILED, error=str(e))

    finally:
        db.close()
