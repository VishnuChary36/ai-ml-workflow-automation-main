"""Model Explainability Service using SHAP, LIME, and other techniques."""
import os
import json
import base64
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Try to import optional explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")


class ExplainabilityService:
    """Generates comprehensive model explanations using various techniques."""
    
    def __init__(self, task_id: str, emit_log_func=None):
        self.task_id = task_id
        self.emit_log = emit_log_func or (lambda *args, **kwargs: None)
        self.max_samples_for_shap = 500  # SHAP can be slow, limit samples
        self.max_features_to_show = 15
        self.random_state = 42
    
    def _log(self, level: str, message: str, source: str = "explainability"):
        """Helper to emit logs."""
        try:
            self.emit_log(self.task_id, level, message, source=source)
        except:
            print(f"[{level}] {message}")
    
    def _plot_to_base64(self) -> str:
        """Convert current matplotlib plot to base64 string."""
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        img_buffer.close()
        plt.close()
        return img_str
    
    def _prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Prepare data for explainability, handling categorical features."""
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()
        
        encoders = {}
        feature_names = X.columns.tolist()
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].apply(lambda x: isinstance(x, str)).any():
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('NaN').astype(str))
                encoders[col] = le
            elif X[col].dtype in ['float64', 'float32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.fillna('Unknown').astype(str)))
            encoders['_target'] = le
        
        return X, y, encoders
    
    def generate_explainability(self, df: pd.DataFrame, model_path: str, 
                                 target_column: str, model_type: str = "classification",
                                 model_name: str = None) -> Dict[str, Any]:
        """Generate comprehensive explainability report."""
        explanations = {}
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "target_column": target_column,
            "n_samples": len(df),
            "n_features": len(df.columns) - 1,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        self._log("INFO", f"🔍 Starting explainability analysis for {model_name or model_type}")
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Prepare data
            X, y, encoders = self._prepare_data(df, target_column)
            feature_names = X.columns.tolist()
            
            # Sample data if too large
            if len(X) > self.max_samples_for_shap:
                sample_idx = np.random.choice(len(X), self.max_samples_for_shap, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx] if isinstance(y, pd.Series) else y[sample_idx]
            else:
                X_sample = X
                y_sample = y
            
            # Make predictions
            y_pred = model.predict(X)
            
            # 1. Permutation Feature Importance
            self._log("INFO", "   📊 Computing Permutation Feature Importance...")
            try:
                perm_importance = self._generate_permutation_importance(model, X_sample, y_sample, feature_names)
                if perm_importance:
                    explanations['permutation_importance'] = perm_importance
            except Exception as e:
                self._log("WARN", f"   Permutation importance failed: {str(e)}")
            
            # 2. SHAP Analysis
            if SHAP_AVAILABLE:
                self._log("INFO", "   🎯 Computing SHAP values...")
                try:
                    shap_results = self._generate_shap_explanations(model, X_sample, feature_names, model_type)
                    explanations.update(shap_results)
                except Exception as e:
                    self._log("WARN", f"   SHAP analysis failed: {str(e)}")
            else:
                self._log("WARN", "   SHAP not installed, skipping SHAP analysis")
            
            # 3. Partial Dependence Plots (for top features)
            self._log("INFO", "   📈 Generating Partial Dependence Plots...")
            try:
                pdp_plots = self._generate_pdp_plots(model, X_sample, feature_names, model_type)
                if pdp_plots:
                    explanations['partial_dependence'] = pdp_plots
            except Exception as e:
                self._log("WARN", f"   PDP generation failed: {str(e)}")
            
            # 4. LIME Explanations (for sample instances)
            if LIME_AVAILABLE:
                self._log("INFO", "   🍋 Generating LIME explanations...")
                try:
                    lime_results = self._generate_lime_explanations(model, X, X_sample, feature_names, model_type)
                    if lime_results:
                        explanations['lime_explanations'] = lime_results
                except Exception as e:
                    self._log("WARN", f"   LIME analysis failed: {str(e)}")
            else:
                self._log("WARN", "   LIME not installed, skipping LIME analysis")
            
            # 5. Surrogate Model (Decision Tree approximation)
            self._log("INFO", "   🌳 Fitting Surrogate Decision Tree...")
            try:
                surrogate = self._generate_surrogate_model(model, X_sample, model_type)
                if surrogate:
                    explanations['surrogate_tree'] = surrogate
            except Exception as e:
                self._log("WARN", f"   Surrogate model failed: {str(e)}")
            
            # 6. Confusion Matrix Analysis (for classification)
            if model_type == "classification":
                self._log("INFO", "   📋 Generating Confusion Matrix Analysis...")
                try:
                    cm_analysis = self._generate_confusion_analysis(y, y_pred, encoders)
                    if cm_analysis:
                        explanations['confusion_analysis'] = cm_analysis
                except Exception as e:
                    self._log("WARN", f"   Confusion analysis failed: {str(e)}")
            
            # 7. Calibration Plot (for classification with predict_proba)
            if model_type == "classification" and hasattr(model, 'predict_proba'):
                self._log("INFO", "   📉 Generating Calibration Plot...")
                try:
                    calibration = self._generate_calibration_plot(model, X, y)
                    if calibration:
                        explanations['calibration_plot'] = calibration
                except Exception as e:
                    self._log("WARN", f"   Calibration plot failed: {str(e)}")
            
            # 8. Feature Distribution Analysis
            self._log("INFO", "   📊 Analyzing Feature Distributions...")
            try:
                distributions = self._generate_feature_distributions(X, feature_names)
                if distributions:
                    explanations['feature_distributions'] = distributions
            except Exception as e:
                self._log("WARN", f"   Feature distribution failed: {str(e)}")
            
            # 9. Correlation Analysis
            self._log("INFO", "   🔗 Computing Feature Correlations...")
            try:
                correlations = self._generate_correlation_analysis(X, feature_names)
                if correlations:
                    explanations['correlation_analysis'] = correlations
            except Exception as e:
                self._log("WARN", f"   Correlation analysis failed: {str(e)}")
            
            self._log("INFO", f"✅ Explainability analysis complete! Generated {len(explanations)} explanations")
            
            return {
                "explanations": explanations,
                "metadata": metadata,
                "available_methods": list(explanations.keys())
            }
            
        except Exception as e:
            self._log("ERROR", f"❌ Explainability analysis failed: {str(e)}")
            raise e
    
    def _generate_permutation_importance(self, model, X, y, feature_names) -> Dict:
        """Generate permutation importance plot."""
        result = permutation_importance(model, X, y, n_repeats=10, random_state=self.random_state, n_jobs=-1)
        
        # Sort by importance
        sorted_idx = result.importances_mean.argsort()[::-1][:self.max_features_to_show]
        
        plt.figure(figsize=(10, max(6, len(sorted_idx) * 0.4)))
        
        importances = result.importances_mean[sorted_idx]
        std = result.importances_std[sorted_idx]
        names = [feature_names[i] for i in sorted_idx]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
        bars = plt.barh(range(len(names)), importances, xerr=std, color=colors, capsize=3)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Permutation Importance (decrease in score)', fontsize=11)
        plt.title('Permutation Feature Importance\n(How much does shuffling each feature hurt predictions?)', 
                  fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (v, s) in enumerate(zip(importances, std)):
            plt.text(v + s + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        return {
            "plot": self._plot_to_base64(),
            "description": "Shows how much model performance decreases when each feature is randomly shuffled. Higher values indicate more important features.",
            "top_features": [{"name": n, "importance": float(i), "std": float(s)} 
                           for n, i, s in zip(names[:5], importances[:5], std[:5])]
        }
    
    def _generate_shap_explanations(self, model, X, feature_names, model_type) -> Dict:
        """Generate SHAP summary and dependence plots."""
        results = {}
        
        # Create SHAP explainer
        try:
            # Try TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except:
            try:
                # Fall back to KernelExplainer
                background = shap.sample(X, min(100, len(X)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X.iloc[:min(100, len(X))])
                X = X.iloc[:min(100, len(X))]
            except Exception as e:
                self._log("WARN", f"Could not create SHAP explainer: {str(e)}")
                return results
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # For multi-class, use the mean absolute SHAP values
            shap_values_plot = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values_plot = shap_values
        
        # 1. SHAP Summary Plot (Beeswarm)
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X, feature_names=feature_names, 
                            show=False, max_display=self.max_features_to_show)
            plt.title('SHAP Summary Plot (Feature Impact on Predictions)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            results['shap_summary'] = {
                "plot": self._plot_to_base64(),
                "description": "Each dot represents a sample. Position on x-axis shows impact on prediction. Color shows feature value (red=high, blue=low)."
            }
        except Exception as e:
            self._log("WARN", f"SHAP summary plot failed: {str(e)}")
        
        # 2. SHAP Bar Plot (Global Importance)
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_plot, X, feature_names=feature_names, 
                            plot_type="bar", show=False, max_display=self.max_features_to_show)
            plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            results['shap_importance'] = {
                "plot": self._plot_to_base64(),
                "description": "Average absolute SHAP value for each feature. Shows overall feature importance across all predictions."
            }
        except Exception as e:
            self._log("WARN", f"SHAP bar plot failed: {str(e)}")
        
        # 3. SHAP Dependence Plot for top feature
        try:
            # Get feature with highest mean SHAP
            if isinstance(shap_values_plot, np.ndarray) and len(shap_values_plot.shape) == 2:
                mean_shap = np.abs(shap_values_plot).mean(axis=0)
                top_feature_idx = np.argmax(mean_shap)
                top_feature = feature_names[top_feature_idx]
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(top_feature_idx, shap_values_plot, X, 
                                    feature_names=feature_names, show=False)
                plt.title(f'SHAP Dependence Plot: {top_feature}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                results['shap_dependence'] = {
                    "plot": self._plot_to_base64(),
                    "feature": top_feature,
                    "description": f"Shows how {top_feature} values affect predictions. Color indicates interaction with another feature."
                }
        except Exception as e:
            self._log("WARN", f"SHAP dependence plot failed: {str(e)}")
        
        return results
    
    def _generate_pdp_plots(self, model, X, feature_names, model_type) -> Dict:
        """Generate Partial Dependence Plots for top features."""
        # Get feature importance to select top features
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:4]  # Top 4 features
        else:
            top_idx = list(range(min(4, len(feature_names))))
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, idx in enumerate(top_idx):
                if i < len(axes):
                    try:
                        PartialDependenceDisplay.from_estimator(
                            model, X, [idx], 
                            feature_names=feature_names,
                            ax=axes[i],
                            kind='average'
                        )
                        axes[i].set_title(f'PDP: {feature_names[idx]}', fontsize=11)
                    except:
                        axes[i].text(0.5, 0.5, f'Could not generate PDP\nfor {feature_names[idx]}',
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f'PDP: {feature_names[idx]} (failed)', fontsize=11)
            
            plt.suptitle('Partial Dependence Plots\n(How each feature affects predictions on average)', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            return {
                "plot": self._plot_to_base64(),
                "features": [feature_names[i] for i in top_idx],
                "description": "Shows the marginal effect of each feature on predictions, averaging out other features."
            }
        except Exception as e:
            self._log("WARN", f"PDP generation failed: {str(e)}")
            return None
    
    def _generate_lime_explanations(self, model, X_full, X_sample, feature_names, model_type) -> Dict:
        """Generate LIME explanations for sample instances."""
        # Create LIME explainer
        if model_type == "classification":
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_full.values,
                feature_names=feature_names,
                class_names=['Class 0', 'Class 1'],  # Will be updated if more classes
                mode='classification',
                random_state=self.random_state
            )
        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_full.values,
                feature_names=feature_names,
                mode='regression',
                random_state=self.random_state
            )
        
        # Explain a few representative instances
        n_instances = min(3, len(X_sample))
        instance_indices = np.random.choice(len(X_sample), n_instances, replace=False)
        
        fig, axes = plt.subplots(1, n_instances, figsize=(5 * n_instances, 6))
        if n_instances == 1:
            axes = [axes]
        
        explanations_data = []
        
        for i, idx in enumerate(instance_indices):
            instance = X_sample.iloc[idx].values
            
            try:
                if model_type == "classification" and hasattr(model, 'predict_proba'):
                    exp = explainer.explain_instance(instance, model.predict_proba, num_features=8)
                else:
                    exp = explainer.explain_instance(instance, model.predict, num_features=8)
                
                # Get explanation as list
                exp_list = exp.as_list()
                
                # Plot
                features = [x[0] for x in exp_list]
                weights = [x[1] for x in exp_list]
                
                colors = ['green' if w > 0 else 'red' for w in weights]
                axes[i].barh(range(len(features)), weights, color=colors, alpha=0.7)
                axes[i].set_yticks(range(len(features)))
                axes[i].set_yticklabels(features, fontsize=9)
                axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                axes[i].set_xlabel('Contribution to Prediction')
                axes[i].set_title(f'Instance #{idx}', fontsize=11)
                
                explanations_data.append({
                    "instance_id": int(idx),
                    "prediction": float(model.predict(instance.reshape(1, -1))[0]),
                    "top_features": [{"feature": f, "weight": float(w)} for f, w in exp_list[:5]]
                })
            except Exception as e:
                axes[i].text(0.5, 0.5, f'LIME failed\n{str(e)[:30]}',
                           ha='center', va='center', transform=axes[i].transAxes)
        
        plt.suptitle('LIME Local Explanations\n(Why did the model predict this for specific instances?)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        return {
            "plot": self._plot_to_base64(),
            "instances": explanations_data,
            "description": "Shows which features pushed the prediction up (green) or down (red) for individual instances."
        }
    
    def _generate_surrogate_model(self, model, X, model_type) -> Dict:
        """Fit a simple decision tree as a surrogate model for global explanation."""
        # Get predictions from the complex model
        y_surrogate = model.predict(X)
        
        # Fit a simple decision tree
        if model_type == "classification":
            surrogate = DecisionTreeClassifier(max_depth=4, random_state=self.random_state)
        else:
            surrogate = DecisionTreeRegressor(max_depth=4, random_state=self.random_state)
        
        surrogate.fit(X, y_surrogate)
        
        # Calculate approximation accuracy
        surrogate_pred = surrogate.predict(X)
        if model_type == "classification":
            accuracy = (surrogate_pred == y_surrogate).mean()
            metric_name = "Fidelity (accuracy)"
        else:
            from sklearn.metrics import r2_score
            accuracy = r2_score(y_surrogate, surrogate_pred)
            metric_name = "Fidelity (R²)"
        
        # Plot the tree
        plt.figure(figsize=(20, 10))
        plot_tree(surrogate, feature_names=X.columns.tolist(), 
                 filled=True, rounded=True, fontsize=9,
                 class_names=[str(c) for c in np.unique(y_surrogate)] if model_type == "classification" else None)
        plt.title(f'Surrogate Decision Tree ({metric_name}: {accuracy:.2%})\n(Simple approximation of the complex model)', 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        return {
            "plot": self._plot_to_base64(),
            "fidelity": float(accuracy),
            "description": f"A simple decision tree that approximates the complex model's decisions. {metric_name}: {accuracy:.2%}"
        }
    
    def _generate_confusion_analysis(self, y_true, y_pred, encoders) -> Dict:
        """Generate detailed confusion matrix analysis - limited to top classes for readability."""
        from collections import Counter
        
        # Get all unique classes
        all_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # Count occurrences of each true class
        class_counts = Counter(y_true)
        
        # Get top N classes by frequency (max 10 for readability)
        max_classes = 10
        top_classes = [cls for cls, _ in class_counts.most_common(max_classes)]
        
        # Get class labels
        if '_target' in encoders:
            all_class_names = encoders['_target'].classes_
            # Create mapping from encoded to original names
            class_name_map = {i: name for i, name in enumerate(all_class_names)}
        else:
            class_name_map = {i: str(i) for i in all_classes}
        
        # Filter to only top classes
        mask = np.isin(y_true, top_classes)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Build confusion matrix for top classes only
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
        
        # Get display names for top classes
        class_labels = [class_name_map.get(c, str(c)) for c in top_classes]
        # Truncate long class names
        class_labels = [name[:15] + '...' if len(str(name)) > 15 else str(name) for name in class_labels]
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Adjust figure size based on number of classes
        n_classes = len(top_classes)
        fig_width = max(10, n_classes * 1.2)
        fig_height = max(6, n_classes * 0.6)
        
        fig, axes = plt.subplots(1, 2, figsize=(fig_width * 2, fig_height))
        
        # Raw counts - use smaller font for many classes
        fontsize = max(8, 12 - n_classes // 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels, ax=axes[0],
                   annot_kws={"size": fontsize})
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('Actual', fontsize=11)
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0].tick_params(axis='y', rotation=0, labelsize=9)
        
        # Normalized (recall per class)
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels, ax=axes[1],
                   annot_kws={"size": fontsize})
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('Actual', fontsize=11)
        axes[1].set_title('Confusion Matrix (Normalized by Row)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1].tick_params(axis='y', rotation=0, labelsize=9)
        
        title_suffix = f" (Top {n_classes} classes)" if len(all_classes) > max_classes else ""
        plt.suptitle(f'Confusion Matrix Analysis{title_suffix}\n(Where does the model make mistakes?)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Calculate per-class metrics for TOP classes only
        per_class = {}
        for i, cls in enumerate(top_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_name = class_name_map.get(cls, str(cls))
            per_class[str(class_name)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(cm[i, :].sum())
            }
        
        return {
            "plot": self._plot_to_base64(),
            "per_class_metrics": per_class,
            "total_classes": len(all_classes),
            "shown_classes": len(top_classes),
            "description": f"Showing top {n_classes} classes by frequency. Left: Raw prediction counts. Right: Row-normalized (recall)."
        }
    
    def _generate_calibration_plot(self, model, X, y) -> Dict:
        """Generate probability calibration plot."""
        y_prob = model.predict_proba(X)
        
        # Handle binary vs multiclass
        if y_prob.shape[1] == 2:
            # Binary classification
            prob_pos = y_prob[:, 1]
            y_binary = y
        else:
            # For multiclass, show calibration for each class
            prob_pos = y_prob.max(axis=1)  # Use max probability
            y_binary = (model.predict(X) == y).astype(int)  # Correct vs incorrect
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, prob_pos, n_bins=10, strategy='uniform'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration plot
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        axes[0].plot(mean_predicted_value, fraction_of_positives, 's-', 
                    color='steelblue', label='Model')
        axes[0].set_xlabel('Mean Predicted Probability', fontsize=11)
        axes[0].set_ylabel('Fraction of Positives', fontsize=11)
        axes[0].set_title('Calibration Curve (Reliability Diagram)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of predicted probabilities
        axes[1].hist(prob_pos, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Predicted Probability', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Distribution of Predicted Probabilities', fontsize=12)
        
        plt.suptitle('Probability Calibration Analysis\n(Can we trust the predicted probabilities?)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        return {
            "plot": self._plot_to_base64(),
            "description": "Left: How well predicted probabilities match actual outcomes. Points on diagonal = well calibrated. Right: Distribution of confidence levels."
        }
    
    def _generate_feature_distributions(self, X, feature_names) -> Dict:
        """Generate feature distribution plots for top features."""
        # Select top 6 features by variance
        variances = X.var()
        top_features = variances.nlargest(6).index.tolist()
        
        n_features = len(top_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                data = X[feature].dropna()
                
                # Check if numeric
                if data.dtype in ['float64', 'float32', 'int64', 'int32']:
                    axes[i].hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                    axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
                    axes[i].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
                    axes[i].legend(fontsize=8)
                else:
                    data.value_counts().head(10).plot(kind='bar', ax=axes[i], color='steelblue', alpha=0.7)
                    axes[i].tick_params(axis='x', rotation=45)
                
                axes[i].set_title(feature, fontsize=10)
                axes[i].set_xlabel('')
        
        # Hide empty subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions\n(Understanding your input data)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        return {
            "plot": self._plot_to_base64(),
            "features": top_features,
            "description": "Distribution of top features by variance. Red line = mean, green line = median."
        }
    
    def _generate_correlation_analysis(self, X, feature_names) -> Dict:
        """Generate feature correlation heatmap."""
        # Select numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        # Limit to top 15 features by variance
        if len(numeric_cols) > 15:
            variances = X[numeric_cols].var()
            numeric_cols = variances.nlargest(15).index.tolist()
        
        corr_matrix = X[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=len(numeric_cols) <= 10,
                   cmap='RdBu_r', center=0, square=True, fmt='.2f',
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix\n(Identifying multicollinearity)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_matrix.iloc[i, j])
                    })
        
        return {
            "plot": self._plot_to_base64(),
            "high_correlations": high_corr_pairs,
            "description": "Shows correlations between features. High correlations (|r| > 0.7) may cause unstable feature attributions."
        }
