"""AI-powered suggestion engine for pipeline and model recommendations."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class SuggestionEngine:
    """Generates AI-powered suggestions for preprocessing and models."""
    
    @staticmethod
    def analyze_dataset_complexity(df: pd.DataFrame, profile: Dict[str, Any], target_column: str) -> Dict[str, Any]:
        """Analyze dataset to determine complexity and best model fit."""
        n_rows = profile["rows"]
        n_cols = profile["columns"] - 1  # Excluding target
        
        # Get target info
        target_info = next((c for c in profile["columns_info"] if c["name"] == target_column), None)
        
        # Calculate metrics for model selection
        analysis = {
            "n_samples": n_rows,
            "n_features": n_cols,
            "sample_feature_ratio": n_rows / max(n_cols, 1),
            "is_small_dataset": n_rows < 1000,
            "is_medium_dataset": 1000 <= n_rows < 50000,
            "is_large_dataset": n_rows >= 50000,
            "is_high_dimensional": n_cols > 50,
            "has_missing_values": any(c["missing_percent"] > 0 for c in profile["columns_info"]),
            "missing_value_ratio": sum(c["missing_percent"] for c in profile["columns_info"]) / len(profile["columns_info"]),
        }
        
        # Categorical feature analysis
        cat_cols = [c for c in profile["columns_info"] if c["type"] in ["object", "category"]]
        num_cols = [c for c in profile["columns_info"] if c["type"] in ["int64", "float64"]]
        
        analysis["n_categorical"] = len(cat_cols)
        analysis["n_numerical"] = len(num_cols)
        analysis["categorical_ratio"] = len(cat_cols) / max(len(profile["columns_info"]), 1)
        
        # Class imbalance for classification
        if target_info and target_info["type"] in ["object", "category", "int64"]:
            try:
                target_values = df[target_column].value_counts()
                if len(target_values) > 1:
                    max_class = target_values.max()
                    min_class = target_values.min()
                    analysis["class_imbalance_ratio"] = max_class / max(min_class, 1)
                    analysis["n_classes"] = len(target_values)
                    analysis["is_imbalanced"] = analysis["class_imbalance_ratio"] > 3
                    analysis["is_multiclass"] = len(target_values) > 2
                else:
                    analysis["class_imbalance_ratio"] = 1
                    analysis["n_classes"] = 1
                    analysis["is_imbalanced"] = False
                    analysis["is_multiclass"] = False
            except:
                analysis["class_imbalance_ratio"] = 1
                analysis["n_classes"] = 2
                analysis["is_imbalanced"] = False
                analysis["is_multiclass"] = False
        else:
            analysis["class_imbalance_ratio"] = 1
            analysis["n_classes"] = 0
            analysis["is_imbalanced"] = False
            analysis["is_multiclass"] = False
        
        # Feature variance analysis for numerical columns
        if num_cols:
            try:
                variances = df[[c["name"] for c in num_cols if c["name"] != target_column]].var()
                analysis["high_variance_features"] = sum(variances > variances.median() * 10) if len(variances) > 0 else 0
            except:
                analysis["high_variance_features"] = 0
        else:
            analysis["high_variance_features"] = 0
        
        return analysis
    
    @staticmethod
    def calculate_model_score(model_name: str, analysis: Dict[str, Any], problem_type: str) -> float:
        """Calculate a score for how suitable a model is for the given dataset."""
        score = 50.0  # Base score
        
        if problem_type == "classification":
            if model_name == "XGBoostClassifier":
                # XGBoost excels with medium to large datasets
                if analysis["is_large_dataset"]:
                    score += 25
                elif analysis["is_medium_dataset"]:
                    score += 20
                else:
                    score += 10
                
                # Handles imbalanced data well
                if analysis["is_imbalanced"]:
                    score += 15
                
                # Handles missing values
                if analysis["has_missing_values"]:
                    score += 10
                
                # Good with mixed features
                if analysis["categorical_ratio"] > 0.3:
                    score += 5
                    
            elif model_name == "RandomForestClassifier":
                # Random Forest is robust for most datasets
                if analysis["is_medium_dataset"]:
                    score += 20
                elif analysis["is_small_dataset"]:
                    score += 15
                else:
                    score += 10
                
                # Good with high dimensional data
                if analysis["is_high_dimensional"]:
                    score += 10
                
                # Handles imbalanced reasonably
                if analysis["is_imbalanced"]:
                    score += 8
                
                # Good interpretability
                score += 5
                
            elif model_name == "LogisticRegression":
                # Best for small, simple datasets
                if analysis["is_small_dataset"] and not analysis["is_high_dimensional"]:
                    score += 20
                elif analysis["is_medium_dataset"] and analysis["n_features"] < 30:
                    score += 10
                else:
                    score -= 10
                
                # Not great for imbalanced
                if analysis["is_imbalanced"]:
                    score -= 5
                
                # Fast training
                score += 10
                
        else:  # Regression
            if model_name == "XGBoostRegressor":
                if analysis["is_large_dataset"]:
                    score += 25
                elif analysis["is_medium_dataset"]:
                    score += 20
                else:
                    score += 10
                
                if analysis["has_missing_values"]:
                    score += 10
                
                if analysis["high_variance_features"] > 0:
                    score += 5
                    
            elif model_name == "RandomForestRegressor":
                if analysis["is_medium_dataset"]:
                    score += 20
                elif analysis["is_small_dataset"]:
                    score += 15
                else:
                    score += 10
                
                if analysis["is_high_dimensional"]:
                    score += 10
                
                score += 5  # Good interpretability
                
            elif model_name == "LinearRegression":
                if analysis["is_small_dataset"] and analysis["n_features"] < 20:
                    score += 25
                elif analysis["is_medium_dataset"] and analysis["n_features"] < 30:
                    score += 10
                else:
                    score -= 15
                
                score += 10  # Fast training
        
        return min(max(score, 0), 100)  # Clamp between 0-100
    
    @staticmethod
    def estimate_training_time(model_name: str, n_samples: int, n_features: int, n_estimators: int = 100) -> Dict[str, Any]:
        """Estimate training time based on dataset size and model complexity."""
        # Base time per sample-feature interaction (in seconds)
        base_times = {
            "XGBoostClassifier": 0.00005,
            "XGBoostRegressor": 0.00005,
            "RandomForestClassifier": 0.00008,
            "RandomForestRegressor": 0.00008,
            "LogisticRegression": 0.00001,
            "LinearRegression": 0.000005,
        }
        
        base = base_times.get(model_name, 0.0001)
        
        # Calculate estimated time
        estimated_seconds = base * n_samples * n_features * (n_estimators / 100)
        
        # Add overhead
        estimated_seconds += 2  # Minimum overhead
        
        # Scale for realistic times
        if "Forest" in model_name or "XGB" in model_name:
            estimated_seconds *= 1.5  # Tree building overhead
        
        return {
            "estimated_seconds": round(estimated_seconds, 1),
            "estimated_minutes": round(estimated_seconds / 60, 2),
            "iterations": n_estimators if "Forest" in model_name or "XGB" in model_name else 1000 if "Logistic" in model_name else 1,
            "training_steps": ["Data preparation", "Feature encoding", "Model initialization", "Training iterations", "Validation", "Metrics calculation"]
        }
    
    @staticmethod
    def suggest_pipeline(df: pd.DataFrame, profile: Dict[str, Any], target_column: Optional[str] = None) -> List[Dict[str, Any]]:
        """Suggest preprocessing pipeline steps based on dataset profile."""
        suggestions = []
        step_counter = 1
        
        # 1. Handle missing values
        for col_info in profile["columns_info"]:
            col_name = col_info["name"]
            missing_pct = col_info["missing_percent"]
            
            if missing_pct > 0:
                if missing_pct > 50:
                    # Drop column if >50% missing
                    suggestions.append({
                        "id": f"step-{step_counter:02d}-drop-{col_name}",
                        "type": "drop_column",
                        "target_columns": [col_name],
                        "params": {},
                        "confidence": 0.95,
                        "rationale": f"Column '{col_name}' has {missing_pct:.1f}% missing values (>50%). Dropping to avoid bias.",
                        "console_preview": [
                            f"INFO {datetime.utcnow().isoformat()}Z | preprocess.drop_column | Analyzing column '{col_name}' -> {missing_pct:.1f}% missing",
                            f"INFO {datetime.utcnow().isoformat()}Z | preprocess.drop_column | Dropped column '{col_name}'"
                        ]
                    })
                    step_counter += 1
                elif col_info["type"] in ["int64", "float64"]:
                    # Numeric imputation
                    strategy = "median" if missing_pct > 10 else "mean"
                    suggestions.append({
                        "id": f"step-{step_counter:02d}-impute-{col_name}",
                        "type": "impute",
                        "target_columns": [col_name],
                        "params": {"strategy": strategy},
                        "confidence": 0.88,
                        "rationale": f"Numeric column '{col_name}' has {missing_pct:.1f}% missing. {strategy.capitalize()} imputation preserves distribution.",
                        "console_preview": [
                            f"INFO {datetime.utcnow().isoformat()}Z | preprocess.impute | Analyzing column '{col_name}' -> dtype={col_info['type']}, nulls={col_info['missing_count']} ({missing_pct:.1f}%)",
                            f"INFO {datetime.utcnow().isoformat()}Z | preprocess.impute | Imputed {col_info['missing_count']} missing values with {strategy}"
                        ]
                    })
                    step_counter += 1
                else:
                    # Categorical imputation
                    suggestions.append({
                        "id": f"step-{step_counter:02d}-impute-{col_name}",
                        "type": "impute",
                        "target_columns": [col_name],
                        "params": {"strategy": "most_frequent"},
                        "confidence": 0.85,
                        "rationale": f"Categorical column '{col_name}' has {missing_pct:.1f}% missing. Mode imputation is appropriate for categorical data.",
                        "console_preview": [
                            f"INFO {datetime.utcnow().isoformat()}Z | preprocess.impute | Analyzing column '{col_name}' -> dtype={col_info['type']}, nulls={col_info['missing_count']}",
                            f"INFO {datetime.utcnow().isoformat()}Z | preprocess.impute | Imputed with most frequent value"
                        ]
                    })
                    step_counter += 1
        
        # 2. Handle duplicates
        if profile["summary"]["duplicate_rows"] > 0:
            dup_pct = profile["summary"]["duplicate_rows"] / profile["rows"] * 100
            suggestions.append({
                "id": f"step-{step_counter:02d}-drop-duplicates",
                "type": "drop_duplicates",
                "target_columns": [],
                "params": {},
                "confidence": 0.92,
                "rationale": f"Found {profile['summary']['duplicate_rows']} duplicate rows ({dup_pct:.1f}%). Removing to avoid data leakage.",
                "console_preview": [
                    f"INFO {datetime.utcnow().isoformat()}Z | preprocess.drop_duplicates | Scanning for duplicates...",
                    f"INFO {datetime.utcnow().isoformat()}Z | preprocess.drop_duplicates | Removed {profile['summary']['duplicate_rows']} duplicate rows"
                ]
            })
            step_counter += 1
        
        # 3. Encode categorical variables
        categorical_cols = [
            col_info["name"] for col_info in profile["columns_info"]
            if col_info["type"] in ["object", "category"] and col_info["name"] != target_column
        ]
        
        for col_name in categorical_cols:
            col_info = next(c for c in profile["columns_info"] if c["name"] == col_name)
            cardinality = col_info["unique_count"]
            
            if cardinality <= 10:
                # One-hot encoding for low cardinality
                suggestions.append({
                    "id": f"step-{step_counter:02d}-onehot-{col_name}",
                    "type": "encode",
                    "target_columns": [col_name],
                    "params": {"method": "onehot"},
                    "confidence": 0.90,
                    "rationale": f"Low cardinality categorical column '{col_name}' ({cardinality} unique values). One-hot encoding suitable for tree-based and linear models.",
                    "console_preview": [
                        f"INFO {datetime.utcnow().isoformat()}Z | preprocess.encode | One-hot encoding column '{col_name}' with {cardinality} categories",
                        f"INFO {datetime.utcnow().isoformat()}Z | preprocess.encode | Created {cardinality} binary columns"
                    ]
                })
            else:
                # Label encoding for high cardinality
                suggestions.append({
                    "id": f"step-{step_counter:02d}-label-{col_name}",
                    "type": "encode",
                    "target_columns": [col_name],
                    "params": {"method": "label"},
                    "confidence": 0.82,
                    "rationale": f"High cardinality categorical column '{col_name}' ({cardinality} unique values). Label encoding to reduce dimensionality.",
                    "console_preview": [
                        f"INFO {datetime.utcnow().isoformat()}Z | preprocess.encode | Label encoding column '{col_name}' with {cardinality} categories",
                        f"INFO {datetime.utcnow().isoformat()}Z | preprocess.encode | Mapped categories to integers 0-{cardinality-1}"
                    ]
                })
            step_counter += 1
        
        # 4. Scale numeric features
        numeric_cols = [
            col_info["name"] for col_info in profile["columns_info"]
            if col_info["type"] in ["int64", "float64"] and col_info["name"] != target_column
        ]
        
        if numeric_cols:
            suggestions.append({
                "id": f"step-{step_counter:02d}-scale",
                "type": "scale",
                "target_columns": numeric_cols,
                "params": {"method": "standard"},
                "confidence": 0.87,
                "rationale": f"Standardizing {len(numeric_cols)} numeric features to mean=0, std=1 for optimal model performance.",
                "console_preview": [
                    f"INFO {datetime.utcnow().isoformat()}Z | preprocess.scale | Scaling {len(numeric_cols)} numeric columns",
                    f"INFO {datetime.utcnow().isoformat()}Z | preprocess.scale | Applied StandardScaler (mean=0, std=1)"
                ]
            })
            step_counter += 1
        
        return suggestions
    
    @staticmethod
    def suggest_models(df: pd.DataFrame, profile: Dict[str, Any], target_column: str, problem_type: str = "auto") -> List[Dict[str, Any]]:
        """Suggest appropriate models based on comprehensive dataset analysis."""
        suggestions = []
        
        # Determine problem type if auto
        if problem_type == "auto":
            if target_column:
                target_info = next((c for c in profile["columns_info"] if c["name"] == target_column), None)
                if target_info:
                    if target_info["type"] in ["int64", "float64"] and target_info["unique_count"] > 20:
                        problem_type = "regression"
                    else:
                        problem_type = "classification"
                else:
                    problem_type = "classification"
            else:
                problem_type = "classification"
        
        n_rows = profile["rows"]
        n_features = profile["columns"] - 1  # Excluding target
        
        # Analyze dataset complexity for smart recommendations
        analysis = SuggestionEngine.analyze_dataset_complexity(df, profile, target_column)
        
        # Define available models based on problem type
        if problem_type == "classification":
            models_config = [
                {
                    "model": "XGBoostClassifier",
                    "params": {
                        "n_estimators": min(200, max(50, n_rows // 100)),
                        "max_depth": 6 if n_rows < 10000 else 8,
                        "learning_rate": 0.1,
                        "objective": "multi:softmax" if analysis.get("is_multiclass") else "binary:logistic",
                        "eval_metric": "mlogloss" if analysis.get("is_multiclass") else "logloss"
                    },
                    "base_rationale": "XGBoost excels with tabular data, handles missing values natively, and provides feature importance."
                },
                {
                    "model": "RandomForestClassifier",
                    "params": {
                        "n_estimators": min(200, max(50, n_rows // 100)),
                        "max_depth": 10 if n_rows < 10000 else 15,
                        "min_samples_split": 5,
                        "n_jobs": -1
                    },
                    "base_rationale": "Random Forest is robust, interpretable, handles non-linear relationships, and rarely overfits."
                },
                {
                    "model": "LogisticRegression",
                    "params": {
                        "C": 1.0,
                        "max_iter": 1000,
                        "solver": "lbfgs",
                        "multi_class": "multinomial" if analysis.get("is_multiclass") else "auto"
                    },
                    "base_rationale": "Fast baseline model with high interpretability, best for linearly separable data."
                }
            ]
        else:  # Regression
            models_config = [
                {
                    "model": "XGBoostRegressor",
                    "params": {
                        "n_estimators": min(200, max(50, n_rows // 100)),
                        "max_depth": 6 if n_rows < 10000 else 8,
                        "learning_rate": 0.1
                    },
                    "base_rationale": "XGBoost provides excellent regression performance with automatic feature interactions."
                },
                {
                    "model": "RandomForestRegressor",
                    "params": {
                        "n_estimators": min(200, max(50, n_rows // 100)),
                        "max_depth": 10 if n_rows < 10000 else 15,
                        "n_jobs": -1
                    },
                    "base_rationale": "Random Forest handles non-linear relationships well and provides robust predictions."
                },
                {
                    "model": "LinearRegression",
                    "params": {},
                    "base_rationale": "Fast baseline with high interpretability for linear relationships."
                }
            ]
        
        # Score and rank models
        for model_config in models_config:
            model_name = model_config["model"]
            score = SuggestionEngine.calculate_model_score(model_name, analysis, problem_type)
            time_estimate = SuggestionEngine.estimate_training_time(
                model_name, n_rows, n_features, 
                model_config["params"].get("n_estimators", 100)
            )
            
            # Build dynamic rationale based on dataset
            rationale_parts = [model_config["base_rationale"]]
            
            if score >= 80:
                if analysis["is_large_dataset"]:
                    rationale_parts.append("Optimized for your large dataset.")
                if analysis["has_missing_values"]:
                    rationale_parts.append("Can handle your missing values.")
                if analysis.get("is_imbalanced") and "XGB" in model_name:
                    rationale_parts.append("Handles class imbalance well.")
            
            suggestions.append({
                "model": model_name,
                "params": model_config["params"],
                "confidence": round(score / 100, 2),
                "score": float(score),
                "rationale": " ".join(rationale_parts),
                "estimated_time_seconds": float(time_estimate["estimated_seconds"]),
                "estimated_time_minutes": float(time_estimate["estimated_minutes"]),
                "training_iterations": int(time_estimate["iterations"]),
                "training_steps": time_estimate["training_steps"],
                "is_recommended": False,  # Will be set later
                "recommendation_reason": "",
                "dataset_analysis": {
                    "samples": int(analysis["n_samples"]),
                    "features": int(analysis["n_features"]),
                    "is_imbalanced": bool(analysis.get("is_imbalanced", False)),
                    "n_classes": int(analysis.get("n_classes", 0)),
                    "problem_type": str(problem_type)
                }
            })
        
        # Sort by score (highest first) and mark the best one as recommended
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        if suggestions:
            best = suggestions[0]
            best["is_recommended"] = True
            
            # Generate recommendation reason
            reasons = []
            if analysis["is_large_dataset"]:
                reasons.append("large dataset size")
            elif analysis["is_small_dataset"]:
                reasons.append("small dataset size")
            else:
                reasons.append("medium dataset size")
                
            if analysis.get("is_imbalanced"):
                reasons.append("class imbalance handling")
            if analysis["has_missing_values"]:
                reasons.append("missing value support")
            if analysis["is_high_dimensional"]:
                reasons.append("high-dimensional data")
                
            best["recommendation_reason"] = f"Best for your data: {', '.join(reasons)}"
        
        return suggestions
