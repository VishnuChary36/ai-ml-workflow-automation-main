"""Dashboard Service - Generates comprehensive data analytics insights for storytelling dashboards."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


class DashboardService:
    """
    Generates Power BI / Excel-style dashboard data with storytelling insights.
    Designed for both senior and beginner data analysts.
    """
    
    def __init__(self, task_id: str = None):
        self.task_id = task_id
    
    def generate_dashboard(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data with insights and visualizations.
        
        Returns a storytelling dashboard with:
        - Executive Summary
        - Key Performance Indicators (KPIs)
        - Data Distribution Charts
        - Correlation Analysis
        - Trend Analysis (if applicable)
        - Categorical Breakdowns
        - Insights and Recommendations
        """
        dashboard = {
            "generated_at": datetime.utcnow().isoformat(),
            "dataset_info": self._get_dataset_info(df),
            "executive_summary": self._generate_executive_summary(df, target_column),
            "kpis": self._calculate_kpis(df, target_column),
            "data_quality": self._analyze_data_quality(df),
            "distributions": self._analyze_distributions(df),
            "correlations": self._analyze_correlations(df),
            "categorical_analysis": self._analyze_categorical(df, target_column),
            "numerical_analysis": self._analyze_numerical(df, target_column),
            "trends": self._analyze_trends(df),
            "insights": self._generate_insights(df, target_column),
            "recommendations": self._generate_recommendations(df, target_column),
            "charts": self._prepare_chart_data(df, target_column)
        }
        
        # Convert all numpy types to Python native types for JSON serialization
        return convert_to_native(dashboard)
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "columns": df.columns.tolist()
        }
    
    def _generate_executive_summary(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Generate executive summary with key findings."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Key statistics
        summary_stats = {}
        for col in numeric_cols[:5]:  # Top 5 numeric columns
            summary_stats[col] = {
                "mean": round(df[col].mean(), 2) if not df[col].isnull().all() else None,
                "median": round(df[col].median(), 2) if not df[col].isnull().all() else None,
                "std": round(df[col].std(), 2) if not df[col].isnull().all() else None
            }
        
        # Target analysis if specified
        target_info = None
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                target_info = {
                    "type": "classification",
                    "classes": df[target_column].nunique(),
                    "distribution": df[target_column].value_counts().head(10).to_dict()
                }
            else:
                target_info = {
                    "type": "regression",
                    "mean": round(df[target_column].mean(), 2),
                    "std": round(df[target_column].std(), 2),
                    "range": [round(df[target_column].min(), 2), round(df[target_column].max(), 2)]
                }
        
        # Generate narrative
        narrative = self._create_narrative(df, target_column)
        
        return {
            "headline": f"Dataset Overview: {len(df):,} Records with {len(df.columns)} Features",
            "narrative": narrative,
            "key_statistics": summary_stats,
            "target_analysis": target_info,
            "data_completeness": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
        }
    
    def _create_narrative(self, df: pd.DataFrame, target_column: str = None) -> str:
        """Create a storytelling narrative about the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        narrative = f"This dataset contains {len(df):,} records with {len(df.columns)} features. "
        
        if len(numeric_cols) > 0:
            narrative += f"There are {len(numeric_cols)} numerical features "
        if len(categorical_cols) > 0:
            narrative += f"and {len(categorical_cols)} categorical features. "
        
        # Data quality narrative
        missing_pct = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
        if missing_pct == 0:
            narrative += "The data is complete with no missing values. "
        elif missing_pct < 5:
            narrative += f"Data quality is excellent with only {missing_pct}% missing values. "
        elif missing_pct < 20:
            narrative += f"There are {missing_pct}% missing values that have been handled during preprocessing. "
        else:
            narrative += f"Note: {missing_pct}% of data points were missing and have been addressed. "
        
        # Target narrative
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                classes = df[target_column].nunique()
                most_common = df[target_column].mode().iloc[0] if not df[target_column].mode().empty else "N/A"
                narrative += f"The target variable '{target_column}' has {classes} unique classes, with '{most_common}' being the most common. "
            else:
                mean_val = df[target_column].mean()
                narrative += f"The target variable '{target_column}' has a mean of {mean_val:.2f}. "
        
        return narrative
    
    def _calculate_kpis(self, df: pd.DataFrame, target_column: str = None) -> List[Dict[str, Any]]:
        """Calculate key performance indicators."""
        kpis = []
        
        # Total Records
        kpis.append({
            "name": "Total Records",
            "value": f"{len(df):,}",
            "icon": "database",
            "color": "blue",
            "description": "Total number of data points in the dataset"
        })
        
        # Data Completeness
        completeness = round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
        kpis.append({
            "name": "Data Completeness",
            "value": f"{completeness}%",
            "icon": "check-circle",
            "color": "green" if completeness >= 95 else "yellow" if completeness >= 80 else "red",
            "description": "Percentage of non-missing values"
        })
        
        # Features
        kpis.append({
            "name": "Features",
            "value": str(len(df.columns)),
            "icon": "layers",
            "color": "purple",
            "description": "Total number of features/columns"
        })
        
        # Numeric Features
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        kpis.append({
            "name": "Numeric Features",
            "value": str(numeric_count),
            "icon": "hash",
            "color": "indigo",
            "description": "Number of numerical columns"
        })
        
        # Categorical Features
        cat_count = len(df.select_dtypes(include=['object', 'category']).columns)
        kpis.append({
            "name": "Categorical Features",
            "value": str(cat_count),
            "icon": "tag",
            "color": "pink",
            "description": "Number of categorical columns"
        })
        
        # Target Distribution (if classification)
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                # Class balance
                value_counts = df[target_column].value_counts()
                if len(value_counts) >= 2:
                    balance = round(value_counts.min() / value_counts.max() * 100, 1)
                    kpis.append({
                        "name": "Class Balance",
                        "value": f"{balance}%",
                        "icon": "pie-chart",
                        "color": "orange" if balance < 50 else "green",
                        "description": "Ratio of minority to majority class"
                    })
        
        return kpis
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        quality_scores = []
        
        for col in df.columns:
            missing_pct = round(df[col].isnull().sum() / len(df) * 100, 2)
            unique_pct = round(df[col].nunique() / len(df) * 100, 2)
            
            # Calculate quality score
            score = 100 - missing_pct
            if df[col].dtype == 'object':
                # Check for potential issues in text columns
                if df[col].str.len().max() > 1000 if hasattr(df[col], 'str') else False:
                    score -= 10
            
            quality_scores.append({
                "column": col,
                "missing_pct": missing_pct,
                "unique_values": df[col].nunique(),
                "unique_pct": unique_pct,
                "quality_score": round(max(0, score), 1),
                "dtype": str(df[col].dtype)
            })
        
        # Overall quality
        overall_score = round(np.mean([q["quality_score"] for q in quality_scores]), 1)
        
        return {
            "overall_score": overall_score,
            "rating": "Excellent" if overall_score >= 95 else "Good" if overall_score >= 80 else "Fair" if overall_score >= 60 else "Needs Improvement",
            "columns": quality_scores
        }
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distributions for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
                
            data = df[col].dropna()
            
            # Calculate histogram bins
            try:
                hist, bin_edges = np.histogram(data, bins=20)
                histogram = {
                    "bins": [round(float(b), 4) for b in bin_edges[:-1]],
                    "counts": [int(c) for c in hist],
                    "bin_labels": [f"{round(bin_edges[i], 2)}-{round(bin_edges[i+1], 2)}" for i in range(len(bin_edges)-1)]
                }
            except:
                histogram = None
            
            # Calculate statistics
            distributions[col] = {
                "min": round(float(data.min()), 4),
                "max": round(float(data.max()), 4),
                "mean": round(float(data.mean()), 4),
                "median": round(float(data.median()), 4),
                "std": round(float(data.std()), 4),
                "skewness": round(float(data.skew()), 4) if len(data) > 2 else 0,
                "kurtosis": round(float(data.kurtosis()), 4) if len(data) > 3 else 0,
                "q1": round(float(data.quantile(0.25)), 4),
                "q3": round(float(data.quantile(0.75)), 4),
                "iqr": round(float(data.quantile(0.75) - data.quantile(0.25)), 4),
                "histogram": histogram,
                "distribution_type": self._detect_distribution_type(data)
            }
        
        return distributions
    
    def _detect_distribution_type(self, data: pd.Series) -> str:
        """Detect the type of distribution."""
        skewness = data.skew() if len(data) > 2 else 0
        
        if abs(skewness) < 0.5:
            return "Normal (Symmetric)"
        elif skewness > 0.5:
            return "Right-skewed (Positive)"
        else:
            return "Left-skewed (Negative)"
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}
        
        # Calculate correlation matrix
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Convert to serializable format
            correlations = {}
            for col in corr_matrix.columns:
                correlations[col] = {c: round(float(corr_matrix.loc[col, c]), 4) for c in corr_matrix.columns}
            
            # Find top correlations
            top_correlations = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr_val):
                        top_correlations.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": round(float(corr_val), 4),
                            "strength": "Strong" if abs(corr_val) >= 0.7 else "Moderate" if abs(corr_val) >= 0.4 else "Weak",
                            "direction": "Positive" if corr_val > 0 else "Negative"
                        })
            
            # Sort by absolute correlation
            top_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "matrix": correlations,
                "columns": list(numeric_cols),
                "top_correlations": top_correlations[:10],
                "heatmap_data": [[round(float(corr_matrix.iloc[i, j]), 4) for j in range(len(numeric_cols))] for i in range(len(numeric_cols))]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_categorical(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Analyze categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        analysis = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            # Get top categories
            top_categories = value_counts.head(10)
            
            # Calculate percentages
            percentages = (top_categories / len(df) * 100).round(2)
            
            analysis[col] = {
                "unique_values": int(df[col].nunique()),
                "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "top_value_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "categories": [
                    {"name": str(cat), "count": int(count), "percentage": float(pct)}
                    for cat, count, pct in zip(top_categories.index, top_categories.values, percentages.values)
                ],
                "chart_data": {
                    "labels": [str(x) for x in top_categories.index.tolist()],
                    "values": [int(x) for x in top_categories.values.tolist()],
                    "percentages": [float(x) for x in percentages.values.tolist()]
                }
            }
            
            # Cross-tabulation with target if applicable
            if target_column and target_column in df.columns and col != target_column:
                if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                    try:
                        crosstab = pd.crosstab(df[col], df[target_column], normalize='index') * 100
                        analysis[col]["target_distribution"] = crosstab.round(2).to_dict()
                    except:
                        pass
        
        return analysis
    
    def _analyze_numerical(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Detailed analysis of numerical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        analysis = {}
        
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
            
            data = df[col].dropna()
            
            # Detect outliers using IQR
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Box plot data
            box_plot = {
                "min": float(data.min()),
                "q1": float(q1),
                "median": float(data.median()),
                "q3": float(q3),
                "max": float(data.max()),
                "lower_fence": float(max(data.min(), lower_bound)),
                "upper_fence": float(min(data.max(), upper_bound))
            }
            
            analysis[col] = {
                "statistics": {
                    "count": int(len(data)),
                    "mean": round(float(data.mean()), 4),
                    "std": round(float(data.std()), 4),
                    "min": round(float(data.min()), 4),
                    "25%": round(float(q1), 4),
                    "50%": round(float(data.median()), 4),
                    "75%": round(float(q3), 4),
                    "max": round(float(data.max()), 4)
                },
                "outliers": {
                    "count": int(len(outliers)),
                    "percentage": round(len(outliers) / len(data) * 100, 2),
                    "lower_bound": round(float(lower_bound), 4),
                    "upper_bound": round(float(upper_bound), 4)
                },
                "box_plot": box_plot
            }
            
            # Target relationship if applicable
            if target_column and target_column in df.columns and col != target_column:
                if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                    try:
                        grouped = df.groupby(target_column)[col].agg(['mean', 'median', 'std']).round(4)
                        analysis[col]["by_target"] = grouped.to_dict('index')
                    except:
                        pass
        
        return analysis
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends if datetime columns exist."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(datetime_cols) == 0:
            # Try to detect date columns from object type
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(df[col].head(100))
                    datetime_cols = pd.Index([col])
                    break
                except:
                    continue
        
        if len(datetime_cols) == 0:
            return {"message": "No datetime columns detected for trend analysis"}
        
        trends = {}
        date_col = datetime_cols[0]
        
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp = df_temp.sort_values(date_col)
            
            for num_col in numeric_cols[:5]:  # Limit to 5 columns
                if df_temp[num_col].isnull().all():
                    continue
                
                # Resample to daily/monthly based on data range
                date_range = (df_temp[date_col].max() - df_temp[date_col].min()).days
                
                if date_range > 365:
                    freq = 'M'
                    freq_name = 'Monthly'
                elif date_range > 30:
                    freq = 'W'
                    freq_name = 'Weekly'
                else:
                    freq = 'D'
                    freq_name = 'Daily'
                
                try:
                    resampled = df_temp.set_index(date_col)[num_col].resample(freq).mean().dropna()
                    
                    trends[num_col] = {
                        "frequency": freq_name,
                        "data_points": [
                            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
                            for d, v in zip(resampled.index, resampled.values)
                        ][-50:],  # Last 50 points
                        "trend_direction": "Increasing" if resampled.iloc[-1] > resampled.iloc[0] else "Decreasing",
                        "change_pct": round((resampled.iloc[-1] - resampled.iloc[0]) / abs(resampled.iloc[0]) * 100, 2) if resampled.iloc[0] != 0 else 0
                    }
                except:
                    continue
                    
        except Exception as e:
            return {"error": str(e)}
        
        return trends
    
    def _generate_insights(self, df: pd.DataFrame, target_column: str = None) -> List[Dict[str, Any]]:
        """Generate actionable insights from the data."""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Insight 1: Data Quality
        missing_cols = df.columns[df.isnull().any()].tolist()
        if len(missing_cols) == 0:
            insights.append({
                "type": "success",
                "category": "Data Quality",
                "title": "Excellent Data Quality",
                "description": "Your dataset has no missing values after preprocessing. This ensures reliable analysis and model training.",
                "icon": "check-circle"
            })
        
        # Insight 2: Feature Importance Hint
        if len(numeric_cols) > 0 and target_column and target_column in df.columns:
            try:
                corr_with_target = df[numeric_cols].corrwith(df[target_column].astype(float)).abs().sort_values(ascending=False)
                top_features = corr_with_target.head(3).index.tolist()
                if len(top_features) > 0:
                    insights.append({
                        "type": "info",
                        "category": "Feature Analysis",
                        "title": "Top Correlated Features",
                        "description": f"The features most correlated with '{target_column}' are: {', '.join(top_features)}. These may be strong predictors.",
                        "icon": "trending-up"
                    })
            except:
                pass
        
        # Insight 3: Outlier Detection
        outlier_cols = []
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_pct = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum() / len(df) * 100
            if outlier_pct > 5:
                outlier_cols.append((col, round(outlier_pct, 1)))
        
        if outlier_cols:
            insights.append({
                "type": "warning",
                "category": "Outliers",
                "title": "Outliers Detected",
                "description": f"Columns with notable outliers: {', '.join([f'{c[0]} ({c[1]}%)' for c in outlier_cols[:3]])}. Consider reviewing these for data errors or genuine extreme values.",
                "icon": "alert-triangle"
            })
        
        # Insight 4: Class Imbalance
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                value_counts = df[target_column].value_counts()
                if len(value_counts) >= 2:
                    imbalance_ratio = value_counts.min() / value_counts.max()
                    if imbalance_ratio < 0.3:
                        insights.append({
                            "type": "warning",
                            "category": "Target Variable",
                            "title": "Class Imbalance Detected",
                            "description": f"The target variable shows class imbalance (ratio: {imbalance_ratio:.2f}). Consider using techniques like SMOTE, class weights, or stratified sampling.",
                            "icon": "pie-chart"
                        })
                    else:
                        insights.append({
                            "type": "success",
                            "category": "Target Variable",
                            "title": "Balanced Classes",
                            "description": f"The target variable classes are reasonably balanced (ratio: {imbalance_ratio:.2f}). This is good for model training.",
                            "icon": "check-circle"
                        })
        
        # Insight 5: High Cardinality
        high_card_cols = []
        for col in categorical_cols:
            if df[col].nunique() > 50:
                high_card_cols.append((col, df[col].nunique()))
        
        if high_card_cols:
            insights.append({
                "type": "info",
                "category": "Categorical Features",
                "title": "High Cardinality Columns",
                "description": f"Columns with many unique values: {', '.join([f'{c[0]} ({c[1]} values)' for c in high_card_cols[:3]])}. Consider grouping rare categories or using target encoding.",
                "icon": "layers"
            })
        
        # Insight 6: Correlation Insights
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        corr_val = abs(corr_matrix.loc[col1, col2])
                        if corr_val > 0.8:
                            high_corr_pairs.append((col1, col2, round(corr_val, 2)))
                
                if high_corr_pairs:
                    insights.append({
                        "type": "info",
                        "category": "Multicollinearity",
                        "title": "Highly Correlated Features",
                        "description": f"Found {len(high_corr_pairs)} feature pairs with correlation > 0.8. Consider removing redundant features: {', '.join([f'{p[0]}-{p[1]}' for p in high_corr_pairs[:3]])}.",
                        "icon": "link"
                    })
            except:
                pass
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame, target_column: str = None) -> List[Dict[str, Any]]:
        """Generate recommendations for next steps."""
        recommendations = []
        
        # Recommendation based on data size
        if len(df) < 1000:
            recommendations.append({
                "priority": "medium",
                "category": "Data Collection",
                "title": "Consider More Data",
                "description": "With less than 1,000 samples, consider collecting more data for robust model training, especially for complex models.",
                "action": "Collect more samples or use simpler models"
            })
        elif len(df) > 100000:
            recommendations.append({
                "priority": "low",
                "category": "Performance",
                "title": "Large Dataset Optimization",
                "description": "With over 100,000 samples, consider using sampling for initial exploration and incremental learning for model training.",
                "action": "Use sampling or distributed processing"
            })
        
        # Model recommendations
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                n_classes = df[target_column].nunique()
                if n_classes == 2:
                    recommendations.append({
                        "priority": "high",
                        "category": "Model Selection",
                        "title": "Binary Classification Task",
                        "description": "Recommended models: Logistic Regression (baseline), Random Forest, XGBoost, or Neural Networks for complex patterns.",
                        "action": "Start with Logistic Regression, then try ensemble methods"
                    })
                else:
                    recommendations.append({
                        "priority": "high",
                        "category": "Model Selection",
                        "title": "Multi-class Classification Task",
                        "description": f"With {n_classes} classes, consider Random Forest, XGBoost, or Neural Networks. Use macro/weighted F1 for evaluation.",
                        "action": "Use stratified cross-validation for reliable metrics"
                    })
            else:
                recommendations.append({
                    "priority": "high",
                    "category": "Model Selection",
                    "title": "Regression Task",
                    "description": "Recommended models: Linear Regression (baseline), Random Forest Regressor, XGBoost, or Gradient Boosting.",
                    "action": "Start with Linear Regression, evaluate with RMSE and R²"
                })
        
        # Feature engineering recommendation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 5:
            recommendations.append({
                "priority": "medium",
                "category": "Feature Engineering",
                "title": "Feature Interactions",
                "description": "With multiple numeric features, consider creating interaction terms or polynomial features for potential improvements.",
                "action": "Try PolynomialFeatures or manual feature creation"
            })
        
        return recommendations
    
    def _prepare_chart_data(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Prepare data for various chart types."""
        charts = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # 1. Bar charts for categorical columns
        bar_charts = {}
        for col in categorical_cols[:5]:  # Limit to 5
            value_counts = df[col].value_counts().head(10)
            bar_charts[col] = {
                "labels": [str(x) for x in value_counts.index.tolist()],
                "values": [int(x) for x in value_counts.values.tolist()],
                "type": "bar"
            }
        charts["bar_charts"] = bar_charts
        
        # 2. Pie chart for target (if categorical)
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                value_counts = df[target_column].value_counts()
                charts["target_pie"] = {
                    "labels": [str(x) for x in value_counts.index.tolist()],
                    "values": [int(x) for x in value_counts.values.tolist()],
                    "type": "pie"
                }
        
        # 3. Histograms for numeric columns
        histograms = {}
        for col in numeric_cols[:5]:
            if df[col].isnull().all():
                continue
            try:
                hist, bin_edges = np.histogram(df[col].dropna(), bins=20)
                histograms[col] = {
                    "bins": [round(float(b), 4) for b in bin_edges[:-1]],
                    "counts": [int(c) for c in hist],
                    "type": "histogram"
                }
            except:
                continue
        charts["histograms"] = histograms
        
        # 4. Scatter plots for top correlated pairs
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                scatter_plots = []
                
                # Find top 3 correlated pairs
                pairs = []
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        if not np.isnan(corr_matrix.loc[col1, col2]):
                            pairs.append((col1, col2, abs(corr_matrix.loc[col1, col2])))
                
                pairs.sort(key=lambda x: x[2], reverse=True)
                
                for col1, col2, corr in pairs[:3]:
                    # Sample data if too large
                    sample_df = df[[col1, col2]].dropna()
                    if len(sample_df) > 500:
                        sample_df = sample_df.sample(500)
                    
                    scatter_plots.append({
                        "x_column": col1,
                        "y_column": col2,
                        "correlation": round(corr, 4),
                        "x_values": [round(float(x), 4) for x in sample_df[col1].values],
                        "y_values": [round(float(x), 4) for x in sample_df[col2].values],
                        "type": "scatter"
                    })
                
                charts["scatter_plots"] = scatter_plots
            except:
                charts["scatter_plots"] = []
        
        # 5. Box plots for numeric columns
        box_plots = {}
        for col in numeric_cols[:5]:
            if df[col].isnull().all():
                continue
            data = df[col].dropna()
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            box_plots[col] = {
                "min": round(float(data.min()), 4),
                "q1": round(float(q1), 4),
                "median": round(float(data.median()), 4),
                "q3": round(float(q3), 4),
                "max": round(float(data.max()), 4),
                "type": "boxplot"
            }
        charts["box_plots"] = box_plots
        
        return charts
