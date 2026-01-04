"""
Production Monitoring Script
Detects data drift and performance degradation.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from datetime import datetime


class DriftMonitor:
    """Monitor input data for drift against training baseline."""
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.baseline = self.config["baseline_statistics"]
        self.thresholds = self.config["drift_thresholds"]
    
    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Compute Population Stability Index."""
        def scale_range(arr, min_val, max_val, buckets):
            return np.clip(((arr - min_val) / (max_val - min_val) * buckets).astype(int), 0, buckets - 1)
        
        breakpoints = np.linspace(0, buckets, buckets + 1)
        
        expected_scaled = scale_range(expected, expected.min(), expected.max(), buckets)
        actual_scaled = scale_range(actual, actual.min(), actual.max(), buckets)
        
        expected_percents = np.histogram(expected_scaled, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual_scaled, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.clip(expected_percents, 0.0001, 1)
        actual_percents = np.clip(actual_percents, 0.0001, 1)
        
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi)
    
    def check_numeric_drift(self, feature: str, new_data: np.ndarray) -> Dict[str, Any]:
        """Check drift for numeric feature."""
        baseline = self.baseline.get(feature, {})
        if baseline.get("type") != "numeric":
            return {"status": "skipped", "reason": "not numeric"}
        
        # Compute KS test
        # Generate synthetic baseline from stats
        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]
        baseline_samples = np.random.normal(baseline_mean, baseline_std, 1000)
        
        ks_stat, ks_pvalue = stats.ks_2samp(baseline_samples, new_data)
        
        # Compute PSI
        psi = self.compute_psi(baseline_samples, new_data)
        
        # Check thresholds
        drift_detected = (
            psi > self.thresholds["psi_threshold"] or
            ks_pvalue < self.thresholds["ks_threshold"]
        )
        
        return {
            "feature": feature,
            "type": "numeric",
            "drift_detected": drift_detected,
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "new_mean": float(new_data.mean()),
            "new_std": float(new_data.std()),
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std
        }
    
    def check_categorical_drift(self, feature: str, new_data: pd.Series) -> Dict[str, Any]:
        """Check drift for categorical feature."""
        baseline = self.baseline.get(feature, {})
        if baseline.get("type") != "categorical":
            return {"status": "skipped", "reason": "not categorical"}
        
        baseline_dist = baseline.get("distribution", {})
        new_dist = new_data.value_counts(normalize=True).to_dict()
        
        # Compute chi-squared test
        all_categories = set(baseline_dist.keys()) | set(new_dist.keys())
        expected = [baseline_dist.get(c, 0.001) for c in all_categories]
        observed = [new_dist.get(c, 0.001) for c in all_categories]
        
        # Normalize
        expected = np.array(expected) / sum(expected)
        observed = np.array(observed) / sum(observed)
        
        chi2, pvalue = stats.chisquare(observed, expected)
        
        drift_detected = pvalue < self.thresholds["chi2_threshold"]
        
        return {
            "feature": feature,
            "type": "categorical",
            "drift_detected": drift_detected,
            "chi2_statistic": float(chi2),
            "chi2_pvalue": float(pvalue),
            "new_categories": list(new_dist.keys())[:10],
            "baseline_categories": list(baseline_dist.keys())[:10]
        }
    
    def check_all_features(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Check drift for all monitored features."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples": len(new_data),
            "features_checked": [],
            "drift_detected": False,
            "drifted_features": []
        }
        
        for feature in self.config["monitoring_features"]:
            if feature not in new_data.columns:
                continue
            
            col_data = new_data[feature].dropna()
            if len(col_data) == 0:
                continue
            
            if self.baseline.get(feature, {}).get("type") == "numeric":
                check_result = self.check_numeric_drift(feature, col_data.values)
            else:
                check_result = self.check_categorical_drift(feature, col_data)
            
            results["features_checked"].append(check_result)
            
            if check_result.get("drift_detected"):
                results["drift_detected"] = True
                results["drifted_features"].append(feature)
        
        return results


if __name__ == "__main__":
    # Example usage
    monitor = DriftMonitor()
    
    # Load new production data
    # new_data = pd.read_csv("production_data.csv")
    # results = monitor.check_all_features(new_data)
    # print(json.dumps(results, indent=2))
