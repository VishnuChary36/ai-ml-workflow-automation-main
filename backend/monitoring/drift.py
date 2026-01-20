"""
Data and Concept Drift Detection

Comprehensive drift detection using multiple statistical methods:
- Population Stability Index (PSI)
- Kullback-Leibler Divergence
- Jensen-Shannon Divergence
- Kolmogorov-Smirnov Test
- Chi-Square Test
- ADWIN (Adaptive Windowing)
- Page-Hinkley Test
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path
from scipy import stats
from collections import deque


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"
    TARGET_DRIFT = "target_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftMethod(str, Enum):
    """Statistical methods for drift detection."""
    PSI = "psi"
    KL_DIVERGENCE = "kl_divergence"
    JS_DIVERGENCE = "js_divergence"
    KS_TEST = "ks_test"
    CHI_SQUARE = "chi_square"
    ADWIN = "adwin"
    PAGE_HINKLEY = "page_hinkley"


class DriftResult:
    """Container for drift detection results."""
    
    def __init__(
        self,
        drift_detected: bool,
        drift_score: float,
        method: str,
        threshold: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.drift_detected = drift_detected
        self.drift_score = drift_score
        self.method = method
        self.threshold = threshold
        self.details = details or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "method": self.method,
            "threshold": self.threshold,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class ADWIN:
    """
    Adaptive Windowing algorithm for concept drift detection.
    
    Maintains a sliding window of recent values and detects
    when the mean of the window changes significantly.
    """
    
    def __init__(self, delta: float = 0.002):
        """
        Initialize ADWIN.
        
        Args:
            delta: Confidence parameter (lower = more sensitive)
        """
        self.delta = delta
        self.window = deque()
        self.sum = 0.0
        self.variance = 0.0
        self.width = 0
    
    def update(self, value: float) -> bool:
        """
        Add new value and check for drift.
        
        Args:
            value: New observation
            
        Returns:
            True if drift detected
        """
        self.window.append(value)
        self.sum += value
        self.width += 1
        
        if self.width < 10:
            return False
        
        # Check for drift
        return self._check_drift()
    
    def _check_drift(self) -> bool:
        """Check if drift occurred using ADWIN algorithm."""
        for i in range(1, self.width):
            # Split window
            left = list(self.window)[:i]
            right = list(self.window)[i:]
            
            if len(left) < 5 or len(right) < 5:
                continue
            
            mean_left = np.mean(left)
            mean_right = np.mean(right)
            
            # Calculate epsilon cut
            m = 1.0 / (1.0 / len(left) + 1.0 / len(right))
            epsilon = np.sqrt(
                (1.0 / (2 * m)) * np.log(4.0 / self.delta)
            )
            
            if abs(mean_left - mean_right) > epsilon:
                # Drift detected, shrink window
                for _ in range(i):
                    removed = self.window.popleft()
                    self.sum -= removed
                    self.width -= 1
                return True
        
        return False
    
    def reset(self):
        """Reset the detector."""
        self.window.clear()
        self.sum = 0.0
        self.variance = 0.0
        self.width = 0


class PageHinkley:
    """
    Page-Hinkley test for abrupt changes in mean.
    """
    
    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
    ):
        """
        Initialize Page-Hinkley detector.
        
        Args:
            delta: Minimum magnitude of change to detect
            threshold: Detection threshold
            alpha: Forgetting factor
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        
        self.sum = 0.0
        self.x_mean = 0.0
        self.count = 0
        self.min_sum = float("inf")
    
    def update(self, value: float) -> bool:
        """
        Add new value and check for drift.
        
        Args:
            value: New observation
            
        Returns:
            True if drift detected
        """
        self.count += 1
        
        # Update mean
        self.x_mean = self.x_mean + (value - self.x_mean) / self.count
        
        # Update sum
        self.sum = self.alpha * self.sum + (value - self.x_mean - self.delta)
        
        # Update minimum
        self.min_sum = min(self.min_sum, self.sum)
        
        # Check threshold
        return (self.sum - self.min_sum) > self.threshold
    
    def reset(self):
        """Reset the detector."""
        self.sum = 0.0
        self.x_mean = 0.0
        self.count = 0
        self.min_sum = float("inf")


class DriftDetector:
    """
    Comprehensive drift detection system.
    
    Features:
    - Multiple detection methods
    - Per-feature drift analysis
    - Threshold configuration
    - Historical tracking
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference/baseline data
            storage_path: Path for storing drift history
        """
        self.reference_data = reference_data
        self.storage_path = Path(
            storage_path or 
            os.getenv("DRIFT_STORAGE_PATH", "./drift_history")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Online detectors
        self._adwin_detectors: Dict[str, ADWIN] = {}
        self._ph_detectors: Dict[str, PageHinkley] = {}
        
        # Drift history
        self.history: List[Dict[str, Any]] = []
        
        # Default thresholds
        self.thresholds = {
            DriftMethod.PSI: 0.2,
            DriftMethod.KL_DIVERGENCE: 0.1,
            DriftMethod.JS_DIVERGENCE: 0.1,
            DriftMethod.KS_TEST: 0.05,  # p-value threshold
            DriftMethod.CHI_SQUARE: 0.05,  # p-value threshold
        }
    
    def set_reference(self, data: pd.DataFrame):
        """Set reference data for comparison."""
        self.reference_data = data.copy()
    
    def set_threshold(self, method: DriftMethod, threshold: float):
        """Set threshold for a detection method."""
        self.thresholds[method] = threshold
    
    def calculate_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
        buckets: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Slight change, monitor
        PSI >= 0.2: Significant change, action required
        """
        if expected.dtype in ['object', 'category']:
            # Categorical PSI
            expected_counts = expected.value_counts(normalize=True)
            actual_counts = actual.value_counts(normalize=True)
            
            all_categories = set(expected_counts.index) | set(actual_counts.index)
            
            psi = 0.0
            for cat in all_categories:
                expected_pct = expected_counts.get(cat, 0.0001)
                actual_pct = actual_counts.get(cat, 0.0001)
                
                expected_pct = max(expected_pct, 0.0001)
                actual_pct = max(actual_pct, 0.0001)
                
                psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            
            return psi
        else:
            # Numeric PSI
            min_val = min(expected.min(), actual.min())
            max_val = max(expected.max(), actual.max())
            
            if min_val == max_val:
                return 0.0
            
            bins = np.linspace(min_val, max_val, buckets + 1)
            
            expected_hist, _ = np.histogram(expected, bins=bins)
            actual_hist, _ = np.histogram(actual, bins=bins)
            
            # Normalize
            expected_pct = (expected_hist + 1) / (expected_hist.sum() + len(expected_hist))
            actual_pct = (actual_hist + 1) / (actual_hist.sum() + len(actual_hist))
            
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            
            return psi
    
    def calculate_kl_divergence(
        self,
        p: pd.Series,
        q: pd.Series,
        buckets: int = 10,
    ) -> float:
        """Calculate Kullback-Leibler divergence."""
        if p.dtype in ['object', 'category']:
            p_counts = p.value_counts(normalize=True)
            q_counts = q.value_counts(normalize=True)
            
            all_vals = set(p_counts.index) | set(q_counts.index)
            
            p_probs = np.array([p_counts.get(v, 0.0001) for v in all_vals])
            q_probs = np.array([q_counts.get(v, 0.0001) for v in all_vals])
        else:
            min_val = min(p.min(), q.min())
            max_val = max(p.max(), q.max())
            
            if min_val == max_val:
                return 0.0
            
            bins = np.linspace(min_val, max_val, buckets + 1)
            
            p_hist, _ = np.histogram(p, bins=bins)
            q_hist, _ = np.histogram(q, bins=bins)
            
            p_probs = (p_hist + 1) / (p_hist.sum() + len(p_hist))
            q_probs = (q_hist + 1) / (q_hist.sum() + len(q_hist))
        
        # KL divergence
        kl = np.sum(p_probs * np.log(p_probs / q_probs))
        
        return kl
    
    def calculate_js_divergence(
        self,
        p: pd.Series,
        q: pd.Series,
        buckets: int = 10,
    ) -> float:
        """Calculate Jensen-Shannon divergence."""
        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        if p.dtype in ['object', 'category']:
            p_counts = p.value_counts(normalize=True)
            q_counts = q.value_counts(normalize=True)
            
            all_vals = list(set(p_counts.index) | set(q_counts.index))
            
            p_probs = np.array([p_counts.get(v, 0.0001) for v in all_vals])
            q_probs = np.array([q_counts.get(v, 0.0001) for v in all_vals])
        else:
            min_val = min(p.min(), q.min())
            max_val = max(p.max(), q.max())
            
            if min_val == max_val:
                return 0.0
            
            bins = np.linspace(min_val, max_val, buckets + 1)
            
            p_hist, _ = np.histogram(p, bins=bins)
            q_hist, _ = np.histogram(q, bins=bins)
            
            p_probs = (p_hist + 1) / (p_hist.sum() + len(p_hist))
            q_probs = (q_hist + 1) / (q_hist.sum() + len(q_hist))
        
        m_probs = 0.5 * (p_probs + q_probs)
        
        kl_pm = np.sum(p_probs * np.log(p_probs / m_probs))
        kl_qm = np.sum(q_probs * np.log(q_probs / m_probs))
        
        return 0.5 * kl_pm + 0.5 * kl_qm
    
    def ks_test(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Returns:
            Tuple of (statistic, p_value)
        """
        if not pd.api.types.is_numeric_dtype(reference):
            raise ValueError("KS test requires numeric data")
        
        statistic, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
        return statistic, p_value
    
    def chi_square_test(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> Tuple[float, float]:
        """
        Perform Chi-Square test for categorical data.
        
        Returns:
            Tuple of (statistic, p_value)
        """
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        all_categories = list(set(ref_counts.index) | set(curr_counts.index))
        
        ref_freq = [ref_counts.get(c, 0) + 1 for c in all_categories]
        curr_freq = [curr_counts.get(c, 0) + 1 for c in all_categories]
        
        # Normalize to same scale
        ref_freq = np.array(ref_freq) * (sum(curr_freq) / sum(ref_freq))
        
        statistic, p_value = stats.chisquare(curr_freq, ref_freq)
        return statistic, p_value
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        methods: Optional[List[DriftMethod]] = None,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect drift in data.
        
        Args:
            current_data: Current data to compare
            methods: Detection methods to use
            columns: Specific columns to check
            
        Returns:
            Drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        methods = methods or [DriftMethod.PSI, DriftMethod.KS_TEST]
        columns = columns or list(self.reference_data.columns)
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": False,
            "columns_with_drift": [],
            "column_results": {},
            "overall_drift_score": 0.0,
        }
        
        drift_scores = []
        
        for col in columns:
            if col not in current_data.columns:
                continue
            
            ref_col = self.reference_data[col].dropna()
            curr_col = current_data[col].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                continue
            
            col_results = {
                "methods": {},
                "drift_detected": False,
            }
            
            for method in methods:
                try:
                    result = self._apply_method(method, ref_col, curr_col)
                    col_results["methods"][method.value] = result.to_dict()
                    
                    if result.drift_detected:
                        col_results["drift_detected"] = True
                        drift_scores.append(result.drift_score)
                        
                except Exception as e:
                    col_results["methods"][method.value] = {
                        "error": str(e)
                    }
            
            results["column_results"][col] = col_results
            
            if col_results["drift_detected"]:
                results["drift_detected"] = True
                results["columns_with_drift"].append(col)
        
        if drift_scores:
            results["overall_drift_score"] = np.mean(drift_scores)
        
        # Save to history
        self.history.append(results)
        self._save_history()
        
        return results
    
    def _apply_method(
        self,
        method: DriftMethod,
        reference: pd.Series,
        current: pd.Series,
    ) -> DriftResult:
        """Apply a specific drift detection method."""
        threshold = self.thresholds.get(method, 0.1)
        
        if method == DriftMethod.PSI:
            score = self.calculate_psi(reference, current)
            return DriftResult(
                drift_detected=score >= threshold,
                drift_score=score,
                method=method.value,
                threshold=threshold,
            )
        
        elif method == DriftMethod.KL_DIVERGENCE:
            score = self.calculate_kl_divergence(reference, current)
            return DriftResult(
                drift_detected=score >= threshold,
                drift_score=score,
                method=method.value,
                threshold=threshold,
            )
        
        elif method == DriftMethod.JS_DIVERGENCE:
            score = self.calculate_js_divergence(reference, current)
            return DriftResult(
                drift_detected=score >= threshold,
                drift_score=score,
                method=method.value,
                threshold=threshold,
            )
        
        elif method == DriftMethod.KS_TEST:
            if not pd.api.types.is_numeric_dtype(reference):
                raise ValueError("KS test requires numeric data")
            statistic, p_value = self.ks_test(reference, current)
            return DriftResult(
                drift_detected=p_value < threshold,
                drift_score=statistic,
                method=method.value,
                threshold=threshold,
                details={"p_value": p_value},
            )
        
        elif method == DriftMethod.CHI_SQUARE:
            statistic, p_value = self.chi_square_test(reference, current)
            return DriftResult(
                drift_detected=p_value < threshold,
                drift_score=statistic,
                method=method.value,
                threshold=threshold,
                details={"p_value": p_value},
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def update_online(
        self,
        feature_name: str,
        value: float,
    ) -> Tuple[bool, bool]:
        """
        Update online drift detectors.
        
        Args:
            feature_name: Feature name
            value: New observation
            
        Returns:
            Tuple of (ADWIN drift, Page-Hinkley drift)
        """
        # Initialize if needed
        if feature_name not in self._adwin_detectors:
            self._adwin_detectors[feature_name] = ADWIN()
            self._ph_detectors[feature_name] = PageHinkley()
        
        adwin_drift = self._adwin_detectors[feature_name].update(value)
        ph_drift = self._ph_detectors[feature_name].update(value)
        
        return adwin_drift, ph_drift
    
    def get_drift_report(
        self,
        last_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate drift report from history.
        
        Args:
            last_n: Number of recent checks to include
            
        Returns:
            Drift report
        """
        recent = self.history[-last_n:] if self.history else []
        
        report = {
            "total_checks": len(self.history),
            "recent_checks": len(recent),
            "drift_rate": 0.0,
            "most_drifted_columns": {},
            "timeline": [],
        }
        
        if not recent:
            return report
        
        # Calculate drift rate
        drift_count = sum(1 for r in recent if r.get("drift_detected", False))
        report["drift_rate"] = drift_count / len(recent)
        
        # Find most drifted columns
        column_drift_counts: Dict[str, int] = {}
        for check in recent:
            for col in check.get("columns_with_drift", []):
                column_drift_counts[col] = column_drift_counts.get(col, 0) + 1
        
        report["most_drifted_columns"] = dict(
            sorted(column_drift_counts.items(), key=lambda x: -x[1])[:10]
        )
        
        # Build timeline
        for check in recent:
            report["timeline"].append({
                "timestamp": check.get("timestamp"),
                "drift_detected": check.get("drift_detected"),
                "score": check.get("overall_drift_score"),
                "drifted_count": len(check.get("columns_with_drift", [])),
            })
        
        return report
    
    def _save_history(self):
        """Save drift history to disk."""
        history_file = self.storage_path / "drift_history.json"
        
        # Keep only last 1000 entries
        history_to_save = self.history[-1000:]
        
        with open(history_file, "w") as f:
            json.dump(history_to_save, f, indent=2, default=str)
    
    def _load_history(self):
        """Load drift history from disk."""
        history_file = self.storage_path / "drift_history.json"
        
        if history_file.exists():
            with open(history_file) as f:
                self.history = json.load(f)
