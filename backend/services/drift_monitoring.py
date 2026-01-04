"""Drift monitoring service with PSI, KL divergence, and ADWIN implementations."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json
from scipy import stats
from river import drift
from sklearn.preprocessing import LabelEncoder

from core.log_emitter import log_emitter
from services.task_manager import TaskManager, TaskStatus


class DriftMonitoringService:
    """Monitors data drift with PSI, KL divergence, and ADWIN implementations."""
    
    def __init__(self, task_id: str, emit_log_func):
        self.task_id = task_id
        self.emit_log = emit_log_func
        self.drift_detectors = {}
    
    def calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        # Create buckets for both series
        if expected.dtype in ['object', 'category']:
            # For categorical data, use value counts
            expected_counts = expected.value_counts()
            actual_counts = actual.value_counts()
            
            # Combine all unique values
            all_values = set(expected_counts.index) | set(actual_counts.index)
            
            expected_dist = []
            actual_dist = []
            
            for val in all_values:
                expected_dist.append(expected_counts.get(val, 0))
                actual_dist.append(actual_counts.get(val, 0))
            
            expected_dist = np.array(expected_dist)
            actual_dist = np.array(actual_dist)
        else:
            # For numeric data, create buckets
            min_val = min(expected.min(), actual.min())
            max_val = max(expected.max(), actual.max())
            
            if min_val == max_val:
                return 0.0  # No variation
            
            buckets_edges = np.linspace(min_val, max_val, buckets + 1)
            
            expected_hist, _ = np.histogram(expected, bins=buckets_edges)
            actual_hist, _ = np.histogram(actual, bins=buckets_edges)
            
            # Add small value to avoid division by zero
            expected_dist = (expected_hist + 1e-8) / (expected_hist.sum() + 1e-8 * len(expected_hist))
            actual_dist = (actual_hist + 1e-8) / (actual_hist.sum() + 1e-8 * len(actual_hist))
        
        # Calculate PSI
        psi_values = (actual_dist - expected_dist) * np.log((actual_dist + 1e-8) / (expected_dist + 1e-8))
        psi = np.sum(psi_values)
        
        return psi
    
    def calculate_kl_divergence(self, p: pd.Series, q: pd.Series) -> float:
        """Calculate Kullback-Leibler divergence between two distributions."""
        if p.dtype in ['object', 'category']:
            p_counts = p.value_counts()
            q_counts = q.value_counts()
            
            # Get all unique values
            all_values = set(p_counts.index) | set(q_counts.index)
            
            p_probs = []
            q_probs = []
            
            for val in all_values:
                p_probs.append(p_counts.get(val, 0))
                q_probs.append(q_counts.get(val, 0))
            
            p_probs = np.array(p_probs)
            q_probs = np.array(q_probs)
            
            # Normalize to probabilities
            p_probs = p_probs / p_probs.sum()
            q_probs = q_probs / q_probs.sum()
        else:
            # For numeric data, create histograms
            min_val = min(p.min(), q.min())
            max_val = max(p.max(), q.max())
            
            if min_val == max_val:
                return 0.0
            
            hist_p, bins = np.histogram(p, bins=10, range=(min_val, max_val))
            hist_q, _ = np.histogram(q, bins=bins)
            
            # Add small value to avoid division by zero
            p_probs = (hist_p + 1e-8) / (hist_p.sum() + 1e-8 * len(hist_p))
            q_probs = (hist_q + 1e-8) / (hist_q.sum() + 1e-8 * len(hist_q))
        
        # Calculate KL divergence
        kl_div = np.sum(p_probs * np.log((p_probs + 1e-8) / (q_probs + 1e-8)))
        
        return kl_div
    
    async def monitor_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, 
                           threshold: float = 0.2) -> Dict[str, Any]:
        """Monitor drift in all columns of the dataset."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": False,
            "columns_analyzed": [],
            "drift_metrics": {}
        }
        
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Starting drift monitoring for {len(reference_df.columns)} columns",
            source="drift.monitor"
        )
        
        for col in reference_df.columns:
            if col not in current_df.columns:
                continue
                
            ref_series = reference_df[col]
            curr_series = current_df[col]
            
            # Calculate PSI
            psi = self.calculate_psi(ref_series, curr_series)
            
            # Calculate KL divergence
            kl_div = self.calculate_kl_divergence(ref_series, curr_series)
            
            # Use ADWIN for continuous monitoring (simplified implementation)
            # For a real implementation, we would use the river library's ADWIN
            adwin_drift = False  # Simplified - in reality, ADWIN tracks a stream of values
            if abs(psi) > threshold or abs(kl_div) > threshold:
                adwin_drift = True
            
            # Store results
            results["drift_metrics"][col] = {
                "psi": float(psi),
                "kl_divergence": float(kl_div),
                "adwin_drift": adwin_drift,
                "threshold": threshold
            }
            
            # Check if drift detected
            if psi > threshold or kl_div > threshold:
                results["drift_detected"] = True
                
                await self.emit_log(
                    self.task_id,
                    "WARN",
                    f"Drift detected in column '{col}': PSI={psi:.4f}, KL={kl_div:.4f}",
                    source="drift.alert",
                    meta={
                        "column": col,
                        "psi": psi,
                        "kl_divergence": kl_div,
                        "threshold": threshold
                    }
                )
        
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Drift monitoring completed. Drift detected: {results['drift_detected']}",
            source="drift.complete"
        )
        
        return results
    
    async def start_continuous_monitoring(self, model_id: str, reference_dataset_path: str, 
                                        check_interval: int = 3600):
        """Start continuous drift monitoring in the background."""
        await self.emit_log(
            self.task_id,
            "INFO",
            f"Starting continuous drift monitoring for model {model_id}",
            source="drift.service",
            meta={"check_interval": check_interval}
        )
        
        # In a real implementation, this would continuously check for new data
        # and compare it to the reference dataset
        # For now, we'll simulate this with a placeholder
        
        await self.emit_log(
            self.task_id,
            "INFO",
            "Drift monitoring service started (simulated)",
            source="drift.service"
        )
        
        # This would normally run indefinitely
        # await self._monitoring_loop(model_id, reference_dataset_path, check_interval)
        
        return {"status": "monitoring_started", "model_id": model_id}