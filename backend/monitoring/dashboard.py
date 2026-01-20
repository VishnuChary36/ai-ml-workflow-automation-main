"""
Monitoring Dashboard

Aggregates monitoring data for visualization and reporting.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from .drift import DriftDetector
from .performance import PerformanceMonitor
from .alerting import AlertManager


class MonitoringDashboard:
    """
    Aggregates monitoring data for dashboards and reports.
    
    Features:
    - Real-time metrics
    - Historical trends
    - Model health scores
    - Alert summary
    """
    
    def __init__(
        self,
        drift_detector: Optional[DriftDetector] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        """
        Initialize dashboard.
        
        Args:
            drift_detector: Drift detection instance
            performance_monitor: Performance monitoring instance
            alert_manager: Alert management instance
        """
        self.drift_detector = drift_detector or DriftDetector()
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager or AlertManager()
    
    def get_overview(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get monitoring overview.
        
        Returns:
            Overview data for dashboard
        """
        overview = {
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": 100.0,
            "status": "healthy",
            "active_alerts": 0,
            "metrics": {},
            "drift": {},
            "performance": {},
        }
        
        # Alert summary
        active_alerts = self.alert_manager.get_active_alerts()
        overview["active_alerts"] = len(active_alerts)
        
        if len(active_alerts) > 0:
            critical = sum(1 for a in active_alerts if a.severity.value == "critical")
            if critical > 0:
                overview["status"] = "critical"
                overview["health_score"] -= 30 * critical
            else:
                overview["status"] = "warning"
                overview["health_score"] -= 10 * len(active_alerts)
        
        # Performance summary
        if self.performance_monitor:
            perf_summary = self.performance_monitor.get_summary()
            overview["performance"] = {
                "total_predictions": perf_summary.get("total_predictions", 0),
                "current_metrics": perf_summary.get("current_metrics", {}),
                "degradation_detected": perf_summary.get("degradation", {}).get("degradation_detected", False),
            }
            
            if overview["performance"]["degradation_detected"]:
                overview["health_score"] -= 20
        
        # Drift summary
        drift_report = self.drift_detector.get_drift_report()
        overview["drift"] = {
            "drift_rate": drift_report.get("drift_rate", 0),
            "most_drifted_columns": list(drift_report.get("most_drifted_columns", {}).keys())[:5],
        }
        
        if drift_report.get("drift_rate", 0) > 0.5:
            overview["health_score"] -= 15
        
        # Clamp health score
        overview["health_score"] = max(0, min(100, overview["health_score"]))
        
        if overview["health_score"] < 50:
            overview["status"] = "critical"
        elif overview["health_score"] < 80:
            overview["status"] = "warning"
        
        return overview
    
    def get_metrics_timeline(
        self,
        metric: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get metric timeline for charting.
        
        Args:
            metric: Metric name
            hours: Hours of history
            
        Returns:
            Timeline data
        """
        if not self.performance_monitor:
            return {"error": "No performance monitor configured"}
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        data_points = []
        for snapshot in self.performance_monitor.history:
            timestamp = snapshot.get("timestamp")
            if timestamp:
                ts = datetime.fromisoformat(timestamp)
                if ts > cutoff:
                    value = snapshot.get("metrics", {}).get(metric)
                    if value is not None:
                        data_points.append({
                            "timestamp": timestamp,
                            "value": value,
                        })
        
        return {
            "metric": metric,
            "hours": hours,
            "data_points": data_points,
            "count": len(data_points),
        }
    
    def get_drift_timeline(
        self,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get drift detection timeline.
        
        Args:
            hours: Hours of history
            
        Returns:
            Drift timeline data
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        data_points = []
        for check in self.drift_detector.history:
            timestamp = check.get("timestamp")
            if timestamp:
                ts = datetime.fromisoformat(timestamp)
                if ts > cutoff:
                    data_points.append({
                        "timestamp": timestamp,
                        "drift_detected": check.get("drift_detected", False),
                        "score": check.get("overall_drift_score", 0),
                        "drifted_columns": len(check.get("columns_with_drift", [])),
                    })
        
        return {
            "hours": hours,
            "data_points": data_points,
            "drift_rate": sum(1 for d in data_points if d["drift_detected"]) / max(len(data_points), 1),
        }
    
    def get_alert_timeline(
        self,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get alert timeline.
        
        Args:
            hours: Hours of history
            
        Returns:
            Alert timeline data
        """
        history = self.alert_manager.get_history(limit=500)
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        data_points = []
        for alert in history:
            fired_at = alert.get("fired_at")
            if fired_at:
                ts = datetime.fromisoformat(fired_at)
                if ts > cutoff:
                    data_points.append({
                        "timestamp": fired_at,
                        "severity": alert.get("severity"),
                        "name": alert.get("name"),
                        "status": alert.get("status"),
                    })
        
        return {
            "hours": hours,
            "data_points": data_points,
            "count": len(data_points),
            "by_severity": self._count_by_key(data_points, "severity"),
        }
    
    def _count_by_key(
        self,
        items: List[Dict],
        key: str,
    ) -> Dict[str, int]:
        """Count items by key value."""
        counts = {}
        for item in items:
            value = item.get(key)
            if value:
                counts[value] = counts.get(value, 0) + 1
        return counts
    
    def get_model_comparison(
        self,
        model_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare metrics across models.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Comparison data
        """
        # This would typically load metrics from storage for each model
        # Simplified implementation
        return {
            "models": model_ids,
            "comparison": {
                "metric": "accuracy",
                "values": {mid: 0.85 + 0.1 * hash(mid) % 10 / 100 for mid in model_ids},
            }
        }
    
    def generate_report(
        self,
        period_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Generate monitoring report.
        
        Args:
            period_hours: Report period in hours
            
        Returns:
            Comprehensive report
        """
        overview = self.get_overview()
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "period_hours": period_hours,
            "summary": {
                "health_score": overview["health_score"],
                "status": overview["status"],
                "active_alerts": overview["active_alerts"],
            },
            "performance": {},
            "drift": {},
            "alerts": {},
            "recommendations": [],
        }
        
        # Performance section
        if self.performance_monitor:
            summary = self.performance_monitor.get_summary()
            report["performance"] = {
                "total_predictions": summary.get("total_predictions", 0),
                "current_metrics": summary.get("current_metrics", {}),
                "trends": summary.get("trends", {}),
                "degradation": summary.get("degradation", {}),
            }
            
            if summary.get("degradation", {}).get("degradation_detected"):
                report["recommendations"].append(
                    "Model performance has degraded. Consider retraining."
                )
        
        # Drift section
        drift_report = self.drift_detector.get_drift_report()
        report["drift"] = {
            "total_checks": drift_report.get("total_checks", 0),
            "drift_rate": drift_report.get("drift_rate", 0),
            "most_drifted_columns": drift_report.get("most_drifted_columns", {}),
        }
        
        if drift_report.get("drift_rate", 0) > 0.3:
            report["recommendations"].append(
                "High drift rate detected. Review data pipeline for changes."
            )
        
        # Alerts section
        alert_stats = self.alert_manager.get_statistics()
        report["alerts"] = {
            "total_alerts": alert_stats.get("total_alerts", 0),
            "last_24h": alert_stats.get("last_24h", 0),
            "by_severity": alert_stats.get("by_severity", {}),
            "active": overview["active_alerts"],
        }
        
        if alert_stats.get("last_24h", 0) > 10:
            report["recommendations"].append(
                "High alert volume. Review alert thresholds."
            )
        
        return report
