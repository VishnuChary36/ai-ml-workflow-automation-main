"""
Inference Metrics

Prometheus-compatible metrics collection for inference service.
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict
import json


class Counter:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, value: int = 1):
        with self._lock:
            self._value += value
    
    @property
    def value(self) -> int:
        return self._value
    
    def reset(self):
        with self._lock:
            self._value = 0


class Gauge:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        with self._lock:
            self._value = value
    
    def inc(self, value: float = 1.0):
        with self._lock:
            self._value += value
    
    def dec(self, value: float = 1.0):
        with self._lock:
            self._value -= value
    
    @property
    def value(self) -> float:
        return self._value


class Histogram:
    """Thread-safe histogram metric with buckets."""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        
        self._counts = {b: 0 for b in self.buckets}
        self._counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        with self._lock:
            self._sum += value
            self._count += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[bucket] += 1
                    break
            else:
                self._counts[float("inf")] += 1
    
    @property
    def sum(self) -> float:
        return self._sum
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum / self._count
    
    def get_percentile(self, percentile: float) -> float:
        """Approximate percentile from histogram."""
        if self._count == 0:
            return 0.0
        
        target = self._count * percentile
        cumulative = 0
        prev_bucket = 0.0
        
        for bucket in self.buckets + [float("inf")]:
            cumulative += self._counts[bucket]
            if cumulative >= target:
                # Linear interpolation
                return (bucket + prev_bucket) / 2
            prev_bucket = bucket
        
        return self.buckets[-1]


class InferenceMetrics:
    """
    Metrics collection for inference service.
    
    Features:
    - Request counts and rates
    - Latency histograms
    - Error tracking
    - Model-specific metrics
    - Prometheus export format
    """
    
    def __init__(self, namespace: str = "mlops"):
        """
        Initialize metrics collector.
        
        Args:
            namespace: Metric namespace prefix
        """
        self.namespace = namespace
        
        # Request metrics
        self.requests_total = Counter(
            f"{namespace}_inference_requests_total",
            "Total inference requests"
        )
        self.requests_success = Counter(
            f"{namespace}_inference_requests_success",
            "Successful inference requests"
        )
        self.requests_error = Counter(
            f"{namespace}_inference_requests_error",
            "Failed inference requests"
        )
        
        # Latency metrics
        self.latency = Histogram(
            f"{namespace}_inference_latency_seconds",
            "Inference latency in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        # Model metrics (per model)
        self.model_requests: Dict[str, Counter] = defaultdict(
            lambda: Counter(f"{namespace}_model_requests", "")
        )
        self.model_latency: Dict[str, Histogram] = {}
        
        # Batch metrics
        self.batch_size = Histogram(
            f"{namespace}_inference_batch_size",
            "Batch size distribution",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
        )
        
        # Active requests gauge
        self.active_requests = Gauge(
            f"{namespace}_inference_active_requests",
            "Currently processing requests"
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            f"{namespace}_cache_hits_total",
            "Cache hit count"
        )
        self.cache_misses = Counter(
            f"{namespace}_cache_misses_total",
            "Cache miss count"
        )
        
        # Error types
        self.error_types: Dict[str, Counter] = defaultdict(
            lambda: Counter(f"{namespace}_errors", "")
        )
        
        # Startup time
        self.startup_time = datetime.utcnow()
    
    def record_request(
        self,
        model_id: str,
        latency_seconds: float,
        success: bool = True,
        batch_size: int = 1,
        cached: bool = False,
        error_type: Optional[str] = None,
    ):
        """
        Record a prediction request.
        
        Args:
            model_id: Model identifier
            latency_seconds: Request latency
            success: Whether request succeeded
            batch_size: Number of predictions
            cached: Whether result was cached
            error_type: Type of error if failed
        """
        self.requests_total.inc()
        
        if success:
            self.requests_success.inc()
        else:
            self.requests_error.inc()
            if error_type:
                self.error_types[error_type].inc()
        
        self.latency.observe(latency_seconds)
        self.batch_size.observe(batch_size)
        
        # Model-specific
        self.model_requests[model_id].inc()
        
        if model_id not in self.model_latency:
            self.model_latency[model_id] = Histogram(
                f"{self.namespace}_model_{model_id}_latency",
                f"Latency for model {model_id}"
            )
        self.model_latency[model_id].observe(latency_seconds)
        
        # Cache
        if cached:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
    
    def request_started(self):
        """Record request start."""
        self.active_requests.inc()
    
    def request_completed(self):
        """Record request completion."""
        self.active_requests.dec()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Metrics summary dict
        """
        uptime = (datetime.utcnow() - self.startup_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "requests": {
                "total": self.requests_total.value,
                "success": self.requests_success.value,
                "error": self.requests_error.value,
                "active": self.active_requests.value,
                "success_rate": self.requests_success.value / max(self.requests_total.value, 1),
            },
            "latency": {
                "mean_ms": self.latency.mean * 1000,
                "p50_ms": self.latency.get_percentile(0.5) * 1000,
                "p95_ms": self.latency.get_percentile(0.95) * 1000,
                "p99_ms": self.latency.get_percentile(0.99) * 1000,
            },
            "cache": {
                "hits": self.cache_hits.value,
                "misses": self.cache_misses.value,
                "hit_rate": self.cache_hits.value / max(
                    self.cache_hits.value + self.cache_misses.value, 1
                ),
            },
            "models": {
                model_id: {
                    "requests": counter.value,
                    "latency_mean_ms": (
                        self.model_latency[model_id].mean * 1000
                        if model_id in self.model_latency else 0
                    ),
                }
                for model_id, counter in self.model_requests.items()
            },
            "errors": {
                error_type: counter.value
                for error_type, counter in self.error_types.items()
            },
        }
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        # Helper to add metric
        def add_metric(name: str, value: float, metric_type: str, help_text: str = ""):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {value}")
        
        # Request counters
        add_metric(
            self.requests_total.name,
            self.requests_total.value,
            "counter",
            self.requests_total.description
        )
        add_metric(
            self.requests_success.name,
            self.requests_success.value,
            "counter",
            self.requests_success.description
        )
        add_metric(
            self.requests_error.name,
            self.requests_error.value,
            "counter",
            self.requests_error.description
        )
        
        # Active requests gauge
        add_metric(
            self.active_requests.name,
            self.active_requests.value,
            "gauge",
            self.active_requests.description
        )
        
        # Latency histogram
        lines.append(f"# HELP {self.latency.name} {self.latency.description}")
        lines.append(f"# TYPE {self.latency.name} histogram")
        
        cumulative = 0
        for bucket in self.latency.buckets:
            cumulative += self.latency._counts[bucket]
            lines.append(f'{self.latency.name}_bucket{{le="{bucket}"}} {cumulative}')
        cumulative += self.latency._counts[float("inf")]
        lines.append(f'{self.latency.name}_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f"{self.latency.name}_sum {self.latency.sum}")
        lines.append(f"{self.latency.name}_count {self.latency.count}")
        
        # Cache metrics
        add_metric(
            self.cache_hits.name,
            self.cache_hits.value,
            "counter",
            self.cache_hits.description
        )
        add_metric(
            self.cache_misses.name,
            self.cache_misses.value,
            "counter",
            self.cache_misses.description
        )
        
        # Model-specific metrics
        for model_id, counter in self.model_requests.items():
            lines.append(
                f'{self.namespace}_model_requests_total{{model_id="{model_id}"}} {counter.value}'
            )
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics."""
        self.requests_total.reset()
        self.requests_success.reset()
        self.requests_error.reset()
        self.cache_hits.reset()
        self.cache_misses.reset()
        self.model_requests.clear()
        self.model_latency.clear()
        self.error_types.clear()


# Global metrics instance
inference_metrics = InferenceMetrics()
