"""
A/B Testing Router

Route predictions between multiple model versions for A/B testing
and canary deployments.
"""

import os
import json
import random
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .predictor import ModelPredictor


class ModelVariant:
    """Represents a model variant for A/B testing."""
    
    def __init__(
        self,
        variant_id: str,
        name: str,
        model_path: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.variant_id = variant_id
        self.name = name
        self.model_path = model_path
        self.weight = weight
        self.metadata = metadata or {}
        
        self.predictor = ModelPredictor(model_path)
        
        # Metrics
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        
        # Conversion tracking
        self.conversions = 0
        self.conversion_rate = 0.0


class ABTestingRouter:
    """
    A/B testing router for model predictions.
    
    Features:
    - Traffic splitting by weight
    - Consistent routing (same user -> same variant)
    - Real-time metrics per variant
    - Conversion tracking
    - Automatic winner detection
    """
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        config_path: Optional[str] = None,
    ):
        """
        Initialize A/B testing router.
        
        Args:
            experiment_id: Unique experiment ID
            name: Experiment name
            config_path: Path to save experiment config
        """
        self.experiment_id = experiment_id
        self.name = name
        self.config_path = Path(
            config_path or 
            os.getenv("AB_TEST_CONFIG_PATH", "./ab_tests")
        )
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        self.variants: Dict[str, ModelVariant] = {}
        self.control_variant_id: Optional[str] = None
        
        self.status = "draft"  # draft, running, completed
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        self._lock = threading.Lock()
    
    def add_variant(
        self,
        variant_id: str,
        name: str,
        model_path: str,
        weight: float = 1.0,
        is_control: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVariant:
        """
        Add a model variant to the experiment.
        
        Args:
            variant_id: Unique variant ID
            name: Variant name
            model_path: Path to model
            weight: Traffic weight (relative)
            is_control: Whether this is the control variant
            metadata: Additional metadata
            
        Returns:
            Created variant
        """
        variant = ModelVariant(
            variant_id=variant_id,
            name=name,
            model_path=model_path,
            weight=weight,
            metadata=metadata,
        )
        
        self.variants[variant_id] = variant
        
        if is_control:
            self.control_variant_id = variant_id
        
        print(f"✓ Added variant: {name} (weight={weight})")
        return variant
    
    def remove_variant(self, variant_id: str) -> bool:
        """Remove a variant from the experiment."""
        if variant_id in self.variants:
            del self.variants[variant_id]
            if self.control_variant_id == variant_id:
                self.control_variant_id = None
            return True
        return False
    
    def update_weight(self, variant_id: str, weight: float):
        """Update variant weight."""
        if variant_id in self.variants:
            self.variants[variant_id].weight = weight
    
    def start(self):
        """Start the experiment."""
        if not self.variants:
            raise ValueError("No variants configured")
        
        if self.control_variant_id is None:
            # Set first variant as control
            self.control_variant_id = list(self.variants.keys())[0]
        
        self.status = "running"
        self.started_at = datetime.utcnow()
        self._save_config()
        print(f"✓ Started A/B test: {self.name}")
    
    def stop(self):
        """Stop the experiment."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self._save_config()
        print(f"✓ Stopped A/B test: {self.name}")
    
    def _select_variant(
        self,
        user_id: Optional[str] = None,
    ) -> ModelVariant:
        """
        Select a variant for routing.
        
        Uses consistent hashing for user_id if provided,
        otherwise random weighted selection.
        """
        total_weight = sum(v.weight for v in self.variants.values())
        
        if user_id:
            # Consistent hashing - same user always gets same variant
            hash_val = int(hashlib.md5(
                f"{self.experiment_id}:{user_id}".encode()
            ).hexdigest(), 16)
            threshold = (hash_val % 1000) / 1000.0 * total_weight
        else:
            threshold = random.random() * total_weight
        
        cumulative = 0.0
        for variant in self.variants.values():
            cumulative += variant.weight
            if threshold <= cumulative:
                return variant
        
        # Fallback to last variant
        return list(self.variants.values())[-1]
    
    def predict(
        self,
        features: Dict[str, Any],
        user_id: Optional[str] = None,
        return_variant: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Route prediction through A/B test.
        
        Args:
            features: Input features
            user_id: User ID for consistent routing
            return_variant: Include variant info in response
            **kwargs: Additional args for predictor
            
        Returns:
            Prediction result with variant info
        """
        if self.status != "running":
            raise RuntimeError(f"Experiment not running: {self.status}")
        
        # Select variant
        variant = self._select_variant(user_id)
        
        with self._lock:
            variant.request_count += 1
        
        try:
            # Make prediction
            result = variant.predictor.predict(features, **kwargs)
            
            with self._lock:
                variant.success_count += 1
                variant.total_latency_ms += result.get("latency_ms", 0)
            
            if return_variant:
                result["variant_id"] = variant.variant_id
                result["variant_name"] = variant.name
                result["experiment_id"] = self.experiment_id
            
            return result
            
        except Exception as e:
            with self._lock:
                variant.error_count += 1
            raise
    
    def record_conversion(
        self,
        user_id: str,
        variant_id: Optional[str] = None,
    ):
        """
        Record a conversion for a user.
        
        Args:
            user_id: User ID
            variant_id: Variant ID (auto-detected if not provided)
        """
        if variant_id is None:
            # Determine variant from user_id
            variant = self._select_variant(user_id)
            variant_id = variant.variant_id
        
        if variant_id in self.variants:
            with self._lock:
                self.variants[variant_id].conversions += 1
                self._update_conversion_rates()
    
    def _update_conversion_rates(self):
        """Update conversion rates for all variants."""
        for variant in self.variants.values():
            if variant.success_count > 0:
                variant.conversion_rate = variant.conversions / variant.success_count
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get experiment results.
        
        Returns:
            Experiment results with per-variant metrics
        """
        with self._lock:
            results = {
                "experiment_id": self.experiment_id,
                "name": self.name,
                "status": self.status,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "control_variant": self.control_variant_id,
                "variants": {},
                "total_requests": 0,
                "winner": None,
            }
            
            for variant in self.variants.values():
                results["variants"][variant.variant_id] = {
                    "name": variant.name,
                    "weight": variant.weight,
                    "request_count": variant.request_count,
                    "success_count": variant.success_count,
                    "error_count": variant.error_count,
                    "error_rate": variant.error_count / max(variant.request_count, 1),
                    "avg_latency_ms": variant.total_latency_ms / max(variant.success_count, 1),
                    "conversions": variant.conversions,
                    "conversion_rate": variant.conversion_rate,
                }
                results["total_requests"] += variant.request_count
            
            # Determine winner
            if results["total_requests"] > 100:
                winner = self._determine_winner()
                if winner:
                    results["winner"] = winner
            
            return results
    
    def _determine_winner(self) -> Optional[str]:
        """
        Determine winning variant using statistical significance.
        
        Simple implementation - production should use proper
        statistical tests (chi-square, Bayesian, etc.)
        """
        if len(self.variants) < 2:
            return None
        
        control = self.variants.get(self.control_variant_id)
        if not control or control.success_count < 50:
            return None
        
        best_variant = None
        best_lift = 0.0
        
        for variant_id, variant in self.variants.items():
            if variant_id == self.control_variant_id:
                continue
            
            if variant.success_count < 50:
                continue
            
            # Calculate relative lift
            if control.conversion_rate > 0:
                lift = (variant.conversion_rate - control.conversion_rate) / control.conversion_rate
            else:
                lift = variant.conversion_rate
            
            if lift > best_lift and lift > 0.05:  # 5% minimum lift
                best_lift = lift
                best_variant = variant_id
        
        return best_variant
    
    def _save_config(self):
        """Save experiment config to disk."""
        config = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "control_variant_id": self.control_variant_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "name": v.name,
                    "model_path": v.model_path,
                    "weight": v.weight,
                    "metadata": v.metadata,
                }
                for v in self.variants.values()
            ],
        }
        
        config_file = self.config_path / f"{self.experiment_id}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(
        cls,
        experiment_id: str,
        config_path: Optional[str] = None,
    ) -> "ABTestingRouter":
        """
        Load experiment from disk.
        
        Args:
            experiment_id: Experiment ID to load
            config_path: Config directory path
            
        Returns:
            Loaded router
        """
        config_dir = Path(
            config_path or 
            os.getenv("AB_TEST_CONFIG_PATH", "./ab_tests")
        )
        config_file = config_dir / f"{experiment_id}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        
        with open(config_file) as f:
            config = json.load(f)
        
        router = cls(
            experiment_id=config["experiment_id"],
            name=config["name"],
            config_path=str(config_dir),
        )
        
        router.status = config["status"]
        router.control_variant_id = config["control_variant_id"]
        
        if config.get("started_at"):
            router.started_at = datetime.fromisoformat(config["started_at"])
        if config.get("completed_at"):
            router.completed_at = datetime.fromisoformat(config["completed_at"])
        
        for v in config["variants"]:
            router.add_variant(
                variant_id=v["variant_id"],
                name=v["name"],
                model_path=v["model_path"],
                weight=v["weight"],
                is_control=(v["variant_id"] == router.control_variant_id),
                metadata=v.get("metadata"),
            )
        
        return router
