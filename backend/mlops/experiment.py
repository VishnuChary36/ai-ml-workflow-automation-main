"""
Experiment Management

Provides experiment tracking, comparison, and analysis capabilities
for ML experiments.
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import numpy as np

from .tracking import mlflow_tracker


class ExperimentManager:
    """
    Manage ML experiments with tracking, comparison, and analysis.
    
    Features:
    - Create and manage experiments
    - Track runs with parameters and metrics
    - Compare experiments and runs
    - Generate experiment reports
    - Hyperparameter analysis
    """
    
    def __init__(self, experiments_path: Optional[str] = None):
        """
        Initialize experiment manager.
        
        Args:
            experiments_path: Path to store experiment data
        """
        self.experiments_path = Path(
            experiments_path or 
            os.getenv("EXPERIMENTS_PATH", "./experiments")
        )
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        
        self.tracker = mlflow_tracker
        
        # Local experiment store
        self.index_file = self.experiments_path / "experiments_index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load experiments index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {"experiments": {}, "runs": {}}
    
    def _save_index(self):
        """Save experiments index."""
        with open(self.index_file, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
    
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Experiment tags
            
        Returns:
            Experiment info dict
        """
        experiment_id = str(uuid.uuid4())
        
        experiment_info = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "run_count": 0,
            "best_run_id": None,
            "best_metric": None,
            "status": "active",
        }
        
        # Create experiment directory
        exp_dir = self.experiments_path / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(experiment_info, f, indent=2)
        
        # Update index
        self._index["experiments"][name] = experiment_info
        self._index["runs"][name] = []
        self._save_index()
        
        # Create in MLflow if available
        if self.tracker.is_available:
            self.tracker.create_experiment(name, tags=tags)
        
        print(f"✓ Created experiment: {name}")
        return experiment_info
    
    def log_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Log a training run to an experiment.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Human-readable run name
            params: Training parameters
            metrics: Evaluation metrics
            artifacts: Artifact paths {name: path}
            tags: Run tags
            model_path: Path to saved model
            
        Returns:
            Run info dict
        """
        # Ensure experiment exists
        if experiment_name not in self._index["experiments"]:
            self.create_experiment(experiment_name)
        
        run_id = str(uuid.uuid4())
        run_name = run_name or f"run-{run_id[:8]}"
        
        run_info = {
            "run_id": run_id,
            "run_name": run_name,
            "experiment_name": experiment_name,
            "params": params or {},
            "metrics": metrics or {},
            "tags": tags or {},
            "artifacts": artifacts or {},
            "model_path": model_path,
            "created_at": datetime.utcnow().isoformat(),
            "status": "completed",
        }
        
        # Save run data
        exp_dir = self.experiments_path / experiment_name
        run_dir = exp_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(run_dir / "run.json", "w") as f:
            json.dump(run_info, f, indent=2)
        
        # Update experiment index
        self._index["runs"][experiment_name].append(run_info)
        self._index["experiments"][experiment_name]["run_count"] += 1
        self._index["experiments"][experiment_name]["updated_at"] = datetime.utcnow().isoformat()
        
        # Update best run if applicable
        self._update_best_run(experiment_name, run_info)
        
        self._save_index()
        
        # Log to MLflow if available
        if self.tracker.is_available:
            with self.tracker.start_run(
                experiment_name=experiment_name,
                run_name=run_name,
                tags=tags,
            ) as ctx:
                if params:
                    ctx.log_params(params)
                if metrics:
                    ctx.log_metrics(metrics)
                if artifacts:
                    for name, path in artifacts.items():
                        if os.path.exists(path):
                            ctx.log_artifact(path, name)
        
        print(f"✓ Logged run: {run_name} to {experiment_name}")
        return run_info
    
    def _update_best_run(
        self,
        experiment_name: str,
        run_info: Dict[str, Any],
        metric_name: str = "accuracy",
        higher_is_better: bool = True,
    ):
        """Update best run for experiment based on metric."""
        exp = self._index["experiments"][experiment_name]
        current_metric = run_info.get("metrics", {}).get(metric_name)
        
        if current_metric is None:
            return
        
        best_metric = exp.get("best_metric")
        
        should_update = (
            best_metric is None or
            (higher_is_better and current_metric > best_metric) or
            (not higher_is_better and current_metric < best_metric)
        )
        
        if should_update:
            exp["best_run_id"] = run_info["run_id"]
            exp["best_metric"] = current_metric
            exp["best_metric_name"] = metric_name
    
    def get_experiment(self, name: str) -> Optional[Dict[str, Any]]:
        """Get experiment info by name."""
        return self._index["experiments"].get(name)
    
    def list_experiments(
        self,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Args:
            status: Filter by status (active, archived)
            
        Returns:
            List of experiment info dicts
        """
        experiments = list(self._index["experiments"].values())
        
        if status:
            experiments = [e for e in experiments if e.get("status") == status]
        
        return sorted(experiments, key=lambda x: x["updated_at"], reverse=True)
    
    def get_runs(
        self,
        experiment_name: str,
        limit: Optional[int] = None,
        order_by: str = "created_at",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get runs for an experiment.
        
        Args:
            experiment_name: Experiment name
            limit: Maximum number of runs
            order_by: Field to sort by
            ascending: Sort order
            
        Returns:
            List of run info dicts
        """
        runs = self._index["runs"].get(experiment_name, [])
        
        # Sort runs
        runs = sorted(
            runs,
            key=lambda x: x.get(order_by, ""),
            reverse=not ascending
        )
        
        if limit:
            runs = runs[:limit]
        
        return runs
    
    def compare_runs(
        self,
        experiment_name: str,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
        params: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple runs from an experiment.
        
        Args:
            experiment_name: Experiment name
            run_ids: List of run IDs to compare
            metrics: Metrics to compare (all if None)
            params: Parameters to compare (all if None)
            
        Returns:
            Comparison results
        """
        runs = self._index["runs"].get(experiment_name, [])
        run_map = {r["run_id"]: r for r in runs}
        
        comparison = {
            "experiment_name": experiment_name,
            "runs": [],
            "metrics_comparison": {},
            "params_comparison": {},
        }
        
        # Collect all metrics and params
        all_metrics = set()
        all_params = set()
        
        for run_id in run_ids:
            run = run_map.get(run_id)
            if not run:
                continue
            
            comparison["runs"].append({
                "run_id": run_id,
                "run_name": run.get("run_name"),
                "created_at": run.get("created_at"),
            })
            
            all_metrics.update(run.get("metrics", {}).keys())
            all_params.update(run.get("params", {}).keys())
        
        # Filter metrics/params if specified
        if metrics:
            all_metrics = all_metrics & set(metrics)
        if params:
            all_params = all_params & set(params)
        
        # Build comparison tables
        for metric in all_metrics:
            comparison["metrics_comparison"][metric] = {}
            for run_id in run_ids:
                run = run_map.get(run_id)
                if run:
                    comparison["metrics_comparison"][metric][run_id] = \
                        run.get("metrics", {}).get(metric)
        
        for param in all_params:
            comparison["params_comparison"][param] = {}
            for run_id in run_ids:
                run = run_map.get(run_id)
                if run:
                    comparison["params_comparison"][param][run_id] = \
                        run.get("params", {}).get(param)
        
        return comparison
    
    def get_best_run(
        self,
        experiment_name: str,
        metric_name: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run for an experiment based on a metric.
        
        Args:
            experiment_name: Experiment name
            metric_name: Metric to optimize
            higher_is_better: Whether higher metric is better
            
        Returns:
            Best run info or None
        """
        runs = self._index["runs"].get(experiment_name, [])
        
        if not runs:
            return None
        
        # Filter runs with the metric
        valid_runs = [
            r for r in runs 
            if metric_name in r.get("metrics", {})
        ]
        
        if not valid_runs:
            return None
        
        # Find best
        return max(
            valid_runs,
            key=lambda x: x["metrics"][metric_name] if higher_is_better 
                else -x["metrics"][metric_name]
        )
    
    def generate_report(
        self,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """
        Generate a summary report for an experiment.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Report dict with statistics and insights
        """
        exp = self._index["experiments"].get(experiment_name)
        runs = self._index["runs"].get(experiment_name, [])
        
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        report = {
            "experiment_name": experiment_name,
            "description": exp.get("description"),
            "created_at": exp.get("created_at"),
            "total_runs": len(runs),
            "best_run": None,
            "metrics_summary": {},
            "params_distribution": {},
            "timeline": [],
        }
        
        if not runs:
            return report
        
        # Collect all metrics
        all_metrics = {}
        for run in runs:
            for metric, value in run.get("metrics", {}).items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Compute metric statistics
        for metric, values in all_metrics.items():
            report["metrics_summary"][metric] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "count": len(values),
            }
        
        # Get best run
        report["best_run"] = self.get_best_run(experiment_name)
        
        # Build timeline
        for run in sorted(runs, key=lambda x: x.get("created_at", "")):
            report["timeline"].append({
                "run_id": run["run_id"],
                "run_name": run.get("run_name"),
                "created_at": run.get("created_at"),
                "primary_metric": run.get("metrics", {}).get("accuracy"),
            })
        
        return report
    
    def delete_experiment(self, name: str) -> bool:
        """
        Delete an experiment and all its runs.
        
        Args:
            name: Experiment name
            
        Returns:
            True if deleted
        """
        if name not in self._index["experiments"]:
            return False
        
        # Archive instead of delete
        self._index["experiments"][name]["status"] = "archived"
        self._index["experiments"][name]["updated_at"] = datetime.utcnow().isoformat()
        
        self._save_index()
        
        print(f"✓ Archived experiment: {name}")
        return True


# Global experiment manager instance
experiment_manager = ExperimentManager()
