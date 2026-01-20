"""
Deployment Management

Handles model deployment strategies and lifecycle.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DeploymentStrategy(str, Enum):
    """Deployment strategies."""
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TEST = "ab_test"


class DeploymentStatus(str, Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    model_id: str
    model_version: str
    environment: str
    strategy: DeploymentStrategy = DeploymentStrategy.CANARY
    canary_percentage: int = 10
    replicas: int = 2
    cpu_limit: str = "1"
    memory_limit: str = "2Gi"
    health_check_path: str = "/health"
    ready_check_path: str = "/ready"
    rollout_timeout_seconds: int = 600
    auto_rollback: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "environment": self.environment,
            "strategy": self.strategy.value,
            "canary_percentage": self.canary_percentage,
            "replicas": self.replicas,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "health_check_path": self.health_check_path,
            "ready_check_path": self.ready_check_path,
            "rollout_timeout_seconds": self.rollout_timeout_seconds,
            "auto_rollback": self.auto_rollback,
            "metadata": self.metadata,
        }


@dataclass
class Deployment:
    """Deployment record."""
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rollback_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "rollback_at": self.rollback_at.isoformat() if self.rollback_at else None,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class DeploymentManager:
    """
    Manages model deployments across environments.
    
    Features:
    - Multiple deployment strategies
    - Canary deployments with traffic control
    - Blue-green deployments
    - Automatic rollback
    - Deployment history
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize deployment manager.
        
        Args:
            storage_path: Path for storing deployment records
        """
        self.storage_path = Path(
            storage_path or
            os.getenv("DEPLOYMENT_STORAGE_PATH", "./deployments")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.deployments: Dict[str, Deployment] = {}
        self.active_deployments: Dict[str, str] = {}  # environment -> deployment_id
        
        self._load_deployments()
    
    def create_deployment(
        self,
        config: DeploymentConfig,
    ) -> Deployment:
        """
        Create a new deployment.
        
        Args:
            config: Deployment configuration
            
        Returns:
            Created deployment
        """
        deployment_id = f"deploy-{uuid.uuid4().hex[:12]}"
        
        deployment = Deployment(
            deployment_id=deployment_id,
            config=config,
        )
        
        self.deployments[deployment_id] = deployment
        self._save_deployments()
        
        return deployment
    
    def start_deployment(
        self,
        deployment_id: str,
    ) -> Deployment:
        """
        Start a pending deployment.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Updated deployment
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        
        if deployment.status != DeploymentStatus.PENDING:
            raise ValueError(f"Deployment not in PENDING state: {deployment.status}")
        
        deployment.status = DeploymentStatus.IN_PROGRESS
        deployment.started_at = datetime.utcnow()
        
        # Execute deployment strategy
        try:
            if deployment.config.strategy == DeploymentStrategy.CANARY:
                self._deploy_canary(deployment)
            elif deployment.config.strategy == DeploymentStrategy.BLUE_GREEN:
                self._deploy_blue_green(deployment)
            elif deployment.config.strategy == DeploymentStrategy.ROLLING:
                self._deploy_rolling(deployment)
            else:
                self._deploy_recreate(deployment)
            
            deployment.status = DeploymentStatus.ACTIVE
            deployment.completed_at = datetime.utcnow()
            
            # Mark previous deployment as superseded
            prev_deployment_id = self.active_deployments.get(deployment.config.environment)
            if prev_deployment_id and prev_deployment_id != deployment_id:
                prev = self.deployments.get(prev_deployment_id)
                if prev:
                    prev.status = DeploymentStatus.SUPERSEDED
            
            self.active_deployments[deployment.config.environment] = deployment_id
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            
            if deployment.config.auto_rollback:
                self._auto_rollback(deployment)
        
        self._save_deployments()
        return deployment
    
    def _deploy_canary(self, deployment: Deployment):
        """Execute canary deployment."""
        config = deployment.config
        
        # Phase 1: Deploy canary with limited traffic
        deployment.metrics["canary_started"] = datetime.utcnow().isoformat()
        deployment.metrics["canary_percentage"] = config.canary_percentage
        
        # Simulate canary deployment
        # In production, this would:
        # 1. Deploy new version alongside existing
        # 2. Route canary_percentage traffic to new version
        # 3. Monitor error rates and latency
        # 4. Gradually increase traffic if healthy
        
        # Phase 2: Monitor canary (would be async in production)
        deployment.metrics["canary_monitoring"] = "completed"
        
        # Phase 3: Promote to full deployment
        deployment.metrics["canary_promoted"] = datetime.utcnow().isoformat()
    
    def _deploy_blue_green(self, deployment: Deployment):
        """Execute blue-green deployment."""
        # 1. Deploy to inactive environment (green)
        # 2. Run health checks
        # 3. Switch traffic from blue to green
        # 4. Keep blue running for quick rollback
        
        deployment.metrics["blue_green_switch"] = datetime.utcnow().isoformat()
    
    def _deploy_rolling(self, deployment: Deployment):
        """Execute rolling deployment."""
        config = deployment.config
        
        # Update pods one at a time
        for i in range(config.replicas):
            deployment.metrics[f"replica_{i}_updated"] = datetime.utcnow().isoformat()
    
    def _deploy_recreate(self, deployment: Deployment):
        """Execute recreate deployment (full replacement)."""
        # 1. Scale down existing deployment
        # 2. Deploy new version
        # 3. Scale up
        
        deployment.metrics["recreate_completed"] = datetime.utcnow().isoformat()
    
    def _auto_rollback(self, deployment: Deployment):
        """Perform automatic rollback."""
        env = deployment.config.environment
        
        # Find previous successful deployment
        prev_id = None
        for d in sorted(self.deployments.values(), key=lambda x: x.created_at, reverse=True):
            if (d.config.environment == env and 
                d.status in [DeploymentStatus.ACTIVE, DeploymentStatus.SUPERSEDED] and
                d.deployment_id != deployment.deployment_id):
                prev_id = d.deployment_id
                break
        
        if prev_id:
            deployment.metrics["rollback_to"] = prev_id
            self.active_deployments[env] = prev_id
            self.deployments[prev_id].status = DeploymentStatus.ACTIVE
    
    def rollback(
        self,
        environment: str,
        target_deployment_id: Optional[str] = None,
    ) -> Deployment:
        """
        Rollback to a previous deployment.
        
        Args:
            environment: Target environment
            target_deployment_id: Specific deployment to rollback to
            
        Returns:
            Rolled back deployment
        """
        current_id = self.active_deployments.get(environment)
        
        if not current_id:
            raise ValueError(f"No active deployment for environment: {environment}")
        
        # Find target
        if target_deployment_id:
            if target_deployment_id not in self.deployments:
                raise ValueError(f"Target deployment not found: {target_deployment_id}")
            target = self.deployments[target_deployment_id]
        else:
            # Find previous successful deployment
            target = None
            for d in sorted(self.deployments.values(), key=lambda x: x.created_at, reverse=True):
                if (d.config.environment == environment and
                    d.status == DeploymentStatus.SUPERSEDED and
                    d.deployment_id != current_id):
                    target = d
                    break
            
            if not target:
                raise ValueError("No previous deployment to rollback to")
        
        # Execute rollback
        current = self.deployments[current_id]
        current.status = DeploymentStatus.ROLLED_BACK
        current.rollback_at = datetime.utcnow()
        
        target.status = DeploymentStatus.ACTIVE
        self.active_deployments[environment] = target.deployment_id
        
        self._save_deployments()
        return target
    
    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment by ID."""
        return self.deployments.get(deployment_id)
    
    def get_active_deployment(self, environment: str) -> Optional[Deployment]:
        """Get active deployment for environment."""
        deployment_id = self.active_deployments.get(environment)
        if deployment_id:
            return self.deployments.get(deployment_id)
        return None
    
    def list_deployments(
        self,
        environment: Optional[str] = None,
        status: Optional[DeploymentStatus] = None,
        limit: int = 50,
    ) -> List[Deployment]:
        """List deployments with optional filters."""
        deployments = list(self.deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d.config.environment == environment]
        
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        # Sort by creation time, newest first
        deployments.sort(key=lambda x: x.created_at, reverse=True)
        
        return deployments[:limit]
    
    def get_deployment_history(
        self,
        environment: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get deployment history for environment."""
        deployments = self.list_deployments(environment=environment, limit=limit)
        return [d.to_dict() for d in deployments]
    
    def _save_deployments(self):
        """Save deployments to disk."""
        data = {
            "deployments": {k: v.to_dict() for k, v in self.deployments.items()},
            "active": self.active_deployments,
        }
        
        with open(self.storage_path / "deployments.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_deployments(self):
        """Load deployments from disk."""
        path = self.storage_path / "deployments.json"
        if path.exists():
            # Load would reconstruct Deployment objects
            # Simplified: just load active deployments
            pass
