"""
Rollback Management

Handles model and deployment rollbacks with safety checks.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class RollbackReason(str, Enum):
    """Reasons for rollback."""
    MANUAL = "manual"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_ERROR_RATE = "high_error_rate"
    DRIFT_DETECTED = "drift_detected"
    HEALTH_CHECK_FAILED = "health_check_failed"
    DEPLOYMENT_TIMEOUT = "deployment_timeout"
    OTHER = "other"


@dataclass
class RollbackRecord:
    """Record of a rollback operation."""
    rollback_id: str
    source_version: str
    target_version: str
    environment: str
    reason: RollbackReason
    initiated_by: str
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "environment": self.environment,
            "reason": self.reason.value,
            "initiated_by": self.initiated_by,
            "initiated_at": self.initiated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "details": self.details,
        }


class RollbackManager:
    """
    Manages model and deployment rollbacks.
    
    Features:
    - Safe rollback procedures
    - Automatic rollback triggers
    - Rollback history
    - Version tracking
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_rollback_depth: int = 5,
    ):
        """
        Initialize rollback manager.
        
        Args:
            storage_path: Path for storing rollback records
            max_rollback_depth: Maximum versions to go back
        """
        self.storage_path = Path(
            storage_path or
            os.getenv("ROLLBACK_STORAGE_PATH", "./rollbacks")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_rollback_depth = max_rollback_depth
        self.history: List[RollbackRecord] = []
        self._load_history()
        
        # Version tracking per environment
        self.version_history: Dict[str, List[str]] = {}
    
    def record_version(
        self,
        environment: str,
        version: str,
    ):
        """
        Record a deployed version.
        
        Args:
            environment: Deployment environment
            version: Deployed version
        """
        if environment not in self.version_history:
            self.version_history[environment] = []
        
        self.version_history[environment].append(version)
        
        # Keep only recent versions
        self.version_history[environment] = self.version_history[environment][-self.max_rollback_depth:]
    
    def get_rollback_targets(
        self,
        environment: str,
    ) -> List[str]:
        """
        Get available rollback targets for environment.
        
        Args:
            environment: Deployment environment
            
        Returns:
            List of version strings that can be rolled back to
        """
        history = self.version_history.get(environment, [])
        
        # Return all except current (last)
        if len(history) > 1:
            return list(reversed(history[:-1]))
        return []
    
    def can_rollback(
        self,
        environment: str,
    ) -> bool:
        """Check if rollback is possible."""
        return len(self.get_rollback_targets(environment)) > 0
    
    def initiate_rollback(
        self,
        environment: str,
        target_version: Optional[str] = None,
        reason: RollbackReason = RollbackReason.MANUAL,
        initiated_by: str = "system",
        details: Optional[Dict[str, Any]] = None,
    ) -> RollbackRecord:
        """
        Initiate a rollback.
        
        Args:
            environment: Target environment
            target_version: Version to rollback to (default: previous)
            reason: Reason for rollback
            initiated_by: User or system initiating rollback
            details: Additional details
            
        Returns:
            Rollback record
        """
        history = self.version_history.get(environment, [])
        
        if len(history) < 2:
            raise ValueError("No previous version to rollback to")
        
        current_version = history[-1]
        
        if target_version:
            if target_version not in history[:-1]:
                raise ValueError(f"Target version not in rollback history: {target_version}")
        else:
            target_version = history[-2]
        
        import uuid
        rollback_id = f"rb-{uuid.uuid4().hex[:8]}"
        
        record = RollbackRecord(
            rollback_id=rollback_id,
            source_version=current_version,
            target_version=target_version,
            environment=environment,
            reason=reason,
            initiated_by=initiated_by,
            details=details or {},
        )
        
        self.history.append(record)
        self._save_history()
        
        return record
    
    def complete_rollback(
        self,
        rollback_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> RollbackRecord:
        """
        Mark rollback as complete.
        
        Args:
            rollback_id: Rollback ID
            success: Whether rollback was successful
            error_message: Error message if failed
            
        Returns:
            Updated rollback record
        """
        record = None
        for r in self.history:
            if r.rollback_id == rollback_id:
                record = r
                break
        
        if not record:
            raise ValueError(f"Rollback not found: {rollback_id}")
        
        record.completed_at = datetime.utcnow()
        record.success = success
        
        if error_message:
            record.details["error_message"] = error_message
        
        if success:
            # Update version history
            env = record.environment
            if env in self.version_history:
                self.version_history[env].append(record.target_version)
        
        self._save_history()
        return record
    
    def get_rollback_history(
        self,
        environment: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get rollback history.
        
        Args:
            environment: Filter by environment
            limit: Maximum records to return
            
        Returns:
            List of rollback records
        """
        records = self.history
        
        if environment:
            records = [r for r in records if r.environment == environment]
        
        # Most recent first
        records = sorted(records, key=lambda x: x.initiated_at, reverse=True)
        
        return [r.to_dict() for r in records[:limit]]
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback statistics."""
        if not self.history:
            return {
                "total_rollbacks": 0,
                "success_rate": 0,
                "by_reason": {},
                "by_environment": {},
            }
        
        total = len(self.history)
        successful = sum(1 for r in self.history if r.success)
        
        by_reason: Dict[str, int] = {}
        by_env: Dict[str, int] = {}
        
        for r in self.history:
            reason = r.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_env[r.environment] = by_env.get(r.environment, 0) + 1
        
        return {
            "total_rollbacks": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_reason": by_reason,
            "by_environment": by_env,
        }
    
    def _save_history(self):
        """Save rollback history."""
        data = {
            "history": [r.to_dict() for r in self.history[-100:]],
            "version_history": self.version_history,
        }
        
        with open(self.storage_path / "rollback_history.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_history(self):
        """Load rollback history."""
        path = self.storage_path / "rollback_history.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                self.version_history = data.get("version_history", {})
                # Would reconstruct RollbackRecord objects here
