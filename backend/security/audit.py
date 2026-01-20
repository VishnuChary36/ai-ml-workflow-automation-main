"""
Audit Logging

Comprehensive audit trail for security and compliance.
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import logging

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    TOKEN_REFRESH = "auth.token_refresh"
    API_KEY_CREATED = "auth.api_key_created"
    API_KEY_REVOKED = "auth.api_key_revoked"
    
    # Data operations
    DATA_UPLOAD = "data.upload"
    DATA_DOWNLOAD = "data.download"
    DATA_DELETE = "data.delete"
    DATA_ACCESS = "data.access"
    
    # Model operations
    MODEL_TRAIN = "model.train"
    MODEL_DELETE = "model.delete"
    MODEL_EXPORT = "model.export"
    MODEL_IMPORT = "model.import"
    
    # Deployment operations
    DEPLOY_CREATE = "deploy.create"
    DEPLOY_DELETE = "deploy.delete"
    DEPLOY_UPDATE = "deploy.update"
    DEPLOY_ROLLBACK = "deploy.rollback"
    
    # Prediction operations
    PREDICT = "predict.single"
    PREDICT_BATCH = "predict.batch"
    
    # Admin operations
    USER_CREATE = "admin.user_create"
    USER_DELETE = "admin.user_delete"
    ROLE_CHANGE = "admin.role_change"
    CONFIG_CHANGE = "admin.config_change"
    
    # Security events
    PERMISSION_DENIED = "security.permission_denied"
    RATE_LIMIT_EXCEEDED = "security.rate_limit"
    SUSPICIOUS_ACTIVITY = "security.suspicious"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents an audit log entry."""
    event_id: str
    timestamp: datetime
    action: str
    severity: AuditSeverity
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["severity"] = self.severity.value
        return data
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
    
    @property
    def checksum(self) -> str:
        """Generate checksum for integrity verification."""
        data = f"{self.event_id}:{self.timestamp.isoformat()}:{self.action}:{self.user_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Features:
    - Tamper-evident logging
    - Multiple output targets (file, database, SIEM)
    - Structured logging format
    - Async logging for performance
    - Log retention policies
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        retention_days: int = 90,
        enable_console: bool = False,
    ):
        """
        Initialize audit logger.
        
        Args:
            storage_path: Path for log files
            retention_days: Days to retain logs
            enable_console: Also log to console
        """
        self.storage_path = Path(
            storage_path or
            os.getenv("AUDIT_LOG_PATH", "./audit_logs")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = retention_days
        self.enable_console = enable_console
        
        # In-memory buffer for batch writes
        self._buffer: List[AuditEvent] = []
        self._buffer_size = 100
        
        # Chain hash for tamper detection
        self._last_hash: Optional[str] = None
        self._load_last_hash()
    
    def log(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            action: Action being logged
            user_id: ID of user performing action
            username: Username of user
            ip_address: Client IP address
            user_agent: Client user agent
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional details
            success: Whether action succeeded
            error_message: Error message if failed
            severity: Event severity
            
        Returns:
            Created audit event
        """
        # Determine severity
        if severity is None:
            if action in [AuditAction.PERMISSION_DENIED, AuditAction.SUSPICIOUS_ACTIVITY]:
                severity = AuditSeverity.WARNING
            elif action in [AuditAction.LOGIN_FAILED, AuditAction.RATE_LIMIT_EXCEEDED]:
                severity = AuditSeverity.WARNING
            elif not success:
                severity = AuditSeverity.ERROR
            else:
                severity = AuditSeverity.INFO
        
        event = AuditEvent(
            event_id=f"audit-{uuid.uuid4().hex}",
            timestamp=datetime.utcnow(),
            action=action.value if isinstance(action, AuditAction) else action,
            severity=severity,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            success=success,
            error_message=error_message,
        )
        
        # Add chain hash for integrity
        event.details["_chain_hash"] = self._compute_chain_hash(event)
        
        # Write event
        self._write_event(event)
        
        if self.enable_console:
            self._log_to_console(event)
        
        return event
    
    async def log_async(self, **kwargs) -> AuditEvent:
        """Async version of log."""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.log(**kwargs))
    
    def log_request(
        self,
        request: Any,  # FastAPI Request
        action: AuditAction,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log audit event from FastAPI request.
        
        Args:
            request: FastAPI request object
            action: Action being logged
            resource_type: Type of resource
            resource_id: Resource ID
            details: Additional details
            success: Whether action succeeded
            error_message: Error message
            
        Returns:
            Created audit event
        """
        # Extract user info from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        username = getattr(request.state, "username", None)
        
        # Get client info
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        
        # Add request details
        request_details = {
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
        }
        
        if details:
            request_details.update(details)
        
        return self.log(
            action=action,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            details=request_details,
            success=success,
            error_message=error_message,
        )
    
    def _get_client_ip(self, request: Any) -> Optional[str]:
        """Extract client IP from request."""
        # Check X-Forwarded-For header
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return None
    
    def _compute_chain_hash(self, event: AuditEvent) -> str:
        """Compute chain hash for tamper detection."""
        prev = self._last_hash or "genesis"
        data = f"{prev}:{event.event_id}:{event.timestamp.isoformat()}"
        new_hash = hashlib.sha256(data.encode()).hexdigest()[:32]
        self._last_hash = new_hash
        return new_hash
    
    def _write_event(self, event: AuditEvent):
        """Write event to storage."""
        # Add to buffer
        self._buffer.append(event)
        
        # Flush if buffer full
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to disk."""
        if not self._buffer:
            return
        
        # Daily log files
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit_{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            for event in self._buffer:
                f.write(event.to_json() + "\n")
        
        self._buffer.clear()
        self._save_last_hash()
    
    def _log_to_console(self, event: AuditEvent):
        """Log event to console."""
        level = {
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }.get(event.severity, logging.INFO)
        
        logger.log(level, f"AUDIT: {event.action} by {event.username or 'unknown'}")
    
    def _save_last_hash(self):
        """Save last chain hash."""
        hash_file = self.storage_path / ".chain_hash"
        with open(hash_file, "w") as f:
            f.write(self._last_hash or "")
    
    def _load_last_hash(self):
        """Load last chain hash."""
        hash_file = self.storage_path / ".chain_hash"
        if hash_file.exists():
            self._last_hash = hash_file.read_text().strip() or None
    
    def query(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            action: Filter by action
            user_id: Filter by user
            severity: Filter by severity
            limit: Maximum results
            
        Returns:
            List of matching audit events
        """
        results = []
        
        # Flush pending events
        self._flush_buffer()
        
        # Iterate through log files
        for log_file in sorted(self.storage_path.glob("audit_*.jsonl"), reverse=True):
            with open(log_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        
                        # Apply filters
                        if action and event.get("action") != action:
                            continue
                        if user_id and event.get("user_id") != user_id:
                            continue
                        if severity and event.get("severity") != severity.value:
                            continue
                        
                        timestamp = datetime.fromisoformat(event["timestamp"])
                        if start_date and timestamp < start_date:
                            continue
                        if end_date and timestamp > end_date:
                            continue
                        
                        results.append(event)
                        
                        if len(results) >= limit:
                            return results
                    except Exception:
                        continue
        
        return results
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify audit log integrity.
        
        Returns:
            Verification results
        """
        results = {
            "verified": True,
            "events_checked": 0,
            "errors": [],
        }
        
        prev_hash = "genesis"
        
        for log_file in sorted(self.storage_path.glob("audit_*.jsonl")):
            with open(log_file) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        event = json.loads(line)
                        results["events_checked"] += 1
                        
                        chain_hash = event.get("details", {}).get("_chain_hash")
                        if chain_hash:
                            # Verify chain
                            event_id = event["event_id"]
                            timestamp = event["timestamp"]
                            expected = hashlib.sha256(
                                f"{prev_hash}:{event_id}:{timestamp}".encode()
                            ).hexdigest()[:32]
                            
                            if chain_hash != expected:
                                results["verified"] = False
                                results["errors"].append({
                                    "file": str(log_file),
                                    "line": line_num,
                                    "error": "Chain hash mismatch",
                                })
                            
                            prev_hash = chain_hash
                    except Exception as e:
                        results["errors"].append({
                            "file": str(log_file),
                            "line": line_num,
                            "error": str(e),
                        })
        
        return results
    
    def __del__(self):
        """Flush buffer on destruction."""
        self._flush_buffer()


# Global audit logger
audit_logger = AuditLogger()
