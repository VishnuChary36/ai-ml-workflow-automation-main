"""
Alerting System

Configurable alerting for model monitoring with multiple channels
and escalation policies.
"""

import os
import json
import uuid
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from pathlib import Path
import threading

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


class AlertChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    LOG = "log"


class AlertRule:
    """
    Defines conditions for triggering alerts.
    """
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown_minutes: int = 30,
        channels: Optional[List[AlertChannel]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize alert rule.
        
        Args:
            rule_id: Unique rule identifier
            name: Human-readable rule name
            condition: Function that returns True when alert should fire
            severity: Alert severity level
            cooldown_minutes: Minimum time between alerts
            channels: Notification channels
            description: Rule description
            metadata: Additional metadata
        """
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.channels = channels or [AlertChannel.LOG]
        self.description = description
        self.metadata = metadata or {}
        
        self.enabled = True
        self.last_fired: Optional[datetime] = None
        self.fire_count = 0
    
    def should_fire(self, data: Dict[str, Any]) -> bool:
        """Check if rule should fire based on condition and cooldown."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_fired:
            cooldown = timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() - self.last_fired < cooldown:
                return False
        
        # Check condition
        try:
            return self.condition(data)
        except Exception:
            return False
    
    def fire(self):
        """Mark rule as fired."""
        self.last_fired = datetime.utcnow()
        self.fire_count += 1


class Alert:
    """
    Represents a triggered alert.
    """
    
    def __init__(
        self,
        alert_id: str,
        rule_id: str,
        name: str,
        severity: AlertSeverity,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.alert_id = alert_id
        self.rule_id = rule_id
        self.name = name
        self.severity = severity
        self.message = message
        self.context = context or {}
        
        self.status = AlertStatus.FIRING
        self.fired_at = datetime.utcnow()
        self.resolved_at: Optional[datetime] = None
        self.acknowledged_at: Optional[datetime] = None
        self.acknowledged_by: Optional[str] = None
    
    def resolve(self):
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
    
    def acknowledge(self, user: str):
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = user
    
    def silence(self):
        """Silence the alert."""
        self.status = AlertStatus.SILENCED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "status": self.status.value,
            "fired_at": self.fired_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
        }


class AlertManager:
    """
    Manages alert rules, notifications, and history.
    
    Features:
    - Rule-based alerting
    - Multiple notification channels
    - Alert history and analytics
    - Escalation policies
    - Alert silencing
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize alert manager.
        
        Args:
            storage_path: Path for storing alert history
            config: Channel configuration
        """
        self.storage_path = Path(
            storage_path or
            os.getenv("ALERTS_STORAGE_PATH", "./alerts")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.history: List[Alert] = []
        
        # Load history
        self._load_history()
        
        # Default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default alert rules."""
        # Drift detection rule
        self.add_rule(AlertRule(
            rule_id="drift_detected",
            name="Data Drift Detected",
            condition=lambda d: d.get("drift_detected", False),
            severity=AlertSeverity.WARNING,
            cooldown_minutes=60,
            description="Alert when data drift is detected",
        ))
        
        # Performance degradation rule
        self.add_rule(AlertRule(
            rule_id="performance_degradation",
            name="Model Performance Degradation",
            condition=lambda d: d.get("degradation", {}).get("degradation_detected", False),
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=30,
            description="Alert when model performance degrades",
        ))
        
        # High error rate rule
        self.add_rule(AlertRule(
            rule_id="high_error_rate",
            name="High Prediction Error Rate",
            condition=lambda d: d.get("error_rate", 0) > 0.1,
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=15,
            description="Alert when prediction error rate exceeds 10%",
        ))
        
        # Low accuracy rule
        self.add_rule(AlertRule(
            rule_id="low_accuracy",
            name="Low Model Accuracy",
            condition=lambda d: d.get("metrics", {}).get("accuracy", 1) < 0.8,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=60,
            description="Alert when model accuracy drops below 80%",
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def enable_rule(self, rule_id: str):
        """Enable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """Disable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    def check(self, data: Dict[str, Any]) -> List[Alert]:
        """
        Check all rules against data.
        
        Args:
            data: Data to check (metrics, drift results, etc.)
            
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for rule in self.rules.values():
            if rule.should_fire(data):
                alert = self._create_alert(rule, data)
                rule.fire()
                triggered.append(alert)
                
                # Send notifications
                self._send_notifications(alert, rule.channels)
        
        return triggered
    
    def _create_alert(
        self,
        rule: AlertRule,
        data: Dict[str, Any],
    ) -> Alert:
        """Create alert from rule."""
        alert_id = f"alert-{uuid.uuid4().hex[:8]}"
        
        message = f"[{rule.severity.value.upper()}] {rule.name}"
        if rule.description:
            message += f": {rule.description}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            name=rule.name,
            severity=rule.severity,
            message=message,
            context=data,
        )
        
        self.active_alerts[alert_id] = alert
        self.history.append(alert)
        self._save_history()
        
        return alert
    
    def _send_notifications(
        self,
        alert: Alert,
        channels: List[AlertChannel],
    ):
        """Send alert to notification channels."""
        for channel in channels:
            try:
                if channel == AlertChannel.LOG:
                    self._notify_log(alert)
                elif channel == AlertChannel.EMAIL:
                    self._notify_email(alert)
                elif channel == AlertChannel.SLACK:
                    self._notify_slack(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._notify_webhook(alert)
                elif channel == AlertChannel.PAGERDUTY:
                    self._notify_pagerduty(alert)
            except Exception as e:
                print(f"Failed to send notification to {channel}: {e}")
    
    def _notify_log(self, alert: Alert):
        """Log alert."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.EMERGENCY: "🚨",
        }
        emoji = severity_emoji.get(alert.severity, "📢")
        print(f"{emoji} ALERT: {alert.message}")
    
    def _notify_email(self, alert: Alert):
        """Send email notification."""
        email_config = self.config.get("email", {})
        
        if not email_config.get("smtp_host"):
            return
        
        msg = MIMEMultipart()
        msg["From"] = email_config.get("from_address", "alerts@mlops.local")
        msg["To"] = email_config.get("to_address", "team@example.com")
        msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.name}"
        
        body = f"""
Alert: {alert.name}
Severity: {alert.severity.value}
Time: {alert.fired_at.isoformat()}

{alert.message}

Context:
{json.dumps(alert.context, indent=2, default=str)}
"""
        msg.attach(MIMEText(body, "plain"))
        
        try:
            with smtplib.SMTP(
                email_config["smtp_host"],
                email_config.get("smtp_port", 587)
            ) as server:
                if email_config.get("smtp_tls", True):
                    server.starttls()
                if email_config.get("smtp_user"):
                    server.login(
                        email_config["smtp_user"],
                        email_config.get("smtp_password", "")
                    )
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    def _notify_slack(self, alert: Alert):
        """Send Slack notification."""
        slack_config = self.config.get("slack", {})
        webhook_url = slack_config.get("webhook_url")
        
        if not webhook_url or not AIOHTTP_AVAILABLE:
            return
        
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.EMERGENCY: "#8b0000",
        }.get(alert.severity, "#808080")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"🔔 {alert.name}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Alert ID", "value": alert.alert_id, "short": True},
                ],
                "ts": int(alert.fired_at.timestamp()),
            }]
        }
        
        # Send async
        asyncio.create_task(self._send_webhook(webhook_url, payload))
    
    def _notify_webhook(self, alert: Alert):
        """Send webhook notification."""
        webhook_config = self.config.get("webhook", {})
        webhook_url = webhook_config.get("url")
        
        if not webhook_url or not AIOHTTP_AVAILABLE:
            return
        
        payload = alert.to_dict()
        asyncio.create_task(self._send_webhook(webhook_url, payload))
    
    def _notify_pagerduty(self, alert: Alert):
        """Send PagerDuty notification."""
        pd_config = self.config.get("pagerduty", {})
        routing_key = pd_config.get("routing_key")
        
        if not routing_key or not AIOHTTP_AVAILABLE:
            return
        
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.EMERGENCY: "critical",
        }
        
        payload = {
            "routing_key": routing_key,
            "event_action": "trigger",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": alert.message,
                "severity": severity_map.get(alert.severity, "warning"),
                "source": "mlops-monitoring",
                "timestamp": alert.fired_at.isoformat(),
                "custom_details": alert.context,
            }
        }
        
        asyncio.create_task(
            self._send_webhook(
                "https://events.pagerduty.com/v2/enqueue",
                payload
            )
        )
    
    async def _send_webhook(self, url: str, payload: Dict[str, Any]):
        """Send webhook request."""
        if not AIOHTTP_AVAILABLE:
            return
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status >= 400:
                    print(f"Webhook failed: {response.status}")
    
    def acknowledge(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledge(user)
            self._save_history()
            return True
        return False
    
    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolve()
            self._save_history()
            return True
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get active alerts."""
        alerts = [
            a for a in self.active_alerts.values()
            if a.status == AlertStatus.FIRING
        ]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.fired_at, reverse=True)
    
    def get_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Dict[str, Any]]:
        """Get alert history."""
        alerts = self.history[-limit:]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return [a.to_dict() for a in reversed(alerts)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        stats = {
            "total_alerts": len(self.history),
            "active_alerts": len(self.get_active_alerts()),
            "by_severity": {},
            "by_rule": {},
            "last_24h": 0,
        }
        
        cutoff = datetime.utcnow() - timedelta(hours=24)
        
        for alert in self.history:
            # By severity
            sev = alert.severity.value
            stats["by_severity"][sev] = stats["by_severity"].get(sev, 0) + 1
            
            # By rule
            rule = alert.rule_id
            stats["by_rule"][rule] = stats["by_rule"].get(rule, 0) + 1
            
            # Last 24h
            if alert.fired_at > cutoff:
                stats["last_24h"] += 1
        
        return stats
    
    def _save_history(self):
        """Save alert history."""
        history_file = self.storage_path / "alert_history.json"
        
        # Keep last 1000
        history_to_save = [a.to_dict() for a in self.history[-1000:]]
        
        with open(history_file, "w") as f:
            json.dump(history_to_save, f, indent=2, default=str)
    
    def _load_history(self):
        """Load alert history."""
        history_file = self.storage_path / "alert_history.json"
        
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                # Note: This loads basic data, full Alert objects would need reconstruction


# Global alert manager instance
alert_manager = AlertManager()
