"""Security audit logging for SQL agents.

Provides forensics and anomaly detection capabilities.

See 2.1 for the complete security architecture.
"""

import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Callable
from pathlib import Path


@dataclass
class AuditEvent:
    """Security audit event."""

    timestamp: str
    event_type: str
    user_id: str
    session_id: str
    action: str
    input_hash: str  # Hash of input (don't log raw for PII)
    sql_generated: str | None
    success: bool
    risk_score: float
    violations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class SecurityAuditor:
    """Audit logger for security events.

    Logs all query attempts with security-relevant metadata
    for forensics and anomaly detection.

    Example:
        >>> auditor = SecurityAuditor()
        >>> event = auditor.log_query_attempt(
        ...     user_id="user-123",
        ...     session_id="session-abc",
        ...     user_input="Show me customer data",
        ...     sql="SELECT * FROM customers",
        ...     success=True,
        ...     risk_score=0.1,
        ...     violations=[],
        ... )
        >>> event.event_type
        'query_attempt'
    """

    def __init__(
        self,
        log_path: str = "logs/security_audit.jsonl",
        alert_callback: Callable[[AuditEvent], None] | None = None,
        alert_threshold: float = 0.7,
    ):
        """Initialize the auditor.

        Args:
            log_path: Path to audit log file
            alert_callback: Function to call for high-risk events
            alert_threshold: Risk score that triggers alerts
        """
        self.log_path = Path(log_path)
        self.alert_callback = alert_callback
        self.alert_threshold = alert_threshold

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_query_attempt(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        sql: str | None,
        success: bool,
        risk_score: float,
        violations: list[str],
        **metadata,
    ) -> AuditEvent:
        """Log a query attempt for security audit.

        Args:
            user_id: User identifier
            session_id: Session identifier
            user_input: Raw user input (will be hashed)
            sql: Generated SQL (if any)
            success: Whether the query succeeded
            risk_score: Calculated risk score
            violations: List of security violations detected
            **metadata: Additional context

        Returns:
            AuditEvent that was logged
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="query_attempt",
            user_id=user_id,
            session_id=session_id,
            action="generate_sql",
            input_hash=self._hash_input(user_input),
            sql_generated=sql,
            success=success,
            risk_score=risk_score,
            violations=violations,
            metadata=metadata,
        )

        self._write_event(event)

        # Alert on high-risk events
        if risk_score >= self.alert_threshold or len(violations) > 2:
            self._send_alert(event)

        return event

    def log_tool_invocation(
        self,
        user_id: str,
        session_id: str,
        tool_name: str,
        action: str,
        allowed: bool,
        reason: str,
        **metadata,
    ) -> AuditEvent:
        """Log a tool invocation attempt.

        Args:
            user_id: User identifier
            session_id: Session identifier
            tool_name: Name of the tool
            action: Action attempted
            allowed: Whether the action was allowed
            reason: Reason for allow/deny
            **metadata: Additional context

        Returns:
            AuditEvent that was logged
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="tool_invocation",
            user_id=user_id,
            session_id=session_id,
            action=f"{tool_name}:{action}",
            input_hash="",
            sql_generated=None,
            success=allowed,
            risk_score=0.0 if allowed else 0.8,
            violations=[reason] if not allowed else [],
            metadata=metadata,
        )

        self._write_event(event)

        if not allowed:
            self._send_alert(event)

        return event

    def _hash_input(self, user_input: str) -> str:
        """Hash input for logging without storing raw PII.

        Uses SHA-256 truncated to 16 characters.
        This allows correlation without storing sensitive content.
        """
        return hashlib.sha256(user_input.encode()).hexdigest()[:16]

    def _write_event(self, event: AuditEvent):
        """Write event to audit log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def _send_alert(self, event: AuditEvent):
        """Send alert for high-risk events.

        Override or inject callback for integration with:
        - PagerDuty
        - Slack
        - SIEM (Splunk, DataDog, etc.)
        - Email
        """
        if self.alert_callback:
            self.alert_callback(event)


def analyze_audit_log(
    log_path: str,
    time_window_hours: int = 24,
) -> dict[str, Any]:
    """Analyze audit log for anomalies.

    Args:
        log_path: Path to audit log
        time_window_hours: Hours to analyze

    Returns:
        Dict with anomaly analysis
    """
    from datetime import timedelta

    events = []
    cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)

    with open(log_path) as f:
        for line in f:
            event = json.loads(line)
            event_time = datetime.fromisoformat(
                event["timestamp"].replace("Z", "+00:00")
            )
            if event_time.replace(tzinfo=None) > cutoff:
                events.append(event)

    if not events:
        return {"total_events": 0, "anomalies": []}

    # Analyze patterns
    high_risk_count = sum(1 for e in events if e["risk_score"] >= 0.7)
    failed_count = sum(1 for e in events if not e["success"])

    # User analysis
    users = {}
    for e in events:
        user_id = e["user_id"]
        if user_id not in users:
            users[user_id] = {"count": 0, "high_risk": 0, "failed": 0}
        users[user_id]["count"] += 1
        if e["risk_score"] >= 0.7:
            users[user_id]["high_risk"] += 1
        if not e["success"]:
            users[user_id]["failed"] += 1

    # Flag suspicious users
    suspicious_users = [
        user_id for user_id, stats in users.items()
        if stats["high_risk"] > 3 or stats["failed"] / max(stats["count"], 1) > 0.5
    ]

    return {
        "total_events": len(events),
        "high_risk_count": high_risk_count,
        "failed_count": failed_count,
        "unique_users": len(users),
        "suspicious_users": suspicious_users,
        "time_window_hours": time_window_hours,
    }
