"""MCP Firewall for tool governance.

Implements governance controls for MCP (Model Context Protocol) tool access:
- Tool allowlisting
- Action-level permissions
- Rate limiting
- Audit logging

See 2.1 for MCP security patterns.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolPermission(str, Enum):
    """Permission levels for MCP tools."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class ToolPolicy:
    """Policy for a single MCP tool.

    Attributes:
        tool_name: Name of the MCP tool
        permission: Base permission level
        allowed_actions: Specific actions permitted (None = all actions)
        rate_limit: Maximum calls per minute (None = unlimited)
        requires_audit: Whether to log all invocations
    """

    tool_name: str
    permission: ToolPermission
    allowed_actions: list[str] | None = None
    rate_limit: int | None = None
    requires_audit: bool = True


class MCPFirewall:
    """Firewall for controlling agent-tool interactions.

    Implements governance controls for MCP tool access:
    - Tool allowlisting (unknown tools denied by default)
    - Action-level permissions
    - Rate limiting (per tool)
    - Integration with audit logging

    Example:
        >>> policies = [
        ...     ToolPolicy("database_query", ToolPermission.ALLOW, rate_limit=60),
        ...     ToolPolicy("database_write", ToolPermission.DENY),
        ... ]
        >>> firewall = MCPFirewall(policies)
        >>> allowed, reason = firewall.check_permission(
        ...     "database_query", "read_only", {}
        ... )
        >>> allowed
        True
    """

    def __init__(self, policies: list[ToolPolicy]):
        """Initialize the firewall with policies.

        Args:
            policies: List of tool policies
        """
        self.policies = {p.tool_name: p for p in policies}
        self.call_counts: dict[str, list[float]] = {}

    def check_permission(
        self,
        tool_name: str,
        action: str,
        user_context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if tool invocation is permitted.

        Args:
            tool_name: Name of the MCP tool
            action: Specific action being performed
            user_context: User/session context (may include "has_approval")

        Returns:
            Tuple of (allowed, reason)
        """
        # Unknown tools are denied by default (allowlist approach)
        if tool_name not in self.policies:
            return False, f"Unknown tool: {tool_name}. Not in allowlist."

        policy = self.policies[tool_name]

        # Check base permission
        if policy.permission == ToolPermission.DENY:
            return False, f"Tool {tool_name} is explicitly denied."

        # Check action-level permission
        if policy.allowed_actions and action not in policy.allowed_actions:
            return False, f"Action '{action}' not allowed for {tool_name}."

        # Check rate limit
        if policy.rate_limit:
            if not self._check_rate_limit(tool_name, policy.rate_limit):
                return False, f"Rate limit exceeded for {tool_name}."

        # Require explicit approval for sensitive tools
        if policy.permission == ToolPermission.REQUIRE_APPROVAL:
            if not user_context.get("has_approval"):
                return False, f"Tool {tool_name} requires explicit user approval."

        return True, "Allowed"

    def record_invocation(self, tool_name: str) -> None:
        """Record a tool invocation for rate limiting.

        Call this after a successful permission check.

        Args:
            tool_name: Name of the tool invoked
        """
        now = time.time()
        if tool_name not in self.call_counts:
            self.call_counts[tool_name] = []
        self.call_counts[tool_name].append(now)

    def _check_rate_limit(self, tool_name: str, limit: int) -> bool:
        """Check if rate limit is exceeded.

        Uses a 60-second sliding window.

        Args:
            tool_name: Tool to check
            limit: Maximum calls per minute

        Returns:
            True if within limit, False if exceeded
        """
        now = time.time()
        window = 60  # 1 minute window

        if tool_name not in self.call_counts:
            self.call_counts[tool_name] = []

        # Clean old entries
        self.call_counts[tool_name] = [
            t for t in self.call_counts[tool_name]
            if now - t < window
        ]

        return len(self.call_counts[tool_name]) < limit

    def get_stats(self) -> dict[str, Any]:
        """Get current firewall statistics.

        Returns:
            Dict with tool invocation counts and rate limit status
        """
        now = time.time()
        window = 60

        stats = {}
        for tool_name, policy in self.policies.items():
            calls = [t for t in self.call_counts.get(tool_name, []) if now - t < window]
            stats[tool_name] = {
                "permission": policy.permission.value,
                "calls_last_minute": len(calls),
                "rate_limit": policy.rate_limit,
                "rate_limit_remaining": (
                    policy.rate_limit - len(calls)
                    if policy.rate_limit
                    else None
                ),
            }
        return stats


# Default policies for SQL agents
DEFAULT_SQL_AGENT_POLICIES = [
    ToolPolicy(
        tool_name="database_query",
        permission=ToolPermission.ALLOW,
        allowed_actions=["read_only_query", "explain_query"],
        rate_limit=60,
        requires_audit=True,
    ),
    ToolPolicy(
        tool_name="database_schema",
        permission=ToolPermission.ALLOW,
        allowed_actions=["list_tables", "describe_table", "list_columns"],
        rate_limit=30,
    ),
    ToolPolicy(
        tool_name="database_write",
        permission=ToolPermission.DENY,
    ),
    ToolPolicy(
        tool_name="external_api",
        permission=ToolPermission.REQUIRE_APPROVAL,
        rate_limit=10,
    ),
    ToolPolicy(
        tool_name="file_system",
        permission=ToolPermission.DENY,
    ),
    ToolPolicy(
        tool_name="code_execution",
        permission=ToolPermission.DENY,
    ),
]


def create_default_firewall() -> MCPFirewall:
    """Create a firewall with default SQL agent policies.

    Returns:
        MCPFirewall configured for SQL agents
    """
    return MCPFirewall(DEFAULT_SQL_AGENT_POLICIES)
