"""Just-in-time permission management.

Implements zero-trust architecture for agent actions:
- Time-limited permission grants
- Scope-based access control
- Automatic expiration

See 2.1 for zero-trust patterns.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Generator
from contextlib import contextmanager


@dataclass
class PermissionGrant:
    """A time-limited permission grant.

    Attributes:
        permission: Name of the permission
        granted_at: Unix timestamp when granted
        expires_at: Unix timestamp when expires
        scope: Constraints on the permission (e.g., specific tables)
        reason: Why the permission was granted
    """

    permission: str
    granted_at: float
    expires_at: float
    scope: dict[str, Any]
    reason: str


class JITPermissionManager:
    """Manage just-in-time permissions for agent actions.

    Implements the principle of least privilege with time-limited grants.
    Permissions are granted only for specific operations and expire
    automatically.

    Example:
        >>> jit = JITPermissionManager(default_ttl=60)
        >>> with jit.temporary_permission("session-1", "read_customer_data"):
        ...     # Permission is active here
        ...     result = execute_query(sql)
        >>> # Permission has been revoked
    """

    def __init__(self, default_ttl: int = 300):
        """Initialize the manager.

        Args:
            default_ttl: Default time-to-live in seconds (default: 5 minutes)
        """
        self.default_ttl = default_ttl
        self.grants: dict[str, list[PermissionGrant]] = {}

    @contextmanager
    def temporary_permission(
        self,
        session_id: str,
        permission: str,
        ttl: int | None = None,
        scope: dict[str, Any] | None = None,
        reason: str = "",
    ) -> Generator[PermissionGrant, None, None]:
        """Grant a temporary permission for a specific operation.

        The permission is automatically revoked when the context exits.

        Args:
            session_id: Session identifier
            permission: Permission to grant
            ttl: Time-to-live in seconds (None = use default)
            scope: Optional scope constraints
            reason: Why the permission is needed

        Yields:
            The PermissionGrant object

        Example:
            >>> with jit.temporary_permission(session, "read_pii", ttl=30):
            ...     # Can access PII for 30 seconds
            ...     result = await execute_query(sql)
        """
        grant = self._create_grant(
            session_id=session_id,
            permission=permission,
            ttl=ttl or self.default_ttl,
            scope=scope or {},
            reason=reason,
        )

        try:
            yield grant
        finally:
            self._revoke_grant(session_id, grant)

    def grant_permission(
        self,
        session_id: str,
        permission: str,
        ttl: int | None = None,
        scope: dict[str, Any] | None = None,
        reason: str = "",
    ) -> PermissionGrant:
        """Grant a permission (manual management).

        Unlike temporary_permission, this doesn't auto-revoke.
        Use check_permission and revoke_all to manage lifecycle.

        Args:
            session_id: Session identifier
            permission: Permission to grant
            ttl: Time-to-live in seconds
            scope: Optional scope constraints
            reason: Why the permission is needed

        Returns:
            The PermissionGrant object
        """
        return self._create_grant(
            session_id=session_id,
            permission=permission,
            ttl=ttl or self.default_ttl,
            scope=scope or {},
            reason=reason,
        )

    def check_permission(
        self,
        session_id: str,
        permission: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if a permission is currently granted.

        Args:
            session_id: Session identifier
            permission: Permission to check
            context: Request context for scope matching

        Returns:
            True if permission is granted and not expired
        """
        now = time.time()
        self._cleanup_expired(session_id)

        if session_id not in self.grants:
            return False

        for grant in self.grants[session_id]:
            if grant.permission != permission:
                continue
            if grant.expires_at < now:
                continue
            if not self._scope_matches(grant.scope, context or {}):
                continue
            return True

        return False

    def get_active_permissions(self, session_id: str) -> list[PermissionGrant]:
        """Get all active (non-expired) permissions for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of active grants
        """
        self._cleanup_expired(session_id)
        return self.grants.get(session_id, []).copy()

    def revoke_all(self, session_id: str) -> int:
        """Revoke all permissions for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of permissions revoked
        """
        if session_id not in self.grants:
            return 0

        count = len(self.grants[session_id])
        del self.grants[session_id]
        return count

    def _create_grant(
        self,
        session_id: str,
        permission: str,
        ttl: int,
        scope: dict[str, Any],
        reason: str,
    ) -> PermissionGrant:
        """Create a new permission grant."""
        now = time.time()
        grant = PermissionGrant(
            permission=permission,
            granted_at=now,
            expires_at=now + ttl,
            scope=scope,
            reason=reason,
        )

        if session_id not in self.grants:
            self.grants[session_id] = []
        self.grants[session_id].append(grant)

        return grant

    def _revoke_grant(self, session_id: str, grant: PermissionGrant) -> None:
        """Revoke a specific grant."""
        if session_id in self.grants:
            self.grants[session_id] = [
                g for g in self.grants[session_id]
                if g is not grant
            ]

    def _cleanup_expired(self, session_id: str) -> None:
        """Remove expired grants for a session."""
        if session_id not in self.grants:
            return

        now = time.time()
        self.grants[session_id] = [
            g for g in self.grants[session_id]
            if g.expires_at > now
        ]

    def _scope_matches(self, grant_scope: dict, request_context: dict) -> bool:
        """Check if request context matches grant scope.

        All scope keys must be present and match in the request context.

        Args:
            grant_scope: Scope constraints from the grant
            request_context: Context from the request

        Returns:
            True if context satisfies scope constraints
        """
        for key, value in grant_scope.items():
            if key not in request_context:
                return False
            if request_context[key] != value:
                return False
        return True


# Common permission constants
class Permissions:
    """Standard permission names for SQL agents."""

    READ_SCHEMA = "read_schema"
    READ_DATA = "read_data"
    READ_PII = "read_pii"
    EXECUTE_QUERY = "execute_query"
    GENERATE_SQL = "generate_sql"
    ACCESS_TOOL = "access_tool"
