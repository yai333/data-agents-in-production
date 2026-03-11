"""Authorization policies for SQL agents.

Defines the policy types and data structures for row-level
and column-level security.

See 2.2 for authorization patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyType(str, Enum):
    """Types of row-level authorization policies."""

    TENANT = "tenant"  # Multi-tenant isolation
    OWNERSHIP = "ownership"  # User owns the record
    ROLE_BASED = "role_based"  # Role determines access
    HIERARCHY = "hierarchy"  # Org hierarchy (manager sees reports)
    CUSTOM = "custom"  # Custom predicate


@dataclass
class RowPolicy:
    """A row-level security policy.

    Defines which rows a user can access in a table.

    Attributes:
        table: Table this policy applies to
        policy_type: Type of policy
        filter_column: Column used for filtering
        context_key: Key in user context to compare
        custom_predicate: SQL predicate for custom policies

    Example:
        >>> policy = RowPolicy(
        ...     table="customers",
        ...     policy_type=PolicyType.TENANT,
        ...     filter_column="tenant_id",
        ...     context_key="tenant_id",
        ... )
    """

    table: str
    policy_type: PolicyType
    filter_column: str = ""
    context_key: str = ""
    custom_predicate: str = ""


@dataclass
class UserContext:
    """User context for authorization.

    Contains all information needed to evaluate policies.

    Attributes:
        user_id: Unique user identifier
        tenant_id: Tenant identifier for multi-tenant apps
        roles: List of roles assigned to user
        department_id: Department for org-based filtering
        manager_of: List of user IDs this user manages
        custom_attributes: Additional attributes for custom policies
    """

    user_id: str
    tenant_id: str | None = None
    roles: list[str] = field(default_factory=list)
    department_id: str | None = None
    manager_of: list[str] = field(default_factory=list)
    custom_attributes: dict[str, Any] = field(default_factory=dict)


# Example policies for common scenarios
def create_multi_tenant_policies(tables: list[str]) -> list[RowPolicy]:
    """Create tenant isolation policies for multiple tables.

    Args:
        tables: List of table names to protect

    Returns:
        List of RowPolicy objects
    """
    return [
        RowPolicy(
            table=table,
            policy_type=PolicyType.TENANT,
            filter_column="tenant_id",
            context_key="tenant_id",
        )
        for table in tables
    ]


def create_ownership_policy(
    table: str,
    user_column: str = "user_id",
) -> RowPolicy:
    """Create an ownership policy for a table.

    Args:
        table: Table name
        user_column: Column containing user ID

    Returns:
        RowPolicy for ownership
    """
    return RowPolicy(
        table=table,
        policy_type=PolicyType.OWNERSHIP,
        filter_column=user_column,
        context_key="user_id",
    )
