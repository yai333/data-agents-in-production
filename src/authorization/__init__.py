"""Authorization module for SQL agents.

Provides row-level security, column filtering, and budget enforcement.

See 2.2 for authorization patterns.
"""

from dataclasses import dataclass
from typing import Any

from src.authorization.policies import (
    RowPolicy,
    PolicyType,
    UserContext,
    create_multi_tenant_policies,
    create_ownership_policy,
)
from src.authorization.row_filter import RowFilterInjector, FilterResult
from src.authorization.column_filter import ColumnFilter, ColumnPolicy
from src.authorization.budget import (
    BudgetManager,
    QueryBudget,
    BudgetTiers,
)


@dataclass
class AuthorizationResult:
    """Complete authorization result.

    Attributes:
        authorized: Whether the query is allowed
        filtered_sql: SQL with filters applied
        row_filters_applied: Description of row filters
        columns_removed: Columns that were filtered
        budget_remaining: Remaining budget info
        warnings: Any warnings generated
        denial_reason: Reason if denied
    """

    authorized: bool
    filtered_sql: str
    row_filters_applied: list[str]
    columns_removed: list[str]
    budget_remaining: dict[str, Any]
    warnings: list[str]
    denial_reason: str | None


class AuthorizationEngine:
    """Complete authorization engine for SQL agents.

    Combines row filtering, column filtering, and budget
    enforcement into a single authorization decision.

    Example:
        >>> from src.authorization import (
        ...     AuthorizationEngine,
        ...     RowPolicy,
        ...     PolicyType,
        ...     UserContext,
        ... )
        >>> policies = [
        ...     RowPolicy("customers", PolicyType.TENANT, "tenant_id", "tenant_id"),
        ... ]
        >>> engine = AuthorizationEngine(policies)
        >>> context = UserContext(user_id="u1", tenant_id="t123", roles=["user"])
        >>> result = engine.authorize(
        ...     sql="SELECT * FROM customers",
        ...     context=context,
        ... )
        >>> result.authorized
        True
        >>> "tenant_id" in result.filtered_sql
        True
    """

    def __init__(
        self,
        row_policies: list[RowPolicy],
        column_policies: list[ColumnPolicy] | None = None,
        default_budget: QueryBudget | None = None,
    ):
        """Initialize the engine.

        Args:
            row_policies: Row-level security policies
            column_policies: Column-level security policies
            default_budget: Default query budget
        """
        self.row_filter = RowFilterInjector(row_policies)
        self.column_filter = (
            ColumnFilter(column_policies) if column_policies else None
        )
        self.budget_manager = BudgetManager(default_budget)

    def authorize(
        self,
        sql: str,
        context: UserContext,
    ) -> AuthorizationResult:
        """Authorize and filter a SQL query.

        Args:
            sql: Original SQL query
            context: User context

        Returns:
            AuthorizationResult with filtered SQL or denial reason
        """
        warnings = []

        # Check budget first (fail fast)
        budget_ok, budget_reason = self.budget_manager.check_budget(
            context.user_id
        )
        if not budget_ok:
            return AuthorizationResult(
                authorized=False,
                filtered_sql="",
                row_filters_applied=[],
                columns_removed=[],
                budget_remaining={},
                warnings=[],
                denial_reason=budget_reason,
            )

        # Apply row-level filters
        row_result = self.row_filter.apply_filters(sql, context)
        current_sql = row_result.filtered_sql

        if row_result.policy_violations:
            warnings.extend(row_result.policy_violations)

        # Apply column-level filters
        columns_removed = []
        if self.column_filter:
            current_sql, columns_removed = self.column_filter.filter_columns(
                current_sql,
                context.roles,
            )

            if not current_sql:
                return AuthorizationResult(
                    authorized=False,
                    filtered_sql="",
                    row_filters_applied=row_result.filters_applied,
                    columns_removed=columns_removed,
                    budget_remaining={},
                    warnings=warnings,
                    denial_reason="All columns filtered - no accessible data",
                )

        # Get budget info
        budget_remaining = self.budget_manager.get_remaining_budget(
            context.user_id
        )

        return AuthorizationResult(
            authorized=True,
            filtered_sql=current_sql,
            row_filters_applied=row_result.filters_applied,
            columns_removed=columns_removed,
            budget_remaining=budget_remaining,
            warnings=warnings,
            denial_reason=None,
        )

    def record_execution(
        self,
        user_id: str,
        cost: float = 0.0,
        rows: int = 0,
    ) -> None:
        """Record query execution for budget tracking.

        Args:
            user_id: User identifier
            cost: Cost of the query
            rows: Number of rows returned
        """
        self.budget_manager.record_query(user_id, cost, rows)


__all__ = [
    # Main engine
    "AuthorizationEngine",
    "AuthorizationResult",
    # Policies
    "RowPolicy",
    "PolicyType",
    "UserContext",
    "ColumnPolicy",
    # Budget
    "QueryBudget",
    "BudgetManager",
    "BudgetTiers",
    # Helpers
    "create_multi_tenant_policies",
    "create_ownership_policy",
]
