"""Row-level security filter injection.

Automatically injects WHERE clauses into SQL queries based on
user context and defined policies.

See 2.2 for row-level security patterns.
"""

import sqlglot
from sqlglot import exp
from dataclasses import dataclass

from src.authorization.policies import (
    RowPolicy,
    PolicyType,
    UserContext,
)


@dataclass
class FilterResult:
    """Result of applying row filters.

    Attributes:
        original_sql: The input SQL
        filtered_sql: SQL with filters applied
        filters_applied: Description of filters added
        policy_violations: Any policy issues encountered
    """

    original_sql: str
    filtered_sql: str
    filters_applied: list[str]
    policy_violations: list[str]


class RowFilterInjector:
    """Inject row-level filters into SQL queries.

    Automatically adds WHERE clauses based on user context
    and defined policies.

    Example:
        >>> policies = [
        ...     RowPolicy("customers", PolicyType.TENANT, "tenant_id", "tenant_id"),
        ... ]
        >>> injector = RowFilterInjector(policies)
        >>> context = UserContext(user_id="u1", tenant_id="t123")
        >>> result = injector.apply_filters(
        ...     sql="SELECT * FROM customers",
        ...     context=context,
        ... )
        >>> "tenant_id" in result.filtered_sql
        True
    """

    def __init__(self, policies: list[RowPolicy]):
        """Initialize with policies.

        Args:
            policies: List of row-level policies
        """
        self.policies = {p.table.lower(): p for p in policies}

    def apply_filters(
        self,
        sql: str,
        context: UserContext,
    ) -> FilterResult:
        """Apply row-level filters to a SQL query.

        Args:
            sql: Original SQL query
            context: User context for authorization

        Returns:
            FilterResult with filtered SQL
        """
        filters_applied = []
        policy_violations = []

        try:
            parsed = sqlglot.parse_one(sql)
        except Exception as e:
            return FilterResult(
                original_sql=sql,
                filtered_sql=sql,
                filters_applied=[],
                policy_violations=[f"Parse error: {e}"],
            )

        # Find all tables in the query
        tables = list(parsed.find_all(exp.Table))

        for table in tables:
            table_name = table.name.lower()
            alias = table.alias or table_name

            if table_name not in self.policies:
                continue

            policy = self.policies[table_name]
            filter_condition = self._build_filter(policy, context, alias)

            if filter_condition is None:
                policy_violations.append(
                    f"Cannot apply policy to {table_name}: missing context"
                )
                continue

            # Inject the filter
            parsed = self._inject_where(parsed, filter_condition)
            filters_applied.append(
                f"{table_name}: {policy.policy_type.value}"
            )

        return FilterResult(
            original_sql=sql,
            filtered_sql=parsed.sql(),
            filters_applied=filters_applied,
            policy_violations=policy_violations,
        )

    def _build_filter(
        self,
        policy: RowPolicy,
        context: UserContext,
        table_alias: str,
    ) -> exp.Expression | None:
        """Build the filter expression for a policy.

        Args:
            policy: The policy to apply
            context: User context
            table_alias: Table alias in the query

        Returns:
            SQL expression for the filter, or None if can't build
        """
        if policy.policy_type == PolicyType.TENANT:
            if context.tenant_id is None:
                return None
            # Use parameterized-style for safety
            return sqlglot.parse_one(
                f"{table_alias}.{policy.filter_column} = '{context.tenant_id}'"
            )

        elif policy.policy_type == PolicyType.OWNERSHIP:
            return sqlglot.parse_one(
                f"{table_alias}.{policy.filter_column} = '{context.user_id}'"
            )

        elif policy.policy_type == PolicyType.HIERARCHY:
            if not context.manager_of:
                # User can only see themselves
                return sqlglot.parse_one(
                    f"{table_alias}.{policy.filter_column} = '{context.user_id}'"
                )
            # Manager can see their reports plus themselves
            ids = "', '".join([context.user_id] + context.manager_of)
            return sqlglot.parse_one(
                f"{table_alias}.{policy.filter_column} IN ('{ids}')"
            )

        elif policy.policy_type == PolicyType.ROLE_BASED:
            # Role-based policies might allow admin to see all
            if "admin" in context.roles:
                return sqlglot.parse_one("1 = 1")  # No filter for admins
            return sqlglot.parse_one(
                f"{table_alias}.{policy.filter_column} = '{context.user_id}'"
            )

        elif policy.policy_type == PolicyType.CUSTOM:
            # Substitute context values into predicate
            predicate = policy.custom_predicate
            for key, value in context.custom_attributes.items():
                predicate = predicate.replace(f"{{{key}}}", str(value))
            return sqlglot.parse_one(predicate)

        return None

    def _inject_where(
        self,
        parsed: exp.Expression,
        filter_condition: exp.Expression,
    ) -> exp.Expression:
        """Inject a filter into the WHERE clause.

        Args:
            parsed: Parsed SQL expression
            filter_condition: Condition to add

        Returns:
            Modified SQL expression
        """
        # Find the SELECT statement
        select = parsed.find(exp.Select)
        if not select:
            return parsed

        # Get existing WHERE clause
        where = select.find(exp.Where)

        if where:
            # AND with existing condition
            new_condition = exp.And(
                this=where.this.copy(),
                expression=filter_condition,
            )
            where.set("this", new_condition)
        else:
            # Add new WHERE clause
            select.set("where", exp.Where(this=filter_condition))

        return parsed
