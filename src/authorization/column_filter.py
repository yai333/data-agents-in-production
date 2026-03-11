"""Column-level security filtering.

Removes or masks columns based on user authorization.

See 2.2 for column-level security patterns.
"""

import re
import sqlglot
from sqlglot import exp
from dataclasses import dataclass


@dataclass
class ColumnPolicy:
    """Column-level security policy.

    Attributes:
        table: Table containing the column
        column: Column to restrict
        allowed_roles: Roles that can see this column
        replacement: SQL expression to show instead (None = exclude)

    Example:
        >>> policy = ColumnPolicy(
        ...     table="employees",
        ...     column="salary",
        ...     allowed_roles=["hr", "admin"],
        ...     replacement="'[REDACTED]'",
        ... )
    """

    table: str
    column: str
    allowed_roles: list[str]
    replacement: str | None = None


class ColumnFilter:
    """Filter columns based on user authorization.

    Removes or masks columns the user isn't authorized to see.

    Example:
        >>> policies = [
        ...     ColumnPolicy("employees", "salary", ["hr"], "'***'"),
        ... ]
        >>> filter = ColumnFilter(policies)
        >>> sql, removed = filter.filter_columns(
        ...     "SELECT name, salary FROM employees",
        ...     user_roles=["user"],
        ... )
        >>> "salary" not in sql or "'***'" in sql
        True
    """

    def __init__(self, policies: list[ColumnPolicy]):
        """Initialize with policies.

        Args:
            policies: List of column policies
        """
        self.policies: dict[tuple[str, str], ColumnPolicy] = {
            (p.table.lower(), p.column.lower()): p
            for p in policies
        }

    def filter_columns(
        self,
        sql: str,
        user_roles: list[str],
    ) -> tuple[str, list[str]]:
        """Filter restricted columns from SQL.

        Args:
            sql: SQL query to filter
            user_roles: User's roles

        Returns:
            Tuple of (filtered_sql, removed_columns)
        """
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception:
            return sql, []

        removed = []
        select = parsed.find(exp.Select)

        if not select:
            return sql, []

        # Get all tables in the query for context
        tables = self._build_table_map(parsed)

        # Handle SELECT * specially
        if self._has_select_star(select):
            # Can't filter SELECT * - would need schema info
            # Log warning but allow (defense in depth at output filter)
            pass

        # Process SELECT columns
        new_expressions = []
        for col_expr in select.expressions:
            col_name = self._get_column_name(col_expr)
            table_name = self._get_table_name(col_expr, tables)

            key = (table_name, col_name.lower()) if table_name else None

            # Check if column is restricted
            if key and key in self.policies:
                policy = self.policies[key]

                # Check if user has required role
                if not any(role in policy.allowed_roles for role in user_roles):
                    removed.append(f"{table_name}.{col_name}")

                    if policy.replacement:
                        # Mask the column
                        new_expr = sqlglot.parse_one(
                            f"{policy.replacement} AS {col_name}"
                        )
                        new_expressions.append(new_expr)
                    # Otherwise, exclude entirely
                    continue

            new_expressions.append(col_expr)

        # Update SELECT expressions
        if new_expressions:
            select.set("expressions", new_expressions)
        else:
            # All columns filtered - return error indicator
            return "", removed

        return parsed.sql(), removed

    def _build_table_map(self, parsed: exp.Expression) -> dict[str, str]:
        """Build mapping of alias -> table name."""
        return {
            (t.alias or t.name).lower(): t.name.lower()
            for t in parsed.find_all(exp.Table)
        }

    def _has_select_star(self, select: exp.Select) -> bool:
        """Check if SELECT uses *."""
        for expr in select.expressions:
            if isinstance(expr, exp.Star):
                return True
        return False

    def _get_column_name(self, expr: exp.Expression) -> str:
        """Extract column name from expression."""
        if isinstance(expr, exp.Column):
            return expr.name
        if isinstance(expr, exp.Alias):
            return expr.alias
        if hasattr(expr, "name"):
            return expr.name
        return str(expr)

    def _get_table_name(
        self,
        expr: exp.Expression,
        tables: dict[str, str],
    ) -> str | None:
        """Extract table name from column expression."""
        if isinstance(expr, exp.Column) and expr.table:
            alias = expr.table.lower()
            return tables.get(alias, alias)
        # For unqualified columns with single table, use that table
        if len(tables) == 1:
            return list(tables.values())[0]
        return None


def is_column_restricted(
    table: str,
    column: str,
    policies: list[ColumnPolicy],
) -> bool:
    """Check if a column has restrictions.

    Args:
        table: Table name
        column: Column name
        policies: List of policies

    Returns:
        True if column has restrictions
    """
    key = (table.lower(), column.lower())
    return any(
        (p.table.lower(), p.column.lower()) == key
        for p in policies
    )
