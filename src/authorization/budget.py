"""Query budget enforcement.

Implements rate limiting and cost controls for SQL agents.

See 2.2 for budget patterns.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryBudget:
    """Budget limits for a user's queries.

    Attributes:
        max_queries_per_minute: Rate limit
        max_rows_per_query: Maximum rows returned
        max_execution_seconds: Query timeout
        max_daily_cost: Token/compute cost limit in dollars

    Example:
        >>> budget = QueryBudget(
        ...     max_queries_per_minute=30,
        ...     max_rows_per_query=1000,
        ...     max_daily_cost=10.0,
        ... )
    """

    max_queries_per_minute: int = 30
    max_rows_per_query: int = 1000
    max_execution_seconds: int = 30
    max_daily_cost: float = 10.0


@dataclass
class BudgetTracker:
    """Track usage for a single user.

    Maintains per-user usage metrics for budget enforcement.
    """

    query_timestamps: list[float] = field(default_factory=list)
    daily_cost: float = 0.0
    daily_reset_time: float = field(default_factory=time.time)
    total_queries: int = 0
    total_rows: int = 0

    def check_rate_limit(self, budget: QueryBudget) -> bool:
        """Check if user is within rate limit.

        Args:
            budget: Budget to check against

        Returns:
            True if within limit
        """
        now = time.time()
        window = 60  # 1 minute

        # Clean old timestamps
        self.query_timestamps = [
            t for t in self.query_timestamps
            if now - t < window
        ]

        return len(self.query_timestamps) < budget.max_queries_per_minute

    def record_query(self, cost: float = 0.0, rows: int = 0) -> None:
        """Record a query execution.

        Args:
            cost: Cost of the query
            rows: Number of rows returned
        """
        now = time.time()
        self.query_timestamps.append(now)
        self.total_queries += 1
        self.total_rows += rows

        # Reset daily cost if new day
        if now - self.daily_reset_time > 86400:  # 24 hours
            self.daily_cost = cost
            self.daily_reset_time = now
        else:
            self.daily_cost += cost

    def check_daily_cost(self, budget: QueryBudget) -> bool:
        """Check if user is within daily cost limit.

        Args:
            budget: Budget to check against

        Returns:
            True if within limit
        """
        return self.daily_cost < budget.max_daily_cost

    def get_usage_stats(self, budget: QueryBudget) -> dict[str, Any]:
        """Get current usage statistics.

        Args:
            budget: Budget for calculating remaining

        Returns:
            Dict with usage stats
        """
        now = time.time()
        queries_in_window = sum(
            1 for t in self.query_timestamps
            if now - t < 60
        )

        return {
            "queries_in_last_minute": queries_in_window,
            "queries_remaining": budget.max_queries_per_minute - queries_in_window,
            "daily_cost": self.daily_cost,
            "cost_remaining": budget.max_daily_cost - self.daily_cost,
            "total_queries": self.total_queries,
            "total_rows": self.total_rows,
        }


class BudgetManager:
    """Manage query budgets across users.

    Provides centralized budget enforcement and tracking.

    Example:
        >>> manager = BudgetManager()
        >>> allowed, reason = manager.check_budget("user-123")
        >>> if allowed:
        ...     execute_query()
        ...     manager.record_query("user-123", cost=0.01)
    """

    def __init__(self, default_budget: QueryBudget | None = None):
        """Initialize the manager.

        Args:
            default_budget: Default budget for users without custom budget
        """
        self.default_budget = default_budget or QueryBudget()
        self.user_budgets: dict[str, QueryBudget] = {}
        self.trackers: dict[str, BudgetTracker] = {}

    def set_budget(self, user_id: str, budget: QueryBudget) -> None:
        """Set custom budget for a user.

        Args:
            user_id: User identifier
            budget: Custom budget
        """
        self.user_budgets[user_id] = budget

    def get_budget(self, user_id: str) -> QueryBudget:
        """Get budget for a user.

        Args:
            user_id: User identifier

        Returns:
            User's budget (custom or default)
        """
        return self.user_budgets.get(user_id, self.default_budget)

    def check_budget(self, user_id: str) -> tuple[bool, str]:
        """Check if user can execute a query.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (allowed, reason)
        """
        budget = self.get_budget(user_id)
        tracker = self._get_tracker(user_id)

        if not tracker.check_rate_limit(budget):
            return False, "Rate limit exceeded"

        if not tracker.check_daily_cost(budget):
            return False, "Daily cost limit exceeded"

        return True, "OK"

    def record_query(
        self,
        user_id: str,
        cost: float = 0.0,
        rows: int = 0,
    ) -> None:
        """Record a query execution for budget tracking.

        Args:
            user_id: User identifier
            cost: Cost of the query
            rows: Number of rows returned
        """
        tracker = self._get_tracker(user_id)
        tracker.record_query(cost, rows)

    def get_remaining_budget(self, user_id: str) -> dict[str, Any]:
        """Get remaining budget for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with remaining budget info
        """
        budget = self.get_budget(user_id)
        tracker = self._get_tracker(user_id)

        stats = tracker.get_usage_stats(budget)

        return {
            "queries_remaining": stats["queries_remaining"],
            "cost_remaining": stats["cost_remaining"],
            "max_rows": budget.max_rows_per_query,
            "timeout_seconds": budget.max_execution_seconds,
        }

    def _get_tracker(self, user_id: str) -> BudgetTracker:
        """Get or create tracker for user.

        Args:
            user_id: User identifier

        Returns:
            BudgetTracker for the user
        """
        if user_id not in self.trackers:
            self.trackers[user_id] = BudgetTracker()
        return self.trackers[user_id]


# Preset budgets for different tiers
class BudgetTiers:
    """Preset budget configurations for user tiers."""

    FREE = QueryBudget(
        max_queries_per_minute=10,
        max_rows_per_query=100,
        max_execution_seconds=10,
        max_daily_cost=1.0,
    )

    BASIC = QueryBudget(
        max_queries_per_minute=30,
        max_rows_per_query=1000,
        max_execution_seconds=30,
        max_daily_cost=10.0,
    )

    PRO = QueryBudget(
        max_queries_per_minute=60,
        max_rows_per_query=10000,
        max_execution_seconds=60,
        max_daily_cost=100.0,
    )

    ENTERPRISE = QueryBudget(
        max_queries_per_minute=120,
        max_rows_per_query=100000,
        max_execution_seconds=120,
        max_daily_cost=1000.0,
    )
