"""SQL Session Manager — stores query templates and handles dynamic pagination.

Key design:
- Stores the BASE SQL query (not results) in session
- Each page request executes SQL with LIMIT/OFFSET
- Optional per-page caching with TTL
- Sessions scoped by context_id for multi-user isolation
"""

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 20
CACHE_TTL_SECONDS = 300  # 5 minutes


@dataclass
class QuerySession:
    """Represents a stored SQL query session with pagination state."""

    query_id: str
    base_sql: str
    count_sql: str
    columns: list[str]
    total_count: int
    page_size: int = DEFAULT_PAGE_SIZE
    created_at: float = field(default_factory=time.time)
    page_cache: dict[int, tuple[list[dict], float]] = field(default_factory=dict)
    original_sql: str = ""
    search_term: str = ""

    def get_total_pages(self) -> int:
        if self.total_count == 0:
            return 1
        return (self.total_count + self.page_size - 1) // self.page_size

    def is_cache_valid(self, page: int) -> bool:
        if page not in self.page_cache:
            return False
        _, timestamp = self.page_cache[page]
        return (time.time() - timestamp) < CACHE_TTL_SECONDS

    def get_cached_page(self, page: int) -> list[dict] | None:
        if self.is_cache_valid(page):
            data, _ = self.page_cache[page]
            return data
        return None

    def cache_page(self, page: int, data: list[dict]) -> None:
        self.page_cache[page] = (data, time.time())


class SQLSessionManager:
    """Manages SQL query sessions for dynamic pagination.

    Instead of fetching all results upfront, this stores the query template
    and executes paginated queries on demand.

    Sessions are scoped by context_id (thread/conversation ID) for user isolation.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Structure: {context_id: {query_id: QuerySession}}
        self.sessions: dict[str, dict[str, QuerySession]] = {}

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _extract_columns(self, cursor: sqlite3.Cursor) -> list[str]:
        return [desc[0] for desc in cursor.description]

    def _build_count_sql(self, base_sql: str) -> str:
        return f"SELECT COUNT(*) as count FROM ({base_sql})"

    def create_session(
        self,
        base_sql: str,
        context_id: str,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> QuerySession:
        """Create a new query session.

        Args:
            base_sql: The base SQL query (without LIMIT/OFFSET).
            context_id: The thread/conversation ID for session scoping.
            page_size: Number of rows per page.

        Returns:
            QuerySession with query metadata.
        """
        query_id = str(uuid.uuid4())[:8]
        count_sql = self._build_count_sql(base_sql)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(count_sql)
            total_count = cursor.fetchone()["count"]

            cursor.execute(f"{base_sql} LIMIT 0")
            columns = self._extract_columns(cursor)

            session = QuerySession(
                query_id=query_id,
                base_sql=base_sql,
                count_sql=count_sql,
                columns=columns,
                total_count=total_count,
                page_size=page_size,
                original_sql=base_sql,
            )

            if context_id not in self.sessions:
                self.sessions[context_id] = {}
            self.sessions[context_id][query_id] = session

            logger.info(
                f"Created session {query_id} (context={context_id}): "
                f"{total_count} rows, {session.get_total_pages()} pages"
            )
            return session

        finally:
            conn.close()

    def get_session(self, context_id: str, query_id: str) -> QuerySession | None:
        """Get an existing query session.

        First tries the specified context_id, then falls back to searching
        all contexts (handles A2A's context ID changes between requests).
        """
        context_sessions = self.sessions.get(context_id, {})
        session = context_sessions.get(query_id)
        if session:
            return session

        # Fallback: search all contexts (safe because query_id is a UUID)
        for ctx_id, ctx_sessions in self.sessions.items():
            if query_id in ctx_sessions:
                logger.info(
                    f"Session {query_id} found in different context "
                    f"(requested={context_id[:8]}..., found={ctx_id[:8]}...)"
                )
                return ctx_sessions[query_id]

        return None

    def fetch_page(
        self,
        context_id: str,
        query_id: str,
        page: int = 1,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Fetch a specific page of results.

        Args:
            context_id: The thread/conversation ID.
            query_id: The query session ID.
            page: Page number (1-indexed).
            use_cache: Whether to use cached results.

        Returns:
            Dict with rows, pagination info, and metadata.
        """
        session = self.get_session(context_id, query_id)
        if not session:
            raise ValueError(f"Query session not found: {query_id}")

        total_pages = session.get_total_pages()
        page = max(1, min(page, total_pages))

        if use_cache:
            cached_data = session.get_cached_page(page)
            if cached_data is not None:
                logger.info(f"Cache hit for query {query_id}, page {page}")
                return self._build_response(session, page, cached_data, from_cache=True)

        offset = (page - 1) * session.page_size
        paginated_sql = f"{session.base_sql} LIMIT {session.page_size} OFFSET {offset}"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(paginated_sql)
            rows = [dict(row) for row in cursor.fetchall()]
            session.cache_page(page, rows)
            return self._build_response(session, page, rows, from_cache=False)

        finally:
            conn.close()

    def _build_response(
        self,
        session: QuerySession,
        page: int,
        rows: list[dict],
        from_cache: bool,
    ) -> dict[str, Any]:
        return {
            "query_id": session.query_id,
            "columns": session.columns,
            "rows": rows,
            "page": page,
            "page_size": session.page_size,
            "total_pages": session.get_total_pages(),
            "total_count": session.total_count,
            "from_cache": from_cache,
        }

    def apply_search(
        self,
        context_id: str,
        query_id: str,
        search_term: str,
        column: str | None = None,
    ) -> dict[str, Any]:
        """Apply a search filter to an existing query session.

        Wraps the original SQL with a WHERE clause. If column is specified,
        only searches that column (index-friendly). Otherwise searches all columns.
        """
        session = self.get_session(context_id, query_id)
        if not session:
            raise ValueError(f"Query session not found: {query_id}")

        safe_term = search_term.replace("'", "''").replace("%", "\\%")

        if column and column in session.columns:
            column_filter = f"CAST({column} AS TEXT) LIKE '%{safe_term}%'"
        else:
            column_filter = " OR ".join(
                f"CAST({col} AS TEXT) LIKE '%{safe_term}%'"
                for col in session.columns
            )

        filtered_sql = f"SELECT * FROM ({session.original_sql}) WHERE {column_filter}"

        session.base_sql = filtered_sql
        session.count_sql = self._build_count_sql(filtered_sql)
        session.search_term = search_term
        session.page_cache.clear()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(session.count_sql)
            session.total_count = cursor.fetchone()["count"]
        finally:
            conn.close()

        logger.info(
            f"Applied search '{search_term}' to query {query_id}: "
            f"{session.total_count} matching rows"
        )

        result = self.fetch_page(context_id, query_id, page=1)
        result["search_term"] = search_term
        return result

    def clear_search(self, context_id: str, query_id: str) -> dict[str, Any]:
        """Clear the search filter and restore original query."""
        session = self.get_session(context_id, query_id)
        if not session:
            raise ValueError(f"Query session not found: {query_id}")

        session.base_sql = session.original_sql
        session.count_sql = self._build_count_sql(session.original_sql)
        session.search_term = ""
        session.page_cache.clear()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(session.count_sql)
            session.total_count = cursor.fetchone()["count"]
        finally:
            conn.close()

        logger.info(f"Cleared search for query {query_id}")

        result = self.fetch_page(context_id, query_id, page=1)
        result["search_term"] = ""
        return result


# Global singleton
_session_manager: SQLSessionManager | None = None


def get_session_manager(db_path: str | None = None) -> SQLSessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        if db_path is None:
            raise ValueError("db_path required for first initialization")
        _session_manager = SQLSessionManager(db_path)
    return _session_manager
