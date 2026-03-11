"""SQL Explorer Agent Executor — handles requests and pagination.

Key feature: Dynamic Pagination
When a page_change userAction is received, this executor:
1. Does NOT call the LLM
2. Fetches the requested page directly from SQLSessionManager
3. Sends only a dataModelUpdate with the new page data

This makes pagination instant and doesn't consume LLM tokens.
"""

import json
import logging
import os

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_parts_message, new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2ui.extension.a2ui_extension import create_a2ui_part, try_activate_a2ui_extension

from agent import SQLExplorerAgent
from sql_session_manager import get_session_manager

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    """Get the path to the Chinook database."""
    db_path = os.getenv("CHINOOK_DB_PATH")
    if db_path and os.path.exists(db_path):
        return db_path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "data", "chinook.db")


class SQLExplorerAgentExecutor(AgentExecutor):
    """Executor for SQL Explorer Agent.

    Routes requests to the appropriate handler:
    - Regular text queries -> LLM agent
    - page_change actions -> direct pagination (NO LLM)
    - search actions -> direct SQL filter (NO LLM)
    - clear_search actions -> restore original query (NO LLM)
    """

    def __init__(self, base_url: str):
        self.ui_agent = SQLExplorerAgent(base_url=base_url, use_ui=True)
        self.text_agent = SQLExplorerAgent(base_url=base_url, use_ui=False)
        get_session_manager(_get_db_path())

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = ""
        ui_event_part = None
        action = None

        use_ui = try_activate_a2ui_extension(context)
        agent = self.ui_agent if use_ui else self.text_agent

        # Parse incoming message parts
        if context.message and context.message.parts:
            for part in context.message.parts:
                if isinstance(part.root, DataPart):
                    if "userAction" in part.root.data:
                        ui_event_part = part.root.data["userAction"]
                elif isinstance(part.root, TextPart):
                    pass  # text parts handled below

        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # Handle userAction events (bypass LLM)
        if ui_event_part:
            action = ui_event_part.get("name")
            ctx = ui_event_part.get("context", {})
            logger.info(f"Action: {action}, context: {ctx}")

            if action == "page_change":
                await self._handle_pagination(ctx, task, updater, event_queue)
                return

            if action == "search":
                await self._handle_search(ctx, task, updater, event_queue)
                return

            if action == "clear_search":
                await self._handle_clear_search(ctx, task, updater, event_queue)
                return

            query = f"User triggered action: {action} with context: {ctx}"
        else:
            query = context.get_user_input()

        logger.info(f"Query for LLM: '{query}'")

        # Stream agent response
        async for item in agent.stream(query, task.context_id):
            is_complete = item["is_task_complete"]

            if not is_complete:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(item["updates"], task.context_id, task.id),
                )
                continue

            content = item["content"]
            final_parts = []

            if "---a2ui_JSON---" in content:
                text_content, json_string = content.split("---a2ui_JSON---", 1)

                if text_content.strip():
                    final_parts.append(Part(root=TextPart(text=text_content.strip())))

                if json_string.strip():
                    try:
                        json_cleaned = json_string.strip().lstrip("```json").rstrip("```").strip()
                        json_data = json.loads(json_cleaned)

                        if isinstance(json_data, list):
                            for message in json_data:
                                final_parts.append(create_a2ui_part(message))
                        else:
                            final_parts.append(create_a2ui_part(json_data))

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse UI JSON: {e}")
                        final_parts.append(Part(root=TextPart(text=json_string)))
            else:
                final_parts.append(Part(root=TextPart(text=content.strip())))

            await updater.update_status(
                TaskState.input_required,
                new_agent_parts_message(final_parts, task.context_id, task.id),
            )
            break

    async def _handle_pagination(
        self, ctx: dict, task: Task, updater: TaskUpdater, event_queue: EventQueue,
    ) -> None:
        """Handle page_change WITHOUT calling LLM — instant response."""
        query_id = ctx.get("queryId")
        direction = ctx.get("direction")
        current_page = ctx.get("currentPage", 1)

        logger.info(f"PAGINATION: query={query_id}, direction={direction}, page={current_page}")

        if not query_id:
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message("Pagination error: missing query ID.", task.context_id, task.id),
            )
            return

        try:
            session_manager = get_session_manager()
            if direction == "next":
                new_page = current_page + 1
            elif direction == "previous":
                new_page = max(1, current_page - 1)
            else:
                new_page = current_page

            result = session_manager.fetch_page(task.context_id, query_id, new_page)

            data_update = {
                "dataModelUpdate": {
                    "surfaceId": "sql-results",
                    "contents": [
                        {"key": "queryId", "valueString": query_id},
                        {"key": "currentPage", "valueNumber": result["page"]},
                        {"key": "totalPages", "valueNumber": result["total_pages"]},
                        {"key": "totalCount", "valueNumber": result["total_count"]},
                        {"key": "rows", "valueArray": result["rows"]},
                    ],
                }
            }

            await updater.update_status(
                TaskState.input_required,
                new_agent_parts_message([create_a2ui_part(data_update)], task.context_id, task.id),
            )

        except ValueError as e:
            logger.error(f"Pagination error: {e}")
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(f"Pagination error: {e}", task.context_id, task.id),
            )

    async def _handle_search(
        self, ctx: dict, task: Task, updater: TaskUpdater, event_queue: EventQueue,
    ) -> None:
        """Handle search WITHOUT calling LLM — applies SQL WHERE filter."""
        query_id = ctx.get("queryId")
        search_term = ctx.get("searchTerm", "")
        search_column = ctx.get("searchColumn")

        if not query_id or not search_term:
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message("Search error: missing query ID or search term.", task.context_id, task.id),
            )
            return

        try:
            session_manager = get_session_manager()
            result = session_manager.apply_search(task.context_id, query_id, search_term, search_column)

            data_update = {
                "dataModelUpdate": {
                    "surfaceId": "sql-results",
                    "contents": [
                        {"key": "queryId", "valueString": query_id},
                        {"key": "currentPage", "valueNumber": result["page"]},
                        {"key": "totalPages", "valueNumber": result["total_pages"]},
                        {"key": "totalCount", "valueNumber": result["total_count"]},
                        {"key": "searchTerm", "valueString": search_term},
                        {"key": "rows", "valueArray": result["rows"]},
                    ],
                }
            }

            await updater.update_status(
                TaskState.input_required,
                new_agent_parts_message([create_a2ui_part(data_update)], task.context_id, task.id),
            )

        except ValueError as e:
            logger.error(f"Search error: {e}")
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(f"Search error: {e}", task.context_id, task.id),
            )

    async def _handle_clear_search(
        self, ctx: dict, task: Task, updater: TaskUpdater, event_queue: EventQueue,
    ) -> None:
        """Handle clear_search WITHOUT calling LLM — restores original query."""
        query_id = ctx.get("queryId")

        if not query_id:
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message("Error: missing query ID.", task.context_id, task.id),
            )
            return

        try:
            session_manager = get_session_manager()
            result = session_manager.clear_search(task.context_id, query_id)

            data_update = {
                "dataModelUpdate": {
                    "surfaceId": "sql-results",
                    "contents": [
                        {"key": "queryId", "valueString": query_id},
                        {"key": "currentPage", "valueNumber": result["page"]},
                        {"key": "totalPages", "valueNumber": result["total_pages"]},
                        {"key": "totalCount", "valueNumber": result["total_count"]},
                        {"key": "searchTerm", "valueString": ""},
                        {"key": "rows", "valueArray": result["rows"]},
                    ],
                }
            }

            await updater.update_status(
                TaskState.input_required,
                new_agent_parts_message([create_a2ui_part(data_update)], task.context_id, task.id),
            )

        except ValueError as e:
            logger.error(f"Clear search error: {e}")
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(f"Error: {e}", task.context_id, task.id),
            )

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
