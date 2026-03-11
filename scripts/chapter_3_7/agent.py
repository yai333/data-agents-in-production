"""SQL Explorer Agent — uses ADK LlmAgent to generate SQL and A2UI to display results.

Two modes:
- UI mode: generates A2UI JSON for interactive DataTable rendering
- Text mode: generates markdown tables for CLI usage

Uses LiteLLM for model abstraction, so works with OpenAI, Gemini, Anthropic, etc.
"""

import json
import logging
import os
from collections.abc import AsyncIterable
from typing import Any

import jsonschema
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from a2ui_examples import SQL_EXPLORER_UI_EXAMPLES
from a2ui_schema import A2UI_SCHEMA
from prompt_builder import get_text_prompt, get_ui_prompt
from tools import execute_sql_query, get_database_schema, get_sample_data

logger = logging.getLogger(__name__)


class SQLExplorerAgent:
    """An agent that helps users explore SQL databases with A2UI visualization."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, base_url: str, use_ui: bool = False):
        self.base_url = base_url
        self.use_ui = use_ui
        self._agent = self._build_agent(base_url, use_ui)
        self._user_id = "sql_explorer_user"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        try:
            single_message_schema = json.loads(A2UI_SCHEMA)
            self.a2ui_schema_object = {"type": "array", "items": single_message_schema}
            logger.info("A2UI schema loaded for validation.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse A2UI schema: {e}")
            self.a2ui_schema_object = None

    def get_processing_message(self) -> str:
        return "Analyzing your request and querying the database..."

    def _build_agent(self, base_url: str, use_ui: bool) -> LlmAgent:
        """Build the LLM agent for SQL exploration."""
        model_name = os.getenv("LITELLM_MODEL", "openai/gpt-5.1-mini")

        if use_ui:
            instruction = get_ui_prompt(base_url, SQL_EXPLORER_UI_EXAMPLES)
        else:
            instruction = get_text_prompt()

        return LlmAgent(
            model=LiteLlm(model=model_name),
            name="sql_explorer_agent",
            description="An agent that explores SQL databases and displays results with rich UI.",
            instruction=instruction,
            tools=[get_database_schema, execute_sql_query, get_sample_data],
        )

    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        """Stream the agent response for a given query."""
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )

        max_retries = 1
        attempt = 0
        current_query_text = query

        if self.use_ui and self.a2ui_schema_object is None:
            logger.error("A2UI schema not loaded.")
            yield {"is_task_complete": True, "content": "Internal configuration error."}
            return

        while attempt <= max_retries:
            attempt += 1
            logger.info(f"--- Agent attempt {attempt}/{max_retries + 1} ---")

            current_message = types.Content(
                role="user", parts=[types.Part.from_text(text=current_query_text)]
            )
            final_response_content = None

            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session.id,
                new_message=current_message,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts and event.content.parts[0].text:
                        final_response_content = "\n".join(
                            [p.text for p in event.content.parts if p.text]
                        )
                    break
                else:
                    yield {"is_task_complete": False, "updates": self.get_processing_message()}

            if final_response_content is None:
                if attempt <= max_retries:
                    current_query_text = f"No response received. Please retry: '{query}'"
                    continue
                else:
                    final_response_content = "Sorry, I couldn't process your request."

            is_valid = False
            error_message = ""

            if self.use_ui:
                try:
                    if "---a2ui_JSON---" not in final_response_content:
                        raise ValueError("Delimiter '---a2ui_JSON---' not found.")

                    _, json_string = final_response_content.split("---a2ui_JSON---", 1)
                    if not json_string.strip():
                        raise ValueError("JSON part is empty.")

                    json_cleaned = json_string.strip().lstrip("```json").rstrip("```").strip()
                    if not json_cleaned:
                        raise ValueError("Cleaned JSON string is empty.")

                    parsed_json = json.loads(json_cleaned)
                    jsonschema.validate(instance=parsed_json, schema=self.a2ui_schema_object)
                    is_valid = True

                except (ValueError, json.JSONDecodeError, jsonschema.exceptions.ValidationError) as e:
                    logger.warning(f"UI validation failed: {e}")
                    error_message = f"Validation failed: {e}"
            else:
                is_valid = True

            if is_valid:
                yield {"is_task_complete": True, "content": final_response_content}
                return

            if attempt <= max_retries:
                current_query_text = (
                    f"Your previous response was invalid. {error_message} "
                    "Please generate a valid A2UI JSON response. "
                    f"Original request: '{query}'"
                )

        yield {
            "is_task_complete": True,
            "content": "Sorry, I'm having trouble generating the UI. Please try again.",
        }
