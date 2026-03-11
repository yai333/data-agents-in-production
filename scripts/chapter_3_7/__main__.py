"""SQL Explorer Agent — A2A server entry point.

Starts a Starlette-based A2A server with:
- A2UI extension for interactive DataTable rendering
- CORS middleware for local development
- Two agent modes: UI (A2UI DataTable) and text-only (markdown)

Usage:
    python -m src.a2ui_explorer            # Start A2A server on localhost:10003
    python -m src.a2ui_explorer --port 8080  # Custom port
"""

import logging
import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2ui.extension.a2ui_extension import get_a2ui_agent_extension
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

from agent import SQLExplorerAgent
from agent_executor import SQLExplorerAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    """Get the path to the Chinook database."""
    db_path = os.getenv("CHINOOK_DB_PATH")
    if db_path and os.path.exists(db_path):
        return db_path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "data", "chinook.db")


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10003)
def main(host, port):
    # Check for Chinook database
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        logger.error(
            f"Chinook database not found at {db_path}. "
            "Run: python scripts/run_chapter_3_7.py --setup-db"
        )
        raise SystemExit(1)

    capabilities = AgentCapabilities(
        streaming=True,
        extensions=[get_a2ui_agent_extension()],
    )

    skill = AgentSkill(
        id="sql_explorer",
        name="SQL Database Explorer",
        description="Explore and query the Chinook music database with natural language.",
        tags=["sql", "database", "query", "explore"],
        examples=[
            "Show me all tables in the database",
            "List all albums by AC/DC",
            "Find the top 10 customers by total purchases",
            "Show me all tracks longer than 5 minutes",
        ],
    )

    base_url = f"http://{host}:{port}"

    agent_card = AgentCard(
        name="SQL Explorer Agent",
        description=(
            "An agent that helps you explore the Chinook music database. "
            "Ask questions in natural language and see results in an interactive table "
            "with pagination support."
        ),
        url=base_url,
        version="1.0.0",
        default_input_modes=SQLExplorerAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=SQLExplorerAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    agent_executor = SQLExplorerAgentExecutor(base_url=base_url)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    app = server.build()

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://localhost:\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info(f"Starting SQL Explorer Agent at {base_url}")
    logger.info(f"Using database: {db_path}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
