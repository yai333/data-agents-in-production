"""Minimal web server for the SQL agent.

Uses aiohttp to serve a frontend and expose the agent as a REST API.
The frontend renders query results and Vega-Lite charts in two tabs.

Endpoints:
    GET  /           — Frontend HTML page
    POST /api/ask    — Run the agent on a question
    GET  /api/health — Health check

See 3.6 for the chart generation pipeline.
"""

import json
import logging
import time
from pathlib import Path

from aiohttp import web
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import SystemMessage, HumanMessage

from src.observability.tracing import get_langfuse_callback

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


async def handle_index(request: web.Request) -> web.Response:
    """Serve the frontend HTML page."""
    env: Environment = request.app["jinja_env"]
    template = env.get_template("index.html")
    html = template.render()
    return web.Response(text=html, content_type="text/html")


async def handle_ask(request: web.Request) -> web.Response:
    """Run the agent on a question and return results as JSON.

    Request body:
        {"question": "How many customers are there?"}

    Response body:
        {"response": "...", "sql": "...", "chart_spec": {...}, ...}
    """
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "Invalid JSON in request body"}, status=400
        )

    question = data.get("question", "").strip()
    if not question:
        return web.json_response(
            {"error": "Missing 'question' field"}, status=400
        )

    logger.info(f"Question: {question}")

    try:
        agent = request.app["agent"]

        initial_state = {
            "original_question": question,
            "disambiguated_question": question,
            "cache_hit": False,
            "cached_sql": "",
            "tables_used": [],
            "schema_overview": "",
            "sql": "",
            "results": "",
            "evaluation_score": 0.0,
            "response": "",
            "messages": [
                SystemMessage(content="You are a SQL agent."),
                HumanMessage(content=question),
            ],
            "chart_spec": None,
            "chart_reasoning": None,
        }

        # Langfuse tracing (carries forward from 3.4/3.5)
        try:
            handler, langfuse_metadata, trace_id = get_langfuse_callback(
                session_id="chapter-3-6-web",
                tags=["chapter-3-6", "web", "chart"],
            )
            config = {
                "configurable": {"thread_id": f"web-{int(time.time())}"},
                "callbacks": [handler],
                "metadata": langfuse_metadata,
                "recursion_limit": 50,
            }
        except Exception:
            config = {
                "configurable": {"thread_id": f"web-{int(time.time())}"},
                "recursion_limit": 50,
            }

        result = await agent.ainvoke(initial_state, config)

        return web.json_response({
            "response": result.get("response", ""),
            "sql": result.get("sql", ""),
            "chart_spec": result.get("chart_spec"),
            "chart_reasoning": result.get("chart_reasoning"),
        }, dumps=lambda obj: json.dumps(obj, default=str))
    except Exception as e:
        logger.exception("Agent error")
        return web.json_response(
            {"error": f"Agent error: {type(e).__name__}: {e}"}, status=500
        )


async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "ok"})


def create_app(agent) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent: Compiled LangGraph agent

    Returns:
        Configured aiohttp Application
    """
    app = web.Application()
    app["agent"] = agent
    app["jinja_env"] = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )

    app.router.add_get("/", handle_index)
    app.router.add_post("/api/ask", handle_ask)
    app.router.add_get("/api/health", handle_health)

    return app
