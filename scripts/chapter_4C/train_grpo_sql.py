"""Launch GRPO SQL agent training via VERL + Agent Lightning.

This script validates training data, defines the VERL config as a Python
dict, wraps the SQL agent as a LitAgent, and trains with
agl.Trainer(...).fit().

The training graph mirrors the production agent (chapters 3.5 and 4A):
  generate_sql ↔ all_tools → respond_sql

Two-phase schema retrieval for scalability:
  Phase 1 (automatic): search_tables returns table names + descriptions
  Phase 2 (agent-driven): model calls get_table_details for columns

Schema exploration tools (same as production MDL layer):
  - get_table_details: columns, types, relationships for one table
  - get_metrics: aggregation patterns and KPIs for a table
  - get_relationships: join paths from a table
  - get_glossary_entries: business term definitions
  - get_additional_descriptions: business context (fiscal year, conventions)

SQL generation tools (same as production):
  - sql_db_query_checker: validate SQL syntax
  - sql_db_query: execute SQL against the database
  - TextToSQLResult: structured output with SQL + explanation

Same prompt as APO output (chapter 4A):
  Loaded from experiments/apo_results/best_prompt.txt

The model learns to explore schema, generate correct SQL, and
self-correct — all through GRPO with execution-based reward.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence, cast

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import agentlightning as agl
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from scripts.chapter_4C.reward_sql import compute_sql_reward
from src.schema.render import render_schema_summary, render_table_card
from src.schema.store import SchemaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PROMPT_PATH = ROOT / "scripts" / "chapter_4C" / "train_prompt.txt"

REQUIRED_KEYS = {"question", "gold_sql"}


# ── Structured output tool (same as chapters 3.5 and 4A) ──

class TextToSQLResult(BaseModel):
    """Structured output from the SQL generation node."""
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(
        description="Brief explanation of the query logic")


# ── Agent state ───────────────────────────────────────────

class TrainingAgentState(TypedDict):
    """State for the GRPO training agent."""
    original_question: str
    sql: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


def load_system_prompt(prompt_path: str | None = None) -> str:
    """Load system prompt from file.

    Priority:
      1. Explicit --prompt argument
      2. Training prompt at scripts/chapter_4C/train_prompt.txt
    """
    if prompt_path:
        p = Path(prompt_path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p.read_text().strip()

    if TRAIN_PROMPT_PATH.exists():
        return TRAIN_PROMPT_PATH.read_text().strip()

    raise FileNotFoundError(
        f"Training prompt not found at {TRAIN_PROMPT_PATH}. "
        f"Use --prompt to specify a prompt file."
    )


# ── VERL config (Python dict, not YAML) ───────────────────
#
# Training mode: transition-level (default).
#
# Why not trajectory-level? The SQL agent uses Hermes-format tool calls
# (<tool_call> tags), which are prone to retokenization mismatch when
# stitching multi-turn conversations into a single token sequence.
# The Agent Lightning blog documents three failure modes that affect us:
#   1. Retokenization context mismatch: BPE merges tool-call tags
#      differently in isolation vs. in a full conversation
#   2. Chat template drift: Hermes turn-boundary tokens span response/
#      template boundaries, making binary mask assignment impossible
#   3. Length limit truncation: trajectory mode needs explicit
#      trajectory_max_* settings that are hard to size correctly
#
# With transition-level, each generate turn is a separate training sample.
# The model still sees full conversation history during rollout (the
# LangGraph message list carries all prior turns). What we lose is
# cross-turn gradient flow — the gradient for turn 3's SQL revision
# doesn't flow through turn 2's error. For 5-10 turn SQL trajectories,
# this is an acceptable trade-off vs. the fragility of trajectory mode.
#
# See: https://agent-lightning.github.io/posts/trajectory_level_aggregation/
VERL_CONFIG = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/sql_train.jsonl",
        "val_files": "data/sql_val.jsonl",
        "train_batch_size": 8,
        "max_prompt_length": 8192,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "name": "vllm",
            "n": 2,
            "tensor_model_parallel_size": 1,
            "gpu_memory_utilization": 0.4,
            "log_prob_micro_batch_size_per_gpu": 2,
            "multi_turn": {"format": "hermes"},
            "enforce_eager": True,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "qwen25_coder",
                    "max_model_len": 4096,
                    "enforce_eager": True,
                    "num_gpu_blocks_override": 256,
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 16,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_max_token_len_per_gpu": 8192,
            "optim": {"lr": 1e-5},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 2,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": os.getenv(
                "MODEL_PATH", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            ),
            "lora_rank": 16,
            "lora_alpha": 32,
            "use_remove_padding": False,
            "enable_gradient_checkpointing": True,
            "override_config": {
                "attn_implementation": "sdpa",
            },
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "total_epochs": 3,
        "val_before_train": True,
        "test_freq": 50,
        "project_name": "sql-agent-grpo",
        "logger": ["console", "wandb"],
    },
}


def build_schema_context(store: SchemaStore) -> str:
    """Build compact schema overview from SchemaStore.

    Returns table names and descriptions only. The model calls
    get_table_details to drill into specific tables it needs.
    """
    tables = store.get_all_tables()
    return render_schema_summary(tables)


def create_mdl_tools(store: SchemaStore) -> list:
    """Create MDL tools for schema exploration.

    Same tools as the production agent (chapters 3.5 and 4A),
    wrapping the sync SchemaStore for training.
    """

    @tool
    def get_table_details(table_name: str) -> str:
        """Get full schema for a table: columns, types, relationships,
        and sample values. Call this before writing SQL to learn the
        exact column names."""
        table = store.get_table(table_name)
        if not table:
            return f"Table '{table_name}' does not exist."
        return render_table_card(table)

    @tool
    def get_metrics(table_name: str) -> str:
        """Get metric definitions (aggregation patterns, KPIs) for a
        table. Use when the question mentions calculated values like
        revenue, count, average."""
        results = store.get_metrics(table_name)
        if not results:
            return f"No metrics defined for '{table_name}'."
        lines = [f"Metrics for {table_name}:"]
        for m in results:
            lines.append(
                f"  - {m['name']}: {m.get('description', '')} "
                f"[{m.get('sql_pattern', '')}]"
            )
        return "\n".join(lines)

    @tool
    def get_relationships(table_name: str) -> str:
        """Get join paths from a table to related tables. Use to
        discover foreign keys and join conditions."""
        rels = store.get_relationships(table_name)
        if not rels:
            return f"No relationships found for '{table_name}'."
        lines = [f"Relationships for {table_name}:"]
        for r in rels:
            lines.append(f"  - {r.name} ({r.join_type}): {r.condition}")
        return "\n".join(lines)

    @tool
    def get_glossary_entries(terms: str) -> str:
        """Look up business term definitions from the glossary.
        Pass comma-separated terms. Use when the question contains
        domain-specific terms like 'active customer' or 'revenue'."""
        term_list = [t.strip().lower() for t in terms.split(",")]
        results = store.get_glossary_entries(term_list)
        if not results:
            return f"No definitions found for: {terms}"
        lines = ["Glossary definitions:"]
        for r in results:
            lines.append(f"  - {r['term']}: {r['definition']}")
        return "\n".join(lines)

    @tool
    def get_additional_descriptions() -> str:
        """Get business context: fiscal year definitions, time
        conventions, default values. Use when the question mentions
        'Q3', 'recent', 'top', or other terms that need business
        rules to interpret correctly."""
        descs = store.get_additional_descriptions()
        if not descs:
            return "No additional business context available."
        lines = ["Business context:"]
        for d in descs:
            lines.append(f"  - {d}")
        return "\n".join(lines)

    return [get_table_details]


def build_sql_training_graph(
    llm,
    system_prompt: str,
    mdl_tools: list,
):
    """Build the SQL training graph for GRPO.

    Minimal tool set for 4096-token context:
      generate_sql ↔ tools → respond_sql

    Only get_table_details + TextToSQLResult. No sql_db_query,
    no checker, no retry loops. The reward function evaluates
    SQL independently after the rollout.

    Args:
        llm: Per-rollout chat model (changes each rollout).
        system_prompt: Injected at each generate_sql call.
        mdl_tools: Schema exploration tools (get_table_details only).
    """
    all_tools = mdl_tools

    model_with_tools = llm.bind_tools(
        all_tools + [TextToSQLResult],
    )

    def generate_sql_node(state):
        """Generate SQL using the APO-optimised prompt."""
        return {
            "messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=system_prompt)]
                    + list(state.get("messages", []))
                )
            ]
        }

    def should_continue_sql(state):
        """Route: tool calls → tools, TextToSQLResult → respond."""
        messages = state.get("messages", [])
        if not messages:
            return "respond_sql"
        last = messages[-1]
        if not hasattr(last, "tool_calls") or not last.tool_calls:
            return "respond_sql"
        # TextToSQLResult means we're done generating
        if any(
            tc.get("name") == "TextToSQLResult"
            for tc in last.tool_calls
        ):
            return "respond_sql"
        return "tools"

    def respond_sql_node(state):
        """Extract SQL from TextToSQLResult tool call."""
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        sql = ""

        if last and hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                if tc.get("name") == "TextToSQLResult":
                    args = tc.get("args", {})
                    sql = args.get("sql", "")
                    break

        return {"sql": sql}

    graph = StateGraph(TrainingAgentState)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("tools", ToolNode(
        all_tools, handle_tool_errors=True,
    ))
    graph.add_node("respond_sql", respond_sql_node)

    graph.set_entry_point("generate_sql")
    graph.add_conditional_edges(
        "generate_sql",
        should_continue_sql,
        {"tools": "tools", "respond_sql": "respond_sql"},
    )
    graph.add_edge("tools", "generate_sql")
    graph.add_edge("respond_sql", END)

    return graph.compile()


class LitSQLAgent(agl.LitAgent[dict[str, Any]]):
    """LitAgent wrapper that runs the SQL training graph.

    Reusable objects (DB, tools, schema context) are created once in
    __init__. Only the chat_model and checker tool change per rollout.
    """

    def __init__(
        self,
        store: SchemaStore,
        db_url: str,
        system_prompt: str,
    ):
        super().__init__()
        self.db_url = db_url
        self.system_prompt = system_prompt

        # Created once, reused across all rollouts
        self.mdl_tools = create_mdl_tools(store)
        self.schema_context = build_schema_context(store)

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        try:
            return self._run_rollout(task, resources, rollout)
        except GraphRecursionError:
            logger.warning(
                "Recursion limit hit for: %s", task["question"][:80],
            )
            return 0.0
        except Exception:
            logger.exception(
                "Rollout failed for: %s", task["question"][:80],
            )
            return 0.0

    def _run_rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        endpoint = llm.get_base_url(
            rollout.rollout_id, rollout.attempt.attempt_id,
        )

        chat_model = init_chat_model(
            llm.model,
            model_provider="openai",
            base_url=endpoint,
            api_key=llm.api_key or "dummy",
            temperature=(
                llm.sampling_parameters.get("temperature", 1.0)
                if rollout.mode == "train" else 0.0
            ),
            max_retries=0,
        )

        graph = build_sql_training_graph(
            chat_model, self.system_prompt,
            self.mdl_tools,
        )

        handler = self.tracer.get_langchain_handler()
        config = {"callbacks": [handler] if handler else []}
        config["recursion_limit"] = 10  # ~5 tool rounds max (get_table_details calls)

        result = graph.invoke(
            {
                "original_question": task["question"],
                "messages": [
                    SystemMessage(content=self.schema_context),
                    ("user", f"Question: {task['question']}"),
                ],
            },
            config,
        )

        # Debug: log message flow to verify tool calling
        for i, msg in enumerate(result.get("messages", [])):
            role = getattr(msg, "type", "?")
            tc = getattr(msg, "tool_calls", None)
            content_preview = str(getattr(msg, "content", ""))[:80]
            if tc:
                tool_names = [t.get("name", "?") for t in tc]
                logger.info(
                    "  msg[%d] %s tool_calls=%s", i, role, tool_names,
                )
            else:
                logger.info(
                    "  msg[%d] %s content=%s", i, role, content_preview,
                )

        generated_sql = self._extract_sql(result)

        reward = compute_sql_reward(
            generated_sql,
            task["gold_sql"],
            self.db_url,
        )

        logger.info(
            "reward=%.3f exec_match=%s comp_f1=%.3f executed=%s "
            "format_ok=%s q=%s sql=%s err=%s",
            reward["total"],
            reward["execution_match"],
            reward["component_f1"],
            reward["executed"],
            reward["format_ok"],
            task["question"][:60],
            generated_sql[:120] if generated_sql else "<empty>",
            reward.get("detail", "")[:120] if reward.get("detail") else "",
        )

        return reward["total"]

    @staticmethod
    def _extract_sql(result: dict) -> str:
        """Extract SQL from graph result with three fallback levels."""
        # Primary: TextToSQLResult (via respond_sql_node)
        sql = result.get("sql", "")
        if sql:
            return sql

        # Fallback 1: extract from last message content
        from scripts.chapter_4C.reward_sql import extract_sql
        last_content = result["messages"][-1].content
        if last_content:
            sql = extract_sql(last_content)
            if sql:
                return sql

        # Fallback 2: last sql_db_query tool call args.
        # Early in training the model may execute correct SQL but
        # skip TextToSQLResult. Without this fallback every such
        # rollout scores 0.0, stalling learning.
        for msg in reversed(result["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "sql_db_query":
                        sql = tc.get("args", {}).get("query", "")
                        if sql:
                            return sql

        return ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_examples(
    examples: list[dict[str, Any]],
    split_name: str,
) -> None:
    """Validate required fields in dataset examples."""
    if not examples:
        raise ValueError(f"{split_name} dataset is empty")

    for i, ex in enumerate(examples):
        missing = REQUIRED_KEYS - set(ex.keys())
        if missing:
            raise ValueError(
                f"{split_name} example {i} missing keys: {sorted(missing)}"
            )
        if not isinstance(ex["gold_sql"], str) or not ex["gold_sql"].strip():
            raise ValueError(
                f"{split_name} example {i} has empty gold_sql"
            )


def train(
    train_path: str,
    val_path: str,
    db_url: str,
    model_path: str | None = None,
    prompt_path: str | None = None,
    dry_run: bool = False,
    base_url: str | None = None,
    dry_run_model: str | None = None,
    trace: bool = False,
) -> None:
    """Preflight + launch GRPO training.

    With --dry-run: uses Trainer.dev() to exercise the full pipeline
    (rollouts, reward, tracing) with a lightweight Baseline algorithm
    instead of VERL. Validates DB connections, LangGraph control flow,
    and confirms rewards are non-zero — without GPU time.
    """
    if trace:
        agl.setup_logging("DEBUG")
        logger.info("Tracing enabled — DEBUG-level AGL logging active")

    train_p = Path(train_path)
    val_p = Path(val_path)

    if not train_p.exists():
        raise FileNotFoundError(f"Train file not found: {train_p}")
    if not val_p.exists():
        raise FileNotFoundError(f"Validation file not found: {val_p}")

    train_data = load_jsonl(train_p)
    val_data = load_jsonl(val_p)
    validate_examples(train_data, "train")
    validate_examples(val_data, "val")

    print(f"Train examples: {len(train_data)}")
    print(f"Val examples:   {len(val_data)}")

    system_prompt = load_system_prompt(prompt_path)
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"  Source: {prompt_path or TRAIN_PROMPT_PATH}")

    # Initialise reward DB connection and schema store
    from scripts.chapter_4C.reward_sql import init_db
    init_db(db_url)

    store = SchemaStore(ROOT / "config" / "chinook_schema.json")
    agent = LitSQLAgent(store, db_url, system_prompt)

    if dry_run:
        # Trainer.dev() uses Baseline algorithm — runs rollouts through
        # the full infrastructure (store, runners, hooks, tracer) without
        # standing up VERL/GRPO. Confirms DB connections, graph execution,
        # and non-zero rewards.
        #
        # Because there is no VERL to spin up vLLM, we provide an LLM
        # resource pointing at an OpenAI-compatible endpoint so rollouts
        # have an LLM to call. Uses agl.LLM (not ProxyLLM) to match the
        # official Agent Lightning pattern for dev mode.
        #
        # Default: local vLLM server at localhost:8000 serving the SFT
        # checkpoint. Start with: make vllm-serve
        # Default to local vLLM server, not OPENAI_API_BASE (which
        # points at cloud APIs that don't serve the fine-tuned model).
        endpoint = base_url or "http://localhost:8000/v1"
        model_name = (
            dry_run_model
            or os.getenv("DRY_RUN_MODEL", "sql-agent")
        )

        print("\n" + "=" * 60)
        print("DRY RUN: exercising pipeline with Trainer.dev()")
        print(f"  endpoint: {endpoint}")
        print(f"  model:    {model_name}")
        print("=" * 60)

        dev_data = train_data[:8]  # Small sample for quick validation

        dev_llm = agl.LLM(
            endpoint=endpoint,
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            sampling_parameters={"temperature": 0.7},
        )

        trainer = agl.Trainer(
            n_workers=1,
            adapter={"agent_match": r"generate_sql"},
            initial_resources={"main_llm": dev_llm},
        )
        trainer.dev(
            agent,
            train_dataset=dev_data,
        )
        print("\nDry run complete — pipeline validated.")
        return

    # Build VERL config — override data paths and model from CLI args
    import copy
    config = copy.deepcopy(VERL_CONFIG)
    config["data"]["train_files"] = train_path
    config["data"]["val_files"] = val_path

    if model_path:
        config["actor_rollout_ref"]["model"]["path"] = model_path

    # Scale down for small datasets
    n_train = len(train_data)
    if n_train <= 32:
        config["data"]["train_batch_size"] = min(
            config["data"]["train_batch_size"], n_train,
        )
        config["trainer"]["total_epochs"] = 1
        config["trainer"]["val_before_train"] = False

    print(f"\nVERL config: batch={config['data']['train_batch_size']}, "
          f"epochs={config['trainer']['total_epochs']}, "
          f"model={config['actor_rollout_ref']['model']['path']}")

    algorithm = agl.VERL(config)
    trainer = agl.Trainer(
        n_runners=1,
        algorithm=algorithm,
        adapter={"agent_match": r"generate_sql"},
    )
    trainer.fit(
        agent,
        train_dataset=train_data,
        val_dataset=val_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SQL agent with GRPO via VERL + Agent Lightning",
    )
    parser.add_argument(
        "--train", default="data/sql_train.jsonl",
        help="Training data JSONL path",
    )
    parser.add_argument(
        "--val", default="data/sql_val.jsonl",
        help="Validation data JSONL path",
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://postgres:postgres@localhost:5432/chinook",
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model path (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="System prompt file (default: experiments/apo_results/best_prompt.txt)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM endpoint for dry-run (default: http://localhost:8000/v1). "
             "Start a server with: make vllm-serve",
    )
    parser.add_argument(
        "--dry-run-model",
        default=None,
        help="Model name for dry-run (default: sql-agent for local vLLM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs only, do not train",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable DEBUG-level tracing for Agent Lightning",
    )
    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        db_url=args.db_url,
        model_path=args.model,
        prompt_path=args.prompt,
        dry_run=args.dry_run,
        base_url=args.base_url,
        dry_run_model=args.dry_run_model,
        trace=args.trace,
    )
