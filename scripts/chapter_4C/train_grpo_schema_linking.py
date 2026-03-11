"""Launch GRPO schema linking training via VERL + Agent Lightning.

Simplified single-turn task: given a question + full DDL context,
predict which tables and columns are needed for the SQL query.

Graph:
  START → schema_context_node → linking_agent_node → END

schema_context_node is deterministic (builds prompt from DDL + glossary).
linking_agent_node is the LLM call that GRPO trains.

Reward: 0.9 * (0.4 * table_F2 + 0.6 * column_F2) + 0.1 * format_score.

Usage:
    # Dry run (local vLLM endpoint)
    python -m scripts.chapter_4C.train_grpo_schema_linking --dry-run

    # Full training
    python -m scripts.chapter_4C.train_grpo_schema_linking --trace
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import agentlightning as agl
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from scripts.chapter_4C.reward_schema_linking import compute_schema_linking_reward
from src.schema.render import render_schema
from src.schema.store import SchemaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_TEMPLATE_PATH = ROOT / "scripts" / "chapter_4C" / "schema_linking_prompt.txt"

REQUIRED_KEYS = {"question", "gold_tables", "gold_columns"}


# ── Agent state ───────────────────────────────────────────

class SchemaLinkingState(TypedDict):
    """State for the schema linking training agent."""
    original_question: str
    predicted_text: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── VERL config ───────────────────────────────────────────
#
# Single-turn task — no tool calling, no multi-turn.
# Prompt is ~4360 tokens (full DDL for all 11 Chinook tables +
# glossary + question + chat template overhead).
# Response is short: just a table/column list (~256 tokens max).
# Total: ~4650 tokens per sequence.
#
VERL_CONFIG = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/schema_linking_train.jsonl",
        "val_files": "data/schema_linking_val.jsonl",
        "train_batch_size": 8,
        "max_prompt_length": 4608,
        "max_response_length": 256,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "name": "vllm",
            "n": 4,                                    # was 2 — match reference, better advantage signal
            "tensor_model_parallel_size": 1,
            "gpu_memory_utilization": 0.4,
            "log_prob_micro_batch_size_per_gpu": 4,    # was 2 — match n=4
            "enforce_eager": True,
            "engine_kwargs": {
                "vllm": {
                    "max_model_len": 5120,
                    "enforce_eager": True,
                    "num_gpu_blocks_override": 384,
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 16,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_max_token_len_per_gpu": 10240,
            "optim": {"lr": 1e-6},                     # was 1e-5 — match reference, grad norm hit 8.5 at 1e-5
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0.01,                     # was 0 — prevent entropy collapse (0.44→0.07 in run 1)
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,                    # was 0.28 — match reference
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 4,    # was 2 — match n=4
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": os.getenv(
                "SFT_MODEL_PATH", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
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
        "total_epochs": 2,                             # was 3 — epoch 3 degraded in run 1, reference uses 2
        "val_before_train": True,
        "test_freq": 16,
        "save_freq": 16,                               # save checkpoint every 16 steps
        "default_local_dir": "checkpoints/schema-linking-grpo",
        "project_name": "schema-linking-grpo",
        "logger": ["console", "wandb"],
    },
}


def load_prompt_template(prompt_path: str | None = None) -> str:
    """Load prompt template from file.

    Priority:
      1. Explicit --prompt argument
      2. Default at scripts/chapter_4C/schema_linking_prompt.txt
    """
    if prompt_path:
        p = Path(prompt_path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p.read_text().strip()

    if PROMPT_TEMPLATE_PATH.exists():
        return PROMPT_TEMPLATE_PATH.read_text().strip()

    raise FileNotFoundError(
        f"Prompt template not found at {PROMPT_TEMPLATE_PATH}. "
        f"Use --prompt to specify a prompt file."
    )


def build_schema_linking_graph(llm):
    """Build the single-turn schema linking graph.

    Two nodes:
      schema_context_node (deterministic): Builds the full prompt from
        pre-computed DDL + glossary stored in the task dict.
      linking_agent_node (LLM call): Predicts tables and columns.

    No tools, no loops, no recursion.
    """

    def schema_context_node(state):
        """Pass through — prompt is already built in the task dict."""
        return {}

    def linking_agent_node(state):
        """Single LLM call — predict tables and columns."""
        response = llm.invoke(list(state.get("messages", [])))
        content = response.content if hasattr(response, "content") else str(response)
        return {
            "messages": [response],
            "predicted_text": content,
        }

    graph = StateGraph(SchemaLinkingState)
    graph.add_node("schema_context", schema_context_node)
    graph.add_node("schema_link", linking_agent_node)

    graph.set_entry_point("schema_context")
    graph.add_edge("schema_context", "schema_link")
    graph.add_edge("schema_link", END)

    return graph.compile()


class LitSchemaLinkingAgent(agl.LitAgent[dict[str, Any]]):
    """LitAgent wrapper for schema linking training.

    The DDL context and glossary are pre-computed in the training data
    (by prepare_schema_linking_data.py). The prompt template is loaded
    once at init.
    """

    def __init__(self, prompt_template: str):
        super().__init__()
        self.prompt_template = prompt_template

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        try:
            return self._run_rollout(task, resources, rollout)
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
        llm_resource: agl.LLM = resources["main_llm"]
        endpoint = llm_resource.get_base_url(
            rollout.rollout_id, rollout.attempt.attempt_id,
        )

        chat_model = init_chat_model(
            llm_resource.model,
            model_provider="openai",
            base_url=endpoint,
            api_key=llm_resource.api_key or "dummy",
            temperature=(
                llm_resource.sampling_parameters.get("temperature", 1.0)
                if rollout.mode == "train" else 0.0
            ),
            max_retries=0,
        )

        graph = build_schema_linking_graph(chat_model)

        # Build the prompt from pre-computed fields in the task.
        # Split into system (instructions + schema) and user (question)
        # so SGLang/vLLM endpoints that require a user message work.
        system_prompt = self.prompt_template.format(
            ddl_context=task.get("ddl_context", ""),
            glossary_context=task.get("glossary_context", "None"),
            question=task["question"],
        )

        handler = self.tracer.get_langchain_handler()
        config = {"callbacks": [handler] if handler else []}

        result = graph.invoke(
            {
                "original_question": task["question"],
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Question: {task['question']}"),
                ],
            },
            config,
        )

        predicted_text = result.get("predicted_text", "")

        # Compute reward
        reward = compute_schema_linking_reward(
            predicted_text,
            task["gold_tables"],
            task["gold_columns"],
        )

        logger.info(
            "reward=%.3f table_f2=%.3f col_f2=%.3f fmt=%.1f q=%s pred=%s",
            reward["total"],
            reward["table_f2"],
            reward["column_f2"],
            reward["format_score"],
            task["question"][:60],
            predicted_text[:120],
        )

        return reward["total"]


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


def train(
    train_path: str,
    val_path: str,
    model_path: str | None = None,
    prompt_path: str | None = None,
    dry_run: bool = False,
    base_url: str | None = None,
    dry_run_model: str | None = None,
    trace: bool = False,
) -> None:
    """Preflight + launch GRPO training for schema linking.

    With --dry-run: uses Trainer.dev() to exercise the full pipeline
    without VERL. Validates prompt formatting, reward computation, and
    model output parsing.
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

    prompt_template = load_prompt_template(prompt_path)
    print(f"Prompt template: {len(prompt_template)} chars")
    print(f"  Source: {prompt_path or PROMPT_TEMPLATE_PATH}")

    agent = LitSchemaLinkingAgent(prompt_template)

    if dry_run:
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

        dev_data = train_data[:8]

        dev_llm = agl.LLM(
            endpoint=endpoint,
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            sampling_parameters={"temperature": 0.7},
        )

        trainer = agl.Trainer(
            n_workers=1,
            adapter={"agent_match": r"schema_link"},
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
        adapter={"agent_match": r"schema_link"},
    )
    trainer.fit(
        agent,
        train_dataset=train_data,
        val_dataset=val_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train schema linking with GRPO via VERL + Agent Lightning",
    )
    parser.add_argument(
        "--train", default="data/schema_linking_train.jsonl",
        help="Training data JSONL path",
    )
    parser.add_argument(
        "--val", default="data/schema_linking_val.jsonl",
        help="Validation data JSONL path",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model path",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Prompt template file",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM endpoint for dry-run (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--dry-run-model",
        default=None,
        help="Model name for dry-run (default: sql-agent for local vLLM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline only, do not train",
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
        model_path=args.model,
        prompt_path=args.prompt,
        dry_run=args.dry_run,
        base_url=args.base_url,
        dry_run_model=args.dry_run_model,
        trace=args.trace,
    )
