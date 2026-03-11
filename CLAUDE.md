# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Companion code for the book **"Data Agents in Production: Build, Deploy, and Scale AI-Powered Analytics with LangGraph"**. Each chapter builds a layer of a production-grade natural-language-to-SQL system — from schema representation through agent loops, security, observability, and fine-tuning.

The book content lives in a separate repo. Code paths referenced in chapters (e.g. `src/adapters/base.py`) refer to this repository.

## Running Python

**ALWAYS run Python using the project's virtual environment:**

```bash
source .venv/bin/activate && python <script.py>
```

The project uses Python 3.12 with modern syntax (`str | None`, `list[str]`). Never use the system Python directly.

## Commands

```bash
# Install for development
make dev

# Run tests
make test              # All tests
make test-unit         # Unit tests only (no API keys needed)
make test-integration  # Integration tests (requires API keys)
pytest tests/test_adapters.py -v  # Single test file

# Code quality
make lint              # ruff + mypy
make format            # Auto-fix with ruff

# Database
make db-up             # Start Postgres with Chinook
make db-down           # Stop database
make db-shell          # psql into Chinook

# Run chapter scripts
make chapter-1_2       # Runs scripts/run_chapter_1_2.py
```

## Architecture

### Provider Adapter Pattern
All LLM interactions go through `src/adapters/`:
- `base.py`: Abstract `LLMAdapter` interface with `generate()`, `generate_structured()`, `generate_with_tools()`
- `openai_adapter.py` / `gemini_adapter.py`: Provider implementations
- `factory.py`: `create_adapter(provider="openai")` creates the right adapter

Code must work with both OpenAI and Gemini. Use `create_adapter()` instead of direct provider imports.

### Configuration
- `config/default.yaml`: Settings with `${ENV_VAR:default}` substitution
- `src/utils/config.py`: `load_config()` returns typed `Settings` object
- `.env`: API keys (copy from `.env.example`)

## Code Changes

- Prefer **minimal targeted edits**. Do NOT refactor surrounding code, add abstractions, or change unrelated defaults.
- If asked to change one value, change only that value. Do not "improve" nearby code.
- Do not add error handling, type annotations, or comments to code you did not change.

## Task Completion

- When a task has multiple phases or files, complete **ALL phases** before declaring done.
- Do not stop after the first phase and wait for the user to say "continue".
- If the task scope is clear, execute it fully in one pass.

## Debugging

- When debugging deployment or API errors, check **environment variables**, database settings that may override env vars, and existing configuration files **BEFORE** making code changes.
- Many production issues stem from config, not code.

## Module Structure

```
src/
├── adapters/       # LLM provider abstraction (OpenAI, Gemini)
├── agent/          # LangGraph agent: graph, error classifier, repair, retry, CLI
├── ambiguity/      # Ambiguity detection and clarification
├── api/            # FastAPI endpoint for the agent
├── authorization/  # Row-level security and access control
├── chart/          # Vega-Lite chart generation from query results
├── context/        # Context engineering (schema + business context assembly)
├── execution/      # SQL execution with guardrails (timeout, row limit, read-only)
├── observability/  # Structured logging and tracing
├── pii/            # PII detection and anonymisation (Presidio)
├── reasoning/      # Prompt templates and chain-of-thought patterns
├── retrieval/      # Few-shot retrieval (BM25, semantic, hybrid with RRF)
├── schema/         # TableCard/ColumnCard schema representation, SchemaStore
├── security/       # Threat modelling utilities
├── structured/     # Structured generation (SQL output parsing)
├── utils/          # Configuration, logging helpers
├── validators/     # SQL validation (schema, safety, join paths)
└── vllm_plugins/   # Custom vLLM parser for Qwen2.5-Coder

scripts/
├── run_chapter_*.py        # Per-chapter runnable scripts
├── chapter_3_2/            # PII protection demo
├── chapter_3_7/            # A2UI (Agent-to-UI) demo app
├── chapter_4A/             # APO (Automated Prompt Optimisation)
├── chapter_4B/             # SFT fine-tuning notebooks
├── chapter_4C/             # GRPO reinforcement learning training
└── data/                   # Data preparation utilities

evals/              # Evaluation framework with golden sets and metrics
data/               # Schema JSON, golden sets, training data
config/             # YAML configuration with env-var substitution
```
