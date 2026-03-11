# Data Agents in Production

Companion code for the book *Data Agents in Production: Build, Deploy, and Scale AI-Powered Analytics with LangGraph*. Each chapter builds a layer of a production-grade natural-language-to-SQL system — from schema representation through agent loops, security, observability, and fine-tuning.

The codebase is designed to work with **OpenAI** or **Google Gemini** as the LLM provider and **PostgreSQL** (Chinook sample database) as the data layer.

## Quick start

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- An API key for OpenAI or Google Gemini

### Install

```bash
git clone <repo-url> && cd text-to-sql-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or GOOGLE_API_KEY
```

### Start the database

```bash
make db-up          # Start Postgres with Docker
make db-setup       # Load Chinook schema and data
```

### Verify your setup

```bash
make chapter-1_5    # Runs the environment check script
```

### Run the agent

```bash
make run                                    # Interactive prompt
python -m src.agent.cli "How many artists are there?"   # Direct question
```

## Project structure

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

## Chapter script map

Each chapter has a companion script you can run with `make chapter-<id>`:

| Make target | Script | Chapter |
|---|---|---|
| `make chapter-1_5` | `scripts/run_chapter_1_5.py` | 1.5 Environment Setup |
| `make chapter-2_1` | `scripts/run_chapter_2_1.py` | 2.1 Measuring Before Building |
| `make chapter-2_2` | `scripts/run_chapter_2_2.py` | 2.2 Schema Representation |
| `make chapter-2_3` | `scripts/run_chapter_2_3.py` | 2.3 Retrieval |
| `make chapter-2_4` | `scripts/run_chapter_2_4.py` | 2.4 Structured Generation |
| `make chapter-2_5` | `scripts/run_chapter_2_5.py` | 2.5 Reasoning Patterns |
| `make chapter-2_6` | `scripts/run_chapter_2_6.py` | 2.6 Agent Loop |
| `make chapter-2_7` | `scripts/run_chapter_2_7.py` | 2.7 Handling Ambiguity |
| `make chapter-2_8` | `scripts/run_chapter_2_8.py` | 2.8 Agent Memory |
| `make chapter-3_1` | `scripts/run_chapter_3_1.py` | 3.1 Threat Modelling |
| `make chapter-3_2` | `scripts/run_chapter_3_2.py` | 3.2 PII Protection |
| `make chapter-3_3` | `scripts/run_chapter_3_3.py` | 3.3 Context Engineering |
| `make chapter-3_4` | `scripts/run_chapter_3_4.py` | 3.4 Observability |
| `make chapter-3_5` | `scripts/run_chapter_3_5.py` | 3.5 Evaluation System |
| `make chapter-3_6` | `scripts/run_chapter_3_6.py` | 3.6 Chart Generation |
| `make chapter-3_7` | `scripts/run_chapter_3_7.py` | 3.7 Agent-Driven UI |

## Available make targets

```bash
# Core
make install            # Install the package
make dev                # Install with dev dependencies
make run                # Run the agent interactively
make clean              # Remove build artifacts

# Database
make db-up              # Start Postgres container
make db-down            # Stop Postgres container
make db-reset           # Wipe and restart Postgres
make db-setup           # Load Chinook schema
make db-shell           # Open psql shell

# Quality
make test               # Run all tests
make test-unit          # Unit tests only (no API keys needed)
make test-integration   # Integration tests (needs API keys + database)
make test-cov           # Tests with coverage report
make lint               # ruff + mypy
make format             # Auto-fix with ruff

# Few-shot indexing
make index-examples     # Index examples into pgvector (default provider)

# Chapter 4A — APO
make install-ch4a       # Install APO dependencies
make apo-quick          # Quick APO run (2 beams, 1 round)
make apo-full           # Full APO optimisation

# Chapter 4C — GRPO
make install-ch4c       # Install GRPO dependencies
make vllm-serve         # Serve SFT model with vLLM
make grpo-train         # Full GRPO training run
make grpo-train-small   # Small-scale test run
make grpo-eval          # Compare cloud vs local model
```

## Configuration

**Environment variables** (`.env`): API keys, database URL, provider selection. Copy `.env.example` to get started.

**YAML config** (`config/default.yaml`): Agent behaviour, safety limits, retrieval settings. Supports `${ENV_VAR:default}` substitution.

See `.env.example` for all available options.

## Optional dependency groups

The base install covers chapters 1–3. Later chapters need additional packages:

```bash
pip install -e ".[apo]"       # Chapter 4A: Automated Prompt Optimisation
pip install -e ".[training]"  # Chapter 4B: SFT fine-tuning
pip install -e ".[grpo]"      # Chapter 4C: GRPO reinforcement learning
```

## Licence

MIT
