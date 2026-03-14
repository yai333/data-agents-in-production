# Data Agents in Production

Companion code for the book *Building Production Data Agents from Scratch: Design, Evaluate, and Scale Production AI Workflows for Data Analytics*.

The codebase uses **PostgreSQL** for the main agent chapters, ships a bundled **SQLite** Chinook file for the A2UI demo in chapter 3.7, and supports **OpenAI** or **Google Gemini** as the LLM provider.

## Quick start

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL-backed chapters)
- An API key for OpenAI or Google Gemini

This repo includes a `.python-version` file and is tested against Python 3.11. Use `python3.11` or a 3.11 virtual environment explicitly if your system `python` points to an older interpreter.

### Install

```bash
git clone <repo-url> mybook-code && cd mybook-code
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env and add OPENAI_API_KEY or GOOGLE_API_KEY
```

### Database setup

For PostgreSQL-backed chapters (1.5 through 3.6, plus most Part 4 training scripts):

```bash
make db-up
make db-setup
```

`make db-setup` downloads the Chinook PostgreSQL bootstrap SQL into `data/chinook/01_chinook.sql` if needed. The repo also includes a checked-in copy of that file.

For chapter 3.7, the repo already includes `data/chinook.db` for the SQLite-based A2UI demo.

### Verify your setup

```bash
make chapter-1_5
```

### Run the full agent

```bash
make chapter-3_8
# or
python scripts/run_chapter_3_8.py "How many artists are there?"
```

## Project structure

```text
src/
├── adapters/       # LLM provider abstraction (OpenAI, Gemini)
├── agent/          # LangGraph agent graph, retry, repair, CLI
├── ambiguity/      # Ambiguity detection and clarification
├── api/            # FastAPI endpoint for the agent
├── authorization/  # Row-level security and access control
├── chart/          # Vega-Lite chart generation from query results
├── context/        # Context engineering and session memory helpers
├── execution/      # SQL execution with guardrails
├── observability/  # Structured logging and tracing
├── pii/            # PII detection and anonymisation
├── reasoning/      # Prompt templates and reasoning patterns
├── retrieval/      # BM25, semantic retrieval, hybrid RRF
├── schema/         # Schema cards, MDL store, schema rendering
├── security/       # Threat-model and firewall helpers
├── structured/     # Structured SQL generation
├── utils/          # Configuration and logging helpers
├── validators/     # SQL validation and safety checks
└── vllm_plugins/   # Custom vLLM tool parser for Qwen2.5-Coder

scripts/
├── run_chapter_*.py        # Chapter entrypoints and runnable wrappers
├── chapter_3_2/            # PII protection demo
├── chapter_3_7/            # A2UI SQL Explorer demo app
├── chapter_4A/             # APO optimization code
├── chapter_4B/             # SFT notebook assets
├── chapter_4C/             # GRPO reward and training code
└── data/                   # Data preparation utilities

evals/                      # Evaluation framework with golden sets and metrics
data/                       # Chinook SQL, SQLite DB, schema JSON, training data
config/                     # YAML configuration with env-var substitution
```

## Chapter runner map

Runnable code begins in chapter 1.5. Chapters 1.1-1.4 are conceptual setup chapters and do not have standalone scripts. Use `make chapter-<id>` or run the script directly for the executable chapters below.

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
| `make chapter-3_1` | `scripts/run_chapter_3_1.py` | 3.1 Threat Modeling |
| `make chapter-3_2` | `scripts/run_chapter_3_2.py` | 3.2 PII Protection |
| `make chapter-3_3` | `scripts/run_chapter_3_3.py` | 3.3 Context Engineering |
| `make chapter-3_4` | `scripts/run_chapter_3_4.py` | 3.4 Observability |
| `make chapter-3_5` | `scripts/run_chapter_3_5.py` | 3.5 Evaluation System |
| `make chapter-3_6` | `scripts/run_chapter_3_6.py` | 3.6 Chart Generation |
| `make chapter-3_7` | `scripts/run_chapter_3_7.py` | 3.7 Agent-Driven UI |
| `make chapter-3_8` | `scripts/run_chapter_3_8.py` | 3.8 The Complete System |
| `make chapter-4A_1` | `scripts/run_chapter_4A_1.py` | 4A.1 How APO Works |
| `make chapter-4A_2` | `scripts/run_chapter_4A_2.py` | 4A.2 Running APO |
| `make chapter-4B_1` | `scripts/run_chapter_4B_1.py` | 4B.1 SFT Fine-Tuning |
| `make chapter-4B_2` | `scripts/run_chapter_4B_2.py` | 4B.2 GRPO Fine-Tuning |
| `make chapter-4C_1` | `scripts/run_chapter_4C_1.py` | 4C.1 Why RL for Agents |
| `make chapter-4C_2` | `scripts/run_chapter_4C_2.py` | 4C.2 Reward Design |
| `make chapter-4C_3` | `scripts/run_chapter_4C_3.py` | 4C.3 SQL GRPO Training Reference |
| `make chapter-4D` | `scripts/run_chapter_4D.py` | 4D Workflow Optimization |

Chapter 4B.1 is notebook-driven. Its wrapper validates the notebook path and can execute it with `jupyter nbconvert --execute` if Jupyter is installed. Chapter 4B.2 uses the schema-linking GRPO wrapper directly.

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
make db-setup           # Download/load Chinook schema bootstrap SQL
make db-shell           # Open psql shell

# Quality
make test               # Run all tests
make test-unit          # Unit tests only
make test-integration   # Integration tests (API keys + database)
make test-cov           # Tests with coverage report
make lint               # ruff + mypy
make format             # Auto-fix with ruff

# Few-shot indexing
make index-examples     # Index examples into pgvector

# Chapter 4A — APO
make install-ch4a       # Install APO dependencies
make apo-quick          # Quick APO run (2 beams, 1 round)
make apo-full           # Full APO optimization

# Chapter 4B — SFT
make install-ch4b       # Install SFT dependencies

# Chapter 4C — GRPO
make install-ch4c       # Install GRPO dependencies
make vllm-serve         # Serve SFT model with vLLM
make grpo-train         # Full GRPO training run
make grpo-train-small   # Small-scale test run
make grpo-eval          # Compare cloud vs local model
```

## Configuration

- `.env`: API keys, database URL, provider selection
- `config/default.yaml`: agent behavior, safety limits, retrieval settings

The primary Postgres connection variable is `CHINOOK_DATABASE_URL`. Chapter 3.7 uses `CHINOOK_DB_PATH` for the SQLite demo.

## Optional dependency groups

The base install covers chapters 1–3. Later chapters need additional packages:

```bash
pip install -e ".[apo]"       # Chapter 4A
pip install -e ".[training]"  # Chapter 4B
pip install -e ".[grpo]"      # Chapter 4C
```

## License

MIT
