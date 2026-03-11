.PHONY: install dev test lint format clean db-up db-down db-reset index-examples \
       install-ch4a install-ch4b install-ch4c apo-install \
       generate-sql-data vllm-serve grpo-dry-run grpo-train grpo-train-small grpo-eval

# Use the venv Python/pip to avoid installing into conda
PYTHON ?= .venv/bin/python
PIP    ?= $(PYTHON) -m pip

# Installation
install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev]"

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/ -v -m "not integration"

test-integration:
	$(PYTHON) -m pytest tests/ -v -m integration

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/

format:
	$(PYTHON) -m ruff check --fix src/ tests/
	$(PYTHON) -m ruff format src/ tests/

# Database
db-up:
	docker compose up -d postgres

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d postgres

db-setup:
	$(PYTHON) scripts/setup_chinook.py

db-shell:
	docker exec sql-agent-postgres psql -U postgres -d chinook

# Index few-shot examples into pgvector
index-examples:
	$(PYTHON) scripts/index_fewshot_examples.py

index-examples-openai:
	$(PYTHON) scripts/index_fewshot_examples.py --provider openai

index-examples-gemini:
	$(PYTHON) scripts/index_fewshot_examples.py --provider gemini

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf output/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development helpers
run:
	$(PYTHON) -m src.agent.cli

# ── Chapter 4A: APO ──────────────────────────
install-ch4a:
	$(PIP) install -e ".[apo]"

apo-install: install-ch4a

apo-quick:
	$(PYTHON) scripts/chapter_4A/run_apo.py --beam-width 2 --rounds 1

apo-full:
	$(PYTHON) scripts/chapter_4A/run_apo.py

# ── Chapter 4B: SFT Fine-tuning ─────────────
install-ch4b:
	$(PIP) install -e ".[training]"

# ── Chapter 4C: GRPO Training ───────────────
install-ch4c:
	$(PIP) install -e ".[grpo]"

generate-sql-data:
	$(PYTHON) -m scripts.data.generate_sql_data

# Serve the SFT checkpoint with vLLM (LoRA adapter on Qwen2.5-Coder-1.5B)
# Override SFT_BASE_MODEL and LORA_PATH for custom checkpoints
SFT_BASE_MODEL ?= unsloth/Qwen2.5-Coder-1.5B-Instruct
LORA_PATH      ?= /home/yai111/works/juypter/output_sql_cot
VLLM_PORT      ?= 8000

vllm-serve:
	$(PYTHON) -m vllm.entrypoints.openai.api_server \
		--model $(SFT_BASE_MODEL) \
		--enable-lora \
		--lora-modules sql-agent=$(LORA_PATH) \
		--port $(VLLM_PORT) \
		--max-model-len 4096 \
		--enable-auto-tool-choice \
		--tool-call-parser hermes

grpo-dry-run:
	$(PYTHON) -m scripts.chapter_4C.train_grpo_sql --dry-run

grpo-train:
	$(PYTHON) -m scripts.chapter_4C.train_grpo_sql

# Small-scale training test (16 train / 4 val, 1 epoch, with tracing)
grpo-train-small:
	$(PYTHON) -m scripts.chapter_4C.train_grpo_sql \
		--train data/sql_train_small.jsonl \
		--val data/sql_val_small.jsonl \
		--model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--trace

# Evaluate: compare cloud vs local model (requires vllm-serve running)
grpo-eval:
	$(PYTHON) scripts/chapter_4C/evaluate_sql.py \
		--local-model $(LORA_PATH) \
		--base-url http://localhost:$(VLLM_PORT)/v1

grpo-eval-cloud:
	$(PYTHON) scripts/chapter_4C/evaluate_sql.py --cloud-only

# Schema linking GRPO (single-turn, no tool calling)
SL_VLLM_PORT   ?= 8001

sl-vllm-serve:
	$(PYTHON) -m vllm.entrypoints.openai.api_server \
		--model $(SFT_BASE_MODEL) \
		--port $(SL_VLLM_PORT) \
		--max-model-len 5120 \
		--enforce-eager \
		--gpu-memory-utilization 0.4

sl-data-prep:
	$(PYTHON) -m scripts.data.prepare_schema_linking_data

sl-dry-run:
	$(PYTHON) -m scripts.chapter_4C.train_grpo_schema_linking \
		--dry-run \
		--base-url http://localhost:$(SL_VLLM_PORT)/v1 \
		--dry-run-model $(SFT_BASE_MODEL)

sl-train:
	$(PYTHON) -m scripts.chapter_4C.train_grpo_schema_linking --trace

sl-train-small:
	$(PYTHON) -m scripts.chapter_4C.train_grpo_schema_linking \
		--train data/schema_linking_train_tiny.jsonl \
		--val data/schema_linking_val_tiny.jsonl \
		--model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--trace

# Chapter scripts
chapter-%:
	$(PYTHON) scripts/run_chapter_$*.py
