# Chapter Script Tester

Test chapter scripts in a sandbox environment following the book's setup instructions.

## Purpose

This skill tests chapter scripts (`scripts/run_chapter_*.py`) to ensure:
1. Scripts run successfully in the correct environment
2. Scripts produce expected output
3. Scripts are consistent with chapter content

## Environment Setup (from Chapter 1)

Before testing any script, ensure the environment is set up:

### 1. Python Environment (requires Python 3.11+)

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### 2. Database Setup

```bash
# Start PostgreSQL with Chinook database
make db-up

# Wait for healthy status
docker ps | grep sql-agent-postgres

# Generate schema file (required for most scripts)
python scripts/generate_chinook_schema.py
```

### 3. Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Add your API key (at least one required)
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...
```

## Testing a Chapter Script

### Step 1: Verify environment

```bash
# Check Python version
python --version  # Should be 3.11+

# Check database is running
docker ps | grep postgres

# Check API key is set
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Gemini:', bool(os.getenv('GOOGLE_API_KEY')))"
```

### Step 2: Run the script

```bash
# Run via make (preferred)
make chapter-2_4

# Or directly
python scripts/run_chapter_2_4.py
```

### Step 3: Verify output

Check that:
- Script completes without errors
- Output matches expected results in script docstring
- Results file is created in `evals/` directory

## Script-Chapter Consistency Checks

When testing a script, verify it matches the chapter content:

### Imports Check
- All imported modules should exist in `src/`
- Imported models should match those defined in the chapter
- Function signatures should match chapter code snippets

### Model Check
For Chapter 2.4:
- Uses `SQLResult` for comparison testing (legacy, acceptable for evals)
- `SQLAgentResponse` is the unified model for new code
- No multi-stage pipeline (single agent approach)

### Output Check
- Script docstring describes expected results
- Actual output should be within reasonable variance
- Results JSON should have correct structure

## Chapter Script Inventory

| Script | Chapter | Key Tests |
|--------|---------|-----------|
| `run_chapter_1_5.py` | 1.5 Setup | Environment, DB connection, basic generation |
| `run_chapter_2_2.py` | 2.2 Schema | Schema loading, table search |
| `run_chapter_2_3.py` | 2.3 Retrieval | Hybrid retrieval, MRR metrics |
| `run_chapter_2_4.py` | 2.4 Structured | Free-form vs structured comparison |

## Troubleshooting

### Python version mismatch
```
TypeError: 'type' object is not subscriptable
```
**Fix**: Use Python 3.11+ (required for modern type hints)

### Database not found
```
connection refused
```
**Fix**: Run `make db-up` and wait for healthy status

### API key missing
```
AuthenticationError
```
**Fix**: Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` in `.env`

### Module not found
```
ModuleNotFoundError: No module named 'src'
```
**Fix**: Run `pip install -e .` from project root

## Usage

To test a chapter script:

1. Specify the chapter number (e.g., "2_4")
2. The skill will:
   - Verify environment setup
   - Run the script
   - Check output against expected results
   - Verify consistency with chapter content

Example:
```
Test chapter 2.4 script
```

The skill will guide you through setup if environment is not ready.
