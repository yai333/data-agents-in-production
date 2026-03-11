"""Chart generation pipeline.

This module implements the three-step chart generation:
1. Preprocess SQL results (extract statistics)
2. Generate chart spec with LLM (select type, create Vega-Lite)
3. Validate and finalize (add data, validate schema)

See 3.6 for the full pipeline documentation.
"""

import json
import logging
from typing import Any, TYPE_CHECKING

import pandas as pd

from langchain_core.messages import SystemMessage, HumanMessage

from src.chart.models import (
    ChartType, ChartSpec, ChartContext,
    MultiChartGenerationResult,
)

if TYPE_CHECKING:
    from src.adapters.base import LLMAdapter

logger = logging.getLogger(__name__)


CHART_SYSTEM_PROMPT = """You are a data visualization expert. Given a user's question,
the SQL query that answered it, and sample data, generate an appropriate Vega-Lite chart.

Rules:
1. Choose the chart type that best represents the data pattern
2. Use 'line' for time series, 'bar' for comparisons, 'pie' for proportions
3. Return empty chart_type if data doesn't suit visualization (single values, wide tables)
4. Generate valid Vega-Lite v5 specs - DO NOT include 'data' field (we'll add it)
5. Use clear axis labels derived from column names
6. Reason through your choice step by step

Chart selection guide:
| Data Pattern | Chart Type |
|--------------|------------|
| Time series (date + metric) | line |
| Category comparison | bar |
| Part-of-whole (proportions) | pie |
| Multiple series over time | multi_line |
| Category × subcategory | grouped_bar |
| Single number | NONE (just display value) |
| Wide table (many columns) | NONE (table is better) |
"""

CHART_USER_PROMPT = """
Question: {query}

SQL: {sql}

Sample data ({row_count} total rows):
{sample_rows}

Column information:
{column_info}

Generate the appropriate chart specification.
"""


def preprocess_for_chart(
    df: pd.DataFrame,
    query: str,
    sql: str,
    sample_size: int = 20,
) -> ChartContext:
    """Extract chart-relevant context from query results.

    Prepares minimal context for the LLM to select chart type
    without sending all data.

    Args:
        df: Query results as DataFrame
        query: Original user question
        sql: Executed SQL query
        sample_size: Number of sample rows to include

    Returns:
        ChartContext with statistics and samples
    """
    column_info = {}

    for col in df.columns:
        col_data = df[col]
        info: dict[str, Any] = {
            "dtype": str(col_data.dtype),
            "unique_count": int(col_data.nunique()),
            "null_count": int(col_data.isnull().sum()),
        }

        # For categorical columns, include sample values
        if col_data.dtype == "object" or col_data.nunique() < 20:
            info["sample_values"] = col_data.dropna().unique()[:10].tolist()

        # For numeric columns, include range
        if pd.api.types.is_numeric_dtype(col_data):
            info["min"] = float(col_data.min()) if not col_data.empty else None
            info["max"] = float(col_data.max()) if not col_data.empty else None

        column_info[col] = info

    return ChartContext(
        query=query,
        sql=sql,
        sample_rows=df.head(sample_size).to_dict(orient="records"),
        column_info=column_info,
        row_count=len(df),
    )


async def generate_chart_spec(
    context: ChartContext,
    adapter: "LLMAdapter",
) -> ChartSpec:
    """Generate Vega-Lite spec for query results.

    Uses LLM to select chart type and generate specification.

    Args:
        context: Preprocessed chart context
        adapter: LLM adapter for generation

    Returns:
        ChartSpec with reasoning and Vega-Lite spec
    """
    prompt = CHART_USER_PROMPT.format(
        query=context.query,
        sql=context.sql,
        row_count=context.row_count,
        sample_rows=_format_sample_rows(context.sample_rows),
        column_info=_format_column_info(context.column_info),
    )

    result = await adapter.generate_structured(
        prompt=prompt,
        system_prompt=CHART_SYSTEM_PROMPT,
        response_model=ChartSpec,
    )

    return result.data


def finalize_chart_spec(
    spec: ChartSpec,
    data: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Add data to spec and validate.

    Merges the LLM-generated spec with actual data and validates
    against Vega-Lite schema.

    Args:
        spec: ChartSpec from LLM
        data: Full query results as list of dicts

    Returns:
        Complete Vega-Lite spec, or None if no chart appropriate
    """
    if spec.chart_type == ChartType.NONE or spec.vega_lite_spec is None:
        return None

    # Merge data into spec with responsive width
    full_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        **spec.vega_lite_spec,
        "data": {"values": data},
    }

    # Validation (simplified - full implementation would use jsonschema)
    if "mark" not in full_spec and "layer" not in full_spec:
        logger.warning("Invalid Vega-Lite spec: missing mark or layer")
        return None

    return full_spec


def _format_sample_rows(rows: list[dict[str, Any]], max_rows: int = 10) -> str:
    """Format sample rows as markdown table for prompt."""
    if not rows:
        return "(empty)"

    # Create markdown table
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows[:max_rows]:
        values = [str(row.get(h, ""))[:30] for h in headers]  # Truncate long values
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _format_column_info(column_info: dict[str, dict[str, Any]]) -> str:
    """Format column statistics for prompt."""
    lines = []
    for col, info in column_info.items():
        parts = [f"{col}: {info['dtype']}"]
        if "sample_values" in info:
            samples = info["sample_values"][:5]
            parts.append(f"values: {samples}")
        if "min" in info:
            parts.append(f"range: [{info['min']}, {info['max']}]")
        lines.append(", ".join(parts))
    return "\n".join(lines)


# ─── Answer-text chart generation (3.6 pattern) ────────────────────────
#
# Instead of re-executing SQL and parsing DataFrames, this approach
# generates charts directly from the final answer text. The LLM
# extracts data sections, labels, and values from the human-readable
# answer and produces Vega-Lite specs with embedded data.

CHART_FROM_ANSWER_SYSTEM_PROMPT = """\
You are a data visualization expert skilled at creating Vega-Lite chart specifications.

### TASK ###
Given a user's question and the final answer text, identify ALL distinct data sections \
and generate a separate Vega-Lite chart for EACH chartable section.

### SUPPORTED CHART TYPES ###
- bar: Comparing values across categories (vertical bars)
- line: Time series, trends over time
- pie: Proportions/percentages (use only when <=6 categories)
- area: Cumulative values over time
- grouped_bar: Comparing multiple metrics across categories
- stacked_bar: Showing composition/breakdown across categories

### MULTI-CHART GUIDELINES ###

1. **Identify Data Sections**:
   - Look for section headers like "By Brand:", "By Customer Status:", "By Age Band:", etc.
   - Each distinct breakdown/grouping should be a separate chart
   - Example: If answer has "By Brand: ..." AND "By Status: ...", generate 2 charts

2. **Data Extraction from Each Section**:
   - Extract category names and numeric values from patterns like "Category: 123 items"
   - Use the EXACT labels from the answer text for axis titles
   - Parse numbered lists, bullet points, or inline data as data rows

3. **Chart Selection per Section**:
   - Time-based data (dates, months, years) -> line or area
   - Category comparisons -> bar
   - Proportions with few categories (<=6) -> pie
   - Multiple metrics per category -> grouped_bar or stacked_bar
   - Large number of categories (>10) -> bar with horizontal orientation

4. **Return empty charts list when**:
   - Only a single value (no comparison possible)
   - Data is empty or text-only without numbers
   - Answer contains error messages or apologies
   - No clear category-value pairs found

5. **Axis Labels (CRITICAL)**:
   - Use the EXACT section headers from the answer as axis titles
   - NEVER use technical column names like "yoi_band" or "customer_type_status"

6. **Chart Schema Structure** (for each chart):
   - $schema: Vega-Lite v5 URL
   - title: Descriptive title for THIS data section
   - data: Extract and include as {"values": [...]}
   - mark: Chart type (bar, line, point, arc, area)
   - encoding: x, y, color with human-readable axis titles
"""

CHART_FROM_ANSWER_USER_PROMPT = """\
### INPUT ###

**User Question**: {question}

**Final Answer** (extract ALL data sections from this):
```
{final_answer}
```

### INSTRUCTIONS ###
1. Read the final answer carefully
2. Identify ALL distinct data sections (look for headers like "By X:", breakdowns, groupings)
3. For EACH data section:
   a. Extract the section header as the axis label
   b. Parse all category-value pairs into chart data
   c. Generate a separate Vega-Lite chart specification
4. Return a chart for each chartable section (may be 1, 2, or more charts)

If the answer doesn't contain chartable data (no numbers, only text, or error messages), \
return empty charts list.
"""


async def generate_chart_from_answer(
    question: str,
    final_answer: str,
    llm_model,
) -> dict[str, Any] | None:
    """Generate Vega-Lite charts from the final answer text.

    The LLM reads the synthesized answer, identifies data sections,
    and produces a chart for each one. Data is extracted directly
    from the answer text, so no SQL re-execution is needed.

    Args:
        question: Original user question
        final_answer: Synthesized answer containing labeled data
        llm_model: LangChain chat model (uses with_structured_output)

    Returns:
        Dict with overall_reasoning and charts list, or None
    """
    if not final_answer or len(final_answer) < 50:
        logger.info("Chart generation skipped: answer too short")
        return None

    # Quick check: answer should contain digits and structured data (colons)
    has_data = any(c.isdigit() for c in final_answer) and ':' in final_answer
    if not has_data:
        logger.info("Chart generation skipped: no numeric data in answer")
        return None

    user_prompt = CHART_FROM_ANSWER_USER_PROMPT.format(
        question=question,
        final_answer=final_answer,
    )

    try:
        structured_llm = llm_model.with_structured_output(MultiChartGenerationResult)

        result = await structured_llm.ainvoke([
            SystemMessage(content=CHART_FROM_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        if not result or not result.charts:
            logger.info("LLM found no chartable data sections")
            return None

        # Parse each chart's JSON spec
        processed = []
        for i, chart in enumerate(result.charts):
            try:
                spec = json.loads(chart.chart_schema_json)
            except json.JSONDecodeError:
                # Try raw_decode as fallback
                try:
                    decoder = json.JSONDecoder()
                    spec, _ = decoder.raw_decode(chart.chart_schema_json.strip())
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Chart {i}: invalid JSON spec, skipping")
                    continue

            if "$schema" not in spec:
                spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
            spec["width"] = "container"

            processed.append({
                "title": chart.title,
                "reasoning": chart.reasoning,
                "chart_type": chart.chart_type,
                "chart_schema": spec,
            })

        if not processed:
            return None

        logger.info(f"Chart generation: {len(processed)} chart(s) from answer text")
        return {
            "overall_reasoning": result.overall_reasoning,
            "charts": processed,
        }

    except Exception as e:
        logger.warning(f"Chart generation failed (non-blocking): {e}")
        return None
