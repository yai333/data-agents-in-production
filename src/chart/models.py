"""Chart models for Vega-Lite generation.

This module defines the Pydantic models for chart generation:
- ChartType: Supported visualization types
- ChartSpec: LLM output with reasoning and Vega-Lite spec
- ChartContext: Preprocessed data context for LLM

See 3.6 for chart type selection logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ChartType(str, Enum):
    """Supported chart types.

    The LLM selects from these based on data patterns:
    - LINE: Time series with trends
    - MULTI_LINE: Multiple series over time
    - BAR: Category comparison
    - GROUPED_BAR: Category × subcategory
    - STACKED_BAR: Composition of categories
    - PIE: Part-of-whole (use sparingly)
    - AREA: Time series with magnitude emphasis
    - NONE: No chart appropriate (single value, wide table)
    """

    LINE = "line"
    MULTI_LINE = "multi_line"
    BAR = "bar"
    GROUPED_BAR = "grouped_bar"
    STACKED_BAR = "stacked_bar"
    PIE = "pie"
    AREA = "area"
    NONE = ""  # No chart appropriate


class ChartSpec(BaseModel):
    """LLM output for chart generation.

    Attributes:
        reasoning: Step-by-step explanation of chart type selection
        chart_type: Selected chart type or NONE if unsuitable
        vega_lite_spec_json: Vega-Lite spec as JSON string (parsed after generation)
    """

    reasoning: str = Field(
        description="Step-by-step reasoning for chart type selection"
    )
    chart_type: ChartType = Field(
        description="Selected chart type or empty string if no chart appropriate"
    )
    vega_lite_spec_json: str | None = Field(
        default=None,
        description=(
            "Complete Vega-Lite v5 JSON specification as a string. "
            "Do NOT include the 'data' field. "
            "Example: {\"mark\": \"bar\", \"encoding\": {\"x\": {\"field\": \"name\", \"type\": \"nominal\"}, \"y\": {\"field\": \"count\", \"type\": \"quantitative\"}}}"
        ),
    )

    @property
    def vega_lite_spec(self) -> dict[str, Any] | None:
        """Parse the JSON string into a dict."""
        if not self.vega_lite_spec_json:
            return None
        import json
        try:
            return json.loads(self.vega_lite_spec_json)
        except json.JSONDecodeError:
            # Try to extract JSON from the string (LLM may add extra text)
            decoder = json.JSONDecoder()
            try:
                obj, _ = decoder.raw_decode(self.vega_lite_spec_json.strip())
                return obj
            except (json.JSONDecodeError, ValueError):
                return None


class ChartGenerationItem(BaseModel):
    """Single chart from a data section in the answer text.

    The LLM identifies distinct data sections in the final answer
    (e.g., "By Genre:", "By Country:") and generates a separate
    Vega-Lite chart for each.
    """

    title: str = Field(description="Descriptive title for this chart")
    reasoning: str = Field(
        description="Why this chart type was selected for this data section"
    )
    chart_type: str = Field(
        description="Chart type: bar, line, pie, area, grouped_bar, stacked_bar"
    )
    chart_schema_json: str = Field(
        description="Complete Vega-Lite v5 specification as a JSON string"
    )


class MultiChartGenerationResult(BaseModel):
    """Output from the answer-text chart generator.

    Supports multiple charts when the answer contains several
    distinct data sections (e.g., breakdown by genre AND by country).
    """

    overall_reasoning: str = Field(
        description="Summary of data sections found and charts generated"
    )
    charts: list[ChartGenerationItem] = Field(
        default_factory=list,
        description="Array of chart specifications, one per data section",
    )


@dataclass
class ChartContext:
    """Context for chart type selection.

    Preprocessed from query results to minimize tokens sent to LLM.

    Attributes:
        query: Original user question
        sql: Executed SQL query
        sample_rows: First N rows of results (typically 10-20)
        column_info: Statistics about each column
        row_count: Total number of rows
    """

    query: str
    sql: str
    sample_rows: list[dict[str, Any]]
    column_info: dict[str, dict[str, Any]]
    row_count: int
