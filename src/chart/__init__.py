"""Chart generation for SQL query results.

This module provides Vega-Lite chart generation for query results.
The LLM selects appropriate chart types and generates valid specs.

See 3.6 for the chart generation pipeline.
"""

from src.chart.models import (
    ChartType, ChartSpec, ChartContext,
    ChartGenerationItem, MultiChartGenerationResult,
)
from src.chart.generator import (
    generate_chart_spec,
    preprocess_for_chart,
    finalize_chart_spec,
    generate_chart_from_answer,
)

__all__ = [
    # Models
    "ChartType",
    "ChartSpec",
    "ChartContext",
    "ChartGenerationItem",
    "MultiChartGenerationResult",
    # Generation (DataFrame-based)
    "generate_chart_spec",
    "preprocess_for_chart",
    "finalize_chart_spec",
    # Generation (answer-text-based, 3.6 pattern)
    "generate_chart_from_answer",
]
