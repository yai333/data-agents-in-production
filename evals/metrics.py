"""Metric calculations for Text-to-SQL evaluation.

Provides aggregate metrics from evaluation results:
- Execution accuracy: % of queries that run without error
- Result accuracy: % of queries that return correct results (Level 2)
- Latency percentiles: Response time distribution
- Slice analysis: Accuracy by category and difficulty

These metrics enable systematic improvement by identifying
specific weaknesses (e.g., "JOIN queries are 20% worse than COUNT queries").
"""

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.runner import EvalResult
    from evals.golden_set import GoldenQuery


def calculate_metrics(
    results: list["EvalResult"],
    golden_set: list["GoldenQuery"],
) -> dict:
    """Calculate aggregate metrics from evaluation results.

    Args:
        results: List of EvalResult from running evaluation
        golden_set: The golden set used for evaluation (for category info)

    Returns:
        Dictionary containing:
        - total_queries: Number of queries evaluated
        - execution_accuracy: Fraction that executed successfully
        - result_accuracy: Fraction with correct results
        - latency_p50_ms: Median latency
        - latency_p95_ms: 95th percentile latency
        - latency_mean_ms: Average latency
        - by_category: Accuracy breakdown by query category
        - by_difficulty: Accuracy breakdown by difficulty
        - results: Raw results for further analysis
    """
    total = len(results)

    if total == 0:
        return {
            "total_queries": 0,
            "execution_accuracy": 0.0,
            "result_accuracy": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_mean_ms": 0.0,
            "by_category": {},
            "by_difficulty": {},
            "results": [],
        }

    # Basic metrics
    executed = sum(1 for r in results if r.executed)
    matched = sum(1 for r in results if r.result_matches)

    # Latency calculations
    latencies = sorted([r.latency_ms for r in results])
    p50_idx = len(latencies) // 2
    p95_idx = int(len(latencies) * 0.95)

    # Build query map for slice analysis
    query_map = {q.id: q for q in golden_set}

    # Slice by category
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})
    by_difficulty: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "executed": 0, "matched": 0})

    for result in results:
        query = query_map.get(result.query_id)
        if query is None:
            continue

        # Category stats
        by_category[query.category]["total"] += 1
        if result.executed:
            by_category[query.category]["executed"] += 1
        if result.result_matches:
            by_category[query.category]["matched"] += 1

        # Difficulty stats
        by_difficulty[query.difficulty]["total"] += 1
        if result.executed:
            by_difficulty[query.difficulty]["executed"] += 1
        if result.result_matches:
            by_difficulty[query.difficulty]["matched"] += 1

    # Convert to accuracy percentages
    category_accuracy = {
        cat: stats["matched"] / stats["total"] if stats["total"] > 0 else 0.0
        for cat, stats in by_category.items()
    }

    difficulty_accuracy = {
        diff: stats["matched"] / stats["total"] if stats["total"] > 0 else 0.0
        for diff, stats in by_difficulty.items()
    }

    return {
        "total_queries": total,
        "execution_accuracy": executed / total,
        "result_accuracy": matched / total,
        "latency_p50_ms": latencies[p50_idx] if latencies else 0.0,
        "latency_p95_ms": latencies[p95_idx] if latencies else 0.0,
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "by_category": category_accuracy,
        "by_difficulty": difficulty_accuracy,
        "by_category_detailed": dict(by_category),
        "by_difficulty_detailed": dict(by_difficulty),
        "results": results,
    }


def format_metrics_report(metrics: dict) -> str:
    """Format metrics as a human-readable report.

    Args:
        metrics: Dictionary from calculate_metrics()

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        "",
        "Overall Metrics",
        "-" * 30,
        f"  Total Queries:      {metrics['total_queries']}",
        f"  Execution Accuracy: {metrics['execution_accuracy']:.1%}",
        f"  Result Accuracy:  {metrics['result_accuracy']:.1%}",
        "",
        "Latency",
        "-" * 30,
        f"  P50: {metrics['latency_p50_ms']:.0f}ms",
        f"  P95: {metrics['latency_p95_ms']:.0f}ms",
        f"  Mean: {metrics['latency_mean_ms']:.0f}ms",
        "",
        "By Category",
        "-" * 30,
    ]

    for cat, acc in sorted(metrics["by_category"].items()):
        detailed = metrics.get("by_category_detailed", {}).get(cat, {})
        matched = detailed.get("matched", 0)
        total = detailed.get("total", 0)
        lines.append(f"  {cat:12} {acc:6.1%}  ({matched}/{total})")

    lines.extend([
        "",
        "By Difficulty",
        "-" * 30,
    ])

    for diff in ["easy", "medium", "hard"]:
        if diff in metrics["by_difficulty"]:
            acc = metrics["by_difficulty"][diff]
            detailed = metrics.get("by_difficulty_detailed", {}).get(diff, {})
            matched = detailed.get("matched", 0)
            total = detailed.get("total", 0)
            lines.append(f"  {diff:12} {acc:6.1%}  ({matched}/{total})")

    lines.append("=" * 60)

    return "\n".join(lines)


def compare_metrics(baseline: dict, current: dict) -> str:
    """Compare two metric sets and show improvements/regressions.

    Args:
        baseline: Metrics from baseline evaluation
        current: Metrics from current evaluation

    Returns:
        Formatted comparison report
    """
    def delta_str(curr: float, base: float) -> str:
        diff = curr - base
        if abs(diff) < 0.001:
            return "="
        return f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"

    lines = [
        "=" * 60,
        "COMPARISON: Baseline vs Current",
        "=" * 60,
        "",
        "Overall",
        "-" * 30,
        f"  Execution: {baseline['execution_accuracy']:.1%} → {current['execution_accuracy']:.1%} ({delta_str(current['execution_accuracy'], baseline['execution_accuracy'])})",
        f"  Result:    {baseline['result_accuracy']:.1%} → {current['result_accuracy']:.1%} ({delta_str(current['result_accuracy'], baseline['result_accuracy'])})",
        "",
        "By Category",
        "-" * 30,
    ]

    all_categories = set(baseline.get("by_category", {}).keys()) | set(current.get("by_category", {}).keys())
    for cat in sorted(all_categories):
        base_acc = baseline.get("by_category", {}).get(cat, 0.0)
        curr_acc = current.get("by_category", {}).get(cat, 0.0)
        lines.append(f"  {cat:12} {base_acc:.1%} → {curr_acc:.1%} ({delta_str(curr_acc, base_acc)})")

    lines.append("=" * 60)

    return "\n".join(lines)
