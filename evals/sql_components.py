"""Spider-style SQL component extraction and comparison.

This module aligns component scoring semantics with Spider's evaluator:
clause-level set matching for SELECT/WHERE/GROUP/HAVING/ORDER plus
AND-OR, IUEN (INTERSECT/UNION/EXCEPT), and keyword overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

AGG_FUNCS = {"COUNT", "SUM", "AVG", "MAX", "MIN"}
WHERE_OPS = (
    "NOT LIKE", "IS NOT", "NOT IN", ">=", "<=", "!=", "<>", "=",
    ">", "<", "LIKE", "IN", "BETWEEN", "IS", "EXISTS",
)


@dataclass
class SQLComponents:
    """Structured representation of SQL query components."""

    select_items: Set[Tuple[str, str]] = field(default_factory=set)
    select_columns: Set[str] = field(default_factory=set)
    select_aggs: Set[str] = field(default_factory=set)
    from_tables: Set[str] = field(default_factory=set)
    where_items: Set[Tuple[str, str]] = field(default_factory=set)
    where_columns: Set[str] = field(default_factory=set)
    group_by_columns: Set[str] = field(default_factory=set)
    having_items: Set[Tuple[str, str]] = field(default_factory=set)
    order_by_columns: Set[str] = field(default_factory=set)
    order_direction: str = ""
    limit_value: Optional[int] = None
    and_or_ops: Set[str] = field(default_factory=set)
    iuen_ops: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    has_subquery: bool = False


def normalize_identifier(identifier: str) -> str:
    ident = identifier.lower().strip()
    ident = re.sub(r"`|\"|'|\[|\]", "", ident)
    ident = re.sub(r"\s+as\s+\w+$", "", ident, flags=re.IGNORECASE).strip()
    if "." in ident:
        ident = ident.split(".")[-1]
    return ident.strip("() ")


def _clean_sql(sql: str) -> str:
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    return " ".join(sql.strip().split())


def _split_csv(expr: str) -> list[str]:
    items: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            items.append(expr[start:i].strip())
            start = i + 1
    tail = expr[start:].strip()
    if tail:
        items.append(tail)
    return items


def _extract_condition_items(clause: str) -> tuple[Set[Tuple[str, str]], Set[str], Set[str]]:
    items: Set[Tuple[str, str]] = set()
    cols: Set[str] = set()
    and_or: Set[str] = set()
    for op in ("AND", "OR"):
        if re.search(rf"\b{op}\b", clause, flags=re.IGNORECASE):
            and_or.add(op.lower())
    parts = re.split(r"\bAND\b|\bOR\b", clause, flags=re.IGNORECASE)
    for part in parts:
        text = part.strip()
        if not text:
            continue
        for op in WHERE_OPS:
            escaped = re.escape(op)
            if re.match(r"^[A-Za-z ]+$", op):
                pat = re.compile(rf"(.+?)\s+{escaped}\b", flags=re.IGNORECASE)
            else:
                pat = re.compile(rf"(.+?)\s+{escaped}(?=\s|$)", flags=re.IGNORECASE)
            m = pat.search(text)
            if not m:
                continue
            col = normalize_identifier(m.group(1))
            norm_op = op.replace(" ", "_").lower()
            if col:
                cols.add(col)
                items.add((col, norm_op))
            break
    return items, cols, and_or


def _extract_keywords(sql: str) -> Set[str]:
    keywords: Set[str] = set()
    spider_keywords = (
        "where", "group", "order", "limit", "join", "or", "not", "in",
        "like", "union", "intersect", "except", "having", "exists",
    )
    for kw in spider_keywords:
        if re.search(rf"\b{kw}\b", sql, flags=re.IGNORECASE):
            keywords.add(kw)
    return keywords


def parse_sql_components(sql: str) -> SQLComponents:
    components = SQLComponents()
    sql = _clean_sql(sql)
    sql_upper = sql.upper()
    components.has_subquery = sql_upper.count("SELECT") > 1
    components.keywords = _extract_keywords(sql)

    if "UNION" in sql_upper:
        components.iuen_ops.add("union")
    if "INTERSECT" in sql_upper:
        components.iuen_ops.add("intersect")
    if "EXCEPT" in sql_upper:
        components.iuen_ops.add("except")

    limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, flags=re.IGNORECASE)
    if limit_match:
        components.limit_value = int(limit_match.group(1))

    select_match = re.search(
        r"\bSELECT\s+(DISTINCT\s+)?(.*?)\s+\bFROM\b",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if select_match:
        select_clause = select_match.group(2)
        for item in _split_csv(select_clause):
            item = re.sub(r"\s+AS\s+\w+$", "", item, flags=re.IGNORECASE).strip()
            agg_match = re.match(
                r"^(COUNT|SUM|AVG|MAX|MIN)\s*\((.*?)\)$",
                item,
                flags=re.IGNORECASE,
            )
            if agg_match:
                agg = agg_match.group(1).upper()
                col = normalize_identifier(agg_match.group(2))
                components.select_aggs.add(agg)
                components.select_columns.add(col)
                components.select_items.add((agg.lower(), col))
            else:
                col = normalize_identifier(item)
                if col:
                    components.select_columns.add(col)
                    components.select_items.add(("none", col))

    from_match = re.search(
        r"\bFROM\s+(.*?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if from_match:
        from_clause = from_match.group(1)
        for m in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.]*)", f"FROM {from_clause}"):
            components.from_tables.add(normalize_identifier(m.group(1)))
        _, _, and_or = _extract_condition_items(from_clause)
        components.and_or_ops.update(and_or)

    where_match = re.search(
        r"\bWHERE\s+(.*?)(?:\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        items, cols, and_or = _extract_condition_items(where_match.group(1))
        components.where_items = items
        components.where_columns = cols
        components.and_or_ops.update(and_or)

    group_match = re.search(
        r"\bGROUP\s+BY\s+(.*?)(?:\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if group_match:
        for col in _split_csv(group_match.group(1)):
            norm = normalize_identifier(col)
            if norm:
                components.group_by_columns.add(norm)

    having_match = re.search(
        r"\bHAVING\s+(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if having_match:
        items, _, and_or = _extract_condition_items(having_match.group(1))
        components.having_items = items
        components.and_or_ops.update(and_or)

    order_match = re.search(
        r"\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if order_match:
        order_clause = order_match.group(1)
        if re.search(r"\bDESC\b", order_clause, flags=re.IGNORECASE):
            components.order_direction = "desc"
        elif re.search(r"\bASC\b", order_clause, flags=re.IGNORECASE):
            components.order_direction = "asc"
        cleaned = re.sub(r"\bASC\b|\bDESC\b", "", order_clause, flags=re.IGNORECASE)
        for col in _split_csv(cleaned):
            norm = normalize_identifier(col)
            if norm:
                components.order_by_columns.add(norm)

    return components


def compute_component_f1(pred_set: Set, gold_set: Set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class ComponentMatchResult:
    """Spider-style per-component match summary."""

    select_f1: float = 0.0
    select_agg_f1: float = 0.0
    select_wo_agg_f1: float = 0.0
    from_f1: float = 0.0
    where_f1: float = 0.0
    where_wo_op_f1: float = 0.0
    group_f1: float = 0.0
    having_f1: float = 0.0
    order_f1: float = 0.0
    and_or_f1: float = 0.0
    iuen_f1: float = 0.0
    keyword_f1: float = 0.0
    exact_match: bool = False
    overall_f1: float = 0.0


def _order_match(pred: SQLComponents, gold: SQLComponents) -> float:
    pred_has = bool(pred.order_by_columns)
    gold_has = bool(gold.order_by_columns)
    if not pred_has and not gold_has:
        return 1.0
    if pred_has != gold_has:
        return 0.0
    pred_limit = pred.limit_value is not None
    gold_limit = gold.limit_value is not None
    if pred_limit != gold_limit:
        return 0.0
    if pred.order_direction != gold.order_direction:
        return 0.0
    return 1.0 if pred.order_by_columns == gold.order_by_columns else 0.0


def compare_sql_components(pred_sql: str, gold_sql: str) -> ComponentMatchResult:
    pred = parse_sql_components(pred_sql)
    gold = parse_sql_components(gold_sql)
    result = ComponentMatchResult()

    result.select_f1 = compute_component_f1(pred.select_items, gold.select_items)
    result.select_agg_f1 = compute_component_f1(pred.select_aggs, gold.select_aggs)
    result.select_wo_agg_f1 = compute_component_f1(
        pred.select_columns,
        gold.select_columns,
    )
    result.from_f1 = compute_component_f1(pred.from_tables, gold.from_tables)
    result.where_f1 = compute_component_f1(pred.where_items, gold.where_items)
    result.where_wo_op_f1 = compute_component_f1(pred.where_columns, gold.where_columns)
    result.group_f1 = compute_component_f1(pred.group_by_columns, gold.group_by_columns)
    result.having_f1 = compute_component_f1(pred.having_items, gold.having_items)
    result.order_f1 = _order_match(pred, gold)
    result.and_or_f1 = compute_component_f1(pred.and_or_ops, gold.and_or_ops)
    result.iuen_f1 = compute_component_f1(pred.iuen_ops, gold.iuen_ops)
    result.keyword_f1 = compute_component_f1(pred.keywords, gold.keywords)

    scores = [
        result.select_f1,
        result.select_wo_agg_f1,
        result.from_f1,
        result.where_f1,
        result.where_wo_op_f1,
        result.group_f1,
        result.having_f1,
        result.order_f1,
        result.and_or_f1,
        result.iuen_f1,
        result.keyword_f1,
    ]
    result.overall_f1 = sum(scores) / len(scores)
    result.exact_match = all(s == 1.0 for s in scores)
    return result


def compute_similarity_from_components(sql1: str, sql2: str) -> float:
    return compare_sql_components(sql1, sql2).overall_f1
