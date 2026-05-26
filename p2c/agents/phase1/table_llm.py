from __future__ import annotations

import json
import re
from typing import Any


FACT_SCOPE_RE = re.compile(r"<fact>(.*?)</fact>.*?<scope>(.*?)</scope>", flags=re.I | re.S)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

TABLE_CRITERIA_SYSTEM_PROMPT = """\
You are an ML reproducibility auditor extracting claims from paper tables.
Read the table, caption, nearby context, and optional visual extraction.
Return only directly reproducible atomic criteria as JSON.

Extract:
- metric results, including error, test error, train error, accuracy, loss, AUC, F1, mean results, best results, and comparisons;
- execution/config parameters, including dataset, epochs, batch size, learning rate, beta, T, K, dropout, weight decay, momentum, optimizer, and seeds.

Use caption/context to infer metric names when table headers are short, nested, or use merged cells.
Skip prose-only observations and method descriptions.
Each item must have a concrete fact and a scope that preserves the table anchor and experimental context.
"""

TABLE_CRITERIA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "scope": {"type": "string"},
                    "facet": {"type": "string", "enum": ["metric_result", "execution_param"]},
                    "metric_name": {"type": ["string", "null"]},
                    "metric_value": {"type": ["number", "string", "null"]},
                    "metric_unit": {"type": ["string", "null"]},
                    "entity": {"type": ["string", "null"]},
                    "comparator": {"type": ["string", "null"]},
                    "dataset_scope": {"type": ["string", "null"]},
                    "table_anchor": {"type": ["string", "null"]},
                    "reason": {"type": ["string", "null"]},
                },
                "required": ["fact", "scope", "facet"],
            },
        }
    },
    "required": ["criteria"],
}


def build_table_criteria_user_prompt(
    *,
    table_anchor: str | None,
    caption: str | None,
    table_html: str | None = None,
    context_before: str | None = None,
    context_after: str | None = None,
    visual_element: dict[str, Any] | None = None,
) -> str:
    visual_json = "{}"
    if visual_element:
        visual_json = json.dumps(visual_element, ensure_ascii=False, sort_keys=True)
        visual_json = visual_json[:12000]

    return (
        "Extract atomic reproducibility criteria from this paper table.\n"
        f"Table anchor: {table_anchor or 'unknown'}\n"
        f"Caption: {caption or ''}\n"
        f"Nearby context before: {(context_before or '')[:2500]}\n"
        f"Nearby context after: {(context_after or '')[:1200]}\n"
        f"Table HTML or markdown:\n{(table_html or '')[:12000]}\n\n"
        f"Visual extraction JSON, if available:\n{visual_json}\n\n"
        "Return JSON with key 'criteria'. Each criterion should include:\n"
        "- fact: e.g. 'EP symmetric test error = 12.45%'\n"
        "- scope: e.g. 'CIFAR-10, Table 1, Squared Error loss, symmetric gradient estimate'\n"
        "- facet: 'metric_result' or 'execution_param'\n"
        "- metric_name and metric_value when applicable.\n"
        "Prefer a small set of high-value claims over every single cell, but include table means, best/final results, and core hyperparameters."
    )


def rows_from_llm_table_response(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    rows = data.get("criteria")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def normalize_table_criteria(
    rows: list[dict[str, Any]],
    *,
    default_anchor: str | None,
    input_unit_id: str | None,
    source_prefix: str,
    default_reason_code: str,
    visual_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        fact, scope = _extract_fact_scope(row)
        if not fact or not scope:
            continue

        facet = str(row.get("facet") or "").strip()
        if facet not in {"metric_result", "execution_param"}:
            facet = "metric_result" if row.get("metric_name") or row.get("metric_value") is not None else "execution_param"

        source_type = f"{source_prefix}_metric" if facet == "metric_result" else f"{source_prefix}_param"
        table_anchor = _clean_str(row.get("table_anchor")) or default_anchor
        metric_value = _float_or_none(row.get("metric_value"))
        if metric_value is None and facet == "metric_result":
            metric_value = _first_number(fact)

        reason_codes = [default_reason_code]
        if row.get("reason"):
            reason_codes.append("LLM_REASON_PROVIDED")

        item = {
            "criterion": f"<fact>{fact}</fact> <scope>{scope}</scope>",
            "fact": fact,
            "scope": scope,
            "facet": facet,
            "source_type": source_type,
            "metric_name": _clean_str(row.get("metric_name")),
            "metric_value": metric_value,
            "metric_unit": _clean_str(row.get("metric_unit")),
            "entity": _clean_str(row.get("entity")),
            "comparator": _clean_str(row.get("comparator")),
            "dataset_scope": _clean_str(row.get("dataset_scope")),
            "table_anchor": table_anchor,
            "input_unit_id": input_unit_id,
            "reason_codes": reason_codes,
        }
        if visual_data:
            item["visual_data"] = dict(visual_data)
        out.append(item)
    return out


def _extract_fact_scope(row: dict[str, Any]) -> tuple[str | None, str | None]:
    criterion = _clean_str(row.get("criterion"))
    if criterion:
        match = FACT_SCOPE_RE.search(criterion)
        if match:
            return _collapse_ws(match.group(1)), _collapse_ws(match.group(2))

    fact = _clean_str(row.get("fact"))
    scope = _clean_str(row.get("scope"))
    if fact and scope:
        return _collapse_ws(fact), _collapse_ws(scope)
    return None, None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = _collapse_ws(str(value))
    return text or None


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = str(value)
    match = NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _first_number(text: str) -> float | None:
    return _float_or_none(text)
