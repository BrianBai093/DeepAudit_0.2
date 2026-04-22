"""EnrichClaimsVisualAgent — enriches atomic criteria with visual data from PDF extraction."""

from __future__ import annotations

import re
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase1.table_llm import (
    TABLE_CRITERIA_SCHEMA,
    TABLE_CRITERIA_SYSTEM_PROMPT,
    build_table_criteria_user_prompt,
    normalize_table_criteria,
    rows_from_llm_table_response,
)
from p2c.schemas import VisualTarget, VisualTargetsDoc


class EnrichClaimsVisualAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="enrich_claims_visual", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        # Load visual elements
        try:
            ve_doc = self.artifacts.read_json("fingerprint/visual_elements.json")
        except Exception:  # noqa: BLE001
            ve_doc = {}

        elements = ve_doc.get("elements", [])
        if not elements:
            self.artifacts.write_json(
                "fingerprint/visual_targets.json",
                VisualTargetsDoc(reason_codes=["NO_VISUAL_ELEMENTS"]).model_dump(),
            )
            self.log("PROGRESS", "No visual elements to enrich with, skipping")
            return {
                "enriched_count": 0,
                "new_from_figures": 0,
                "new_from_tables": 0,
                "visual_targets": 0,
            }

        # Load existing atomic criteria
        try:
            criteria = self.artifacts.read_json("fingerprint/atomic_criteria.json")
        except Exception:  # noqa: BLE001
            self.log("PROGRESS", "No atomic_criteria.json found, creating visual-derived criteria doc")
            criteria = {"criteria": [], "reason_codes": ["VISUAL_ONLY_ATOMIC_CRITERIA"]}

        criteria_key = "criteria" if isinstance(criteria.get("criteria"), list) else "accepted"
        criterion_rows = criteria.get(criteria_key, [])
        if not isinstance(criterion_rows, list):
            criterion_rows = []
            criteria[criteria_key] = criterion_rows

        # Build visual element index by anchor (e.g., "Table 1" → element)
        anchor_index: dict[str, dict] = {}
        for elem in elements:
            anchor = str(elem.get("visual_anchor", "")).strip().lower()
            if anchor:
                anchor_index[anchor] = elem
            # Also index by element_id
            eid = str(elem.get("element_id", "")).strip().lower()
            if eid:
                anchor_index[eid] = elem

        # Enrich existing criteria with visual data
        enriched_count = 0
        for criterion in criterion_rows:
            table_anchor = _extract_table_anchor(criterion)
            if not table_anchor:
                continue

            elem = anchor_index.get(table_anchor.lower())
            if not elem:
                continue

            # Add visual_data to the criterion
            if "visual_data" not in criterion:
                criterion["visual_data"] = {}
            criterion["visual_data"]["chart_type"] = elem.get("chart_type")
            criterion["visual_data"]["axis_labels"] = elem.get("axis_labels", {})
            criterion["visual_data"]["legend_entries"] = elem.get("legend_entries", [])
            criterion["visual_data"]["data_series"] = elem.get("data_series", [])
            criterion["visual_data"]["element_id"] = elem.get("element_id")
            _copy_visual_metadata(elem, criterion["visual_data"])

            if "reason_codes" not in criterion:
                criterion["reason_codes"] = []
            if "VISUAL_ENRICHED" not in criterion["reason_codes"]:
                criterion["reason_codes"].append("VISUAL_ENRICHED")
            enriched_count += 1

        # Keep figure data as object-level reconstruction targets instead of point-level claims.
        new_from_figures: list[dict] = []
        new_from_tables = self._generate_criteria_from_tables(elements, criterion_rows)
        if new_from_tables:
            criterion_rows.extend(new_from_tables)

        visual_targets = self._build_visual_targets(elements, criterion_rows)
        visual_targets_doc = VisualTargetsDoc(
            visual_targets=visual_targets,
            reason_codes=["OBJECT_LEVEL_VISUAL_TARGETS"],
        )

        # Write back
        criteria[criteria_key] = criterion_rows
        self.artifacts.write_json("fingerprint/atomic_criteria.json", criteria)
        self.artifacts.write_json("fingerprint/visual_targets.json", visual_targets_doc.model_dump())

        self.log(
            "DONE",
            f"Enriched {enriched_count} criteria, added "
            f"{len(new_from_figures)} new from figures, {len(new_from_tables)} new from tables, "
            f"and built {len(visual_targets)} visual targets",
        )
        return {
            "enriched_count": enriched_count,
            "new_from_figures": len(new_from_figures),
            "new_from_tables": len(new_from_tables),
            "visual_targets": len(visual_targets),
        }

    @staticmethod
    def _build_visual_targets(
        elements: list[dict],
        criteria_rows: list[dict],
    ) -> list[VisualTarget]:
        """Build one reconstruction target per visual element."""
        visual_targets: list[VisualTarget] = []
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            related_criteria = [row for row in criteria_rows if _criterion_matches_element(row, elem)]
            series_names = _series_names_for_element(elem)
            metric_names = _metric_names_for_element(elem, related_criteria)
            model_names = _model_names_for_element(elem)
            reason_codes = ["OBJECT_LEVEL_VISUAL_TARGET"]
            if elem.get("element_type") == "figure":
                reason_codes.append("FIGURE_POINTS_NOT_EXPANDED")

            visual_targets.append(
                VisualTarget(
                    element_id=str(elem.get("element_id") or ""),
                    visual_anchor=str(elem.get("visual_anchor") or ""),
                    element_type=str(elem.get("element_type") or "figure"),
                    chart_type=elem.get("chart_type"),
                    caption=str(elem.get("caption") or ""),
                    page=elem.get("page"),
                    reference_image_path=(
                        str(elem.get("crop_path") or elem.get("raw_page_image") or "") or None
                    ),
                    axis_labels={
                        str(key): str(value)
                        for key, value in (elem.get("axis_labels") or {}).items()
                        if str(value).strip()
                    },
                    legend_entries=_dedupe_strings(str(x) for x in elem.get("legend_entries", [])),
                    series_names=series_names,
                    metric_names=metric_names,
                    model_names=model_names,
                    sampling_strategy=(
                        str(elem.get("sampling_strategy")).strip()
                        if elem.get("sampling_strategy") not in (None, "")
                        else None
                    ),
                    semantic_summary=_semantic_summary_for_element(
                        elem,
                        metric_names=metric_names,
                        series_names=series_names,
                    ),
                    reconstruction_instructions=_reconstruction_instructions_for_element(
                        elem,
                        metric_names=metric_names,
                        series_names=series_names,
                    ),
                    associated_claim_ids=[],
                    reason_codes=reason_codes,
                )
            )

        return visual_targets

    def _generate_criteria_from_tables(
        self,
        elements: list[dict],
        existing_criteria: list[dict],
    ) -> list[dict]:
        existing_keys = {_criterion_key(c) for c in existing_criteria}
        new_criteria: list[dict] = []

        for elem in elements:
            if elem.get("element_type") != "table":
                continue
            if not elem.get("data_series"):
                continue

            visual_anchor = str(elem.get("visual_anchor") or elem.get("element_id") or "").strip() or None
            visual_data = _visual_data_from_elem(elem)
            user = build_table_criteria_user_prompt(
                table_anchor=visual_anchor,
                caption=str(elem.get("caption") or ""),
                table_html="",
                context_before="",
                context_after="",
                visual_element=elem,
            )
            data, err = self.safe_chat_json(
                schema=TABLE_CRITERIA_SCHEMA,
                system=TABLE_CRITERIA_SYSTEM_PROMPT,
                user=user,
            )
            rows = rows_from_llm_table_response(data)
            criteria = normalize_table_criteria(
                rows,
                default_anchor=visual_anchor,
                input_unit_id=str(elem.get("element_id") or ""),
                source_prefix="visual_table",
                default_reason_code="VISUAL_TABLE_EXTRACTED",
                visual_data=visual_data,
            )
            if err and not criteria:
                continue

            for criterion in criteria:
                key = _criterion_key(criterion)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                new_criteria.append(criterion)

        return new_criteria


def _extract_table_anchor(criterion: dict) -> str | None:
    """Extract a table/figure anchor like 'Table 1' from criterion text."""
    direct_anchor = criterion.get("table_anchor")
    if isinstance(direct_anchor, str) and direct_anchor.strip():
        return direct_anchor.strip()

    scope = str(criterion.get("scope", ""))
    fact = str(criterion.get("fact", ""))
    text = f"{scope} {fact}"

    m = re.search(r"((?:Table|Figure|Fig\.?)\s*\d+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _criterion_matches_element(criterion: dict, elem: dict) -> bool:
    anchor_candidates = {
        str(elem.get("visual_anchor") or "").strip().lower(),
        str(elem.get("element_id") or "").strip().lower(),
    }
    anchor_candidates.discard("")
    table_anchor = str(criterion.get("table_anchor") or "").strip().lower()
    if table_anchor and table_anchor in anchor_candidates:
        return True
    visual_data = criterion.get("visual_data")
    if isinstance(visual_data, dict):
        element_id = str(visual_data.get("element_id") or "").strip().lower()
        if element_id and element_id in anchor_candidates:
            return True
    return False


def _criterion_key(criterion: dict) -> tuple[str, str, str, str]:
    anchor = str(criterion.get("table_anchor") or "").strip().lower()
    metric = str(criterion.get("metric_name") or "").strip().lower()
    value = str(criterion.get("metric_value") if criterion.get("metric_value") is not None else "").strip()
    fact = re.sub(r"\s+", " ", str(criterion.get("fact") or "")).strip().lower()
    return anchor, metric, value, fact


def _visual_data_from_elem(elem: dict) -> dict:
    visual_data = {
        "element_id": elem.get("element_id"),
        "chart_type": elem.get("chart_type"),
        "axis_labels": elem.get("axis_labels", {}),
        "legend_entries": elem.get("legend_entries", []),
        "data_series": elem.get("data_series", []),
    }
    _copy_visual_metadata(elem, visual_data)
    return visual_data


def _copy_visual_metadata(elem: dict, target: dict) -> None:
    for key in (
        "bbox",
        "raw_page_image",
        "crop_path",
        "x_axis_range",
        "y_axis_range",
        "series_semantics",
        "model_names",
        "sampling_strategy",
        "numeric_confidence",
    ):
        value = elem.get(key)
        if value not in (None, [], {}):
            target[key] = value


def _dedupe_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _series_names_for_element(elem: dict) -> list[str]:
    names: list[str] = []
    names.extend(str(x) for x in elem.get("legend_entries", []))
    for series in elem.get("data_series", []):
        if isinstance(series, dict):
            names.append(str(series.get("name") or ""))
    for row in elem.get("series_semantics", []):
        if isinstance(row, dict):
            names.append(str(row.get("name") or ""))
    return _dedupe_strings(names)


def _metric_names_for_element(elem: dict, related_criteria: list[dict]) -> list[str]:
    names: list[str] = []
    axis_labels = elem.get("axis_labels") or {}
    y_axis = str(axis_labels.get("y") or "").strip()
    if y_axis:
        names.append(y_axis)
    for row in related_criteria:
        names.append(str(row.get("metric_name") or row.get("metric") or ""))
    for row in elem.get("series_semantics", []):
        if isinstance(row, dict):
            names.append(str(row.get("metric") or ""))
    for series in elem.get("data_series", []):
        if not isinstance(series, dict):
            continue
        for value_row in series.get("values", []):
            if not isinstance(value_row, dict):
                continue
            for key in value_row:
                if _looks_like_metric_label(key):
                    names.append(str(key))
    return _dedupe_strings(names)


def _model_names_for_element(elem: dict) -> list[str]:
    names: list[str] = []
    names.extend(str(x) for x in elem.get("model_names", []))
    for row in elem.get("series_semantics", []):
        if isinstance(row, dict):
            names.append(str(row.get("model") or ""))
    return _dedupe_strings(names)


def _looks_like_metric_label(text: str) -> bool:
    lower = str(text or "").strip().lower()
    if not lower:
        return False
    return any(
        token in lower
        for token in (
            "accuracy",
            "acc",
            "error",
            "loss",
            "precision",
            "recall",
            "f1",
            "auc",
            "bleu",
            "rouge",
            "mse",
            "mae",
            "rmse",
            "perplexity",
        )
    )


def _semantic_summary_for_element(
    elem: dict,
    *,
    metric_names: list[str],
    series_names: list[str],
) -> str:
    anchor = str(elem.get("visual_anchor") or elem.get("element_id") or "This visual").strip()
    element_type = str(elem.get("element_type") or "visual").strip()
    chart_type = str(elem.get("chart_type") or "").strip()
    axis_labels = elem.get("axis_labels") or {}
    x_axis = str(axis_labels.get("x") or "").strip()

    if element_type == "table":
        parts = [f"{anchor} is a table"]
    elif chart_type:
        parts = [f"{anchor} is a {chart_type} {element_type}"]
    else:
        parts = [f"{anchor} is a {element_type}"]

    if metric_names and x_axis:
        parts.append(f"summarizing {', '.join(metric_names[:3])} against {x_axis}")
    elif metric_names:
        parts.append(f"summarizing {', '.join(metric_names[:3])}")
    if series_names:
        parts.append(f"with series {', '.join(series_names[:4])}")

    summary = " ".join(parts).strip()
    if not summary.endswith("."):
        summary += "."
    return summary


def _reconstruction_instructions_for_element(
    elem: dict,
    *,
    metric_names: list[str],
    series_names: list[str],
) -> list[str]:
    instructions: list[str] = []
    if elem.get("crop_path"):
        instructions.append("Use the saved paper crop as the phase 3 reference image.")
    elif elem.get("raw_page_image"):
        instructions.append("Use the saved paper page image as the phase 3 reference image.")

    if elem.get("element_type") == "table":
        if metric_names:
            instructions.append(
                f"Aggregate phase 2 outputs into table rows and columns for {', '.join(metric_names[:4])}."
            )
        else:
            instructions.append("Aggregate phase 2 outputs into the reported table rows and columns.")
        instructions.append("Render a reproduced table that preserves the paper anchor, row groups, and column labels.")
        return instructions

    if metric_names:
        instructions.append(
            f"Collect phase 2 outputs needed to plot {', '.join(metric_names[:4])} for this visual."
        )
    else:
        instructions.append("Collect the phase 2 outputs needed to recreate the plotted evidence.")
    if series_names:
        instructions.append(f"Plot separate series for {', '.join(series_names[:4])} and preserve the legend semantics.")
    chart_type = str(elem.get("chart_type") or "chart").strip()
    instructions.append(f"Match the paper's {chart_type} layout, axis labels, and visual anchor in phase 3.")
    return instructions
