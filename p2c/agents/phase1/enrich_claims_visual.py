"""EnrichClaimsVisualAgent — enriches atomic criteria with visual data from PDF extraction."""

from __future__ import annotations

import re
from typing import Any

from p2c.agents.base import BaseAgent


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
            self.log("PROGRESS", "No visual elements to enrich with, skipping")
            return {}

        # Load existing atomic criteria
        try:
            criteria = self.artifacts.read_json("fingerprint/atomic_criteria.json")
        except Exception:  # noqa: BLE001
            self.log("PROGRESS", "No atomic_criteria.json found, skipping enrichment")
            return {}

        accepted = criteria.get("accepted", [])
        if not accepted:
            self.log("PROGRESS", "No accepted criteria to enrich")
            return {}

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
        for criterion in accepted:
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

            if "reason_codes" not in criterion:
                criterion["reason_codes"] = []
            if "VISUAL_ENRICHED" not in criterion["reason_codes"]:
                criterion["reason_codes"].append("VISUAL_ENRICHED")
            enriched_count += 1

        # Generate new criteria from figure data_series not covered by existing criteria
        new_criteria = self._generate_criteria_from_figures(elements, accepted)
        if new_criteria:
            accepted.extend(new_criteria)

        # Write back
        criteria["accepted"] = accepted
        self.artifacts.write_json("fingerprint/atomic_criteria.json", criteria)

        self.log(
            "DONE",
            f"Enriched {enriched_count} criteria, added {len(new_criteria)} new from figures",
        )
        return {"enriched_count": enriched_count, "new_from_figures": len(new_criteria)}

    @staticmethod
    def _generate_criteria_from_figures(
        elements: list[dict],
        existing_criteria: list[dict],
    ) -> list[dict]:
        """Create new criteria from figure data_series not already covered."""
        # Collect existing metric+value pairs to avoid duplicates
        existing_values: set[tuple[str, str]] = set()
        for c in existing_criteria:
            metric = str(c.get("metric", "")).strip().lower()
            target = str(c.get("target", ""))
            if metric:
                existing_values.add((metric, target))

        new_criteria: list[dict] = []
        for elem in elements:
            if elem.get("element_type") != "figure":
                continue
            if elem.get("chart_type") in ("diagram", "other", None):
                continue

            for series in elem.get("data_series", []):
                series_name = str(series.get("name", "")).strip()
                for point in series.get("values", []):
                    y_val = point.get("y")
                    x_label = str(point.get("x", ""))
                    if y_val is None or not isinstance(y_val, (int, float)):
                        continue

                    # Infer metric name from axis label or series name
                    y_axis = elem.get("axis_labels", {}).get("y", "")
                    metric_name = y_axis.strip().lower() if y_axis else series_name.lower()
                    if not metric_name:
                        continue

                    key = (metric_name, str(y_val))
                    if key in existing_values:
                        continue
                    existing_values.add(key)

                    scope = f"from {elem.get('visual_anchor', elem.get('element_id', ''))}"
                    if x_label:
                        scope += f", {x_label}"
                    if series_name:
                        scope += f", {series_name}"

                    new_criteria.append({
                        "fact": f"{metric_name} = {y_val}",
                        "scope": scope,
                        "metric": metric_name,
                        "target": y_val,
                        "source": "visual_extraction",
                        "visual_data": {
                            "element_id": elem.get("element_id"),
                            "chart_type": elem.get("chart_type"),
                        },
                        "reason_codes": ["VISUAL_FIGURE_EXTRACTED"],
                    })

        return new_criteria


def _extract_table_anchor(criterion: dict) -> str | None:
    """Extract a table/figure anchor like 'Table 1' from criterion text."""
    scope = str(criterion.get("scope", ""))
    fact = str(criterion.get("fact", ""))
    text = f"{scope} {fact}"

    m = re.search(r"((?:Table|Figure|Fig\.?)\s*\d+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None
