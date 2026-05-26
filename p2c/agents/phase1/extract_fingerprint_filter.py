from __future__ import annotations

import hashlib
import json
import os
import re

from p2c.agents.base import BaseAgent
from p2c.agents.phase1.fingerprint_prompt_templates import FILTER_SYSTEM_PROMPT, FILTER_USER_PROMPT_TEMPLATE
from p2c.schemas import (
    Fingerprint,
    FingerprintClaim,
    FingerprintConfigurations,
    FingerprintEvidenceAnchors,
    FingerprintTolerance,
)

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TABLE_DERIVED_SOURCES = {
    "table_metric",
    "llm_table_metric",
    "llm_table_param",
    "visual_table_metric",
    "visual_table_param",
}

_DEFAULT_MAX_FINGERPRINT_CLAIMS = 120


def _compact_visual_reference(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    element_id = str(raw.get("element_id") or "").strip()
    if not element_id:
        return {}
    return {"element_id": element_id}


class ExtractFingerprintFilterAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="extract_fingerprint_filter", *args, **kwargs)

    @staticmethod
    def _norm(text: str | None) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _fact_core(text: str) -> str:
        lower = re.sub(r"\s+", " ", text.lower()).strip()
        return re.sub(r"\d+(?:\.\d+)?", "#", lower)

    def _cluster_key(self, row: dict) -> str:
        facet = self._norm(str(row.get("facet") or "execution_param"))
        metric = self._norm(str(row.get("metric_name") or ""))
        entity = self._norm(str(row.get("entity") or ""))
        dataset_scope = self._norm(str(row.get("dataset_scope") or row.get("scope") or ""))
        fact_core = self._fact_core(str(row.get("fact") or row.get("criterion") or ""))
        numeric_value = row.get("metric_value")
        if numeric_value is None:
            num_match = NUMBER_RE.search(str(row.get("fact") or ""))
            numeric_value = num_match.group(0) if num_match else ""
        comparator = self._norm(str(row.get("comparator") or ""))

        # For metric rows, prioritize semantic signature over raw wording so paraphrases dedup.
        if facet == "metric_result":
            fact_core = metric or "metric"
        return "|".join(
            [
                facet,
                metric,
                entity,
                dataset_scope,
                fact_core,
                str(numeric_value),
                comparator,
            ]
        )

    @staticmethod
    def _claim_type(row: dict) -> str:
        facet = str(row.get("facet") or "")
        if facet == "metric_result":
            return "result"
        return "config"

    @staticmethod
    def _verification_logic(claim_type: str) -> str:
        return "exact_match"

    @staticmethod
    def _extract_hparams(text: str) -> dict[str, object]:
        lower = text.lower()
        out: dict[str, object] = {}

        lr = re.search(r"(?:learning rate|\blr\b)\s*(?:=|of|:)?\s*(\d+(?:\.\d+)?(?:e-?\d+)?)", lower)
        if lr:
            try:
                out["learning_rate"] = float(lr.group(1))
            except ValueError:
                out["learning_rate"] = lr.group(1)

        bs = re.search(r"batch(?:\s+size)?\s*(?:=|of|:)?\s*(\d+)", lower)
        if bs:
            out["batch_size"] = int(bs.group(1))

        ep = re.search(r"epochs?\s*(?:=|of|:)?\s*(\d+)", lower)
        if ep:
            out["epochs"] = int(ep.group(1))

        dr = re.search(r"dropout\s*(?:=|of|:)?\s*(\d+(?:\.\d+)?)", lower)
        if dr:
            out["dropout"] = float(dr.group(1))

        seed = re.search(r"seed\s*(?:=|of|:)?\s*(\d+)", lower)
        if seed:
            out["seed"] = int(seed.group(1))

        opt = re.search(r"optimizer\s*(?:=|is|:)?\s*([a-z0-9_-]+)", lower)
        if opt:
            out["optimizer"] = opt.group(1)

        return out

    @staticmethod
    def _append_unique_dict(rows: list[dict], item: dict) -> None:
        norm = (item.get("detail", "").strip().lower(), item.get("scope", "").strip().lower())
        seen = {(x.get("detail", "").strip().lower(), x.get("scope", "").strip().lower()) for x in rows}
        if norm not in seen:
            rows.append(item)

    def _build_configurations(self, selected_rows: list[dict]) -> FingerprintConfigurations:
        dataset_specs: list[dict] = []
        hyperparameters: dict[str, object] = {}
        environment: dict[str, object] = {}
        evaluation_metrics: list[str] = []

        for row in selected_rows:
            fact = str(row.get("fact") or "")
            scope = str(row.get("scope") or "")
            facet = str(row.get("facet") or "execution_param")
            source = f"{fact} {scope}".strip()

            if facet == "execution_param":
                self._append_unique_dict(
                    dataset_specs,
                    {
                        "facet": facet,
                        "detail": fact,
                        "scope": scope,
                    },
                )

            for k, v in self._extract_hparams(source).items():
                hyperparameters[k] = v

            lower = source.lower()
            if "pytorch" in lower:
                environment["framework"] = "pytorch"
            if "tensorflow" in lower:
                environment["framework"] = "tensorflow"
            cuda = re.search(r"cuda\s*(\d+(?:\.\d+)?)", lower)
            if cuda:
                environment["cuda"] = cuda.group(1)
            if "a100" in lower:
                environment["hardware"] = "A100"
            if "v100" in lower:
                environment["hardware"] = "V100"

            metric_name = row.get("metric_name")
            if isinstance(metric_name, str) and metric_name and metric_name not in evaluation_metrics:
                evaluation_metrics.append(metric_name)

        return FingerprintConfigurations(
            dataset_specs=dataset_specs,
            hyperparameters=hyperparameters,
            environment=environment,
            evaluation_metrics=evaluation_metrics,
        )

    @staticmethod
    def _fingerprint_id(
        *,
        claims: list[FingerprintClaim],
        configurations: FingerprintConfigurations,
        reason_codes: list[str],
    ) -> str:
        payload = {
            "claims": [claim.model_dump() for claim in claims],
            "configurations": configurations.model_dump(),
            "reason_codes": sorted(set(reason_codes)),
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"fp_{digest[:16]}"

    @staticmethod
    def _is_actionable_row(row: dict) -> bool:
        source_type = str(row.get("source_type") or "")
        if source_type in TABLE_DERIVED_SOURCES:
            return True

        facet = str(row.get("facet") or "")
        fact = str(row.get("fact") or "")

        if facet == "metric_result":
            return row.get("metric_value") is not None or bool(re.search(r"\d", fact))
        if facet == "execution_param":
            return bool(re.search(r"\d", fact))
        return False

    @classmethod
    def _limit_selected_indices(
        cls,
        criteria: list[dict],
        selected_indices: list[int],
        max_claims: int,
    ) -> tuple[list[int], int]:
        """Keep phase-2-facing claims bounded while preserving high-value rows."""
        if max_claims <= 0 or len(selected_indices) <= max_claims:
            return selected_indices, 0

        def priority(idx: int) -> tuple[int, int, int, int]:
            row = criteria[idx]
            claim_type = cls._claim_type(row)
            has_metric_value = row.get("metric_value") is not None
            source_type = str(row.get("source_type") or "")
            is_table_result = source_type in TABLE_DERIVED_SOURCES and claim_type == "result"
            return (
                0 if claim_type == "result" else 1,
                0 if has_metric_value else 1,
                0 if is_table_result else 1,
                idx,
            )

        kept = sorted(sorted(set(selected_indices), key=priority)[:max_claims])
        return kept, max(0, len(set(selected_indices)) - len(kept))

    def execute(self, ctx: dict) -> dict:
        atomic = self.artifacts.read_json("fingerprint/atomic_criteria.json")
        criteria = [c for c in atomic.get("criteria", []) if isinstance(c, dict)]
        filtered_criteria: list[dict] = []
        dropped_non_actionable = 0
        for row in criteria:
            if self._is_actionable_row(row):
                filtered_criteria.append(row)
            else:
                dropped_non_actionable += 1
        criteria = filtered_criteria

        if not criteria:
            reason = "NO_ATOMIC_CRITERIA_AFTER_FILTERING" if dropped_non_actionable else "NO_ATOMIC_CRITERIA"
            fingerprint = Fingerprint(reason_codes=[reason])
            self.artifacts.write_json("fingerprint/filter_clusters.json", {"clusters": [], "reason_codes": ["NO_ATOMIC_CRITERIA"]})
            self.artifacts.write_json("fingerprint/filter_selected.json", {"selected_indices": [], "reason_codes": [reason]})
            self.artifacts.write_json("fingerprint/fingerprint.json", fingerprint.model_dump())
            return {"fingerprint": fingerprint.model_dump()}

        reason_codes: list[str] = []
        if dropped_non_actionable:
            reason_codes.append("FILTER_DROPPED_NON_ACTIONABLE")
        clusters: dict[str, list[int]] = {}
        table_selected: list[int] = []
        table_seen: set[tuple[str, str, str, str]] = set()

        for idx, row in enumerate(criteria):
            if row.get("source_type") in TABLE_DERIVED_SOURCES:
                table_key = (
                    str(row.get("table_anchor") or ""),
                    str(row.get("entity") or row.get("model") or ""),
                    str(row.get("metric_name") or ""),
                    str(row.get("metric_value") or row.get("value_raw") or row.get("fact") or ""),
                )
                if table_key not in table_seen:
                    table_seen.add(table_key)
                    table_selected.append(idx)
                continue
            key = self._cluster_key(row)
            clusters.setdefault(key, []).append(idx)

        llm_budget = int(os.getenv("P2C_FILTER_LLM_CLUSTER_BUDGET", "20"))
        llm_calls = 0

        selected_indices: list[int] = []
        cluster_debug: list[dict] = []

        for key, indices in clusters.items():
            cluster_rows = [criteria[i] for i in indices]
            selected_local = 0
            selection_reason = "deterministic_first"

            if len(indices) > 1 and llm_calls < llm_budget:
                llm_calls += 1
                group = "\n".join(f"{i}. {cluster_rows[i].get('criterion') or cluster_rows[i].get('fact') or ''}" for i in range(len(cluster_rows)))
                llm_schema = {
                    "type": "object",
                    "properties": {
                        "selected_index": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["selected_index", "reason"],
                }
                llm_user = FILTER_USER_PROMPT_TEMPLATE.format(cluster_of_similar_criteria=group)
                llm_data, llm_err = self.safe_chat_json(schema=llm_schema, system=FILTER_SYSTEM_PROMPT, user=llm_user)
                if llm_data and isinstance(llm_data.get("selected_index"), int):
                    picked = int(llm_data["selected_index"])
                    if 0 <= picked < len(cluster_rows):
                        selected_local = picked
                        selection_reason = str(llm_data.get("reason") or "llm")
                    else:
                        selection_reason = "llm_out_of_range"
                else:
                    selection_reason = "llm_empty"

                if llm_err:
                    reason_codes.append("LLM_UNAVAILABLE")

            selected_global = indices[selected_local]
            selected_indices.append(selected_global)
            cluster_debug.append(
                {
                    "cluster_key": key,
                    "candidate_indices": indices,
                    "selected_index": selected_global,
                    "selection_reason": selection_reason,
                }
            )

        selected_indices = sorted(set(selected_indices).union(table_selected))
        try:
            max_claims = int(os.getenv("P2C_MAX_FINGERPRINT_CLAIMS", str(_DEFAULT_MAX_FINGERPRINT_CLAIMS)))
        except ValueError:
            max_claims = _DEFAULT_MAX_FINGERPRINT_CLAIMS
        selected_indices, dropped_by_cap = self._limit_selected_indices(criteria, selected_indices, max_claims)
        if dropped_by_cap:
            reason_codes.append("FINGERPRINT_CLAIMS_CAPPED")

        selected_rows = [criteria[i] for i in selected_indices]

        claims: list[FingerprintClaim] = []
        for out_idx, crit_idx in enumerate(selected_indices, start=1):
            row = criteria[crit_idx]
            fact = str(row.get("fact") or row.get("criterion") or "").strip() or "unknown fact"
            scope = str(row.get("scope") or "from paper context").strip() or "from paper context"
            claim_type = self._claim_type(row)
            verification_logic = self._verification_logic(claim_type)

            is_result = claim_type == "result"
            tolerance = FingerprintTolerance(
                abs=0.005 if is_result else None,
                rel=0.02 if is_result else None,
                text="default metric tolerance" if is_result else "exact config/value match",
            )

            claims.append(
                FingerprintClaim(
                    id=f"claim_{out_idx:02d}",
                    claim_type=claim_type,
                    fact=fact,
                    scope=scope,
                    comparator=row.get("comparator"),
                    verification_logic=verification_logic,
                    tolerance=tolerance,
                    evidence_anchors=FingerprintEvidenceAnchors(
                        text_anchor=f"atomic_criteria[{crit_idx}]",
                        visual_anchor=row.get("table_anchor") or None,
                        visual_data=_compact_visual_reference(row.get("visual_data")),
                    ),
                    reason_codes=[str(x) for x in row.get("reason_codes", [])],
                )
            )

        configurations = self._build_configurations(selected_rows)

        fingerprint = Fingerprint(
            fingerprint_id=self._fingerprint_id(
                claims=claims,
                configurations=configurations,
                reason_codes=reason_codes,
            ),
            configurations=configurations,
            claims=claims,
            reason_codes=sorted(set(reason_codes)),
            notes="Generated via strict guide -> atomic -> filter pipeline focused on results and execution parameters",
        )

        self.artifacts.write_json("fingerprint/filter_clusters.json", {"clusters": cluster_debug, "reason_codes": []})
        self.artifacts.write_json(
            "fingerprint/filter_selected.json",
            {
                "selected_indices": selected_indices,
                "dropped_by_cap": dropped_by_cap,
                "max_claims": max_claims,
                "reason_codes": ["FINGERPRINT_CLAIMS_CAPPED"] if dropped_by_cap else [],
            },
        )
        self.artifacts.write_json("fingerprint/fingerprint.json", fingerprint.model_dump())
        return {"fingerprint": fingerprint.model_dump()}
