from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import ClaimItem, ClaimsIR, Experiment

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_DEFAULT_EXPERIMENT_PROMPT_ENTRYPOINTS = 30
_DEFAULT_EXPERIMENT_PROMPT_DEP_PROFILES = 20

EXPERIMENT_SYSTEM_PROMPT = """\
You are an expert ML paper analyst. Given extracted claims from a paper, you will:

1. Identify each DISTINCT EXPERIMENT described in the paper.
   An experiment is a separate evaluation setting — different dataset, different \
   model variant, different test condition, etc. Two rows in the same table are \
   NOT separate experiments unless they test fundamentally different things.

2. Classify each claim as either:
   - "result": a numeric outcome that can be reproduced (accuracy, F1, loss, etc.)
   - "config": a parameter or setup detail (dataset size, epochs, learning rate, etc.)
   Only result claims with a clear metric name and numeric target are useful for \
   reproducibility verification.

IMPORTANT:
- The claim IDs from the fingerprint are authoritative. Reuse those exact IDs.
- Do NOT invent new claim IDs, renumber claims, or rewrite a claim into a different fact.
- Use the claims array only to annotate existing fingerprint claims with experiment_id, \
  table_anchor, scope, is_primary, and optional notes/reason.

Return a JSON object with this EXACT structure:
{
  "experiments": [
    {
      "experiment_id": "exp_01",
      "name": "short descriptive name",
      "description": "what this experiment tests",
      "dataset": "dataset name or null",
      "table_anchor": "Table X or null",
      "notes": "why you think it is/isn't in the repo"
    }
  ],
  "claims": [
    {
      "claim_id": "claim_01",
      "type": "result",
      "predicate": "accuracy = 0.97",
      "metric": "accuracy",
      "target": 0.97,
      "experiment_id": "exp_01",
      "table_anchor": "Table 1",
      "scope": "evaluation on test set",
      "is_primary": true,
      "reason": "why this claim matters or doesn't"
    }
  ]
}

RULES:
- Every result claim MUST have metric (string) and target (float).
- target should be a ratio in [0,1] for bounded metrics (accuracy, F1, precision, recall).
  Convert percentages: 96.85% → 0.9685
- Config claims (dataset sizes, hyperparams, etc.) should be included but marked type="config" \
  with metric=null and target=null.
- is_primary=true for the main result claims the paper emphasizes. is_primary=false for \
  secondary/supporting claims.
- Do NOT invent metrics or values. Only use what appears in the input.
"""


def _build_experiment_user_prompt(
    fingerprint: dict,
) -> str:
    sections = []

    # Paper claims from fingerprint
    sections.append("## Extracted Claims from Paper")
    for i, claim in enumerate(fingerprint.get("claims", [])):
        anchors = claim.get("evidence_anchors", {})
        sections.append(
            f"{i+1}. claim_id={claim.get('id', f'claim_{i+1:02d}')} "
            f"[{claim.get('claim_type', '?')}] "
            f"fact=\"{claim.get('fact', '')}\" "
            f"scope=\"{claim.get('scope', '')}\" "
            f"table={anchors.get('visual_anchor', 'N/A')} "
            f"reason_codes={claim.get('reason_codes', [])}"
        )

    # Paper configurations
    configs = fingerprint.get("configurations", {})
    if configs:
        sections.append("\n## Paper Configurations")
        for spec in configs.get("dataset_specs", []):
            sections.append(f"- {spec.get('detail', '')} ({spec.get('scope', '')})")
        if configs.get("hyperparameters"):
            sections.append(f"- Hyperparameters: {json.dumps(configs['hyperparameters'])}")
        if configs.get("evaluation_metrics"):
            sections.append(f"- Evaluation metrics: {configs['evaluation_metrics']}")

    return "\n".join(sections)


def _compact_visual_reference(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    element_id = str(raw.get("element_id") or "").strip()
    if not element_id:
        return {}
    return {"element_id": element_id}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BuildClaimsIRAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="build_claims_ir", *args, **kwargs)

    # ------------------------------------------------------------------
    # Deterministic helpers (kept as fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metric_and_target(text: str) -> tuple[str | None, float | None]:
        lower = text.lower()
        metric = None
        for name in [
            "accuracy", "acc", "f1", "bleu", "auc", "loss",
            "precision", "recall", "mse", "mae", "perplexity", "rmse", "rouge",
        ]:
            if name in lower:
                metric = name
                break

        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m_pct:
            val = float(m_pct.group(1)) / 100.0
            BOUNDED = {"accuracy", "acc", "f1", "auc", "precision", "recall", "bleu", "rouge"}
            if metric in BOUNDED and val > 1.0:
                return metric, None
            return metric, val
        m_dec = re.search(r"\b(0\.\d+|1\.0+)\b", text)
        if m_dec:
            return metric, float(m_dec.group(1))

        m_int = re.search(r"\b(\d+)\b", text)
        if m_int and metric:
            val = float(m_int.group(1))
            BOUNDED = {"accuracy", "acc", "f1", "auc", "precision", "recall", "bleu", "rouge"}
            if metric in BOUNDED and val > 100.0:
                return metric, None
            if metric in BOUNDED and 1.0 < val <= 100.0:
                return metric, val / 100.0

        return metric, None

    # ------------------------------------------------------------------
    # LLM-based experiment identification
    # ------------------------------------------------------------------

    def _build_claims_ir_via_llm(
        self,
        fingerprint: dict,
        base_claims: list[ClaimItem],
    ) -> ClaimsIR | None:
        """Use LLM to identify experiments and group claims using paper evidence only."""
        user_prompt = _build_experiment_user_prompt(fingerprint)

        schema = {
            "type": "object",
            "properties": {
                "experiments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "experiment_id": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "dataset": {"type": ["string", "null"]},
                            "table_anchor": {"type": ["string", "null"]},
                            "notes": {"type": ["string", "null"]},
                        },
                        "required": ["experiment_id", "name"],
                    },
                },
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "type": {"type": "string", "enum": ["result", "config"]},
                            "predicate": {"type": "string"},
                            "metric": {"type": ["string", "null"]},
                            "target": {"type": ["number", "null"]},
                            "experiment_id": {"type": ["string", "null"]},
                            "table_anchor": {"type": ["string", "null"]},
                            "scope": {"type": ["string", "null"]},
                            "is_primary": {"type": "boolean"},
                            "reason": {"type": ["string", "null"]},
                        },
                        "required": ["claim_id", "type", "predicate"],
                    },
                },
            },
            "required": ["experiments", "claims"],
        }

        llm_data, llm_err = self.safe_chat_json(schema, EXPERIMENT_SYSTEM_PROMPT, user_prompt)
        if not llm_data:
            self.log("PROGRESS", f"LLM unavailable for experiment identification: {llm_err}")
            return None

        base_claim_ids = [claim.claim_id for claim in base_claims]
        base_claim_id_set = set(base_claim_ids)

        # Parse experiments
        experiments: list[Experiment] = []
        for exp in llm_data.get("experiments", []):
            try:
                experiments.append(Experiment(
                    experiment_id=exp.get("experiment_id", f"exp_{len(experiments)+1}"),
                    name=exp.get("name", "unknown"),
                    description=exp.get("description", ""),
                    dataset=exp.get("dataset"),
                    table_anchor=exp.get("table_anchor"),
                    notes=exp.get("notes"),
                ))
            except Exception:  # noqa: BLE001
                continue

        claims = [claim.model_copy(deep=True) for claim in base_claims]
        llm_claim_rows = [row for row in llm_data.get("claims", []) if isinstance(row, dict)]
        llm_claim_ids = {
            str(row.get("claim_id") or "").strip()
            for row in llm_claim_rows
            if str(row.get("claim_id") or "").strip()
        }
        reason_codes = ["LLM_EXPERIMENT_IDENTIFICATION", "SOURCE_FINGERPRINT_CLAIMS"]

        if llm_claim_rows and llm_claim_ids == base_claim_id_set:
            merged_claims: list[ClaimItem] = []
            llm_claim_map = {
                str(row.get("claim_id")).strip(): row
                for row in llm_claim_rows
                if str(row.get("claim_id") or "").strip() in base_claim_id_set
            }
            for claim in claims:
                row = llm_claim_map.get(claim.claim_id, {})
                conditions = dict(claim.conditions)
                if not conditions.get("scope") and row.get("scope"):
                    conditions["scope"] = row["scope"]
                if not conditions.get("table_anchor") and row.get("table_anchor"):
                    conditions["table_anchor"] = row["table_anchor"]
                if row.get("experiment_id"):
                    conditions["experiment_id"] = row["experiment_id"]
                if row.get("is_primary") is not None:
                    conditions["is_primary"] = bool(row["is_primary"])

                metric = claim.metric
                target = claim.target
                if metric is None and row.get("metric"):
                    metric = row.get("metric")
                if target is None and row.get("target") is not None:
                    target = row.get("target")
                if claim.type == "result" and (metric is None or target is None):
                    parsed_metric, parsed_target = self._extract_metric_and_target(
                        row.get("predicate", "") or claim.predicate
                    )
                    metric = metric or parsed_metric
                    target = target if target is not None else parsed_target

                merged_claims.append(
                    claim.model_copy(
                        update={
                            "metric": metric,
                            "target": target,
                            "conditions": conditions,
                            "unverifiable_from_paper": (
                                claim.type == "result" and target is None
                            ),
                            "code_verifiable": (
                                (claim.type == "result" and metric is not None and target is not None)
                                or claim.type == "config"
                            ),
                            "notes": row.get("reason") or claim.notes,
                        }
                    )
                )
            claims = merged_claims
        elif llm_claim_rows:
            reason_codes.append("LLM_CLAIM_SET_MISMATCH")
            self.log(
                "PROGRESS",
                "LLM claim set mismatched fingerprint claims; preserving fingerprint claims and merging experiments only",
            )

        claims_ir = ClaimsIR(
            experiments=experiments,
            claims=claims,
            reason_codes=reason_codes,
        )
        return self._attach_experiment_rollups(claims_ir)

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------

    def _claims_from_fingerprint(self, fingerprint: dict) -> tuple[list[ClaimItem], list[str]]:
        """Fallback: deterministic claim extraction when LLM is unavailable."""
        rows = [r for r in fingerprint.get("claims", []) if isinstance(r, dict)]
        if not rows:
            return [], []

        reason_codes = ["SOURCE_FINGERPRINT_CLAIMS", "DETERMINISTIC_FALLBACK"]
        out: list[ClaimItem] = []

        for idx, row in enumerate(rows, start=1):
            claim_id = str(row.get("id") or f"C{idx}")
            fact = str(row.get("fact") or "").strip()
            scope = str(row.get("scope") or "").strip()
            claim_type = str(row.get("claim_type") or "config")
            tol = row.get("tolerance") or {}

            metric, target = self._extract_metric_and_target(fact)
            mapped_type = "result" if claim_type == "result" else "config"

            conditions: dict[str, Any] = {}
            if scope:
                conditions["scope"] = scope
            evidence_anchors = row.get("evidence_anchors") or {}
            if evidence_anchors.get("visual_anchor"):
                conditions["table_anchor"] = str(evidence_anchors["visual_anchor"])
            if evidence_anchors.get("visual_data"):
                compact_visual = _compact_visual_reference(evidence_anchors["visual_data"])
                if compact_visual:
                    conditions["visual_data"] = compact_visual

            out.append(
                ClaimItem(
                    claim_id=claim_id,
                    type=mapped_type,
                    predicate=fact or "fingerprint claim",
                    metric=metric,
                    target=target,
                    baseline=None,
                    conditions=conditions,
                    aggregation="best" if mapped_type == "result" else None,
                    evidence_set=[str(evidence_anchors.get("text_anchor") or "fingerprint")],
                    tolerance_policy={
                        "abs_eps": float(tol.get("abs", 0.02) or 0.02),
                        "rel_eps": float(tol.get("rel", 0.03) or 0.03),
                    },
                    unverifiable_from_paper=(target is None and mapped_type == "result"),
                    code_verifiable=(mapped_type == "result" and metric is not None and target is not None)
                    or mapped_type == "config",
                    reason_codes=[str(x) for x in row.get("reason_codes", [])],
                    notes=str(fingerprint.get("notes") or "") or None,
                )
            )

        return out, reason_codes

    @staticmethod
    def _attach_experiment_rollups(claims_ir: ClaimsIR) -> ClaimsIR:
        claims = [claim.model_copy(deep=True) for claim in claims_ir.claims]
        experiments = [exp.model_copy(deep=True) for exp in claims_ir.experiments]

        experiment_by_id = {exp.experiment_id: exp for exp in experiments}
        synthetic_counter = len(experiments)

        def ensure_experiment(experiment_id: str, *, anchor: str | None = None) -> Experiment:
            nonlocal synthetic_counter
            exp = experiment_by_id.get(experiment_id)
            if exp is not None:
                return exp
            synthetic_counter += 1
            exp = Experiment(
                experiment_id=experiment_id,
                name=experiment_id.replace("_", " "),
                description="",
                dataset=None,
                table_anchor=anchor,
                notes="Synthesized from claim grouping.",
            )
            experiment_by_id[experiment_id] = exp
            experiments.append(exp)
            return exp

        if not experiments:
            fallback_id = "exp_01"
            experiments.append(
                Experiment(
                    experiment_id=fallback_id,
                    name="paper experiment 1",
                    description="Fallback experiment synthesized from paper claims.",
                    notes="Synthesized because experiment extraction returned no explicit experiments.",
                )
            )
            experiment_by_id[fallback_id] = experiments[0]

        for idx, claim in enumerate(claims, start=1):
            conditions = dict(claim.conditions)
            experiment_id = str(conditions.get("experiment_id") or "").strip()
            if not experiment_id:
                anchor = str(conditions.get("table_anchor") or "").strip()
                if anchor:
                    normalized = re.sub(r"[^a-z0-9]+", "_", anchor.lower()).strip("_")
                    experiment_id = normalized or f"exp_{idx:02d}"
                else:
                    experiment_id = experiments[0].experiment_id
                conditions["experiment_id"] = experiment_id
                claims[idx - 1] = claim.model_copy(update={"conditions": conditions})
                claim = claims[idx - 1]
            ensure_experiment(experiment_id, anchor=conditions.get("table_anchor"))

        updated_experiments: list[Experiment] = []
        dropped_empty_experiments = 0
        for exp in experiments:
            related_claims = [
                claim for claim in claims
                if str((claim.conditions or {}).get("experiment_id") or "").strip() == exp.experiment_id
            ]
            claim_ids = [claim.claim_id for claim in related_claims]
            if not claim_ids:
                dropped_empty_experiments += 1
                continue
            primary_metrics: list[str] = []
            is_primary = False
            dataset = exp.dataset
            table_anchor = exp.table_anchor
            description = exp.description
            notes = exp.notes

            for claim in related_claims:
                conditions = claim.conditions or {}
                if not dataset and conditions.get("dataset"):
                    dataset = str(conditions["dataset"])
                if not table_anchor and conditions.get("table_anchor"):
                    table_anchor = str(conditions["table_anchor"])
                if claim.metric and claim.metric not in primary_metrics and (
                    conditions.get("is_primary", False) or claim.type == "result"
                ):
                    primary_metrics.append(str(claim.metric))
                if conditions.get("is_primary", False):
                    is_primary = True

            if not description and related_claims:
                description = related_claims[0].predicate
            if not notes and not related_claims:
                notes = "No claims were grouped into this experiment."

            updated_experiments.append(
                exp.model_copy(
                    update={
                        "dataset": dataset,
                        "table_anchor": table_anchor,
                        "primary_metrics": primary_metrics,
                        "is_primary": is_primary or bool(primary_metrics),
                        "description": description,
                        "notes": notes,
                    }
                )
            )

        reason_codes = list(claims_ir.reason_codes)
        if "EXPERIMENT_ROLLUPS_ATTACHED" not in reason_codes:
            reason_codes.append("EXPERIMENT_ROLLUPS_ATTACHED")
        if dropped_empty_experiments:
            reason_codes.append("EXPERIMENTS_WITHOUT_CLAIMS_DROPPED")
        return claims_ir.model_copy(update={"experiments": updated_experiments, "claims": claims, "reason_codes": reason_codes})

    @staticmethod
    def _append_unique(values: list[str], candidate: str) -> None:
        item = str(candidate or "").strip()
        if item and item not in values:
            values.append(item)

    def _sync_visual_associations(self, claims_ir: ClaimsIR) -> None:
        claim_ids_by_element: dict[str, list[str]] = {}
        claim_ids_by_anchor: dict[str, list[str]] = {}

        for claim in claims_ir.claims:
            conditions = claim.conditions if isinstance(claim.conditions, dict) else {}
            anchor = str(conditions.get("table_anchor") or "").strip().lower()
            if anchor:
                claim_ids_by_anchor.setdefault(anchor, [])
                self._append_unique(claim_ids_by_anchor[anchor], claim.claim_id)

            visual_data = conditions.get("visual_data")
            if isinstance(visual_data, dict):
                element_id = str(visual_data.get("element_id") or "").strip().lower()
                if element_id:
                    claim_ids_by_element.setdefault(element_id, [])
                    self._append_unique(claim_ids_by_element[element_id], claim.claim_id)

        self._write_visual_claim_links(
            "fingerprint/visual_elements.json",
            "elements",
            claim_ids_by_element,
            claim_ids_by_anchor,
            "VISUAL_ELEMENTS_LINKED_TO_CLAIMS",
        )
        self._write_visual_claim_links(
            "fingerprint/visual_targets.json",
            "visual_targets",
            claim_ids_by_element,
            claim_ids_by_anchor,
            "VISUAL_TARGETS_LINKED_TO_CLAIMS",
        )

    def _write_visual_claim_links(
        self,
        artifact_path: str,
        rows_key: str,
        claim_ids_by_element: dict[str, list[str]],
        claim_ids_by_anchor: dict[str, list[str]],
        reason_code: str,
    ) -> None:
        try:
            doc = self.artifacts.read_json(artifact_path)
        except Exception:  # noqa: BLE001
            return

        rows = doc.get(rows_key)
        if not isinstance(rows, list) or not rows:
            return

        updated_rows: list[dict[str, Any]] = []
        changed = False
        for row in rows:
            if not isinstance(row, dict):
                updated_rows.append(row)
                continue
            claim_ids: list[str] = []
            element_id = str(row.get("element_id") or "").strip().lower()
            anchor = str(row.get("visual_anchor") or "").strip().lower()
            for claim_id in claim_ids_by_element.get(element_id, []):
                self._append_unique(claim_ids, claim_id)
            for claim_id in claim_ids_by_anchor.get(anchor, []):
                self._append_unique(claim_ids, claim_id)
            if row.get("associated_claim_ids") != claim_ids:
                next_row = dict(row)
                next_row["associated_claim_ids"] = claim_ids
                updated_rows.append(next_row)
                changed = True
            else:
                updated_rows.append(row)

        if not changed:
            return

        next_doc = dict(doc)
        next_doc[rows_key] = updated_rows
        reason_codes = [str(code) for code in next_doc.get("reason_codes", []) if str(code).strip()]
        if reason_code not in reason_codes:
            reason_codes.append(reason_code)
        next_doc["reason_codes"] = reason_codes
        self.artifacts.write_json(artifact_path, next_doc)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, ctx: dict) -> dict:
        fingerprint = self.artifacts.read_json("fingerprint/fingerprint.json")
        claims_from_fp, reason_codes = self._claims_from_fingerprint(fingerprint)

        # Primary path: LLM-based experiment identification
        claims_ir = self._build_claims_ir_via_llm(fingerprint, claims_from_fp)

        if claims_ir is None:
            # Fallback: deterministic extraction
            self.log("PROGRESS", "Falling back to deterministic claim extraction")
            claims_ir = self._attach_experiment_rollups(
                ClaimsIR(claims=claims_from_fp, reason_codes=reason_codes)
            )

        self.artifacts.write_json("fingerprint/claims_ir.json", claims_ir.model_dump())
        self._sync_visual_associations(claims_ir)
        return {"claims_ir": claims_ir.model_dump()}
