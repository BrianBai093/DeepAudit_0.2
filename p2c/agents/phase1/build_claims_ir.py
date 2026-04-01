from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import ClaimItem, ClaimsIR, Experiment

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

EXPERIMENT_SYSTEM_PROMPT = """\
You are an expert ML paper analyst. Given extracted claims from a paper and a \
repository analysis, you will:

1. Identify each DISTINCT EXPERIMENT described in the paper.
   An experiment is a separate evaluation setting — different dataset, different \
   model variant, different test condition, etc. Two rows in the same table are \
   NOT separate experiments unless they test fundamentally different things.

2. For each experiment, determine whether the repository contains code to run it.

3. Classify each claim as either:
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
      "claim_ids": ["claim_01", "claim_02"],
      "repo_coverage": "implemented | partial | not_found",
      "repo_entrypoint": "path/to/file.py or null",
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
- For repo_coverage: "implemented" means there is clear code to run this experiment. \
  "partial" means some code exists but it's incomplete. "not_found" means no code for this.
- Do NOT invent metrics or values. Only use what appears in the input.
"""


def _build_experiment_user_prompt(
    fingerprint: dict,
    repo_analysis: dict | None,
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

    # Repo analysis
    if repo_analysis:
        sections.append("\n## Repository Analysis")
        entrypoints = repo_analysis.get("entrypoint_candidates", []) or repo_analysis.get("entrypoints", [])
        for ep in entrypoints:
            sections.append(
                f"- Entrypoint: {ep.get('path', '')} "
                f"(command: {ep.get('command', '')}, confidence: {ep.get('confidence', '?')})"
            )
        for dep in repo_analysis.get("dependency_profiles", []):
            sections.append(
                f"- Dependencies: {dep.get('profile_id', '')} "
                f"(manifests: {dep.get('manifest_paths', [])})"
            )

    return "\n".join(sections)


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
        repo_analysis: dict | None,
        base_claims: list[ClaimItem],
    ) -> ClaimsIR | None:
        """Use LLM to identify experiments, group claims, and assess repo coverage."""
        user_prompt = _build_experiment_user_prompt(fingerprint, repo_analysis)

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
                            "claim_ids": {"type": "array", "items": {"type": "string"}},
                            "repo_coverage": {
                                "type": "string",
                                "enum": ["implemented", "partial", "not_found"],
                            },
                            "repo_entrypoint": {"type": ["string", "null"]},
                            "notes": {"type": ["string", "null"]},
                        },
                        "required": ["experiment_id", "name", "claim_ids", "repo_coverage"],
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
                claim_ids = []
                seen_claim_ids: set[str] = set()
                for claim_id in exp.get("claim_ids", []):
                    normalized = str(claim_id or "").strip()
                    if not normalized or normalized not in base_claim_id_set or normalized in seen_claim_ids:
                        continue
                    claim_ids.append(normalized)
                    seen_claim_ids.add(normalized)
                if not claim_ids:
                    continue
                experiments.append(Experiment(
                    experiment_id=exp.get("experiment_id", f"exp_{len(experiments)+1}"),
                    name=exp.get("name", "unknown"),
                    description=exp.get("description", ""),
                    dataset=exp.get("dataset"),
                    table_anchor=exp.get("table_anchor"),
                    claim_ids=claim_ids,
                    repo_coverage=exp.get("repo_coverage", "not_found"),
                    repo_entrypoint=exp.get("repo_entrypoint"),
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
        return self._postprocess_repo_coverage(claims_ir, repo_analysis)

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
                conditions["visual_data"] = evidence_anchors["visual_data"]

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
    def _primary_repo_entrypoint(repo_analysis: dict | None) -> str | None:
        if not isinstance(repo_analysis, dict):
            return None
        entrypoints = [row for row in repo_analysis.get("entrypoint_candidates", []) if isinstance(row, dict)]
        if not entrypoints:
            return None
        primary_id = str(repo_analysis.get("primary_entrypoint_id") or "").strip()
        if primary_id:
            for row in entrypoints:
                if str(row.get("entrypoint_id") or "").strip() == primary_id:
                    path = str(row.get("path") or "").strip()
                    if path:
                        return path
        for row in entrypoints:
            path = str(row.get("path") or "").strip()
            if path:
                return path
        return None

    def _postprocess_repo_coverage(self, claims_ir: ClaimsIR, repo_analysis: dict | None) -> ClaimsIR:
        if not isinstance(repo_analysis, dict):
            return claims_ir
        entrypoints = [row for row in repo_analysis.get("entrypoint_candidates", []) if isinstance(row, dict)]
        if not entrypoints:
            return claims_ir

        primary_path = self._primary_repo_entrypoint(repo_analysis)
        updated = False
        experiments: list[Experiment] = []
        for exp in claims_ir.experiments:
            repo_coverage = exp.repo_coverage
            repo_entrypoint = exp.repo_entrypoint or primary_path
            notes = exp.notes
            exp_updated = False

            if repo_coverage == "not_found":
                repo_coverage = "partial"
                exp_updated = True
            if repo_entrypoint != exp.repo_entrypoint:
                exp_updated = True
            if exp_updated and (
                not notes
                or "no repository analysis content" in str(notes).lower()
                or "cannot be confirmed" in str(notes).lower()
            ):
                entrypoint_note = f" via `{repo_entrypoint}`" if repo_entrypoint else ""
                notes = (
                    "Repository analysis found runnable entrypoint(s)"
                    f"{entrypoint_note}, but the paper experiment is only partially aligned to the repo implementation."
                )
            experiments.append(
                exp.model_copy(
                    update={
                        "repo_coverage": repo_coverage,
                        "repo_entrypoint": repo_entrypoint,
                        "notes": notes,
                    }
                )
            )
            updated = updated or exp_updated

        if not updated:
            return claims_ir
        reason_codes = list(claims_ir.reason_codes)
        if "REPO_COVERAGE_POSTPROCESSED" not in reason_codes:
            reason_codes.append("REPO_COVERAGE_POSTPROCESSED")
        return claims_ir.model_copy(update={"experiments": experiments, "reason_codes": reason_codes})

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, ctx: dict) -> dict:
        fingerprint = self.artifacts.read_json("fingerprint/fingerprint.json")
        claims_from_fp, reason_codes = self._claims_from_fingerprint(fingerprint)

        # Try to load repo analysis for cross-referencing
        repo_analysis: dict | None = None
        try:
            repo_analysis = self.artifacts.read_json("task/repo_analysis.json")
        except Exception:  # noqa: BLE001
            pass

        # Primary path: LLM-based experiment identification
        claims_ir = self._build_claims_ir_via_llm(fingerprint, repo_analysis, claims_from_fp)

        if claims_ir is None:
            # Fallback: deterministic extraction
            self.log("PROGRESS", "Falling back to deterministic claim extraction")
            claims_ir = ClaimsIR(claims=claims_from_fp, reason_codes=reason_codes)
            claims_ir = self._postprocess_repo_coverage(claims_ir, repo_analysis)

        self.artifacts.write_json("fingerprint/claims_ir.json", claims_ir.model_dump())
        return {"claims_ir": claims_ir.model_dump()}
