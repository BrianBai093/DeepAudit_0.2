from __future__ import annotations

import json
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import (
    ClaimVerdict,
    EvaluabilityDoc,
    EvaluabilityVerdictDoc,
    EvaluabilityVerdictRow,
    MetricRecord,
    VerdictDoc,
)

SYSTEM_PROMPT = """\
You are an expert ML reproducibility auditor. You will receive:

1. **Paper claims** (claims_ir.json): what the paper says — each claim has a type \
(result or config), metric name, target value, and conditions (scope, table_anchor).
2. **Execution metrics** (parsed_evidence.json): what we actually measured by running \
the repository code — matched metric records with values and sources.
3. **Run manifest summary**: what steps were executed, what succeeded/failed, \
what metrics each step produced.
4. **Evaluability assessment**: which claims could be evaluated and which could not.

Your job is to produce a JSON verdict for each claim. For each claim, assess:

- **SUPPORTED**: The reproduced value is within tolerance of the paper's claimed value.
- **NOT_SUPPORTED**: We ran the relevant experiment but the value differs beyond tolerance.
- **INCONCLUSIVE**: We could not run the relevant experiment, or the repo does not \
implement this specific experiment, or the claim is a config parameter that cannot be \
numerically verified.

CRITICAL RULES:
- For result claims with matched metrics: compute |reproduced - target| and compare \
against threshold = max(abs_eps, rel_eps * |target|).
- For result claims WITHOUT matched metrics: check the missing_reason. If it says \
"could not be aligned" or "ALIGNMENT_AMBIGUOUS", the repo likely does not implement \
 this exact paper configuration — verdict is INCONCLUSIVE with reason.
- If repo_coverage says "not_found" but execution produced same-named metrics or \
 successful related steps, treat that as an alignment gap or evidence mismatch, not \
 as proof that the repository lacks all relevant implementation.
- For config claims: these describe dataset sizes, hyperparameters, etc. Mark them \
INCONCLUSIVE with a brief note on whether the execution implicitly satisfied them.
- Do NOT fabricate values. Only use the metrics provided.
- Be precise about WHY each claim got its verdict.

Return a JSON object with this exact structure:
{
  "claim_verdicts": [
    {
      "claim_id": "claim_XX",
      "status": "SUPPORTED|NOT_SUPPORTED|INCONCLUSIVE",
      "detail": "brief explanation",
      "compared_value": <float or null>,
      "target_value": <float or null>,
      "reason_codes": ["..."]
    }
  ],
  "experiments_summary": {
    "paper_experiments": ["list of distinct experiments/evaluations described in the paper claims"],
    "repo_experiments": ["list of experiments actually implemented and executed in the repo"],
    "missing_experiments": ["experiments in paper but not in repo"],
    "coverage_ratio": <float 0-1>
  }
}
"""


def _build_user_prompt(
    claims_doc: dict,
    parsed_evidence: dict,
    evaluability: dict,
    run_manifest: dict,
) -> str:
    sections = []

    # Experiments (if available from LLM-based Phase 1)
    experiments = claims_doc.get("experiments", [])
    if experiments:
        sections.append("## Paper Experiments")
        for exp in experiments:
            sections.append(
                f"- {exp.get('experiment_id')}: {exp.get('name')} "
                f"(dataset={exp.get('dataset', 'N/A')}, table={exp.get('table_anchor', 'N/A')}, "
                f"repo_coverage={exp.get('repo_coverage', '?')}, "
                f"entrypoint={exp.get('repo_entrypoint', 'N/A')})"
            )
            sections.append(f"  claims: {exp.get('claim_ids', [])}")
            if exp.get("notes"):
                sections.append(f"  notes: {exp['notes']}")

    sections.append("\n## Paper Claims (claims_ir.json)")
    for c in claims_doc.get("claims", []):
        cond = c.get("conditions", {})
        sections.append(
            f"- {c['claim_id']} [{c.get('type')}]: {c.get('predicate')} "
            f"(metric={c.get('metric')}, target={c.get('target')}, "
            f"scope={cond.get('scope', 'N/A')}, table={cond.get('table_anchor', 'N/A')}, "
            f"experiment={cond.get('experiment_id', 'N/A')})"
        )
        tol = c.get("tolerance_policy", {})
        sections.append(f"  tolerance: abs_eps={tol.get('abs_eps')}, rel_eps={tol.get('rel_eps')}")

    sections.append("\n## Execution Evidence (parsed_evidence.json)")
    for ev in parsed_evidence.get("claim_evidence", []):
        cid = ev.get("claim_id")
        records = ev.get("matched_records", [])
        if records:
            vals = [f"{r.get('value')} (from {r.get('source')})" for r in records]
            sections.append(f"- {cid}: matched values = {', '.join(vals)}")
        else:
            sections.append(f"- {cid}: NO MATCH — {ev.get('missing_reason', 'unknown')}")

    sections.append("\n## Evaluability")
    for entry in evaluability.get("entries", []):
        sections.append(
            f"- {entry.get('claim_id')}: evaluable={entry.get('evaluable')}, "
            f"reason={entry.get('reason', 'N/A')}"
        )

    sections.append("\n## Run Manifest Summary")
    for run in run_manifest.get("runs", []):
        status = run.get("status", "unknown")
        metrics = run.get("metrics", {})
        sections.append(
            f"- {run.get('run_id')}: status={status}, "
            f"runtime={run.get('runtime_sec', '?')}s, "
            f"metrics={json.dumps(metrics)}"
        )

    return "\n".join(sections)


def _fallback_evaluate(
    claim: dict[str, Any],
    matched_records: list[MetricRecord],
    missing_reason: str | None = None,
) -> ClaimVerdict:
    """Deterministic fallback when LLM is unavailable."""
    claim_id = claim.get("claim_id", "unknown")
    ctype = claim.get("type", "other")
    target = claim.get("target")
    tol = claim.get("tolerance_policy", {}) or {}
    abs_eps = float(tol.get("abs_eps", 0.01))
    rel_eps = float(tol.get("rel_eps", 0.02))

    if ctype == "config":
        return ClaimVerdict(
            claim_id=claim_id,
            status="INCONCLUSIVE",
            detail=(
                missing_reason
                or "Configuration claim requires direct code/config evidence; execution success alone does not verify the paper setup."
            ),
            compared_value=None,
            target_value=float(target) if target is not None else None,
            reason_codes=["CONFIG_CLAIM", "NO_DIRECT_CONFIG_EVIDENCE"],
        )

    valued_records = [r for r in matched_records if r.value is not None]
    if not valued_records:
        reason_codes = ["MISSING_RECORDS"]
        detail = "No numeric records available for this claim."
        if missing_reason and "could not be aligned" in missing_reason:
            reason_codes = ["ALIGNMENT_AMBIGUOUS"]
            detail = missing_reason
        elif missing_reason:
            detail = missing_reason
        return ClaimVerdict(
            claim_id=claim_id,
            status="INCONCLUSIVE",
            detail=detail,
            reason_codes=reason_codes,
            target_value=float(target) if target is not None else None,
        )

    if ctype == "result":
        if target is None:
            return ClaimVerdict(
                claim_id=claim_id,
                status="INCONCLUSIVE",
                detail="Target value missing for result claim",
                reason_codes=["MISSING_TARGET"],
            )
        chosen = min(valued_records, key=lambda record: abs(float(record.value) - float(target)))
        x_rep = float(chosen.value)
        threshold = max(abs_eps, rel_eps * abs(float(target)))
        abs_error = abs(x_rep - float(target))
        ok = abs_error <= threshold
        return ClaimVerdict(
            claim_id=claim_id,
            status="SUPPORTED" if ok else "NOT_SUPPORTED",
            detail=(
                f"Matched {chosen.metric_name} value {x_rep:.4f} from {chosen.source}; "
                f"absolute error {abs_error:.4f} is "
                f"{'within' if ok else 'outside'} tolerance {threshold:.4f} for target {float(target):.4f}."
            ),
            compared_value=x_rep,
            target_value=float(target),
            reason_codes=["MATCHED_METRIC", "WITHIN_TOLERANCE" if ok else "OUTSIDE_TOLERANCE"],
        )

    return ClaimVerdict(
        claim_id=claim_id,
        status="INCONCLUSIVE",
        detail="Claim type unsupported",
        reason_codes=["UNSUPPORTED_CLAIM_TYPE"],
    )


class VerifyClaimsAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="verify_claims", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        parsed_doc = self.artifacts.read_json("results/parsed_evidence.json")
        evaluability_doc_raw = self.artifacts.read_json("results/evaluability.json")
        evaluability_doc = EvaluabilityDoc(**evaluability_doc_raw)
        run_manifest = self.artifacts.read_json("execution/codex_outputs/run_manifest.json")
        evidence_map: dict[str, list[MetricRecord]] = {}
        missing_reason_map: dict[str, str | None] = {}
        for row in parsed_doc.get("claim_evidence", []):
            cid = row.get("claim_id", "")
            evidence_map[cid] = [MetricRecord(**r) for r in row.get("matched_records", [])]
            missing_reason_map[cid] = row.get("missing_reason")

        verdicts = []
        for claim in claims_doc.get("claims", []):
            cid = claim.get("claim_id")
            verdicts.append(
                _fallback_evaluate(
                    claim,
                    evidence_map.get(cid, []),
                    missing_reason=missing_reason_map.get(cid),
                )
            )
        experiments_summary = self._summarize_experiments(claims_doc, run_manifest)

        # ── Build overall verdict ────────────────────────────────────
        if not verdicts:
            verdict = VerdictDoc(
                status="INCONCLUSIVE",
                claim_verdicts=[],
                reason_codes=["NO_CLAIMS_AVAILABLE"],
                summary="No claims to evaluate.",
            )
        else:
            statuses = [v.status for v in verdicts]
            if all(s == "SUPPORTED" for s in statuses):
                overall = "SUPPORTED"
            elif all(s == "NOT_SUPPORTED" for s in statuses):
                overall = "NOT_SUPPORTED"
            elif any(s == "INCONCLUSIVE" for s in statuses):
                overall = "INCONCLUSIVE"
            else:
                overall = "PARTIALLY_SUPPORTED"
            verdict = VerdictDoc(
                status=overall,
                claim_verdicts=verdicts,
                reason_codes=[],
                summary=f"Evaluated {len(verdicts)} claims.",
            )

        # ── Evaluability verdict (unchanged) ─────────────────────────
        eval_rows: list[EvaluabilityVerdictRow] = []
        for row in evaluability_doc.entries:
            if row.evaluable == "yes":
                status = "EVALUABLE"
            elif row.evaluable == "partial":
                status = "PARTIAL"
            else:
                status = "NOT_EVALUABLE"
            eval_rows.append(
                EvaluabilityVerdictRow(
                    claim_id=row.claim_id,
                    status=status,
                    detail=row.reason or "",
                    reason_codes=[],
                )
            )

        if not eval_rows:
            eval_verdict = EvaluabilityVerdictDoc(
                status="NOT_EVALUABLE",
                claim_rows=[],
                reason_codes=["NO_EVALUABILITY_ROWS"],
                summary="No claim evaluability rows available.",
            )
        else:
            row_statuses = [r.status for r in eval_rows]
            if all(s == "EVALUABLE" for s in row_statuses):
                eval_status = "EVALUABLE"
            elif all(s == "NOT_EVALUABLE" for s in row_statuses):
                eval_status = "NOT_EVALUABLE"
            else:
                eval_status = "PARTIAL"
            eval_verdict = EvaluabilityVerdictDoc(
                status=eval_status,
                claim_rows=eval_rows,
                reason_codes=[],
                summary=f"Evaluability: {len(eval_rows)} claims assessed.",
            )

        # ── Persist ──────────────────────────────────────────────────
        verdict_data = verdict.model_dump()
        if experiments_summary:
            verdict_data["experiments_summary"] = experiments_summary

        self.artifacts.write_json("results/verdict.json", verdict_data)
        self.artifacts.write_json("results/evaluability_verdict.json", eval_verdict.model_dump())
        return {"verdict": verdict_data, "evaluability_verdict": eval_verdict.model_dump()}

    @staticmethod
    def _summarize_experiments(claims_doc: dict[str, Any], run_manifest: dict[str, Any]) -> dict[str, Any] | None:
        experiments = [row for row in claims_doc.get("experiments", []) if isinstance(row, dict)]
        if not experiments:
            return None
        paper_experiments = [str(exp.get("name") or exp.get("experiment_id") or "unknown") for exp in experiments]
        repo_experiments = [
            str(run.get("run_id") or "unknown")
            for run in run_manifest.get("runs", [])
            if str(run.get("status") or "") in {"ok", "partial"}
        ]
        missing_experiments = [
            str(exp.get("name") or exp.get("experiment_id") or "unknown")
            for exp in experiments
            if str(exp.get("repo_coverage") or "not_found") == "not_found"
        ]
        covered = sum(1 for exp in experiments if str(exp.get("repo_coverage") or "") in {"implemented", "partial"})
        return {
            "paper_experiments": paper_experiments,
            "repo_experiments": repo_experiments,
            "missing_experiments": missing_experiments,
            "coverage_ratio": covered / len(experiments) if experiments else 0.0,
        }
