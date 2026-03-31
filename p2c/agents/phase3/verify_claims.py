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
this experiment — verdict is INCONCLUSIVE with reason.
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

    sections.append("## Paper Claims (claims_ir.json)")
    for c in claims_doc.get("claims", []):
        cond = c.get("conditions", {})
        sections.append(
            f"- {c['claim_id']} [{c.get('type')}]: {c.get('predicate')} "
            f"(metric={c.get('metric')}, target={c.get('target')}, "
            f"scope={cond.get('scope', 'N/A')}, table={cond.get('table_anchor', 'N/A')})"
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

    values = [r.value for r in matched_records if r.value is not None]
    if not values:
        reason_codes = ["MISSING_RECORDS"]
        detail = "No numeric records available"
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
        )

    x_rep = max(values)

    if ctype == "result":
        if target is None:
            return ClaimVerdict(
                claim_id=claim_id,
                status="INCONCLUSIVE",
                detail="Target value missing for result claim",
                reason_codes=["MISSING_TARGET"],
            )
        threshold = max(abs_eps, rel_eps * abs(float(target)))
        ok = abs(x_rep - float(target)) <= threshold
        return ClaimVerdict(
            claim_id=claim_id,
            status="SUPPORTED" if ok else "NOT_SUPPORTED",
            detail=f"|x_rep-x_paper| <= eps evaluated with eps={threshold:.4f}",
            compared_value=x_rep,
            target_value=float(target),
            reason_codes=[],
        )

    if ctype == "config":
        return ClaimVerdict(
            claim_id=claim_id,
            status="INCONCLUSIVE",
            detail="Config claim; verified by successful execution",
            compared_value=x_rep,
            reason_codes=["CONFIG_CLAIM"],
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

        # ── Try LLM-based verdict ────────────────────────────────────
        user_prompt = _build_user_prompt(claims_doc, parsed_doc, evaluability_doc_raw, run_manifest)

        verdict_schema = {
            "type": "object",
            "properties": {
                "claim_verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]},
                            "detail": {"type": "string"},
                            "compared_value": {"type": ["number", "null"]},
                            "target_value": {"type": ["number", "null"]},
                            "reason_codes": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["claim_id", "status", "detail"],
                    },
                },
                "experiments_summary": {
                    "type": "object",
                    "properties": {
                        "paper_experiments": {"type": "array", "items": {"type": "string"}},
                        "repo_experiments": {"type": "array", "items": {"type": "string"}},
                        "missing_experiments": {"type": "array", "items": {"type": "string"}},
                        "coverage_ratio": {"type": "number"},
                    },
                },
            },
            "required": ["claim_verdicts"],
        }

        llm_result, llm_err = self.safe_chat_json(verdict_schema, SYSTEM_PROMPT, user_prompt)

        experiments_summary = None

        if llm_result and llm_result.get("claim_verdicts"):
            self.log("PROGRESS", "LLM verdict received, validating")
            verdicts = []
            for v in llm_result["claim_verdicts"]:
                try:
                    verdicts.append(ClaimVerdict(
                        claim_id=v.get("claim_id", "unknown"),
                        status=v.get("status", "INCONCLUSIVE"),
                        detail=v.get("detail", ""),
                        compared_value=v.get("compared_value"),
                        target_value=v.get("target_value"),
                        reason_codes=v.get("reason_codes", []),
                    ))
                except Exception:  # noqa: BLE001
                    continue
            experiments_summary = llm_result.get("experiments_summary")
        else:
            # ── Fallback: deterministic rules ────────────────────────
            self.log("PROGRESS", f"LLM unavailable ({llm_err}), using deterministic fallback")
            evidence_map: dict[str, list[MetricRecord]] = {}
            missing_reason_map: dict[str, str | None] = {}
            for row in parsed_doc.get("claim_evidence", []):
                cid = row.get("claim_id", "")
                evidence_map[cid] = [MetricRecord(**r) for r in row.get("matched_records", [])]
                missing_reason_map[cid] = row.get("missing_reason")

            verdicts = []
            for claim in claims_doc.get("claims", []):
                cid = claim.get("claim_id")
                verdicts.append(_fallback_evaluate(
                    claim,
                    evidence_map.get(cid, []),
                    missing_reason=missing_reason_map.get(cid),
                ))

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
