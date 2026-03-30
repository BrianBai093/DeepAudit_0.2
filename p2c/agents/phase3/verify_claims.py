from __future__ import annotations

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

SYSTEM_PROMPT = "You verify claims against evidence using deterministic rules. Output JSON only."
USER_PROMPT_TEMPLATE = "Input: claims_ir + parsed_evidence + evaluability. Output: verdict + evaluability_verdict"


def evaluate_claim(
    claim: dict[str, Any],
    matched_records: list[MetricRecord],
    missing_reason: str | None = None,
) -> ClaimVerdict:
    claim_id = claim.get("claim_id", "unknown")
    ctype = claim.get("type", "other")
    target = claim.get("target")
    baseline = claim.get("baseline")
    tol = claim.get("tolerance_policy", {}) or {}
    abs_eps = float(tol.get("abs_eps", 0.01))
    rel_eps = float(tol.get("rel_eps", 0.02))

    values = [r.value for r in matched_records if r.value is not None]
    if not values:
        # Distinguish "metric exists but can't align" from "metric not found"
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
        detail="Claim type unsupported by MVP verifier",
        reason_codes=["UNSUPPORTED_CLAIM_TYPE"],
    )


class VerifyClaimsAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="verify_claims", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        parsed_doc = self.artifacts.read_json("results/parsed_evidence.json")
        evaluability_doc = EvaluabilityDoc(**self.artifacts.read_json("results/evaluability.json"))

        evidence_map: dict[str, list[MetricRecord]] = {}
        missing_reason_map: dict[str, str | None] = {}
        for row in parsed_doc.get("claim_evidence", []):
            cid = row.get("claim_id", "")
            evidence_map[cid] = [MetricRecord(**r) for r in row.get("matched_records", [])]
            missing_reason_map[cid] = row.get("missing_reason")

        verdicts: list[ClaimVerdict] = []
        for claim in claims_doc.get("claims", []):
            cid = claim.get("claim_id")
            verdicts.append(evaluate_claim(
                claim,
                evidence_map.get(cid, []),
                missing_reason=missing_reason_map.get(cid),
            ))

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
                summary=f"Numeric track: evaluated {len(verdicts)} claims.",
            )

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
                summary=f"Evaluability track: {len(eval_rows)} claims assessed.",
            )

        verdict.summary = (
            f"Numeric={verdict.status}; Evaluability={eval_verdict.status}; "
            f"claims={len(verdict.claim_verdicts)}"
        )

        self.artifacts.write_json("results/verdict.json", verdict.model_dump())
        self.artifacts.write_json("results/evaluability_verdict.json", eval_verdict.model_dump())
        return {"verdict": verdict.model_dump(), "evaluability_verdict": eval_verdict.model_dump()}
