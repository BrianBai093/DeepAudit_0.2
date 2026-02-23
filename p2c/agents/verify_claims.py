from __future__ import annotations

from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import ClaimVerdict, MetricRecord, VerdictDoc

SYSTEM_PROMPT = "You verify claims against evidence using deterministic rules. Output JSON only."
USER_PROMPT_TEMPLATE = "Input: claims_ir + parsed_evidence. Output: results/verdict.json"


def evaluate_claim(claim: dict[str, Any], matched_records: list[MetricRecord]) -> ClaimVerdict:
    claim_id = claim.get("claim_id", "unknown")
    ctype = claim.get("type", "other")
    target = claim.get("target")
    baseline = claim.get("baseline")
    tol = claim.get("tolerance_policy", {}) or {}
    abs_eps = float(tol.get("abs_eps", 0.01))
    rel_eps = float(tol.get("rel_eps", 0.02))

    values = [r.value for r in matched_records if r.value is not None]
    if not values:
        return ClaimVerdict(
            claim_id=claim_id,
            status="INCONCLUSIVE",
            detail="No numeric records available",
            reason_codes=["MISSING_RECORDS"],
        )

    x_rep = max(values)

    if ctype == "absolute":
        if target is None:
            return ClaimVerdict(
                claim_id=claim_id,
                status="INCONCLUSIVE",
                detail="Target value missing for absolute claim",
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

    if ctype == "relative":
        if target is None:
            return ClaimVerdict(
                claim_id=claim_id,
                status="INCONCLUSIVE",
                detail="Target missing for relative claim",
                reason_codes=["MISSING_TARGET"],
            )
        if baseline is None:
            # Fallback: require reproduced metric near target.
            ok = x_rep >= float(target) - abs_eps
            return ClaimVerdict(
                claim_id=claim_id,
                status="SUPPORTED" if ok else "NOT_SUPPORTED",
                detail="Baseline missing; fallback to x_rep >= target - eps",
                compared_value=x_rep,
                target_value=float(target),
            )
        delta_paper = float(target) - float(baseline)
        delta_rep = x_rep - float(baseline)
        ok = delta_rep >= delta_paper - abs_eps
        return ClaimVerdict(
            claim_id=claim_id,
            status="SUPPORTED" if ok else "NOT_SUPPORTED",
            detail="Checked delta_rep >= delta_paper - eps",
            compared_value=delta_rep,
            target_value=delta_paper,
        )

    if ctype == "ranking":
        # MVP fallback: if there is any parsed metric, treat as partial support due to missing rank labels.
        return ClaimVerdict(
            claim_id=claim_id,
            status="INCONCLUSIVE",
            detail="Ranking evidence requires labeled model leaderboard records",
            compared_value=x_rep,
            reason_codes=["RANKING_EVIDENCE_INSUFFICIENT"],
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

        evidence_map = {
            row.get("claim_id"): [MetricRecord(**r) for r in row.get("matched_records", [])]
            for row in parsed_doc.get("claim_evidence", [])
        }

        verdicts: list[ClaimVerdict] = []
        for claim in claims_doc.get("claims", []):
            cid = claim.get("claim_id")
            verdicts.append(evaluate_claim(claim, evidence_map.get(cid, [])))

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
                # Missing evidence must remain inconclusive.
                overall = "INCONCLUSIVE"
            else:
                overall = "PARTIALLY_SUPPORTED"
            verdict = VerdictDoc(
                status=overall,
                claim_verdicts=verdicts,
                reason_codes=[],
                summary=f"Evaluated {len(verdicts)} claims.",
            )

        self.artifacts.write_json("results/verdict.json", verdict.model_dump())
        return {"verdict": verdict.model_dump()}
