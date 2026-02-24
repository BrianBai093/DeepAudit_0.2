from __future__ import annotations

from p2c.agents.base import BaseAgent
from p2c.schemas import (
    ClaimAlignmentDoc,
    ClaimEvidence,
    EvaluabilityDoc,
    EvaluabilityEntry,
    MetricRecord,
    ParsedEvidence,
)

SYSTEM_PROMPT = "You align claims with metric records and evaluability signals. Output JSON only."
USER_PROMPT_TEMPLATE = (
    "Input: fingerprint/claims_ir.json + execution/codex_outputs/claim_alignment.json + results/metrics.json. "
    "Output: results/parsed_evidence.json and results/evaluability.json"
)


class AlignEvidenceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="align_evidence", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        metrics_doc = self.artifacts.read_json("results/metrics.json")
        alignment_payload = self.artifacts.read_json("execution/codex_outputs/claim_alignment.json")
        alignment = ClaimAlignmentDoc(**alignment_payload)

        records = [MetricRecord(**r) for r in metrics_doc.get("records", [])]

        align_map = {row.claim_id: row for row in alignment.claims}

        evidence_rows: list[ClaimEvidence] = []
        eval_rows: list[EvaluabilityEntry] = []

        for claim in claims_doc.get("claims", []):
            cid = claim.get("claim_id", "")
            metric_name = (claim.get("metric") or "").lower().strip()
            matched = [r for r in records if metric_name and r.metric_name.lower() == metric_name]
            if not matched:
                matched = [r for r in records if r.metric_name.lower() != "unknown"]

            if matched:
                evidence_rows.append(ClaimEvidence(claim_id=cid, matched_records=matched, missing_reason=None))
            else:
                evidence_rows.append(
                    ClaimEvidence(
                        claim_id=cid,
                        matched_records=[],
                        missing_reason="No matching metrics records",
                    )
                )

            aligned = align_map.get(cid)
            if aligned is not None:
                eval_rows.append(
                    EvaluabilityEntry(
                        claim_id=cid,
                        evaluable=aligned.evaluable,
                        source=aligned.source,
                        reason=aligned.reason,
                    )
                )
            else:
                fallback = "yes" if matched else "no"
                eval_rows.append(
                    EvaluabilityEntry(
                        claim_id=cid,
                        evaluable=fallback,
                        source=["results/metrics.json"] if matched else [],
                        reason="fallback_from_metrics" if matched else "missing_claim_alignment",
                    )
                )

        parsed = ParsedEvidence(claim_evidence=evidence_rows, reason_codes=[])
        evaluability = EvaluabilityDoc(entries=eval_rows, reason_codes=[])
        self.artifacts.write_json("results/parsed_evidence.json", parsed.model_dump())
        self.artifacts.write_json("results/evaluability.json", evaluability.model_dump())
        return {
            "parsed_evidence": parsed.model_dump(),
            "evaluability": evaluability.model_dump(),
        }
