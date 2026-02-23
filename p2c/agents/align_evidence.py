from __future__ import annotations

from p2c.agents.base import BaseAgent
from p2c.schemas import ClaimEvidence, MetricRecord, ParsedEvidence

SYSTEM_PROMPT = "You align claims with metric records. Output JSON only and include reason_codes when missing evidence."
USER_PROMPT_TEMPLATE = "Input: fingerprint/claims_ir.json + results/metrics.json. Output: results/parsed_evidence.json"


class AlignEvidenceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="align_evidence", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        metrics_doc = self.artifacts.read_json("results/metrics.json")
        records = [MetricRecord(**r) for r in metrics_doc.get("records", [])]

        evidence_rows: list[ClaimEvidence] = []
        for claim in claims_doc.get("claims", []):
            cid = claim.get("claim_id", "")
            metric_name = (claim.get("metric") or "").lower()
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

        parsed = ParsedEvidence(claim_evidence=evidence_rows, reason_codes=[])
        self.artifacts.write_json("results/parsed_evidence.json", parsed.model_dump())
        return {"parsed_evidence": parsed.model_dump()}
