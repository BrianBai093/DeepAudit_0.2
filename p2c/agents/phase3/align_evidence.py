from __future__ import annotations

from collections import Counter
from typing import Any

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

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        metrics_doc = self.artifacts.read_json("results/metrics.json")
        alignment_payload = self.artifacts.read_json("execution/codex_outputs/claim_alignment.json")
        alignment = ClaimAlignmentDoc(**alignment_payload)

        records = [MetricRecord(**r) for r in metrics_doc.get("records", [])]
        claims = claims_doc.get("claims", [])
        experiments = claims_doc.get("experiments", [])

        align_map = {row.claim_id: row for row in alignment.claims}

        # Build experiment coverage map for richer missing_reason messages
        exp_coverage: dict[str, str] = {}  # experiment_id → repo_coverage
        exp_names: dict[str, str] = {}  # experiment_id → name
        for exp in experiments:
            eid = exp.get("experiment_id", "")
            exp_coverage[eid] = exp.get("repo_coverage", "not_found")
            exp_names[eid] = exp.get("name", eid)

        # Detect ambiguous metric names: >1 result claim sharing the same metric
        metric_claim_count: Counter[str] = Counter()
        for claim in claims:
            if claim.get("type") == "result" and claim.get("metric"):
                metric_claim_count[claim["metric"]] += 1

        evidence_rows: list[ClaimEvidence] = []
        eval_rows: list[EvaluabilityEntry] = []

        for claim in claims:
            cid = claim.get("claim_id", "")
            metric_name = (claim.get("metric") or "").lower().strip()
            conditions = claim.get("conditions", {})
            target = claim.get("target")
            claim_type = claim.get("type", "config")
            aligned = align_map.get(cid)

            # ── Gate 1: experiment not implemented → skip matching entirely ──
            exp_id = conditions.get("experiment_id", "")
            exp_cov = exp_coverage.get(exp_id, "") if exp_id else ""
            if exp_cov == "not_found":
                exp_name = exp_names.get(exp_id, exp_id)
                reason = (
                    f"Experiment '{exp_name}' ({exp_id}) is not implemented in the "
                    f"repository (repo_coverage=not_found). Cannot evaluate this claim."
                )
                evidence_rows.append(ClaimEvidence(claim_id=cid, matched_records=[], missing_reason=reason))
                eval_rows.append(EvaluabilityEntry(
                    claim_id=cid, evaluable="no",
                    source=aligned.source if aligned else [],
                    reason=reason,
                ))
                continue

            # ── Gate 2: config claims need code evidence, not metric matching ──
            if claim_type == "config":
                reason = (
                    "Configuration claim requires direct code/config evidence; "
                    "execution metrics alone do not verify the paper setup."
                )
                evidence_rows.append(ClaimEvidence(claim_id=cid, matched_records=[], missing_reason=reason))
                eval_rows.append(EvaluabilityEntry(
                    claim_id=cid, evaluable="no",
                    source=aligned.source if aligned else [],
                    reason=reason,
                ))
                continue

            # ── Normal path: match records for result claims ──
            matched = self._match_records(
                metric_name=metric_name,
                target=target,
                conditions=conditions,
                records=records,
                is_ambiguous=metric_claim_count.get(metric_name, 0) > 1,
            )

            if matched:
                evidence_rows.append(ClaimEvidence(claim_id=cid, matched_records=matched, missing_reason=None))
            else:
                evidence_rows.append(
                    ClaimEvidence(
                        claim_id=cid,
                        matched_records=[],
                        missing_reason=self._missing_reason(
                            metric_name, records, conditions,
                            exp_coverage=exp_coverage, exp_names=exp_names,
                        ),
                    )
                )

            # Evaluability: use Phase 2 alignment as a hint, but override
            # based on actual match quality.
            if matched:
                eval_rows.append(
                    EvaluabilityEntry(
                        claim_id=cid,
                        evaluable="yes",
                        source=(aligned.source if aligned else []) or ["results/metrics.json"],
                        reason=None,
                    )
                )
            elif aligned is not None:
                # Phase 2 said partial/yes but we couldn't match — downgrade
                eval_rows.append(
                    EvaluabilityEntry(
                        claim_id=cid,
                        evaluable="no" if aligned.evaluable == "no" else "partial",
                        source=aligned.source,
                        reason=aligned.reason or "metric exists but cannot be aligned to this specific claim",
                    )
                )
            else:
                eval_rows.append(
                    EvaluabilityEntry(
                        claim_id=cid,
                        evaluable="no",
                        source=[],
                        reason="missing_claim_alignment",
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

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    @staticmethod
    def _match_records(
        *,
        metric_name: str,
        target: float | None,
        conditions: dict[str, Any],
        records: list[MetricRecord],
        is_ambiguous: bool,
    ) -> list[MetricRecord]:
        """Match metric records to a claim, considering experiment context.

        When multiple claims share the same metric name (ambiguous), we use
        target proximity to pick the best-matching record rather than
        returning all records with that name — which would cause every
        claim to compare against the same max value.
        """
        if not metric_name:
            return []

        # Step 1: find all records matching by metric name
        name_matched = [r for r in records if r.metric_name.lower() == metric_name]

        if not name_matched:
            # Fallback: try prefixed variants (val_accuracy, test_accuracy, etc.)
            prefixed = [
                r for r in records
                if r.metric_name.lower().endswith(f"_{metric_name}")
                or r.metric_name.lower().startswith(f"{metric_name}_")
            ]
            if prefixed:
                name_matched = prefixed

        if not name_matched:
            return []

        # Step 2: if only one claim uses this metric, return all matches (no ambiguity)
        if not is_ambiguous:
            return name_matched

        # Step 3: ambiguous case — multiple claims want the same metric name.
        # Use target proximity to select the best-matching record(s).
        if target is not None:
            valued = [(r, abs((r.value or 0.0) - target)) for r in name_matched if r.value is not None]
            if valued:
                valued.sort(key=lambda x: x[1])
                best_dist = valued[0][1]
                # Reject if even the closest record is far from the target.
                # Use a relative+absolute gate: must be within 10% of target
                # or 0.05 absolute, whichever is larger.
                max_acceptable = max(0.05, 0.10 * abs(target))
                if best_dist > max_acceptable:
                    return []  # no record is close enough
                # Return records within a reasonable band of the closest match
                band = max(0.02, best_dist * 2.0)
                return [r for r, d in valued if d <= band]

        # No target or no valued records — can't disambiguate, return nothing
        # to force INCONCLUSIVE rather than a wrong NOT_SUPPORTED
        return []

    @staticmethod
    def _missing_reason(
        metric_name: str,
        records: list[MetricRecord],
        conditions: dict[str, Any],
        *,
        exp_coverage: dict[str, str] | None = None,
        exp_names: dict[str, str] | None = None,
    ) -> str:
        """Generate a human-readable reason why matching failed."""
        available = {r.metric_name.lower() for r in records if r.value is not None}
        table_anchor = conditions.get("table_anchor", "")
        experiment_id = conditions.get("experiment_id", "")

        # Check if we know the experiment is not in the repo
        exp_status = ""
        if experiment_id and exp_coverage:
            coverage = exp_coverage.get(experiment_id, "")
            name = (exp_names or {}).get(experiment_id, experiment_id)
            if coverage == "not_found":
                exp_status = f" Experiment '{name}' is not implemented in the repository."
            elif coverage == "partial":
                exp_status = f" Experiment '{name}' is only partially implemented in the repository."

        if metric_name and metric_name in available:
            ctx = f" ({table_anchor})" if table_anchor else ""
            return (
                f"Metric '{metric_name}' was collected but could not be aligned "
                f"to this specific claim{ctx}.{exp_status}"
            )
        elif metric_name:
            return f"No metric records matching '{metric_name}'.{exp_status}"
        else:
            return f"Claim has no metric name specified.{exp_status}"
