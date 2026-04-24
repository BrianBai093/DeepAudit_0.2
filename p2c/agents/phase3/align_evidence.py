from __future__ import annotations

import re
from collections import Counter
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import (
    ClaimEvidence,
    EvaluabilityDoc,
    EvaluabilityEntry,
    MetricRecord,
    ParsedEvidence,
    RunManifestDoc,
)

SYSTEM_PROMPT = "You align claims with metric records and execution runs. Output JSON only."
USER_PROMPT_TEMPLATE = (
    "Input: fingerprint/claims_ir.json + execution/executor_outputs/run_manifest.json + results/metrics.json. "
    "Output: results/parsed_evidence.json and results/evaluability.json"
)


class AlignEvidenceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="align_evidence", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        metrics_doc = self.artifacts.read_json("results/metrics.json")
        manifest_payload = self.artifacts.read_json("execution/executor_outputs/run_manifest.json")
        manifest = RunManifestDoc(**manifest_payload)

        records = [MetricRecord(**row) for row in metrics_doc.get("records", [])]
        claims = [row for row in claims_doc.get("claims", []) if isinstance(row, dict)]
        experiments = {
            str(exp.get("experiment_id") or ""): exp
            for exp in claims_doc.get("experiments", [])
            if isinstance(exp, dict) and exp.get("experiment_id")
        }
        runs = list(manifest.runs)

        metric_claim_count: Counter[str] = Counter()
        for claim in claims:
            metric = str(claim.get("metric") or "").strip().lower()
            if claim.get("type") == "result" and metric:
                metric_claim_count[metric] += 1

        evidence_rows: list[ClaimEvidence] = []
        evaluability_rows: list[EvaluabilityEntry] = []

        for claim in claims:
            candidate_runs = self._candidate_runs_for_claim(claim, runs)
            matched = self._match_records(
                claim=claim,
                candidate_runs=candidate_runs,
                records=records,
                ambiguous_metric=metric_claim_count.get(str(claim.get("metric") or "").lower(), 0) > 1,
            )

            if matched:
                evidence_rows.append(ClaimEvidence(claim_id=claim["claim_id"], matched_records=matched, missing_reason=None))
                evaluability_rows.append(
                    EvaluabilityEntry(
                        claim_id=claim["claim_id"],
                        evaluable="yes",
                        source=sorted({record.source for record in matched}),
                        reason=None,
                    )
                )
                continue

            missing_reason = self._missing_reason(claim, candidate_runs, experiments)
            evidence_rows.append(
                ClaimEvidence(
                    claim_id=claim["claim_id"],
                    matched_records=[],
                    missing_reason=missing_reason,
                )
            )
            evaluability_rows.append(
                EvaluabilityEntry(
                    claim_id=claim["claim_id"],
                    evaluable=self._evaluable_status(claim, candidate_runs, missing_reason),
                    source=self._candidate_sources(candidate_runs),
                    reason=missing_reason,
                )
            )

        parsed = ParsedEvidence(claim_evidence=evidence_rows, reason_codes=[])
        evaluability = EvaluabilityDoc(entries=evaluability_rows, reason_codes=[])
        self.artifacts.write_json("results/parsed_evidence.json", parsed.model_dump())
        self.artifacts.write_json("results/evaluability.json", evaluability.model_dump())
        return {"parsed_evidence": parsed.model_dump(), "evaluability": evaluability.model_dump()}

    @staticmethod
    def _candidate_runs_for_claim(claim: dict[str, Any], runs: list[Any]) -> list[Any]:
        conditions = claim.get("conditions", {}) if isinstance(claim.get("conditions"), dict) else {}
        experiment_id = str(conditions.get("experiment_id") or "").strip()
        selected = []
        for run in runs:
            if experiment_id and str(run.experiment_id or "") == experiment_id:
                selected.append(run)
        return selected

    @classmethod
    def _match_records(
        cls,
        *,
        claim: dict[str, Any],
        candidate_runs: list[Any],
        records: list[MetricRecord],
        ambiguous_metric: bool,
    ) -> list[MetricRecord]:
        metric_name = str(claim.get("metric") or "").strip().lower()
        if not metric_name:
            return []

        candidate_sources = set(cls._candidate_sources(candidate_runs))
        candidates = cls._metric_candidates_for_claim(claim)
        matched = [
            record for record in records
            if record.metric_name.lower() in candidates
            and (not candidate_sources or record.source in candidate_sources)
        ]
        if matched:
            if ambiguous_metric and claim.get("target") is not None:
                target = float(claim["target"])
                valued = [record for record in matched if record.value is not None]
                if valued:
                    valued.sort(key=lambda record: abs(float(record.value) - target))
                    best = valued[0]
                    band = max(0.02, 0.10 * abs(target))
                    return [record for record in valued if abs(float(record.value) - target) <= band] or [best]
            return matched

        if candidate_sources:
            return []

        fallback = [record for record in records if record.metric_name.lower() in candidates]
        if ambiguous_metric and len({record.source for record in fallback}) > 1:
            return []
        return fallback

    @staticmethod
    def _candidate_sources(candidate_runs: list[Any]) -> list[str]:
        sources: list[str] = []
        for run in candidate_runs:
            manifest_source = f"execution/executor_outputs/run_manifest.json:{run.run_id}"
            if manifest_source not in sources:
                sources.append(manifest_source)
            logs = getattr(run, "logs", None)
            stdout_path = getattr(logs, "stdout", None) if logs is not None else None
            if stdout_path and stdout_path not in sources:
                sources.append(stdout_path)
        return sources

    @staticmethod
    def _metric_candidates_for_claim(claim: dict[str, Any]) -> list[str]:
        metric_name = str(claim.get("metric") or "").lower().strip()
        if not metric_name:
            return []
        predicate = str(claim.get("predicate") or "").lower()
        candidates: list[str] = []

        def add(candidate: str) -> None:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        if metric_name in {"precision", "recall", "f1"}:
            if re.search(rf"\b0\s+{re.escape(metric_name)}\b", predicate):
                add(f"class_0_{metric_name}")
            elif re.search(rf"\b1\s+{re.escape(metric_name)}\b", predicate):
                add(f"class_1_{metric_name}")
            elif "avg / total" in predicate or "avg/total" in predicate or "avg total" in predicate:
                add(f"avg_total_{metric_name}")
                add(f"weighted_{metric_name}")
            elif "weighted avg" in predicate or "weighted" in predicate:
                add(f"weighted_{metric_name}")
            elif "macro avg" in predicate or "macro" in predicate:
                add(f"macro_{metric_name}")

        add(metric_name)
        return candidates

    @staticmethod
    def _missing_reason(claim: dict[str, Any], candidate_runs: list[Any], experiments: dict[str, dict[str, Any]]) -> str:
        if claim.get("type") == "config":
            return "Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup."

        conditions = claim.get("conditions", {}) if isinstance(claim.get("conditions"), dict) else {}
        experiment_id = str(conditions.get("experiment_id") or "").strip()
        metric = str(claim.get("metric") or "").strip()

        if experiment_id and not candidate_runs:
            experiment = experiments.get(experiment_id, {})
            name = experiment.get("name") or experiment_id
            return f"No recorded run for experiment `{name}`."

        if candidate_runs and metric:
            return f"Experiment run exists but metric `{metric}` could not be aligned for this claim."

        return "No aligned metric record found."

    @staticmethod
    def _evaluable_status(claim: dict[str, Any], candidate_runs: list[Any], missing_reason: str) -> str:
        if claim.get("type") == "config":
            return "no"
        if candidate_runs:
            return "partial"
        if "No recorded run" in missing_reason:
            return "no"
        return "partial"
