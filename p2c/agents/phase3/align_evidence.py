from __future__ import annotations

import re
from collections import Counter
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase3.claim_inputs import load_effective_claims_ir
from p2c.agents.phase3.execution_summary_evidence import load_effective_run_manifest
from p2c.schemas import (
    ClaimEvidence,
    EvaluabilityDoc,
    EvaluabilityEntry,
    MetricRecord,
    ParsedEvidence,
    RunManifestDoc,
)

SYSTEM_PROMPT = """\
You align claims with metric records and execution runs. Output JSON only.
Respect claim scope qualifiers before numeric proximity: algorithm (BP/FA/DFA/DRTP/PEPITA),
dataset (MNIST/CIFAR10/CIFAR100), and architecture (fully connected/FC vs convolutional/conv)
must match the metric provenance when those qualifiers are present.
"""
USER_PROMPT_TEMPLATE = (
    "Input: results/effective_claims_ir.json + results/effective_run_manifest.json + results/metrics.json. "
    "Output: results/parsed_evidence.json and results/evaluability.json"
)


class AlignEvidenceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="align_evidence", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        claims_doc = load_effective_claims_ir(self.artifacts)
        metrics_doc = self.artifacts.read_json("results/metrics.json")
        manifest_payload = load_effective_run_manifest(self.artifacts)
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
        candidate_experiment_ids = {
            str(getattr(run, "experiment_id", "") or getattr(run, "run_id", "") or "")
            for run in candidate_runs
        }
        candidates = cls._metric_candidates_for_claim(claim)
        matched_all = [
            record for record in records
            if cls._metric_matches(record.metric_name, candidates)
            and (
                not candidate_sources
                or record.source in candidate_sources
                or str(record.experiment_id or "") in candidate_experiment_ids
            )
        ]
        matched = cls._scope_filtered_records(claim, matched_all, candidate_runs)
        if matched:
            if ambiguous_metric and claim.get("target") is not None:
                target = float(claim["target"])
                valued = cls._highest_fidelity_records([record for record in matched if record.value is not None])
                if valued:
                    valued.sort(key=lambda record: abs(float(record.value) - target))
                    best = valued[0]
                    band = max(0.02, 0.10 * abs(target))
                    return [record for record in valued if abs(float(record.value) - target) <= band] or [best]
            return matched

        if candidate_sources:
            return []

        fallback_all = [record for record in records if cls._metric_matches(record.metric_name, candidates)]
        fallback = cls._scope_filtered_records(claim, fallback_all, candidate_runs)
        if ambiguous_metric and len({record.source for record in fallback}) > 1:
            return []
        return fallback

    @staticmethod
    def _candidate_sources(candidate_runs: list[Any]) -> list[str]:
        sources: list[str] = []
        for run in candidate_runs:
            effective_source = f"results/effective_run_manifest.json:{run.run_id}"
            if effective_source not in sources:
                sources.append(effective_source)
            manifest_source = f"execution/executor_outputs/run_manifest.json:{run.run_id}"
            if manifest_source not in sources:
                sources.append(manifest_source)
            experiment_id = str(getattr(run, "experiment_id", "") or "").strip()
            if experiment_id:
                for summary_name in (
                    "EXECUTION_SUMMARY_FINAL.md",
                    "EXECUTION_SUMMARY.md",
                ):
                    summary_source = f"execution/executor_outputs/{summary_name}:{experiment_id}"
                    if summary_source not in sources:
                        sources.append(summary_source)
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
    def _metric_matches(record_metric: str, candidates: list[str]) -> bool:
        metric = str(record_metric or "").lower().strip()
        if not metric:
            return False
        for candidate in candidates:
            cand = str(candidate or "").lower().strip()
            if not cand:
                continue
            if metric == cand or metric.endswith(f"_{cand}") or metric.startswith(f"{cand}_"):
                return True
            tokens = [tok for tok in re.split(r"[^a-z0-9]+", metric) if tok]
            if cand in tokens:
                return True
        return False

    @classmethod
    def _scope_filtered_records(
        cls,
        claim: dict[str, Any],
        records: list[MetricRecord],
        candidate_runs: list[Any],
    ) -> list[MetricRecord]:
        profile = cls._claim_scope_profile(claim)
        if not any(profile.values()):
            return cls._prefer_test_records(records, claim)

        run_text_by_id = {
            str(getattr(run, "run_id", "") or ""): cls._normalize_scope_text(
                " ".join(
                    str(value or "")
                    for value in (
                        getattr(run, "run_id", None),
                        getattr(run, "experiment_id", None),
                        getattr(run, "experiment_name", None),
                        getattr(run, "dataset", None),
                        getattr(run, "command", None),
                        " ".join(getattr(run, "commands_attempted", []) or []),
                    )
                )
            )
            for run in candidate_runs
        }
        filtered = [
            record for record in records
            if cls._record_satisfies_scope(record, profile, run_text_by_id)
        ]
        return cls._prefer_test_records(filtered, claim)

    @classmethod
    def _record_satisfies_scope(
        cls,
        record: MetricRecord,
        profile: dict[str, set[str]],
        run_text_by_id: dict[str, set[str]],
    ) -> bool:
        metric_tokens = cls._normalize_scope_text(record.metric_name)
        run_tokens = run_text_by_id.get(str(record.run_id or ""), set())
        combined = metric_tokens | run_tokens

        algorithms = profile.get("algorithms", set())
        if algorithms and not any(cls._algorithm_matches(algorithm, metric_tokens) for algorithm in algorithms):
            return False

        datasets = profile.get("datasets", set())
        if datasets and not datasets <= metric_tokens:
            return False

        architectures = profile.get("architectures", set())
        if "conv" in architectures and not ({"conv", "convolutional"} & combined):
            return False
        if "fc" in architectures and {"conv", "convolutional"} & combined:
            return False

        return True

    @staticmethod
    def _algorithm_matches(algorithm: str, tokens: set[str]) -> bool:
        aliases = {
            "bp": {"bp"},
            "fa": {"fa"},
            "dfa": {"dfa"},
            "drtp": {"drtp"},
            "pepita": {"pepita", "erin"},
        }
        return bool(aliases.get(algorithm, {algorithm}) & tokens)

    @classmethod
    def _claim_scope_profile(cls, claim: dict[str, Any]) -> dict[str, set[str]]:
        conditions = claim.get("conditions", {}) if isinstance(claim.get("conditions"), dict) else {}
        text = " ".join(
            str(value or "")
            for value in (
                claim.get("predicate"),
                conditions.get("scope"),
                conditions.get("dataset"),
                conditions.get("table_anchor"),
            )
        ).lower()
        tokens = cls._normalize_scope_text(text)
        algorithms: set[str] = set()
        for name in ("dfa", "drtp", "pepita", "bp", "fa"):
            if name in tokens:
                algorithms.add(name)
        if "erin" in tokens:
            algorithms.add("pepita")

        datasets: set[str] = set()
        for dataset in ("mnist", "cifar10", "cifar100"):
            if dataset in tokens:
                datasets.add(dataset)

        architectures: set[str] = set()
        if {"fc", "fully", "connected"} & tokens or "fully_connected" in text:
            architectures.add("fc")
        if {"conv", "convolutional"} & tokens:
            architectures.add("conv")

        return {"algorithms": algorithms, "datasets": datasets, "architectures": architectures}

    @staticmethod
    def _normalize_scope_text(text: Any) -> set[str]:
        normalized = str(text or "").lower()
        normalized = normalized.replace("cifar-10", "cifar10").replace("cifar 10", "cifar10")
        normalized = normalized.replace("cifar-100", "cifar100").replace("cifar 100", "cifar100")
        normalized = normalized.replace("fully-connected", "fully connected").replace("fully_connected", "fully connected")
        tokens = {tok for tok in re.split(r"[^a-z0-9]+", normalized) if tok}
        for token in list(tokens):
            if "conv" in token:
                tokens.add("conv")
            if re.search(r"(?:^|[0-9])fc(?:$|[0-9a-z])", token):
                tokens.add("fc")
        return tokens

    @staticmethod
    def _prefer_test_records(records: list[MetricRecord], claim: dict[str, Any]) -> list[MetricRecord]:
        if not records:
            return []
        claim_text = f"{claim.get('predicate', '')} {claim.get('conditions', {})}".lower()
        if "train" not in claim_text:
            non_train = [record for record in records if "train" not in record.metric_name.lower()]
            if non_train:
                records = non_train
        test_records = [record for record in records if "test" in record.metric_name.lower()]
        if test_records:
            records = test_records
        trend_records = [
            record for record in records
            if "trend" in record.metric_name.lower() or getattr(record, "fidelity", None) in {"trend", "artifact", "full"}
        ]
        return trend_records or records

    @staticmethod
    def _highest_fidelity_records(records: list[MetricRecord]) -> list[MetricRecord]:
        if not records:
            return []

        def rank(record: MetricRecord) -> int:
            outcome = str(getattr(record, "execution_outcome", None) or "")
            fidelity = str(getattr(record, "fidelity", None) or "")
            metric_name = str(getattr(record, "metric_name", "") or "").lower()
            if "_full_" in metric_name or metric_name.startswith("full_") or metric_name.endswith("_full"):
                return 5
            if "_artifact_" in metric_name or metric_name.startswith("artifact_") or metric_name.endswith("_artifact"):
                return 4
            if "_trend_" in metric_name or metric_name.startswith("trend_") or metric_name.endswith("_trend"):
                return 3
            if "_smoke_" in metric_name or metric_name.startswith("smoke_") or metric_name.endswith("_smoke"):
                return 2
            if outcome == "FULLY_REPRODUCED" or fidelity == "full":
                return 4
            if fidelity == "artifact":
                return 3
            if outcome == "TREND_SUPPORTED" or fidelity == "trend":
                return 2
            if outcome == "EXECUTABLE" or fidelity == "smoke":
                return 1
            return 0

        best = max(rank(record) for record in records)
        return [record for record in records if rank(record) == best]

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
