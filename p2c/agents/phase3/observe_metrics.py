from __future__ import annotations

from p2c.agents.base import BaseAgent
from p2c.agents.phase3.execution_summary_evidence import load_effective_run_manifest
from p2c.agents.phase2.result_extraction import (
    extract_metric_records_from_stdout,
    is_static_inspection_command,
)
from p2c.schemas import MetricContract, MetricRecord, MetricsDoc, RunManifestDoc

SYSTEM_PROMPT = "You parse metrics from effective execution evidence; do not fabricate and return strict JSON only."
USER_PROMPT_TEMPLATE = "Input: results/effective_run_manifest.json + execution summary evidence. Output: results/metrics.json"

_GENERIC_SCALAR_METRICS = {
    "accuracy",
    "auc",
    "bleu",
    "f1",
    "loss",
    "mae",
    "mse",
    "perplexity",
    "pr_auc",
    "precision",
    "recall",
    "rmse",
    "roc_auc",
    "rouge",
}

_BOUNDED_METRICS = {
    "accuracy",
    "acc",
    "auc",
    "bleu",
    "f1",
    "precision",
    "pr_auc",
    "recall",
    "roc_auc",
    "rouge",
    "true positive rate",
    "false positive rate",
}


def _is_bounded_metric(metric_name: str | None) -> bool:
    lowered = str(metric_name or "").lower()
    if lowered in _BOUNDED_METRICS:
        return True
    tokens = {tok for tok in lowered.replace("-", "_").split("_") if tok}
    return bool(tokens & _BOUNDED_METRICS)


class ObserveMetricsAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="observe_metrics", *args, **kwargs)

    @staticmethod
    def _to_float(value, metric_name: str | None = None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            v = float(value)
        else:
            s = str(value).strip().rstrip("%")
            try:
                v = float(s)
            except ValueError:
                return None
        if _is_bounded_metric(metric_name) and v > 1.0:
            v = v / 100.0
        return v

    @staticmethod
    def _is_relevant_metric(metric_name: str, required: set[str]) -> bool:
        if not required:
            return True
        lowered = metric_name.lower()
        if lowered in required:
            return True
        if lowered in {"accuracy", "roc_auc", "pr_auc", "recommended_threshold", "best_f1_threshold"}:
            return True
        return any(
            lowered.endswith(f"_{metric}") or lowered.startswith(f"{metric}_")
            for metric in required
        )

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        manifest_payload = load_effective_run_manifest(self.artifacts)
        manifest = RunManifestDoc(**manifest_payload)
        summary_evidence = self.artifacts.read_json("results/execution_summary_evidence.json")

        contract_payload = self.artifacts.read_json("task/metric_contract.json")
        contract = MetricContract(**contract_payload) if contract_payload else MetricContract()
        required = {str(x).lower() for x in contract.required_metrics if str(x).strip()}

        records: list[MetricRecord] = []
        seen: set[tuple[str, float | None, str]] = set()

        def append_record(
            metric_name: str,
            value,
            source: str,
            reason_codes: list[str] | None = None,
            *,
            run=None,
            run_id: str | None = None,
            experiment_id: str | None = None,
            fidelity=None,
            execution_outcome=None,
            evidence_source=None,
        ) -> None:
            lowered = str(metric_name).lower()
            if lowered.endswith("_all"):
                return
            if not self._is_relevant_metric(lowered, required):
                return
            parsed_value = self._to_float(value, lowered)
            key = (lowered, parsed_value, source)
            if key in seen:
                return
            seen.add(key)
            records.append(
                MetricRecord(
                    metric_name=lowered,
                    value=parsed_value,
                    unit="ratio" if parsed_value is not None else None,
                    source=source,
                    run_id=getattr(run, "run_id", None) if run is not None else run_id,
                    experiment_id=getattr(run, "experiment_id", None) if run is not None else experiment_id,
                    fidelity=getattr(run, "fidelity", None) if run is not None else fidelity,
                    execution_outcome=getattr(run, "execution_outcome", None) if run is not None else execution_outcome,
                    evidence_source=getattr(run, "evidence_source", None) if run is not None else evidence_source,
                    parsed=parsed_value is not None,
                    reason_codes=reason_codes or ([] if parsed_value is not None else ["VALUE_PARSE_FAILED"]),
                )
            )

        for run in manifest.runs:
            if is_static_inspection_command(run.command):
                continue

            stdout_log_ref = run.logs.stdout if getattr(run, "logs", None) else None
            stdout_log_is_session = bool(stdout_log_ref and str(stdout_log_ref).endswith("/session_stdout.log"))
            stdout_log = None
            if stdout_log_ref and not stdout_log_is_session:
                stdout_log = self.artifacts.path(stdout_log_ref) if not str(stdout_log_ref).startswith("/") else None
                if str(stdout_log_ref).startswith("/"):
                    from pathlib import Path
                    stdout_log = Path(stdout_log_ref)
            stdout_text = ""
            if stdout_log and stdout_log.exists():
                stdout_text = stdout_log.read_text(encoding="utf-8", errors="ignore")
            elif run.stdout_tail:
                stdout_text = run.stdout_tail
            stdout_records = []
            if stdout_text:
                stdout_records = extract_metric_records_from_stdout(
                    stdout_text,
                    contract=contract,
                    source=stdout_log_ref or f"results/effective_run_manifest.json:{run.run_id}",
                    command=run.command,
                )
            for record in stdout_records:
                append_record(
                    metric_name=record["metric_name"],
                    value=record["value"],
                    source=record["source"],
                    reason_codes=record.get("reason_codes", []),
                    run=run,
                )

            observed_stdout_names = {str(r.get("metric_name") or "").lower() for r in stdout_records}
            for name, raw in run.metrics.items():
                if not self._allow_manifest_metric(str(name), observed_stdout_names):
                    continue
                append_record(
                    metric_name=str(name),
                    value=raw,
                    source=f"results/effective_run_manifest.json:{run.run_id}",
                    run=run,
                )

        summary_path = summary_evidence.get("summary_path") if isinstance(summary_evidence, dict) else None
        if summary_path:
            for run in summary_evidence.get("summary_runs", []):
                if not isinstance(run, dict):
                    continue
                exp_id = str(run.get("experiment_id") or run.get("run_id") or "").strip()
                for name, raw in (run.get("metrics") or {}).items():
                    append_record(
                        metric_name=str(name),
                        value=raw,
                        source=f"{summary_path}:{exp_id}",
                        reason_codes=["SUMMARY_PRIORITY_EVIDENCE"],
                        run_id=str(run.get("run_id") or exp_id),
                        experiment_id=exp_id or None,
                        fidelity=run.get("fidelity"),
                        execution_outcome=run.get("execution_outcome"),
                        evidence_source=run.get("evidence_source"),
                    )

        if not records:
            records.append(
                MetricRecord(
                    metric_name="unknown",
                    value=None,
                    unit=None,
                    source="results/effective_run_manifest.json",
                    parsed=False,
                    reason_codes=["NO_METRIC_MATCH"],
                )
            )

        metrics = MetricsDoc(records=records, reason_codes=[])
        self.artifacts.write_json("results/metrics.json", metrics.model_dump())
        return {"metrics": metrics.model_dump()}

    @staticmethod
    def _allow_manifest_metric(metric_name: str, observed_stdout_names: set[str]) -> bool:
        lowered = metric_name.lower()
        if lowered.endswith("_all"):
            return False
        if lowered in observed_stdout_names:
            return False
        if observed_stdout_names and lowered in _GENERIC_SCALAR_METRICS:
            return False
        return True
