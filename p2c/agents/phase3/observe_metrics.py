from __future__ import annotations

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.result_extraction import (
    extract_metric_records_from_stdout,
    is_static_inspection_command,
)
from p2c.schemas import MetricContract, MetricRecord, MetricsDoc, RunManifestDoc

SYSTEM_PROMPT = "You parse metrics from run manifest; do not fabricate and return strict JSON only."
USER_PROMPT_TEMPLATE = "Input: execution/executor_outputs/run_manifest.json. Output: results/metrics.json"

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
        if metric_name and metric_name.lower() in _BOUNDED_METRICS and v > 1.0:
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

        manifest_payload = self.artifacts.read_json("execution/executor_outputs/run_manifest.json")
        manifest = RunManifestDoc(**manifest_payload)

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
                    run_id=getattr(run, "run_id", None) if run is not None else None,
                    experiment_id=getattr(run, "experiment_id", None) if run is not None else None,
                    fidelity=getattr(run, "fidelity", None) if run is not None else None,
                    execution_outcome=getattr(run, "execution_outcome", None) if run is not None else None,
                    evidence_source=getattr(run, "evidence_source", None) if run is not None else None,
                    parsed=parsed_value is not None,
                    reason_codes=reason_codes or ([] if parsed_value is not None else ["VALUE_PARSE_FAILED"]),
                )
            )

        for run in manifest.runs:
            if is_static_inspection_command(run.command):
                continue

            stdout_log_ref = run.logs.stdout if getattr(run, "logs", None) else None
            stdout_log = self.artifacts.path(stdout_log_ref) if stdout_log_ref and not str(stdout_log_ref).startswith("/") else None
            if stdout_log_ref and str(stdout_log_ref).startswith("/"):
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
                    source=stdout_log_ref or f"execution/executor_outputs/run_manifest.json:{run.run_id}",
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
                    source=f"execution/executor_outputs/run_manifest.json:{run.run_id}",
                    run=run,
                )

        if not records:
            records.append(
                MetricRecord(
                    metric_name="unknown",
                    value=None,
                    unit=None,
                    source="execution/executor_outputs/run_manifest.json",
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
