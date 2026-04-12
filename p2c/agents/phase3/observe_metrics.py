from __future__ import annotations

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.result_extraction import (
    extract_metric_records_from_stdout,
    is_static_inspection_command,
)
from p2c.schemas import MetricContract, MetricRecord, MetricsDoc, RunManifestDoc

SYSTEM_PROMPT = "You parse metrics from run manifest; do not fabricate and return strict JSON only."
USER_PROMPT_TEMPLATE = "Input: execution/codex_outputs/run_manifest.json. Output: results/metrics.json"

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

        manifest_payload = self.artifacts.read_json("execution/codex_outputs/run_manifest.json")
        manifest = RunManifestDoc(**manifest_payload)
        plan_payload = self.artifacts.read_json("execution/execution_plan.json")
        planned_steps = self._planned_step_map(plan_payload)

        contract_payload = self.artifacts.read_json("task/metric_contract.json")
        contract = MetricContract(**contract_payload) if contract_payload else MetricContract()
        required = {str(x).lower() for x in contract.required_metrics if str(x).strip()}

        records: list[MetricRecord] = []
        seen: set[tuple[str, float | None, str]] = set()

        def append_record(metric_name: str, value, source: str, reason_codes: list[str] | None = None) -> None:
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
                    parsed=parsed_value is not None,
                    reason_codes=reason_codes or ([] if parsed_value is not None else ["VALUE_PARSE_FAILED"]),
                )
            )

        for run in manifest.runs:
            planned_step = planned_steps.get(run.run_id, {})
            planned_command = str(planned_step.get("command") or "")
            if (
                is_static_inspection_command(run.command)
                or self._is_metricless_planned_step(planned_step)
                or (
                    planned_command
                    and is_static_inspection_command(planned_command)
                    and "COMMAND_DRIFT" in set(run.reason_codes or [])
                )
            ):
                continue

            stdout_log = self.artifacts.path(f"execution/codex_outputs/step_{run.run_id}_stdout.log")
            stdout_text = ""
            if stdout_log.exists():
                stdout_text = stdout_log.read_text(encoding="utf-8", errors="ignore")
            elif run.stdout_tail:
                stdout_text = run.stdout_tail
            stdout_records = []
            if stdout_text:
                stdout_records = extract_metric_records_from_stdout(
                    stdout_text,
                    contract=contract,
                    source=f"execution/codex_outputs/step_{run.run_id}_stdout.log",
                    command=run.command,
                )
            for record in stdout_records:
                append_record(
                    metric_name=record["metric_name"],
                    value=record["value"],
                    source=record["source"],
                    reason_codes=record.get("reason_codes", []),
                )

            observed_stdout_names = {str(r.get("metric_name") or "").lower() for r in stdout_records}
            for name, raw in run.metrics.items():
                if not self._allow_manifest_metric(str(name), observed_stdout_names):
                    continue
                append_record(
                    metric_name=str(name),
                    value=raw,
                    source=f"execution/codex_outputs/run_manifest.json:{run.run_id}",
                )

        if not records:
            records.append(
                MetricRecord(
                    metric_name="unknown",
                    value=None,
                    unit=None,
                    source="execution/codex_outputs/run_manifest.json",
                    parsed=False,
                    reason_codes=["NO_METRIC_MATCH"],
                )
            )

        metrics = MetricsDoc(records=records, reason_codes=[])
        self.artifacts.write_json("results/metrics.json", metrics.model_dump())
        return {"metrics": metrics.model_dump()}

    @staticmethod
    def _planned_step_map(plan_payload: dict) -> dict[str, dict]:
        if not isinstance(plan_payload, dict):
            return {}
        rows = plan_payload.get("execution_steps", [])
        if not isinstance(rows, list):
            return {}
        return {
            str(row.get("step_id")): row
            for row in rows
            if isinstance(row, dict) and row.get("step_id")
        }

    @staticmethod
    def _is_metricless_planned_step(step: dict) -> bool:
        if not step:
            return False
        expected = step.get("expected_metrics") or []
        produced = step.get("produced_artifacts") or []
        if expected or produced:
            return False
        if step.get("is_setup"):
            return True
        return is_static_inspection_command(str(step.get("command") or ""))

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
