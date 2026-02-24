from __future__ import annotations

from p2c.agents.base import BaseAgent
from p2c.schemas import MetricRecord, MetricsDoc, RunManifestDoc

SYSTEM_PROMPT = "You parse metrics from run manifest; do not fabricate and return strict JSON only."
USER_PROMPT_TEMPLATE = "Input: execution/codex_outputs/run_manifest.json. Output: results/metrics.json"


class ObserveMetricsAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="observe_metrics", *args, **kwargs)

    @staticmethod
    def _to_float(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            v = float(value)
        else:
            s = str(value).strip().rstrip("%")
            try:
                v = float(s)
            except ValueError:
                return None
        if v > 1.0:
            v = v / 100.0
        return v

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        manifest_payload = self.artifacts.read_json("execution/codex_outputs/run_manifest.json")
        manifest = RunManifestDoc(**manifest_payload)

        contract = self.artifacts.read_json("task/metric_contract.json")
        required = {str(x).lower() for x in contract.get("required_metrics", []) if str(x).strip()}

        records: list[MetricRecord] = []
        for run in manifest.runs:
            for name, raw in run.metrics.items():
                metric_name = str(name).lower()
                if required and metric_name not in required:
                    continue
                value = self._to_float(raw)
                records.append(
                    MetricRecord(
                        metric_name=metric_name,
                        value=value,
                        unit="ratio" if value is not None else None,
                        source=f"execution/codex_outputs/run_manifest.json:{run.run_id}",
                        parsed=value is not None,
                        reason_codes=[] if value is not None else ["VALUE_PARSE_FAILED"],
                    )
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
