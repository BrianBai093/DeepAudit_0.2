from __future__ import annotations

import re

from p2c.agents.base import BaseAgent
from p2c.schemas import MetricRecord, MetricsDoc

SYSTEM_PROMPT = "You parse metrics from logs; do not fabricate and return strict JSON only."
USER_PROMPT_TEMPLATE = "Input: task/metric_contract.json + execution/run.log. Output: results/metrics.json"


class ObserveMetricsAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="observe_metrics", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        contract = self.artifacts.read_json("task/metric_contract.json")
        run_log = self.artifacts.path("execution/run.log").read_text(encoding="utf-8", errors="ignore")

        records: list[MetricRecord] = []
        for parser in contract.get("parsers", []):
            regex = parser.get("regex", "")
            metric_name = parser.get("metric_name", "metric")
            for m in re.finditer(regex, run_log, flags=re.I):
                raw = m.group(1) if m.groups() else m.group(0)
                value = None
                try:
                    value = float(raw)
                    if value > 1.0:
                        value = value / 100.0
                except Exception:  # noqa: BLE001
                    value = None
                records.append(
                    MetricRecord(
                        metric_name=metric_name,
                        value=value,
                        unit="ratio" if value is not None else None,
                        source="execution/run.log",
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
                    source="execution/run.log",
                    parsed=False,
                    reason_codes=["NO_METRIC_MATCH"],
                )
            )

        metrics = MetricsDoc(records=records, reason_codes=[])
        self.artifacts.write_json("results/metrics.json", metrics.model_dump())
        return {"metrics": metrics.model_dump()}
