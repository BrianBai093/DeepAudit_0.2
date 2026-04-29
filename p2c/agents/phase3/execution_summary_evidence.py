from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from p2c.agents.base import BaseAgent

SUMMARY_PRIORITY_NOTE = (
    "EXECUTION_COMPLETE.md, EXECUTION_SUMMARY_FINAL.md, executor_results.json, and run_manifest.json "
    "are same-origin Phase 2 execution evidence. Phase 3 should use EXECUTION_COMPLETE.md first "
    "when it exists, EXECUTION_SUMMARY_FINAL.md second, executor_results.json third, and raw "
    "run_manifest.json only as lower-priority audit context."
)

_VALID_STATUS = {"ok", "partial", "failed", "skipped"}
_VALID_FIDELITY = {"artifact", "smoke", "trend", "full"}
_VALID_EVIDENCE_SOURCE = {"fresh_run", "checkpoint_eval", "existing_logs", "existing_results", "mixed"}
_VALID_STOP_REASON = {
    "checkpoint_eval",
    "existing_artifact",
    "budget_bound",
    "early_stop_evidence",
    "full_run_complete",
    "repo_missing_path",
    "runtime_failure",
    "guardrail_blocked",
    "skipped_nonessential",
}


def load_effective_run_manifest(artifacts) -> dict[str, Any]:
    """Return Phase 3's preferred manifest, falling back to the raw Phase 2 manifest."""
    effective = artifacts.read_json("results/effective_run_manifest.json")
    if isinstance(effective, dict) and effective.get("runs"):
        return effective
    return artifacts.read_json("execution/executor_outputs/run_manifest.json")


class ExecutionSummaryEvidenceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="execution_summary_evidence", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        raw_manifest = self.artifacts.read_json("execution/executor_outputs/run_manifest.json")
        executor_results = self.artifacts.read_json("execution/executor_outputs/executor_results.json")
        summary_path, summary_text = self._read_summary_text()
        summary_runs = self._parse_summary_runs(summary_text)

        effective, conflicts = self._build_effective_manifest(
            raw_manifest=raw_manifest,
            executor_results=executor_results,
            summary_runs=summary_runs,
            summary_path=summary_path,
        )

        evidence = {
            "same_origin_note": SUMMARY_PRIORITY_NOTE,
            "priority_order": [
                "execution/executor_outputs/EXECUTION_COMPLETE.md",
                "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md",
                "execution/executor_outputs/executor_results.json",
                "execution/executor_outputs/run_manifest.json",
            ],
            "summary_path": summary_path,
            "summary_text": summary_text[:20000],
            "summary_runs": summary_runs,
            "executor_results_path": "execution/executor_outputs/executor_results.json",
            "raw_manifest_path": "execution/executor_outputs/run_manifest.json",
            "effective_manifest_path": "results/effective_run_manifest.json",
            "conflicts": conflicts,
            "reason_codes": ["SUMMARY_PRIORITY_EVIDENCE_LAYER_BUILT"],
        }

        self.artifacts.write_json("results/execution_summary_evidence.json", evidence)
        self.artifacts.write_json("results/effective_run_manifest.json", effective)
        return {"execution_summary_evidence": evidence, "effective_run_manifest": effective}

    def _read_summary_text(self) -> tuple[str | None, str]:
        for rel in (
            "execution/executor_outputs/EXECUTION_COMPLETE.md",
            "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md",
            "execution/executor_outputs/EXECUTION_SUMMARY.md",
        ):
            path = self.artifacts.path(rel)
            if path.exists() and path.stat().st_size > 0:
                return rel, path.read_text(encoding="utf-8", errors="ignore")
        return None, ""

    @classmethod
    def _build_effective_manifest(
        cls,
        *,
        raw_manifest: dict[str, Any],
        executor_results: dict[str, Any],
        summary_runs: list[dict[str, Any]],
        summary_path: str | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        conflicts: list[dict[str, Any]] = []
        run_by_id: dict[str, dict[str, Any]] = {}
        ordered_ids: list[str] = []

        def put(run_id: str, run: dict[str, Any]) -> None:
            if run_id not in run_by_id:
                ordered_ids.append(run_id)
            run_by_id[run_id] = run

        for raw_run in raw_manifest.get("runs", []) if isinstance(raw_manifest, dict) else []:
            if not isinstance(raw_run, dict):
                continue
            run = cls._normalize_run(raw_run, source_reason="RAW_RUN_MANIFEST_BASE")
            put(cls._run_key(run), run)

        for raw_run in executor_results.get("runs", []) if isinstance(executor_results, dict) else []:
            if not isinstance(raw_run, dict):
                continue
            run = cls._normalize_run(raw_run, source_reason="EFFECTIVE_FROM_EXECUTOR_RESULTS")
            key = cls._run_key(run)
            existing = run_by_id.get(key)
            if existing:
                merged = cls._merge_run(
                    existing,
                    run,
                    source_reason="EFFECTIVE_FROM_EXECUTOR_RESULTS",
                    conflict_reason="CONFLICT_WITH_ORIGINAL_RUN_MANIFEST",
                    conflicts=conflicts,
                    source_label="executor_results.json",
                )
                put(key, merged)
            else:
                put(key, run)

        for summary_run in summary_runs:
            run = cls._normalize_run(
                summary_run,
                source_reason="SUMMARY_FINAL_PRIORITY",
                summary_path=summary_path,
            )
            key = cls._run_key(run)
            existing = run_by_id.get(key)
            if existing:
                merged = cls._merge_run(
                    existing,
                    run,
                    source_reason="SUMMARY_FINAL_PRIORITY",
                    conflict_reason="CONFLICT_WITH_LOWER_PRIORITY_EXECUTION_EVIDENCE",
                    conflicts=conflicts,
                    source_label=summary_path or "execution_summary",
                    sparse_override=True,
                )
                put(key, merged)
            else:
                put(key, run)

        reason_codes = list(raw_manifest.get("reason_codes", [])) if isinstance(raw_manifest, dict) else []
        for code in ("EFFECTIVE_MANIFEST_BUILT", "SUMMARY_HIGHEST_PRIORITY"):
            if code not in reason_codes:
                reason_codes.append(code)
        if conflicts and "EFFECTIVE_MANIFEST_CONFLICTS_RECORDED" not in reason_codes:
            reason_codes.append("EFFECTIVE_MANIFEST_CONFLICTS_RECORDED")

        return {"runs": [run_by_id[key] for key in ordered_ids], "reason_codes": reason_codes}, conflicts

    @classmethod
    def _merge_run(
        cls,
        base: dict[str, Any],
        incoming: dict[str, Any],
        *,
        source_reason: str,
        conflict_reason: str,
        conflicts: list[dict[str, Any]],
        source_label: str,
        sparse_override: bool = False,
    ) -> dict[str, Any]:
        merged = deepcopy(base)
        conflict_count_before = len(conflicts)
        run_id = str(merged.get("run_id") or incoming.get("run_id") or incoming.get("experiment_id") or "")

        scalar_fields = [
            "experiment_id",
            "experiment_name",
            "dataset",
            "command",
            "cwd",
            "exit_code",
            "status",
            "fidelity",
            "execution_outcome",
            "evidence_source",
            "stop_reason",
            "runtime_sec",
            "stdout_tail",
            "stderr_tail",
        ]
        for field in scalar_fields:
            new_value = incoming.get(field)
            if sparse_override and cls._is_blank(new_value):
                continue
            if new_value != merged.get(field):
                if field in {"status", "fidelity", "execution_outcome", "evidence_source", "stop_reason"}:
                    conflicts.append(
                        {
                            "run_id": run_id,
                            "field": field,
                            "lower_priority_value": merged.get(field),
                            "higher_priority_value": new_value,
                            "higher_priority_source": source_label,
                            "reason_code": conflict_reason,
                        }
                    )
                merged[field] = new_value

        for field in ("commands_attempted", "override_args", "observed_signals", "artifacts"):
            values = list(merged.get(field, []) if isinstance(merged.get(field), list) else [])
            for item in incoming.get(field, []) if isinstance(incoming.get(field), list) else []:
                if item not in values:
                    values.append(item)
            merged[field] = values

        if isinstance(incoming.get("params"), dict):
            params = dict(merged.get("params", {}) if isinstance(merged.get("params"), dict) else {})
            params.update({k: v for k, v in incoming["params"].items() if not cls._is_blank(v)})
            merged["params"] = params

        if isinstance(incoming.get("metrics"), dict):
            metrics = dict(merged.get("metrics", {}) if isinstance(merged.get("metrics"), dict) else {})
            for name, value in incoming["metrics"].items():
                if name in metrics and metrics[name] != value:
                    conflicts.append(
                        {
                            "run_id": run_id,
                            "field": f"metrics.{name}",
                            "lower_priority_value": metrics[name],
                            "higher_priority_value": value,
                            "higher_priority_source": source_label,
                            "reason_code": conflict_reason,
                        }
                    )
                metrics[name] = value
            merged["metrics"] = metrics

        if isinstance(incoming.get("logs"), dict):
            logs = dict(merged.get("logs", {}) if isinstance(merged.get("logs"), dict) else {})
            for name, value in incoming["logs"].items():
                if not cls._is_blank(value):
                    logs[name] = value
            merged["logs"] = logs

        notes = [str(x).strip() for x in (merged.get("notes"), incoming.get("notes")) if str(x or "").strip()]
        if notes:
            merged["notes"] = " ".join(dict.fromkeys(notes))

        reason_codes = [
            code
            for code in merged.get("reason_codes", [])
            if code not in {"COMMAND_NOT_OBSERVED", "UNTRACEABLE_METRICS"} or incoming.get("status") not in {"ok", "partial"}
        ]
        for code in incoming.get("reason_codes", []) if isinstance(incoming.get("reason_codes"), list) else []:
            if code not in reason_codes:
                reason_codes.append(code)
        local_conflict = len(conflicts) > conflict_count_before
        for code in (source_reason, conflict_reason if local_conflict else None):
            if code and code not in reason_codes:
                reason_codes.append(code)
        merged["reason_codes"] = reason_codes
        return cls._normalize_run(merged, source_reason=None)

    @classmethod
    def _normalize_run(
        cls,
        raw_run: dict[str, Any],
        *,
        source_reason: str | None,
        summary_path: str | None = None,
    ) -> dict[str, Any]:
        exp_id = str(raw_run.get("experiment_id") or raw_run.get("run_id") or "").strip()
        run_id = str(raw_run.get("run_id") or exp_id or "run").strip()
        status = cls._normalize_status(raw_run.get("status"))
        fidelity = cls._normalize_fidelity(raw_run.get("fidelity"), raw_run.get("notes"))
        evidence_source = cls._normalize_evidence_source(raw_run.get("evidence_source"))
        stop_reason = cls._normalize_stop_reason(raw_run.get("stop_reason") or raw_run.get("reason"))
        execution_outcome = cls._normalize_execution_outcome(raw_run.get("execution_outcome"), fidelity, status)

        exit_code_raw = raw_run.get("exit_code")
        try:
            exit_code = int(exit_code_raw)
        except (TypeError, ValueError):
            exit_code = 0 if status in {"ok", "partial", "skipped"} else 1

        metrics = raw_run.get("metrics") if isinstance(raw_run.get("metrics"), dict) else {}
        logs = raw_run.get("logs") if isinstance(raw_run.get("logs"), dict) else {}
        if summary_path:
            logs = {**logs, "narrative": summary_path}

        reason_codes = list(raw_run.get("reason_codes", []) if isinstance(raw_run.get("reason_codes"), list) else [])
        if source_reason and source_reason not in reason_codes:
            reason_codes.append(source_reason)

        return {
            "run_id": run_id,
            "experiment_id": exp_id or run_id,
            "experiment_name": raw_run.get("experiment_name") or raw_run.get("name") or run_id,
            "dataset": raw_run.get("dataset"),
            "command": str(raw_run.get("command") or ""),
            "commands_attempted": list(raw_run.get("commands_attempted", []) if isinstance(raw_run.get("commands_attempted"), list) else []),
            "params": dict(raw_run.get("params", {}) if isinstance(raw_run.get("params"), dict) else {}),
            "cwd": str(raw_run.get("cwd") or "."),
            "exit_code": exit_code,
            "status": status,
            "fidelity": fidelity,
            "execution_outcome": execution_outcome,
            "evidence_source": evidence_source,
            "override_args": list(raw_run.get("override_args", []) if isinstance(raw_run.get("override_args"), list) else []),
            "observed_signals": list(raw_run.get("observed_signals", []) if isinstance(raw_run.get("observed_signals"), list) else []),
            "stop_reason": stop_reason,
            "notes": raw_run.get("notes"),
            "runtime_sec": cls._to_float(raw_run.get("runtime_sec")),
            "stdout_tail": raw_run.get("stdout_tail"),
            "stderr_tail": raw_run.get("stderr_tail"),
            "artifacts": list(raw_run.get("artifacts", []) if isinstance(raw_run.get("artifacts"), list) else []),
            "metrics": metrics,
            "logs": logs,
            "reason_codes": reason_codes,
        }

    @classmethod
    def _parse_summary_runs(cls, summary_text: str) -> list[dict[str, Any]]:
        if not summary_text.strip():
            return []

        complete_runs = cls._parse_execution_complete_table(summary_text)
        if complete_runs:
            return complete_runs

        matches = list(re.finditer(r"^#{2,6}\s+((?:Exp|exp)_\d+|table_\d+)[:\s-]*(.*)$", summary_text, flags=re.MULTILINE))
        runs: list[dict[str, Any]] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(summary_text)
            chunk = summary_text[start:end]
            exp_id = match.group(1).lower()
            name = match.group(2).strip(" :-") or exp_id

            status = cls._extract_markdown_field(chunk, "Status")
            fidelity = cls._extract_markdown_field(chunk, "Fidelity")
            reason = cls._extract_markdown_field(chunk, "Reason") or cls._extract_markdown_field(chunk, "Stop Reason")

            metrics = cls._extract_summary_metrics(chunk)
            artifacts = re.findall(r"`([^`]*(?:res_|results?)[^`]*)`", chunk)
            notes = " ".join(line.strip(" -") for line in chunk.splitlines() if line.strip())[:2000]

            runs.append(
                {
                    "run_id": exp_id,
                    "experiment_id": exp_id,
                    "experiment_name": name,
                    "status": cls._normalize_status(status),
                    "fidelity": cls._normalize_fidelity(fidelity, chunk),
                    "evidence_source": "mixed" if status else None,
                    "stop_reason": cls._normalize_stop_reason(reason),
                    "metrics": metrics,
                    "artifacts": artifacts,
                    "notes": notes,
                    "reason_codes": ["SUMMARY_FINAL_PRIORITY"],
                }
            )
        return runs

    @classmethod
    def _parse_execution_complete_table(cls, summary_text: str) -> list[dict[str, Any]]:
        rows = cls._parse_markdown_table(summary_text, required_headers={"experiment", "config", "fidelity", "accuracy"})
        if not rows:
            return []

        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in rows:
            experiment = row.get("experiment", "")
            config = row.get("config", "")
            fidelity = cls._normalize_fidelity(row.get("fidelity"), config)
            status_cell = row.get("status", "")
            status = "failed" if "\\u274c" in status_cell.encode("unicode_escape").decode("ascii") or "fail" in status_cell.lower() else "ok"
            if status != "ok":
                continue

            canonical_id = cls._canonical_experiment_id(experiment=experiment, config=config)
            if not canonical_id:
                continue
            if canonical_id not in grouped:
                grouped[canonical_id] = {
                    "run_id": canonical_id,
                    "experiment_id": canonical_id,
                    "experiment_name": cls._canonical_experiment_name(canonical_id, config),
                    "status": "ok",
                    "fidelity": fidelity,
                    "evidence_source": "fresh_run",
                    "stop_reason": "full_run_complete" if fidelity == "full" else "budget_bound",
                    "metrics": {},
                    "artifacts": [],
                    "notes": [],
                    "reason_codes": ["SUMMARY_FINAL_PRIORITY", "EXECUTION_COMPLETE_TABLE_PARSED"],
                }
                order.append(canonical_id)

            run = grouped[canonical_id]
            run["fidelity"] = cls._higher_fidelity(run.get("fidelity"), fidelity)
            if run["fidelity"] == "full":
                run["stop_reason"] = "full_run_complete"
            metric_name = cls._scoped_accuracy_metric_name(config=config, fidelity=fidelity)
            accuracy = cls._extract_percent_ratio(row.get("accuracy", ""))
            if metric_name and accuracy is not None:
                run["metrics"][metric_name] = accuracy
            note = f"{experiment} {config} {fidelity or ''} accuracy={row.get('accuracy', '').strip()}".strip()
            if note:
                run["notes"].append(note)

        out = []
        for run_id in order:
            run = grouped[run_id]
            run["notes"] = " ".join(run["notes"])[:2000]
            out.append(run)
        return out

    @staticmethod
    def _parse_markdown_table(text: str, *, required_headers: set[str]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        lines = [line.strip() for line in text.splitlines()]
        for idx, line in enumerate(lines):
            if not line.startswith("|") or idx + 1 >= len(lines):
                continue
            headers = [cell.strip().lower() for cell in line.strip("|").split("|")]
            if not required_headers <= set(headers):
                continue
            separator = lines[idx + 1]
            if not separator.startswith("|") or not set(separator.replace("|", "").strip()) <= {"-", ":"}:
                continue
            for data_line in lines[idx + 2:]:
                if not data_line.startswith("|"):
                    break
                cells = [cell.strip() for cell in data_line.strip("|").split("|")]
                if len(cells) != len(headers):
                    continue
                rows.append(dict(zip(headers, cells, strict=False)))
            if rows:
                break
        return rows

    @staticmethod
    def _canonical_experiment_id(*, experiment: str, config: str) -> str | None:
        text = f"{experiment} {config}".lower()
        if "conv" in text or "convolutional" in text:
            return "exp_02"
        if "fc" in text or "fully connected" in text:
            return "exp_01"
        return None

    @staticmethod
    def _canonical_experiment_name(experiment_id: str, config: str) -> str:
        if experiment_id == "exp_02":
            return "Table 1 convolutional benchmark"
        if experiment_id == "exp_01":
            return "Table 1 fully connected benchmark"
        return config or experiment_id

    @staticmethod
    def _scoped_accuracy_metric_name(*, config: str, fidelity: str | None) -> str | None:
        text = config.lower().replace("-", "")
        dataset = None
        for name in ("mnist", "cifar10", "cifar100"):
            if name in text:
                dataset = name
                break
        algorithm = None
        for name in ("pepita", "erin", "drtp", "dfa", "fa", "bp"):
            if name in text:
                algorithm = "pepita" if name == "erin" else name
                break
        architecture = "conv" if "conv" in text or "convolutional" in text else "fc" if "fc" in text else None
        if not dataset or not algorithm:
            return None
        pieces = [algorithm, dataset]
        if architecture:
            pieces.append(architecture)
        if fidelity:
            pieces.append(fidelity)
        pieces.extend(["test", "accuracy"])
        return "_".join(pieces)

    @staticmethod
    def _extract_percent_ratio(text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if match:
            return float(match.group(1)) / 100.0
        try:
            value = float(str(text).strip())
        except ValueError:
            return None
        return value / 100.0 if value > 1.0 else value

    @staticmethod
    def _higher_fidelity(left: str | None, right: str | None) -> str | None:
        order = {None: 0, "smoke": 1, "trend": 2, "artifact": 2, "full": 3}
        return right if order.get(right, 0) > order.get(left, 0) else left

    @staticmethod
    def _extract_markdown_field(text: str, label: str) -> str | None:
        pattern = rf"\*\*{re.escape(label)}:\*\*\s*([^|\n]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        pattern = rf"{re.escape(label)}:\s*([^|\n]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_summary_metrics(text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        counter: dict[str, int] = {}
        metric_names = ("accuracy", "acc", "f1", "auc", "precision", "recall")
        for line in text.splitlines():
            lowered = line.lower()
            matched_metric = next((name for name in metric_names if name in lowered), None)
            if not matched_metric:
                continue
            for raw_value in re.findall(r"(\d+(?:\.\d+)?)\s*%", line):
                value = float(raw_value) / 100.0
                base = "accuracy" if matched_metric == "acc" else matched_metric
                counter[base] = counter.get(base, 0) + 1
                name = base if counter[base] == 1 else f"{base}_{counter[base]}"
                metrics[name] = value
        return metrics

    @staticmethod
    def _normalize_status(value: Any) -> str:
        raw = str(value or "").strip().lower()
        raw = re.sub(r"[^a-z_]+", "_", raw).strip("_")
        aliases = {
            "success": "ok",
            "succeeded": "ok",
            "complete": "ok",
            "completed": "ok",
            "pass": "ok",
            "passed": "ok",
            "partially_successful": "partial",
            "degraded_success": "partial",
            "not_run": "skipped",
            "skip": "skipped",
        }
        return aliases.get(raw, raw if raw in _VALID_STATUS else "failed")

    @staticmethod
    def _normalize_fidelity(value: Any, context: Any = None) -> str | None:
        value_text = str(value or "").strip().lower()
        value_normalized = re.sub(r"[^a-z0-9]+", "_", value_text).strip("_")
        if value_normalized in _VALID_FIDELITY:
            return value_normalized
        if value_normalized in {"smoke_trend", "mixed_smoke_trend", "mixed_smoke_trend_"}:
            return "trend"
        if value_normalized in {"smoke_artifact", "artifact_evaluation", "mixed_artifact"}:
            return "artifact"

        raw = f"{value_text} {context or ''}".lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if "full" in normalized:
            return "full"
        if "artifact" in normalized or "existing" in normalized:
            return "artifact"
        if "trend" in normalized or "smoke_trend" in normalized or "mixed_smoke_trend" in normalized:
            return "trend"
        if "smoke" in normalized:
            return "smoke"
        return None

    @staticmethod
    def _normalize_evidence_source(value: Any) -> str | None:
        raw = str(value or "").strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        aliases = {
            "fresh_runs": "fresh_run",
            "existing_artifact": "existing_results",
            "existing_artifacts": "existing_results",
            "artifact": "existing_results",
            "artifact_evaluation": "existing_results",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in _VALID_EVIDENCE_SOURCE else None

    @staticmethod
    def _normalize_stop_reason(value: Any) -> str | None:
        raw = str(value or "").strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if normalized in _VALID_STOP_REASON:
            return normalized
        if "budget" in normalized:
            return "budget_bound"
        if "artifact" in normalized:
            return "existing_artifact"
        if "skip" in normalized or "config" in normalized or "method" in normalized:
            return "skipped_nonessential"
        if "guard" in normalized or "implementation" in normalized:
            return "guardrail_blocked"
        if "complete" in normalized:
            return "full_run_complete"
        if "runtime" in normalized or "fail" in normalized:
            return "runtime_failure"
        return None

    @staticmethod
    def _normalize_execution_outcome(value: Any, fidelity: str | None, status: str) -> str | None:
        raw = str(value or "").strip()
        if raw in {"EXECUTABLE", "TREND_SUPPORTED", "FULLY_REPRODUCED"}:
            return raw
        if status not in {"ok", "partial"}:
            return None
        if fidelity == "full":
            return "FULLY_REPRODUCED"
        if fidelity in {"trend", "artifact"}:
            return "TREND_SUPPORTED"
        if fidelity == "smoke":
            return "EXECUTABLE"
        return None

    @staticmethod
    def _run_key(run: dict[str, Any]) -> str:
        return str(run.get("experiment_id") or run.get("run_id") or "").strip()

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_blank(value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}
