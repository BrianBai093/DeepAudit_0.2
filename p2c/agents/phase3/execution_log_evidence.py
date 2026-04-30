"""Extract structured Phase 2 evidence from raw executor output logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent

LOG_EVIDENCE_PATH = "results/execution_log_evidence.json"


class ExecutionLogEvidenceAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="execution_log_evidence", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        outputs_dir = self.artifacts.path("execution/executor_outputs")
        logs = extract_execution_log_evidence(outputs_dir, self.artifacts.run_root)
        doc = {
            "schema_version": "execution_log_evidence.v1",
            "logs": logs,
            "reason_codes": ["EXECUTION_LOG_SCAN_COMPLETE"],
        }
        self.artifacts.write_json(LOG_EVIDENCE_PATH, doc)
        useful = sum(1 for row in logs if row.get("metrics") or row.get("curves") or row.get("skip_reason"))
        self.log("DONE", f"scanned {len(logs)} logs; extracted evidence from {useful}")
        return {"execution_log_evidence": doc}


def extract_execution_log_evidence(outputs_dir: Path, run_root: Path | None = None) -> list[dict[str, Any]]:
    if not outputs_dir.exists():
        return []
    run_root = run_root or outputs_dir.parent.parent
    rows = []
    for path in sorted(outputs_dir.glob("*.log")):
        rel_path = _relative_to_run_root(path, run_root)
        text = path.read_text(encoding="utf-8", errors="ignore")
        meta = _metadata_from_log_path(path)
        metrics = _extract_scalar_metrics(text, source=rel_path, meta=meta)
        curves = _extract_epoch_curves(text, source=rel_path, meta=meta)
        skip_reason = _extract_skip_reason(text)
        error_summary = _extract_error_summary(text)
        reason_codes = _reason_codes_for_log(meta, metrics, curves, skip_reason, error_summary)
        rows.append(
            {
                "path": rel_path,
                "log_kind": meta.get("log_kind"),
                "experiment_id": meta.get("experiment_id"),
                "fidelity": meta.get("fidelity"),
                "dataset": meta.get("dataset"),
                "algorithm": meta.get("algorithm"),
                "model_family": meta.get("model_family"),
                "config_name": meta.get("config_name"),
                "metrics": metrics,
                "curves": curves,
                "skip_reason": skip_reason,
                "error_summary": error_summary,
                "reason_codes": reason_codes,
            }
        )
    return rows


def _metadata_from_log_path(path: Path) -> dict[str, Any]:
    stem = path.stem
    name = path.name
    log_kind = "unknown"
    if stem.endswith("_stdout"):
        log_kind = "stdout"
        stem = stem[: -len("_stdout")]
    elif stem.endswith("_stderr"):
        log_kind = "stderr"
        stem = stem[: -len("_stderr")]
    elif stem.endswith("_narrative"):
        log_kind = "narrative"
        stem = stem[: -len("_narrative")]
    elif name in {"session_stdout.log", "session_stderr.log"}:
        log_kind = "stdout" if "stdout" in name else "stderr"
    elif name == "executor_agent.log":
        log_kind = "narrative"

    meta: dict[str, Any] = {"log_kind": log_kind}
    match = re.search(r"experiment_(exp_\d+)(?:_(.*))?$", stem)
    if match:
        meta["experiment_id"] = match.group(1)
        config = match.group(2) or ""
        meta["config_name"] = config
        tokens = [tok for tok in re.split(r"[_\s-]+", config) if tok]
        lowered = {tok.lower() for tok in tokens}
        for token in lowered:
            if token in {"smoke", "trend", "full"}:
                meta["fidelity"] = token
            elif token in {"mnist", "mn"}:
                meta["dataset"] = "mnist"
            elif token in {"cifar10", "cif10", "cif"}:
                meta["dataset"] = "cifar10"
            elif token in {"cifar100", "cif100"}:
                meta["dataset"] = "cifar100"
            elif token in {"bp", "pepita", "fa", "dfa", "drtp", "rp"}:
                meta["algorithm"] = token
            elif token in {"fc", "fullyconnected"}:
                meta["model_family"] = "fc"
            elif token in {"conv", "cnn"}:
                meta["model_family"] = "conv"
    return meta


def _extract_epoch_curves(text: str, *, source: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    epoch_rows: dict[int, dict[str, float]] = {}
    current_epoch: int | None = None
    for line in text.splitlines():
        epoch_match = re.search(r"\[(\d+),\s*\d+\]\s+loss:\s*([\d.eE+-]+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epoch_rows.setdefault(current_epoch, {})["loss"] = float(epoch_match.group(2))
            continue
        test_match = re.search(r"Test accuracy:\s*([\d.eE+-]+)\s*%?", line, re.IGNORECASE)
        if test_match and current_epoch is not None:
            epoch_rows.setdefault(current_epoch, {})["test_accuracy"] = _percent_to_ratio(float(test_match.group(1)))
            continue
        train_match = re.search(r"Training accuracy\s*=\s*([\d.eE+-]+)", line, re.IGNORECASE)
        if train_match:
            epoch_rows.setdefault(current_epoch or 1, {})["train_accuracy"] = _percent_to_ratio(float(train_match.group(1)))
            continue
        val_match = re.search(r"Validation accuracy\s*=\s*([\d.eE+-]+)", line, re.IGNORECASE)
        if val_match:
            epoch_rows.setdefault(current_epoch or 1, {})["val_accuracy"] = _percent_to_ratio(float(val_match.group(1)))

    curves = []
    for metric_name in ("test_accuracy", "train_accuracy", "val_accuracy", "loss"):
        points = [
            {"x": epoch, "y": values[metric_name]}
            for epoch, values in sorted(epoch_rows.items())
            if metric_name in values
        ]
        if points:
            curves.append({"metric_name": metric_name, "points": points, "source": source, **_scope_meta(meta)})
    return curves


def _extract_scalar_metrics(text: str, *, source: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []

    def add(metric_name: str, raw: str, reason_code: str) -> None:
        try:
            value = _percent_to_ratio(float(str(raw).strip().rstrip("%")))
        except ValueError:
            return
        metrics.append({"metric_name": metric_name, "value": value, "source": source, **_scope_meta(meta), "reason_codes": [reason_code]})

    for match in re.finditer(r"Training accuracy\s*=\s*([\d.eE+-]+)", text, re.IGNORECASE):
        add("train_accuracy", match.group(1), "TRAIN_ACCURACY")
    for match in re.finditer(r"Validation accuracy\s*=\s*([\d.eE+-]+)", text, re.IGNORECASE):
        add("val_accuracy", match.group(1), "VALIDATION_ACCURACY")
    for match in re.finditer(r"Mean train accuracy\s*=\s*\[?([\d.eE+-]+)\]?", text, re.IGNORECASE):
        add("mean_train_accuracy", match.group(1), "MEAN_TRAIN_ACCURACY")
    for match in re.finditer(r"Mean test accuracy\s*=\s*\[?([\d.eE+-]+)\]?", text, re.IGNORECASE):
        add("mean_test_accuracy", match.group(1), "MEAN_TEST_ACCURACY")
    for match in re.finditer(r"Final accuracy\s*=\s*\[?([\d.eE+-]+)\]?", text, re.IGNORECASE):
        add("final_accuracy", match.group(1), "FINAL_ACCURACY")
    test_matches = list(re.finditer(r"Test accuracy:\s*([\d.eE+-]+)\s*%?", text, re.IGNORECASE))
    if test_matches:
        add("final_test_accuracy", test_matches[-1].group(1), "FINAL_TEST_ACCURACY")
    loss_matches = list(re.finditer(r"\[(\d+),\s*\d+\]\s+loss:\s*([\d.eE+-]+)", text))
    if loss_matches:
        raw = loss_matches[-1].group(2)
        metrics.append({"metric_name": "final_loss", "value": float(raw), "source": source, **_scope_meta(meta), "reason_codes": ["FINAL_LOSS"]})
    return _dedupe_metrics(metrics)


def _extract_skip_reason(text: str) -> str | None:
    match = re.search(r"(exp_\d+\s+skipped:\s*.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "skipped:" in text.lower():
        return text.strip().splitlines()[0][:500]
    return None


def _extract_error_summary(text: str) -> str | None:
    if "traceback" in text.lower():
        idx = text.lower().rfind("traceback")
        return text[idx: idx + 800]
    if "error" in text.lower() or "exception" in text.lower():
        for line in text.splitlines():
            if "error" in line.lower() or "exception" in line.lower():
                return line[:500]
    return None


def _reason_codes_for_log(
    meta: dict[str, Any],
    metrics: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    skip_reason: str | None,
    error_summary: str | None,
) -> list[str]:
    codes: list[str] = []
    if metrics:
        codes.append("EXECUTED_METRIC")
    if curves:
        codes.append("EXECUTED_CURVE")
    if meta.get("fidelity") == "smoke":
        codes.append("SMOKE_ONLY")
    if skip_reason:
        codes.append("SKIPPED_REASON")
    if meta.get("experiment_id") == "exp_05":
        codes.append("CONFIG_ONLY")
    if error_summary:
        codes.append("ERROR_LOG")
    if not codes:
        codes.append("PARSE_LOW_CONFIDENCE")
    return codes


def _scope_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": meta.get("experiment_id"),
        "fidelity": meta.get("fidelity"),
        "algorithm": meta.get("algorithm"),
        "dataset": meta.get("dataset"),
        "model_family": meta.get("model_family"),
        "config_name": meta.get("config_name"),
    }


def _dedupe_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for metric in metrics:
        key = (metric.get("metric_name"), metric.get("value"), metric.get("source"))
        if key in seen:
            continue
        seen.add(key)
        out.append(metric)
    return out


def _percent_to_ratio(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _relative_to_run_root(path: Path, run_root: Path) -> str:
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except Exception:  # noqa: BLE001
        return path.as_posix()
