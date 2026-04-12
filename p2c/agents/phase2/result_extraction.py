"""Shared utilities for metric extraction and Phase 3-compatible output building."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from p2c.schemas import (
    ClaimAlignmentDoc,
    ClaimAlignmentItem,
    ClaimsIR,
    CodexRun,
    MetricContract,
    RunManifestDoc,
)


# ---------------------------------------------------------------------------
# Metric extraction (multi-record strategy)
# ---------------------------------------------------------------------------


_METRIC_LABEL_ALIASES = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "auc": "auc",
    "bleu": "bleu",
    "f1": "f1",
    "f1_score": "f1",
    "f1score": "f1",
    "f1-score": "f1",
    "loss": "loss",
    "mae": "mae",
    "mse": "mse",
    "perplexity": "perplexity",
    "ppl": "perplexity",
    "pr_auc": "pr_auc",
    "pr-auc": "pr_auc",
    "prauc": "pr_auc",
    "precision": "precision",
    "recall": "recall",
    "rmse": "rmse",
    "roc_auc": "roc_auc",
    "roc-auc": "roc_auc",
    "rocauc": "roc_auc",
    "rouge": "rouge",
}

_BOUNDED_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "auc", "bleu", "rouge"}


def _normalize_metric_name(name: str) -> str:
    normalized = re.sub(r"[\s\-]+", "_", str(name or "").strip().lower()).strip("_")
    return _METRIC_LABEL_ALIASES.get(normalized, normalized)


def _coerce_metric_value(raw: Any, metric_name: str | None = None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(str(raw).strip().rstrip("%"))
    except ValueError:
        return None
    if metric_name in _BOUNDED_METRICS and value > 1.0:
        value = value / 100.0
    return value


def _dedupe_metric_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, float | None, str]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = (
            str(record.get("metric_name") or ""),
            record.get("value"),
            str(record.get("source") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def is_static_inspection_command(command: str | None) -> bool:
    """Return True when a command only prints source/config text and should not emit metrics."""
    raw = str(command or "").strip()
    if not raw:
        return False
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    if not tokens:
        return False

    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        name, _, value = tokens[0].partition("=")
        if name.isidentifier() and value:
            tokens = tokens[1:]
            continue
        break
    if not tokens:
        return False

    head = tokens[0]
    if head in {"sed", "cat", "head", "tail"}:
        return True

    if head == "python" and len(tokens) >= 3 and tokens[1] == "-c":
        snippet = " ".join(tokens[2:]).lower()
        if "read_text(" in snippet or ".read_text(" in snippet:
            return True
        if "open(" in snippet and ".read(" in snippet:
            return True

    return False


def extract_metric_records_from_stdout(
    stdout: str,
    contract: MetricContract,
    *,
    source: str = "stdout",
    command: str | None = None,
) -> list[dict[str, Any]]:
    """Extract multiple metric records from a single stdout blob."""
    if is_static_inspection_command(command):
        return []

    records: list[dict[str, Any]] = []

    def add_record(metric_name: str, raw_value: Any, reason_code: str) -> None:
        normalized_name = _normalize_metric_name(metric_name)
        value = _coerce_metric_value(raw_value, normalized_name)
        if value is None:
            return
        records.append(
            {
                "metric_name": normalized_name,
                "value": value,
                "source": source,
                "parsed": True,
                "reason_codes": [reason_code],
            }
        )

    # Layer 1 — contract regex parsers. Use finditer instead of search so we keep all matches.
    for parser in contract.parsers:
        try:
            matches = list(re.finditer(parser.regex, stdout, re.IGNORECASE | re.MULTILINE))
        except re.error:
            continue
        for match in matches:
            raw = match.group(1) if match.lastindex else match.group(0)
            if parser.transform == "float":
                add_record(parser.metric_name, raw, "CONTRACT_PARSER")
            else:
                add_record(parser.metric_name, raw, "CONTRACT_PARSER_RAW")

    # Layer 2 — explicit METRIC:name=value lines.
    for match in re.finditer(r"METRIC:([\w.][\w.]*)=([\d.eE+-]+)", stdout):
        add_record(match.group(1), match.group(2), "EXPLICIT_METRIC_LINE")

    # Layer 3 — prefixed train/val/test metrics.
    prefixed_pattern = re.compile(
        r"(test|val|eval|valid|validation|train|training)[_ ]?"
        r"(accuracy|acc|loss|f1|auc|bleu|rouge|precision|recall|mse|mae|rmse|perplexity|ppl)"
        r"\s*[:=]\s*([\d.eE+-]+)",
        re.IGNORECASE,
    )
    for match in prefixed_pattern.finditer(stdout):
        prefix = match.group(1).lower()
        metric_name = _normalize_metric_name(match.group(2))
        if prefix in ("val", "valid", "validation", "eval"):
            prefix = "val"
        elif prefix in ("train", "training"):
            prefix = "train"
        add_record(f"{prefix}_{metric_name}", match.group(3), "PREFIXED_METRIC")
        if prefix in ("val", "test"):
            add_record(metric_name, match.group(3), "PREFIXED_METRIC_PRIMARY")

    # Layer 4 — common labeled metrics (works for `Precision: 0.03` and inline summary rows).
    labeled_pattern = re.compile(
        r"(?i)(roc[-_ ]auc|pr[-_ ]auc|precision|recall|f1(?:-score)?|accuracy|loss|auc|bleu|rouge|mse|mae|rmse|perplexity|ppl)"
        r"\s*[:=]\s*([\d.eE+-]+)"
    )
    for match in labeled_pattern.finditer(stdout):
        line_start = stdout.rfind("\n", 0, match.start()) + 1
        line_prefix = stdout[line_start:match.start()].lower()
        if re.search(r"(?:^|[^a-z0-9_])(train|training|val|valid|validation|eval|test)[_ ]*$", line_prefix):
            continue
        add_record(match.group(1), match.group(2), "LABELED_METRIC")

    # Layer 5 — dictionary-style metric summaries (`{'precision': 0.1, 'f1': 0.2}`).
    dict_pattern = re.compile(
        r"['\"](accuracy|precision|recall|f1|roc_auc|pr_auc|auc)['\"]\s*:\s*([\d.eE+-]+)",
        re.IGNORECASE,
    )
    for match in dict_pattern.finditer(stdout):
        add_record(match.group(1), match.group(2), "DICT_METRIC")

    # Layer 6 — sklearn classification report rows.
    report_row_pattern = re.compile(
        r"(?im)^\s*(0|1|macro avg|weighted avg|avg / total)"
        r"\s+(0?\.\d+|1(?:\.0+)?)"
        r"\s+(0?\.\d+|1(?:\.0+)?)"
        r"\s+(0?\.\d+|1(?:\.0+)?)"
        r"\s+\d+\s*$"
    )
    row_prefixes = {
        "0": "class_0",
        "1": "class_1",
        "macro avg": "macro",
        "weighted avg": "weighted",
        "avg / total": "avg_total",
    }
    for match in report_row_pattern.finditer(stdout):
        row_name = row_prefixes.get(match.group(1).lower(), _normalize_metric_name(match.group(1)))
        add_record(f"{row_name}_precision", match.group(2), "CLASSIFICATION_REPORT")
        add_record(f"{row_name}_recall", match.group(3), "CLASSIFICATION_REPORT")
        add_record(f"{row_name}_f1", match.group(4), "CLASSIFICATION_REPORT")

    # Layer 7 — threshold sweep table rows.
    threshold_row_pattern = re.compile(
        r"(?im)^\s*(0?\.\d+|1(?:\.0+)?)"
        r"\s+(0?\.\d+|1(?:\.0+)?)"
        r"\s+(0?\.\d+|1(?:\.0+)?)"
        r"\s+(0?\.\d+|1(?:\.0+)?)"
        r"\s+\d+\s+\d+\s+\d+\s+\d+\s*$"
    )
    for match in threshold_row_pattern.finditer(stdout):
        threshold = str(match.group(1)).replace(".", "_")
        add_record(f"threshold_{threshold}_precision", match.group(2), "THRESHOLD_ROW")
        add_record(f"threshold_{threshold}_recall", match.group(3), "THRESHOLD_ROW")
        add_record(f"threshold_{threshold}_f1", match.group(4), "THRESHOLD_ROW")

    recommended_pattern = re.search(
        r"Recommended threshold.*?:\s*([\d.]+).*?"
        r"Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+),\s*F1:\s*([\d.]+)",
        stdout,
        re.IGNORECASE | re.DOTALL,
    )
    if recommended_pattern:
        add_record("recommended_threshold", recommended_pattern.group(1), "RECOMMENDED_THRESHOLD")
        add_record("recommended_precision", recommended_pattern.group(2), "RECOMMENDED_THRESHOLD")
        add_record("recommended_recall", recommended_pattern.group(3), "RECOMMENDED_THRESHOLD")
        add_record("recommended_f1", recommended_pattern.group(4), "RECOMMENDED_THRESHOLD")

    best_f1_block = re.search(r"BEST_F1_ROW:\s*(.*?)(?:\n\s*\n|$)", stdout, re.IGNORECASE | re.DOTALL)
    if best_f1_block:
        for line in best_f1_block.group(1).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            raw_key = "_".join(parts[:-1])
            raw_value = parts[-1]
            add_record(f"best_f1_{raw_key}", raw_value, "BEST_F1_ROW")

    return _dedupe_metric_records(records)


def extract_metrics_from_stdout(
    stdout: str,
    contract: MetricContract,
    *,
    command: str | None = None,
) -> dict[str, Any]:
    """Extract metrics from stdout and keep canonical last-seen values plus `_all` lists."""
    metrics: dict[str, Any] = {}
    values_by_metric: dict[str, list[float]] = {}

    for record in extract_metric_records_from_stdout(stdout, contract, command=command):
        metric_name = str(record.get("metric_name") or "")
        value = record.get("value")
        if not metric_name or value is None:
            continue
        values_by_metric.setdefault(metric_name, []).append(value)

    for metric_name, values in values_by_metric.items():
        metrics[metric_name] = values[-1]
        if len(values) > 1:
            metrics[f"{metric_name}_all"] = values

    return metrics


def extract_metrics_from_file(file_path: str | Path) -> dict[str, Any]:
    """Read a JSON file written by codex and return the metrics dict inside it."""
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("metrics", data)
        return {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(r"(?:No module named|ModuleNotFoundError|ImportError)\s+['\"]?([\w.]+)")
_OOM_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "OOM", "RuntimeError: out of memory")

# New taxonomy import (used by classify_error_v2; legacy wrapper kept for compat)
from p2c.failure_taxonomy import (  # noqa: E402
    FailureSpec,
    classify_failure as _classify_failure_v2,
    failure_to_legacy,
)


def classify_error(stdout: str, stderr: str, exit_code: int) -> str:
    """Return one of the canonical error_type literals (backward-compatible).

    Internally delegates to the v2 taxonomy and maps back to the legacy
    7-literal system.
    """
    spec = classify_error_v2(stdout, stderr, exit_code)
    return failure_to_legacy(spec)


def classify_error_v2(
    stdout: str,
    stderr: str,
    exit_code: int,
    *,
    metrics: dict | None = None,
    expected_metrics: list[str] | None = None,
) -> FailureSpec:
    """Return a full FailureSpec from the v2 taxonomy.

    Use this for rich failure information (repair strategy, confidence, etc.).
    """
    return _classify_failure_v2(
        stdout, stderr, exit_code,
        metrics=metrics,
        expected_metrics=expected_metrics,
    )


def extract_traceback(stderr: str) -> str | None:
    """Pull the last Python traceback from stderr."""
    idx = stderr.rfind("Traceback (most recent call last)")
    if idx == -1:
        return None
    return stderr[idx:][:3000]


def is_fast_fail(stdout: str, stderr: str) -> bool:
    """Return True if the failure is unrecoverable (OOM, segfault, disk full)."""
    spec = classify_error_v2(stdout, stderr, exit_code=1)
    return spec.is_fast_fail


# ---------------------------------------------------------------------------
# Phase 3-compatible output builders
# ---------------------------------------------------------------------------


def build_run_manifest(
    runs: list[dict[str, Any]],
    reason_codes: list[str] | None = None,
) -> RunManifestDoc:
    """Build a ``RunManifestDoc`` that Phase 3 can consume."""
    codex_runs = []
    for r in runs:
        codex_runs.append(
            CodexRun(
                run_id=r.get("step_id", f"run_{len(codex_runs)}"),
                command=r.get("command", ""),
                params=r.get("params", {}),
                cwd=r.get("cwd", "."),
                exit_code=int(r.get("exit_code", 1)),
                status=str(r.get("status") or ("ok" if int(r.get("exit_code", 1)) == 0 else "failed")),
                runtime_sec=r.get("runtime_sec"),
                stdout_tail=(r.get("stdout_tail") or "")[-2000:],
                stderr_tail=(r.get("stderr_tail") or "")[-2000:],
                artifacts=r.get("artifacts", []),
                metrics=r.get("metrics", {}),
                reason_codes=r.get("reason_codes", []),
            )
        )
    return RunManifestDoc(runs=codex_runs, reason_codes=reason_codes or [])


def build_claim_alignment(
    claims_ir: ClaimsIR,
    collected_metrics: dict[str, Any],
    metric_sources: dict[str, list[str]] | None = None,
) -> ClaimAlignmentDoc:
    """Build a ``ClaimAlignmentDoc`` mapping claims to discovered metrics.

    Phase 2 is intentionally conservative here: it reports *what* metrics
    were collected and marks claims as ``"partial"`` when the metric name
    matches but we cannot be certain the collected value corresponds to the
    specific experiment described by the claim (e.g. same metric from
    different tables / datasets).  Full alignment is deferred to Phase 3,
    which has access to richer claim context (table_anchor, scope, etc.).
    """
    items: list[ClaimAlignmentItem] = []
    metric_sources = metric_sources or {}

    # Count how many result-type claims share each metric name
    from collections import Counter
    metric_claim_count: Counter[str] = Counter()
    for claim in claims_ir.claims:
        if claim.type == "result" and claim.metric:
            metric_claim_count[claim.metric] += 1

    for claim in claims_ir.claims:
        required = [claim.metric] if claim.metric else []
        matching_metric_names = [
            name for name in collected_metrics
            if claim.metric
            and (
                name == claim.metric
                or name.endswith(f"_{claim.metric}")
                or name.startswith(f"{claim.metric}_")
            )
        ]
        has_metric = bool(matching_metric_names)
        claim_sources = []
        for metric_name in matching_metric_names:
            for source in metric_sources.get(metric_name, []):
                if source not in claim_sources:
                    claim_sources.append(source)

        if not has_metric:
            evaluable = "no"
            reason = None
        elif claim.metric and metric_claim_count.get(claim.metric, 0) > 1:
            # Multiple claims reference the same metric name (e.g. 3 different
            # "accuracy" values from different tables).  We have *a* value but
            # cannot determine which claim it corresponds to — mark partial and
            # let Phase 3 resolve with experiment context.
            evaluable = "partial"
            reason = (
                f"metric '{claim.metric}' collected but {metric_claim_count[claim.metric]} "
                f"claims reference it; alignment deferred to Phase 3"
            )
        else:
            evaluable = "yes"
            reason = None

        items.append(
            ClaimAlignmentItem(
                claim_id=claim.claim_id,
                required_metrics=required,
                source=claim_sources or ["codex_local_execution"],
                evaluable=evaluable,
                reason=reason,
            )
        )
    return ClaimAlignmentDoc(claims=items)
