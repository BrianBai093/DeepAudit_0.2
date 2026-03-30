"""Shared utilities for metric extraction and Phase 3-compatible output building."""

from __future__ import annotations

import json
import re
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
# Metric extraction (three-layer strategy)
# ---------------------------------------------------------------------------


def extract_metrics_from_stdout(stdout: str, contract: MetricContract) -> dict[str, Any]:
    """Extract metrics from stdout using three layers of patterns.

    Layer 1: ``MetricContract.parsers`` regex patterns (highest priority).
    Layer 2: ``METRIC:{name}={value}`` lines (our structured prompt format).
    Layer 3: Common ``test accuracy: 0.95`` / ``val_loss = 0.23`` heuristics.
    Layer 4: Best/max aggregation — when multiple values exist for the same
             metric, keep both ``{metric}`` (best) and ``{metric}_all`` (list).
    """
    metrics: dict[str, Any] = {}

    # Layer 1 — MetricContract regex parsers
    for parser in contract.parsers:
        try:
            match = re.search(parser.regex, stdout)
        except re.error:
            continue
        if match:
            raw = match.group(1) if match.lastindex else match.group(0)
            if parser.transform == "float":
                try:
                    metrics[parser.metric_name] = float(raw)
                except ValueError:
                    metrics[parser.metric_name] = raw
            else:
                metrics[parser.metric_name] = raw

    # Layer 2 — METRIC:{name}={value}
    # Collect ALL values per metric name (there may be multiple METRIC:accuracy lines)
    _metric_values: dict[str, list[float]] = {}
    for m in re.finditer(r"METRIC:([\w.][\w.]*)=([\d.eE+-]+)", stdout):
        name, val = m.group(1), m.group(2)
        try:
            _metric_values.setdefault(name, []).append(float(val))
        except ValueError:
            pass
    for name, vals in _metric_values.items():
        if name not in metrics:
            # Use the LAST reported value (most likely the final/best)
            metrics[name] = vals[-1]

    # Layer 3 — common prefixed patterns (val_accuracy, test_loss, etc.)
    # These get priority names like "val_accuracy" to distinguish from train
    _PREFIXED = re.compile(
        r"(test|val|eval|valid|validation|train|training)[_ ]?"
        r"(accuracy|acc|loss|f1|auc|bleu|rouge|precision|recall|mse|mae|rmse|perplexity|ppl)"
        r"\s*[:=]\s*([\d.]+)",
        re.IGNORECASE,
    )
    for m in _PREFIXED.finditer(stdout):
        prefix = m.group(1).lower()
        metric_name = m.group(2).lower()
        val = m.group(2 + 1)
        # Normalize prefix
        if prefix in ("val", "valid", "validation", "eval"):
            prefix = "val"
        elif prefix in ("train", "training"):
            prefix = "train"
        # else: "test"
        full_name = f"{prefix}_{metric_name}"
        if full_name not in metrics:
            try:
                metrics[full_name] = float(val)
            except ValueError:
                pass
        # Also set the unprefixed name if not already set (prefer val/test over train)
        if metric_name not in metrics and prefix in ("val", "test", "eval"):
            try:
                metrics[metric_name] = float(val)
            except ValueError:
                pass

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
                status="ok" if int(r.get("exit_code", 1)) == 0 else "failed",
                runtime_sec=r.get("runtime_sec"),
                stdout_tail=(r.get("stdout_tail") or "")[-2000:],
                stderr_tail=(r.get("stderr_tail") or "")[-2000:],
                metrics=r.get("metrics", {}),
            )
        )
    return RunManifestDoc(runs=codex_runs, reason_codes=reason_codes or [])


def build_claim_alignment(
    claims_ir: ClaimsIR,
    collected_metrics: dict[str, Any],
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

    # Count how many result-type claims share each metric name
    from collections import Counter
    metric_claim_count: Counter[str] = Counter()
    for claim in claims_ir.claims:
        if claim.type == "result" and claim.metric:
            metric_claim_count[claim.metric] += 1

    for claim in claims_ir.claims:
        required = [claim.metric] if claim.metric else []
        has_metric = bool(claim.metric and claim.metric in collected_metrics)

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
                source=["codex_local_execution"],
                evaluable=evaluable,
                reason=reason,
            )
        )
    return ClaimAlignmentDoc(claims=items)
