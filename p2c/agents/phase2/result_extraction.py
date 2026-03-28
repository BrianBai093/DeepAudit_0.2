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
    for m in re.finditer(r"METRIC:([\w.][\w.]*)=([\d.eE+-]+)", stdout):
        name, val = m.group(1), m.group(2)
        if name not in metrics:
            try:
                metrics[name] = float(val)
            except ValueError:
                metrics[name] = val

    # Layer 3 — common patterns
    _COMMON = re.compile(
        r"(?:test|val|eval|valid|validation)[_ ]?"
        r"(accuracy|acc|loss|f1|auc|bleu|rouge|precision|recall|mse|mae|rmse|perplexity|ppl)"
        r"\s*[:=]\s*([\d.]+)",
        re.IGNORECASE,
    )
    for m in _COMMON.finditer(stdout):
        name, val = m.group(1).lower(), m.group(2)
        if name not in metrics:
            try:
                metrics[name] = float(val)
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
    """Build a ``ClaimAlignmentDoc`` mapping claims to discovered metrics."""
    items: list[ClaimAlignmentItem] = []
    for claim in claims_ir.claims:
        required = [claim.metric] if claim.metric else []
        has_metric = bool(claim.metric and claim.metric in collected_metrics)
        items.append(
            ClaimAlignmentItem(
                claim_id=claim.claim_id,
                required_metrics=required,
                source=["codex_local_execution"],
                evaluable="yes" if has_metric else "no",
            )
        )
    return ClaimAlignmentDoc(claims=items)
