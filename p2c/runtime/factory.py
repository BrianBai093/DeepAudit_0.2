from __future__ import annotations

import os
from typing import Any

from p2c.io_artifacts import ArtifactManager
from p2c.runtime.base import ExecutionRuntime
from p2c.runtime.e2b_runtime import E2BRuntime
from p2c.runtime.local_runtime import LocalRuntime

CTX_KEY = "_p2c_runtime"


def _make_runtime() -> ExecutionRuntime:
    backend = (os.getenv("P2C_RUNTIME_BACKEND") or "e2b").strip().lower()
    if backend == "local":
        return LocalRuntime()
    if backend == "e2b":
        timeout = int(os.getenv("P2C_SANDBOX_TIMEOUT_SEC", "3600"))
        return E2BRuntime(timeout_sec=timeout)
    raise RuntimeError(f"Unsupported runtime backend: {backend}")


def ensure_runtime(ctx: dict[str, Any], artifacts: ArtifactManager) -> ExecutionRuntime:
    rt = ctx.get(CTX_KEY)
    if rt is None:
        rt = _make_runtime()
        rt.ensure_started()
        ctx[CTX_KEY] = rt
        artifacts.append_text("execution/run.log", f"[runtime] backend={rt.metadata()}\n")
    return rt


def close_runtime(ctx: dict[str, Any], artifacts: ArtifactManager) -> None:
    rt = ctx.get(CTX_KEY)
    if rt is None:
        return
    try:
        rt.close()
        artifacts.append_text("execution/run.log", "[runtime] closed\n")
    finally:
        ctx.pop(CTX_KEY, None)
