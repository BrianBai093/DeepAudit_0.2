from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_log(agent: str, state: str, step: str, message: str) -> str:
    return f"[{utc_now_iso()}] [agent={agent}] [state={state}] [step={step}] {message}"
