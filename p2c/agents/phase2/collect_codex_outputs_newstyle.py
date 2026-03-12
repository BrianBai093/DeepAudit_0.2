from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import ExecutionSummaryDoc

SYSTEM_PROMPT = "You collect and validate new-style Codex output artifacts."
USER_PROMPT_TEMPLATE = "Input: workspace outputs directory. Output: execution/codex_outputs/*"


class CollectCodexOutputsNewstyleAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="collect_codex_outputs_newstyle", *args, **kwargs)

    @staticmethod
    def _extract_last_json(text: str) -> dict[str, Any] | None:
        raw = str(text or "")
        for idx in range(len(raw) - 1, -1, -1):
            if raw[idx] != "{":
                continue
            try:
                obj = json.loads(raw[idx:])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        runtime = ensure_runtime(ctx, self.artifacts)
        if (getattr(runtime, "backend_name", "") or "").lower() != "e2b":
            raise RuntimeError("collect_codex_outputs_newstyle requires P2C_RUNTIME_BACKEND=e2b")

        outputs_dir = str(ctx.get("workspace_outputs_dir") or "")
        if not outputs_dir:
            raise RuntimeError("collect_codex_outputs_newstyle missing workspace_outputs_dir")

        required_files = {
            f"{outputs_dir}/codex_exec.log": "execution/codex_outputs/codex_exec.log",
            f"{outputs_dir}/codex_main.log": "execution/codex_outputs/codex_main.log",
        }
        optional_files = {
            f"{outputs_dir}/execution_summary.json": "execution/codex_outputs/execution_summary.json",
            f"{outputs_dir}/patches.diff": "execution/codex_outputs/patches.diff",
            f"{outputs_dir}/codex_exec.stream.log": "execution/codex_outputs/codex_exec.stream.log",
            f"{outputs_dir}/dependency_bootstrap.log": "execution/codex_outputs/dependency_bootstrap.log",
            f"{outputs_dir}/toolchain_probe.json": "execution/codex_outputs/toolchain_probe.json",
            f"{outputs_dir}/codex_failure.json": "execution/codex_outputs/codex_failure.json",
        }

        for remote, rel in required_files.items():
            runtime.download_file(remote, self.artifacts.path(rel))
        for remote, rel in optional_files.items():
            try:
                runtime.download_file(remote, self.artifacts.path(rel))
            except Exception:  # noqa: BLE001
                if rel.endswith(".log") or rel.endswith(".diff"):
                    self.artifacts.write_text(rel, "")

        summary_path = self.artifacts.path("execution/codex_outputs/execution_summary.json")
        payload: dict[str, Any] | None = None
        if summary_path.exists() and summary_path.stat().st_size > 0:
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                payload = None
        if not isinstance(payload, dict):
            main_log = self.artifacts.path("execution/codex_outputs/codex_main.log").read_text(
                encoding="utf-8", errors="ignore"
            )
            payload = self._extract_last_json(main_log)
            if not isinstance(payload, dict):
                raise RuntimeError("new-style collect could not recover a valid execution_summary.json")
            self.artifacts.write_json("execution/codex_outputs/execution_summary.json", payload)

        summary_doc = ExecutionSummaryDoc(**payload)
        self.artifacts.write_json("execution/codex_outputs/execution_summary.json", summary_doc.model_dump())

        return {
            "codex_outputs": {
                "execution_succeeded": summary_doc.execution_succeeded,
                "success_basis": summary_doc.success_basis,
                "attempt_count": summary_doc.attempt_count,
                "task_results": len(summary_doc.task_results),
                "remaining_blockers": list(summary_doc.remaining_blockers),
            }
        }
