"""Phase2Orchestrator — environment setup + autonomous executor."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.executor_agent import ExecutorAgent
from p2c.agents.phase2.tool_agent import ToolAgent
from p2c.schemas import (
    CondaDependency,
    ExecutionFailure,
    Phase2State,
    RunManifestDoc,
)


class Phase2Orchestrator(BaseAgent):
    """Phase 2 controller without planning/replanning."""

    def __init__(
        self,
        *,
        tool_agent: ToolAgent,
        executor_agent: ExecutorAgent,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="phase2_orchestrator", **kwargs)
        self.tool_agent = tool_agent
        self.executor_agent = executor_agent

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        max_attempts = max(1, int(os.getenv("P2C_MAX_ENV_PATCH", "2")))
        budget_sec = int(ctx.get("budget_minutes", 30)) * 60
        state = Phase2State(max_attempts=max_attempts, total_budget_sec=budget_sec)
        started = time.time()

        try:
            env_spec = self.tool_agent.build_env_spec(ctx)
            state.env_spec = env_spec
            ctx["_p2_env_spec"] = env_spec
            self._persist_state(state, started)

            while state.attempt < state.max_attempts:
                state.attempt += 1
                state.status = "env_setup" if state.attempt == 1 else "repairing"
                self._persist_state(state, started)
                remaining = budget_sec - state.elapsed_sec
                if remaining <= 60:
                    self.log("PROGRESS", "budget nearly exhausted, stopping phase 2")
                    break

                if state.attempt > 1:
                    self.tool_agent.cleanup()

                env_result_dict = self.tool_agent.run(ctx)
                env_result = env_result_dict.get("env_result")
                state.env_result = env_result
                self._persist_state(state, started)

                state.status = "executing"
                ctx["_p2_env_mgr"] = self.tool_agent.env_manager
                ctx["_p2_remaining_sec"] = max(120, remaining - (time.time() - started - state.elapsed_sec))
                ctx["_p2_attempt"] = state.attempt
                self._persist_state(state, started)

                exec_result = self.executor_agent.run(ctx)
                if exec_result.get("success"):
                    state.status = "success"
                    state.final_manifest = exec_result.get("run_manifest")
                    self._persist_state(state, started)
                    self._write_success_state(state)
                    break

                failure = exec_result.get("failure")
                if isinstance(failure, dict):
                    failure = ExecutionFailure(**failure)
                if not isinstance(failure, ExecutionFailure):
                    failure = ExecutionFailure(
                        attempt=state.attempt,
                        stage="execution",
                        overall_error="unknown executor failure",
                    )
                state.failures.append(failure)
                self._persist_state(state, started)

                if state.attempt >= state.max_attempts:
                    break
                if not self._patch_env(failure, self.tool_agent.env_manager):
                    break

            if state.status != "success":
                state.status = "failed"
                self._persist_state(state, started)
                self._write_failure_state(state)
        finally:
            self._persist_state(state, started)
            if not os.getenv("P2C_KEEP_CONDA_ENV"):
                self.tool_agent.cleanup()

        return state.model_dump()

    def _patch_env(self, failure: ExecutionFailure, env_mgr: Any) -> bool:
        patched_any = False
        for sf in failure.step_failures:
            code = sf.failure_code or ""
            if code == "DEP_MISSING_PACKAGE":
                pkg = self._extract_missing_package(sf.stderr_tail or sf.error_message)
                if pkg:
                    self.log("PROGRESS", f"env-patch: pip install {pkg}")
                    result = env_mgr.install_pip_packages([pkg])
                    patched_any = patched_any or (result.returncode == 0)
            elif code == "DEP_VERSION_CONFLICT":
                pkg = self._extract_missing_package(sf.stderr_tail or sf.error_message)
                if pkg:
                    base_pkg = pkg.split("==")[0].split(">=")[0].split("<=")[0]
                    self.log("PROGRESS", f"env-patch: pip install {base_pkg}")
                    result = env_mgr.install_pip_packages([base_pkg])
                    patched_any = patched_any or (result.returncode == 0)
            elif code in ("DEP_CUDA_MISMATCH", "CFG_WRONG_DEVICE"):
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                patched_any = True
            elif code == "DEP_BUILD_FAILURE":
                pkg = self._extract_missing_package(sf.stderr_tail or sf.error_message)
                if pkg:
                    env_mgr.install_conda_packages([
                        CondaDependency(package=pkg.split("==")[0], channel="conda-forge", pip_fallback=True),
                    ])
                    patched_any = True
        return patched_any

    @staticmethod
    def _extract_missing_package(text: str) -> str | None:
        match = re.search(r"No module named ['\"]?([\w.]+)", text)
        if match:
            return match.group(1).split(".")[0]
        match = re.search(r"Failed building wheel for ([\w-]+)", text)
        return match.group(1) if match else None

    def _write_success_state(self, state: Phase2State) -> None:
        if state.final_manifest:
            payload = state.final_manifest.model_dump() if hasattr(state.final_manifest, "model_dump") else state.final_manifest
            self.artifacts.write_json("execution/executor_outputs/run_manifest.json", payload)

    def _write_failure_state(self, state: Phase2State) -> None:
        self.artifacts.write_json(
            "execution/executor_outputs/run_manifest.json",
            RunManifestDoc(reason_codes=["PHASE2_FAILED"]).model_dump(),
        )
        self.artifacts.write_json("execution/execution_failures.json", [f.model_dump() for f in state.failures])
        self.log("DONE", f"phase 2 failed after {state.attempt} attempts")

    def _persist_state(self, state: Phase2State, started: float) -> None:
        state.elapsed_sec = time.time() - started
        self.artifacts.write_json("execution/phase2_state.json", state.model_dump())
