"""Phase2Orchestrator — environment setup + autonomous executor."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.code_compat_agent import CodeCompatAgent
from p2c.agents.phase2.env_repair_agent import EnvRepairAgent
from p2c.agents.phase2.executor_agent import ExecutorAgent
from p2c.agents.phase2.tool_agent import ToolAgent
from p2c.schemas import (
    CodeCompatResult,
    CondaDependency,
    EnvRepairResult,
    EnvSetupResult,
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
        env_repair_agent: EnvRepairAgent | None = None,
        code_compat_agent: CodeCompatAgent | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="phase2_orchestrator", **kwargs)
        self.tool_agent = tool_agent
        self.env_repair_agent = env_repair_agent
        self.code_compat_agent = code_compat_agent
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
            force_env_repair = bool(ctx.get("phase2_force_env_repair"))
            self._persist_state(state, started)

            while state.attempt < state.max_attempts:
                state.attempt += 1
                state.status = "repairing" if force_env_repair or state.attempt > 1 else "env_setup"
                self._persist_state(state, started)
                remaining = budget_sec - state.elapsed_sec
                if remaining <= 60:
                    self.log("PROGRESS", "budget nearly exhausted, stopping phase 2")
                    break

                if state.attempt > 1:
                    self.tool_agent.cleanup()
                    if self.env_repair_agent is not None:
                        self.env_repair_agent.cleanup()

                if force_env_repair:
                    if not self._run_repair_branch(ctx, state, started):
                        break
                else:
                    env_result_dict = self.tool_agent.run(ctx)
                    env_result = self._coerce_env_setup_result(env_result_dict.get("env_result"))
                    state.env_result = env_result
                    self._persist_state(state, started)

                    env_failure = self._env_setup_failure(env_result, state.attempt)
                    if env_failure is not None:
                        state.failures.append(env_failure)
                        if self._should_run_repair_branch(env_result, env_spec):
                            ctx["_p2_env_failure"] = env_result
                            state.status = "repairing"
                            self._persist_state(state, started)
                            if not self._run_repair_branch(ctx, state, started):
                                break
                        else:
                            state.status = "failed"
                            codes = ",".join(env_failure.reason_codes) or "unknown"
                            self.log("PROGRESS", f"environment setup failed ({codes}); skipping executor")
                            self._persist_state(state, started)
                            break
                    else:
                        ctx["_p2_env_mgr"] = self.tool_agent.env_manager

                state.status = "executing"
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
                if not self._patch_env(failure, ctx.get("_p2_env_mgr")):
                    break

            if state.status != "success":
                state.status = "failed"
                self._persist_state(state, started)
                self._write_failure_state(state)
        finally:
            self._persist_state(state, started)
            if not os.getenv("P2C_KEEP_CONDA_ENV"):
                self.tool_agent.cleanup()
                if self.env_repair_agent is not None:
                    self.env_repair_agent.cleanup()

        return state.model_dump()

    def _run_repair_branch(self, ctx: dict[str, Any], state: Phase2State, started: float) -> bool:
        if self.env_repair_agent is None or self.code_compat_agent is None:
            failure = ExecutionFailure(
                attempt=state.attempt,
                stage="env_setup",
                overall_error="environment repair branch is not configured",
                is_dependency_issue=True,
                reason_codes=["ENV_REPAIR_AGENT_MISSING"],
            )
            state.failures.append(failure)
            state.status = "failed"
            self._persist_state(state, started)
            return False

        repair_result_dict = self.env_repair_agent.run(ctx)
        repair_result = self._coerce_env_repair_result(repair_result_dict.get("env_repair_result"))
        state.env_repair_result = repair_result
        self._persist_state(state, started)
        if not isinstance(repair_result, EnvRepairResult) or repair_result.status != "success":
            failure = ExecutionFailure(
                attempt=state.attempt,
                stage="env_setup",
                overall_error="environment repair failed",
                is_dependency_issue=True,
                reason_codes=(repair_result.reason_codes if isinstance(repair_result, EnvRepairResult) else ["ENV_REPAIR_FAILED"]),
            )
            state.failures.append(failure)
            state.status = "failed"
            self._persist_state(state, started)
            return False

        env_mgr = self.env_repair_agent.env_manager or repair_result_dict.get("env_manager")
        if env_mgr is None:
            failure = ExecutionFailure(
                attempt=state.attempt,
                stage="env_setup",
                overall_error="environment repair succeeded without an environment manager",
                is_dependency_issue=True,
                reason_codes=["ENV_REPAIR_MANAGER_MISSING"],
            )
            state.failures.append(failure)
            state.status = "failed"
            self._persist_state(state, started)
            return False

        ctx["_p2_env_mgr"] = env_mgr
        ctx["_p2_env_repair_result"] = repair_result
        if ctx.get("phase2_force_env_repair"):
            state.code_compat_result = CodeCompatResult(
                status="skipped",
                validation_passed=False,
                notes="Skipped because --phase2_force_env_repair trusts the repaired environment and proceeds directly to executor.",
                reason_codes=["CODE_COMPAT_SKIPPED_FORCE_ENV_REPAIR"],
            )
            self._persist_state(state, started)
            return True

        compat_result_dict = self.code_compat_agent.run(ctx)
        compat_result = self._coerce_code_compat_result(compat_result_dict.get("code_compat_result"))
        state.code_compat_result = compat_result
        self._persist_state(state, started)
        if not isinstance(compat_result, CodeCompatResult) or compat_result.status != "success":
            failure = ExecutionFailure(
                attempt=state.attempt,
                stage="execution",
                overall_error="code compatibility validation failed",
                is_dependency_issue=False,
                reason_codes=(compat_result.reason_codes if isinstance(compat_result, CodeCompatResult) else ["CODE_COMPAT_FAILED"]),
            )
            state.failures.append(failure)
            state.status = "failed"
            self._persist_state(state, started)
            return False
        return True

    @staticmethod
    def _env_setup_failure(env_result: Any, attempt: int) -> ExecutionFailure | None:
        env_result = Phase2Orchestrator._coerce_env_setup_result(env_result)
        if not isinstance(env_result, EnvSetupResult):
            return ExecutionFailure(
                attempt=attempt,
                stage="env_setup",
                overall_error="tool agent did not return an environment setup result",
                is_dependency_issue=True,
                reason_codes=["ENV_SETUP_RESULT_MISSING"],
            )

        reason_codes = list(env_result.reason_codes)
        hard_failure_codes = {
            "ENV_CREATE_FAILED",
            "NATIVE_CONDA_ENV_CREATE_FAILED",
            "NATIVE_CONDA_ENV_CREATE_TIMEOUT",
        }
        matched_codes = [code for code in reason_codes if code in hard_failure_codes]
        if not matched_codes:
            return None

        attempted = "; ".join(env_result.install_commands) or "environment creation command"
        return ExecutionFailure(
            attempt=attempt,
            stage="env_setup",
            overall_error=f"{attempted} failed; executor was not launched",
            is_dependency_issue=True,
            reason_codes=matched_codes,
        )

    def _patch_env(self, failure: ExecutionFailure, env_mgr: Any) -> bool:
        if env_mgr is None:
            return False
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
    def _should_run_repair_branch(env_result: EnvSetupResult | None, env_spec: Any) -> bool:
        if not isinstance(env_result, EnvSetupResult):
            return False
        if not getattr(env_spec, "native_environment_file", None):
            return False
        return bool(
            {"NATIVE_CONDA_ENV_CREATE_FAILED", "NATIVE_CONDA_ENV_CREATE_TIMEOUT"}.intersection(env_result.reason_codes)
        )

    @staticmethod
    def _coerce_env_setup_result(raw: Any) -> EnvSetupResult | None:
        if isinstance(raw, EnvSetupResult):
            return raw
        if isinstance(raw, dict):
            return EnvSetupResult(**raw)
        return None

    @staticmethod
    def _coerce_env_repair_result(raw: Any) -> EnvRepairResult | None:
        if isinstance(raw, EnvRepairResult):
            return raw
        if isinstance(raw, dict):
            return EnvRepairResult(**raw)
        return None

    @staticmethod
    def _coerce_code_compat_result(raw: Any) -> CodeCompatResult | None:
        if isinstance(raw, CodeCompatResult):
            return raw
        if isinstance(raw, dict):
            return CodeCompatResult(**raw)
        return None

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
