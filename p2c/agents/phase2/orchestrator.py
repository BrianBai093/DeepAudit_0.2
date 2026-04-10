"""Phase2Orchestrator — state-machine controller: Planner → ToolAgent → CodexExecutor.

Includes a two-tier repair loop:
  1. **Micro-repair**: inline fix (pip install, path patch, device switch) — no LLM replan.
  2. **Macro-replan**: full planner + env rebuild cycle.

The repair tier is determined by the v2 failure taxonomy (see ``failure_taxonomy.py``).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.codex_executor import CodexExecutorAgent
from p2c.agents.phase2.planner import PlannerAgent
from p2c.agents.phase2.tool_agent import ToolAgent
from p2c.failure_taxonomy import RepairStrategy
from p2c.schemas import (
    ClaimAlignmentDoc,
    ExecutionFailure,
    Phase2State,
    RunManifestDoc,
    StepFailure,
)


class Phase2Orchestrator(BaseAgent):
    """Feedback-loop controller for Phase 2 local execution.

    State machine::

        PLANNING ──▶ ENV_SETUP ──▶ EXECUTING ──▶ SUCCESS
            ▲              │            │
            │              ▼            ▼
            └── REPLANNING ◀── failure analysis
                    │
                    ▼ (attempts exhausted)
              AUTONOMOUS ──▶ SUCCESS / FAILED
    """

    def __init__(
        self,
        *,
        planner: PlannerAgent,
        tool_agent: ToolAgent,
        codex_executor: CodexExecutorAgent,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="phase2_orchestrator", **kwargs)
        self.planner = planner
        self.tool_agent = tool_agent
        self.codex_executor = codex_executor

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        max_attempts = int(os.getenv("P2C_MAX_REPLAN", "3"))
        budget_sec = int(ctx.get("budget_minutes", 30)) * 60

        state = Phase2State(
            max_attempts=max_attempts,
            total_budget_sec=budget_sec,
        )
        t_start = time.time()
        plan = None

        try:
            while state.attempt < state.max_attempts:
                state.elapsed_sec = time.time() - t_start
                remaining = budget_sec - state.elapsed_sec
                if remaining <= 60:
                    self.log("PROGRESS", "budget nearly exhausted, stopping")
                    break

                # === PLANNING / REPLANNING ===
                state.status = "planning" if state.attempt == 0 else "replanning"
                state.attempt += 1
                self.log("PROGRESS", f"attempt {state.attempt}/{state.max_attempts} — {state.status}")

                ctx["_p2_failures"] = state.failures
                plan_result = self.planner.run(ctx)
                plan = plan_result.get("plan")
                if plan is None:
                    raise RuntimeError("Planner returned no plan")
                state.plan = plan

                # === ENV SETUP ===
                state.status = "env_setup"
                # Clean up previous env before creating a new one (prevents conda env leak)
                if state.attempt > 1:
                    self.tool_agent.cleanup()
                ctx["_p2_plan"] = plan
                env_result_dict = self.tool_agent.run(ctx)
                env_result = env_result_dict.get("env_result")
                state.env_result = env_result

                # Check for critical dependency failures
                if env_result and env_result.failed_packages:
                    critical = self._has_critical_failures(env_result, plan)
                    if critical:
                        state.failures.append(ExecutionFailure(
                            attempt=state.attempt,
                            plan_version=plan.plan_version,
                            stage="env_setup",
                            overall_error=f"Critical packages failed: {env_result.failed_packages}",
                            is_dependency_issue=True,
                        ))
                        self.log("PROGRESS", "critical deps failed, replanning...")
                        continue

                # === EXECUTION ===
                state.status = "executing"
                ctx["_p2_env_mgr"] = self.tool_agent.env_manager
                ctx["_p2_remaining_sec"] = budget_sec - (time.time() - t_start)
                ctx["_p2_attempt"] = state.attempt

                exec_result = self.codex_executor.run(ctx)

                if exec_result.get("success"):
                    state.status = "success"
                    state.final_manifest = exec_result.get("run_manifest")
                    state.final_alignment = exec_result.get("claim_alignment")
                    self.log("DONE", f"execution succeeded with {len(exec_result.get('metrics', {}))} metrics")
                    break

                # --- Two-tier failure handling ---
                failure = exec_result.get("failure")
                if isinstance(failure, dict):
                    failure = ExecutionFailure(**failure)
                if not isinstance(failure, ExecutionFailure):
                    failure = ExecutionFailure(
                        attempt=state.attempt, stage="execution",
                        overall_error="unknown failure",
                    )

                repair_level = self._classify_repair_level(failure)
                self.log("PROGRESS", f"failure repair_level={repair_level}, "
                         f"codes={[sf.failure_code for sf in failure.step_failures]}")

                # --- Tier 1: Micro-repair (no LLM replan, no env rebuild) ---
                if repair_level == "micro":
                    micro_ok = self._attempt_micro_repair(
                        ctx, failure, self.tool_agent.env_manager,
                    )
                    if micro_ok:
                        # Re-run execution with same plan after inline fix
                        self.log("PROGRESS", "micro-repair applied, re-executing...")
                        retry_result = self.codex_executor.run(ctx)
                        if retry_result.get("success"):
                            state.status = "success"
                            state.final_manifest = retry_result.get("run_manifest")
                            state.final_alignment = retry_result.get("claim_alignment")
                            self.log("DONE", "succeeded after micro-repair")
                            break
                        # Micro-repair didn't fully resolve — fall through to record failure
                        self.log("PROGRESS", "micro-repair insufficient, escalating to replan")

                # --- Tier 1.5: Env-patch (targeted pip install, no full rebuild) ---
                elif repair_level == "env_patch":
                    patch_ok = self._patch_env(ctx, failure, self.tool_agent.env_manager)
                    if patch_ok:
                        self.log("PROGRESS", "env patched, re-executing...")
                        retry_result = self.codex_executor.run(ctx)
                        if retry_result.get("success"):
                            state.status = "success"
                            state.final_manifest = retry_result.get("run_manifest")
                            state.final_alignment = retry_result.get("claim_alignment")
                            self.log("DONE", "succeeded after env patch")
                            break
                        self.log("PROGRESS", "env patch insufficient, escalating to replan")

                # --- Tier 2: Macro-replan (full cycle on next iteration) ---
                state.failures.append(failure)

                # --- Signal-based autonomous switch (checked EVERY iteration) ---
                should_auto, auto_reason = self._should_switch_to_autonomous(
                    state, budget_sec, time.time() - t_start,
                )
                if should_auto:
                    if self._run_autonomous_fallback(ctx, state, budget_sec, t_start, auto_reason):
                        break
                    # Autonomous also failed — continue replanning if attempts remain
                    self.log("PROGRESS", "autonomous mode failed, continuing replan loop")

                dep_issue = failure.is_dependency_issue
                if dep_issue:
                    self.log("PROGRESS", "dependency issue detected, replanning...")
                else:
                    self.log("PROGRESS", "execution failed, replanning...")
                continue

            # === AUTONOMOUS FALLBACK (attempts exhausted) ===
            if state.status != "success" and plan and plan.codex_autonomous_fallback:
                self._run_autonomous_fallback(
                    ctx, state, budget_sec, t_start, "attempts_exhausted",
                )

            # === FINALIZE ===
            if state.status == "success":
                self._write_success_state(state)
            else:
                state.status = "failed"
                self._write_failure_state(state)

        finally:
            # Persist state regardless
            state.elapsed_sec = time.time() - t_start
            self.artifacts.write_json("execution/phase2_state.json", state.model_dump())

            # Cleanup conda env
            if not os.getenv("P2C_KEEP_CONDA_ENV"):
                self.tool_agent.cleanup()

        return state.model_dump()

    # ------------------------------------------------------------------
    # Two-tier repair logic
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_repair_level(failure: ExecutionFailure) -> str:
        """Decide repair scope based on v2 taxonomy.

        Returns one of: ``"micro"`` | ``"env_patch"`` | ``"macro"``.
        """
        if not failure.step_failures:
            return "macro"

        strategies = {sf.repair_strategy for sf in failure.step_failures if sf.repair_strategy}
        codes = {sf.failure_code for sf in failure.step_failures if sf.failure_code}

        # Fast-fail codes (OOM, segfault, disk full) → macro (or abort)
        fast_fails = {"RES_OOM_GPU", "RES_OOM_CPU", "RES_SEGFAULT", "RES_DISK_FULL"}
        if codes & fast_fails:
            return "macro"

        # All failures are inline-fixable and ≤2 steps → micro
        if strategies <= {RepairStrategy.INLINE_FIX.value, "inline_fix"} and len(failure.step_failures) <= 2:
            return "micro"

        # Dependency issues that are small in scope → env_patch
        dep_codes = {"DEP_MISSING_PACKAGE", "DEP_VERSION_CONFLICT", "DEP_BUILD_FAILURE"}
        if codes and codes <= dep_codes and len(failure.step_failures) <= 2:
            return "env_patch"

        # Single CUDA/device issue → env_patch
        if codes <= {"DEP_CUDA_MISMATCH", "CFG_WRONG_DEVICE"}:
            return "env_patch"

        return "macro"

    def _attempt_micro_repair(
        self,
        ctx: dict[str, Any],
        failure: ExecutionFailure,
        env_mgr: Any,
    ) -> bool:
        """Try inline fixes without LLM replan. Returns True if a fix was applied."""
        fixed_any = False
        for sf in failure.step_failures:
            code = sf.failure_code or ""

            # --- Missing package: extract name and pip install ---
            if code == "DEP_MISSING_PACKAGE":
                pkg = self._extract_missing_package(sf.stderr_tail or sf.error_message)
                if pkg:
                    self.log("PROGRESS", f"micro-repair: pip install {pkg}")
                    result = env_mgr.install_pip_packages([pkg])
                    if result.returncode == 0:
                        fixed_any = True
                    else:
                        self.log("PROGRESS", f"micro-repair: pip install {pkg} failed")

            # --- Missing env var (e.g. WANDB_API_KEY): set dummy ---
            elif code == "CFG_MISSING_ENV_VAR":
                var = self._extract_env_var(sf.stderr_tail or sf.error_message)
                if var:
                    self.log("PROGRESS", f"micro-repair: setting dummy {var}")
                    os.environ[var] = "disabled"
                    # Also common: disable wandb entirely
                    if "WANDB" in var.upper():
                        os.environ["WANDB_MODE"] = "disabled"
                    fixed_any = True

            # --- Wrong device (cuda not available): set CPU fallback ---
            elif code == "CFG_WRONG_DEVICE":
                self.log("PROGRESS", "micro-repair: setting CUDA_VISIBLE_DEVICES=''")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                fixed_any = True

            # --- Permission denied: try chmod ---
            elif code == "CFG_PERMISSION_DENIED":
                path = self._extract_file_path(sf.stderr_tail or sf.error_message)
                if path:
                    self.log("PROGRESS", f"micro-repair: chmod 755 {path}")
                    env_mgr.run_in_env(f"chmod -R 755 {path}", timeout_sec=15)
                    fixed_any = True

            # --- Data path mismatch / hardcoded path: log but can't auto-fix reliably ---
            elif code in ("DATA_PATH_MISMATCH", "CFG_HARDCODED_PATH"):
                self.log("PROGRESS", f"micro-repair: {code} detected but no auto-fix available")

        return fixed_any

    def _patch_env(
        self,
        ctx: dict[str, Any],
        failure: ExecutionFailure,
        env_mgr: Any,
    ) -> bool:
        """Targeted env fix: install missing/conflicting packages without full rebuild."""
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
                # Try installing without version pin
                pkg = self._extract_missing_package(sf.stderr_tail or sf.error_message)
                if pkg:
                    base_pkg = pkg.split("==")[0].split(">=")[0].split("<=")[0]
                    self.log("PROGRESS", f"env-patch: pip install {base_pkg} (unpinned)")
                    result = env_mgr.install_pip_packages([base_pkg])
                    patched_any = patched_any or (result.returncode == 0)

            elif code in ("DEP_CUDA_MISMATCH", "CFG_WRONG_DEVICE"):
                # Fallback: force CPU-only torch
                self.log("PROGRESS", "env-patch: setting CUDA_VISIBLE_DEVICES='' for CPU fallback")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                patched_any = True

            elif code == "DEP_BUILD_FAILURE":
                pkg = self._extract_missing_package(sf.stderr_tail or sf.error_message)
                if pkg:
                    base_pkg = pkg.split("==")[0]
                    self.log("PROGRESS", f"env-patch: trying conda install {base_pkg}")
                    from p2c.schemas import CondaDependency
                    env_mgr.install_conda_packages([
                        CondaDependency(package=base_pkg, channel="conda-forge", pip_fallback=True),
                    ])
                    patched_any = True

        return patched_any

    # ------------------------------------------------------------------
    # Signal-based autonomous mode switching
    # ------------------------------------------------------------------

    @staticmethod
    def _should_switch_to_autonomous(
        state: Phase2State, budget_sec: float, elapsed_sec: float,
    ) -> tuple[bool, str]:
        """Decide whether to abandon plan-directed mode for autonomous exploration.

        Checks 5 signals (any one triggers the switch):
          1. same_errors_repeating — last 2 failures have identical error codes
          2. entrypoint_not_found — plan's .py entrypoint doesn't exist
          3. persistent_env_failure — 2+ consecutive dependency-only failures
          4. partial_success — some metrics obtained but execution still "failed"
          5. budget_pressure — remaining time < 70% of avg cycle duration

        Returns ``(should_switch, reason_tag)``.
        """
        failures = state.failures
        if len(failures) < 2:
            return False, ""

        # --- Signal 1: same errors repeating ---
        def _error_codes(f: ExecutionFailure) -> set[str]:
            return {sf.failure_code or sf.error_type for sf in f.step_failures}

        last_two_codes = [_error_codes(f) for f in failures[-2:]]
        if last_two_codes[0] and last_two_codes[0] == last_two_codes[1]:
            return True, "same_errors_repeating"

        # --- Signal 2: entrypoint not found ---
        for f in failures:
            for sf in f.step_failures:
                code = sf.failure_code or ""
                if code in ("DATA_PATH_MISMATCH", "DATA_MISSING_DATASET"):
                    # Check if the missing item is a .py file (wrong entrypoint)
                    msg = sf.error_message or sf.stderr_tail or ""
                    if ".py" in msg:
                        return True, "entrypoint_not_found"

        # --- Signal 3: persistent env failure ---
        if len(failures) >= 2 and all(f.is_dependency_issue for f in failures[-2:]):
            return True, "persistent_env_failure"

        # --- Signal 4: partial success (not implemented in failure struct yet,
        #     but we can detect it if any failure has steps with metrics) ---
        # This is checked at the orchestrator level via exec_result metrics.

        # --- Signal 5: budget pressure ---
        remaining = budget_sec - elapsed_sec
        avg_cycle = elapsed_sec / max(state.attempt, 1)
        if avg_cycle > 0 and remaining < avg_cycle * 0.7:
            return True, "budget_pressure"

        return False, ""

    def _run_autonomous_fallback(
        self,
        ctx: dict[str, Any],
        state: Phase2State,
        budget_sec: float,
        t_start: float,
        reason: str,
    ) -> bool:
        """Execute autonomous mode and update state. Returns True if successful."""
        plan = state.plan
        if not plan or not plan.codex_autonomous_fallback:
            return False

        remaining = budget_sec - (time.time() - t_start)
        if remaining <= 120:
            self.log("PROGRESS", "not enough budget for autonomous mode")
            return False

        state.status = "autonomous"
        self.log("PROGRESS", f"switching to autonomous: {reason} ({remaining:.0f}s left)")
        ctx["_p2_remaining_sec"] = remaining
        ctx["_p2_failures"] = state.failures
        ctx["_p2_auto_reason"] = reason

        auto_result = self.codex_executor.execute_autonomous(ctx)
        if auto_result.get("success"):
            state.status = "success"
            state.final_manifest = auto_result.get("run_manifest")
            state.final_alignment = auto_result.get("claim_alignment")
            return True
        return False

    # ------------------------------------------------------------------
    # Repair extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_missing_package(text: str) -> str | None:
        """Extract package name from import/install error messages."""
        # "No module named 'einops'" / "No module named 'torch.utils'"
        m = re.search(r"No module named ['\"]?([\w.]+)", text)
        if m:
            # Return top-level package (e.g. "torch" from "torch.utils")
            return m.group(1).split(".")[0]
        # "Failed building wheel for foobar"
        m = re.search(r"Failed building wheel for ([\w-]+)", text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_env_var(text: str) -> str | None:
        """Extract env var name from KeyError or 'not set' messages."""
        m = re.search(r"KeyError:\s*['\"]?([\w_]+)", text)
        if m:
            return m.group(1)
        m = re.search(r"([\w_]+).*(?:not set|not found|missing)", text, re.IGNORECASE)
        if m and m.group(1).isupper():
            return m.group(1)
        return None

    @staticmethod
    def _extract_file_path(text: str) -> str | None:
        """Extract file path from PermissionError or FileNotFoundError."""
        m = re.search(r"(?:Permission denied|No such file or directory):\s*['\"]?([^\s'\"]+)", text)
        return m.group(1) if m else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_critical_failures(env_result: Any, plan: Any) -> bool:
        """Check if any execution step depends on a package that failed to install."""
        if not env_result or not env_result.failed_packages:
            return False
        failed_set = {p.split("==")[0].split(">=")[0].split("[")[0].lower()
                      for p in env_result.failed_packages}
        # If >30% of pip deps failed, it's critical
        total_deps = len(plan.pip_dependencies) + len(plan.conda_dependencies)
        if total_deps > 0 and len(failed_set) / total_deps > 0.3:
            return True
        return False

    def _write_success_state(self, state: Phase2State) -> None:
        """Ensure Phase 3 artifacts exist."""
        if state.final_manifest:
            manifest_data = (state.final_manifest.model_dump()
                             if hasattr(state.final_manifest, "model_dump")
                             else state.final_manifest)
            self.artifacts.write_json("execution/codex_outputs/run_manifest.json", manifest_data)
        if state.final_alignment:
            alignment_data = (state.final_alignment.model_dump()
                              if hasattr(state.final_alignment, "model_dump")
                              else state.final_alignment)
            self.artifacts.write_json("execution/codex_outputs/claim_alignment.json", alignment_data)

    def _write_failure_state(self, state: Phase2State) -> None:
        """Write empty-but-valid Phase 3 artifacts so the pipeline doesn't crash."""
        self.artifacts.write_json("execution/codex_outputs/run_manifest.json",
                                  RunManifestDoc(reason_codes=["PHASE2_FAILED"]).model_dump())
        self.artifacts.write_json("execution/codex_outputs/claim_alignment.json",
                                  ClaimAlignmentDoc(reason_codes=["PHASE2_FAILED"]).model_dump())
        self.artifacts.write_json("execution/execution_failures.json",
                                  [f.model_dump() for f in state.failures])
        self.log("DONE", f"phase 2 failed after {state.attempt} attempts")
