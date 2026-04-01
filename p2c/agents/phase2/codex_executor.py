"""CodexExecutorAgent — runs codex CLI locally following the execution plan."""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.local_prompt_templates import (
    build_autonomous_exploration_prompt,
    build_step_execution_prompt,
)
from p2c.agents.phase2.result_extraction import (
    build_claim_alignment,
    build_run_manifest,
    classify_error,
    classify_error_v2,
    extract_metrics_from_file,
    extract_metrics_from_stdout,
    extract_traceback,
    is_fast_fail,
)
from p2c.runtime.conda_env import CondaEnvManager
from p2c.schemas import (
    ClaimsIR,
    ExecutionFailure,
    ExecutionPlan,
    ExecutionStep,
    MetricContract,
    StepFailure,
)

DEFAULT_CODEX_MODEL = "gpt-5.4"


class CodexExecutorAgent(BaseAgent):
    """Execute plan steps via local ``codex exec --full-auto``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="codex_executor", *args, **kwargs)

    # ------------------------------------------------------------------
    # Mode A: Plan-directed execution
    # ------------------------------------------------------------------

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        plan: ExecutionPlan = ctx["_p2_plan"]
        env_mgr: CondaEnvManager = ctx["_p2_env_mgr"]
        remaining_sec: float = ctx.get("_p2_remaining_sec", plan.total_budget_sec)
        repo_dir = str(ctx["repo_dir"])

        contract_data = self.artifacts.read_json("task/metric_contract.json")
        contract = MetricContract(**contract_data) if contract_data else MetricContract()
        claims_ir_data = self.artifacts.read_json("fingerprint/claims_ir.json")
        claims_ir = ClaimsIR(**claims_ir_data) if claims_ir_data.get("claims") else ClaimsIR()

        outputs_dir = str(self.artifacts.path("execution/codex_outputs"))
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)

        all_runs: list[dict[str, Any]] = []
        all_metrics: dict[str, Any] = {}
        step_failures: list[StepFailure] = []
        any_success = False
        t_start = time.time()

        # Execution journal — accumulates results across steps so each
        # subsequent step can see what happened before (inter-step feedback).
        execution_journal: list[dict[str, Any]] = []

        # Topological sort (simple: respect depends_on ordering)
        ordered_steps = self._topo_sort(plan.execution_steps)

        for step in ordered_steps:
            elapsed = time.time() - t_start
            if elapsed >= remaining_sec:
                self.log("PROGRESS", f"budget exhausted, skipping {step.step_id}")
                break

            step_remaining = min(step.timeout_sec, remaining_sec - elapsed)
            self.log("PROGRESS", f"step {step.step_id}: {step.description[:80]}")

            # Build compressed prior-step context for the prompt
            prior_context = self._summarize_journal(execution_journal)

            run_result = self._execute_step(
                step=step,
                env_mgr=env_mgr,
                repo_dir=repo_dir,
                contract=contract,
                outputs_dir=outputs_dir,
                timeout_sec=int(step_remaining),
                prior_step_results=prior_context,
            )

            # Append to journal BEFORE processing next step
            execution_journal.append({
                "step_id": step.step_id,
                "description": step.description[:120],
                "exit_code": run_result.get("exit_code", 1),
                "metrics": run_result.get("metrics", {}),
                "error_type": run_result.get("error_type"),
                "failure_code": run_result.get("failure_code"),
                "stdout_tail": (run_result.get("stdout_tail") or "")[-500:],
                "stderr_tail": (run_result.get("stderr_tail") or "")[-300:],
            })

            all_runs.append(run_result)
            if run_result.get("metrics"):
                all_metrics.update(run_result["metrics"])
                any_success = True
            elif run_result.get("exit_code", 1) != 0:
                sf = StepFailure(
                    step_id=step.step_id,
                    command=run_result.get("command", step.command),
                    exit_code=int(run_result.get("exit_code", 1)),
                    error_type=run_result.get("error_type", "unknown"),
                    error_message=run_result.get("error_message", ""),
                    stdout_tail=run_result.get("stdout_tail", "")[-2000:],
                    stderr_tail=run_result.get("stderr_tail", "")[-2000:],
                    traceback=run_result.get("traceback"),
                    # v2 taxonomy fields for repair routing
                    failure_code=run_result.get("failure_code"),
                    failure_layer=run_result.get("failure_layer"),
                    repair_strategy=run_result.get("repair_strategy"),
                    repair_action=run_result.get("repair_action"),
                    auto_repair_confidence=run_result.get("auto_repair_confidence"),
                )
                step_failures.append(sf)
                if run_result.get("fast_fail"):
                    self.log("PROGRESS", f"fast-fail on {step.step_id}, stopping")
                    break

        # Build Phase 3 outputs
        manifest = build_run_manifest(all_runs, reason_codes=["LOCAL_CODEX_EXEC"])
        alignment = build_claim_alignment(claims_ir, all_metrics)

        if any_success:
            self.artifacts.write_json("execution/codex_outputs/run_manifest.json", manifest.model_dump())
            self.artifacts.write_json("execution/codex_outputs/claim_alignment.json", alignment.model_dump())
            return {
                "success": True,
                "run_manifest": manifest,
                "claim_alignment": alignment,
                "metrics": all_metrics,
            }

        failure = ExecutionFailure(
            attempt=int(ctx.get("_p2_attempt", 1)),
            plan_version=plan.plan_version,
            stage="execution",
            step_failures=step_failures,
            overall_error=f"{len(step_failures)} steps failed",
            is_dependency_issue=any(
                sf.error_type in ("dependency", "import") for sf in step_failures
            ),
        )
        self.artifacts.write_json("execution/execution_failures.json",
                                  [failure.model_dump()])
        return {"success": False, "failure": failure}

    # ------------------------------------------------------------------
    # Mode B: Autonomous exploration fallback
    # ------------------------------------------------------------------

    def execute_autonomous(self, ctx: dict[str, Any]) -> dict[str, Any]:
        env_mgr: CondaEnvManager = ctx["_p2_env_mgr"]
        repo_dir = str(ctx["repo_dir"])
        remaining_sec: float = ctx.get("_p2_remaining_sec", 600)
        failures: list[ExecutionFailure] = ctx.get("_p2_failures", [])

        contract_data = self.artifacts.read_json("task/metric_contract.json")
        contract = MetricContract(**contract_data) if contract_data else MetricContract()
        claims_ir_data = self.artifacts.read_json("fingerprint/claims_ir.json")
        claims_ir = ClaimsIR(**claims_ir_data) if claims_ir_data.get("claims") else ClaimsIR()

        outputs_dir = str(self.artifacts.path("execution/codex_outputs"))
        expected_results = self.artifacts.read_json("execution/execution_plan.json").get("expected_results", [])

        prompt = build_autonomous_exploration_prompt(
            repo_dir=repo_dir,
            failure_history_json=json.dumps(
                [f.model_dump() if hasattr(f, "model_dump") else f for f in failures],
                indent=2, ensure_ascii=False,
            ),
            expected_results_json=json.dumps(expected_results, indent=2, ensure_ascii=False),
            outputs_dir=outputs_dir,
        )

        self.log("PROGRESS", "starting autonomous exploration mode...")
        timeout = max(300, int(remaining_sec))
        t0 = time.time()
        proc = self._run_codex(env_mgr, prompt, repo_dir, timeout_sec=timeout)
        runtime = time.time() - t0

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        self.artifacts.write_text("execution/codex_outputs/autonomous_stdout.log", stdout)
        self.artifacts.write_text("execution/codex_outputs/autonomous_stderr.log", stderr)

        # Try file-based extraction first
        metrics = extract_metrics_from_file(f"{outputs_dir}/autonomous_results.json")
        # Merge stdout-based
        stdout_metrics = extract_metrics_from_stdout(stdout, contract)
        for k, v in stdout_metrics.items():
            if k not in metrics:
                metrics[k] = v

        run_entry = {
            "step_id": "autonomous",
            "command": "codex autonomous exploration",
            "cwd": repo_dir,
            "exit_code": proc.returncode,
            "runtime_sec": runtime,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "metrics": metrics,
        }

        manifest = build_run_manifest([run_entry], reason_codes=["AUTONOMOUS_EXPLORATION"])
        alignment = build_claim_alignment(claims_ir, metrics)

        if metrics:
            self.artifacts.write_json("execution/codex_outputs/run_manifest.json", manifest.model_dump())
            self.artifacts.write_json("execution/codex_outputs/claim_alignment.json", alignment.model_dump())
            return {"success": True, "run_manifest": manifest, "claim_alignment": alignment, "metrics": metrics}

        return {
            "success": False,
            "failure": ExecutionFailure(
                attempt=0, stage="autonomous",
                overall_error="Autonomous exploration produced no metrics",
            ),
        }

    # ------------------------------------------------------------------
    # Internal: execute a single step
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        *,
        step: ExecutionStep,
        env_mgr: CondaEnvManager,
        repo_dir: str,
        contract: MetricContract,
        outputs_dir: str,
        timeout_sec: int,
        prior_step_results: str | None = None,
    ) -> dict[str, Any]:
        parsers = [p.model_dump() for p in contract.parsers]
        prompt = build_step_execution_prompt(
            repo_dir=repo_dir,
            step_description=step.description,
            step_command=step.command,
            expected_metrics=step.expected_metrics,
            metric_parsers=parsers,
            outputs_dir=outputs_dir,
            step_id=step.step_id,
            prior_step_results=prior_step_results,
        )

        cwd = str(Path(repo_dir) / step.cwd)
        t0 = time.time()
        proc = self._run_codex(env_mgr, prompt, cwd, timeout_sec=timeout_sec)
        runtime = time.time() - t0

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # Persist step logs
        self.artifacts.write_text(f"execution/codex_outputs/step_{step.step_id}_stdout.log", stdout)
        self.artifacts.write_text(f"execution/codex_outputs/step_{step.step_id}_stderr.log", stderr)

        # Extract metrics (file first, then stdout)
        metrics = extract_metrics_from_file(f"{outputs_dir}/step_{step.step_id}_result.json")
        stdout_metrics = extract_metrics_from_stdout(stdout, contract)
        for k, v in stdout_metrics.items():
            if k not in metrics:
                metrics[k] = v

        # Rich v2 classification (provides repair routing info)
        failure_spec = classify_error_v2(
            stdout, stderr, proc.returncode,
            metrics=metrics,
            expected_metrics=step.expected_metrics,
        )

        result: dict[str, Any] = {
            "step_id": step.step_id,
            "command": step.command,
            "cwd": step.cwd,
            "exit_code": proc.returncode,
            "runtime_sec": runtime,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "metrics": metrics,
            "fast_fail": failure_spec.is_fast_fail,
            "error_type": failure_spec.legacy_error_type,
            "error_message": stderr[-500:] if proc.returncode != 0 else "",
            "traceback": extract_traceback(stderr),
            # v2 taxonomy fields
            "failure_code": failure_spec.code,
            "failure_layer": failure_spec.layer,
            "repair_strategy": failure_spec.repair_strategy.value,
            "repair_action": failure_spec.repair_action,
            "auto_repair_confidence": failure_spec.auto_repair_confidence,
        }
        return result

    # ------------------------------------------------------------------
    # Execution journal summarization
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_journal(journal: list[dict[str, Any]]) -> str | None:
        """Compress the execution journal for inclusion in the next step's prompt.

        Strategy: keep the last 3 entries in full detail, compress earlier entries
        to just step_id + exit_code + metrics (saves prompt tokens).
        Returns None if the journal is empty.
        """
        if not journal:
            return None

        MAX_FULL = 3  # number of recent entries to keep in full detail
        if len(journal) <= MAX_FULL:
            return json.dumps(journal, indent=2, ensure_ascii=False, default=str)

        # Compress older entries: drop stdout/stderr tails
        compressed = []
        for entry in journal[:-MAX_FULL]:
            compressed.append({
                "step_id": entry["step_id"],
                "exit_code": entry["exit_code"],
                "metrics": entry.get("metrics", {}),
                "failure_code": entry.get("failure_code"),
            })
        # Keep recent entries in full
        full_recent = journal[-MAX_FULL:]
        return json.dumps(compressed + full_recent, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Codex CLI invocation
    # ------------------------------------------------------------------

    @staticmethod
    def _run_codex(
        env_mgr: CondaEnvManager,
        prompt: str,
        cwd: str,
        timeout_sec: int = 600,
    ) -> Any:
        """Run ``codex exec --full-auto`` inside the managed environment."""
        model = (os.getenv("P2C_CODEX_MODEL") or DEFAULT_CODEX_MODEL).strip()
        codex_bin = (
            os.getenv("P2C_CODEX_BIN")
            or CondaEnvManager._resolve_codex_bin()
            or "codex"
        )
        codex_cmd = (
            f"{shlex.quote(codex_bin)} exec --full-auto"
            f" -m {shlex.quote(model)} {shlex.quote(prompt)}"
        )
        return env_mgr.run_in_env(codex_cmd, cwd=cwd, timeout_sec=timeout_sec)

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    @staticmethod
    def _topo_sort(steps: list[ExecutionStep]) -> list[ExecutionStep]:
        """Sort steps respecting depends_on; falls back to original order on cycles."""
        by_id = {s.step_id: s for s in steps}
        visited: set[str] = set()
        result: list[ExecutionStep] = []

        def visit(sid: str) -> None:
            if sid in visited:
                return
            visited.add(sid)
            step = by_id.get(sid)
            if not step:
                return
            for dep in step.depends_on:
                if dep not in visited:
                    visit(dep)
            result.append(step)

        for s in steps:
            visit(s.step_id)
        return result
