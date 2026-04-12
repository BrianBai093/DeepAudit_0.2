"""CodexExecutorAgent — Claude Code Agent SDK as primary execution engine.

v0.5: Every execution step is handled by Claude Code via the Agent SDK.
Claude Code's ``Bash`` tool inherits the host process environment,
eliminating the PATH-reset problem that plagued Codex CLI.  The agent
can self-heal failures, install missing packages, and iterate — all
within a single SDK session per step.

Architecture (autoresearch-inspired):
  1. Build a task prompt describing the step + expected metrics.
  2. Invoke ``claude-agent-sdk.query()`` — Claude Code reads files, runs
     commands via its Bash tool, inspects errors, and retries.
  3. Collect all Bash stdout/stderr + assistant text → extract metrics.
  4. No separate "direct execution" layer — Claude Code IS the executor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from claude_agent_sdk import (  # type: ignore[import-untyped]
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        UserMessage,
        query,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    # Placeholder types so the module can be imported without the SDK
    # (e.g. during tests or static analysis).  Runtime calls to
    # _run_claude() will fail fast with a clear error message.
    AssistantMessage = type("AssistantMessage", (), {})  # type: ignore[misc,assignment]
    ClaudeAgentOptions = type("ClaudeAgentOptions", (), {})  # type: ignore[misc,assignment]
    ResultMessage = type("ResultMessage", (), {})  # type: ignore[misc,assignment]
    UserMessage = type("UserMessage", (), {})  # type: ignore[misc,assignment]

    async def query(**kwargs):  # type: ignore[misc]
        raise RuntimeError("claude-agent-sdk is not installed")
        yield  # noqa: make it an async generator

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.local_prompt_templates import (
    build_autonomous_exploration_prompt,
    build_step_execution_prompt,
)
from p2c.agents.phase2.result_extraction import (
    build_claim_alignment,
    build_run_manifest,
    classify_error_v2,
    extract_metrics_from_file,
    extract_metrics_from_stdout,
    extract_traceback,
    is_static_inspection_command,
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

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_GENERIC_SCALAR_METRICS = {
    "accuracy",
    "auc",
    "bleu",
    "f1",
    "loss",
    "mae",
    "mse",
    "perplexity",
    "pr_auc",
    "precision",
    "recall",
    "rmse",
    "roc_auc",
    "rouge",
}

# ---------------------------------------------------------------------------
# Keys forwarded from the host environment into the Claude Code session.
# ---------------------------------------------------------------------------
_FORWARD_ENV_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "HOME", "USER", "PATH", "LANG", "SHELL",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
    "CONDA_EXE", "CONDA_PREFIX",
)


@dataclass
class ClaudeResult:
    """Structured result from a Claude Code agent session.

    Mirrors ``subprocess.CompletedProcess[str]`` so downstream consumers
    (metric extraction, failure classification) work unchanged.

    ``narrative`` carries Claude's own assistant text (reasoning, status
    commentary) on a separate channel so it does not contaminate stdout
    used for metric extraction.
    """
    stdout: str
    stderr: str
    returncode: int
    narrative: str = ""


class CodexExecutorAgent(BaseAgent):
    """Execute plan steps via Claude Code Agent SDK (primary execution engine)."""

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

        outputs_dir = str(self.artifacts.path("execution/codex_outputs").resolve())
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)

        all_runs: list[dict[str, Any]] = []
        all_metrics: dict[str, Any] = {}
        step_failures: list[StepFailure] = []
        any_success = False
        any_completed_step = False
        t_start = time.time()

        execution_journal: list[dict[str, Any]] = []
        ordered_steps = self._topo_sort(plan.execution_steps)

        for step in ordered_steps:
            elapsed = time.time() - t_start
            if elapsed >= remaining_sec:
                self.log("PROGRESS", f"budget exhausted, skipping {step.step_id}")
                break

            step_remaining = min(step.timeout_sec, remaining_sec - elapsed)
            self.log("PROGRESS", f"step {step.step_id} (Claude Code): {step.description[:80]}")

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
            if int(run_result.get("exit_code", 1)) == 0:
                any_completed_step = True
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

        manifest = build_run_manifest(all_runs, reason_codes=["CLAUDE_CODE_EXEC"])
        alignment = build_claim_alignment(
            claims_ir,
            all_metrics,
            metric_sources=self._collect_metric_sources(all_runs),
        )

        if any_success:
            self.artifacts.write_json("execution/codex_outputs/run_manifest.json", manifest.model_dump())
            self.artifacts.write_json("execution/codex_outputs/claim_alignment.json", alignment.model_dump())
            return {
                "success": True,
                "run_manifest": manifest,
                "claim_alignment": alignment,
                "metrics": all_metrics,
            }

        if any_completed_step and not step_failures:
            step_failures.append(
                StepFailure(
                    step_id="phase2_metrics",
                    command="phase2 execution",
                    exit_code=1,
                    error_type="unknown",
                    error_message="Execution completed without producing expected metrics",
                    stdout_tail="",
                    stderr_tail="",
                    failure_code="RESULT_MISSING_METRICS",
                    failure_layer="result",
                    repair_strategy="replan",
                    repair_action="Adjust planner/extraction so successful runs emit metrics",
                    auto_repair_confidence=0.4,
                )
            )

        failure = ExecutionFailure(
            attempt=int(ctx.get("_p2_attempt", 1)),
            plan_version=plan.plan_version,
            stage="execution",
            step_failures=step_failures,
            overall_error=(
                "Execution completed but produced no metrics"
                if any_completed_step and not any_success
                else f"{len(step_failures)} steps failed"
            ),
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

        outputs_dir = str(self.artifacts.path("execution/codex_outputs").resolve())
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

        self.log("PROGRESS", "starting autonomous exploration mode (Claude Code)...")
        timeout = max(300, int(remaining_sec))
        t0 = time.time()
        proc = self._run_claude(env_mgr, prompt, repo_dir, timeout_sec=timeout)
        runtime = time.time() - t0

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        self.artifacts.write_text("execution/codex_outputs/autonomous_stdout.log", stdout)
        self.artifacts.write_text("execution/codex_outputs/autonomous_stderr.log", stderr)
        autonomous_narrative = getattr(proc, "narrative", "") or ""
        if autonomous_narrative:
            self.artifacts.write_text(
                "execution/codex_outputs/autonomous_claude_narrative.log",
                autonomous_narrative,
            )

        # Try file-based extraction first
        metrics = extract_metrics_from_file(f"{outputs_dir}/autonomous_results.json")
        # Merge stdout-based
        stdout_metrics = extract_metrics_from_stdout(stdout, contract, command="claude autonomous exploration")
        for k, v in stdout_metrics.items():
            if k not in metrics:
                metrics[k] = v

        run_entry = {
            "step_id": "autonomous",
            "command": "claude autonomous exploration",
            "cwd": repo_dir,
            "exit_code": proc.returncode,
            "runtime_sec": runtime,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "metrics": metrics,
        }

        manifest = build_run_manifest([run_entry], reason_codes=["AUTONOMOUS_EXPLORATION"])
        alignment = build_claim_alignment(
            claims_ir,
            metrics,
            metric_sources=self._collect_metric_sources([run_entry]),
        )

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
    # Internal: execute a single step via Claude Code (primary executor)
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
        cwd = str(Path(repo_dir) / step.cwd)
        parsers = [p.model_dump() for p in contract.parsers]
        step_result_relative = f"execution/codex_outputs/step_{step.step_id}_result.json"
        step_result_path = self.artifacts.path(step_result_relative)
        step_result_path.parent.mkdir(parents=True, exist_ok=True)
        step_result_path.unlink(missing_ok=True)

        started = time.time()

        # Build prompt — Claude Code will execute, self-heal, and iterate.
        prompt = build_step_execution_prompt(
            repo_dir=repo_dir,
            step_description=step.description,
            step_command=step.command,
            expected_metrics=step.expected_metrics,
            metric_parsers=parsers,
            outputs_dir=outputs_dir,
            step_id=step.step_id,
            failure_context=None,  # no prior failure — Claude Code is primary
            prior_step_results=prior_step_results,
        )

        proc = self._run_claude(env_mgr, prompt, cwd, timeout_sec=timeout_sec)
        runtime_sec = time.time() - started

        attempt_records = [{
            "mode": "claude:primary",
            "command": step.command,
            "executed_command": step.command,
            "exit_code": proc.returncode,
            "runtime_sec": runtime_sec,
            "attempt_index": 1,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }]

        return self._finalize_step_result(
            step=step,
            contract=contract,
            step_result_relative=step_result_relative,
            default_command=step.command,
            proc=proc,
            runtime_sec=runtime_sec,
            stdout_text=proc.stdout or "",
            stderr_text=proc.stderr or "",
            attempt_records=attempt_records,
            execution_mode="claude_primary",
            effective_cwd=step.cwd,
        )

    def _finalize_step_result(
        self,
        *,
        step: ExecutionStep,
        contract: MetricContract,
        step_result_relative: str,
        default_command: str,
        proc: Any,
        runtime_sec: float,
        stdout_text: str,
        stderr_text: str,
        attempt_records: list[dict[str, Any]],
        execution_mode: str,
        effective_cwd: str,
    ) -> dict[str, Any]:
        stored = self.artifacts.read_json(step_result_relative)
        stored_metrics = stored.get("metrics") if isinstance(stored.get("metrics"), dict) else {}
        ignore_metrics = self._should_ignore_step_metrics(step)
        if ignore_metrics:
            metrics: dict[str, Any] = {}
        else:
            stdout_metrics = extract_metrics_from_stdout(stdout_text, contract, command=step.command)
            stdout_metric_names = {str(name).lower() for name in stdout_metrics}
            metrics = dict(stdout_metrics)
            for key, value in stored_metrics.items():
                if self._allow_stored_metric(str(key), stdout_metric_names):
                    metrics[key] = value

        selected_attempt = self._find_attempt_for_stored_result(stored, attempt_records)
        effective_exit_code = proc.returncode
        stored_exit_code = stored.get("exit_code")
        if isinstance(stored_exit_code, (int, float)) and not isinstance(stored_exit_code, bool):
            effective_exit_code = int(stored_exit_code)
        elif proc.returncode != 0:
            selected_attempt = self._choose_effective_failure_attempt(attempt_records)
            if selected_attempt is not None:
                effective_exit_code = int(selected_attempt.get("exit_code", proc.returncode))

        if selected_attempt is None and attempt_records:
            selected_attempt = attempt_records[-1]

        selected_stdout = str((selected_attempt or {}).get("stdout") or proc.stdout or "")
        selected_stderr = str((selected_attempt or {}).get("stderr") or proc.stderr or "")
        forced_failure = False
        if effective_exit_code == 0 and self._looks_like_false_shell_success(
            step=step,
            metrics=metrics,
            stdout=selected_stdout,
            stderr=selected_stderr,
        ):
            effective_exit_code = 1
            forced_failure = True

        if isinstance(stored.get("command"), str):
            effective_command = stored.get("command")
        elif selected_attempt is not None and isinstance(selected_attempt.get("command"), str):
            effective_command = selected_attempt.get("command")
        else:
            effective_command = default_command
        notes = stored.get("notes") if isinstance(stored.get("notes"), str) else self._build_attempt_notes(attempt_records)
        primary_attempt = attempt_records[0] if attempt_records else None
        degraded_success = self._is_degraded_success(primary_attempt, selected_attempt, effective_exit_code, effective_command)
        run_status = "partial" if degraded_success else ("ok" if effective_exit_code == 0 else "failed")
        run_params: dict[str, Any] = {
            "effective_cwd": effective_cwd,
            "path_resolution_mode": step.path_resolution_mode or "default",
            "derived_from_wrapper": step.derived_from_wrapper,
            "planned_command": step.command,
            "expected_metrics": list(step.expected_metrics),
            "is_setup": step.is_setup,
        }
        run_reason_codes: list[str] = []
        if ignore_metrics and stored_metrics:
            run_reason_codes.append("METRICS_IGNORED_FOR_INSPECTION_STEP")
        if effective_command.strip() != step.command.strip() and not self._commands_share_target(step.command, effective_command):
            run_reason_codes.append("COMMAND_DRIFT")
        if degraded_success and primary_attempt is not None:
            run_params.update({
                "primary_exit_code": int(primary_attempt.get("exit_code", 1)),
                "fallback_used": True,
                "degraded_success": True,
            })
            run_reason_codes = list(dict.fromkeys([
                *run_reason_codes,
                "PRIMARY_FAILED_FALLBACK_SUCCEEDED",
                "NON_EQUIVALENT_FALLBACK",
            ]))
        if forced_failure:
            run_status = "failed"
            run_reason_codes = list(dict.fromkeys([*run_reason_codes, "SHELL_WRAPPER_FALSE_SUCCESS", "PATH_RESOLUTION_FAILURE"]))
            notes = f"{notes}\n- shell wrapper returned success despite path-resolution failure; forced to failed."

        self.artifacts.write_json(
            step_result_relative,
            {
                "command": effective_command,
                "exit_code": effective_exit_code,
                "metrics": metrics,
                "notes": notes,
            },
        )
        self.artifacts.write_text(f"execution/codex_outputs/step_{step.step_id}_stdout.log", stdout_text)
        self.artifacts.write_text(f"execution/codex_outputs/step_{step.step_id}_stderr.log", stderr_text)
        narrative_text = getattr(proc, "narrative", "") or ""
        if narrative_text:
            self.artifacts.write_text(
                f"execution/codex_outputs/step_{step.step_id}_claude_narrative.log",
                narrative_text,
            )

        classify_stdout = selected_stdout or stdout_text
        classify_stderr = selected_stderr or stderr_text
        if effective_exit_code != 0:
            classify_stdout = (selected_attempt or {}).get("stdout", classify_stdout) or stdout_text
            classify_stderr = (selected_attempt or {}).get("stderr", classify_stderr) or stderr_text

        failure_spec = classify_error_v2(
            classify_stdout,
            classify_stderr,
            effective_exit_code,
            metrics=metrics,
            expected_metrics=step.expected_metrics,
        )

        error_source = classify_stderr or classify_stdout or stderr_text or stdout_text
        return {
            "step_id": step.step_id,
            "command": effective_command,
            "cwd": step.cwd,
            "params": run_params,
            "exit_code": effective_exit_code,
            "status": run_status,
            "runtime_sec": runtime_sec,
            "stdout_tail": stdout_text[-2000:],
            "stderr_tail": stderr_text[-2000:],
            "metrics": metrics,
            "reason_codes": run_reason_codes,
            "fast_fail": failure_spec.is_fast_fail,
            "error_type": failure_spec.legacy_error_type,
            "error_message": error_source[-500:] if effective_exit_code != 0 else "",
            "traceback": extract_traceback(stderr_text),
            "failure_code": failure_spec.code,
            "failure_layer": failure_spec.layer,
            "repair_strategy": failure_spec.repair_strategy.value,
            "repair_action": failure_spec.repair_action,
            "auto_repair_confidence": failure_spec.auto_repair_confidence,
            "execution_mode": execution_mode,
            "attempted_commands": attempt_records,
        }

    @staticmethod
    def _should_ignore_step_metrics(step: ExecutionStep) -> bool:
        """Metricless setup/inspection steps should not contribute result evidence."""
        if step.expected_metrics or step.produced_artifacts:
            return False
        if step.is_setup:
            return True
        return is_static_inspection_command(step.command)

    @staticmethod
    def _allow_stored_metric(metric_name: str, observed_stdout_names: set[str]) -> bool:
        lowered = metric_name.lower()
        if lowered.endswith("_all"):
            return False
        if lowered in observed_stdout_names:
            return False
        if observed_stdout_names and lowered in _GENERIC_SCALAR_METRICS:
            return False
        return True

    @classmethod
    def _is_shell_step(cls, command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        # Skip leading VAR=value assignments
        remaining = list(tokens)
        while remaining and "=" in remaining[0] and not remaining[0].startswith("-"):
            name, _, value = remaining[0].partition("=")
            if name.isidentifier() and value:
                remaining = remaining[1:]
                continue
            break
        if not remaining:
            return False
        head = remaining[0]
        if head in {"bash", "sh"} and len(remaining) >= 2 and remaining[1].endswith(".sh"):
            return True
        return (head.startswith("./") or head.startswith("../")) and head.endswith(".sh")

    @staticmethod
    def _is_shell_compound_command(command: str) -> bool:
        raw = str(command or "")
        if not raw.strip():
            return False
        if "\n" in raw:
            return True
        markers = ("|", "&&", ";", "||")
        if any(marker in raw for marker in markers):
            return True
        return raw.lstrip().startswith("cd ")

    @staticmethod
    def _contains_path_resolution_error(stdout: str, stderr: str) -> bool:
        combined = f"{stdout}\n{stderr}".lower()
        patterns = (
            "no such file or directory",
            "can't open file",
            "cannot open",
            "cd:",
        )
        return any(pattern in combined for pattern in patterns)

    @staticmethod
    def _contains_runtime_failure_signal(stdout: str, stderr: str) -> bool:
        combined = f"{stdout}\n{stderr}".lower()
        patterns = (
            "traceback (most recent call last)",
            "modulenotfounderror",
            "importerror",
            "syntaxerror",
            "nameerror",
            "typeerror",
            "valueerror",
            "assertionerror",
            "runtimeerror",
        )
        return any(pattern in combined for pattern in patterns)

    @classmethod
    def _looks_like_false_shell_success(
        cls,
        *,
        step: ExecutionStep,
        metrics: dict[str, Any],
        stdout: str,
        stderr: str,
    ) -> bool:
        shell_like = cls._is_shell_step(step.command) or cls._is_shell_compound_command(step.command)
        if not shell_like:
            return False
        if not (cls._contains_path_resolution_error(stdout, stderr) or cls._contains_runtime_failure_signal(stdout, stderr)):
            return False
        if step.produced_artifacts or step.expected_metrics:
            return not metrics
        return True

    @staticmethod
    def _append_attempt_output(target: list[str], label: str, command: str, content: str) -> None:
        body = content or ""
        if body and not body.endswith("\n"):
            body += "\n"
        target.append(f"===== {label} | {command} =====\n{body}")

    @staticmethod
    def _build_attempt_notes(attempt_records: list[dict[str, Any]]) -> str:
        lines = ["Execution attempts:"]
        for attempt in attempt_records:
            executed = str(attempt.get("executed_command") or attempt["command"])
            details = f"- {attempt['mode']}: `{attempt['command']}`"
            if executed != attempt["command"]:
                details += f" -> `{executed}`"
            details += f" (exit {attempt['exit_code']}, {attempt['runtime_sec']:.2f}s)"
            lines.append(details)
        return "\n".join(lines)

    @staticmethod
    def _extract_python_script_target(command: str) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            name, _, value = tokens[0].partition("=")
            if name.isidentifier() and value:
                tokens = tokens[1:]
                continue
            break
        tokens = CodexExecutorAgent._strip_conda_run_tokens(tokens)
        if len(tokens) < 2 or tokens[0] != "python":
            return None
        script = tokens[1]
        if script.endswith(".py"):
            return script
        return None

    @classmethod
    def _commands_share_target(cls, primary_command: str, effective_command: str) -> bool:
        if primary_command.strip() == effective_command.strip():
            return True
        primary_tokens = cls._command_tokens_without_conda(primary_command)
        effective_tokens = cls._command_tokens_without_conda(effective_command)
        if primary_tokens and primary_tokens == effective_tokens:
            return True
        primary_script = cls._extract_python_script_target(primary_command)
        effective_script = cls._extract_python_script_target(effective_command)
        return primary_script is not None and primary_script == effective_script

    @staticmethod
    def _command_tokens_without_conda(command: str) -> list[str] | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            name, _, value = tokens[0].partition("=")
            if name.isidentifier() and value:
                tokens = tokens[1:]
                continue
            break
        return CodexExecutorAgent._strip_conda_run_tokens(tokens)

    @staticmethod
    def _strip_conda_run_tokens(tokens: list[str]) -> list[str]:
        if len(tokens) < 2 or tokens[0] != "conda" or tokens[1] != "run":
            return tokens
        i = 2
        options_with_values = {"-n", "--name", "-p", "--prefix"}
        while i < len(tokens):
            token = tokens[i]
            if token in options_with_values:
                i += 2
                continue
            if token.startswith("--") and "=" in token:
                i += 1
                continue
            if token.startswith("-"):
                i += 1
                continue
            break
        return tokens[i:]

    @classmethod
    def _is_degraded_success(
        cls,
        primary_attempt: dict[str, Any] | None,
        selected_attempt: dict[str, Any] | None,
        effective_exit_code: int,
        effective_command: str,
    ) -> bool:
        if effective_exit_code != 0 or primary_attempt is None or selected_attempt is None:
            return False
        if int(primary_attempt.get("exit_code", 0)) == 0:
            return False
        mode = str(selected_attempt.get("mode") or "")
        if not mode.startswith("direct:fallback_"):
            return False
        primary_command = str(primary_attempt.get("command") or "")
        return not cls._commands_share_target(primary_command, effective_command)

    @staticmethod
    def _is_infrastructure_attempt(attempt: dict[str, Any]) -> bool:
        stdout = str(attempt.get("stdout") or "")
        stderr = str(attempt.get("stderr") or "")
        combined = f"{stdout}\n{stderr}".lower()
        if "traceback (most recent call last)" in combined:
            return False
        return any(
            pattern in combined
            for pattern in (
                "command not found",
                "env: ‘node’",
                "claude code agent error",
                "claude-agent-sdk not installed",
                "no such file or directory",
            )
        )

    @classmethod
    def _choose_effective_failure_attempt(cls, attempt_records: list[dict[str, Any]]) -> dict[str, Any] | None:
        failed_attempts = [attempt for attempt in attempt_records if int(attempt.get("exit_code", 0)) != 0]
        if not failed_attempts:
            return None

        def mode_priority(mode: str) -> int:
            if mode == "direct:primary":
                return 0
            if mode.startswith("direct:fallback_"):
                return 1
            return 2

        ranked = sorted(
            failed_attempts,
            key=lambda attempt: (
                1 if cls._is_infrastructure_attempt(attempt) else 0,
                mode_priority(str(attempt.get("mode") or "")),
                int(attempt.get("attempt_index") or 0),
            ),
        )
        return ranked[0]

    @staticmethod
    def _find_attempt_for_stored_result(
        stored: dict[str, Any],
        attempt_records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        stored_command = stored.get("command")
        stored_exit_code = stored.get("exit_code")
        if not isinstance(stored_exit_code, (int, float)) or isinstance(stored_exit_code, bool):
            return None
        normalized_exit = int(stored_exit_code)

        for attempt in attempt_records:
            if (
                attempt.get("command") == stored_command
                and int(attempt.get("exit_code", 0)) == normalized_exit
            ):
                return attempt
        for attempt in attempt_records:
            if int(attempt.get("exit_code", 0)) == normalized_exit:
                return attempt
        return None

    @staticmethod
    def _collect_metric_sources(runs: list[dict[str, Any]]) -> dict[str, list[str]]:
        metric_sources: dict[str, list[str]] = {}
        for run in runs:
            run_id = str(run.get("step_id") or run.get("run_id") or "unknown")
            source = f"execution/codex_outputs/run_manifest.json:{run_id}"
            for name, value in (run.get("metrics") or {}).items():
                if str(name).endswith("_all") or isinstance(value, (list, dict, tuple)):
                    continue
                metric_sources.setdefault(str(name), [])
                if source not in metric_sources[str(name)]:
                    metric_sources[str(name)].append(source)
        return metric_sources

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
    # Claude Code Agent SDK invocation (primary execution engine)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_claude(
        env_mgr: CondaEnvManager,
        prompt: str,
        cwd: str,
        timeout_sec: int = 600,
    ) -> ClaudeResult:
        """Run a Claude Code agent session via the Agent SDK.

        This is the **primary** execution method for every step.  Claude
        Code's ``Bash`` tool runs commands in the host shell, inheriting
        the current PATH.  The system prompt instructs Claude to use
        ``conda run -n <env>`` so the managed environment's Python and
        packages are used.
        """
        model = (os.getenv("P2C_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip()
        max_turns = max(10, min(50, timeout_sec // 20))
        env_name = env_mgr.env_name

        # System prompt: tell Claude Code how to activate the conda env.
        system_prompt = (
            f"You are executing code in a managed conda environment '{env_name}'.\n"
            f"For ALL python/pip commands, prefix with: "
            f"conda run --no-capture-output -n {env_name}\n"
            f"Example: conda run --no-capture-output -n {env_name} python train.py\n"
            "Do NOT create new conda/venv environments. One is already active.\n"
            "Always use `python` (not `python3`) to run scripts.\n"
            "\n"
            "EXECUTION DISCIPLINE (must follow):\n"
            "1. Run every command in the FOREGROUND. Do NOT use background mode "
            "(no `&`, no `nohup`, no `run_in_background=true`). Wait for each "
            "command to finish before issuing the next one.\n"
            "2. Run each script EXACTLY ONCE. Do not relaunch a long-running "
            "training script just because you have not seen output yet — be "
            "patient and wait for it to complete.\n"
            "3. Do not modify, patch, rewrite, or wrap repository source files. "
            "Use the file as-is. If a script needs an argument, pass it on the "
            "command line; do not edit the script.\n"
            "4. Keep your assistant-text commentary minimal. Do not narrate "
            "every step. Status checklists, ✅ summaries, and progress updates "
            "are unnecessary."
        )

        # Forward essential host env vars into the Claude Code session.
        child_env = {
            k: v for k, v in os.environ.items()
            if k in _FORWARD_ENV_KEYS and v
        }

        async def _execute() -> ClaudeResult:
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            narrative_parts: list[str] = []
            last_exit_code = 0

            async for msg in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=cwd,
                    allowed_tools=["Bash", "Read", "Glob", "Grep"],
                    permission_mode="bypassPermissions",
                    max_turns=max_turns,
                    model=model,
                    system_prompt=system_prompt,
                    env=child_env,
                ),
            ):
                # --- AssistantMessage: Claude's own text (reasoning, status).
                # Routed to narrative channel ONLY so it never contaminates
                # the stdout used for metric extraction.
                if isinstance(msg, AssistantMessage):
                    for block in getattr(msg, "content", []):
                        if hasattr(block, "text"):
                            narrative_parts.append(block.text)

                # --- UserMessage: tool-result blocks (real Bash stdout/stderr)
                elif isinstance(msg, UserMessage):
                    for block in getattr(msg, "content", []):
                        content = getattr(block, "content", None)
                        if isinstance(content, str):
                            stdout_parts.append(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    stdout_parts.append(item["text"])
                        # Track error status from tool results
                        if getattr(block, "is_error", False):
                            stderr_parts.append(
                                content if isinstance(content, str) else str(content)
                            )
                            last_exit_code = 1

                # --- ResultMessage: final summary — narrative, not stdout.
                elif isinstance(msg, ResultMessage):
                    if getattr(msg, "result", None):
                        narrative_parts.append(msg.result)
                    subtype = getattr(msg, "subtype", "")
                    if subtype and subtype != "success":
                        last_exit_code = 1

            return ClaudeResult(
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
                returncode=last_exit_code,
                narrative="\n".join(narrative_parts),
            )

        try:
            return asyncio.run(
                asyncio.wait_for(_execute(), timeout=float(timeout_sec))
            )
        except asyncio.TimeoutError:
            return ClaudeResult(
                stdout="",
                stderr=f"Claude Code agent timed out after {timeout_sec}s",
                returncode=1,
            )
        except Exception as exc:
            logger.exception("Claude Code agent error")
            return ClaudeResult(
                stdout="",
                stderr=f"Claude Code agent error: {exc}",
                returncode=1,
            )

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
