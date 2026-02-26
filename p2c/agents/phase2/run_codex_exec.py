from __future__ import annotations

import os
import shlex
import time
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.codex_exec_support import (
    CodexBackgroundExecutor,
    CodexCapabilityGate,
    CodexFailureReporter,
    CodexOutputValidator,
    is_rate_limit_failure,
)
from p2c.agents.phase2.codex_prompt_templates import build_codex_main_prompt, build_codex_repair_prompt
from p2c.runtime.factory import ensure_runtime

SYSTEM_PROMPT = "You orchestrate Codex execution in sandbox with strict output contracts."
USER_PROMPT_TEMPLATE = "Input: task_spec + claims_ir. Output: /workspace/outputs/*.json"
DEFAULT_CODEX_MODEL = "gpt-5.1-codex-mini"


class RunCodexExecAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="run_codex_exec", *args, **kwargs)
        self.validator = CodexOutputValidator()
        self.reporter = CodexFailureReporter(self.artifacts, self.log)
        self.bg = CodexBackgroundExecutor(self.log)
        self.capability_gate = CodexCapabilityGate()

    @staticmethod
    def _build_codex_cmd(prompt: str, extra_args: list[str] | None = None) -> str:
        model = (os.getenv("P2C_CODEX_MODEL") or DEFAULT_CODEX_MODEL).strip()
        parts = ["codex", "exec"]
        if extra_args:
            parts.extend(extra_args)
        if model:
            parts.extend(["-m", model])
        parts.append(prompt)
        return " ".join(shlex.quote(x) for x in parts)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = str(item or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _run_stage(
        self,
        runtime,
        *,
        label: str,
        cmd: str,
        repo_dir: str,
        outputs_dir: str,
        workspace_root: str,
        timeout_sec: int,
        reason_codes: list[str],
        capability_snapshot: dict[str, Any] | None = None,
        dependency_bootstrap_trace: list[str] | None = None,
    ) -> dict:
        try:
            result = self.bg.run(
                runtime,
                cmd=cmd,
                cwd=repo_dir,
                outputs_dir=outputs_dir,
                label=label,
                timeout_sec=timeout_sec,
                workspace_root=workspace_root,
            )
        except Exception as e:  # noqa: BLE001
            self.reporter.handle_stage_exception(
                stage=label,
                cmd=cmd,
                error=e,
                runtime=runtime,
                outputs_dir=outputs_dir,
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            raise RuntimeError(
                f"run_codex_exec {label} stage failed: {e}. "
                "See artifacts/<run_id>/execution/codex_failure.json"
            ) from e

        self.artifacts.append_text(
            "execution/run.log",
            (
                f"\n# codex {label}\n"
                f"$ {cmd}\n"
                f"pid_path={result['pid_path']}\n"
                f"exit_path={result['exit_path']}\n"
                f"polls={result['polls']}\n"
                f"timed_out={result['timed_out']}\n"
                f"rc={result['rc']}\n"
            ),
        )
        return result

    def _record_nonzero_stage_rc(
        self,
        runtime,
        *,
        stage: str,
        stage_cmd: str,
        stage_result: dict,
        outputs_dir: str,
        reason_codes: list[str],
        capability_snapshot: dict[str, Any] | None = None,
        dependency_bootstrap_trace: list[str] | None = None,
    ) -> None:
        if stage_result["timed_out"]:
            reason_codes.append(f"CODEX_{stage.upper()}_TIMEOUT")
        if stage_result["rc"] == 0:
            return
        reason_codes.append(f"CODEX_{stage.upper()}_RC_{stage_result['rc']}")
        log_tail = self.reporter.safe_remote_log_tail(runtime, stage_result["log_path"])
        reason_codes.append(f"CODEX_{stage.upper()}_LOG_TAIL:{log_tail}")
        pip_diag = self.reporter.collect_pip_log_tail(runtime, outputs_dir)
        if pip_diag.get("has_conflict_signal", False):
            reason_codes.append("DEPENDENCY_INSTALL_CONFLICT")
        elif pip_diag.get("has_pip_activity", False):
            reason_codes.append("DEPENDENCY_INSTALL_ACTIVITY_DETECTED")
        self.reporter.write_failure_artifact(
            stage="main" if stage.startswith("main") else (stage if stage in {"precheck", "repair"} else "postcheck"),
            last_command=stage_cmd,
            exit_code=int(stage_result["rc"]),
            stdout_tail=str(stage_result.get("launch_stdout") or ""),
            stderr_tail=str(stage_result.get("launch_stderr") or ""),
            codex_exec_log_tail=log_tail,
            pip_log_tail=str(pip_diag.get("tail") or ""),
            reason_codes=reason_codes,
            capability_snapshot=capability_snapshot,
            dependency_bootstrap_trace=dependency_bootstrap_trace,
        )

    def _write_capability_artifacts(self, runtime, *, outputs_dir: str, capability_snapshot: dict[str, Any]) -> None:
        self.artifacts.write_json("execution/codex_outputs/capability_probe.json", capability_snapshot)
        runtime.write_text(
            f"{outputs_dir}/capability_probe.json",
            self.artifacts.path("execution/codex_outputs/capability_probe.json").read_text(
                encoding="utf-8", errors="ignore"
            ),
        )

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        runtime = ensure_runtime(ctx, self.artifacts)
        if (getattr(runtime, "backend_name", "") or "").lower() != "e2b":
            raise RuntimeError("run_codex_exec requires P2C_RUNTIME_BACKEND=e2b")

        required_ctx = [
            "workspace_root",
            "workspace_repo_dir",
            "workspace_outputs_dir",
            "workspace_inputs_dir",
        ]
        missing = [k for k in required_ctx if not ctx.get(k)]
        if missing:
            raise RuntimeError(f"run_codex_exec missing workspace context keys: {missing}")
        if ctx.get("_dep_gate_terminal"):
            raise RuntimeError("dependency gate previously failed in this run; see codex_failure.json")
        runtime_meta = runtime.metadata()

        workspace_root = str(ctx["workspace_root"])
        repo_dir = str(ctx["workspace_repo_dir"])
        outputs_dir = str(ctx["workspace_outputs_dir"])
        inputs_dir = str(ctx["workspace_inputs_dir"])
        max_iters = int(ctx.get("max_self_heal_iters", 2))
        budget_minutes = int(ctx.get("budget_minutes", 30))
        main_timeout = max(900, budget_minutes * 60 + 300)
        repair_timeout = min(900, max(300, main_timeout // 2))
        rate_limit_retries = int(os.getenv("P2C_RATE_LIMIT_RETRIES", "2"))
        rate_limit_backoff_sec = float(os.getenv("P2C_RATE_LIMIT_BACKOFF_SEC", "30"))
        rate_limit_backoff_multiplier = float(os.getenv("P2C_RATE_LIMIT_BACKOFF_MULTIPLIER", "2.0"))

        runtime.run_command(
            f"mkdir -p {shlex.quote(outputs_dir)} {shlex.quote(inputs_dir)}",
            cwd=workspace_root,
            timeout_sec=30,
        )

        key_probe = runtime.run_command("bash -lc 'test -n \"$OPENAI_API_KEY\"'", cwd=workspace_root, timeout_sec=20)
        if key_probe.rc != 0:
            self.reporter.write_failure_artifact(
                stage="precheck",
                last_command=key_probe.command,
                exit_code=key_probe.rc,
                stdout_tail=key_probe.stdout,
                stderr_tail=key_probe.stderr,
                codex_exec_log_tail="",
                pip_log_tail="",
                reason_codes=["PRECHECK_OPENAI_API_KEY_MISSING"],
            )
            raise RuntimeError("OPENAI_API_KEY is not available inside sandbox runtime environment")

        codex_probe = runtime.run_command("bash -lc 'command -v codex >/dev/null 2>&1'", cwd=workspace_root, timeout_sec=20)
        if codex_probe.rc != 0:
            self.reporter.write_failure_artifact(
                stage="precheck",
                last_command=codex_probe.command,
                exit_code=codex_probe.rc,
                stdout_tail=codex_probe.stdout,
                stderr_tail=codex_probe.stderr,
                codex_exec_log_tail="",
                pip_log_tail="",
                reason_codes=["PRECHECK_CODEX_CLI_MISSING"],
            )
            raise RuntimeError("codex CLI is not available inside sandbox (template mismatch or install issue)")

        reason_codes: list[str] = [
            "DEPENDENCY_SOLVER_STARTED",
            "CODEX_SKIP_GIT_FLAG_USED",
            "CODEX_DANGEROUS_BYPASS_USED",
        ]
        cmd_args: list[str] = [
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]

        capability_snapshot = self.capability_gate.probe_python_capabilities(runtime, workspace_root)
        capability_snapshot["runtime_metadata"] = runtime_meta
        reason_codes.extend(capability_snapshot.get("reason_codes", []))
        self._write_capability_artifacts(runtime, outputs_dir=outputs_dir, capability_snapshot=capability_snapshot)

        bootstrap = self.capability_gate.bootstrap_dependencies(
            runtime,
            repo_dir=repo_dir,
            outputs_dir=outputs_dir,
            workspace_root=workspace_root,
            capability_snapshot=capability_snapshot,
        )
        dependency_bootstrap_trace = list(bootstrap.get("trace") or [])
        capability_snapshot = dict(bootstrap.get("snapshot_after") or capability_snapshot)
        capability_snapshot["runtime_metadata"] = runtime_meta
        capability_snapshot["sudo_diagnostics"] = bootstrap.get("sudo_diag", {})
        reason_codes.extend(bootstrap.get("reason_codes") or [])
        reason_codes = self._dedupe(reason_codes)
        self._write_capability_artifacts(runtime, outputs_dir=outputs_dir, capability_snapshot=capability_snapshot)
        try:
            dep_log_text = runtime.read_text(f"{outputs_dir}/dependency_bootstrap.log")
            self.artifacts.write_text("execution/codex_outputs/dependency_bootstrap.log", dep_log_text)
        except Exception:  # noqa: BLE001
            self.artifacts.write_text("execution/codex_outputs/dependency_bootstrap.log", "")

        if not bool(bootstrap.get("ready")):
            reason_codes.append("DEPENDENCY_UNRESOLVED")
            probe = self.capability_gate.probe_entrypoints_once(
                runtime,
                repo_dir=repo_dir,
                task_spec_path=f"{inputs_dir}/task_spec.json",
            )
            run_rows = list(probe.get("runs") or [])
            worklog_events = list(bootstrap.get("worklog_events") or []) + list(probe.get("worklog_events") or [])
            self.capability_gate.render_fallback_outputs(
                runtime,
                outputs_dir=outputs_dir,
                claims_ir_path=f"{inputs_dir}/claims_ir.json",
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
                runs=run_rows,
                worklog_events=worklog_events,
                reason_codes=reason_codes,
                dependency_solver_payload=bootstrap.get("dependency_solver"),
            )
            self.reporter.write_failure_artifact(
                stage="precheck",
                last_command="capability_gate",
                exit_code=1,
                stdout_tail="",
                stderr_tail="Dependency bootstrap could not satisfy required runtime modules",
                codex_exec_log_tail=self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log"),
                pip_log_tail=str(self.reporter.collect_pip_log_tail(runtime, outputs_dir).get("tail") or ""),
                reason_codes=self._dedupe(reason_codes),
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            ctx["_dep_gate_terminal"] = True
            raise RuntimeError(
                "Dependency capability gate failed before codex main execution. "
                "Fallback outputs were written; see execution/codex_failure.json"
            )

        main_prompt = build_codex_main_prompt(
            max_self_heal_iters=max_iters,
            repo_dir=repo_dir,
            inputs_task_spec=f"{inputs_dir}/task_spec.json",
            inputs_claims_ir=f"{inputs_dir}/claims_ir.json",
            outputs_dir=outputs_dir,
        )
        main_cmd = self._build_codex_cmd(main_prompt, extra_args=cmd_args)
        main_result: dict[str, Any] | None = None
        for attempt in range(rate_limit_retries + 1):
            label = "main" if attempt == 0 else f"main_retry_{attempt}"
            main_result = self._run_stage(
                runtime,
                label=label,
                cmd=main_cmd,
                repo_dir=repo_dir,
                outputs_dir=outputs_dir,
                workspace_root=workspace_root,
                timeout_sec=main_timeout,
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            self._record_nonzero_stage_rc(
                runtime,
                stage=label,
                stage_cmd=main_cmd,
                stage_result=main_result,
                outputs_dir=outputs_dir,
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            if int(main_result.get("rc", 1)) == 0:
                break
            stage_tail = self.reporter.safe_remote_log_tail(runtime, str(main_result.get("log_path") or ""))
            if is_rate_limit_failure(stage_tail) and attempt < rate_limit_retries:
                sleep_sec = rate_limit_backoff_sec * (rate_limit_backoff_multiplier**attempt)
                reason_codes.append(f"CODEX_RATE_LIMIT_BACKOFF_RETRY_{attempt + 1}")
                self.log("PROGRESS", f"rate-limit detected; backoff {sleep_sec:.1f}s before retry {attempt + 1}")
                time.sleep(sleep_sec)
                continue
            if is_rate_limit_failure(stage_tail):
                reason_codes.append("CODEX_RATE_LIMIT_BACKOFF_EXHAUSTED")
            break

        self.artifacts.append_text(
            "execution/run.log",
            (
                f"workspace_root={workspace_root}\n"
                f"workspace_repo_dir={repo_dir}\n"
                f"workspace_inputs_dir={inputs_dir}\n"
                f"workspace_outputs_dir={outputs_dir}\n"
            ),
        )

        ready, output_issues = self.validator.outputs_ready(runtime, outputs_dir)
        reason_codes.extend(output_issues[:8])
        if self.validator.outputs_missing(output_issues):
            reason_codes.append("CODEX_OUTPUTS_MISSING_AFTER_MAIN")

        if not ready:
            repair_cmd = self._build_codex_cmd(
                build_codex_repair_prompt(outputs_dir),
                extra_args=cmd_args,
            )
            repair_result = self._run_stage(
                runtime,
                label="repair",
                cmd=repair_cmd,
                repo_dir=repo_dir,
                outputs_dir=outputs_dir,
                workspace_root=workspace_root,
                timeout_sec=repair_timeout,
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            self._record_nonzero_stage_rc(
                runtime,
                stage="repair",
                stage_cmd=repair_cmd,
                stage_result=repair_result,
                outputs_dir=outputs_dir,
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )

        ready, output_issues = self.validator.outputs_ready(runtime, outputs_dir)
        reason_codes.extend(output_issues[:8])
        reason_codes = self._dedupe(reason_codes)
        total_runs, success_runs, dep_failed_runs = self.validator.dependency_failure_count(runtime, outputs_dir)
        if dep_failed_runs > 0:
            reason_codes.append("ENTRYPOINT_UNRUNNABLE_DEPENDENCY")
        if self.validator.all_entrypoints_unrunnable_due_dependency(runtime, outputs_dir):
            reason_codes.append("DEPENDENCY_UNRESOLVED")
            log_tail = self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_main.log")
            pip_diag = self.reporter.collect_pip_log_tail(runtime, outputs_dir)
            self.reporter.write_failure_artifact(
                stage="postcheck",
                last_command=main_cmd,
                exit_code=1,
                stdout_tail="",
                stderr_tail="",
                codex_exec_log_tail=log_tail,
                pip_log_tail=str(pip_diag.get("tail") or ""),
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            raise RuntimeError(
                f"Codex execution failed due to unresolved dependencies across all entrypoints; "
                f"runs={total_runs} success={success_runs} dep_failed={dep_failed_runs} "
                f"reason_codes={reason_codes}"
            )

        if not ready:
            log_tail = self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log")
            pip_diag = self.reporter.collect_pip_log_tail(runtime, outputs_dir)
            if pip_diag.get("has_conflict_signal", False):
                reason_codes.append("DEPENDENCY_INSTALL_CONFLICT")
            elif pip_diag.get("has_pip_activity", False):
                reason_codes.append("DEPENDENCY_INSTALL_ACTIVITY_DETECTED")
            self.reporter.write_failure_artifact(
                stage="postcheck",
                last_command=main_cmd,
                exit_code=1,
                stdout_tail="",
                stderr_tail="",
                codex_exec_log_tail=log_tail,
                pip_log_tail=str(pip_diag.get("tail") or ""),
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            raise RuntimeError(
                f"Codex execution completed but required outputs are missing or invalid under {outputs_dir}; "
                f"reason_codes={reason_codes}; see artifacts/<run_id>/execution/codex_failure.json"
            )

        return {
            "codex_exec": {
                "reason_codes": self._dedupe(reason_codes),
                "workspace_outputs_dir": outputs_dir,
            }
        }
