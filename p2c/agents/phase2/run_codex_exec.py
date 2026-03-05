from __future__ import annotations

import json
import os
import re
import shlex
import time
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.codex_exec_support import (
    CodexBackgroundExecutor,
    CodexCapabilityGate,
    CodexFailureReporter,
    extract_task_items,
    is_rate_limit_failure,
    timeout_for_class,
)
from p2c.agents.phase2.codex_prompt_templates import (
    build_codex_single_task_prompt,
    build_codex_single_task_repair_prompt,
)
from p2c.runtime.factory import ensure_runtime

SYSTEM_PROMPT = "You orchestrate Codex execution in sandbox with strict output contracts."
USER_PROMPT_TEMPLATE = "Input: task_spec. Output: task execution artifacts under /workspace/outputs."
DEFAULT_CODEX_MODEL = "gpt-5.1"


class RunCodexExecAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="run_codex_exec", *args, **kwargs)
        self.reporter = CodexFailureReporter(self.artifacts, self.log)
        self.bg = CodexBackgroundExecutor(self.log, artifacts=self.artifacts)
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
    def _prepend_path(cmd: str, workspace_bin_dir: str) -> str:
        path_q = shlex.quote(workspace_bin_dir)
        return f"PATH={path_q}:$PATH {cmd}"

    def _probe_toolchain(self, runtime, *, workspace_root: str, workspace_bin_dir: str) -> dict[str, Any]:
        tools = ["python", "python3", "pip", "pip3", "apply_patch"]
        paths: dict[str, str | None] = {}
        versions: dict[str, str] = {}
        reason_codes: list[str] = []
        for tool in tools:
            cmd = "bash -lc " + shlex.quote(
                f"PATH={workspace_bin_dir}:$PATH; command -v {tool} 2>/dev/null || true"
            )
            probe = runtime.run_command(cmd, cwd=workspace_root, timeout_sec=20)
            tool_path = (probe.stdout or "").strip().splitlines()[0].strip() if (probe.stdout or "").strip() else ""
            paths[tool] = tool_path or None
            if not tool_path:
                reason_codes.append(f"TOOL_MISSING_{tool.upper()}")
                continue
            version_flag = "-V" if tool in {"python", "python3"} else "--version"
            v_cmd = "bash -lc " + shlex.quote(
                f"PATH={workspace_bin_dir}:$PATH; {tool} {version_flag} 2>&1 || true"
            )
            v_probe = runtime.run_command(v_cmd, cwd=workspace_root, timeout_sec=20)
            version_text = (v_probe.stdout or v_probe.stderr or "").strip()
            versions[tool] = version_text[-300:] if version_text else ""
        return {
            "paths": paths,
            "versions": versions,
            "path_prefix": workspace_bin_dir,
            "reason_codes": self._dedupe(reason_codes),
        }

    def _write_toolchain_artifacts(self, runtime, *, outputs_dir: str, toolchain_probe: dict[str, Any]) -> None:
        self.artifacts.write_json("execution/codex_outputs/toolchain_probe.json", toolchain_probe)
        runtime.write_text(
            f"{outputs_dir}/toolchain_probe.json",
            self.artifacts.path("execution/codex_outputs/toolchain_probe.json").read_text(
                encoding="utf-8", errors="ignore"
            ),
        )

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

    @staticmethod
    def _safe_label(value: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_\-.]", "_", str(value or "task"))
        s = s.strip("_.")
        return s or "task"

    @staticmethod
    def _normalize_run(
        row: dict[str, Any],
        *,
        default_task: dict[str, Any] | None = None,
        default_rc: int = 1,
        default_reason: list[str] | None = None,
    ) -> dict[str, Any]:
        task = default_task or {}
        command = str(row.get("command") or task.get("command") or "").strip()
        task_id = str(row.get("task_id") or row.get("run_id") or task.get("task_id") or "task_unknown")
        entrypoint = str(row.get("entrypoint") or task.get("entrypoint") or "")
        rc_raw = row.get("exit_code", default_rc)
        try:
            rc = int(rc_raw)
        except Exception:  # noqa: BLE001
            rc = default_rc

        status = str(row.get("status") or ("ok" if rc == 0 else "failed"))
        if rc == 124:
            status = "timeout"

        stdout_tail = str(row.get("stdout_tail") or "")
        stderr_tail = str(row.get("stderr_tail") or "")

        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
        reason_codes = row.get("reason_codes")
        if not isinstance(reason_codes, list):
            reason_codes = []
        if default_reason:
            reason_codes = list(reason_codes) + list(default_reason)

        # Normalize legacy TF1 API failures under TF2 runtime so downstream analysis
        # is stable even when Codex output is incomplete.
        combined_tail = f"{stderr_tail}\n{stdout_tail}".lower()
        tf1_markers = (
            "tf.placeholder",
            "tf.set_random_seed",
            "tf.contrib",
            "no attribute 'placeholder'",
            "no attribute 'set_random_seed'",
            "no attribute 'contrib'",
        )
        if rc != 0 and any(marker in combined_tail for marker in tf1_markers):
            if status != "timeout":
                status = "failed_dependency"
            reason_codes = list(reason_codes) + ["TF1_API_INCOMPATIBLE_WITH_TF2"]

        deduped_reasons: list[str] = []
        seen_reasons: set[str] = set()
        for rc_item in reason_codes:
            key = str(rc_item).strip()
            if not key or key in seen_reasons:
                continue
            seen_reasons.add(key)
            deduped_reasons.append(key)

        return {
            "run_id": task_id,
            "task_id": task_id,
            "entrypoint": entrypoint,
            "command": command,
            "params": {
                "timeout_class": str(task.get("timeout_class") or "medium"),
                "hyperparams": dict(task.get("hyperparams") or {}),
            },
            "cwd": str(row.get("cwd") or ""),
            "exit_code": rc,
            "status": status,
            "runtime_sec": float(row.get("runtime_sec") or 0.0),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "artifacts": artifacts,
            "metrics": metrics,
            "reason_codes": deduped_reasons,
        }

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
        local_stream_path: str | None = None,
        stream_sync_every_sec: int = 20,
    ) -> dict[str, Any]:
        try:
            result = self.bg.run(
                runtime,
                cmd=cmd,
                cwd=repo_dir,
                outputs_dir=outputs_dir,
                label=label,
                timeout_sec=timeout_sec,
                workspace_root=workspace_root,
                local_stream_path=local_stream_path,
                stream_sync_every_sec=stream_sync_every_sec,
                stream_flush_on_exit=True,
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

    def _write_capability_artifacts(self, runtime, *, outputs_dir: str, capability_snapshot: dict[str, Any]) -> None:
        self.artifacts.write_json("execution/codex_outputs/capability_probe.json", capability_snapshot)
        runtime.write_text(
            f"{outputs_dir}/capability_probe.json",
            self.artifacts.path("execution/codex_outputs/capability_probe.json").read_text(
                encoding="utf-8", errors="ignore"
            ),
        )

    def _load_remote_json(self, runtime, path: str) -> dict[str, Any]:
        try:
            obj = json.loads(runtime.read_text(path))
        except Exception:  # noqa: BLE001
            return {}
        return obj if isinstance(obj, dict) else {}

    def _build_single_task_spec(self, task_spec: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        return {
            "tasks": [task],
            "constraints": dict(task_spec.get("constraints") or {}),
            "entrypoints": [],
            "metric_observers": list(task_spec.get("metric_observers") or []),
            "run_matrix": [task],
            "selection_notes": list(task_spec.get("selection_notes") or []) + [
                f"single_task_session:{task.get('task_id', '')}"
            ],
            "reason_codes": self._dedupe(list(task_spec.get("reason_codes") or []) + ["TASK_SERIAL_MODE"]),
        }

    def _collect_single_task_run(
        self,
        runtime,
        *,
        outputs_dir: str,
        task: dict[str, Any],
        fallback_rc: int,
        fallback_reason: list[str] | None = None,
    ) -> dict[str, Any]:
        task_id = str(task.get("task_id") or "")
        payload = self._load_remote_json(runtime, f"{outputs_dir}/task_run_results.json")
        runs = payload.get("runs")
        candidate: dict[str, Any] | None = None
        if isinstance(runs, list):
            for row in runs:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("task_id") or row.get("run_id") or "")
                if key == task_id:
                    candidate = row
        if candidate is None:
            # Backward-compat: some Codex runs only emit run_manifest.json.
            manifest = self._load_remote_json(runtime, f"{outputs_dir}/run_manifest.json")
            manifest_runs = manifest.get("runs")
            if isinstance(manifest_runs, list):
                for row in manifest_runs:
                    if not isinstance(row, dict):
                        continue
                    key = str(row.get("task_id") or row.get("run_id") or "")
                    if key == task_id:
                        candidate = row
                        break
                if candidate is None and len(manifest_runs) == 1 and isinstance(manifest_runs[0], dict):
                    candidate = manifest_runs[0]
        if candidate is None:
            return self._normalize_run(
                {
                    "run_id": task_id,
                    "task_id": task_id,
                    "entrypoint": task.get("entrypoint"),
                    "command": task.get("command"),
                    "exit_code": fallback_rc,
                    "status": "timeout" if int(fallback_rc) == 124 else ("ok" if int(fallback_rc) == 0 else "failed"),
                    "reason_codes": ["TASK_RESULT_MISSING_FROM_CODEX"],
                },
                default_task=task,
                default_rc=fallback_rc,
                default_reason=fallback_reason,
            )
        return self._normalize_run(
            candidate,
            default_task=task,
            default_rc=fallback_rc,
            default_reason=fallback_reason,
        )

    def _build_claim_alignment_local(
        self,
        *,
        claims_payload: dict[str, Any],
        run_rows: list[dict[str, Any]],
        reason_codes: list[str],
    ) -> dict[str, Any]:
        metric_sources: dict[str, list[str]] = {}
        for run in run_rows:
            if int(run.get("exit_code", 1)) != 0:
                continue
            run_id = str(run.get("run_id") or "")
            metrics = run.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            for key in metrics.keys():
                m = str(key).strip().lower()
                if not m:
                    continue
                metric_sources.setdefault(m, []).append(f"execution/codex_outputs/run_manifest.json:{run_id}")

        claims = claims_payload.get("claims")
        if not isinstance(claims, list):
            claims = []

        rows: list[dict[str, Any]] = []
        for idx, claim in enumerate(claims):
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id") or f"claim_{idx+1:02d}")
            metric = str(claim.get("metric") or "").strip().lower()
            required_metrics = [metric] if metric else []
            code_verifiable = bool(claim.get("code_verifiable", not claim.get("unverifiable_from_paper", False)))
            if not code_verifiable:
                rows.append(
                    {
                        "claim_id": claim_id,
                        "required_metrics": required_metrics,
                        "source": [],
                        "evaluable": "no",
                        "reason": "claim marked non-code-verifiable in phase1",
                    }
                )
                continue
            if not required_metrics:
                rows.append(
                    {
                        "claim_id": claim_id,
                        "required_metrics": [],
                        "source": ["execution/codex_outputs/run_manifest.json"],
                        "evaluable": "partial",
                        "reason": "claim has no metric mapping",
                    }
                )
                continue
            sources = metric_sources.get(metric, [])
            rows.append(
                {
                    "claim_id": claim_id,
                    "required_metrics": required_metrics,
                    "source": sources or ["execution/codex_outputs/run_manifest.json"],
                    "evaluable": "yes" if sources else "no",
                    "reason": "metric observed in task runs" if sources else "metric not observed in task runs",
                }
            )

        return {"claims": rows, "reason_codes": self._dedupe(reason_codes)}

    def _write_runner_outputs(
        self,
        runtime,
        *,
        outputs_dir: str,
        run_manifest: dict[str, Any],
        claim_alignment: dict[str, Any],
        worklog_events: list[dict[str, Any]],
        stream_local_path: str,
    ) -> None:
        self.artifacts.write_json("execution/codex_outputs/run_manifest.json", run_manifest)
        self.artifacts.write_json("execution/codex_outputs/claim_alignment.json", claim_alignment)
        worklog_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in worklog_events if isinstance(x, dict)).strip()
        if worklog_text:
            worklog_text += "\n"
        self.artifacts.write_text("execution/codex_outputs/codex_worklog.jsonl", worklog_text)

        if not self.artifacts.path(stream_local_path).exists():
            self.artifacts.write_text(stream_local_path, "")

        try:
            patch_text = runtime.read_text(f"{outputs_dir}/patches.diff")
        except Exception:  # noqa: BLE001
            patch_text = ""
        self.artifacts.write_text("execution/codex_outputs/patches.diff", patch_text)
        runtime.write_text(f"{outputs_dir}/run_manifest.json", json.dumps(run_manifest, ensure_ascii=False, indent=2))
        runtime.write_text(f"{outputs_dir}/claim_alignment.json", json.dumps(claim_alignment, ensure_ascii=False, indent=2))
        runtime.write_text(f"{outputs_dir}/codex_worklog.jsonl", worklog_text)
        runtime.write_text(f"{outputs_dir}/patches.diff", patch_text)

        for name in [
            "codex_exec.log",
            "codex_main.log",
            "codex_repair.log",
            "dependency_solver.json",
            "pip_install.log",
            "task_run_results.json",
        ]:
            try:
                self.artifacts.write_text(f"execution/codex_outputs/{name}", runtime.read_text(f"{outputs_dir}/{name}"))
            except Exception:  # noqa: BLE001
                if name.endswith(".json"):
                    self.artifacts.write_json(
                        f"execution/codex_outputs/{name}",
                        {"reason_codes": ["MISSING_REMOTE_OUTPUT"], "source": f"{outputs_dir}/{name}"},
                    )
                else:
                    self.artifacts.write_text(f"execution/codex_outputs/{name}", "")

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
        workspace_bin_dir = str(ctx.get("workspace_bin_dir") or f"{workspace_root}/bin")

        budget_minutes = int(ctx.get("budget_minutes", 30))
        global_timeout_sec = min(45 * 60, max(900, budget_minutes * 60 + 300))
        repair_timeout = min(900, max(300, global_timeout_sec // 2))
        global_deadline = time.time() + global_timeout_sec
        runtime_started_at = float(ctx.get("_runtime_started_at") or 0.0)
        runtime_timeout_sec = int(runtime_meta.get("timeout_sec") or 0)

        # Task-level scheduling and retries.
        task_continue_on_failure = (os.getenv("P2C_TASK_CONTINUE_ON_FAILURE") or "1").strip() != "0"
        task_rate_limit_retries = int(os.getenv("P2C_TASK_RATE_LIMIT_RETRIES", os.getenv("P2C_RATE_LIMIT_RETRIES", "1")))
        task_rate_limit_backoff_sec = float(
            os.getenv("P2C_TASK_RATE_LIMIT_BACKOFF_SEC", os.getenv("P2C_RATE_LIMIT_BACKOFF_SEC", "60"))
        )
        task_rate_limit_backoff_multiplier = float(
            os.getenv(
                "P2C_TASK_RATE_LIMIT_BACKOFF_MULTIPLIER",
                os.getenv("P2C_RATE_LIMIT_BACKOFF_MULTIPLIER", "2.0"),
            )
        )

        # Local stream sync settings.
        stream_sync_enable = (os.getenv("P2C_STREAM_SYNC_ENABLE") or "1").strip() != "0"
        stream_sync_every_sec = int(os.getenv("P2C_STREAM_SYNC_INTERVAL_SEC", "20"))
        stream_local_path = (os.getenv("P2C_STREAM_LOCAL_PATH") or "execution/codex_outputs/codex_exec.stream.log").strip()
        if not stream_local_path or stream_local_path.startswith("/"):
            stream_local_path = "execution/codex_outputs/codex_exec.stream.log"

        runtime.run_command(
            f"mkdir -p {shlex.quote(outputs_dir)} {shlex.quote(inputs_dir)}",
            cwd=workspace_root,
            timeout_sec=30,
        )
        self.artifacts.write_text(stream_local_path, "")

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

        codex_probe = runtime.run_command(
            "bash -lc " + shlex.quote(f"PATH={workspace_bin_dir}:$PATH; command -v codex >/dev/null 2>&1"),
            cwd=workspace_root,
            timeout_sec=20,
        )
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

        toolchain_probe = self._probe_toolchain(
            runtime,
            workspace_root=workspace_root,
            workspace_bin_dir=workspace_bin_dir,
        )
        self._write_toolchain_artifacts(runtime, outputs_dir=outputs_dir, toolchain_probe=toolchain_probe)

        reason_codes: list[str] = [
            "RUNNER_TASK_ONLY_MODE",
            "CLAIMS_LOCAL_ONLY",
            "DEPENDENCY_SOLVER_STARTED",
            "CODEX_SKIP_GIT_FLAG_USED",
            "CODEX_DANGEROUS_BYPASS_USED",
            "P2C_TASK_SERIAL_MODE",
        ]
        reason_codes.extend(toolchain_probe.get("reason_codes") or [])
        if stream_sync_enable:
            reason_codes.append("STREAM_SYNC_ENABLED")

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

        claims_payload = self.artifacts.read_json("fingerprint/claims_ir.json")
        if not isinstance(claims_payload.get("claims"), list) or not claims_payload.get("claims"):
            remote_claims = self._load_remote_json(runtime, f"{inputs_dir}/claims_ir.json")
            if isinstance(remote_claims.get("claims"), list) and remote_claims.get("claims"):
                claims_payload = remote_claims
        task_spec_payload = self.artifacts.read_json("task/task_spec.json")

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
                claims_payload=claims_payload,
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

        if runtime_started_at > 0 and runtime_timeout_sec > 0:
            min_lifetime_before_task = int(os.getenv("P2C_SANDBOX_MIN_LIFETIME_BEFORE_TASK_SEC", "480"))
            elapsed = max(0.0, time.time() - runtime_started_at)
            remaining_lifetime = max(0.0, float(runtime_timeout_sec) - elapsed)
            self.artifacts.append_text(
                "execution/run.log",
                (
                    "[run_codex_exec] sandbox_lifetime_check "
                    f"timeout_sec={runtime_timeout_sec} elapsed_sec={elapsed:.1f} "
                    f"remaining_sec={remaining_lifetime:.1f} threshold_sec={min_lifetime_before_task}\n"
                ),
            )
            if remaining_lifetime < float(min_lifetime_before_task):
                reason_codes.append("SANDBOX_LIFETIME_TOO_LOW_BEFORE_TASK")
                self.reporter.write_failure_artifact(
                    stage="precheck",
                    last_command="sandbox_lifetime_guard",
                    exit_code=124,
                    stdout_tail="",
                    stderr_tail=(
                        f"remaining sandbox lifetime {remaining_lifetime:.1f}s below "
                        f"threshold {min_lifetime_before_task}s"
                    ),
                    codex_exec_log_tail=self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log"),
                    pip_log_tail=str(self.reporter.collect_pip_log_tail(runtime, outputs_dir).get("tail") or ""),
                    reason_codes=self._dedupe(reason_codes),
                    capability_snapshot=capability_snapshot,
                    dependency_bootstrap_trace=dependency_bootstrap_trace,
                )
                raise RuntimeError(
                    "Sandbox lifetime guard failed before task execution. "
                    "See execution/codex_failure.json"
                )

        tasks = extract_task_items(task_spec_payload)
        if not tasks:
            tasks = [
                {
                    "task_id": "legacy_task_01",
                    "entrypoint": "",
                    "command": "python3 -c 'print(\"phase2 task fallback\")'",
                    "timeout_class": "short",
                    "expected_metrics": [],
                    "hyperparams": {},
                }
            ]
            reason_codes.append("TASKS_MISSING_FALLBACK_SINGLE_SESSION")

        worklog_events: list[dict[str, Any]] = list(bootstrap.get("worklog_events") or [])
        run_rows: list[dict[str, Any]] = []
        saw_global_timeout = False
        last_task_cmd = ""

        for task in tasks:
            if time.time() >= global_deadline:
                saw_global_timeout = True
                reason_codes.append("GLOBAL_TIMEOUT_45M")
                break

            task_id = str(task.get("task_id") or f"task_{len(run_rows)+1:02d}")
            safe_task_id = self._safe_label(task_id)
            single_spec_path = f"{inputs_dir}/task_spec.single.{safe_task_id}.json"
            runtime.write_text(
                single_spec_path,
                json.dumps(self._build_single_task_spec(task_spec_payload, task), ensure_ascii=False, indent=2),
            )

            task_prompt = build_codex_single_task_prompt(
                repo_dir=repo_dir,
                inputs_task_spec=single_spec_path,
                inputs_metric_contract=f"{inputs_dir}/metric_contract.json",
                outputs_dir=outputs_dir,
                task_id=task_id,
            )
            task_cmd = self._prepend_path(
                self._build_codex_cmd(task_prompt, extra_args=cmd_args),
                workspace_bin_dir=workspace_bin_dir,
            )
            last_task_cmd = task_cmd

            task_timeout_cap = timeout_for_class(str(task.get("timeout_class") or "medium"), fallback_sec=600)
            remaining = max(1, int(global_deadline - time.time()))
            task_timeout = max(60, min(task_timeout_cap, remaining))

            worklog_events.append(
                {
                    "type": "run",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "details": {"stage": "main", "task_id": task_id, "command": task_cmd},
                    "result": {"status": "start", "timeout_sec": task_timeout},
                }
            )

            main_result: dict[str, Any] | None = None
            task_reason_codes: list[str] = []
            for attempt in range(task_rate_limit_retries + 1):
                label = f"task_{safe_task_id}_main" if attempt == 0 else f"task_{safe_task_id}_retry_{attempt}"
                main_result = self._run_stage(
                    runtime,
                    label=label,
                    cmd=task_cmd,
                    repo_dir=repo_dir,
                    outputs_dir=outputs_dir,
                    workspace_root=workspace_root,
                    timeout_sec=task_timeout,
                    reason_codes=reason_codes,
                    capability_snapshot=capability_snapshot,
                    dependency_bootstrap_trace=dependency_bootstrap_trace,
                    local_stream_path=stream_local_path if stream_sync_enable else None,
                    stream_sync_every_sec=stream_sync_every_sec,
                )
                if int(main_result.get("rc", 1)) == 0:
                    break

                stage_tail = self.reporter.safe_remote_log_tail(runtime, str(main_result.get("log_path") or ""))
                if is_rate_limit_failure(stage_tail):
                    task_reason_codes.append("STREAM_DISCONNECT_CONTINUE_POLL")
                    reason_codes.append("STREAM_DISCONNECT_CONTINUE_POLL")
                    if attempt < task_rate_limit_retries:
                        backoff = task_rate_limit_backoff_sec * (task_rate_limit_backoff_multiplier**attempt)
                        reason_codes.append(f"TASK_RATE_LIMIT_BACKOFF_RETRY_{attempt + 1}")
                        reason_codes.append(f"CODEX_RATE_LIMIT_BACKOFF_RETRY_{attempt + 1}")
                        sleep_sec = max(0.0, min(backoff, max(0.0, global_deadline - time.time() - 1.0)))
                        if sleep_sec > 0:
                            self.log("PROGRESS", f"task {task_id}: rate-limit detected; backoff {sleep_sec:.1f}s")
                            time.sleep(sleep_sec)
                        continue
                    reason_codes.append("TASK_RATE_LIMIT_BACKOFF_EXHAUSTED")
                    reason_codes.append("CODEX_RATE_LIMIT_BACKOFF_EXHAUSTED")
                break

            if main_result is None:
                main_result = {
                    "rc": 1,
                    "polls": 0,
                    "timed_out": False,
                    "pid_path": "",
                    "exit_path": "",
                    "log_path": "",
                    "launch_stdout": "",
                    "launch_stderr": "",
                }

            if bool(main_result.get("timed_out")):
                task_reason_codes.append("TASK_TIMEOUT")
                reason_codes.append("GLOBAL_TIMEOUT_45M")

            if int(main_result.get("rc", 1)) != 0 and not bool(main_result.get("timed_out")):
                repair_cmd = self._build_codex_cmd(
                    build_codex_single_task_repair_prompt(
                        outputs_dir=outputs_dir,
                        task_id=task_id,
                        task_spec_path=single_spec_path,
                    ),
                    extra_args=cmd_args,
                )
                repair_cmd = self._prepend_path(repair_cmd, workspace_bin_dir=workspace_bin_dir)
                remaining_after_main = max(1, int(global_deadline - time.time()))
                task_repair_timeout = max(60, min(repair_timeout, remaining_after_main))
                _ = self._run_stage(
                    runtime,
                    label=f"task_{safe_task_id}_repair",
                    cmd=repair_cmd,
                    repo_dir=repo_dir,
                    outputs_dir=outputs_dir,
                    workspace_root=workspace_root,
                    timeout_sec=task_repair_timeout,
                    reason_codes=reason_codes,
                    capability_snapshot=capability_snapshot,
                    dependency_bootstrap_trace=dependency_bootstrap_trace,
                    local_stream_path=stream_local_path if stream_sync_enable else None,
                    stream_sync_every_sec=stream_sync_every_sec,
                )

            row = self._collect_single_task_run(
                runtime,
                outputs_dir=outputs_dir,
                task=task,
                fallback_rc=int(main_result.get("rc", 1)),
                fallback_reason=task_reason_codes,
            )
            if bool(main_result.get("timed_out")):
                row["exit_code"] = 124
                row["status"] = "timeout"
                row["reason_codes"] = self._dedupe(list(row.get("reason_codes") or []) + ["TASK_TIMEOUT"])

            run_rows.append(row)
            worklog_events.append(
                {
                    "type": "run",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "details": {"stage": "main", "task_id": task_id},
                    "result": {
                        "status": "end",
                        "rc": int(main_result.get("rc", 1)),
                        "timed_out": bool(main_result.get("timed_out")),
                        "log_path": str(main_result.get("log_path") or ""),
                    },
                }
            )

            if int(row.get("exit_code", 1)) != 0 and not task_continue_on_failure:
                reason_codes.append("TASK_STOP_ON_FAILURE")
                break

        # If global timeout happened, mark pending tasks explicitly.
        if saw_global_timeout and len(run_rows) < len(tasks):
            for task in tasks[len(run_rows) :]:
                run_rows.append(
                    self._normalize_run(
                        {
                            "run_id": str(task.get("task_id") or "task_unknown"),
                            "task_id": str(task.get("task_id") or "task_unknown"),
                            "entrypoint": str(task.get("entrypoint") or ""),
                            "command": str(task.get("command") or ""),
                            "exit_code": 124,
                            "status": "timeout",
                            "reason_codes": ["TASK_SKIPPED_GLOBAL_TIMEOUT"],
                        },
                        default_task=task,
                        default_rc=124,
                    )
                )

        run_manifest = {
            "runs": run_rows,
            "reason_codes": self._dedupe(reason_codes),
        }
        claim_alignment = self._build_claim_alignment_local(
            claims_payload=claims_payload,
            run_rows=run_rows,
            reason_codes=reason_codes,
        )

        worklog_events.append(
            {
                "type": "output",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "details": {"stage": "runner_assemble"},
                "result": {
                    "files": [
                        "run_manifest.json",
                        "claim_alignment.json",
                        "codex_worklog.jsonl",
                        "patches.diff",
                        stream_local_path,
                    ]
                },
            }
        )

        self._write_runner_outputs(
            runtime,
            outputs_dir=outputs_dir,
            run_manifest=run_manifest,
            claim_alignment=claim_alignment,
            worklog_events=worklog_events,
            stream_local_path=stream_local_path,
        )

        success_count = sum(1 for row in run_rows if int(row.get("exit_code", 1)) == 0)
        dep_failed_count = 0
        for row in run_rows:
            if int(row.get("exit_code", 0)) == 0:
                continue
            status = str(row.get("status") or "").lower()
            row_reasons = [str(x).lower() for x in (row.get("reason_codes") or [])]
            row_text = " ".join([status] + row_reasons + [str(row.get("stderr_tail") or "").lower()])
            if "dependency" in row_text or "module not found" in row_text or "modulenotfounderror" in row_text:
                dep_failed_count += 1
        if run_rows and success_count == 0 and dep_failed_count == len(run_rows):
            reason_codes.append("DEPENDENCY_UNRESOLVED")
        run_manifest["reason_codes"] = self._dedupe(reason_codes)
        claim_alignment["reason_codes"] = self._dedupe(reason_codes)

        if saw_global_timeout or success_count == 0 or not run_rows:
            pip_diag = self.reporter.collect_pip_log_tail(runtime, outputs_dir)
            final_reason_codes = self._dedupe(reason_codes)
            if pip_diag.get("has_conflict_signal", False):
                final_reason_codes.append("DEPENDENCY_INSTALL_CONFLICT")
            elif pip_diag.get("has_pip_activity", False):
                final_reason_codes.append("DEPENDENCY_INSTALL_ACTIVITY_DETECTED")
            final_reason_codes = self._dedupe(final_reason_codes)
            self.reporter.write_failure_artifact(
                stage="main",
                last_command=last_task_cmd or "task_serial_execution",
                exit_code=124 if saw_global_timeout else 1,
                stdout_tail="",
                stderr_tail=(
                    "global timeout reached" if saw_global_timeout else "no successful tasks in serial execution"
                ),
                codex_exec_log_tail=self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log"),
                pip_log_tail=str(pip_diag.get("tail") or ""),
                reason_codes=final_reason_codes,
                capability_snapshot=capability_snapshot,
                dependency_bootstrap_trace=dependency_bootstrap_trace,
            )
            if saw_global_timeout:
                raise RuntimeError(
                    "Codex execution exceeded the 45-minute global timeout. "
                    "Partial artifacts were written to execution/codex_outputs/*"
                )
            raise RuntimeError(
                "Codex execution finished but no task succeeded. "
                "See execution/codex_failure.json and execution/codex_outputs/*"
            )

        return {
            "codex_exec": {
                "reason_codes": self._dedupe(reason_codes),
                "workspace_outputs_dir": outputs_dir,
                "run_count": len(run_rows),
                "success_count": success_count,
            }
        }
