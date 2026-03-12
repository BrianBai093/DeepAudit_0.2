from __future__ import annotations

import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.codex_exec_support import (
    CodexBackgroundExecutor,
    CodexCapabilityGate,
    CodexFailureReporter,
    CodexOutputValidator,
    extract_task_items,
    is_rate_limit_failure,
)
from p2c.agents.phase2.codex_prompt_templates import (
    build_autonomous_discovery_prompt,
    build_autonomous_execution_prompt,
    build_autonomous_repair_prompt,
)
from p2c.runtime.factory import ensure_runtime

SYSTEM_PROMPT = "You orchestrate autonomous Codex execution in E2B sandbox with two-stage discovery and execution."
USER_PROMPT_TEMPLATE = "Input: task_spec. Output: discovery summary + task execution artifacts under /workspace/outputs."
DEFAULT_CODEX_MODEL = "gpt-5.1"


class RunCodexExecAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="run_codex_exec", *args, **kwargs)
        self.reporter = CodexFailureReporter(self.artifacts, self.log)
        self.bg = CodexBackgroundExecutor(self.log, artifacts=self.artifacts)
        self.capability_gate = CodexCapabilityGate()
        self.validator = CodexOutputValidator()

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

    def _required_toolchain_checks(self, *, require_rscript: bool = False) -> list[tuple[str, str]]:
        checks = [
            ("python3", "python3 --version"),
            ("pip", "python3 -m pip --version"),
            ("poetry", "poetry --version"),
            ("uv", "uv --version"),
            ("node", "node --version"),
            ("npm", "npm --version"),
            ("codex", "codex --version"),
        ]
        if require_rscript:
            checks.append(("Rscript", "Rscript --version"))
        return checks

    def _bootstrap_toolchain(
        self,
        runtime,
        *,
        workspace_root: str,
        workspace_bin_dir: str,
        outputs_dir: str,
        install_rscript: bool = False,
    ) -> dict[str, Any]:
        log_remote = f"{outputs_dir}/dependency_bootstrap.log"
        rscript_block = ""
        verify_commands = [
            "python3 --version",
            "python3 -m pip --version",
            "poetry --version",
            "uv --version",
            "node --version",
            "npm --version",
            "codex --version",
        ]
        if install_rscript:
            rscript_block = f"""
if ! command -v Rscript >/dev/null 2>&1; then
  log "install Rscript via micromamba"
  MM_CACHE="$HOME/.cache/p2c/micromamba"
  MM_BIN="$HOME/.local/bin/micromamba"
  MM_ROOT="$HOME/.local/micromamba"
  mkdir -p "$MM_CACHE" "$HOME/.local/bin" "$MM_ROOT"
  if [ ! -x "$MM_BIN" ]; then
    if command -v curl >/dev/null 2>&1; then
      curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MM_CACHE" bin/micromamba >> {shlex.quote(log_remote)} 2>&1 || true
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MM_CACHE" bin/micromamba >> {shlex.quote(log_remote)} 2>&1 || true
    fi
    if [ -x "$MM_CACHE/bin/micromamba" ]; then
      cp "$MM_CACHE/bin/micromamba" "$MM_BIN" >> {shlex.quote(log_remote)} 2>&1 || true
      chmod +x "$MM_BIN" >> {shlex.quote(log_remote)} 2>&1 || true
    fi
  fi
  if [ -x "$MM_BIN" ]; then
    export MAMBA_ROOT_PREFIX="$MM_ROOT"
    "$MM_BIN" create -y -p "$MM_ROOT/envs/r-base" -c conda-forge r-base >> {shlex.quote(log_remote)} 2>&1 || true
    if [ -x "$MM_ROOT/envs/r-base/bin/Rscript" ]; then
      ln -sf "$MM_ROOT/envs/r-base/bin/Rscript" "$HOME/.local/bin/Rscript" >> {shlex.quote(log_remote)} 2>&1 || true
    fi
  fi
fi
"""
            verify_commands.append("Rscript --version")
        verify_lines = " \\\n".join(f'  "{cmd}"' for cmd in verify_commands)
        script = f"""
set -eu
export HOME="${{HOME:-/home/user}}"
export PATH="{workspace_bin_dir}:$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.local/bin" "$HOME/.cache/p2c" {shlex.quote(workspace_bin_dir)} {shlex.quote(outputs_dir)}
: > {shlex.quote(log_remote)}
log() {{
  printf '%s %s\\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> {shlex.quote(log_remote)}
}}
run_log() {{
  log "$1"
  shift
  "$@" >> {shlex.quote(log_remote)} 2>&1 || return $?
}}

log "bootstrap start"
if ! command -v python3 >/dev/null 2>&1; then
  log "python3 missing"
  exit 97
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
  log "install pip via ensurepip"
  python3 -m ensurepip --upgrade >> {shlex.quote(log_remote)} 2>&1 || true
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  log "install pip via get-pip.py"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$HOME/.cache/p2c/get-pip.py" >> {shlex.quote(log_remote)} 2>&1 || true
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$HOME/.cache/p2c/get-pip.py" https://bootstrap.pypa.io/get-pip.py >> {shlex.quote(log_remote)} 2>&1 || true
  fi
  if [ -f "$HOME/.cache/p2c/get-pip.py" ]; then
    python3 "$HOME/.cache/p2c/get-pip.py" --user >> {shlex.quote(log_remote)} 2>&1 || true
  fi
fi

if ! command -v uv >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
  log "install uv"
  python3 -m pip install --user uv >> {shlex.quote(log_remote)} 2>&1 || true
fi
if ! command -v poetry >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
  log "install poetry"
  python3 -m pip install --user poetry >> {shlex.quote(log_remote)} 2>&1 || true
fi
if ! command -v codex >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  log "install codex"
  npm install -g --prefix "$HOME/.local" @openai/codex >> {shlex.quote(log_remote)} 2>&1 || true
fi
{rscript_block}

for tool in python python3 pip pip3 poetry uv node npm codex Rscript apply_patch; do
  p="$(command -v "$tool" 2>/dev/null || true)"
  if [ -n "$p" ]; then
    ln -sf "$p" {shlex.quote(workspace_bin_dir)}/"$tool" >> {shlex.quote(log_remote)} 2>&1 || true
  fi
done

log "bootstrap verification"
for verify in \
{verify_lines}
do
  log "$verify"
  bash -lc "$verify" >> {shlex.quote(log_remote)} 2>&1 || true
done
log "bootstrap end"
"""
        result = runtime.run_command(
            "bash -lc " + shlex.quote(script),
            cwd=workspace_root,
            timeout_sec=15 * 60,
        )
        try:
            bootstrap_log = runtime.read_text(log_remote)
        except Exception:  # noqa: BLE001
            bootstrap_log = ""
        self.artifacts.write_text("execution/codex_outputs/dependency_bootstrap.log", bootstrap_log)
        return {
            "rc": int(result.rc),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "log_remote": log_remote,
            "reason_codes": [] if int(result.rc) == 0 else ["TOOLCHAIN_BOOTSTRAP_FAILED"],
        }

    def _probe_toolchain(self, runtime, *, workspace_root: str, workspace_bin_dir: str) -> dict[str, Any]:
        tools = ["python", "python3", "pip", "pip3", "poetry", "uv", "node", "npm", "codex", "Rscript", "apply_patch"]
        paths: dict[str, str | None] = {}
        versions: dict[str, str] = {}
        reason_codes: list[str] = []
        for tool in tools:
            if tool == "pip":
                v_cmd = "bash -lc " + shlex.quote(
                    f"PATH={workspace_bin_dir}:$PATH; python3 -m pip --version 2>&1 || true"
                )
                v_probe = runtime.run_command(v_cmd, cwd=workspace_root, timeout_sec=20)
                version_text = (v_probe.stdout or v_probe.stderr or "").strip()
                if version_text and "no module named pip" not in version_text.lower():
                    paths[tool] = "python3 -m pip"
                    versions[tool] = version_text[-300:]
                else:
                    paths[tool] = None
                    reason_codes.append("TOOL_MISSING_PIP")
                continue
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

    def _missing_required_tools(self, toolchain_probe: dict[str, Any], *, require_rscript: bool = False) -> list[str]:
        paths = toolchain_probe.get("paths") or {}
        missing: list[str] = []
        for tool, _ in self._required_toolchain_checks(require_rscript=require_rscript):
            tool_key = "pip" if tool == "pip" else tool
            if not str(paths.get(tool_key) or "").strip():
                missing.append(tool)
        return missing

    @staticmethod
    def _task_spec_requires_r(task_spec_payload: dict[str, Any]) -> bool:
        for section in ("tasks", "entrypoints"):
            rows = task_spec_payload.get(section)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in ("entrypoint", "command", "path", "runtime"):
                    value = str(row.get(key) or "")
                    lower = value.lower()
                    if lower.endswith(".r") or "rscript" in lower or ".r " in lower:
                        return True
        return False

    @staticmethod
    def _repo_requires_r(local_repo_dir: str) -> bool:
        repo_path = Path(str(local_repo_dir or ""))
        if not repo_path.exists():
            return False
        try:
            return any(repo_path.rglob("*.R"))
        except Exception:  # noqa: BLE001
            return False

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
                f"pid_path={result.get('pid_path', '')}\n"
                f"exit_path={result.get('exit_path', '')}\n"
                f"polls={result.get('polls', '')}\n"
                f"timed_out={result.get('timed_out', '')}\n"
                f"rc={result.get('rc', '')}\n"
            ),
        )
        return result

    def _load_remote_json(self, runtime, path: str) -> dict[str, Any]:
        try:
            obj = json.loads(runtime.read_text(path))
        except Exception:  # noqa: BLE001
            return {}
        return obj if isinstance(obj, dict) else {}

    def _collect_autonomous_results(
        self,
        runtime,
        *,
        outputs_dir: str,
        tasks: list[dict[str, Any]],
        fallback_rc: int,
    ) -> list[dict[str, Any]]:
        """Collect task results from task_run_results.json written by Codex Stage 2."""
        payload = self._load_remote_json(runtime, f"{outputs_dir}/task_run_results.json")
        runs = payload.get("runs")
        run_rows: list[dict[str, Any]] = []

        if isinstance(runs, list) and runs:
            for row in runs:
                if not isinstance(row, dict):
                    continue
                task_id = str(row.get("task_id") or row.get("run_id") or "")
                matching_task = next((t for t in tasks if str(t.get("task_id") or "") == task_id), None)
                run_rows.append(self._normalize_run(row, default_task=matching_task, default_rc=fallback_rc))

        # For any tasks not represented in output, add a fallback row.
        seen_ids = {str(r.get("task_id") or "") for r in run_rows}
        for task in tasks:
            tid = str(task.get("task_id") or "")
            if tid and tid not in seen_ids:
                run_rows.append(
                    self._normalize_run(
                        {
                            "task_id": tid,
                            "entrypoint": task.get("entrypoint"),
                            "command": task.get("command"),
                            "exit_code": fallback_rc,
                            "status": "timeout" if fallback_rc == 124 else "failed",
                            "reason_codes": ["TASK_RESULT_MISSING_FROM_CODEX"],
                        },
                        default_task=task,
                        default_rc=fallback_rc,
                    )
                )

        # Fallback: if no tasks were defined and nothing collected, create a single placeholder.
        if not run_rows:
            manifest = self._load_remote_json(runtime, f"{outputs_dir}/run_manifest.json")
            manifest_runs = manifest.get("runs")
            if isinstance(manifest_runs, list):
                for row in manifest_runs:
                    if isinstance(row, dict):
                        run_rows.append(self._normalize_run(row, default_rc=fallback_rc))

        return run_rows

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
            "discovery_summary.json",
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

        workspace_root = str(ctx["workspace_root"])
        repo_dir = str(ctx["workspace_repo_dir"])
        outputs_dir = str(ctx["workspace_outputs_dir"])
        inputs_dir = str(ctx["workspace_inputs_dir"])
        workspace_bin_dir = str(ctx.get("workspace_bin_dir") or f"{workspace_root}/bin")
        codex_skill_remote = str(ctx.get("workspace_codex_skill_remote") or "").strip()

        budget_minutes = int(ctx.get("budget_minutes", 30))
        global_timeout_sec = min(45 * 60, max(900, budget_minutes * 60 + 300))
        global_deadline = time.time() + global_timeout_sec

        # Stream sync settings.
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

        task_spec_local_artifact = str(ctx.get("workspace_task_spec_local_artifact") or "").strip()
        task_spec_payload = (
            self.artifacts.read_json(task_spec_local_artifact)
            if task_spec_local_artifact
            else self.artifacts.read_json("task/task_spec.json")
        )
        require_rscript = self._task_spec_requires_r(task_spec_payload) or self._repo_requires_r(ctx.get("repo_dir", ""))

        bootstrap_result = self._bootstrap_toolchain(
            runtime,
            workspace_root=workspace_root,
            workspace_bin_dir=workspace_bin_dir,
            outputs_dir=outputs_dir,
            install_rscript=require_rscript,
        )

        # --- Pre-flight checks ---
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

        # Toolchain diagnostic probe (non-blocking).
        toolchain_probe = self._probe_toolchain(
            runtime,
            workspace_root=workspace_root,
            workspace_bin_dir=workspace_bin_dir,
        )
        self._write_toolchain_artifacts(runtime, outputs_dir=outputs_dir, toolchain_probe=toolchain_probe)
        missing_required_tools = self._missing_required_tools(toolchain_probe, require_rscript=require_rscript)
        if missing_required_tools:
            self.reporter.write_failure_artifact(
                stage="precheck",
                last_command="toolchain_probe",
                exit_code=1,
                stdout_tail="",
                stderr_tail=f"missing required sandbox toolchain: {', '.join(missing_required_tools)}",
                codex_exec_log_tail="",
                pip_log_tail=self.artifacts.path("execution/codex_outputs/dependency_bootstrap.log").read_text(
                    encoding="utf-8", errors="ignore"
                )[-1200:],
                reason_codes=[f"PRECHECK_TOOL_MISSING_{tool.upper()}" for tool in missing_required_tools],
            )
            raise RuntimeError(
                "sandbox toolchain bootstrap incomplete; missing required commands: "
                + ", ".join(missing_required_tools)
            )

        # Capability diagnostic probe (non-blocking).
        capability_snapshot = self.capability_gate.probe_python_capabilities(runtime, workspace_root)
        self.artifacts.write_json("execution/codex_outputs/capability_probe.json", capability_snapshot)

        reason_codes: list[str] = [
            "AUTONOMOUS_EXECUTION_MODE",
            "CODEX_SKIP_GIT_FLAG_USED",
            "CODEX_DANGEROUS_BYPASS_USED",
        ]
        if require_rscript:
            reason_codes.append("RUNTIME_REQUIRES_RSCRIPT")
        reason_codes.extend(bootstrap_result.get("reason_codes") or [])
        reason_codes.extend(toolchain_probe.get("reason_codes") or [])
        reason_codes.extend(capability_snapshot.get("reason_codes") or [])
        if stream_sync_enable:
            reason_codes.append("STREAM_SYNC_ENABLED")

        cmd_args: list[str] = [
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]

        # Load task spec and claims for context.
        claims_payload = self.artifacts.read_json("fingerprint/claims_ir.json")
        if not isinstance(claims_payload.get("claims"), list) or not claims_payload.get("claims"):
            remote_claims = self._load_remote_json(runtime, f"{inputs_dir}/claims_ir.json")
            if isinstance(remote_claims.get("claims"), list) and remote_claims.get("claims"):
                claims_payload = remote_claims
        tasks = extract_task_items(task_spec_payload)

        worklog_events: list[dict[str, Any]] = []

        # === Stage 1: Autonomous Discovery ===
        discovery_timeout = max(300, int((global_deadline - time.time()) * 0.4))
        discovery_prompt = build_autonomous_discovery_prompt(
            repo_dir=repo_dir,
            outputs_dir=outputs_dir,
            skill_path=codex_skill_remote or None,
        )
        discovery_cmd = self._prepend_path(
            self._build_codex_cmd(discovery_prompt, extra_args=cmd_args),
            workspace_bin_dir=workspace_bin_dir,
        )

        worklog_events.append(
            {
                "type": "run",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "details": {"stage": "discovery", "command": discovery_cmd},
                "result": {"status": "start", "timeout_sec": discovery_timeout},
            }
        )

        self.log("PROGRESS", "Stage 1: Autonomous discovery and dependency installation...")
        discovery_result = self._run_stage(
            runtime,
            label="discovery_main",
            cmd=discovery_cmd,
            repo_dir=repo_dir,
            outputs_dir=outputs_dir,
            workspace_root=workspace_root,
            timeout_sec=discovery_timeout,
            reason_codes=reason_codes,
            capability_snapshot=capability_snapshot,
            local_stream_path=stream_local_path if stream_sync_enable else None,
            stream_sync_every_sec=stream_sync_every_sec,
        )

        worklog_events.append(
            {
                "type": "run",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "details": {"stage": "discovery"},
                "result": {
                    "status": "end",
                    "rc": int(discovery_result.get("rc", 1)),
                    "timed_out": bool(discovery_result.get("timed_out")),
                },
            }
        )

        # Collect discovery summary (non-blocking — Stage 2 proceeds regardless).
        discovery_summary, discovery_issues = self.validator.validate_discovery_summary(runtime, outputs_dir)
        if discovery_summary:
            self.artifacts.write_json("execution/codex_outputs/discovery_summary.json", discovery_summary)
            self.log("PROGRESS", f"Discovery: project_type={discovery_summary.get('project_type')}, "
                      f"env_ready={discovery_summary.get('environment_ready')}")
        if discovery_issues:
            reason_codes.extend(discovery_issues)

        if bool(discovery_result.get("timed_out")):
            reason_codes.append("DISCOVERY_STAGE_TIMEOUT")

        # === Stage 2: Task Execution ===
        remaining = max(300, int(global_deadline - time.time()))
        execution_prompt = build_autonomous_execution_prompt(
            repo_dir=repo_dir,
            outputs_dir=outputs_dir,
            task_spec_path=f"{inputs_dir}/task_spec.json",
            metric_contract_path=f"{inputs_dir}/metric_contract.json",
            skill_path=codex_skill_remote or None,
        )
        execution_cmd = self._prepend_path(
            self._build_codex_cmd(execution_prompt, extra_args=cmd_args),
            workspace_bin_dir=workspace_bin_dir,
        )

        worklog_events.append(
            {
                "type": "run",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "details": {"stage": "execution", "command": execution_cmd},
                "result": {"status": "start", "timeout_sec": remaining},
            }
        )

        self.log("PROGRESS", "Stage 2: Executing tasks from task_spec...")
        execution_result = self._run_stage(
            runtime,
            label="execution_main",
            cmd=execution_cmd,
            repo_dir=repo_dir,
            outputs_dir=outputs_dir,
            workspace_root=workspace_root,
            timeout_sec=remaining,
            reason_codes=reason_codes,
            capability_snapshot=capability_snapshot,
            local_stream_path=stream_local_path if stream_sync_enable else None,
            stream_sync_every_sec=stream_sync_every_sec,
        )

        worklog_events.append(
            {
                "type": "run",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "details": {"stage": "execution"},
                "result": {
                    "status": "end",
                    "rc": int(execution_result.get("rc", 1)),
                    "timed_out": bool(execution_result.get("timed_out")),
                },
            }
        )

        saw_global_timeout = bool(execution_result.get("timed_out"))
        if saw_global_timeout:
            reason_codes.append("GLOBAL_TIMEOUT_45M")

        # Repair attempt if Stage 2 failed.
        if int(execution_result.get("rc", 1)) != 0 and not saw_global_timeout:
            repair_remaining = max(60, int(global_deadline - time.time()))
            repair_timeout = min(900, repair_remaining)
            repair_prompt = build_autonomous_repair_prompt(
                outputs_dir=outputs_dir,
                task_spec_path=f"{inputs_dir}/task_spec.json",
                skill_path=codex_skill_remote or None,
            )
            repair_cmd = self._prepend_path(
                self._build_codex_cmd(repair_prompt, extra_args=cmd_args),
                workspace_bin_dir=workspace_bin_dir,
            )
            self.log("PROGRESS", "Stage 2 failed, attempting repair...")
            _ = self._run_stage(
                runtime,
                label="execution_repair",
                cmd=repair_cmd,
                repo_dir=repo_dir,
                outputs_dir=outputs_dir,
                workspace_root=workspace_root,
                timeout_sec=repair_timeout,
                reason_codes=reason_codes,
                capability_snapshot=capability_snapshot,
                local_stream_path=stream_local_path if stream_sync_enable else None,
                stream_sync_every_sec=stream_sync_every_sec,
            )

        # === Collect Results ===
        fallback_rc = 124 if saw_global_timeout else int(execution_result.get("rc", 1))
        run_rows = self._collect_autonomous_results(
            runtime,
            outputs_dir=outputs_dir,
            tasks=tasks,
            fallback_rc=fallback_rc,
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
                        "discovery_summary.json",
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

        if saw_global_timeout:
            pip_diag = self.reporter.collect_pip_log_tail(runtime, outputs_dir)
            self.reporter.write_failure_artifact(
                stage="main",
                last_command="autonomous_execution",
                exit_code=124,
                stdout_tail="",
                stderr_tail="global timeout reached during autonomous execution",
                codex_exec_log_tail=self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log"),
                pip_log_tail=str(pip_diag.get("tail") or ""),
                reason_codes=self._dedupe(reason_codes),
                capability_snapshot=capability_snapshot,
            )
            raise RuntimeError(
                "Codex execution exceeded the global timeout. "
                "Partial artifacts were written to execution/codex_outputs/*"
            )

        if success_count == 0 and run_rows:
            pip_diag = self.reporter.collect_pip_log_tail(runtime, outputs_dir)
            self.reporter.write_failure_artifact(
                stage="main",
                last_command="autonomous_execution",
                exit_code=1,
                stdout_tail="",
                stderr_tail="no successful tasks in autonomous execution",
                codex_exec_log_tail=self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log"),
                pip_log_tail=str(pip_diag.get("tail") or ""),
                reason_codes=self._dedupe(reason_codes),
                capability_snapshot=capability_snapshot,
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
