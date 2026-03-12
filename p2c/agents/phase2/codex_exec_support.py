from __future__ import annotations

import json
import os
import re
import shlex
import time
from datetime import datetime, timezone
from typing import Any, Callable


def _tail(text: str, n: int = 800) -> str:
    if not text:
        return ""
    return text[-n:]


def _quote(path: str) -> str:
    return shlex.quote(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


DEPENDENCY_SIGNAL_NEEDLES = [
    "dependency",
    "resolutionimpossible",
    "no matching distribution found",
    "could not find a version",
    "module not found",
    "modulenotfounderror",
    "importerror",
    "pip",
]

TIMEOUT_BY_CLASS_SEC = {
    "short": 60,
    "medium": 10 * 60,
    "long": 60 * 60,
}


def _has_dependency_signal(text: str) -> bool:
    low = (text or "").lower()
    return any(x in low for x in DEPENDENCY_SIGNAL_NEEDLES)


def is_rate_limit_failure(log_text: str) -> bool:
    low = (log_text or "").lower()
    needles = [
        "429",
        "rate limit",
        "tpm",
        "stream disconnected",
        "too many requests",
        "retrying",
    ]
    return any(x in low for x in needles)


def timeout_for_class(timeout_class: str, fallback_sec: int = 600) -> int:
    return int(TIMEOUT_BY_CLASS_SEC.get(str(timeout_class or "").lower(), fallback_sec))


def extract_task_items(task_spec: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = task_spec.get("tasks")
    if isinstance(tasks, list) and tasks:
        out: list[dict[str, Any]] = []
        for row in tasks:
            if not isinstance(row, dict):
                continue
            cmd = str(row.get("command") or "").strip()
            entrypoint = str(row.get("entrypoint") or "").strip()
            if not cmd:
                continue
            out.append(
                {
                    "task_id": str(row.get("task_id") or f"task_{len(out)+1:02d}"),
                    "entrypoint": entrypoint,
                    "command": cmd,
                    "timeout_class": str(row.get("timeout_class") or "medium"),
                    "expected_metrics": list(row.get("expected_metrics") or []),
                    "hyperparams": dict(row.get("hyperparams") or {}),
                }
            )
        if out:
            return out

    # Legacy fallback from entrypoints.
    entrypoints = task_spec.get("entrypoints")
    if not isinstance(entrypoints, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(entrypoints, start=1):
        if not isinstance(row, dict):
            continue
        cmd = str(row.get("command") or "").strip()
        path = str(row.get("path") or "").strip()
        if not cmd:
            continue
        out.append(
            {
                "task_id": f"legacy_task_{idx:02d}",
                "entrypoint": path,
                "command": cmd,
                "timeout_class": "medium",
                "expected_metrics": [],
                "hyperparams": {},
            }
        )
    return out


class CodexOutputValidator:
    def _json_read(self, runtime, remote_path: str) -> tuple[dict | None, str | None]:
        try:
            raw = runtime.read_text(remote_path)
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            return None, f"JSON_READ_FAILED:{remote_path}:{e}"
        if not isinstance(data, dict):
            return None, f"JSON_NOT_OBJECT:{remote_path}"
        return data, None

    def outputs_ready(self, runtime, outputs_dir: str) -> tuple[bool, list[str]]:
        issues: list[str] = []
        run_manifest, err = self._json_read(runtime, f"{outputs_dir}/run_manifest.json")
        if err:
            issues.append(err)
        else:
            runs = run_manifest.get("runs")
            if not isinstance(runs, list) or not runs:
                issues.append("RUN_MANIFEST_EMPTY_RUNS")
            else:
                required_run_keys = {
                    "run_id",
                    "command",
                    "params",
                    "cwd",
                    "exit_code",
                    "status",
                    "metrics",
                }
                for idx, run_item in enumerate(runs):
                    if not isinstance(run_item, dict):
                        issues.append(f"RUN_MANIFEST_ITEM_NOT_OBJECT:{idx}")
                        continue
                    missing = sorted(required_run_keys - set(run_item.keys()))
                    if missing:
                        issues.append(f"RUN_MANIFEST_ITEM_MISSING:{idx}:{','.join(missing)}")
                    if not isinstance(run_item.get("metrics"), dict):
                        issues.append(f"RUN_MANIFEST_ITEM_METRICS_NOT_OBJECT:{idx}")

        claim_alignment, err = self._json_read(runtime, f"{outputs_dir}/claim_alignment.json")
        if err:
            issues.append(err)
        else:
            claims = claim_alignment.get("claims")
            if not isinstance(claims, list) or not claims:
                issues.append("CLAIM_ALIGNMENT_EMPTY_CLAIMS")
            else:
                required_claim_keys = {"claim_id", "evaluable", "source"}
                for idx, claim_item in enumerate(claims):
                    if not isinstance(claim_item, dict):
                        issues.append(f"CLAIM_ALIGNMENT_ITEM_NOT_OBJECT:{idx}")
                        continue
                    missing = sorted(required_claim_keys - set(claim_item.keys()))
                    if missing:
                        issues.append(f"CLAIM_ALIGNMENT_ITEM_MISSING:{idx}:{','.join(missing)}")
                    source = claim_item.get("source")
                    if not isinstance(source, list) or not source:
                        issues.append(f"CLAIM_ALIGNMENT_ITEM_SOURCE_EMPTY:{idx}")

        for path in ("codex_worklog.jsonl", "patches.diff", "codex_exec.log"):
            try:
                runtime.read_text(f"{outputs_dir}/{path}")
            except Exception as e:  # noqa: BLE001
                issues.append(f"MISSING_OUTPUT:{path}:{e}")
        return (len(issues) == 0), issues

    @staticmethod
    def outputs_missing(issues: list[str]) -> bool:
        return any(x.startswith("JSON_READ_FAILED:") or x.startswith("MISSING_OUTPUT:") for x in issues)

    @staticmethod
    def _run_failed_for_dependency(run_item: dict[str, Any]) -> bool:
        status = str(run_item.get("status") or "").lower()
        rc = run_item.get("exit_code")
        if rc in (0, "0"):
            return False
        reason_codes = [str(x).lower() for x in (run_item.get("reason_codes") or [])]
        stdout_tail = str(run_item.get("stdout_tail") or "").lower()
        stderr_tail = str(run_item.get("stderr_tail") or "").lower()
        signals = reason_codes + [status, stdout_tail, stderr_tail]
        return any(n in s for s in signals for n in DEPENDENCY_SIGNAL_NEEDLES)

    def all_entrypoints_unrunnable_due_dependency(self, runtime, outputs_dir: str) -> bool:
        manifest, err = self._json_read(runtime, f"{outputs_dir}/run_manifest.json")
        if err or not manifest:
            return False
        runs = manifest.get("runs")
        if not isinstance(runs, list) or not runs:
            return False
        success_count = 0
        for row in runs:
            if not isinstance(row, dict):
                continue
            if row.get("exit_code") in (0, "0"):
                success_count += 1
        if success_count > 0:
            return False
        return all(isinstance(row, dict) and self._run_failed_for_dependency(row) for row in runs)

    def dependency_failure_count(self, runtime, outputs_dir: str) -> tuple[int, int, int]:
        manifest, err = self._json_read(runtime, f"{outputs_dir}/run_manifest.json")
        if err or not manifest:
            return (0, 0, 0)
        runs = manifest.get("runs")
        if not isinstance(runs, list):
            return (0, 0, 0)
        total = len(runs)
        success = 0
        dep_failed = 0
        for row in runs:
            if not isinstance(row, dict):
                continue
            if row.get("exit_code") in (0, "0"):
                success += 1
            elif self._run_failed_for_dependency(row):
                dep_failed += 1
        return (total, success, dep_failed)

    def validate_discovery_summary(self, runtime, outputs_dir: str) -> tuple[dict | None, list[str]]:
        """Validate discovery_summary.json produced by autonomous Stage 1."""
        issues: list[str] = []
        data, err = self._json_read(runtime, f"{outputs_dir}/discovery_summary.json")
        if err:
            issues.append(err)
            return None, issues
        required_keys = {"project_type", "language", "dependency_steps", "discovered_entrypoints", "environment_ready"}
        missing = sorted(required_keys - set(data.keys()))
        if missing:
            issues.append(f"DISCOVERY_SUMMARY_MISSING_KEYS:{','.join(missing)}")
        if not isinstance(data.get("dependency_steps"), list):
            issues.append("DISCOVERY_SUMMARY_DEPENDENCY_STEPS_NOT_LIST")
        if not isinstance(data.get("discovered_entrypoints"), list):
            issues.append("DISCOVERY_SUMMARY_ENTRYPOINTS_NOT_LIST")
        return data, issues


class CodexFailureReporter:
    def __init__(self, artifacts, log_fn: Callable[[str, str], None]):
        self.artifacts = artifacts
        self.log_fn = log_fn

    def safe_remote_log_tail(self, runtime, path: str, n: int = 800) -> str:
        try:
            return _tail(runtime.read_text(path), n=n)
        except Exception:  # noqa: BLE001
            return ""

    def collect_pip_log_tail(self, runtime, outputs_dir: str, n: int = 2000) -> dict[str, Any]:
        conflict_needles = [
            "resolutionimpossible",
            "no matching distribution found",
            "could not find a version",
            "conflict",
            "version solving failed",
            "unsatisfiable",
            "cannot install",
        ]
        activity_needles = [
            "pip install",
            "pip3 install",
            "python -m pip",
            "uv pip",
            "poetry install",
            "conda install",
            "collecting ",
            "installing collected packages",
        ]
        pip_text = ""
        codex_text = ""
        pip_log_path = f"{outputs_dir}/pip_install.log"
        try:
            pip_text = runtime.read_text(pip_log_path)
        except Exception:  # noqa: BLE001
            pip_text = ""
        codex_text = self.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log", n=6000)
        merged = "\n".join([x for x in [pip_text, codex_text] if x]).lower()
        has_conflict_signal = any(x in merged for x in conflict_needles)
        has_pip_activity = any(x in merged for x in activity_needles) or has_conflict_signal
        tail_src = pip_text if pip_text else codex_text
        return {
            "tail": _tail(tail_src, n=n),
            "has_conflict_signal": has_conflict_signal,
            "has_pip_activity": has_pip_activity,
        }

    @staticmethod
    def infer_tool_reason_codes(*texts: str) -> list[str]:
        low = "\n".join([str(x or "") for x in texts]).lower()
        out: list[str] = []
        if "update_plan" in low:
            out.append("UNKNOWN_TOOL_COMMAND_ATTEMPTED")
        if "apply_patch: not found" in low or "apply_patch not found" in low:
            out.append("PATCH_TOOL_MISSING_IN_SESSION")
        return _dedupe(out)

    def write_failure_artifact(
        self,
        *,
        stage: str,
        last_command: str,
        exit_code: int,
        stdout_tail: str,
        stderr_tail: str,
        codex_exec_log_tail: str,
        pip_log_tail: str,
        reason_codes: list[str],
        capability_snapshot: dict[str, Any] | None = None,
        dependency_bootstrap_trace: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "stage": stage,
            "last_command": last_command,
            "exit_code": exit_code,
            "stdout_tail": _tail(stdout_tail, n=3000),
            "stderr_tail": _tail(stderr_tail, n=3000),
            "codex_exec_log_tail": _tail(codex_exec_log_tail, n=5000),
            "pip_log_tail": _tail(pip_log_tail, n=3000),
            "reason_codes": _dedupe(list(reason_codes)),
            "capability_snapshot": capability_snapshot or {},
            "dependency_bootstrap_trace": list(dependency_bootstrap_trace or []),
        }
        payload["reason_codes"] = _dedupe(
            list(payload["reason_codes"])
            + self.infer_tool_reason_codes(
                payload["stderr_tail"],
                payload["codex_exec_log_tail"],
                payload["pip_log_tail"],
            )
        )
        self.artifacts.write_json("execution/codex_failure.json", payload)
        self.artifacts.write_json("execution/codex_outputs/codex_failure.json", payload)
        self.artifacts.append_text(
            "execution/run.log",
            (
                "[run_codex_exec] failure snapshot written: "
                f"stage={stage} exit_code={exit_code} "
                f"reason_codes={','.join(payload['reason_codes'][:8])}\n"
            ),
        )

    @staticmethod
    def extract_launcher_rc_reason(message: str) -> str | None:
        m = re.search(r"CODEX_BACKGROUND_LAUNCHER_RC_(\d+)", message or "")
        if not m:
            return None
        return f"CODEX_BACKGROUND_LAUNCHER_RC_{m.group(1)}"

    def handle_stage_exception(
        self,
        *,
        stage: str,
        cmd: str,
        error: Exception,
        runtime,
        outputs_dir: str,
        reason_codes: list[str],
        capability_snapshot: dict[str, Any] | None = None,
        dependency_bootstrap_trace: list[str] | None = None,
    ) -> None:
        codex_tail = self.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log")
        pip_tail = self.collect_pip_log_tail(runtime, outputs_dir)
        rc = 1
        msg = str(error)
        if "code 124" in msg.lower() or "timeout" in msg.lower():
            rc = 124
        extra = list(reason_codes) + [f"STAGE_EXCEPTION_{stage.upper()}", "CODEX_BACKGROUND_LAUNCH_FAILED"]
        launcher_rc_reason = self.extract_launcher_rc_reason(msg)
        if launcher_rc_reason:
            extra.append(launcher_rc_reason)
        if pip_tail.get("has_conflict_signal", False):
            extra.append("DEPENDENCY_INSTALL_CONFLICT")
        elif pip_tail.get("has_pip_activity", False):
            extra.append("DEPENDENCY_INSTALL_ACTIVITY_DETECTED")
        self.write_failure_artifact(
            stage=stage if stage in {"precheck", "main", "repair", "postcheck"} else "postcheck",
            last_command=cmd,
            exit_code=rc,
            stdout_tail="",
            stderr_tail=msg,
            codex_exec_log_tail=codex_tail,
            pip_log_tail=str(pip_tail.get("tail") or ""),
            reason_codes=extra,
            capability_snapshot=capability_snapshot,
            dependency_bootstrap_trace=dependency_bootstrap_trace,
        )


class CodexBackgroundExecutor:
    def __init__(self, log_fn: Callable[[str, str], None], artifacts=None):
        self.log_fn = log_fn
        self.artifacts = artifacts
        self._stream_cursor_by_remote: dict[str, int] = {}

    @staticmethod
    def stage_log_name(label: str) -> str:
        normalized = str(label or "").lower()
        if normalized == "main" or normalized.endswith("_main") or "_retry_" in normalized:
            return "codex_main.log"
        if normalized == "repair" or normalized.endswith("_repair"):
            return "codex_repair.log"
        return f"codex_{label}.log"

    def _start_background(
        self,
        runtime,
        *,
        cmd: str,
        cwd: str,
        outputs_dir: str,
        label: str,
    ) -> dict[str, str]:
        pid_name = "codex_exec.pid" if label == "main" else f"codex_{label}.pid"
        exit_name = "codex_exec.rc" if label == "main" else f"codex_{label}.rc"
        pid_path = f"{outputs_dir}/{pid_name}"
        exit_path = f"{outputs_dir}/{exit_name}"
        log_path = f"{outputs_dir}/{self.stage_log_name(label)}"
        combined_log_path = f"{outputs_dir}/codex_exec.log"
        script = (
            f"mkdir -p {_quote(outputs_dir)}; "
            f"touch {_quote(combined_log_path)}; "
            f"rm -f {_quote(pid_path)} {_quote(exit_path)}; "
            f"( {cmd}; rc=$?; printf '%s' \"$rc\" > {_quote(exit_path)} ) 2>&1 | "
            f"tee -a {_quote(log_path)} >> {_quote(combined_log_path)} & "
            f"echo $! > {_quote(pid_path)}"
        )
        launcher_cmd = f"bash -lc {shlex.quote(script)}"

        def _is_deadline_error(err: Exception) -> bool:
            text = str(err or "").lower()
            return "context deadline exceeded" in text or "deadline_exceeded" in text

        def _probe_background_started() -> bool:
            probe_script = (
                f"if [ -f {_quote(exit_path)} ]; then exit 0; fi; "
                f"if [ -f {_quote(pid_path)} ]; then "
                f"  pid=$(cat {_quote(pid_path)} 2>/dev/null || true); "
                "  if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then exit 0; fi; "
                "fi; "
                "exit 1"
            )
            try:
                probe = runtime.run_command(
                    f"bash -lc {shlex.quote(probe_script)}",
                    cwd=cwd,
                    timeout_sec=20,
                )
                return probe.rc == 0
            except Exception:  # noqa: BLE001
                return False

        try:
            launcher = runtime.run_command(
                launcher_cmd,
                cwd=cwd,
                timeout_sec=0,
            )
        except Exception as e:  # noqa: BLE001
            if _is_deadline_error(e) and _probe_background_started():
                return {
                    "pid_path": pid_path,
                    "exit_path": exit_path,
                    "log_path": log_path,
                    "combined_log_path": combined_log_path,
                    "launch_stdout": "",
                    "launch_stderr": str(e),
                }
            raise RuntimeError(
                f"failed to launch codex {label}: runtime.run_command raised: {e}; "
                f"launcher_cmd={launcher_cmd}"
            ) from e
        if launcher.rc != 0:
            reason = f"CODEX_BACKGROUND_LAUNCHER_RC_{launcher.rc}"
            raise RuntimeError(
                f"failed to launch codex {label} command {reason}; "
                f"script={script!r}; "
                f"stdout_tail={_tail(launcher.stdout or '', 500)!r}; "
                f"stderr_tail={_tail(launcher.stderr or '', 500)!r}; "
                f"launcher_cmd={launcher_cmd}"
            )
        return {
            "pid_path": pid_path,
            "exit_path": exit_path,
            "log_path": log_path,
            "combined_log_path": combined_log_path,
            "launch_stdout": launcher.stdout or "",
            "launch_stderr": launcher.stderr or "",
        }

    def _emit_log_delta(self, runtime, *, log_path: str, cursor: int, label: str) -> int:
        try:
            text = runtime.read_text(log_path)
        except Exception:  # noqa: BLE001
            return cursor

        if len(text) < cursor:
            cursor = 0
        delta = text[cursor:]
        if not delta:
            return len(text)

        for line in delta.splitlines():
            if not line.strip():
                continue
            self.log_fn("PROGRESS", f"[codex:{label}] {line[:800]}")
        return len(text)

    def _safe_append_local_stream(self, local_stream_path: str, content: str) -> None:
        if not local_stream_path or not content or self.artifacts is None:
            return
        try:
            self.artifacts.append_text(local_stream_path, content)
        except Exception as e:  # noqa: BLE001
            self.log_fn("PROGRESS", f"[codex:stream] local append failed: {e}")

    def _sync_remote_log_delta_to_local(
        self,
        runtime,
        *,
        remote_log_path: str,
        local_stream_path: str | None,
    ) -> None:
        if not local_stream_path or self.artifacts is None:
            return
        try:
            text = runtime.read_text(remote_log_path)
        except Exception as e:  # noqa: BLE001
            self.log_fn("PROGRESS", f"[codex:stream] remote read failed: {e}")
            return

        cursor = self._stream_cursor_by_remote.get(remote_log_path, 0)
        if len(text) < cursor:
            cursor = 0
        delta = text[cursor:]
        if delta:
            self._safe_append_local_stream(local_stream_path, delta)
        self._stream_cursor_by_remote[remote_log_path] = len(text)

    def _poll_background(
        self,
        runtime,
        *,
        pid_path: str,
        exit_path: str,
        log_path: str,
        combined_log_path: str,
        label: str,
        cwd: str,
        timeout_sec: int,
        local_stream_path: str | None = None,
        stream_sync_every_sec: int = 20,
        stream_flush_on_exit: bool = True,
        poll_sec: int = 5,
    ) -> tuple[int, int, bool]:
        deadline = time.time() + timeout_sec
        polls = 0
        cursor = 0
        last_stream_sync = 0.0
        exit_probe_cmd = f"bash -lc {shlex.quote(f'test -f {_quote(exit_path)}')}"
        alive_script = (
            "pid=''; "
            f"if [ -f {_quote(pid_path)} ]; then "
            f"  pid=$(cat {_quote(pid_path)} 2>/dev/null || true); "
            "fi; "
            "[ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null"
        )
        alive_cmd = f"bash -lc {shlex.quote(alive_script)}"

        while time.time() < deadline:
            polls += 1
            now = time.time()
            if local_stream_path and (now - last_stream_sync >= max(1, int(stream_sync_every_sec))):
                self._sync_remote_log_delta_to_local(
                    runtime,
                    remote_log_path=combined_log_path,
                    local_stream_path=local_stream_path,
                )
                last_stream_sync = now
            cursor = self._emit_log_delta(runtime, log_path=log_path, cursor=cursor, label=label)
            exited = runtime.run_command(exit_probe_cmd, cwd=cwd, timeout_sec=20)
            if exited.rc == 0:
                cursor = self._emit_log_delta(runtime, log_path=log_path, cursor=cursor, label=label)
                if stream_flush_on_exit:
                    self._sync_remote_log_delta_to_local(
                        runtime,
                        remote_log_path=combined_log_path,
                        local_stream_path=local_stream_path,
                    )
                raw = runtime.read_text(exit_path).strip()
                try:
                    return int(raw), polls, False
                except ValueError:
                    return 1, polls, False

            alive = runtime.run_command(alive_cmd, cwd=cwd, timeout_sec=20)
            if alive.rc != 0:
                time.sleep(min(2, poll_sec))
                cursor = self._emit_log_delta(runtime, log_path=log_path, cursor=cursor, label=label)
                exited = runtime.run_command(exit_probe_cmd, cwd=cwd, timeout_sec=20)
                if exited.rc == 0:
                    cursor = self._emit_log_delta(runtime, log_path=log_path, cursor=cursor, label=label)
                    if stream_flush_on_exit:
                        self._sync_remote_log_delta_to_local(
                            runtime,
                            remote_log_path=combined_log_path,
                            local_stream_path=local_stream_path,
                        )
                    raw = runtime.read_text(exit_path).strip()
                    try:
                        return int(raw), polls, False
                    except ValueError:
                        return 1, polls, False
                return 1, polls, False

            time.sleep(poll_sec)

        kill_script = (
            f"if [ -f {_quote(pid_path)} ]; then "
            f"  pid=$(cat {_quote(pid_path)} 2>/dev/null || true); "
            "  if [ -n \"$pid\" ]; then "
            "    kill \"$pid\" 2>/dev/null || true; "
            "    sleep 1; "
            "    kill -9 \"$pid\" 2>/dev/null || true; "
            "  fi; "
            "fi"
        )
        runtime.run_command(f"bash -lc {shlex.quote(kill_script)}", cwd=cwd, timeout_sec=30)
        self._emit_log_delta(runtime, log_path=log_path, cursor=cursor, label=label)
        if stream_flush_on_exit:
            self._sync_remote_log_delta_to_local(
                runtime,
                remote_log_path=combined_log_path,
                local_stream_path=local_stream_path,
            )
        return 124, polls, True

    def run(
        self,
        runtime,
        *,
        cmd: str,
        cwd: str,
        outputs_dir: str,
        label: str,
        timeout_sec: int,
        workspace_root: str,
        local_stream_path: str | None = None,
        stream_sync_every_sec: int = 20,
        stream_flush_on_exit: bool = True,
    ) -> dict[str, Any]:
        launch_info = self._start_background(
            runtime,
            cmd=cmd,
            cwd=cwd,
            outputs_dir=outputs_dir,
            label=label,
        )
        pid_path = launch_info["pid_path"]
        exit_path = launch_info["exit_path"]
        log_path = launch_info["log_path"]
        rc, polls, timed_out = self._poll_background(
            runtime,
            pid_path=pid_path,
            exit_path=exit_path,
            log_path=log_path,
            combined_log_path=launch_info["combined_log_path"],
            label=label,
            cwd=workspace_root,
            timeout_sec=timeout_sec,
            local_stream_path=local_stream_path,
            stream_sync_every_sec=stream_sync_every_sec,
            stream_flush_on_exit=stream_flush_on_exit,
        )
        return {
            "rc": rc,
            "polls": polls,
            "timed_out": timed_out,
            "pid_path": pid_path,
            "exit_path": exit_path,
            "log_path": log_path,
            "launch_stdout": launch_info.get("launch_stdout", ""),
            "launch_stderr": launch_info.get("launch_stderr", ""),
        }


class CodexCapabilityGate:
    def __init__(self) -> None:
        raw_modules = (os.getenv("P2C_CAPABILITY_REQUIRED_MODULES") or "numpy").strip()
        self.required_modules = [x.strip() for x in raw_modules.split(",") if x.strip()]
        self.entrypoint_probe_timeout_sec = int(os.getenv("P2C_ENTRYPOINT_PROBE_TIMEOUT_SEC", "120"))

    @staticmethod
    def _run_probe(runtime, *, cmd: str, cwd: str, timeout_sec: int = 30):
        return runtime.run_command(f"bash -lc {shlex.quote(cmd)}", cwd=cwd, timeout_sec=timeout_sec)

    @staticmethod
    def _as_bool_probe(result) -> bool:
        if result.rc != 0:
            return False
        raw = (result.stdout or "").strip()
        return raw in {"1", "true", "True"}

    @staticmethod
    def _append_remote_text(runtime, path: str, text: str) -> None:
        try:
            current = runtime.read_text(path)
        except Exception:  # noqa: BLE001
            current = ""
        runtime.write_text(path, current + text)

    def probe_python_capabilities(self, runtime, workspace_root: str) -> dict[str, Any]:
        py = self._run_probe(runtime, cmd="python3 -c \"import sys; print(sys.version)\"", cwd=workspace_root, timeout_sec=30)
        pip = self._run_probe(
            runtime,
            cmd="python3 -c \"import importlib.util as u; print(1 if u.find_spec('pip') else 0)\"",
            cwd=workspace_root,
            timeout_sec=30,
        )
        ensurepip = self._run_probe(
            runtime,
            cmd="python3 -c \"import importlib.util as u; print(1 if u.find_spec('ensurepip') else 0)\"",
            cwd=workspace_root,
            timeout_sec=30,
        )

        tool_paths: dict[str, str | None] = {}
        tool_versions: dict[str, str] = {}
        for tool in ["python", "python3", "pip", "pip3"]:
            path_probe = self._run_probe(
                runtime,
                cmd=f"command -v {tool} 2>/dev/null || true",
                cwd=workspace_root,
                timeout_sec=30,
            )
            path_val = (path_probe.stdout or "").strip().splitlines()[0].strip() if (path_probe.stdout or "").strip() else ""
            tool_paths[tool] = path_val or None
            if path_val:
                flag = "-V" if tool in {"python", "python3"} else "--version"
                version_probe = self._run_probe(
                    runtime,
                    cmd=f"{tool} {flag} 2>&1 || true",
                    cwd=workspace_root,
                    timeout_sec=30,
                )
                tool_versions[tool] = _tail((version_probe.stdout or version_probe.stderr or "").strip(), 300)

        modules_available: dict[str, bool] = {}
        module_probe_detail: dict[str, dict[str, Any]] = {}
        for module_name in self.required_modules:
            probe = self._run_probe(
                runtime,
                cmd=(
                    "python3 -c "
                    + shlex.quote(
                        f"import importlib.util as u; print(1 if u.find_spec('{module_name}') else 0)"
                    )
                ),
                cwd=workspace_root,
                timeout_sec=30,
            )
            module_ok = self._as_bool_probe(probe)
            modules_available[module_name] = module_ok
            module_probe_detail[module_name] = {
                "rc": probe.rc,
                "stdout_tail": _tail(probe.stdout or "", 200),
                "stderr_tail": _tail(probe.stderr or "", 200),
            }

        reason_codes: list[str] = []
        if py.rc != 0:
            reason_codes.append("PYTHON3_MISSING")
        if not self._as_bool_probe(pip):
            reason_codes.append("PIP_NOT_AVAILABLE")
        if not self._as_bool_probe(ensurepip):
            reason_codes.append("ENSUREPIP_MISSING")
        if tool_paths.get("pip3"):
            reason_codes.append("PIP3_COMMAND_AVAILABLE")
        for module_name, is_ok in modules_available.items():
            if not is_ok:
                code = re.sub(r"[^A-Za-z0-9_]", "_", module_name.upper())
                reason_codes.append(f"REQUIRED_MODULE_MISSING_{code}")

        return {
            "python_ok": py.rc == 0,
            "python_version": (py.stdout or "").strip(),
            "pip_available": self._as_bool_probe(pip),
            "pip_command_available": bool(tool_paths.get("pip") or tool_paths.get("pip3")),
            "ensurepip_available": self._as_bool_probe(ensurepip),
            "required_modules_available": modules_available,
            "module_probe_detail": module_probe_detail,
            "tool_paths": tool_paths,
            "tool_versions": tool_versions,
            "reason_codes": _dedupe(reason_codes),
        }

