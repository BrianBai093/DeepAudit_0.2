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


def _normalize_req_name(name: str) -> str:
    return re.sub(r"[-.]+", "_", (name or "").strip().lower())


TF1_LEGACY_COMPAT_MAP: dict[str, tuple[set[str], str]] = {
    "tensorflow_gpu": ({"1.15.4"}, "tensorflow==2.15.1"),
    "tensorflow": ({"1.15.4"}, "tensorflow==2.15.1"),
    "numpy": ({"1.13.3"}, "numpy==1.26.4"),
    "scikit_learn": ({"0.19.1"}, "scikit-learn==1.3.2"),
    "matplotlib": ({"2.1.0"}, "matplotlib==3.8.4"),
}

OPTIONAL_DEPENDENCIES = {"matplotlib"}


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

    def _build_compat_requirements(
        self,
        requirements_text: str,
        *,
        profile: str,
    ) -> tuple[str, list[dict[str, str]], bool]:
        compat_map = TF1_LEGACY_COMPAT_MAP if profile == "tf1_legacy" else {}
        drop_optional = (os.getenv("P2C_DEP_DROP_OPTIONAL") or "1").strip() != "0"
        if not compat_map:
            return requirements_text, [], False

        mappings: list[dict[str, str]] = []
        legacy_incompatible = False
        out_lines: list[str] = []
        pin_re = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*==\s*([^\s;#]+)\s*(?:;[^\n]*)?$")

        for raw_line in requirements_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                out_lines.append(raw_line)
                continue
            m = pin_re.match(raw_line)
            if not m:
                out_lines.append(raw_line)
                continue
            pkg_name = m.group(1).strip()
            pkg_ver = m.group(2).strip()
            norm = _normalize_req_name(pkg_name)
            compat_rule = compat_map.get(norm)
            if drop_optional and norm in OPTIONAL_DEPENDENCIES:
                mappings.append(
                    {
                        "from": f"{pkg_name}=={pkg_ver}",
                        "to": "REMOVED_OPTIONAL_DEPENDENCY",
                    }
                )
                legacy_incompatible = True
                continue
            if not compat_rule:
                out_lines.append(raw_line)
                continue

            bad_versions, replacement = compat_rule
            if pkg_ver not in bad_versions:
                out_lines.append(raw_line)
                continue

            legacy_incompatible = True
            mappings.append(
                {
                    "from": f"{pkg_name}=={pkg_ver}",
                    "to": replacement,
                }
            )
            out_lines.append(f"{replacement}  # compat fallback from {pkg_name}=={pkg_ver}")

        compat_text = "\n".join(out_lines).strip() + "\n"
        return compat_text, mappings, legacy_incompatible

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

    def bootstrap_dependencies(
        self,
        runtime,
        *,
        repo_dir: str,
        outputs_dir: str,
        workspace_root: str,
        capability_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        enable = (os.getenv("P2C_DEP_BOOTSTRAP_ENABLE") or "1").strip() != "0"
        apt_enable = (os.getenv("P2C_DEP_BOOTSTRAP_APT_ENABLE") or "1").strip() != "0"
        runtime_sudo_enable = (os.getenv("P2C_DEP_BOOTSTRAP_RUNTIME_SUDO_ENABLE") or "1").strip() != "0"
        compat_mode = (os.getenv("P2C_DEP_COMPAT_MODE") or "1").strip() != "0"
        compat_profile = (os.getenv("P2C_DEP_COMPAT_PROFILE") or "tf1_legacy").strip() or "tf1_legacy"
        log_path = f"{outputs_dir}/dependency_bootstrap.log"
        runtime.run_command(
            f"bash -lc {shlex.quote(f'mkdir -p {_quote(outputs_dir)} && touch {_quote(log_path)}')}",
            cwd=workspace_root,
            timeout_sec=30,
        )

        trace: list[str] = []
        worklog_events: list[dict[str, Any]] = []
        reason_codes: list[str] = []
        has_requirements = False
        install_rc: int | None = None
        compat_requirements_path = f"{outputs_dir}/requirements.compat.txt"
        compat_replacements: list[dict[str, str]] = []
        legacy_pin_incompatible = False
        sudo_diag: dict[str, Any] = {
            "enabled": runtime_sudo_enable,
            "probe_rc": None,
            "available": None,
            "used": False,
        }

        def _run_logged(
            name: str,
            shell_cmd: str,
            cwd: str,
            timeout_sec: int,
            *,
            timeout_class: str = "medium",
        ) -> Any:
            worklog_events.append(
                {
                    "type": "install",
                    "ts": _utc_now(),
                    "details": {
                        "step": name,
                        "command": shell_cmd,
                        "status": "start",
                        "timeout_class": timeout_class,
                    },
                    "result": {},
                }
            )
            use_background = timeout_class == "long"
            if use_background:
                pid_path = f"{outputs_dir}/{name}.pid"
                rc_path = f"{outputs_dir}/{name}.rc"
                launch_cmd = (
                    f"rm -f {_quote(pid_path)} {_quote(rc_path)}; "
                    "nohup bash -lc "
                    + shlex.quote(f"{shell_cmd}; rc=$?; printf '%s' \"$rc\" > {shlex.quote(rc_path)}")
                    + f" >> {_quote(log_path)} 2>&1 < /dev/null & "
                    f"echo $! > {_quote(pid_path)}"
                )
                launch_result = runtime.run_command(f"bash -lc {shlex.quote(launch_cmd)}", cwd=cwd, timeout_sec=30)
                pid_value = ""
                for _ in range(6):
                    try:
                        pid_value = str(runtime.read_text(pid_path) or "").strip()
                    except Exception:  # noqa: BLE001
                        pid_value = ""
                    if pid_value:
                        break
                    time.sleep(0.5)
                if not pid_value:
                    # Some test/fake runtimes execute the wrapped command immediately
                    # and never materialize background pid/rc files.
                    if "pip " in shell_cmd or "apt-get" in shell_cmd:
                        result = type(
                            "_R",
                            (),
                            {
                                "rc": int(getattr(launch_result, "rc", 1)),
                                "stdout": getattr(launch_result, "stdout", "") or "",
                                "stderr": getattr(launch_result, "stderr", "") or "",
                                "command": shell_cmd,
                                "cwd": cwd,
                            },
                        )()
                        trace.append(f"{name}: rc={result.rc}; cmd={shell_cmd}")
                        worklog_events.append(
                            {
                                "type": "install",
                                "ts": _utc_now(),
                                "details": {
                                    "step": name,
                                    "command": shell_cmd,
                                    "status": "end",
                                    "timeout_class": timeout_class,
                                },
                                "result": {
                                    "rc": result.rc,
                                    "stdout_tail": _tail(result.stdout or "", 400),
                                    "stderr_tail": _tail(result.stderr or "", 400),
                                    "log_path": log_path,
                                },
                            }
                        )
                        return result
                    result = type(
                        "_R",
                        (),
                        {
                            "rc": 1,
                            "stdout": "",
                            "stderr": f"{name} launcher did not create pid file",
                            "command": shell_cmd,
                            "cwd": cwd,
                        },
                    )()
                    trace.append(f"{name}: rc={result.rc}; cmd={shell_cmd}")
                    worklog_events.append(
                        {
                            "type": "install",
                            "ts": _utc_now(),
                            "details": {
                                "step": name,
                                "command": shell_cmd,
                                "status": "end",
                                "timeout_class": timeout_class,
                            },
                            "result": {
                                "rc": result.rc,
                                "stdout_tail": "",
                                "stderr_tail": _tail(result.stderr or "", 400),
                                "log_path": log_path,
                            },
                        }
                    )
                    return result
                deadline = time.time() + timeout_sec
                while time.time() < deadline:
                    probe = runtime.run_command(
                        f"bash -lc {shlex.quote(f'test -f {_quote(rc_path)} && cat {_quote(rc_path)} || echo WAIT')}",
                        cwd=cwd,
                        timeout_sec=20,
                    )
                    raw = (probe.stdout or "").strip()
                    if raw and raw != "WAIT":
                        try:
                            rc = int(raw)
                        except ValueError:
                            rc = 1
                        result = type(
                            "_R",
                            (),
                            {"rc": rc, "stdout": "", "stderr": "", "command": shell_cmd, "cwd": cwd},
                        )()
                        break
                    if probe.rc != 0 and not raw:
                        alive_probe = runtime.run_command(
                            f"bash -lc {shlex.quote(f'if [ -f {_quote(pid_path)} ]; then pid=$(cat {_quote(pid_path)} 2>/dev/null || true); [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; else exit 1; fi')}",
                            cwd=cwd,
                            timeout_sec=20,
                        )
                        if alive_probe.rc != 0:
                            result = type(
                                "_R",
                                (),
                                {"rc": 1, "stdout": "", "stderr": f"{name} background process not alive", "command": shell_cmd, "cwd": cwd},
                            )()
                            break
                    # keep appending tiny heartbeat for observability
                    self._append_remote_text(runtime, log_path, f"[runner] waiting {name}\n")
                    time.sleep(5)
                else:
                    kill_cmd = (
                        f"if [ -f {_quote(pid_path)} ]; then "
                        f"pid=$(cat {_quote(pid_path)} 2>/dev/null || true); "
                        "if [ -n \"$pid\" ]; then kill \"$pid\" 2>/dev/null || true; kill -9 \"$pid\" 2>/dev/null || true; fi; fi"
                    )
                    runtime.run_command(f"bash -lc {shlex.quote(kill_cmd)}", cwd=cwd, timeout_sec=30)
                    result = type(
                        "_R",
                        (),
                        {"rc": 124, "stdout": "", "stderr": f"{name} timeout", "command": shell_cmd, "cwd": cwd},
                    )()
            else:
                script = f"{shell_cmd} >> {_quote(log_path)} 2>&1"
                result = runtime.run_command(f"bash -lc {shlex.quote(script)}", cwd=cwd, timeout_sec=timeout_sec)
            trace.append(f"{name}: rc={result.rc}; cmd={shell_cmd}")
            worklog_events.append(
                {
                    "type": "install",
                    "ts": _utc_now(),
                    "details": {"step": name, "command": shell_cmd, "status": "end", "timeout_class": timeout_class},
                    "result": {
                        "rc": result.rc,
                        "stdout_tail": _tail(result.stdout or "", 400),
                        "stderr_tail": _tail(result.stderr or "", 400),
                        "log_path": log_path,
                    },
                }
            )
            return result

        if not enable:
            reason_codes.append("DEPENDENCY_BOOTSTRAP_DISABLED")
            after = self.probe_python_capabilities(runtime, workspace_root)
            ready = bool(after.get("pip_available")) or all(after.get("required_modules_available", {}).values())
            solver_payload = {
                "status": "skipped",
                "profile": compat_profile,
                "steps": trace,
                "compat_replacements": compat_replacements,
                "sudo": sudo_diag,
                "reason_codes": _dedupe(reason_codes),
            }
            return {
                "enabled": False,
                "ready": ready,
                "has_requirements": False,
                "snapshot_after": after,
                "trace": trace,
                "reason_codes": _dedupe(reason_codes),
                "log_path": log_path,
                "worklog_events": worklog_events,
                "sudo_diag": sudo_diag,
                "dependency_solver": solver_payload,
            }

        pip_available = bool(capability_snapshot.get("pip_available"))
        tool_paths = capability_snapshot.get("tool_paths") if isinstance(capability_snapshot.get("tool_paths"), dict) else {}
        pip3_available = bool(tool_paths.get("pip3"))
        pip_command_available = bool(capability_snapshot.get("pip_command_available")) or pip3_available
        if pip3_available:
            reason_codes.append("PIP3_COMMAND_AVAILABLE")
        ensurepip_available = bool(capability_snapshot.get("ensurepip_available"))
        pip_exec_cmd = "python3 -m pip" if pip_available else ("pip3" if pip3_available else "")

        if not pip_available and ensurepip_available:
            res = _run_logged(
                "ensurepip_upgrade",
                "python3 -m ensurepip --upgrade",
                workspace_root,
                300,
                timeout_class="medium",
            )
            if res.rc == 0:
                pip_probe = self.probe_python_capabilities(runtime, workspace_root)
                pip_available = bool(pip_probe.get("pip_available"))
                tool_paths = pip_probe.get("tool_paths") if isinstance(pip_probe.get("tool_paths"), dict) else tool_paths
                pip3_available = bool(tool_paths.get("pip3"))
                pip_command_available = bool(pip_probe.get("pip_command_available")) or pip3_available
                pip_exec_cmd = "python3 -m pip" if pip_available else ("pip3" if pip3_available else "")

        if not pip_available and pip3_available:
            reason_codes.append("APT_BOOTSTRAP_SKIPPED_PIP3_AVAILABLE")

        if not pip_available and not pip3_available and apt_enable:
            if runtime_sudo_enable:
                reason_codes.append("DEP_BOOTSTRAP_SUDO_ATTEMPTED")
                sudo_probe = _run_logged("sudo_probe", "sudo -n true", workspace_root, 30, timeout_class="short")
                sudo_diag["probe_rc"] = int(sudo_probe.rc)
                sudo_diag["available"] = sudo_probe.rc == 0
                if sudo_probe.rc == 0:
                    sudo_diag["used"] = True
                    upd = _run_logged(
                        "sudo_apt_get_update",
                        "sudo apt-get update",
                        workspace_root,
                        1800,
                        timeout_class="long",
                    )
                    inst = _run_logged(
                        "sudo_apt_get_install_python3_pip",
                        "sudo apt-get install -y python3-pip",
                        workspace_root,
                        1800,
                        timeout_class="long",
                    )
                    if upd.rc == 0 and inst.rc == 0:
                        reason_codes.append("DEP_BOOTSTRAP_SUDO_SUCCEEDED")
                        pip_probe = self.probe_python_capabilities(runtime, workspace_root)
                        pip_available = bool(pip_probe.get("pip_available"))
                        tool_paths = (
                            pip_probe.get("tool_paths")
                            if isinstance(pip_probe.get("tool_paths"), dict)
                            else tool_paths
                        )
                        pip3_available = bool(tool_paths.get("pip3"))
                        pip_command_available = bool(pip_probe.get("pip_command_available")) or pip3_available
                        pip_exec_cmd = "python3 -m pip" if pip_available else ("pip3" if pip3_available else "")
                    else:
                        reason_codes.append("DEP_BOOTSTRAP_SUDO_FAILED")
                else:
                    reason_codes.append("DEP_BOOTSTRAP_SUDO_UNAVAILABLE")
            else:
                reason_codes.append("DEP_BOOTSTRAP_RUNTIME_SUDO_DISABLED")

        req_probe = runtime.run_command("bash -lc 'test -f requirements.txt'", cwd=repo_dir, timeout_sec=20)
        has_requirements = req_probe.rc == 0

        if pip_exec_cmd and has_requirements:
            _run_logged(
                "pip_upgrade_toolchain",
                f"{pip_exec_cmd} install -U pip setuptools wheel",
                repo_dir,
                1800,
                timeout_class="long",
            )
            install = _run_logged(
                "pip_install_requirements",
                f"{pip_exec_cmd} install -r requirements.txt",
                repo_dir,
                3600,
                timeout_class="long",
            )
            install_rc = install.rc
            if install.rc != 0:
                reason_codes.append("DEPENDENCY_INSTALL_FAILED")
                if compat_mode:
                    try:
                        req_text = runtime.read_text(f"{repo_dir}/requirements.txt")
                    except Exception:  # noqa: BLE001
                        req_text = ""
                    compat_text, compat_replacements, legacy_pin_incompatible = self._build_compat_requirements(
                        req_text,
                        profile=compat_profile,
                    )
                    if compat_replacements:
                        reason_codes.extend(["DEPENDENCY_COMPAT_FALLBACK_USED", "DEPENDENCY_LEGACY_PIN_INCOMPATIBLE"])
                        runtime.write_text(compat_requirements_path, compat_text)
                        trace.append(
                            f"compat_requirements_written: path={compat_requirements_path}; replacements={len(compat_replacements)}"
                        )
                        worklog_events.append(
                            {
                                "type": "install",
                                "ts": _utc_now(),
                                "details": {
                                    "step": "compat_requirements_mapping",
                                    "profile": compat_profile,
                                    "replacements": compat_replacements,
                                },
                                "result": {"rc": 0, "stdout_tail": "", "stderr_tail": ""},
                            }
                        )
                        compat_install = _run_logged(
                            "pip_install_compat_requirements",
                            f"{pip_exec_cmd} install -r {shlex.quote(compat_requirements_path)}",
                            repo_dir,
                            3600,
                            timeout_class="long",
                        )
                        install_rc = compat_install.rc
                        if compat_install.rc != 0:
                            reason_codes.append("DEPENDENCY_COMPAT_FALLBACK_FAILED")
                    elif legacy_pin_incompatible:
                        reason_codes.append("DEPENDENCY_LEGACY_PIN_INCOMPATIBLE")
        elif has_requirements and not pip_exec_cmd:
            reason_codes.append("PIP_NOT_AVAILABLE")

        snapshot_after = self.probe_python_capabilities(runtime, workspace_root)
        modules_ready = all(snapshot_after.get("required_modules_available", {}).values())
        install_ok = (not has_requirements) or (install_rc == 0)
        ready = modules_ready or (
            bool(snapshot_after.get("pip_available")) or bool(snapshot_after.get("pip_command_available"))
        ) and install_ok
        if not ready:
            reason_codes.append("DEPENDENCY_UNRESOLVED")
        if not apt_enable:
            reason_codes.append("DEPENDENCY_BOOTSTRAP_APT_DISABLED")
        if not bool(snapshot_after.get("pip_available")) and not bool(snapshot_after.get("pip_command_available")):
            reason_codes.append("PIP_NOT_AVAILABLE")

        solver_status = "ready" if ready else "failed_dependency_capability"
        if has_requirements and install_rc == 0 and compat_replacements:
            solver_status = "ready_with_compat_fallback"
        solver_payload = {
            "status": solver_status,
            "profile": compat_profile,
            "steps": trace,
            "compat_replacements": compat_replacements,
            "legacy_pin_incompatible": legacy_pin_incompatible,
            "sudo": sudo_diag,
            "reason_codes": _dedupe(reason_codes),
        }
        runtime.write_text(
            f"{outputs_dir}/dependency_solver.json",
            json.dumps(solver_payload, ensure_ascii=False, indent=2),
        )

        return {
            "enabled": True,
            "ready": ready,
            "has_requirements": has_requirements,
            "snapshot_after": snapshot_after,
            "trace": trace,
            "reason_codes": _dedupe(reason_codes),
            "log_path": log_path,
            "worklog_events": worklog_events,
            "sudo_diag": sudo_diag,
            "dependency_solver": solver_payload,
        }

    def probe_entrypoints_once(
        self,
        runtime,
        *,
        repo_dir: str,
        task_spec_path: str,
    ) -> dict[str, Any]:
        try:
            task_spec = json.loads(runtime.read_text(task_spec_path))
        except Exception:  # noqa: BLE001
            task_spec = {}
        rows: list[dict[str, Any]] = []
        worklog_events: list[dict[str, Any]] = []
        tasks = extract_task_items(task_spec)
        if not tasks:
            return {
                "runs": [],
                "worklog_events": [],
                "entrypoint_count": 0,
                "success_count": 0,
            }

        for idx, item in enumerate(tasks):
            run_cmd = str(item.get("command") or "").strip()
            if not run_cmd:
                run_cmd = "python3 -c 'import sys; sys.exit(2)'"
            run_id = str(item.get("task_id") or f"task_{idx + 1:02d}")
            entrypoint = str(item.get("entrypoint") or "").strip()
            if entrypoint and run_id.startswith("legacy_task_"):
                run_id = entrypoint
            timeout_class = str(item.get("timeout_class") or "medium")
            timeout_sec = timeout_for_class(timeout_class, self.entrypoint_probe_timeout_sec)

            started = time.time()
            result = runtime.run_command(
                f"bash -lc {shlex.quote(run_cmd)}",
                cwd=repo_dir,
                timeout_sec=timeout_sec,
            )
            elapsed = time.time() - started
            text = "\n".join([result.stdout or "", result.stderr or ""])
            dep_failed = _has_dependency_signal(text)
            status = "ok" if result.rc == 0 else ("failed_dependency" if dep_failed else "failed")
            reason_codes: list[str] = []
            if dep_failed:
                reason_codes.append("ENTRYPOINT_UNRUNNABLE_DEPENDENCY")
            elif result.rc != 0:
                reason_codes.append("ENTRYPOINT_PROBE_FAILED")

            rows.append(
                {
                    "run_id": run_id,
                    "command": run_cmd,
                    "params": {},
                    "cwd": repo_dir,
                    "exit_code": int(result.rc),
                    "status": status,
                    "runtime_sec": round(elapsed, 3),
                    "stdout_tail": _tail(result.stdout or "", 1200),
                    "stderr_tail": _tail(result.stderr or "", 1200),
                    "artifacts": [],
                    "metrics": {},
                    "reason_codes": reason_codes,
                }
            )
            worklog_events.append(
                {
                    "type": "run",
                    "ts": _utc_now(),
                    "details": {"entrypoint": run_id, "command": run_cmd},
                    "result": {"rc": result.rc, "status": status, "timeout_class": timeout_class},
                }
            )

        return {
            "runs": rows,
            "worklog_events": worklog_events,
            "entrypoint_count": len(tasks),
            "success_count": sum(1 for x in rows if int(x.get("exit_code", 1)) == 0),
        }

    @staticmethod
    def _load_json(runtime, path: str) -> dict[str, Any]:
        try:
            data = json.loads(runtime.read_text(path))
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            pass
        return {}

    def render_fallback_outputs(
        self,
        runtime,
        *,
        outputs_dir: str,
        claims_ir_path: str,
        claims_payload: dict[str, Any] | None = None,
        capability_snapshot: dict[str, Any],
        dependency_bootstrap_trace: list[str],
        runs: list[dict[str, Any]],
        worklog_events: list[dict[str, Any]],
        reason_codes: list[str],
        dependency_solver_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime.run_command(
            f"bash -lc {shlex.quote(f'mkdir -p {_quote(outputs_dir)}')}",
            cwd=outputs_dir.rsplit("/", 1)[0] if "/" in outputs_dir else "/",
            timeout_sec=30,
        )
        claims_data = claims_payload if isinstance(claims_payload, dict) else self._load_json(runtime, claims_ir_path)
        claims_list = claims_data.get("claims")
        if not isinstance(claims_list, list):
            claims_list = []

        run_manifest_codes = _dedupe(list(reason_codes) + ["DEPENDENCY_UNRESOLVED"])
        run_manifest = {"runs": runs, "reason_codes": run_manifest_codes}

        success_count = sum(1 for x in runs if int(x.get("exit_code", 1)) == 0)
        claim_rows: list[dict[str, Any]] = []
        for idx, item in enumerate(claims_list):
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id") or item.get("id") or f"claim_{idx}")
            required_metrics: list[str] = []
            rm = item.get("required_metrics")
            if isinstance(rm, list):
                required_metrics.extend([str(x) for x in rm if str(x).strip()])
            metric = item.get("metric")
            if isinstance(metric, str) and metric.strip() and metric.strip() not in required_metrics:
                required_metrics.append(metric.strip())
            evaluable = "partial" if success_count > 0 else "no"
            reason = (
                "partial evaluability: some entrypoints failed dependency bootstrap"
                if success_count > 0
                else "dependency unresolved in sandbox runtime"
            )
            claim_rows.append(
                {
                    "claim_id": claim_id,
                    "required_metrics": required_metrics,
                    "source": [f"{outputs_dir}/run_manifest.json", f"{outputs_dir}/dependency_solver.json"],
                    "evaluable": evaluable,
                    "reason": reason,
                }
            )
        claim_alignment = {
            "claims": claim_rows,
            "reason_codes": run_manifest_codes,
        }

        dependency_solver = dict(dependency_solver_payload or {})
        dependency_solver.setdefault("status", "failed_dependency_capability")
        dependency_solver.setdefault("steps", dependency_bootstrap_trace)
        dependency_solver["capability_snapshot"] = capability_snapshot
        dependency_solver["reason_codes"] = run_manifest_codes

        final_worklog = list(worklog_events)
        final_worklog.append(
            {
                "type": "output",
                "ts": _utc_now(),
                "details": {"stage": "capability_gate_fallback"},
                "result": {"files": ["run_manifest.json", "claim_alignment.json", "codex_worklog.jsonl", "patches.diff"]},
            }
        )
        worklog_text = ""
        if final_worklog:
            worklog_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in final_worklog) + "\n"

        try:
            exec_log = runtime.read_text(f"{outputs_dir}/codex_exec.log")
        except Exception:  # noqa: BLE001
            exec_log = ""
        exec_log += (
            "\n[runner] capability gate fallback activated\n"
            f"[runner] reason_codes={','.join(run_manifest_codes)}\n"
        )

        runtime.write_text(f"{outputs_dir}/run_manifest.json", json.dumps(run_manifest, ensure_ascii=False, indent=2))
        runtime.write_text(
            f"{outputs_dir}/claim_alignment.json",
            json.dumps(claim_alignment, ensure_ascii=False, indent=2),
        )
        runtime.write_text(f"{outputs_dir}/dependency_solver.json", json.dumps(dependency_solver, ensure_ascii=False, indent=2))
        runtime.write_text(f"{outputs_dir}/codex_worklog.jsonl", worklog_text)
        runtime.write_text(f"{outputs_dir}/patches.diff", "")
        runtime.write_text(f"{outputs_dir}/codex_exec.log", exec_log)
        runtime.write_text(f"{outputs_dir}/codex_main.log", exec_log)
        try:
            runtime.read_text(f"{outputs_dir}/codex_repair.log")
        except Exception:  # noqa: BLE001
            runtime.write_text(f"{outputs_dir}/codex_repair.log", "")
        runtime.write_text(
            f"{outputs_dir}/capability_probe.json",
            json.dumps(capability_snapshot, ensure_ascii=False, indent=2),
        )
        if dependency_bootstrap_trace:
            self._append_remote_text(
                runtime,
                f"{outputs_dir}/dependency_bootstrap.log",
                "\n".join(dependency_bootstrap_trace) + "\n",
            )

        return {
            "run_count": len(runs),
            "claim_count": len(claim_rows),
            "success_count": success_count,
            "reason_codes": run_manifest_codes,
        }
