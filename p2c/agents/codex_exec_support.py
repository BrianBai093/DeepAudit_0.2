from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any, Callable


def _tail(text: str, n: int = 800) -> str:
    if not text:
        return ""
    return text[-n:]


def _quote(path: str) -> str:
    return shlex.quote(path)


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
        needles = [
            "dependency",
            "install",
            "resolutionimpossible",
            "no matching distribution found",
            "could not find a version",
            "module not found",
            "modulenotfounderror",
            "importerror",
            "pip",
        ]
        return any(n in s for s in signals for n in needles)

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
    ) -> None:
        payload: dict[str, Any] = {
            "stage": stage,
            "last_command": last_command,
            "exit_code": exit_code,
            "stdout_tail": _tail(stdout_tail, n=3000),
            "stderr_tail": _tail(stderr_tail, n=3000),
            "codex_exec_log_tail": _tail(codex_exec_log_tail, n=5000),
            "pip_log_tail": _tail(pip_log_tail, n=3000),
            "reason_codes": list(reason_codes),
        }
        self.artifacts.write_json("execution/codex_failure.json", payload)
        self.artifacts.append_text(
            "execution/run.log",
            (
                "[run_codex_exec] failure snapshot written: "
                f"stage={stage} exit_code={exit_code} "
                f"reason_codes={','.join(reason_codes[:8])}\n"
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
        )


class CodexBackgroundExecutor:
    def __init__(self, log_fn: Callable[[str, str], None]):
        self.log_fn = log_fn

    @staticmethod
    def stage_log_name(label: str) -> str:
        if label == "main":
            return "codex_main.log"
        if label == "repair":
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
        try:
            launcher = runtime.run_command(
                launcher_cmd,
                cwd=cwd,
                timeout_sec=60,
            )
        except Exception as e:  # noqa: BLE001
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

    def _poll_background(
        self,
        runtime,
        *,
        pid_path: str,
        exit_path: str,
        log_path: str,
        label: str,
        cwd: str,
        timeout_sec: int,
        poll_sec: int = 5,
    ) -> tuple[int, int, bool]:
        deadline = time.time() + timeout_sec
        polls = 0
        cursor = 0
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
            cursor = self._emit_log_delta(runtime, log_path=log_path, cursor=cursor, label=label)
            exited = runtime.run_command(exit_probe_cmd, cwd=cwd, timeout_sec=20)
            if exited.rc == 0:
                cursor = self._emit_log_delta(runtime, log_path=log_path, cursor=cursor, label=label)
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
            label=label,
            cwd=workspace_root,
            timeout_sec=timeout_sec,
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

