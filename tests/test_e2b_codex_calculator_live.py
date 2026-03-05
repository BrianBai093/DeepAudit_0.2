from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    pytest = None

from p2c.runtime.e2b_runtime import E2BRuntime


def _pick_workspace_root(runtime: E2BRuntime) -> str:
    override = (os.getenv("P2C_WORKSPACE_ROOT") or "").strip()
    candidates = [override] if override else []
    candidates.extend(["/workspace", "/home/user/workspace", "/home/sandbox/workspace", "/tmp/workspace"])
    seen: set[str] = set()
    uniq = [x for x in candidates if x and not (x in seen or seen.add(x))]
    errors: list[str] = []
    for root in uniq:
        try:
            probe = runtime.run_command(
                f"bash -lc {shlex.quote(f'mkdir -p {shlex.quote(root)} && test -w {shlex.quote(root)}')}",
                cwd="/",
                timeout_sec=20,
            )
            if probe.rc == 0:
                return root
            errors.append(f"{root}: rc={probe.rc} stderr={probe.stderr[:120]}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{root}: {e}")
            continue
    detail = "; ".join(errors[-4:]) if errors else "no candidates checked"
    raise RuntimeError(f"no writable workspace root found from: {uniq}; detail={detail}")


def _approval_args(help_text: str) -> list[str]:
    #if "--approval-mode" in help_text:
        #return ["--approval-mode", "full-auto"]
    #if "--full-auto" in help_text:
        #return ["--full-auto"]
    return []


def _build_codex_cmd(*, prompt: str, help_text: str) -> str:
    model = (os.getenv("P2C_CODEX_MODEL") or "gpt-5.1").strip()
    # Default: do not pass --sandbox in this live test; allow CLI native defaults.
    # Optional override: set P2C_CODEX_SANDBOX_MODE (e.g. workspace-write).
    sandbox_mode = (os.getenv("P2C_CODEX_SANDBOX_MODE") or "").strip()
    unsafe_bypass = (os.getenv("P2C_CODEX_UNSAFE_BYPASS", "1") or "1").strip() != "0"
    parts = ["codex", "exec"]
    parts.extend(_approval_args(help_text))
    if sandbox_mode and "--sandbox" in help_text:
        parts.extend(["--sandbox", sandbox_mode])
    if unsafe_bypass and "--dangerously-bypass-approvals-and-sandbox" in help_text:
        parts.append("--dangerously-bypass-approvals-and-sandbox")
    if "--skip-git-repo-check" in help_text:
        parts.append("--skip-git-repo-check")
    if model:
        parts.extend(["-m", model])
    parts.append(prompt)
    return " ".join(shlex.quote(x) for x in parts)


def _stream_until_done(
    runtime: E2BRuntime,
    *,
    cwd: str,
    pid_path: str,
    exit_path: str,
    log_path: str,
    timeout_sec: int,
    poll_sec: int,
) -> int:
    deadline = time.time() + timeout_sec
    cursor = 0
    probe_cmd = f"if [ -f {shlex.quote(exit_path)} ]; then echo READY; else echo WAIT; fi"
    while time.time() < deadline:
        try:
            text = runtime.read_text(log_path)
            if len(text) < cursor:
                cursor = 0
            delta = text[cursor:]
            cursor = len(text)
            for line in delta.splitlines():
                if line.strip():
                    print(f"[codex-live] {line}", flush=True)
        except Exception:
            pass

        exited = runtime.run_command(
            f"bash -lc {shlex.quote(probe_cmd)}",
            cwd=cwd,
            timeout_sec=20,
        )
        if "READY" in (exited.stdout or ""):
            raw = runtime.read_text(exit_path).strip()
            try:
                return int(raw)
            except ValueError:
                return 1

        time.sleep(poll_sec)

    runtime.run_command(
        "bash -lc "
        + shlex.quote(
            f"if [ -f {shlex.quote(pid_path)} ]; then "
            f"pid=$(cat {shlex.quote(pid_path)} 2>/dev/null || true); "
            "if [ -n \"$pid\" ]; then kill \"$pid\" 2>/dev/null || true; kill -9 \"$pid\" 2>/dev/null || true; fi; "
            "fi"
        ),
        cwd=cwd,
        timeout_sec=20,
    )
    raise TimeoutError(f"codex live test timed out after {timeout_sec}s")


def run_live_codex_calculator(output_file: Path, timeout_sec: int = 600, poll_sec: int = 2) -> Path:
    key = (os.getenv("E2B_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("E2B_API_KEY is required")
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    runtime = E2BRuntime(timeout_sec=max(1200, timeout_sec + 300))
    runtime.ensure_started()
    try:
        workspace_root = _pick_workspace_root(runtime)
        repo_dir = f"{workspace_root}/repo_live_calc"
        outputs_dir = f"{workspace_root}/outputs_live_calc"
        setup = runtime.run_command(
            f"bash -lc {shlex.quote(f'mkdir -p {shlex.quote(repo_dir)} {shlex.quote(outputs_dir)}')}",
            cwd=workspace_root,
            timeout_sec=30,
        )
        if setup.rc != 0:
            raise RuntimeError(f"failed to setup workspace: {setup.stderr}")

        # Create a minimal repository root marker.
        runtime.run_command("bash -lc 'printf \"# live codex test\\n\" > README.md'", cwd=repo_dir, timeout_sec=20)

        help_out = runtime.run_command("bash -lc 'codex exec --help 2>&1 || true'", cwd=repo_dir, timeout_sec=30)
        help_text = (help_out.stdout or "") + "\n" + (help_out.stderr or "")

        if "--skip-git-repo-check" not in help_text:
            runtime.run_command("bash -lc 'git init >/dev/null 2>&1 || true'", cwd=repo_dir, timeout_sec=20)

        prompt = (
            "Create a file named calculator.py in the current directory. "
            "Requirements: define add(a,b), subtract(a,b), multiply(a,b), divide(a,b) where divide raises ValueError "
            "on division by zero. Add a CLI using argparse so command format is: "
            "python calculator.py <op> <a> <b> with op in {add,sub,mul,div}. "
            "Print only the numeric result. Do not create any other files."
        )
        codex_cmd = _build_codex_cmd(prompt=prompt, help_text=help_text)

        stage_log = f"{outputs_dir}/codex_calc.log"
        merged_log = f"{outputs_dir}/codex_exec.log"
        prep = runtime.run_command(
            f"bash -lc {shlex.quote(f'mkdir -p {shlex.quote(outputs_dir)} && : > {shlex.quote(stage_log)} && : > {shlex.quote(merged_log)}')}",
            cwd=workspace_root,
            timeout_sec=20,
        )
        if prep.rc != 0:
            raise RuntimeError(
                f"failed to prepare logs rc={prep.rc}; stdout={prep.stdout[-300:]!r}; stderr={prep.stderr[-300:]!r}"
            )
        run_script = (
            f"{codex_cmd} 2>&1 | tee -a {shlex.quote(stage_log)} | tee -a {shlex.quote(merged_log)}"
        )
        try:
            run_out = runtime.run_command(
                f"bash -lc {shlex.quote(run_script)}",
                cwd=repo_dir,
                timeout_sec=0,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "failed to run codex exec in sandbox; "
                f"run_script={run_script!r}; error={e}"
            ) from e
        if run_out.rc != 0:
            try:
                stage_tail = runtime.read_text(stage_log)[-2500:]
            except Exception:
                stage_tail = ""
            raise RuntimeError(
                f"codex exec failed rc={run_out.rc}; "
                f"stdout_tail={str(run_out.stdout)[-500:]!r}; stderr_tail={str(run_out.stderr)[-500:]!r}; "
                f"log_tail={stage_tail!r}; run_script={run_script!r}"
            )

        exists = runtime.run_command(
            "bash -lc 'if [ -f calculator.py ]; then echo READY; else echo MISSING; fi'",
            cwd=repo_dir,
            timeout_sec=20,
        )
        if "READY" not in (exists.stdout or ""):
            tail = runtime.read_text(stage_log)[-2000:]
            raise RuntimeError(f"calculator.py not created; log_tail={tail!r}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        runtime.download_file(f"{repo_dir}/calculator.py", output_file)

        # Download sandbox logs for audit.
        runtime.download_file(stage_log, output_file.parent / "codex_calc.log")
        runtime.download_file(merged_log, output_file.parent / "codex_exec.log")
        return output_file
    finally:
        runtime.close()


if pytest is not None:

    @pytest.mark.skipif(
        (os.getenv("RUN_E2B_LIVE_TEST") or "0").strip() != "1",
        reason="set RUN_E2B_LIVE_TEST=1 to run live E2B Codex test",
    )
    def test_e2b_codex_writes_calculator() -> None:
        out = Path("artifacts/e2b_live_calc/calculator.py")
        created = run_live_codex_calculator(out)
        content = created.read_text(encoding="utf-8", errors="ignore")
        assert "def add(" in content
        assert "def subtract(" in content
        assert "def multiply(" in content
        assert "def divide(" in content
        assert "argparse" in content


def _run_as_script() -> int:
    out = Path("artifacts/e2b_live_calc/calculator.py")
    created = run_live_codex_calculator(out)
    content = created.read_text(encoding="utf-8", errors="ignore")
    required = ["def add(", "def subtract(", "def multiply(", "def divide(", "argparse"]
    missing = [x for x in required if x not in content]
    if missing:
        print(f"[e2b-live] calculator.py created but missing markers: {missing}")
        return 2
    print(f"[e2b-live] success: wrote {created}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_run_as_script())
