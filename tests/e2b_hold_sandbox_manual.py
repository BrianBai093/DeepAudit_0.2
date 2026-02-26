from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from p2c.agents.phase2.prepare_sandbox import PrepareSandboxAgent
from p2c.runtime.e2b_runtime import E2BRuntime


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.task_spec and args.claims_ir:
        task_spec = Path(args.task_spec)
        claims_ir = Path(args.claims_ir)
    else:
        if not args.run_id:
            raise ValueError("Either provide both --task_spec/--claims_ir or provide --run_id.")
        run_root = Path(args.artifacts_dir) / args.run_id
        task_spec = run_root / "task/task_spec.json"
        claims_ir = run_root / "fingerprint/claims_ir.json"

    if not task_spec.exists():
        raise FileNotFoundError(f"task_spec not found: {task_spec}")
    if not claims_ir.exists():
        raise FileNotFoundError(f"claims_ir not found: {claims_ir}")
    return task_spec, claims_ir


def _setup_workspace(runtime: E2BRuntime, repo_dir: Path, task_spec: Path, claims_ir: Path, include_git: bool) -> dict:
    workspace_root = PrepareSandboxAgent._pick_workspace_root(runtime)
    workspace_repo_dir = f"{workspace_root}/repo"
    workspace_inputs_dir = f"{workspace_root}/inputs"
    workspace_outputs_dir = f"{workspace_root}/outputs"

    repo_q = shlex.quote(workspace_repo_dir)
    inputs_q = shlex.quote(workspace_inputs_dir)
    outputs_q = shlex.quote(workspace_outputs_dir)
    mk = runtime.run_command(
        f"mkdir -p {repo_q} {inputs_q} {outputs_q}",
        cwd=workspace_root,
        timeout_sec=30,
    )
    if mk.rc != 0:
        raise RuntimeError(f"failed to prepare workspace dirs: {mk.stderr[:300]}")

    repo_excludes = None if include_git else [".git", ".git/**"]
    runtime.upload_dir(local_dir=repo_dir, remote_dir=workspace_repo_dir, exclude_globs=repo_excludes)
    runtime.upload_file(local_file=task_spec, remote_path=f"{workspace_inputs_dir}/task_spec.json")
    runtime.upload_file(local_file=claims_ir, remote_path=f"{workspace_inputs_dir}/claims_ir.json")

    checks = [
        f"test -d {shlex.quote(workspace_repo_dir)}",
        f"test -f {shlex.quote(workspace_inputs_dir + '/task_spec.json')}",
        f"test -f {shlex.quote(workspace_inputs_dir + '/claims_ir.json')}",
    ]
    probe = runtime.run_command(" && ".join(checks), cwd=workspace_root, timeout_sec=20)
    if probe.rc != 0:
        raise RuntimeError("sandbox upload verification failed: repo/json files not found in workspace")

    return {
        "workspace_root": workspace_root,
        "workspace_repo_dir": workspace_repo_dir,
        "workspace_inputs_dir": workspace_inputs_dir,
        "workspace_outputs_dir": workspace_outputs_dir,
    }


def run(args: argparse.Namespace) -> int:
    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists() or not repo_dir.is_dir():
        raise FileNotFoundError(f"repo_dir not found: {repo_dir}")

    task_spec, claims_ir = _resolve_inputs(args)

    timeout_sec = int(args.sandbox_timeout_sec)
    if timeout_sec > 3600:
        print(
            f"[hold-sandbox] requested timeout {timeout_sec}s exceeds E2B limit; clamped to 3600s",
            flush=True,
        )
        timeout_sec = 3600
    runtime = E2BRuntime(timeout_sec=timeout_sec)
    runtime.ensure_started()
    metadata = runtime.metadata()
    sandbox_id = metadata.get("sandbox_id")

    workspace = _setup_workspace(
        runtime=runtime,
        repo_dir=repo_dir,
        task_spec=task_spec,
        claims_ir=claims_ir,
        include_git=args.include_git,
    )

    info = {
        "sandbox_id": sandbox_id,
        "runtime": metadata,
        "repo_local": str(repo_dir),
        "task_spec_local": str(task_spec),
        "claims_ir_local": str(claims_ir),
        "workspace": workspace,
        "include_git": bool(args.include_git),
        "connection_hint": f"e2b sandbox connect {sandbox_id}",
    }

    info_path = Path(args.info_out)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[hold-sandbox] sandbox ready", flush=True)
    print(f"[hold-sandbox] sandbox_id={sandbox_id}", flush=True)
    print(f"[hold-sandbox] workspace_root={workspace['workspace_root']}", flush=True)
    print(f"[hold-sandbox] info_file={info_path}", flush=True)
    print(f"[hold-sandbox] connect_hint=e2b sandbox connect {sandbox_id}", flush=True)
    print("[hold-sandbox] no further actions will be executed.", flush=True)
    print("[hold-sandbox] keeping sandbox alive for manual E2B CLI debugging...", flush=True)

    try:
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
        else:
            while True:
                time.sleep(5)
    except KeyboardInterrupt:
        print("[hold-sandbox] interrupted by user.", flush=True)
    finally:
        if args.close_on_exit:
            runtime.close()
            print("[hold-sandbox] sandbox closed (--close_on_exit enabled).", flush=True)
        else:
            print("[hold-sandbox] sandbox left running (default behavior).", flush=True)
            print(f"[hold-sandbox] reconnect with: e2b sandbox connect {sandbox_id}", flush=True)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an E2B Codex sandbox, upload repo + task_spec.json + claims_ir.json, "
            "then hold the process for manual CLI debugging."
        )
    )
    parser.add_argument("--repo_dir", default="Target/code")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--task_spec", default="")
    parser.add_argument("--claims_ir", default="")
    parser.add_argument("--include_git", action="store_true")
    parser.add_argument("--sandbox_timeout_sec", type=int, default=1800)
    parser.add_argument("--hold_seconds", type=int, default=0, help="0 means hold forever.")
    parser.add_argument("--close_on_exit", action="store_true")
    parser.add_argument("--info_out", default="artifacts/e2b_manual_hold/sandbox_info.json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
