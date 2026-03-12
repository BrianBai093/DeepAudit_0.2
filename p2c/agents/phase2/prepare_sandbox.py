from __future__ import annotations

import json
import os
import platform
import shlex
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import DataManifest, DataManifestEntry, SystemInfo

SYSTEM_PROMPT = (
    "You summarize runtime system info. Return strict JSON and avoid fabrication."
)

USER_PROMPT_TEMPLATE = "Input: local runtime info. Output: execution/system_info.json"

DATA_DIR_NAMES = {"data", "dataset", "datasets", "input", "inputs"}
DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".npy", ".npz", ".txt", ".pt", ".pth"}

PYTHON_SHIM = """#!/usr/bin/env sh
set -eu
exec python3 "$@"
"""

PIP_SHIM = """#!/usr/bin/env sh
set -eu
if python3 -m pip --version >/dev/null 2>&1; then
  exec python3 -m pip "$@"
fi
exec pip3 "$@"
"""

APPLY_PATCH_SHIM = """#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python3 "${SCRIPT_DIR}/p2c_apply_patch.py" "$@"
"""

APPLY_PATCH_PY = """#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def _to_unified(payload: str) -> str:
    if "*** Begin Patch" not in payload:
        return payload

    lines = payload.splitlines()
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("*** Begin Patch") or line.startswith("*** End Patch"):
            idx += 1
            continue
        if line.startswith("*** Update File: "):
            path = line[len("*** Update File: ") :].strip()
            out.append(f"--- {path}")
            out.append(f"+++ {path}")
            idx += 1
            if idx < len(lines) and lines[idx].startswith("*** Move to: "):
                idx += 1
            while idx < len(lines):
                cur = lines[idx]
                if cur.startswith("*** "):
                    break
                if cur.startswith("@@") or cur.startswith("+") or cur.startswith("-") or cur.startswith(" "):
                    out.append(cur)
                idx += 1
            continue
        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: ") :].strip()
            added: list[str] = []
            idx += 1
            while idx < len(lines):
                cur = lines[idx]
                if cur.startswith("*** "):
                    break
                if cur.startswith("+"):
                    added.append(cur[1:])
                idx += 1
            out.append("--- /dev/null")
            out.append(f"+++ {path}")
            out.append(f"@@ -0,0 +1,{len(added)} @@")
            out.extend([f"+{x}" for x in added])
            continue
        idx += 1
    if not out:
        return ""
    return "\\n".join(out) + "\\n"


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    patch_text = _to_unified(raw)
    if not patch_text.strip():
        sys.stderr.write("p2c_apply_patch: empty converted patch\\n")
        return 2
    proc = subprocess.run(["patch", "-p0"], input=patch_text, text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
"""

CODEX_EXECUTION_SKILL = """# Codex Execution Skill

Follow these rules before running the repository:

1. If an error contains `No module named ...`, identify the most likely installable package for that module, install it with `python3 -m pip` into the sandbox user's local environment, and retry. Do not stop at the first import error if a reasonable package guess exists.
2. If the README contains data download, dataset setup, processing, or vectorization commands, execute those README steps during setup. Treat documented dataset preparation as required unless the README explicitly says the step is optional or unavailable in this environment.
3. Do not create a virtual environment. Install tools and packages only into the sandbox user's local environment, for example with `--user` or under `~/.local`.
4. Keep logs compact and record the actual install and data-setup commands you ran.
"""


class PrepareSandboxAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="prepare_sandbox", *args, **kwargs)

    @staticmethod
    def _discover_data_entries(repo_dir: Path) -> list[Path]:
        picked: list[Path] = []
        seen: set[Path] = set()

        # Prefer canonical data directories.
        for p in sorted(repo_dir.rglob("*")):
            if p.is_dir() and p.name.lower() in DATA_DIR_NAMES:
                if p not in seen:
                    picked.append(p)
                    seen.add(p)

        # Add loose data files not already covered by picked directories.
        for p in sorted(repo_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in DATA_SUFFIXES:
                continue
            if any(parent in seen for parent in p.parents):
                continue
            if p not in seen:
                picked.append(p)
                seen.add(p)

        return picked

    @staticmethod
    def _pick_workspace_root(runtime) -> str:
        override = (os.getenv("P2C_WORKSPACE_ROOT") or "").strip()
        preferred = [override] if override else []
        preferred.extend(["/workspace", "/home/user/workspace", "/home/sandbox/workspace", "/tmp/workspace"])
        # Deduplicate while preserving order.
        uniq: list[str] = []
        for p in preferred:
            if p and p not in uniq:
                uniq.append(p)
        errors: list[str] = []
        for root in uniq:
            try:
                root_q = shlex.quote(root)
                probe = runtime.run_command(
                    f"mkdir -p {root_q} && test -w {root_q}",
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
        raise RuntimeError(f"no writable workspace root found; candidates={uniq}; detail={detail}")

    @staticmethod
    def _rewrite_code_scoped_task_spec(payload: dict, *, uploaded_code_subdir: bool) -> dict:
        if not isinstance(payload, dict):
            return {}
        if not uploaded_code_subdir:
            return payload

        def _strip_code_prefix(value: object) -> object:
            raw = str(value or "").strip()
            if not raw:
                return value
            if raw == "code":
                return "."
            if raw.startswith("code/"):
                return raw[len("code/") :]
            return value

        def _rewrite_command(value: object) -> object:
            raw = str(value or "").strip()
            if not raw:
                return value
            return raw.replace(" code/", " ").replace("code/", "", 1) if raw.startswith("code/") else raw.replace(" code/", " ")

        rewritten = json.loads(json.dumps(payload))

        for row in rewritten.get("tasks") or []:
            if not isinstance(row, dict):
                continue
            row["entrypoint"] = _strip_code_prefix(row.get("entrypoint"))
            row["command"] = _rewrite_command(row.get("command"))
            if str(row.get("cwd") or "").strip() == "code":
                row["cwd"] = "."

        for row in rewritten.get("entrypoints") or []:
            if not isinstance(row, dict):
                continue
            row["path"] = _strip_code_prefix(row.get("path"))
            row["command"] = _rewrite_command(row.get("command"))
            if str(row.get("cwd") or "").strip() == "code":
                row["cwd"] = "."
            entrypoint_id = str(row.get("entrypoint_id") or "")
            if entrypoint_id.startswith("python-file:code/"):
                row["entrypoint_id"] = "python-file:" + entrypoint_id[len("python-file:code/") :]

        constraints = rewritten.get("constraints")
        if isinstance(constraints, dict):
            if str(constraints.get("allowed_modification_scope") or "").strip() == "Target/code":
                constraints["allowed_modification_scope"] = "repo"

        notes = rewritten.get("selection_notes")
        if isinstance(notes, list):
            new_notes: list[object] = []
            for note in notes:
                raw = str(note or "")
                new_notes.append(raw.replace("code/", "", 1) if "primary_entrypoint_path=code/" in raw else raw)
            rewritten["selection_notes"] = new_notes

        return rewritten

    def execute(self, ctx: dict) -> dict:
        runtime = ensure_runtime(ctx, self.artifacts)
        if (getattr(runtime, "backend_name", "") or "").lower() != "e2b":
            raise RuntimeError("prepare_sandbox requires P2C_RUNTIME_BACKEND=e2b")
        llm_schema = {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["notes", "reason_codes"],
        }
        llm_data, _ = self.safe_chat_json(llm_schema, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        repo_dir = Path(ctx.get("repo_dir", ""))
        if not repo_dir.exists():
            raise FileNotFoundError(f"repo_dir not found: {repo_dir}")

        task_spec_local = self.artifacts.path("task/task_spec.json")
        metric_contract_local = self.artifacts.path("task/metric_contract.json")
        claims_ir_local = self.artifacts.path("fingerprint/claims_ir.json")
        if not task_spec_local.exists():
            raise FileNotFoundError(f"missing required artifact: {task_spec_local}")
        if not metric_contract_local.exists():
            raise FileNotFoundError(f"missing required artifact: {metric_contract_local}")

        workspace_root = self._pick_workspace_root(runtime)
        workspace_repo_dir = f"{workspace_root}/repo"
        workspace_data_dir = f"{workspace_root}/data"
        workspace_inputs_dir = f"{workspace_root}/inputs"
        workspace_outputs_dir = f"{workspace_root}/outputs"
        workspace_bin_dir = f"{workspace_root}/bin"

        ctx["workspace_root"] = workspace_root
        ctx["workspace_repo_dir"] = workspace_repo_dir
        ctx["workspace_data_dir"] = workspace_data_dir
        ctx["workspace_inputs_dir"] = workspace_inputs_dir
        ctx["workspace_outputs_dir"] = workspace_outputs_dir
        ctx["workspace_bin_dir"] = workspace_bin_dir
        # Keep backward-compatible keys for agents that still look for runtime_*.
        ctx["runtime_repo_dir"] = workspace_repo_dir

        repo_q = shlex.quote(workspace_repo_dir)
        data_q = shlex.quote(workspace_data_dir)
        inputs_q = shlex.quote(workspace_inputs_dir)
        outputs_q = shlex.quote(workspace_outputs_dir)
        bin_q = shlex.quote(workspace_bin_dir)
        mk = runtime.run_command(
            f"mkdir -p {repo_q} {data_q} {inputs_q} {outputs_q} {bin_q}",
            cwd=workspace_root,
            timeout_sec=30,
        )
        if mk.rc != 0:
            raise RuntimeError(f"failed to prepare workspace dirs: {mk.stderr[:300]}")

        # Install deterministic tool shims to avoid command guessing.
        tools_local_dir = "execution/tools"
        python_shim_local = self.artifacts.write_text(f"{tools_local_dir}/python", PYTHON_SHIM)
        pip_shim_local = self.artifacts.write_text(f"{tools_local_dir}/pip", PIP_SHIM)
        apply_patch_shim_local = self.artifacts.write_text(f"{tools_local_dir}/apply_patch", APPLY_PATCH_SHIM)
        apply_patch_py_local = self.artifacts.write_text(f"{tools_local_dir}/p2c_apply_patch.py", APPLY_PATCH_PY)

        tools_remote_dir = f"{workspace_inputs_dir}/tools"
        runtime.upload_file(local_file=python_shim_local, remote_path=f"{tools_remote_dir}/python")
        runtime.upload_file(local_file=pip_shim_local, remote_path=f"{tools_remote_dir}/pip")
        runtime.upload_file(local_file=apply_patch_shim_local, remote_path=f"{tools_remote_dir}/apply_patch")
        runtime.upload_file(local_file=apply_patch_py_local, remote_path=f"{tools_remote_dir}/p2c_apply_patch.py")

        install_tools = runtime.run_command(
            "bash -lc "
            + shlex.quote(
                f"mkdir -p {workspace_bin_dir} {tools_remote_dir} && "
                f"cp {tools_remote_dir}/python {workspace_bin_dir}/python && "
                f"cp {tools_remote_dir}/pip {workspace_bin_dir}/pip && "
                f"cp {tools_remote_dir}/apply_patch {workspace_bin_dir}/apply_patch && "
                f"cp {tools_remote_dir}/p2c_apply_patch.py {workspace_bin_dir}/p2c_apply_patch.py && "
                f"chmod +x {workspace_bin_dir}/python {workspace_bin_dir}/pip "
                f"{workspace_bin_dir}/apply_patch {workspace_bin_dir}/p2c_apply_patch.py"
            ),
            cwd=workspace_root,
            timeout_sec=30,
        )
        if install_tools.rc != 0:
            raise RuntimeError(f"failed to install tool shims: {install_tools.stderr[:300]}")

        tool_summary_probe = runtime.run_command(
            "bash -lc "
            + shlex.quote(
                f"PATH={workspace_bin_dir}:$PATH; "
                "for t in python python3 pip pip3 apply_patch; do "
                "  p=$(command -v \"$t\" 2>/dev/null || true); "
                "  if [ -n \"$p\" ]; then echo \"$t=$p\"; else echo \"$t=MISSING\"; fi; "
                "done"
            ),
            cwd=workspace_root,
            timeout_sec=20,
        )
        tool_summary = "; ".join([x.strip() for x in (tool_summary_probe.stdout or "").splitlines() if x.strip()])
        self.artifacts.append_text("execution/run.log", f"[prepare_sandbox] toolchain={tool_summary}\n")

        upload_repo_dir = repo_dir / "code" if (repo_dir / "code").is_dir() else repo_dir
        uploaded_code_subdir = upload_repo_dir != repo_dir

        raw_task_spec_payload = self.artifacts.read_json("task/task_spec.json")
        sandbox_task_spec_payload = self._rewrite_code_scoped_task_spec(
            raw_task_spec_payload,
            uploaded_code_subdir=uploaded_code_subdir,
        )
        sandbox_task_spec_local = self.artifacts.write_json(
            "execution/task_spec.sandbox.json",
            sandbox_task_spec_payload,
        )
        codex_skill_local = self.artifacts.write_text(
            "execution/codex_execution_skill.md",
            CODEX_EXECUTION_SKILL,
        )
        ctx["workspace_task_spec_remote"] = f"{workspace_inputs_dir}/task_spec.json"
        ctx["workspace_task_spec_local_artifact"] = "execution/task_spec.sandbox.json"
        ctx["workspace_codex_skill_remote"] = f"{workspace_inputs_dir}/codex_execution_skill.md"
        ctx["workspace_codex_skill_local_artifact"] = "execution/codex_execution_skill.md"

        include_git = (os.getenv("P2C_INCLUDE_GIT") or "").strip() == "1"
        repo_excludes = None if include_git else [".git", ".git/**"]
        self.artifacts.append_text(
            "execution/run.log",
            f"[prepare_sandbox] repo upload mode=local_dir include_git={1 if include_git else 0} "
            f"upload_root={upload_repo_dir}\n",
        )
        runtime.upload_dir(local_dir=upload_repo_dir, remote_dir=workspace_repo_dir, exclude_globs=repo_excludes)
        runtime.upload_file(local_file=sandbox_task_spec_local, remote_path=f"{workspace_inputs_dir}/task_spec.json")
        runtime.upload_file(local_file=codex_skill_local, remote_path=f"{workspace_inputs_dir}/codex_execution_skill.md")
        runtime.upload_file(local_file=metric_contract_local, remote_path=f"{workspace_inputs_dir}/metric_contract.json")
        upload_claims = (os.getenv("P2C_UPLOAD_CLAIMS_TO_SANDBOX") or "").strip() == "1"
        if upload_claims and claims_ir_local.exists():
            runtime.upload_file(local_file=claims_ir_local, remote_path=f"{workspace_inputs_dir}/claims_ir.json")
        self.artifacts.append_text(
            "execution/run.log",
            f"[prepare_sandbox] upload_claims_to_sandbox={1 if upload_claims else 0}\n",
        )

        entries: list[DataManifestEntry] = []
        for src in self._discover_data_entries(upload_repo_dir):
            rel = src.relative_to(upload_repo_dir)
            remote = f"{workspace_data_dir}/{str(rel).replace('\\', '/')}"
            try:
                if src.is_dir():
                    runtime.upload_dir(local_dir=src, remote_dir=remote)
                    size = None
                else:
                    runtime.upload_file(local_file=src, remote_path=remote)
                    size = src.stat().st_size
                entries.append(
                    DataManifestEntry(
                        path=str(rel),
                        exists=True,
                        size_bytes=size,
                        sandbox_path=remote,
                    )
                )
            except Exception:  # noqa: BLE001
                entries.append(
                    DataManifestEntry(
                        path=str(rel),
                        exists=False,
                        size_bytes=None,
                        sandbox_path=remote,
                    )
                )

        manifest = DataManifest(
            entries=entries,
            unresolved=len(entries) == 0,
            reason_codes=["NO_DATA_FILES_DISCOVERED"] if len(entries) == 0 else [],
        )
        self.artifacts.write_json("execution/data_manifest.json", manifest.model_dump())

        mem_gb = None
        try:
            probe = runtime.run_command(
                "python3 - <<'PY'\nimport os,platform\nprint(platform.system())\nprint(platform.release())\nprint(platform.python_version())\ntry:\n p=os.sysconf('SC_PAGE_SIZE')\n n=os.sysconf('SC_PHYS_PAGES')\n print(round((p*n)/(1024**3),2))\nexcept Exception:\n print('')\nPY",
                cwd=workspace_repo_dir,
                timeout_sec=30,
            )
            lines = [x.strip() for x in (probe.stdout or "").splitlines() if x.strip()]
            platform_name = lines[0] if len(lines) > 0 else platform.system()
            release = lines[1] if len(lines) > 1 else platform.release()
            pyver = lines[2] if len(lines) > 2 else platform.python_version()
            if len(lines) > 3:
                try:
                    mem_gb = float(lines[3])
                except ValueError:
                    mem_gb = None
        except Exception:  # noqa: BLE001
            platform_name = platform.system()
            release = platform.release()
            pyver = platform.python_version()

        info = SystemInfo(
            platform=platform_name,
            platform_release=release,
            python_version=pyver,
            cpu_count=None,
            memory_gb=mem_gb,
            reason_codes=(llm_data or {}).get("reason_codes", []),
        )
        self.artifacts.write_json("execution/system_info.json", info.model_dump())

        return {
            "system_info": info.model_dump(),
            "data_manifest": manifest.model_dump(),
            "workspace": {
                "repo": workspace_repo_dir,
                "data": workspace_data_dir,
                "inputs": workspace_inputs_dir,
                "outputs": workspace_outputs_dir,
                "bin": workspace_bin_dir,
            },
        }
