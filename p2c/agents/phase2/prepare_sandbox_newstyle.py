from __future__ import annotations

import platform
import shlex
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.prepare_sandbox import (
    APPLY_PATCH_PY,
    APPLY_PATCH_SHIM,
    CODEX_EXECUTION_SKILL,
    PIP_SHIM,
    PYTHON_SHIM,
    PrepareSandboxAgent,
)
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import SystemInfo

SYSTEM_PROMPT = "You summarize runtime system info. Return strict JSON and avoid fabrication."
USER_PROMPT_TEMPLATE = "Input: local runtime info. Output: execution/system_info.json"


class PrepareSandboxNewstyleAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="prepare_sandbox_newstyle", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        runtime = ensure_runtime(ctx, self.artifacts)
        if (getattr(runtime, "backend_name", "") or "").lower() != "e2b":
            raise RuntimeError("prepare_sandbox_newstyle requires P2C_RUNTIME_BACKEND=e2b")

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
        if not task_spec_local.exists():
            raise FileNotFoundError(f"missing required artifact: {task_spec_local}")

        workspace_root = PrepareSandboxAgent._pick_workspace_root(runtime)
        workspace_repo_dir = f"{workspace_root}/repo"
        workspace_inputs_dir = f"{workspace_root}/inputs"
        workspace_outputs_dir = f"{workspace_root}/outputs"
        workspace_bin_dir = f"{workspace_root}/bin"

        ctx["workspace_root"] = workspace_root
        ctx["workspace_repo_dir"] = workspace_repo_dir
        ctx["workspace_inputs_dir"] = workspace_inputs_dir
        ctx["workspace_outputs_dir"] = workspace_outputs_dir
        ctx["workspace_bin_dir"] = workspace_bin_dir
        ctx["runtime_repo_dir"] = workspace_repo_dir

        mk = runtime.run_command(
            f"mkdir -p {shlex.quote(workspace_repo_dir)} {shlex.quote(workspace_inputs_dir)} "
            f"{shlex.quote(workspace_outputs_dir)} {shlex.quote(workspace_bin_dir)}",
            cwd=workspace_root,
            timeout_sec=30,
        )
        if mk.rc != 0:
            raise RuntimeError(f"failed to prepare workspace dirs: {mk.stderr[:300]}")

        tools_local_dir = "execution/tools_newstyle"
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

        upload_repo_dir = repo_dir / "code" if (repo_dir / "code").is_dir() else repo_dir
        uploaded_code_subdir = upload_repo_dir != repo_dir
        sandbox_task_spec_payload = PrepareSandboxAgent._rewrite_code_scoped_task_spec(
            self.artifacts.read_json("task/task_spec.json"),
            uploaded_code_subdir=uploaded_code_subdir,
        )
        sandbox_task_spec_local = self.artifacts.write_json(
            "execution/task_spec.newstyle.sandbox.json",
            sandbox_task_spec_payload,
        )
        codex_skill_local = self.artifacts.write_text(
            "execution/codex_execution_skill.newstyle.md",
            CODEX_EXECUTION_SKILL,
        )
        ctx["workspace_task_spec_remote"] = f"{workspace_inputs_dir}/task_spec.json"
        ctx["workspace_task_spec_local_artifact"] = "execution/task_spec.newstyle.sandbox.json"
        ctx["workspace_codex_skill_remote"] = f"{workspace_inputs_dir}/codex_execution_skill.md"
        ctx["workspace_codex_skill_local_artifact"] = "execution/codex_execution_skill.newstyle.md"

        include_git = (ctx.get("phase2_include_git") or "").strip() == "1"
        repo_excludes = None if include_git else [".git", ".git/**"]
        runtime.upload_dir(local_dir=upload_repo_dir, remote_dir=workspace_repo_dir, exclude_globs=repo_excludes)
        runtime.upload_file(local_file=sandbox_task_spec_local, remote_path=f"{workspace_inputs_dir}/task_spec.json")
        runtime.upload_file(local_file=codex_skill_local, remote_path=f"{workspace_inputs_dir}/codex_execution_skill.md")

        try:
            probe = runtime.run_command(
                "python3 - <<'PY'\nimport os,platform\nprint(platform.system())\nprint(platform.release())\nprint(platform.python_version())\nPY",
                cwd=workspace_repo_dir,
                timeout_sec=30,
            )
            lines = [x.strip() for x in (probe.stdout or "").splitlines() if x.strip()]
            platform_name = lines[0] if len(lines) > 0 else platform.system()
            release = lines[1] if len(lines) > 1 else platform.release()
            pyver = lines[2] if len(lines) > 2 else platform.python_version()
        except Exception:  # noqa: BLE001
            platform_name = platform.system()
            release = platform.release()
            pyver = platform.python_version()

        info = SystemInfo(
            platform=platform_name,
            platform_release=release,
            python_version=pyver,
            cpu_count=None,
            memory_gb=None,
            reason_codes=(llm_data or {}).get("reason_codes", []),
        )
        self.artifacts.write_json("execution/system_info.json", info.model_dump())

        return {
            "system_info": info.model_dump(),
            "workspace": {
                "repo": workspace_repo_dir,
                "inputs": workspace_inputs_dir,
                "outputs": workspace_outputs_dir,
                "bin": workspace_bin_dir,
            },
        }
