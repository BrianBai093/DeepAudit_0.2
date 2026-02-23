from __future__ import annotations

import os
import platform
import shlex
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import SystemInfo

SYSTEM_PROMPT = (
    "You summarize runtime system info. Return strict JSON and avoid fabrication."
)

USER_PROMPT_TEMPLATE = "Input: local runtime info. Output: execution/system_info.json"


class PrepareSandboxAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="prepare_sandbox", *args, **kwargs)

    def _pick_runtime_root(self, runtime, preferred_root: str) -> str:
        candidates = [preferred_root]
        for root in [
            "/home/user/p2c_sandbox",
            "/home/sandbox/p2c_sandbox",
            "/workspace/p2c_sandbox",
            "/tmp/p2c_sandbox",
        ]:
            if root not in candidates:
                candidates.append(root)

        for root in candidates:
            tmp_dir_q = shlex.quote(f"{root}/tmp")
            probe = runtime.run_command(f"mkdir -p {tmp_dir_q} && test -w {tmp_dir_q}", cwd="/", timeout_sec=20)
            if probe.rc == 0:
                return root
        return preferred_root

    def execute(self, ctx: dict) -> dict:
        runtime = ensure_runtime(ctx, self.artifacts)
        llm_schema = {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["notes", "reason_codes"],
        }
        llm_data, _ = self.safe_chat_json(llm_schema, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        # Ensure repo is uploaded to sandbox runtime and collect runtime system facts from sandbox.
        backend = (getattr(runtime, "backend_name", "") or "").strip().lower()
        default_root = "/home/user/p2c_sandbox" if backend == "e2b" else "/tmp/p2c_sandbox"
        runtime_root_dir = ctx.get("runtime_root_dir") or os.getenv("P2C_RUNTIME_ROOT_DIR") or default_root
        if backend == "e2b":
            runtime_root_dir = self._pick_runtime_root(runtime, runtime_root_dir)
        repo_dir = ctx.get("repo_dir")
        runtime_repo_dir = ctx.get("runtime_repo_dir") or f"{runtime_root_dir}/repo"
        runtime_mini_dir = ctx.get("runtime_mini_dir") or f"{runtime_root_dir}/mini-swe-agent"
        runtime_tmp_dir = ctx.get("runtime_tmp_dir") or f"{runtime_root_dir}/tmp"
        ctx["runtime_root_dir"] = runtime_root_dir
        ctx["runtime_repo_dir"] = runtime_repo_dir
        ctx["runtime_mini_dir"] = runtime_mini_dir
        ctx["runtime_tmp_dir"] = runtime_tmp_dir
        runtime_root_q = shlex.quote(runtime_root_dir)
        runtime_repo_q = shlex.quote(runtime_repo_dir)
        runtime_mini_q = shlex.quote(runtime_mini_dir)
        runtime_tmp_q = shlex.quote(runtime_tmp_dir)
        runtime.run_command(
            f"mkdir -p {runtime_root_q} {runtime_repo_q} {runtime_mini_q} {runtime_tmp_q}",
            cwd="/",
            timeout_sec=30,
        )
        if repo_dir:
            runtime.upload_dir(local_dir=Path(repo_dir), remote_dir=runtime_repo_dir)
        if (Path.cwd() / "mini-swe-agent").exists():
            runtime.upload_dir(local_dir=Path.cwd() / "mini-swe-agent", remote_dir=runtime_mini_dir)

        mem_gb = None
        try:
            probe = runtime.run_command(
                "python3 - <<'PY'\nimport os,platform\nprint(platform.system())\nprint(platform.release())\nprint(platform.python_version())\ntry:\n p=os.sysconf('SC_PAGE_SIZE')\n n=os.sysconf('SC_PHYS_PAGES')\n print(round((p*n)/(1024**3),2))\nexcept Exception:\n print('')\nPY",
                cwd=runtime_repo_dir,
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
            cpu_count=os.cpu_count(),
            memory_gb=mem_gb,
            reason_codes=(llm_data or {}).get("reason_codes", []),
        )
        self.artifacts.write_json("execution/system_info.json", info.model_dump())
        return {"system_info": info.model_dump()}
