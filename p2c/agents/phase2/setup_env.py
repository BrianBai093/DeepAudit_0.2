from __future__ import annotations

import shlex

from p2c.agents.base import BaseAgent
from p2c.runtime.factory import ensure_runtime

SYSTEM_PROMPT = "You provide concise environment capture guidance as JSON only."
USER_PROMPT_TEMPLATE = "Output target: execution/env_lock/pip_freeze.txt"


class SetupEnvAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="setup_env", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        runtime = ensure_runtime(ctx, self.artifacts)
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        backend = (getattr(runtime, "backend_name", "") or "").strip().lower()
        default_root = "/home/user/p2c_sandbox" if backend == "e2b" else "/tmp/p2c_sandbox"
        runtime_repo_dir = ctx.get("runtime_repo_dir", f"{default_root}/repo")
        runtime_tmp_dir = ctx.get("runtime_tmp_dir", f"{default_root}/tmp")

        runtime_repo_dir_q = shlex.quote(runtime_repo_dir)
        runtime_tmp_dir_q = shlex.quote(runtime_tmp_dir)
        pip_upgrade_log_q = shlex.quote(f"{runtime_tmp_dir}/p2c_pip_upgrade.log")
        repo_install_log_q = shlex.quote(f"{runtime_tmp_dir}/p2c_repo_install.log")

        reason_codes: list[str] = []
        install_cmds = [
            f"mkdir -p {runtime_tmp_dir_q}",
            f"python3 -m pip install -U pip >{pip_upgrade_log_q} 2>&1 || true",
            f"if [ -f {runtime_repo_dir_q}/requirements.txt ]; then python3 -m pip install -r {runtime_repo_dir_q}/requirements.txt >{repo_install_log_q} 2>&1 || true; fi",
        ]
        for cmd in install_cmds:
            proc = runtime.run_command(cmd, cwd=runtime_repo_dir, timeout_sec=600)
            if proc.rc != 0:
                reason_codes.append("SANDBOX_DEP_INSTALL_FAILED")
                self.log("PROGRESS", f"sandbox install command failed rc={proc.rc}")

        freeze = runtime.run_command("python3 -m pip freeze", cwd=runtime_repo_dir, timeout_sec=120)
        if freeze.rc == 0:
            content = freeze.stdout
        else:
            content = ""
            reason_codes.append("PIP_FREEZE_FAILED")
            self.log("PROGRESS", f"pip freeze failed rc={freeze.rc}: {freeze.stderr[:200]}")

        self.artifacts.write_text("execution/env_lock/pip_freeze.txt", content)
        return {"env_lock": {"pip_freeze": "execution/env_lock/pip_freeze.txt", "reason_codes": reason_codes}}
