from __future__ import annotations

import json
import os
import shlex
import time
from typing import Any

from p2c.agents.phase2.codex_exec_support import is_rate_limit_failure
from p2c.agents.phase2.codex_prompt_templates_newstyle import build_newstyle_execution_prompt
from p2c.agents.phase2.run_codex_exec import RunCodexExecAgent
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import ExecutionSummaryDoc

SYSTEM_PROMPT = "You orchestrate new-style autonomous Codex execution in E2B sandbox with a single exec session."
USER_PROMPT_TEMPLATE = "Input: task_spec. Output: execution_summary.json + logs under /workspace/outputs."


class RunCodexExecNewstyleAgent(RunCodexExecAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "run_codex_exec_newstyle"

    @staticmethod
    def _extract_last_json(text: str) -> dict[str, Any] | None:
        raw = str(text or "")
        for idx in range(len(raw) - 1, -1, -1):
            if raw[idx] != "{":
                continue
            try:
                obj = json.loads(raw[idx:])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _load_execution_summary(self, runtime, outputs_dir: str) -> dict[str, Any]:
        try:
            payload = json.loads(runtime.read_text(f"{outputs_dir}/execution_summary.json"))
            if isinstance(payload, dict):
                return payload
        except Exception:  # noqa: BLE001
            pass
        main_log = self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_main.log", n=120000)
        recovered = self._extract_last_json(main_log)
        return recovered if isinstance(recovered, dict) else {}

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        runtime = ensure_runtime(ctx, self.artifacts)
        if (getattr(runtime, "backend_name", "") or "").lower() != "e2b":
            raise RuntimeError("run_codex_exec_newstyle requires P2C_RUNTIME_BACKEND=e2b")

        required_ctx = ["workspace_root", "workspace_repo_dir", "workspace_outputs_dir", "workspace_inputs_dir"]
        missing = [k for k in required_ctx if not ctx.get(k)]
        if missing:
            raise RuntimeError(f"run_codex_exec_newstyle missing workspace context keys: {missing}")

        workspace_root = str(ctx["workspace_root"])
        repo_dir = str(ctx["workspace_repo_dir"])
        outputs_dir = str(ctx["workspace_outputs_dir"])
        inputs_dir = str(ctx["workspace_inputs_dir"])
        workspace_bin_dir = str(ctx.get("workspace_bin_dir") or f"{workspace_root}/bin")
        task_spec_local_artifact = str(ctx.get("workspace_task_spec_local_artifact") or "").strip()
        codex_skill_remote = str(ctx.get("workspace_codex_skill_remote") or "").strip()

        task_spec_payload = (
            self.artifacts.read_json(task_spec_local_artifact)
            if task_spec_local_artifact
            else self.artifacts.read_json("task/task_spec.json")
        )
        tasks = self._load_remote_json(runtime, f"{inputs_dir}/task_spec.json") or task_spec_payload
        require_rscript = self._task_spec_requires_r(tasks) or self._repo_requires_r(ctx.get("repo_dir", ""))

        runtime.run_command(
            f"mkdir -p {shlex.quote(outputs_dir)} {shlex.quote(inputs_dir)}",
            cwd=workspace_root,
            timeout_sec=30,
        )
        stream_local_path = "execution/codex_outputs/codex_exec.stream.log"
        self.artifacts.write_text(stream_local_path, "")

        bootstrap_result = self._bootstrap_toolchain(
            runtime,
            workspace_root=workspace_root,
            workspace_bin_dir=workspace_bin_dir,
            outputs_dir=outputs_dir,
            install_rscript=require_rscript,
        )
        key_probe = runtime.run_command("bash -lc 'test -n \"$OPENAI_API_KEY\"'", cwd=workspace_root, timeout_sec=20)
        if key_probe.rc != 0:
            raise RuntimeError("OPENAI_API_KEY is not available inside sandbox runtime environment")

        codex_probe = runtime.run_command(
            "bash -lc " + shlex.quote(f"PATH={workspace_bin_dir}:$PATH; command -v codex >/dev/null 2>&1"),
            cwd=workspace_root,
            timeout_sec=20,
        )
        if codex_probe.rc != 0:
            raise RuntimeError("codex CLI is not available inside sandbox (template mismatch or install issue)")

        toolchain_probe = self._probe_toolchain(
            runtime,
            workspace_root=workspace_root,
            workspace_bin_dir=workspace_bin_dir,
        )
        self._write_toolchain_artifacts(runtime, outputs_dir=outputs_dir, toolchain_probe=toolchain_probe)
        missing_required_tools = self._missing_required_tools(toolchain_probe, require_rscript=require_rscript)
        if missing_required_tools:
            raise RuntimeError(
                "sandbox toolchain bootstrap incomplete; missing required commands: "
                + ", ".join(missing_required_tools)
            )

        prompt = build_newstyle_execution_prompt(
            repo_dir=repo_dir,
            task_spec_path=f"{inputs_dir}/task_spec.json",
            summary_output_path=f"{outputs_dir}/execution_summary.json",
            patches_output_path=f"{outputs_dir}/patches.diff",
            skill_path=codex_skill_remote or None,
        )
        cmd = self._prepend_path(
            self._build_codex_cmd(
                prompt,
                extra_args=["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"],
            ),
            workspace_bin_dir=workspace_bin_dir,
        )

        budget_minutes = int(ctx.get("budget_minutes", 30))
        timeout_sec = min(45 * 60, max(900, budget_minutes * 60 + 300))
        max_retries = int(os.getenv("P2C_CODEX_RATE_LIMIT_RETRIES", "1")) + 1
        backoff_sec = int(os.getenv("P2C_CODEX_RATE_LIMIT_BACKOFF_SEC", "10"))
        result: dict[str, Any] | None = None

        for attempt in range(1, max_retries + 1):
            result = self._run_stage(
                runtime,
                label="main",
                cmd=cmd,
                repo_dir=repo_dir,
                outputs_dir=outputs_dir,
                workspace_root=workspace_root,
                timeout_sec=timeout_sec,
                reason_codes=["NEWSTYLE_SINGLE_EXEC", *list(bootstrap_result.get("reason_codes") or [])],
                local_stream_path=stream_local_path,
                stream_sync_every_sec=20,
            )
            if int(result.get("rc", 1)) == 0:
                break
            main_log = self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_main.log", n=5000)
            exec_log = self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log", n=5000)
            if attempt >= max_retries or not is_rate_limit_failure(f"{main_log}\n{exec_log}"):
                break
            time.sleep(max(0, backoff_sec))

        summary_payload = self._load_execution_summary(runtime, outputs_dir)
        if not summary_payload:
            main_log = self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_main.log", n=5000)
            exec_log = self.reporter.safe_remote_log_tail(runtime, f"{outputs_dir}/codex_exec.log", n=5000)
            self.reporter.write_failure_artifact(
                stage="main",
                last_command=cmd,
                exit_code=int((result or {}).get("rc", 1)),
                stdout_tail="",
                stderr_tail="execution_summary.json missing or invalid",
                codex_exec_log_tail=exec_log,
                pip_log_tail=self.artifacts.path("execution/codex_outputs/dependency_bootstrap.log").read_text(
                    encoding="utf-8", errors="ignore"
                )[-2000:],
                reason_codes=["NEWSTYLE_SUMMARY_MISSING"],
            )
            raise RuntimeError("new-style phase2 finished without a valid execution_summary.json")

        summary_doc = ExecutionSummaryDoc(**summary_payload)
        self.artifacts.write_json("execution/codex_outputs/execution_summary.json", summary_doc.model_dump())
        try:
            runtime.write_text(
                f"{outputs_dir}/execution_summary.json",
                self.artifacts.path("execution/codex_outputs/execution_summary.json").read_text(encoding="utf-8"),
            )
        except Exception:  # noqa: BLE001
            pass

        for name in ["codex_exec.log", "codex_main.log", "patches.diff", "codex_exec.stream.log"]:
            try:
                self.artifacts.write_text(f"execution/codex_outputs/{name}", runtime.read_text(f"{outputs_dir}/{name}"))
            except Exception:  # noqa: BLE001
                if name.endswith(".diff") or name.endswith(".log"):
                    self.artifacts.write_text(f"execution/codex_outputs/{name}", "")

        return {"execution_summary": summary_doc.model_dump()}
