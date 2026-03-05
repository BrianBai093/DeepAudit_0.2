from __future__ import annotations

import re
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import CommandRecord, MetricRecord, MetricsDoc, RepoState
from p2c.utils.console import utc_now_iso

SYSTEM_PROMPT = (
    "You are an execution orchestrator. Produce concise command strategy text only. "
    "Do not fabricate outcomes."
)

USER_PROMPT_TEMPLATE = (
    "Input: task/task_spec.json. Output: execution/commands.jsonl, execution/run.log, "
    "execution/patch.diff, execution/repo_state.json, results/metrics.json"
)

UNSAFE_PATTERNS = [
    re.compile(r"(^|\s)sudo(\s|$)"),
    re.compile(r"curl\b[^\n|]*\|\s*(bash|sh)\b"),
    re.compile(r"wget\b[^\n|]*\|\s*(bash|sh)\b"),
]


class ExecuteAndHealAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="execute_and_heal", *args, **kwargs)

    @staticmethod
    def _is_unsafe(command: str) -> bool:
        return any(p.search(command) for p in UNSAFE_PATTERNS)

    def _record_command(self, cwd: str, cmd: str, rc: int, stdout: str, stderr: str) -> None:
        rec = CommandRecord(
            ts=utc_now_iso(),
            cwd=cwd,
            cmd=cmd,
            rc=rc,
            stdout_summary=stdout[:300] if stdout else None,
            stderr_summary=stderr[:300] if stderr else None,
            resource_usage=None,
        )
        self.artifacts.append_jsonl("execution/commands.jsonl", rec.model_dump())

    @staticmethod
    def _default_runtime_root(runtime) -> str:
        backend = (getattr(runtime, "backend_name", "") or "").strip().lower()
        return "/home/user/p2c_sandbox" if backend == "e2b" else "/tmp/p2c_sandbox"

    def _capture_repo_state(self, runtime, repo_dir: str) -> None:
        head = None
        branch = None
        diff_summary = None
        submodules: list[str] = []
        reason_codes: list[str] = []

        head_p = runtime.run_command("git rev-parse HEAD", cwd=repo_dir, timeout_sec=10)
        branch_p = runtime.run_command("git rev-parse --abbrev-ref HEAD", cwd=repo_dir, timeout_sec=10)
        diff_p = runtime.run_command("git diff -- .", cwd=repo_dir, timeout_sec=20)
        subm_p = runtime.run_command("git submodule status", cwd=repo_dir, timeout_sec=20)

        if head_p.rc == 0:
            head = head_p.stdout.strip() or None
        else:
            reason_codes.append("GIT_HEAD_UNAVAILABLE")

        if branch_p.rc == 0:
            branch = branch_p.stdout.strip() or None
        else:
            reason_codes.append("GIT_BRANCH_UNAVAILABLE")

        self.artifacts.write_text("execution/patch.diff", diff_p.stdout if diff_p.rc == 0 else "")
        diff_summary = (diff_p.stdout or "")[:1200]

        if subm_p.rc == 0:
            submodules = [line for line in (subm_p.stdout or "").splitlines() if line.strip()]

        repo_state = RepoState(
            head=head,
            branch=branch,
            diff_summary=diff_summary,
            submodules=submodules,
            reason_codes=reason_codes,
        )
        self.artifacts.write_json("execution/repo_state.json", repo_state.model_dump())

    @staticmethod
    def _extract_metrics(output_text: str) -> list[MetricRecord]:
        records: list[MetricRecord] = []
        for m in re.finditer(r"accuracy[^0-9]*(\d+(?:\.\d+)?)\s*%", output_text, flags=re.I):
            val = float(m.group(1)) / 100.0
            records.append(MetricRecord(metric_name="accuracy", value=val, unit="ratio", source="run.log"))
        for m in re.finditer(r"accuracy[^0-9]*(0\.\d+|1\.0+)", output_text, flags=re.I):
            val = float(m.group(1))
            records.append(MetricRecord(metric_name="accuracy", value=val, unit="ratio", source="run.log"))

        if not records:
            records.append(
                MetricRecord(
                    metric_name="unknown",
                    value=None,
                    unit=None,
                    source="run.log",
                    parsed=False,
                    reason_codes=["UNPARSED_METRICS"],
                )
            )
        return records

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        runtime = ensure_runtime(ctx, self.artifacts)
        default_root = self._default_runtime_root(runtime)
        runtime_repo_dir = ctx.get("runtime_repo_dir", f"{default_root}/repo")
        repo_dir_local = Path(ctx.get("repo_dir", ""))
        if not ctx.get("runtime_repo_dir") and repo_dir_local.exists():
            runtime.upload_dir(local_dir=repo_dir_local, remote_dir=runtime_repo_dir)
            ctx["runtime_repo_dir"] = runtime_repo_dir

        task_spec = self.artifacts.read_json("task/task_spec.json")
        run_matrix = task_spec.get("run_matrix", [])
        timeout_sec = int((run_matrix[0] or {}).get("timeout_sec", 900)) if run_matrix else 900

        entrypoints = task_spec.get("entrypoints", [])
        reason_codes: list[str] = []

        output_blob = ""
        if entrypoints:
            command = entrypoints[0].get("command", "")
            if not command:
                reason_codes.append("EMPTY_ENTRYPOINT_COMMAND")
            elif self._is_unsafe(command):
                reason_codes.append("BLOCKED_UNSAFE_COMMAND")
                self.artifacts.append_text(
                    "execution/run.log",
                    f"\nBlocked unsafe command: {command}\n",
                )
            else:
                self.log("PROGRESS", f"executing command in sandbox: {command}")
                proc = runtime.run_command(command, cwd=runtime_repo_dir, timeout_sec=timeout_sec)
                self._record_command(runtime_repo_dir, command, proc.rc, proc.stdout, proc.stderr)
                output_blob = f"$ {command}\n{proc.stdout}\n{proc.stderr}\n"
                self.artifacts.append_text("execution/run.log", "\n" + output_blob)
                if proc.rc != 0:
                    reason_codes.append(f"ENTRYPOINT_RC_{proc.rc}")
        else:
            reason_codes.append("NO_ENTRYPOINT")

        self._capture_repo_state(runtime, runtime_repo_dir)

        metrics = MetricsDoc(records=self._extract_metrics(output_blob), reason_codes=reason_codes)
        self.artifacts.write_json("results/metrics.json", metrics.model_dump())

        return {"metrics": metrics.model_dump()}
