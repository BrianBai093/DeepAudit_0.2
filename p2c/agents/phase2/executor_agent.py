"""ExecutorAgent — single-session Claude Code execution for paper experiments."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from claude_agent_sdk import (  # type: ignore[import-untyped]
        AssistantMessage,
        ClaudeAgentOptions,
        PermissionResultAllow,
        PermissionResultDeny,
        ResultMessage,
        ToolPermissionContext,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    AssistantMessage = type("AssistantMessage", (), {})  # type: ignore[misc,assignment]
    ClaudeAgentOptions = type("ClaudeAgentOptions", (), {})  # type: ignore[misc,assignment]
    PermissionResultAllow = type("PermissionResultAllow", (), {})  # type: ignore[misc,assignment]
    PermissionResultDeny = type("PermissionResultDeny", (), {})  # type: ignore[misc,assignment]
    ResultMessage = type("ResultMessage", (), {})  # type: ignore[misc,assignment]
    ToolPermissionContext = type("ToolPermissionContext", (), {})  # type: ignore[misc,assignment]
    ToolResultBlock = type("ToolResultBlock", (), {})  # type: ignore[misc,assignment]
    ToolUseBlock = type("ToolUseBlock", (), {})  # type: ignore[misc,assignment]
    UserMessage = type("UserMessage", (), {})  # type: ignore[misc,assignment]

    async def query(**kwargs):  # type: ignore[misc]
        raise RuntimeError("claude-agent-sdk is not installed")
        yield

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.result_extraction import (
    build_run_manifest,
    classify_error_v2,
    extract_metrics_from_stdout,
)
from p2c.runtime.conda_env import CondaEnvManager
from p2c.schemas import (
    ClaimsIR,
    ExecutionFailure,
    ExecutorResultsDoc,
    MetricContract,
    StepFailure,
)

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_FORWARD_ENV_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "HOME", "USER", "PATH", "LANG", "SHELL",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
    "CONDA_EXE", "CONDA_PREFIX",
)
_ALLOWED_OVERRIDE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^--.*epoch.*$",
        r"^--.*step.*$",
        r"^--iters?$",
        r"^--n_batches?$",
        r"^--.*subset.*$",
        r"^--.*debug.*$",
        r"^--.*dev.*$",
        r"^--.*samples?.*$",
        r"^--fast_dev_run$",
    )
]
_SUSPICIOUS_OVERRIDE_KEYWORDS = ("epoch", "step", "iter", "subset", "debug", "dev", "sample", "fast_dev_run", "n_batch")
_ARTIFACT_EVIDENCE_SOURCES = {"checkpoint_eval", "existing_logs", "existing_results", "mixed"}
SMOKE_MIN_EPOCHS = 3
_BOUNDED_PACKAGE_METRIC_TOKENS = {
    "accuracy",
    "acc",
    "auc",
    "bleu",
    "f1",
    "precision",
    "pr_auc",
    "recall",
    "roc_auc",
    "rouge",
}
_PACKAGE_FIDELITY_RANK = {None: 0, "smoke": 1, "trend": 2, "artifact": 2, "full": 3}
_RUNTIME_ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
}
_RUNTIME_ARTIFACT_NAME_TOKENS = (
    "checkpoint",
    "checkpoints",
    "model",
    "models",
    "output",
    "outputs",
    "result",
    "results",
    "stat",
    "stats",
)
_REPO_MUTATION_GUARD_REL_PATH = "execution/repo_mutation_guard.json"


@dataclass
class ExecutorSessionResult:
    stdout: str
    stderr: str
    returncode: int
    narrative: str = ""


@dataclass(frozen=True)
class _ExecutorRuntimeSpec:
    backend: str
    env_name: str
    env_path: str
    python_command: str
    pip_command: str


def _compact_log_text(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _normalize_shell_command(command: str) -> str:
    return " ".join(str(command or "").split())


def _path_is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


@dataclass
class _LiveSessionSink:
    artifacts: Any
    repo_dir: Path
    parent_pid: int
    model: str
    env_name: str
    timeout_sec: int
    started_at: float
    activity_callback: Callable[..., None]
    active_bash_calls: dict[str, tuple[str, float]] = field(default_factory=dict)
    observed_bash_commands: list[str] = field(default_factory=list)

    def reset_files(self) -> None:
        self.artifacts.write_text("execution/executor_outputs/executor_agent.log", "")
        self.artifacts.write_text("execution/executor_outputs/session_stdout.log", "")
        self.artifacts.write_text("execution/executor_outputs/session_stderr.log", "")
        self.write_runtime_snapshot(status="initialized", message="Executor live sink initialized.")

    def append_narrative(self, text: str) -> None:
        content = str(text or "")
        if content:
            self.artifacts.append_text("execution/executor_outputs/executor_agent.log", content.rstrip() + "\n")

    def append_stdout(self, text: str) -> None:
        content = str(text or "")
        if content:
            self.artifacts.append_text("execution/executor_outputs/session_stdout.log", content.rstrip() + "\n")

    def append_stderr(self, text: str) -> None:
        content = str(text or "")
        if content:
            self.artifacts.append_text("execution/executor_outputs/session_stderr.log", content.rstrip() + "\n")

    def log_activity(
        self,
        *,
        event: str,
        message: str,
        status: str = "running",
        exit_code: int | None = None,
        artifacts: list[str] | None = None,
    ) -> None:
        self.activity_callback(
            event=event,
            experiment_id=None,
            cwd=str(self.repo_dir),
            command="executor_session",
            status=status,
            exit_code=exit_code,
            duration_sec=max(0.0, time.time() - self.started_at),
            artifacts=artifacts or [],
            message=_compact_log_text(message),
        )

    def heartbeat(self) -> None:
        children = self.snapshot_children()
        child_summary = ", ".join(f"{row['pid']}:{row['cmd']}" for row in children[:4]) or "none"
        self.write_runtime_snapshot(status="running", message=f"Heartbeat with children: {child_summary}")
        self.log_activity(
            event="heartbeat",
            message=f"Claude session still running; child_processes={child_summary}",
            status="running",
        )

    def record_command_start(self, tool_use_id: str, command: str) -> None:
        normalized = _normalize_shell_command(command)
        self.active_bash_calls[tool_use_id] = (normalized, time.time())
        if normalized and normalized not in self.observed_bash_commands:
            self.observed_bash_commands.append(normalized)
        self.log_activity(
            event="command_start",
            message=f"Executing Bash command: {normalized}",
            status="started",
            artifacts=[],
        )

    def record_command_end(self, tool_use_id: str, content: str, *, is_error: bool = False) -> None:
        command, started_at = self.active_bash_calls.pop(tool_use_id, ("", self.started_at))
        self.activity_callback(
            event="command_end",
            experiment_id=None,
            cwd=str(self.repo_dir),
            command=command,
            status="failed" if is_error else "ok",
            exit_code=1 if is_error else 0,
            duration_sec=max(0.0, time.time() - started_at),
            artifacts=[],
            message=_compact_log_text(content or ("command failed" if is_error else "command completed")),
        )

    def snapshot_children(self) -> list[dict[str, str]]:
        proc = subprocess.run(
            ["ps", "-o", "pid=,ppid=,etime=,stat=,cmd=", "--ppid", str(self.parent_pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        rows: list[dict[str, str]] = []
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(None, 4)
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "pid": parts[0],
                    "ppid": parts[1],
                    "elapsed": parts[2],
                    "stat": parts[3],
                    "cmd": parts[4],
                }
            )
        return rows

    def write_runtime_snapshot(self, *, status: str, message: str) -> None:
        self.artifacts.write_json(
            "execution/executor_outputs/executor_runtime.json",
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": status,
                "message": message,
                "parent_pid": self.parent_pid,
                "model": self.model,
                "env_name": self.env_name,
                "timeout_sec": self.timeout_sec,
                "children": self.snapshot_children(),
            },
        )


class ExecutorAgent(BaseAgent):
    """Phase 2 autonomous execution agent."""

    _live_sink: _LiveSessionSink | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="executor_agent", *args, **kwargs)

    @staticmethod
    def _build_runtime_spec(env_mgr: CondaEnvManager) -> _ExecutorRuntimeSpec:
        env_name = env_mgr.env_name
        env_path_getter = getattr(env_mgr, "env_path_actual", None)
        env_path = env_path_getter() if callable(env_path_getter) else ""
        if getattr(env_mgr, "_use_venv_fallback", False):
            if not env_path:
                env_path = str(Path("/tmp") / f"p2c_venv_{env_name}")
            return _ExecutorRuntimeSpec(
                backend="venv",
                env_name=env_name,
                env_path=env_path,
                python_command=str(Path(env_path) / "bin" / "python"),
                pip_command=str(Path(env_path) / "bin" / "pip"),
            )

        raw_conda_bin = getattr(env_mgr, "_conda_bin", None) or "conda"
        conda_bin = CondaEnvManager._resolve_binary(str(raw_conda_bin)) or str(raw_conda_bin)
        run_prefix = [conda_bin, "run"]
        if Path(conda_bin).name != "mamba":
            run_prefix.append("--no-capture-output")
        run_prefix.extend(["-n", env_name])
        python_command = " ".join(shlex.quote(part) for part in [*run_prefix, "python"])
        pip_command = " ".join(shlex.quote(part) for part in [*run_prefix, "pip"])
        return _ExecutorRuntimeSpec(
            backend=Path(conda_bin).name or "conda",
            env_name=env_name,
            env_path=env_path,
            python_command=python_command,
            pip_command=pip_command,
        )

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        env_mgr: CondaEnvManager = ctx["_p2_env_mgr"]
        repo_dir = Path(str(ctx["repo_dir"])).resolve()
        remaining_sec = max(300, int(ctx.get("_p2_remaining_sec", 1800)))

        claims_ir_payload = self.artifacts.read_json("fingerprint/claims_ir.json")
        claims_ir = ClaimsIR(**claims_ir_payload) if claims_ir_payload else ClaimsIR()
        experiments = [exp.model_dump() if hasattr(exp, "model_dump") else exp for exp in claims_ir.experiments]
        if not experiments:
            failure = self._fail_fast(
                "No experiments available in fingerprint/claims_ir.json",
                failure_code="NO_EXPERIMENTS_AVAILABLE",
            )
            return {"success": False, "failure": failure}

        contract_payload = self.artifacts.read_json("task/metric_contract.json")
        contract = MetricContract(**contract_payload) if contract_payload else MetricContract()
        repo_analysis = self.artifacts.read_json("task/repo_analysis.json")
        readme_content = self._read_readme(repo_dir)
        dependency_files = self._dependency_file_contents(repo_dir, repo_analysis)
        canonical_outputs_dir = self.artifacts.path("execution/executor_outputs").resolve()
        outputs_dir = self._executor_visible_outputs_dir(repo_dir, canonical_outputs_dir)
        if outputs_dir != canonical_outputs_dir and outputs_dir.exists():
            shutil.rmtree(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        model = (os.getenv("P2C_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip()
        soft_budget_sec_per_experiment = max(300, remaining_sec // max(1, len(experiments)))
        runtime_spec = self._build_runtime_spec(env_mgr)

        live_sink = _LiveSessionSink(
            artifacts=self.artifacts,
            repo_dir=repo_dir,
            parent_pid=os.getpid(),
            model=model,
            env_name=env_mgr.env_name,
            timeout_sec=remaining_sec,
            started_at=time.time(),
            activity_callback=self._append_activity,
        )
        live_sink.reset_files()

        prompt = self._build_prompt(
            repo_dir=repo_dir,
            experiments=experiments,
            repo_analysis=repo_analysis,
            readme_content=readme_content,
            dependency_files=dependency_files,
            runtime_spec=runtime_spec,
            outputs_dir=outputs_dir,
            budget_sec=remaining_sec,
            soft_budget_sec_per_experiment=soft_budget_sec_per_experiment,
        )

        repo_guard_ignore_roots = self._repo_guard_ignore_roots(repo_dir)
        baseline = self._capture_repo_state(repo_dir, ignore_roots=repo_guard_ignore_roots)
        self._append_activity(
            event="session_start",
            experiment_id=None,
            cwd=str(repo_dir),
            command="executor_session",
            status="started",
            exit_code=None,
            duration_sec=0.0,
            artifacts=[],
            message=f"Starting autonomous execution for {len(experiments)} experiments with model={model}.",
        )
        live_sink.write_runtime_snapshot(status="starting", message="Launching Claude executor session.")

        started = time.time()
        type(self)._live_sink = live_sink
        try:
            session_result = self._run_executor_session(env_mgr, prompt, str(repo_dir), timeout_sec=remaining_sec)
        finally:
            type(self)._live_sink = None
        runtime_sec = time.time() - started
        self._terminate_descendants(live_sink.parent_pid)

        narrative_path = self.artifacts.path("execution/executor_outputs/executor_agent.log")
        stdout_path = self.artifacts.path("execution/executor_outputs/session_stdout.log")
        stderr_path = self.artifacts.path("execution/executor_outputs/session_stderr.log")
        if session_result.narrative and narrative_path.stat().st_size == 0:
            live_sink.append_narrative(session_result.narrative)
        if session_result.stdout and stdout_path.stat().st_size == 0:
            live_sink.append_stdout(session_result.stdout)
        if session_result.stderr and stderr_path.stat().st_size == 0:
            live_sink.append_stderr(session_result.stderr)
        live_sink.write_runtime_snapshot(status="completed", message="Claude executor session returned to host.")

        mutated_files = self._detect_repo_mutation(repo_dir, baseline, ignore_roots=repo_guard_ignore_roots)
        artifact_mutations, blocking_mutations = self._partition_repo_mutations(mutated_files)
        self._write_repo_mutation_guard(
            mutated_files=mutated_files,
            artifact_mutations=artifact_mutations,
            blocking_mutations=blocking_mutations,
        )
        self._append_activity(
            event="guard_check",
            experiment_id=None,
            cwd=str(repo_dir),
            command="repo_guard",
            status="failed" if blocking_mutations else ("warning" if artifact_mutations else "ok"),
            exit_code=1 if blocking_mutations else 0,
            duration_sec=0.0,
            artifacts=mutated_files[:20],
            message=(
                "Tracked source/config mutation detected."
                if blocking_mutations else
                "Tracked runtime artifact mutation recorded; continuing."
                if artifact_mutations else
                "Repository guard passed."
            ),
        )
        if blocking_mutations:
            failure = ExecutionFailure(
                attempt=int(ctx.get("_p2_attempt", 1)),
                stage="execution",
                step_failures=[
                    StepFailure(
                        step_id="repo_guard",
                        command="executor session",
                        exit_code=1,
                        error_type="runtime",
                        error_message="Tracked source files were modified during execution.",
                        stdout_tail="\n".join(blocking_mutations[:20]),
                        stderr_tail="",
                        failure_code="SOURCE_MUTATION_DETECTED",
                        failure_layer="source_guard",
                        repair_strategy="abort",
                        repair_action="Reject run and inspect executor activity log.",
                        auto_repair_confidence=0.0,
                    )
                ],
                overall_error="Tracked source mutation detected",
                is_dependency_issue=False,
                reason_codes=["SOURCE_MUTATION_DETECTED"],
            )
            self.artifacts.write_json("execution/execution_failures.json", [failure.model_dump()])
            return {"success": False, "failure": failure}

        if outputs_dir != canonical_outputs_dir:
            self._sync_executor_outputs(outputs_dir, canonical_outputs_dir)
        self._recover_misplaced_executor_outputs(repo_dir, canonical_outputs_dir)

        result_path = self._select_executor_results_path(canonical_outputs_dir)
        runs = self._load_executor_runs(
            result_path,
            contract,
            session_result.stdout,
            experiments,
            canonical_outputs_dir,
            repo_dir=repo_dir,
            observed_commands=live_sink.observed_bash_commands,
        )
        self._backfill_activity_from_runs(runs)
        manifest_reason_codes = ["EXECUTOR_AGENT_RUN"]
        if artifact_mutations:
            manifest_reason_codes.append("REPO_RUNTIME_ARTIFACT_MUTATION_RECORDED")
        manifest = build_run_manifest(runs, reason_codes=manifest_reason_codes)
        self.artifacts.write_json("execution/executor_outputs/run_manifest.json", manifest.model_dump())
        raw_runs = self._load_raw_executor_result_runs(result_path)
        claims = [
            claim.model_dump() if hasattr(claim, "model_dump") else claim
            for claim in claims_ir.claims
        ]
        phase2_package = self._build_phase2_execution_package(
            raw_runs=raw_runs,
            experiments=experiments,
            claims=claims,
            contract=contract,
            raw_manifest=manifest.model_dump(),
            repo_dir=repo_dir,
            observed_commands=live_sink.observed_bash_commands,
            executor_results_rel_path=f"execution/executor_outputs/{result_path.name}",
        )
        self.artifacts.write_json("execution/executor_outputs/phase2_execution_package.json", phase2_package)
        self.artifacts.write_text(
            "execution/executor_outputs/PHASE2_RESULTS.md",
            self._render_phase2_results_markdown(phase2_package),
        )

        all_metrics = {
            metric_name: value
            for run in runs
            for metric_name, value in (run.get("metrics") or {}).items()
            if not str(metric_name).endswith("_all")
        }

        self._append_activity(
            event="session_end",
            experiment_id=None,
            cwd=str(repo_dir),
            command="executor_session",
            status="ok" if runs else "failed",
            exit_code=session_result.returncode,
            duration_sec=runtime_sec,
            artifacts=[
                "execution/executor_outputs/run_manifest.json",
                "execution/executor_outputs/phase2_execution_package.json",
                "execution/executor_outputs/PHASE2_RESULTS.md",
            ],
            message=f"Recorded {len(runs)} experiment runs.",
        )

        if runs:
            return {
                "success": True,
                "run_manifest": manifest,
                "phase2_execution_package": phase2_package,
                "metrics": all_metrics,
            }

        failure = self._fail_fast(
            "Executor session completed without producing executor_results.json runs.",
            failure_code="RESULTS_NOT_WRITTEN",
            stdout_tail=session_result.stdout,
            stderr_tail=session_result.stderr,
        )
        return {"success": False, "failure": failure}

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    @staticmethod
    def _read_readme(repo_dir: Path) -> str:
        for name in ("README.md", "readme.md"):
            path = repo_dir / name
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="ignore")[:16000]
        return ""

    @staticmethod
    def _dependency_file_contents(repo_dir: Path, repo_analysis: dict[str, Any]) -> dict[str, str]:
        payload: dict[str, str] = {}
        seen: set[str] = set()
        for profile in repo_analysis.get("dependency_profiles", []):
            for manifest_path in profile.get("manifest_paths", []):
                rel = str(manifest_path or "").strip()
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                path = repo_dir / rel
                if path.is_file():
                    payload[rel] = path.read_text(encoding="utf-8", errors="ignore")[:8000]
        return payload

    def _executor_visible_outputs_dir(self, repo_dir: Path, canonical_outputs_dir: Path) -> Path:
        """Return the output directory shown to the executor session.

        Downstream P2C stages still consume canonical artifacts under ArtifactManager.
        This method only moves the executor-writable scratch directory out of the target
        repo when the canonical artifact directory would otherwise be inside it.
        """
        repo_dir = repo_dir.resolve()
        canonical_outputs_dir = canonical_outputs_dir.resolve()
        if not _path_is_relative_to(canonical_outputs_dir, repo_dir):
            return canonical_outputs_dir

        override = os.getenv("P2C_EXECUTOR_OUTPUTS_DIR")
        if override:
            base = Path(override).expanduser().resolve()
            if base.name == "executor_outputs":
                return base
            return base / self.artifacts.run_id / "execution" / "executor_outputs"
        return Path(tempfile.gettempdir()) / "p2c_executor_outputs" / self.artifacts.run_id / "execution" / "executor_outputs"

    def _sync_executor_outputs(self, source_dir: Path, canonical_outputs_dir: Path) -> None:
        """Mirror externally written executor outputs back into canonical artifacts."""
        if not source_dir.exists():
            return
        canonical_outputs_dir.mkdir(parents=True, exist_ok=True)
        preserve_existing = {
            "executor_agent.log",
            "executor_runtime.json",
            "session_stdout.log",
            "session_stderr.log",
        }
        copied: list[str] = []
        for child in source_dir.iterdir():
            if not child.is_file():
                continue
            if child.stat().st_size > 20 * 1024 * 1024:
                continue
            dest = canonical_outputs_dir / child.name
            if dest.exists() and child.name in preserve_existing:
                continue
            try:
                shutil.copy2(child, dest)
                copied.append(f"execution/executor_outputs/{child.name}")
            except OSError:
                continue
        if copied:
            self._append_activity(
                event="executor_outputs_synced",
                experiment_id=None,
                cwd=str(source_dir),
                command="sync_external_executor_outputs",
                status="ok",
                exit_code=0,
                duration_sec=0.0,
                artifacts=copied,
                message=f"Synced executor outputs from external directory: {source_dir}",
            )

    @staticmethod
    def _long_horizon_policy_text() -> str:
        return (
            "## Long-Horizon Training Policy\n"
            "- Treat runs with explicit 100+ epoch schedules, nominal epochs >= 50, or unknown per-epoch cost as long-horizon by default.\n"
            f"- If repo-supported budget flags exist, start with a smoke run using at least {SMOKE_MIN_EPOCHS} epochs when an epoch flag is available; otherwise use the smallest faithful budget >= 5% of the declared schedule when possible.\n"
            "- If smoke succeeds and logs or metrics move, run one trend pass using <= 10 epochs or <= 20% of the declared schedule, whichever is smaller.\n"
            "- Do not start a long full run until every experiment has at least one artifact, smoke, trend, or skipped record.\n"
            "- Estimate full runtime from observed wall-clock cost of smoke or trend runs when possible.\n"
            "- Only run full when artifact, smoke, and trend evidence are insufficient and the estimated full runtime is <= 80% of the remaining global budget.\n"
            "- If the estimate exceeds budget, stop at trend or skipped rather than launching an unbounded long run.\n"
            "- If the repo exposes no supported budget flag, do not invent one: choose artifact, full, or skipped.\n\n"
        )

    @staticmethod
    def _build_prompt(
        *,
        repo_dir: Path,
        experiments: list[dict[str, Any]],
        repo_analysis: dict[str, Any],
        readme_content: str,
        dependency_files: dict[str, str],
        runtime_spec: _ExecutorRuntimeSpec,
        outputs_dir: Path,
        budget_sec: int,
        soft_budget_sec_per_experiment: int,
    ) -> str:
        dep_sections = "\n".join(
            f"### {name}\n```\n{content}\n```"
            for name, content in dependency_files.items()
        ) or "(none)"
        return (
            "You are executing a research repository to reproduce paper experiments.\n"
            f"Repository root: {repo_dir}\n"
            f"Managed environment backend: {runtime_spec.backend}\n"
            f"Managed environment name: {runtime_spec.env_name}\n"
            f"Managed environment path: {runtime_spec.env_path}\n"
            f"Managed python command: {runtime_spec.python_command}\n"
            f"Managed pip command: {runtime_spec.pip_command}\n"
            f"Budget: {budget_sec} seconds\n"
            f"Soft budget per experiment: {soft_budget_sec_per_experiment} seconds\n"
            f"External output directory (outside repository): {outputs_dir}\n\n"
            "## Paper Experiments (authoritative)\n"
            f"```json\n{json.dumps(experiments, ensure_ascii=False, indent=2)}\n```\n\n"
            "## Repository Analysis\n"
            f"```json\n{json.dumps(repo_analysis, ensure_ascii=False, indent=2)[:12000]}\n```\n\n"
            "## README\n"
            f"```\n{readme_content[:12000]}\n```\n\n"
            "## Dependency Files\n"
            f"{dep_sections}\n\n"
            "## Execution Policy\n"
            "1. Treat the experiments JSON as the only execution objective source.\n"
            "2. Read the repo and README to decide how to run the code.\n"
            "3. Use this search order for every experiment: artifact -> smoke -> trend -> full.\n"
            "4. Prefer eval-only, test-only, inference-only, checkpoint-based, or existing artifact paths first.\n"
            "5. For reduced-fidelity runs, only use repo-supported CLI flags already exposed by the repo.\n"
            "6. Do not edit, patch, rewrite, or modify repository-tracked source/config/script/notebook files.\n"
            "7. Breadth-first budget policy: do not spend full-run budget on one experiment until every experiment has a non-failed artifact/smoke/trend attempt or an explicit skipped record.\n"
            "8. If you shorten epochs/steps/data, you MUST mark the run as reduced-fidelity and never present it as full reproduction.\n"
            "9. If no supported CLI budget flag exists, choose artifact, full, or skipped instead of editing code.\n"
            "10. Keep a clear audit trail.\n"
            f"11. For Python commands, always start with `{runtime_spec.python_command}`.\n"
            f"12. For pip commands, always start with `{runtime_spec.pip_command}`.\n"
            "13. Never guess a different environment name or reuse an old environment from another run.\n"
            "14. Write audit metadata, executor_results.json, activity logs, and per-experiment logs only to the external output directory above, never under the repository root.\n"
            "15. If a training command supports output/log/checkpoint directory flags, point them to the external output directory; otherwise do not create extra audit directories inside the repo.\n\n"
            f"{ExecutorAgent._long_horizon_policy_text()}"
            "## Required Files To Write\n"
            "Use the absolute external output directory below. Do not write these files to `./artifacts`, `./outputs`, or any path inside the repository root.\n"
            f"1. `{outputs_dir}/executor_activity.jsonl`\n"
            "   Each line is a JSON object with keys: "
            "`ts`, `event`, `experiment_id`, `cwd`, `command`, `status`, `exit_code`, "
            "`duration_sec`, `artifacts`, `message`.\n"
            f"2. `{outputs_dir}/executor_results.json`\n"
            "   Schema:\n"
            "   {\n"
            "     \"runs\": [\n"
            "       {\n"
            "         \"experiment_id\": \"...\",\n"
            "         \"experiment_name\": \"...\",\n"
            "         \"dataset\": \"... or null\",\n"
            "         \"command\": \"primary command\",\n"
            "         \"commands_attempted\": [\"all commands in order\"],\n"
            "         \"cwd\": \".\",\n"
            "         \"exit_code\": 0,\n"
            "         \"status\": \"ok|partial|failed|skipped\",\n"
            "         \"fidelity\": \"artifact|smoke|trend|full\",\n"
            "         \"evidence_source\": \"fresh_run|checkpoint_eval|existing_logs|existing_results|mixed|null\",\n"
            "         \"override_args\": [\"repo-supported override flags actually used\"],\n"
            "         \"observed_signals\": [\"loss_decreasing|val_metric_seen|artifact_written|log_format_confirmed|...\"],\n"
            "         \"stop_reason\": \"checkpoint_eval|existing_artifact|budget_bound|early_stop_evidence|full_run_complete|repo_missing_path|runtime_failure|guardrail_blocked|skipped_nonessential|null\",\n"
            "         \"runtime_sec\": 0.0,\n"
            "         \"artifacts\": [\"relative or absolute paths\"],\n"
            "         \"metrics\": {\"metric_name\": 0.0},\n"
            "         \"notes\": \"what happened\",\n"
            "         \"logs\": {\n"
            "           \"stdout\": \"path/to/experiment_<id>_stdout.log\",\n"
            "           \"stderr\": \"path/to/experiment_<id>_stderr.log\",\n"
            "           \"narrative\": \"path/to/experiment_<id>_narrative.log\",\n"
            "           \"activity\": \"path/to/executor_activity.jsonl\"\n"
            "         },\n"
            "         \"reason_codes\": [\"optional\"]\n"
            "       }\n"
            "     ]\n"
            "   }\n\n"
            "For each experiment, always emit exactly one result row even if the experiment is skipped or fails.\n"
            "For each experiment, record command_start, command_end, artifact_recorded, metric_observed, and failure events when applicable.\n"
            "For each experiment, write dedicated logs named "
            "`experiment_<experiment_id>_stdout.log`, `experiment_<experiment_id>_stderr.log`, and "
            "`experiment_<experiment_id>_narrative.log` under the external output directory. "
            "Do not point per-experiment logs to session_stdout.log or session_stderr.log; those files are global session logs only."
        )

    @staticmethod
    def _build_system_prompt(runtime_spec: _ExecutorRuntimeSpec) -> str:
        return (
            "You are the Phase 2 executor for research repositories.\n"
            "Authority split:\n"
            "- The experiments JSON is the ONLY authority for what experiments must be attempted.\n"
            "- The repository, README, and dependency files are the ONLY authority for how to run them.\n"
            "- Do NOT use paper prose or numeric targets to invent commands.\n\n"
            "Execution policy:\n"
            "- Prefer artifact/checkpoint/eval-only paths before training.\n"
            "- Then prefer short bounded smoke or trend runs.\n"
            "- Only attempt full training when repo evidence requires it and budget allows.\n"
            "- If you shorten epochs/steps/data, mark the run as reduced-fidelity.\n"
            "- Never claim reduced-fidelity or artifact evidence is full reproduction.\n\n"
            "Long-horizon training policy:\n"
            "- Treat runs with explicit 100+ epoch schedules, nominal epochs >= 50, or unknown per-epoch cost as long-horizon by default.\n"
            f"- If repo-supported budget flags exist, start with a smoke run using at least {SMOKE_MIN_EPOCHS} epochs when an epoch flag is available; otherwise use the smallest faithful budget >= 5% of the declared schedule when possible.\n"
            "- If smoke succeeds and logs or metrics move, run one trend pass using <= 10 epochs or <= 20% of the declared schedule, whichever is smaller.\n"
            "- Do not start a long full run until every experiment has at least one artifact, smoke, trend, or skipped record.\n"
            "- Estimate full runtime from observed wall-clock cost of smoke or trend runs when possible.\n"
            "- Only run full when artifact, smoke, and trend evidence are insufficient and the estimated full runtime is <= 80% of the remaining global budget.\n"
            "- If the estimate exceeds budget, stop at trend or skipped rather than launching an unbounded long run.\n"
            "- If the repo exposes no supported budget flag, do not invent one: choose artifact, full, or skipped.\n\n"
            "Runtime policy:\n"
            f"- Managed backend: {runtime_spec.backend}\n"
            f"- Managed env name: {runtime_spec.env_name}\n"
            f"- Managed env path: {runtime_spec.env_path}\n"
            f"- For python commands, always start with `{runtime_spec.python_command}`.\n"
            f"- For pip commands, always start with `{runtime_spec.pip_command}`.\n"
            "- Never guess a different environment name or reuse a leftover environment from another run.\n\n"
            "Mutation policy:\n"
            "- Do NOT modify repository-tracked source, config, script, or notebook files.\n"
            "- Write audit metadata and executor logs/results only to the provided external execution output directory, not under the repository root.\n"
            "- If repo commands support output/log/checkpoint directory flags, direct those runtime outputs to the external output directory.\n\n"
            "Shell discipline:\n"
            "- Run commands in the foreground only.\n"
            "- No background jobs, no nohup, no screen, no tmux.\n"
            "- Use the managed python/pip commands above rather than a guessed shell environment.\n"
        )

    @staticmethod
    def _extract_bash_command(tool_input: dict[str, Any]) -> str:
        for key in ("command", "cmd", "bash_command"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in tool_input.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _split_shell_segments(command: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"\s*(?:&&|\|\||;)\s*", str(command or "")) if segment.strip()]

    @staticmethod
    def _strip_leading_assignments(tokens: list[str]) -> list[str]:
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token == "env":
                idx += 1
                continue
            name, eq, value = token.partition("=")
            if eq and name.replace("_", "").isalnum() and value:
                idx += 1
                continue
            break
        return tokens[idx:]

    @classmethod
    def _segment_tokens(cls, segment: str) -> list[str]:
        try:
            tokens = cls._strip_leading_assignments(shlex.split(segment))
        except ValueError:
            tokens = cls._strip_leading_assignments(segment.split())
        return tokens

    @staticmethod
    def _coerce_runtime_spec(runtime: _ExecutorRuntimeSpec | str) -> _ExecutorRuntimeSpec:
        if isinstance(runtime, _ExecutorRuntimeSpec):
            return runtime
        env_name = str(runtime or "").strip()
        prefix = f"conda run --no-capture-output -n {env_name}"
        return _ExecutorRuntimeSpec(
            backend="conda",
            env_name=env_name,
            env_path="",
            python_command=f"{prefix} python",
            pip_command=f"{prefix} pip",
        )

    @classmethod
    def _classify_python_or_pip_command(cls, segment: str) -> tuple[list[str], str | None]:
        tokens = cls._segment_tokens(segment)
        if not tokens:
            return tokens, None

        head = Path(tokens[0]).name
        if head in {"python", "python3", "pip"}:
            return tokens, head

        if head in {"conda", "mamba"} and len(tokens) > 1 and tokens[1] == "run":
            idx = 2
            while idx < len(tokens):
                token = tokens[idx]
                if token == "--no-capture-output":
                    idx += 1
                    continue
                if token in {"-n", "--name", "-p", "--prefix"}:
                    idx += 2
                    continue
                if token.startswith("-"):
                    idx += 1
                    continue
                break
            if idx < len(tokens):
                nested = Path(tokens[idx]).name
                if nested in {"python", "python3", "pip"}:
                    return tokens, nested

        return tokens, None

    @classmethod
    def _has_required_runtime_prefix(cls, segment: str, runtime: _ExecutorRuntimeSpec | str) -> bool:
        runtime_spec = cls._coerce_runtime_spec(runtime)
        tokens, command_name = cls._classify_python_or_pip_command(segment)
        if not command_name:
            return True

        if runtime_spec.backend == "venv":
            env_bin = Path(runtime_spec.env_path) / "bin"
            if command_name in {"python", "python3"}:
                return tokens[0] in {str(env_bin / "python"), str(env_bin / "python3")}
            return tokens[0] == str(env_bin / "pip")

        required_command = runtime_spec.python_command if command_name in {"python", "python3"} else runtime_spec.pip_command
        required_tokens = cls._segment_tokens(required_command)
        if tokens[: len(required_tokens)] == required_tokens:
            return True
        if command_name == "python3" and required_tokens:
            alt_tokens = [*required_tokens[:-1], "python3"]
            return tokens[: len(alt_tokens)] == alt_tokens
        return False

    @staticmethod
    def _contains_background_execution(segment: str) -> bool:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        if any(token in {"nohup", "screen", "tmux"} for token in tokens):
            return True
        return any(token == "&" for token in tokens)

    @staticmethod
    def _contains_destructive_mutation(segment: str) -> bool:
        normalized = _normalize_shell_command(segment).lower()
        return any(
            pattern in normalized
            for pattern in ("git checkout", "git reset", "sed -i", "perl -pi")
        )

    @staticmethod
    def _option_matches_allowed_override(option: str) -> bool:
        lowered = option.strip().lower()
        return any(pattern.match(lowered) for pattern in _ALLOWED_OVERRIDE_PATTERNS)

    @classmethod
    def _contains_disallowed_override(cls, segment: str) -> bool:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        for token in tokens:
            if not token.startswith("--"):
                continue
            option = token.split("=", 1)[0]
            lowered = option.lower()
            if any(keyword in lowered for keyword in _SUSPICIOUS_OVERRIDE_KEYWORDS) and not cls._option_matches_allowed_override(option):
                return True
        return False

    @classmethod
    def _extract_override_args_from_commands(cls, commands: list[str]) -> list[str]:
        override_args: list[str] = []
        for command in commands:
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = command.split()
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                option = token.split("=", 1)[0]
                if token.startswith("--") and cls._option_matches_allowed_override(option):
                    if "=" in token:
                        entry = token
                    elif idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                        entry = f"{token}={tokens[idx + 1]}"
                    else:
                        entry = token
                    if entry not in override_args:
                        override_args.append(entry)
                idx += 1
        return override_args

    @classmethod
    def _evaluate_bash_guardrail(cls, command: str, runtime: _ExecutorRuntimeSpec | str) -> tuple[bool, str | None, str]:
        runtime_spec = cls._coerce_runtime_spec(runtime)
        normalized = _normalize_shell_command(command)
        for segment in cls._split_shell_segments(normalized):
            if cls._contains_background_execution(segment):
                return False, "BACKGROUND_PROCESS_BLOCKED", "Background or detached execution is not allowed."
            if cls._contains_destructive_mutation(segment):
                return False, "DESTRUCTIVE_COMMAND_BLOCKED", "Repository-destructive mutation commands are blocked."
            if cls._contains_disallowed_override(segment):
                return False, "OVERRIDE_FLAG_NOT_ALLOWED", "Reduced-fidelity overrides outside the whitelist are blocked."
            _, command_name = cls._classify_python_or_pip_command(segment)
            if command_name and not cls._has_required_runtime_prefix(segment, runtime_spec):
                code = "MANAGED_RUNTIME_REQUIRED" if runtime_spec.backend == "venv" else "CONDA_PREFIX_REQUIRED"
                return False, code, "Python and pip commands must use the managed runtime command."
        return True, None, "allowed"

    # ------------------------------------------------------------------
    # Session execution
    # ------------------------------------------------------------------

    @staticmethod
    def _stream_prompt(prompt: str) -> AsyncIterator[dict[str, Any]]:
        async def _prompt_messages() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "user",
                "session_id": "",
                "message": {
                    "role": "user",
                    "content": prompt,
                },
                "parent_tool_use_id": None,
            }

        return _prompt_messages()

    @staticmethod
    def _run_executor_session(
        env_mgr: CondaEnvManager,
        prompt: str,
        cwd: str,
        timeout_sec: int = 1800,
    ) -> ExecutorSessionResult:
        model = (os.getenv("P2C_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip()
        max_turns = max(20, min(80, timeout_sec // 20))
        runtime_spec = ExecutorAgent._build_runtime_spec(env_mgr)
        use_tool_guardrails = os.getenv("P2C_USE_CLAUDE_TOOL_GUARDRAILS", "0") == "1"
        sink = ExecutorAgent._live_sink
        system_prompt = ExecutorAgent._build_system_prompt(runtime_spec)
        child_env = {key: value for key, value in os.environ.items() if key in _FORWARD_ENV_KEYS and value}
        if sink is not None:
            sink.log_activity(
                event="session_progress",
                message=f"Launching Claude SDK session model={model} max_turns={max_turns} timeout_sec={timeout_sec}.",
                status="starting",
            )
            sink.write_runtime_snapshot(status="launching", message="Claude SDK query() starting.")

        async def _can_use_tool(tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext):
            if tool_name != "Bash":
                return PermissionResultAllow()
            command = ExecutorAgent._extract_bash_command(tool_input)
            allowed, reason_code, reason = ExecutorAgent._evaluate_bash_guardrail(command, runtime_spec)
            if allowed:
                return PermissionResultAllow()
            if sink is not None:
                sink.append_stderr(f"[guardrail_blocked] {command}\n{reason}")
                sink.log_activity(
                    event="failure",
                    message=f"{reason_code}: {reason}",
                    status="failed",
                    exit_code=1,
                    artifacts=[],
                )
            return PermissionResultDeny(message=f"{reason_code}: {reason}", interrupt=False)

        async def _execute() -> ExecutorSessionResult:
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            narrative_parts: list[str] = []
            last_exit_code = 0
            heartbeat_task: asyncio.Task[None] | None = None

            async def _heartbeat() -> None:
                while True:
                    await asyncio.sleep(60)
                    if sink is not None:
                        sink.heartbeat()

            prompt_messages = ExecutorAgent._stream_prompt(prompt)
            if sink is not None:
                heartbeat_task = asyncio.create_task(_heartbeat())

            try:
                options_kwargs: dict[str, Any] = {
                    "cwd": cwd,
                    "allowed_tools": ["Read", "Glob", "Grep"] if use_tool_guardrails else ["Bash", "Read", "Glob", "Grep"],
                    "permission_mode": "default" if use_tool_guardrails else "bypassPermissions",
                    "max_turns": max_turns,
                    "model": model,
                    "system_prompt": system_prompt,
                    "env": child_env,
                }
                if use_tool_guardrails:
                    options_kwargs["can_use_tool"] = _can_use_tool
                async for msg in query(
                    prompt=prompt_messages,
                    options=ClaudeAgentOptions(**options_kwargs),
                ):
                    if isinstance(msg, AssistantMessage):
                        for block in getattr(msg, "content", []):
                            if isinstance(block, ToolUseBlock) and getattr(block, "name", "") == "Bash":
                                command = ExecutorAgent._extract_bash_command(getattr(block, "input", {}) or {})
                                if sink is not None and command:
                                    sink.record_command_start(getattr(block, "id", ""), command)
                                continue
                            text = getattr(block, "text", None)
                            if isinstance(text, str) and text.strip():
                                narrative_parts.append(text)
                                if sink is not None:
                                    sink.append_narrative(f"[assistant]\n{text}")
                                    sink.log_activity(
                                        event="session_progress",
                                        message=f"Assistant: {_compact_log_text(text)}",
                                    )
                    elif isinstance(msg, UserMessage):
                        for block in getattr(msg, "content", []):
                            if isinstance(block, ToolResultBlock):
                                tool_content = getattr(block, "content", None)
                                if isinstance(tool_content, list):
                                    text = "\n".join(
                                        str(item.get("text") or "")
                                        for item in tool_content
                                        if isinstance(item, dict) and item.get("type") == "text"
                                    )
                                else:
                                    text = str(tool_content or "")
                                if sink is not None:
                                    sink.record_command_end(
                                        getattr(block, "tool_use_id", ""),
                                        text,
                                        is_error=bool(getattr(block, "is_error", False)),
                                    )
                                if text:
                                    stdout_parts.append(text)
                                    if sink is not None:
                                        sink.append_stdout(text)
                                        sink.append_narrative(f"[tool]\n{text}")
                                if getattr(block, "is_error", False):
                                    stderr_parts.append(text)
                                    if sink is not None:
                                        sink.append_stderr(text)
                                        sink.append_narrative(f"[tool_error]\n{text}")
                                    last_exit_code = 1
                                continue
                            fragments: list[str] = []
                            content = getattr(block, "content", None)
                            if isinstance(content, str):
                                fragments.append(content)
                            elif isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        fragments.append(str(item.get("text") or ""))
                            text = "\n".join(fragment for fragment in fragments if fragment)
                            if text:
                                stdout_parts.append(text)
                                if sink is not None:
                                    sink.append_stdout(text)
                                    sink.append_narrative(f"[tool]\n{text}")
                                    sink.log_activity(
                                        event="session_progress",
                                        message=f"Tool output: {_compact_log_text(text)}",
                                    )
                            if getattr(block, "is_error", False):
                                stderr_text = text or str(content)
                                stderr_parts.append(stderr_text)
                                if sink is not None:
                                    sink.append_stderr(stderr_text)
                                    sink.append_narrative(f"[tool_error]\n{stderr_text}")
                                    sink.log_activity(
                                        event="failure",
                                        message=f"Tool error: {_compact_log_text(stderr_text)}",
                                        status="failed",
                                        exit_code=1,
                                    )
                                last_exit_code = 1
                    elif isinstance(msg, ResultMessage):
                        if getattr(msg, "result", None):
                            narrative_parts.append(msg.result)
                            if sink is not None:
                                sink.append_narrative(f"[result]\n{msg.result}")
                        subtype = getattr(msg, "subtype", "")
                        if sink is not None:
                            sink.log_activity(
                                event="session_progress",
                                message=f"Claude result subtype={subtype or 'success'}.",
                                status="running" if not subtype or subtype == "success" else "failed",
                                exit_code=0 if not subtype or subtype == "success" else 1,
                            )
                        if subtype and subtype != "success":
                            last_exit_code = 1
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

            return ExecutorSessionResult(
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
                returncode=last_exit_code,
                narrative="\n".join(narrative_parts),
            )

        try:
            return asyncio.run(asyncio.wait_for(_execute(), timeout=float(timeout_sec)))
        except asyncio.TimeoutError:
            ExecutorAgent._terminate_descendants(sink.parent_pid if sink is not None else os.getpid())
            if sink is not None:
                sink.append_stderr(f"Executor session timed out after {timeout_sec}s")
                sink.log_activity(
                    event="failure",
                    message=f"Executor session timed out after {timeout_sec}s.",
                    status="failed",
                    exit_code=1,
                )
                sink.write_runtime_snapshot(status="timeout", message="Claude executor session timed out.")
            return ExecutorSessionResult("", f"Executor session timed out after {timeout_sec}s", 1)
        except Exception as exc:
            logger.exception("Executor session error")
            ExecutorAgent._terminate_descendants(sink.parent_pid if sink is not None else os.getpid())
            if sink is not None:
                sink.append_stderr(f"Executor session error: {exc}")
                sink.log_activity(
                    event="failure",
                    message=f"Executor session error: {exc}",
                    status="failed",
                    exit_code=1,
                )
                sink.write_runtime_snapshot(status="error", message=f"Claude executor session error: {exc}")
            return ExecutorSessionResult("", f"Executor session error: {exc}", 1)

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------

    def _load_executor_runs(
        self,
        result_path: Path,
        contract: MetricContract,
        session_stdout: str,
        experiments: list[dict[str, Any]],
        outputs_dir: Path,
        *,
        repo_dir: Path | None = None,
        observed_commands: list[str],
    ) -> list[dict[str, Any]]:
        payload = {}
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                payload = {}

        raw_runs: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            try:
                parsed = ExecutorResultsDoc(**payload)
                raw_runs = [row.model_dump() for row in parsed.runs]
            except Exception:  # noqa: BLE001
                raw_runs = payload.get("runs", []) if isinstance(payload.get("runs"), list) else []

        runs_by_experiment: dict[str, dict[str, Any]] = {}
        for raw in raw_runs:
            if not isinstance(raw, dict):
                continue
            experiment_id = str(raw.get("experiment_id") or "").strip()
            if not experiment_id:
                continue
            runs_by_experiment[experiment_id] = dict(raw)

        normalized_runs: list[dict[str, Any]] = []
        observed_command_set = {_normalize_shell_command(command) for command in observed_commands if command}

        for experiment in experiments:
            experiment_id = str(experiment.get("experiment_id") or "").strip()
            if not experiment_id:
                continue
            raw_present = experiment_id in runs_by_experiment
            raw = runs_by_experiment.get(experiment_id, {})
            logs = raw.get("logs") if isinstance(raw.get("logs"), dict) else {}
            stdout_log = self._existing_executor_log_or_fallback(
                logs.get("stdout") or f"execution/executor_outputs/experiment_{experiment_id}_stdout.log",
                experiment_id,
                "stdout",
            )
            stderr_log = self._existing_executor_log_or_fallback(
                logs.get("stderr") or f"execution/executor_outputs/experiment_{experiment_id}_stderr.log",
                experiment_id,
                "stderr",
            )
            narrative_log = self._existing_executor_log_or_fallback(
                logs.get("narrative") or f"execution/executor_outputs/experiment_{experiment_id}_narrative.log",
                experiment_id,
                "narrative",
            )
            activity_log = self._existing_executor_log_or_fallback(
                logs.get("activity") or "execution/executor_outputs/executor_activity.jsonl",
                experiment_id,
                "activity",
            )

            if raw_present:
                stdout_log, stderr_log, narrative_log, synthetic_reason = self._ensure_per_run_logs(
                    experiment_id=experiment_id,
                    raw=raw,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    narrative_log=narrative_log,
                )
            else:
                synthetic_reason = None

            stdout_text = self._read_log(stdout_log)
            stderr_text = self._read_log(stderr_log)
            traceable_stdout_text = "" if stdout_log.endswith("/session_stdout.log") else stdout_text
            metrics_from_stdout = (
                extract_metrics_from_stdout(traceable_stdout_text, contract, command=raw.get("command"))
                if traceable_stdout_text else {}
            )
            raw_metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
            metrics = dict(metrics_from_stdout)
            for name, value in raw_metrics.items():
                metrics.setdefault(str(name), value)

            commands_attempted = [
                _normalize_shell_command(cmd)
                for cmd in raw.get("commands_attempted", [])
                if str(cmd).strip()
            ]
            primary_command = _normalize_shell_command(raw.get("command", ""))
            if primary_command and not commands_attempted:
                commands_attempted = [primary_command]
            if not primary_command and commands_attempted:
                primary_command = commands_attempted[0]

            override_args = self._extract_override_args_from_commands(commands_attempted)
            for entry in raw.get("override_args", []) if isinstance(raw.get("override_args"), list) else []:
                normalized_entry = str(entry).strip()
                if normalized_entry and normalized_entry not in override_args:
                    override_args.append(normalized_entry)

            artifacts = [str(path) for path in raw.get("artifacts", []) if str(path).strip()]
            existing_artifacts = [path for path in artifacts if self._artifact_exists(path, repo_dir=repo_dir)]
            evidence_source = self._normalize_evidence_source(raw.get("evidence_source"), traceable_stdout_text, existing_artifacts, commands_attempted)
            status = self._normalize_run_status(raw.get("status"), raw.get("exit_code", 1), raw_present)
            exit_code = int(raw.get("exit_code")) if raw.get("exit_code") is not None else (0 if status in {"ok", "partial", "skipped"} else 1)
            reason_codes = [str(code) for code in raw.get("reason_codes", []) if str(code).strip()]
            if synthetic_reason:
                reason_codes.append(synthetic_reason)
            observed_signals = [
                str(signal_name)
                for signal_name in raw.get("observed_signals", [])
                if str(signal_name).strip()
            ]
            observed_signals = self._merge_signal_hints(
                observed_signals,
                stdout_text=traceable_stdout_text,
                metrics=metrics,
                artifacts=existing_artifacts,
            )
            fidelity = self._normalize_fidelity(raw.get("fidelity"), evidence_source, override_args, metrics, observed_signals)
            stop_reason = self._normalize_stop_reason(raw.get("stop_reason"), status, fidelity, evidence_source, observed_signals)
            notes = raw.get("notes") if raw.get("notes") is None or isinstance(raw.get("notes"), str) else str(raw.get("notes"))

            if not raw_present:
                status = "failed"
                stop_reason = "runtime_failure"
                reason_codes.append("EXPERIMENT_RESULT_MISSING")

            missing_logs = [
                path for path in (stdout_log, stderr_log, narrative_log, activity_log)
                if path and not self._artifact_exists(path, repo_dir=repo_dir)
            ]
            if raw_present and missing_logs:
                reason_codes.append("DECLARED_LOG_MISSING")
                if status in {"ok", "partial"}:
                    status = "failed"
                    stop_reason = "runtime_failure"

            unobserved_commands = [
                command for command in commands_attempted
                if command and not self._command_was_observed(command, observed_command_set, observed_commands)
            ]
            if raw_present and observed_command_set and unobserved_commands:
                reason_codes.append("COMMAND_NOT_OBSERVED")
                traceable_success_evidence = bool(metrics or metrics_from_stdout or existing_artifacts or traceable_stdout_text)
                if status in {"ok", "partial"} and not traceable_success_evidence:
                    status = "failed" if evidence_source in {None, "fresh_run"} else "partial"
                    stop_reason = "guardrail_blocked"

            if metrics and not metrics_from_stdout and evidence_source not in _ARTIFACT_EVIDENCE_SOURCES and not existing_artifacts:
                reason_codes.append("UNTRACEABLE_METRICS")
                metrics = {}
                if status in {"ok", "partial"}:
                    status = "failed"
                    stop_reason = "runtime_failure"

            if fidelity == "full" and override_args:
                reason_codes.append("FULL_WITH_OVERRIDE_ARGS")
                fidelity = "trend" if metrics or observed_signals else "smoke"

            execution_outcome = self._compute_execution_outcome(
                fidelity=fidelity,
                status=status,
                evidence_source=evidence_source,
                override_args=override_args,
                metrics=metrics,
                observed_signals=observed_signals,
            )
            if execution_outcome == "FULLY_REPRODUCED" and evidence_source != "fresh_run":
                reason_codes.append("FULLY_REPRODUCED_REQUIRES_FRESH_RUN")
                execution_outcome = "TREND_SUPPORTED" if status in {"ok", "partial"} else None

            normalized_runs.append(
                {
                    "run_id": experiment_id,
                    "experiment_id": experiment_id,
                    "experiment_name": raw.get("experiment_name") or experiment.get("name"),
                    "dataset": raw.get("dataset") if raw.get("dataset") is not None else experiment.get("dataset"),
                    "command": primary_command,
                    "commands_attempted": commands_attempted,
                    "cwd": str(raw.get("cwd") or "."),
                    "exit_code": exit_code,
                    "status": status,
                    "fidelity": fidelity,
                    "execution_outcome": execution_outcome,
                    "evidence_source": evidence_source,
                    "override_args": override_args,
                    "observed_signals": observed_signals,
                    "stop_reason": stop_reason,
                    "notes": notes,
                    "runtime_sec": raw.get("runtime_sec"),
                    "stdout_tail": stdout_text[-2000:],
                    "stderr_tail": stderr_text[-2000:],
                    "artifacts": artifacts,
                    "metrics": metrics,
                    "logs": {
                        "stdout": stdout_log,
                        "stderr": stderr_log,
                        "narrative": narrative_log,
                        "activity": activity_log,
                    },
                    "params": {
                        "experiment_name": experiment.get("name"),
                        "dataset": experiment.get("dataset"),
                        "table_anchor": experiment.get("table_anchor"),
                    },
                    "reason_codes": list(dict.fromkeys(reason_codes)),
                }
            )
        return normalized_runs

    @staticmethod
    def _load_raw_executor_result_runs(result_path: Path) -> list[dict[str, Any]]:
        """Read executor_results.json without collapsing duplicate experiment ids."""
        if not result_path.is_file():
            return []
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        rows = payload.get("runs") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict)]

    @classmethod
    def _select_executor_results_path(cls, outputs_dir: Path) -> Path:
        """Choose the freshest executor results payload with runs.

        Some executor sessions produce an improved final payload such as
        ``executor_results_final.json`` after an initial diagnostic
        ``executor_results.json``. Phase 3 should consume the best completed
        payload, not a stale early diagnostic file.
        """
        candidates = [
            outputs_dir / "executor_results.json",
            outputs_dir / "executor_results_final.json",
            *sorted(outputs_dir.glob("executor_results_*.json")),
        ]
        valid: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if cls._load_raw_executor_result_runs(path):
                valid.append(path)
        if not valid:
            return outputs_dir / "executor_results.json"
        return max(
            valid,
            key=lambda path: (
                path.stat().st_mtime,
                1 if path.name != "executor_results.json" else 0,
            ),
        )

    def _build_phase2_execution_package(
        self,
        *,
        raw_runs: list[dict[str, Any]],
        experiments: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        contract: MetricContract,
        raw_manifest: dict[str, Any],
        repo_dir: Path,
        observed_commands: list[str],
        executor_results_rel_path: str = "execution/executor_outputs/executor_results.json",
    ) -> dict[str, Any]:
        """Build the single canonical Phase2 package consumed by Phase3."""
        experiment_by_id = {
            str(exp.get("experiment_id") or "").strip(): exp
            for exp in experiments
            if str(exp.get("experiment_id") or "").strip()
        }
        package_experiments: dict[str, dict[str, Any]] = {}
        for experiment_id, exp in experiment_by_id.items():
            package_experiments[experiment_id] = {
                "experiment_id": experiment_id,
                "name": exp.get("name"),
                "description": exp.get("description"),
                "dataset": exp.get("dataset"),
                "table_anchor": exp.get("table_anchor"),
                "primary_metrics": list(exp.get("primary_metrics", []) or []),
                "aliases": [],
                "paper_target_refs": self._paper_target_refs_for_experiment(experiment_id, claims),
                "attempts": [],
                "best_attempts_by_scope": {},
                "metrics": [],
                "failures": [],
                "logs": [],
                "summary_for_llm": "",
            }

        source_files = {
            "claims_ir": "fingerprint/claims_ir.json",
            "executor_results": executor_results_rel_path,
            "raw_run_manifest": "execution/executor_outputs/run_manifest.json",
            "phase2_results": "execution/executor_outputs/PHASE2_RESULTS.md",
            "raw_summaries": [
                rel for rel in (
                    "execution/executor_outputs/EXECUTION_COMPLETE.md",
                    "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md",
                    "execution/executor_outputs/EXECUTION_SUMMARY.md",
                    "execution/executor_outputs/PHASE2_EXECUTION_SUMMARY.md",
                )
                if self.artifacts.path(rel).exists() and self.artifacts.path(rel).stat().st_size > 0
            ],
        }
        if self.artifacts.path(_REPO_MUTATION_GUARD_REL_PATH).exists():
            source_files["repo_mutation_guard"] = _REPO_MUTATION_GUARD_REL_PATH

        observed_command_set = {_normalize_shell_command(command) for command in observed_commands if command}
        used_attempt_ids: set[str] = set()
        for index, raw in enumerate(raw_runs, start=1):
            raw_experiment_id = str(raw.get("experiment_id") or raw.get("run_id") or f"raw_{index}").strip()
            scope = self._infer_attempt_scope(raw)
            canonical_experiment_id = self._canonical_phase2_experiment_id(
                raw_experiment_id,
                raw,
                scope,
                experiment_by_id,
            )
            if canonical_experiment_id not in package_experiments:
                package_experiments[canonical_experiment_id] = {
                    "experiment_id": canonical_experiment_id,
                    "name": raw.get("experiment_name") or canonical_experiment_id,
                    "description": None,
                    "dataset": raw.get("dataset"),
                    "table_anchor": None,
                    "primary_metrics": [],
                    "aliases": [],
                    "paper_target_refs": self._paper_target_refs_for_experiment(canonical_experiment_id, claims),
                    "attempts": [],
                    "best_attempts_by_scope": {},
                    "metrics": [],
                    "failures": [],
                    "logs": [],
                    "summary_for_llm": "",
                }
            experiment_row = package_experiments[canonical_experiment_id]
            if raw_experiment_id and raw_experiment_id != canonical_experiment_id:
                aliases = experiment_row.setdefault("aliases", [])
                if raw_experiment_id not in aliases:
                    aliases.append(raw_experiment_id)

            commands_attempted = [
                _normalize_shell_command(cmd)
                for cmd in raw.get("commands_attempted", [])
                if str(cmd).strip()
            ]
            primary_command = _normalize_shell_command(raw.get("command", ""))
            if primary_command and not commands_attempted:
                commands_attempted = [primary_command]
            if not primary_command and commands_attempted:
                primary_command = commands_attempted[0]

            override_args = self._extract_override_args_from_commands(commands_attempted)
            explicit_override_args: list[str] = []
            for entry in raw.get("override_args", []) if isinstance(raw.get("override_args"), list) else []:
                normalized_entry = str(entry).strip()
                if normalized_entry:
                    explicit_override_args.append(normalized_entry)
                if normalized_entry and normalized_entry not in override_args:
                    override_args.append(normalized_entry)

            raw_logs = raw.get("logs") if isinstance(raw.get("logs"), dict) else {}
            stdout_log = self._existing_executor_log_or_fallback(
                raw_logs.get("stdout") or f"execution/executor_outputs/experiment_{raw_experiment_id}_stdout.log",
                raw_experiment_id,
                "stdout",
            )
            stderr_log = self._existing_executor_log_or_fallback(
                raw_logs.get("stderr") or f"execution/executor_outputs/experiment_{raw_experiment_id}_stderr.log",
                raw_experiment_id,
                "stderr",
            )
            narrative_log = self._existing_executor_log_or_fallback(
                raw_logs.get("narrative") or f"execution/executor_outputs/experiment_{raw_experiment_id}_narrative.log",
                raw_experiment_id,
                "narrative",
            )
            activity_log = self._existing_executor_log_or_fallback(
                raw_logs.get("activity") or "execution/executor_outputs/executor_activity.jsonl",
                raw_experiment_id,
                "activity",
            )

            raw_artifacts = [str(path) for path in raw.get("artifacts", []) if str(path).strip()]
            existing_artifacts = [path for path in raw_artifacts if self._artifact_exists(path, repo_dir=repo_dir)]
            status = self._normalize_run_status(raw.get("status"), raw.get("exit_code", 1), True)
            raw_metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
            observed_signals = [
                str(signal_name)
                for signal_name in raw.get("observed_signals", [])
                if str(signal_name).strip()
            ]
            fidelity = self._normalize_fidelity(
                raw.get("fidelity"),
                self._normalize_evidence_source(raw.get("evidence_source"), "", existing_artifacts, commands_attempted),
                override_args,
                raw_metrics,
                observed_signals,
            )
            attempt_id = self._unique_attempt_id(
                canonical_experiment_id=canonical_experiment_id,
                scope=scope,
                fidelity=fidelity,
                used=used_attempt_ids,
                fallback_index=index,
            )
            log_stem = self._safe_identifier(attempt_id)
            stdout_log, stderr_log, narrative_log, synthetic_reason = self._ensure_per_run_logs(
                experiment_id=raw_experiment_id,
                raw=raw,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                narrative_log=narrative_log,
                log_stem=log_stem,
            )
            stdout_text = self._read_log(stdout_log)
            stderr_text = self._read_log(stderr_log)
            traceable_stdout_text = "" if stdout_log.endswith("/session_stdout.log") else stdout_text
            metrics_from_stdout = (
                extract_metrics_from_stdout(traceable_stdout_text, contract, command=primary_command)
                if traceable_stdout_text else {}
            )
            merged_metrics = dict(metrics_from_stdout)
            for name, value in raw_metrics.items():
                merged_metrics.setdefault(str(name), value)

            evidence_source = self._normalize_evidence_source(
                raw.get("evidence_source"),
                traceable_stdout_text,
                existing_artifacts,
                commands_attempted,
            )
            observed_signals = self._merge_signal_hints(
                observed_signals,
                stdout_text=traceable_stdout_text,
                metrics=merged_metrics,
                artifacts=existing_artifacts,
            )
            fidelity = self._normalize_fidelity(raw.get("fidelity"), evidence_source, override_args, merged_metrics, observed_signals)
            stop_reason = self._normalize_stop_reason(raw.get("stop_reason"), status, fidelity, evidence_source, observed_signals)
            reason_codes = [str(code) for code in raw.get("reason_codes", []) if str(code).strip()]
            if synthetic_reason and synthetic_reason not in reason_codes:
                reason_codes.append(synthetic_reason)

            unobserved_commands = [
                command for command in commands_attempted
                if command and not self._command_was_observed(command, observed_command_set, observed_commands)
            ]
            if observed_command_set and unobserved_commands and "COMMAND_NOT_OBSERVED" not in reason_codes:
                reason_codes.append("COMMAND_NOT_OBSERVED")

            if fidelity == "full" and explicit_override_args:
                reason_codes.append("FULL_WITH_OVERRIDE_ARGS")
                fidelity = "trend" if merged_metrics or observed_signals else "smoke"

            outcome_override_args = explicit_override_args if fidelity == "full" else override_args
            execution_outcome = self._compute_execution_outcome(
                fidelity=fidelity,
                status=status,
                evidence_source=evidence_source,
                override_args=outcome_override_args,
                metrics=merged_metrics,
                observed_signals=observed_signals,
            )
            if execution_outcome == "FULLY_REPRODUCED" and evidence_source != "fresh_run":
                reason_codes.append("FULLY_REPRODUCED_REQUIRES_FRESH_RUN")
                execution_outcome = "TREND_SUPPORTED" if status in {"ok", "partial"} else None

            metric_entries = self._package_metric_entries(
                metrics=merged_metrics,
                scope=scope,
                fidelity=fidelity,
                attempt_id=attempt_id,
            )
            logs = {
                "stdout": stdout_log,
                "stderr": stderr_log,
                "narrative": narrative_log,
                "activity": activity_log,
            }
            attempt = {
                "attempt_id": attempt_id,
                "experiment_id": canonical_experiment_id,
                "source_experiment_id": raw_experiment_id,
                "experiment_name": raw.get("experiment_name") or experiment_row.get("name"),
                "config_name": raw.get("config_name"),
                "scope": scope,
                "command": primary_command,
                "commands_attempted": commands_attempted,
                "cwd": str(raw.get("cwd") or "."),
                "exit_code": self._safe_int(raw.get("exit_code"), 0 if status in {"ok", "partial", "skipped"} else 1),
                "status": status,
                "fidelity": fidelity,
                "execution_outcome": execution_outcome,
                "evidence_source": evidence_source,
                "override_args": override_args,
                "observed_signals": observed_signals,
                "stop_reason": stop_reason,
                "runtime_sec": raw.get("runtime_sec"),
                "artifacts": raw_artifacts,
                "metrics": metric_entries,
                "logs": logs,
                "stdout_tail": stdout_text[-2000:],
                "stderr_tail": stderr_text[-2000:],
                "notes": raw.get("notes"),
                "reason_codes": list(dict.fromkeys(reason_codes)),
            }
            experiment_row["attempts"].append(attempt)
            for log_ref in logs.values():
                if log_ref and log_ref not in experiment_row["logs"]:
                    experiment_row["logs"].append(log_ref)
            for metric in metric_entries:
                experiment_row["metrics"].append(metric)
                scope_key = self._metric_scope_key(metric)
                current = experiment_row["best_attempts_by_scope"].get(scope_key)
                if not current or self._attempt_preferred(attempt, current, experiment_row["attempts"]):
                    experiment_row["best_attempts_by_scope"][scope_key] = attempt_id
            if status in {"failed", "skipped"}:
                experiment_row["failures"].append(
                    {
                        "attempt_id": attempt_id,
                        "status": status,
                        "fidelity": fidelity,
                        "stop_reason": stop_reason,
                        "reason_codes": attempt["reason_codes"],
                        "stderr_tail": attempt["stderr_tail"],
                        "notes": raw.get("notes"),
                    }
                )

        for experiment_row in package_experiments.values():
            experiment_row["metrics"] = self._dedupe_package_metrics(experiment_row["metrics"])
            experiment_row["summary_for_llm"] = self._package_experiment_summary(experiment_row)

        return {
            "schema_version": "phase2_execution_package.v1",
            "source_files": source_files,
            "experiments": list(package_experiments.values()),
            "raw_manifest_reason_codes": list(raw_manifest.get("reason_codes", [])) if isinstance(raw_manifest, dict) else [],
            "reason_codes": ["PHASE2_EXECUTION_PACKAGE_BUILT"],
        }

    @staticmethod
    def _paper_target_refs_for_experiment(experiment_id: str, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for claim in claims:
            conditions = claim.get("conditions", {}) if isinstance(claim.get("conditions"), dict) else {}
            if str(conditions.get("experiment_id") or "").strip() != experiment_id:
                continue
            if claim.get("type") != "result":
                continue
            refs.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "predicate": claim.get("predicate"),
                    "metric": claim.get("metric"),
                    "target": claim.get("target"),
                    "scope": conditions.get("scope"),
                    "tolerance_policy": claim.get("tolerance_policy", {}),
                }
            )
        return refs

    @classmethod
    def _canonical_phase2_experiment_id(
        cls,
        raw_experiment_id: str,
        raw: dict[str, Any],
        scope: dict[str, Any],
        experiment_by_id: dict[str, dict[str, Any]],
    ) -> str:
        if raw_experiment_id in experiment_by_id:
            return raw_experiment_id
        match = re.match(r"^(exp)_(\d+)", raw_experiment_id.lower())
        if match:
            padded = f"exp_{int(match.group(2)):02d}"
            if padded in experiment_by_id:
                return padded
            unpadded = f"exp_{int(match.group(2))}"
            if unpadded in experiment_by_id:
                return unpadded
        text = cls._scope_text(raw, scope)
        if ("conv" in text or "convolutional" in text) and "exp_02" in experiment_by_id:
            return "exp_02"
        if ("fc" in text or "fully connected" in text or "fully_connected" in text) and "exp_01" in experiment_by_id:
            return "exp_01"
        return raw_experiment_id or "unknown_experiment"

    @classmethod
    def _infer_attempt_scope(cls, raw: dict[str, Any]) -> dict[str, Any]:
        text = cls._scope_text(raw, {})
        dataset = cls._infer_dataset(raw, text)
        algorithm = cls._infer_algorithm(text)
        model_family = cls._infer_model_family(text, raw)
        epochs = cls._infer_epochs(text)
        return {
            "algorithm": algorithm,
            "dataset": dataset,
            "model_family": model_family,
            "epochs": epochs,
        }

    @staticmethod
    def _scope_text(raw: dict[str, Any], scope: dict[str, Any]) -> str:
        values = [
            raw.get("experiment_id"),
            raw.get("experiment_name"),
            raw.get("config_name"),
            raw.get("dataset"),
            raw.get("command"),
            " ".join(raw.get("commands_attempted", []) or []) if isinstance(raw.get("commands_attempted"), list) else "",
            raw.get("notes"),
            scope.get("algorithm"),
            scope.get("dataset"),
            scope.get("model_family"),
        ]
        text = " ".join(str(value or "") for value in values).lower()
        return (
            text.replace("cifar-10", "cifar10")
            .replace("cifar 10", "cifar10")
            .replace("cifar-100", "cifar100")
            .replace("cifar 100", "cifar100")
            .replace("fully-connected", "fully connected")
            .replace("fully_connected", "fully connected")
        )

    @staticmethod
    def _infer_dataset(raw: dict[str, Any], text: str) -> str | None:
        explicit = str(raw.get("dataset") or "").strip().lower()
        explicit = explicit.replace("cifar-10", "cifar10").replace("cifar-100", "cifar100")
        if explicit in {"mnist", "cifar10", "cifar100"}:
            return explicit
        if "cifar100" in text or "--dataset cif100" in text:
            return "cifar100"
        if "cifar10" in text or "--cifar10" in text or "--dataset cif" in text:
            return "cifar10"
        if "mnist" in text or "--mnist" in text or "--dataset mn" in text:
            return "mnist"
        return None

    @staticmethod
    def _infer_algorithm(text: str) -> str | None:
        learn_type = re.search(r"--learn[_-]?type\s+([a-z0-9_]+)", text)
        if learn_type:
            return "pepita" if learn_type.group(1) == "erin" else learn_type.group(1)
        tokens = {tok for tok in re.split(r"[^a-z0-9]+", text) if tok}
        for name in ("pepita", "erin", "drtp", "dfa", "bp", "fa"):
            if name in tokens:
                return "pepita" if name == "erin" else name
        return None

    @staticmethod
    def _infer_model_family(text: str, raw: dict[str, Any]) -> str | None:
        if "conv" in text or "convolutional" in text or "main_pytorch.py" in text:
            return "conv"
        if " fc" in f" {text} " or "_fc" in text or "fully connected" in text or "main.py" in text:
            return "fc"
        model = str(raw.get("model") or "").lower()
        if "conv" in model:
            return "conv"
        return None

    @staticmethod
    def _infer_epochs(text: str) -> int | None:
        for pattern in (r"--train[_-]?epochs\s+(\d+)", r"--epochs\s+(\d+)", r"with\s+(\d+)\s+epochs?"):
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def _unique_attempt_id(
        cls,
        *,
        canonical_experiment_id: str,
        scope: dict[str, Any],
        fidelity: str | None,
        used: set[str],
        fallback_index: int,
    ) -> str:
        parts = [
            canonical_experiment_id,
            scope.get("algorithm") or "unknown",
            scope.get("dataset") or "unknown",
            scope.get("model_family") or "model",
            fidelity or "unknown",
        ]
        if scope.get("epochs") is not None:
            parts.append(f"{scope['epochs']}ep")
        base = ".".join(cls._safe_identifier(part) for part in parts if part)
        if not base or base in {"unknown"}:
            base = f"{canonical_experiment_id}.attempt_{fallback_index}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}.{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    @staticmethod
    def _safe_identifier(value: Any) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip()).strip("_.") or "unknown"

    @classmethod
    def _package_metric_entries(
        cls,
        *,
        metrics: dict[str, Any],
        scope: dict[str, Any],
        fidelity: str | None,
        attempt_id: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, float | None, str]] = set()
        for raw_name, raw_value in metrics.items():
            if str(raw_name).endswith("_all"):
                continue
            raw_metric_name = cls._normalize_package_metric_name(str(raw_name))
            metric_name = cls._scoped_package_metric_name(raw_metric_name, scope, fidelity)
            parsed_value = cls._to_metric_float(raw_value)
            bounded = cls._is_bounded_package_metric(metric_name) or cls._is_bounded_package_metric(raw_metric_name)
            value_ratio = parsed_value
            if bounded and value_ratio is not None and value_ratio > 1.0:
                value_ratio = value_ratio / 100.0
            key = (metric_name, value_ratio, str(raw_value))
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "metric_name": metric_name,
                    "raw_metric_name": raw_metric_name,
                    "value_ratio": value_ratio if bounded else None,
                    "value": value_ratio if bounded else parsed_value,
                    "raw_value": raw_value,
                    "unit": "ratio" if bounded else "raw",
                    "algorithm": scope.get("algorithm"),
                    "dataset": scope.get("dataset"),
                    "model_family": scope.get("model_family"),
                    "fidelity": fidelity,
                    "source_attempt_id": attempt_id,
                }
            )
        return entries

    @staticmethod
    def _normalize_package_metric_name(metric_name: str) -> str:
        normalized = re.sub(r"[\s\-]+", "_", metric_name.strip().lower()).strip("_")
        aliases = {"acc": "accuracy", "f1_score": "f1", "f1score": "f1", "f1-score": "f1"}
        return aliases.get(normalized, normalized)

    @classmethod
    def _scoped_package_metric_name(cls, raw_metric_name: str, scope: dict[str, Any], fidelity: str | None) -> str:
        pieces = [scope.get("algorithm"), scope.get("dataset"), scope.get("model_family")]
        if not all(pieces):
            return raw_metric_name
        metric_tail = raw_metric_name
        if raw_metric_name in {"accuracy", "acc"}:
            metric_tail = "test_accuracy"
        prefix = [str(piece) for piece in pieces if piece]
        if fidelity:
            prefix.append(fidelity)
        return "_".join(prefix + [metric_tail])

    @staticmethod
    def _to_metric_float(raw_value: Any) -> float | None:
        if raw_value is None or isinstance(raw_value, bool):
            return None
        try:
            return float(str(raw_value).strip().rstrip("%"))
        except ValueError:
            return None

    @staticmethod
    def _safe_int(raw_value: Any, default: int) -> int:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_bounded_package_metric(metric_name: str | None) -> bool:
        lowered = str(metric_name or "").lower()
        if lowered in _BOUNDED_PACKAGE_METRIC_TOKENS:
            return True
        tokens = {tok for tok in re.split(r"[^a-z0-9]+", lowered) if tok}
        return bool(tokens & _BOUNDED_PACKAGE_METRIC_TOKENS)

    @staticmethod
    def _metric_scope_key(metric: dict[str, Any]) -> str:
        return "|".join(
            str(metric.get(name) or "")
            for name in ("algorithm", "dataset", "model_family", "raw_metric_name")
        )

    @staticmethod
    def _attempt_preferred(
        incoming: dict[str, Any],
        current_attempt_id: str,
        attempts: list[dict[str, Any]],
    ) -> bool:
        current = next((attempt for attempt in attempts if attempt.get("attempt_id") == current_attempt_id), None)
        if current is None:
            return True

        def rank(attempt: dict[str, Any]) -> tuple[int, int, float]:
            status_rank = 2 if attempt.get("status") == "ok" else 1 if attempt.get("status") == "partial" else 0
            fidelity_rank = _PACKAGE_FIDELITY_RANK.get(attempt.get("fidelity"), 0)
            runtime = float(attempt.get("runtime_sec") or 0.0)
            return status_rank, fidelity_rank, runtime

        return rank(incoming) > rank(current)

    @staticmethod
    def _dedupe_package_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, Any, str]] = set()
        for metric in metrics:
            key = (
                str(metric.get("metric_name") or ""),
                metric.get("value_ratio") if metric.get("value_ratio") is not None else metric.get("value"),
                str(metric.get("source_attempt_id") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(metric)
        return deduped

    @staticmethod
    def _package_experiment_summary(experiment: dict[str, Any]) -> str:
        attempts = experiment.get("attempts", [])
        if not attempts:
            return "No Phase 2 execution attempt was recorded for this experiment."
        ok = [attempt for attempt in attempts if attempt.get("status") in {"ok", "partial"}]
        metrics = experiment.get("metrics", [])
        metric_bits = []
        for metric in metrics[:5]:
            value = metric.get("value_ratio") if metric.get("value_ratio") is not None else metric.get("value")
            if value is None:
                continue
            metric_bits.append(f"{metric.get('metric_name')}={value}")
        return (
            f"{len(attempts)} attempts recorded; {len(ok)} produced executable evidence. "
            f"Best scoped attempts: {json.dumps(experiment.get('best_attempts_by_scope', {}), ensure_ascii=False)}. "
            f"Metrics: {', '.join(metric_bits) if metric_bits else 'none'}."
        )

    @staticmethod
    def _render_phase2_results_markdown(package: dict[str, Any]) -> str:
        lines = [
            "# Phase 2 Results",
            "",
            "Canonical source: `execution/executor_outputs/phase2_execution_package.json`.",
            "Raw executor summaries and manifests are same-origin debug/audit evidence only.",
            "",
            "| Experiment | Scope | Attempt | Status | Fidelity | Metric | Value | Notes |",
            "|------------|-------|---------|--------|----------|--------|-------|-------|",
        ]
        for experiment in package.get("experiments", []):
            experiment_label = experiment.get("name") or experiment.get("experiment_id")
            attempts = experiment.get("attempts", [])
            if not attempts:
                lines.append(
                    f"| {experiment_label} | - | - | not_run | - | - | - | No Phase 2 attempt recorded. |"
                )
                continue
            for attempt in attempts:
                scope = attempt.get("scope", {})
                scope_label = "/".join(
                    str(scope.get(key) or "-")
                    for key in ("algorithm", "dataset", "model_family")
                )
                metrics = attempt.get("metrics", [])
                if not metrics:
                    reason = ",".join(attempt.get("reason_codes", []) or [])
                    lines.append(
                        f"| {experiment_label} | {scope_label} | {attempt.get('attempt_id')} | "
                        f"{attempt.get('status')} | {attempt.get('fidelity')} | - | - | "
                        f"{attempt.get('stop_reason') or reason or attempt.get('notes') or ''} |"
                    )
                    continue
                for metric in metrics:
                    value = metric.get("value_ratio") if metric.get("value_ratio") is not None else metric.get("value")
                    if metric.get("unit") == "ratio" and isinstance(value, (int, float)):
                        value_text = f"{value * 100:.2f}%"
                    else:
                        value_text = str(value)
                    lines.append(
                        f"| {experiment_label} | {scope_label} | {attempt.get('attempt_id')} | "
                        f"{attempt.get('status')} | {attempt.get('fidelity')} | {metric.get('metric_name')} | "
                        f"{value_text} | {attempt.get('notes') or ''} |"
                    )
        return "\n".join(lines) + "\n"

    def _recover_misplaced_executor_outputs(self, repo_dir: Path, outputs_dir: Path) -> Path | None:
        """Recover result files when the executor writes artifacts relative to the target repo."""
        canonical_results = outputs_dir / "executor_results.json"
        if canonical_results.is_file():
            return None

        candidate_dirs = [
            repo_dir / "artifacts" / self.artifacts.run_id / "execution" / "executor_outputs",
        ]
        preserve_existing = {
            "executor_activity.jsonl",
            "executor_agent.log",
            "executor_runtime.json",
            "run_manifest.json",
            "session_stdout.log",
            "session_stderr.log",
        }
        copied: list[str] = []
        for candidate_dir in candidate_dirs:
            candidate_results = candidate_dir / "executor_results.json"
            if not candidate_results.is_file():
                continue
            outputs_dir.mkdir(parents=True, exist_ok=True)
            for child in candidate_dir.iterdir():
                if not child.is_file():
                    continue
                dest = outputs_dir / child.name
                if dest.exists() and child.name in preserve_existing:
                    continue
                if dest.exists() and child.name != "executor_results.json":
                    continue
                try:
                    if child.stat().st_size > 20 * 1024 * 1024:
                        continue
                    shutil.copy2(child, dest)
                    copied.append(f"execution/executor_outputs/{child.name}")
                except OSError:
                    continue
            self._append_activity(
                event="executor_outputs_recovered",
                experiment_id=None,
                cwd=str(repo_dir),
                command="recover_executor_outputs",
                status="ok" if canonical_results.is_file() else "failed",
                exit_code=0 if canonical_results.is_file() else 1,
                duration_sec=0.0,
                artifacts=copied,
                message=f"Recovered executor outputs from misplaced target-repo path: {candidate_dir}",
            )
            return candidate_dir
        return None

    def _existing_executor_log_or_fallback(self, path_value: Any, experiment_id: str, kind: str) -> str:
        candidate = self._canonical_executor_output_ref(path_value)
        if candidate and self._artifact_exists(candidate):
            return candidate

        fallbacks_by_kind = {
            "stdout": [
                f"execution/executor_outputs/experiment_{experiment_id}_stdout.log",
            ],
            "stderr": [
                f"execution/executor_outputs/experiment_{experiment_id}_stderr.log",
            ],
            "narrative": [
                f"execution/executor_outputs/experiment_{experiment_id}_narrative.log",
                "execution/executor_outputs/executor_agent.log",
            ],
            "activity": ["execution/executor_outputs/executor_activity.jsonl"],
        }
        for fallback in fallbacks_by_kind.get(kind, []):
            if self._artifact_exists(fallback):
                return fallback
        return candidate

    def _ensure_per_run_logs(
        self,
        *,
        experiment_id: str,
        raw: dict[str, Any],
        stdout_log: str,
        stderr_log: str,
        narrative_log: str,
        log_stem: str | None = None,
    ) -> tuple[str, str, str, str | None]:
        stem = log_stem or experiment_id
        expected_stdout = f"execution/executor_outputs/experiment_{stem}_stdout.log"
        expected_stderr = f"execution/executor_outputs/experiment_{stem}_stderr.log"
        expected_narrative = f"execution/executor_outputs/experiment_{stem}_narrative.log"

        needs_synthetic = (
            not stdout_log
            or stdout_log.endswith("/session_stdout.log")
            or not self._artifact_exists(stdout_log)
        )
        if not needs_synthetic:
            synthetic_reason = None
            if not stderr_log or not self._artifact_exists(stderr_log):
                self.artifacts.write_text(expected_stderr, str(raw.get("stderr_tail") or ""))
                stderr_log = expected_stderr
                synthetic_reason = "SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS"
            if not narrative_log or not self._artifact_exists(narrative_log):
                notes = str(raw.get("notes") or "").strip()
                narrative_text = notes or "Synthetic per-run narrative generated from executor_results.json."
                self.artifacts.write_text(expected_narrative, narrative_text + ("\n" if narrative_text else ""))
                narrative_log = expected_narrative
                synthetic_reason = "SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS"
            if synthetic_reason:
                return stdout_log, stderr_log, narrative_log, synthetic_reason
            return stdout_log, stderr_log, narrative_log, None

        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        stdout_lines: list[str] = []
        notes = str(raw.get("notes") or "").strip()
        if notes:
            stdout_lines.append(notes)
        for metric_name, value in metrics.items():
            if isinstance(value, bool) or value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            stdout_lines.append(f"METRIC:{metric_name}={value}")
        if raw.get("stdout_tail"):
            stdout_lines.append(str(raw.get("stdout_tail")))

        stderr_text = str(raw.get("stderr_tail") or "")
        narrative_text = notes or "Synthetic per-run narrative generated from executor_results.json."

        self.artifacts.write_text(expected_stdout, "\n".join(stdout_lines).strip() + ("\n" if stdout_lines else ""))
        self.artifacts.write_text(expected_stderr, stderr_text)
        self.artifacts.write_text(expected_narrative, narrative_text + ("\n" if narrative_text else ""))
        return expected_stdout, expected_stderr, expected_narrative, "SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS"

    @staticmethod
    def _canonical_executor_output_ref(path_value: Any) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return ""
        normalized = raw.replace("\\", "/")
        marker = "execution/executor_outputs/"
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return normalized[marker_index:]
        return raw

    @staticmethod
    def _command_was_observed(
        command: str,
        observed_command_set: set[str],
        observed_commands: list[str],
    ) -> bool:
        normalized = _normalize_shell_command(command)
        if not normalized:
            return True
        if normalized in observed_command_set:
            return True
        normalized_suffix = ExecutorAgent._runtime_stripped_command(normalized)
        if normalized_suffix in observed_command_set:
            return True
        return any(
            normalized in observed
            or ExecutorAgent._command_suffix_match(normalized, observed)
            for observed in observed_commands
        )

    @staticmethod
    def _runtime_stripped_command(command: str) -> str:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if not tokens:
            return ""
        for idx, token in enumerate(tokens):
            if Path(token).name in {"python", "python3", "pip"}:
                return " ".join(tokens[idx:])
        return " ".join(tokens)

    @staticmethod
    def _command_suffix_match(declared: str, observed: str) -> bool:
        declared_suffix = ExecutorAgent._runtime_stripped_command(declared)
        observed_suffix = ExecutorAgent._runtime_stripped_command(observed)
        if not declared_suffix or not observed_suffix:
            return False
        return declared_suffix == observed_suffix or declared_suffix in observed_suffix or observed_suffix in declared_suffix

    def _read_log(self, path_value: str) -> str:
        if not path_value:
            return ""
        path = Path(path_value)
        if not path.is_absolute():
            path = self.artifacts.path(path_value)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")

    def _artifact_exists(self, path_value: str, *, repo_dir: Path | None = None) -> bool:
        if not path_value:
            return False
        path = Path(path_value)
        if path.is_absolute():
            return path.exists()
        if self.artifacts.path(path_value).exists():
            return True
        return bool(repo_dir and (repo_dir / path_value).exists())

    @staticmethod
    def _normalize_run_status(raw_status: Any, exit_code: Any, raw_present: bool) -> str:
        candidate = str(raw_status or "").strip().lower()
        if candidate in {"ok", "partial", "failed", "skipped"}:
            return candidate
        if candidate in {"success", "succeeded", "complete", "completed", "pass", "passed"}:
            return "ok"
        if candidate in {"partially_successful", "degraded_success"}:
            return "partial"
        if candidate in {"skip", "not_run"}:
            return "skipped"
        if not raw_present:
            return "failed"
        return "ok" if int(exit_code or 1) == 0 else "failed"

    @staticmethod
    def _normalize_evidence_source(raw_value: Any, stdout_text: str, existing_artifacts: list[str], commands_attempted: list[str]) -> str | None:
        candidate = re.sub(r"[^a-z0-9]+", "_", str(raw_value or "").strip().lower()).strip("_")
        if candidate in {"existing_artifact", "existing_artifacts"}:
            return "existing_results"
        if candidate == "fresh_runs":
            return "fresh_run"
        if candidate in {"fresh_run", "checkpoint_eval", "existing_logs", "existing_results", "mixed"}:
            return candidate
        if existing_artifacts and stdout_text:
            return "mixed"
        if existing_artifacts:
            return "existing_results"
        if stdout_text or commands_attempted:
            return "fresh_run"
        return None

    @staticmethod
    def _merge_signal_hints(
        existing: list[str],
        *,
        stdout_text: str,
        metrics: dict[str, Any],
        artifacts: list[str],
    ) -> list[str]:
        out = [signal_name for signal_name in existing if signal_name]
        for signal_name, predicate in (
            ("log_format_confirmed", bool(stdout_text)),
            ("metric_observed", bool(metrics)),
            ("artifact_written", bool(artifacts)),
            ("val_metric_seen", any(str(name).startswith(("val_", "test_")) for name in metrics)),
        ):
            if predicate and signal_name not in out:
                out.append(signal_name)
        return out

    @staticmethod
    def _normalize_fidelity(
        raw_value: Any,
        evidence_source: str | None,
        override_args: list[str],
        metrics: dict[str, Any],
        observed_signals: list[str],
    ) -> str | None:
        candidate = str(raw_value or "").strip().lower()
        compact = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
        if candidate in {"artifact", "smoke", "trend", "full"}:
            return candidate
        if candidate in {"smoke+trend", "mixed_smoke_trend", "mixed (smoke + trend)"} or compact in {
            "smoke_trend",
            "mixed_smoke_trend",
            "trend_smoke",
        }:
            return "trend"
        if candidate in {"smoke+artifact", "artifact_evaluation", "mixed_artifact"} or compact in {
            "smoke_artifact",
            "mixed_artifact",
        }:
            return "artifact"
        if evidence_source in _ARTIFACT_EVIDENCE_SOURCES:
            return "artifact"
        if override_args:
            return "trend" if metrics or observed_signals else "smoke"
        return "full" if metrics or observed_signals else "smoke"

    @staticmethod
    def _normalize_stop_reason(
        raw_value: Any,
        status: str,
        fidelity: str | None,
        evidence_source: str | None,
        observed_signals: list[str],
    ) -> str | None:
        candidate = str(raw_value or "").strip().lower()
        if candidate in {
            "checkpoint_eval",
            "existing_artifact",
            "budget_bound",
            "early_stop_evidence",
            "full_run_complete",
            "repo_missing_path",
            "runtime_failure",
            "guardrail_blocked",
            "skipped_nonessential",
        }:
            return candidate
        if "budget" in candidate:
            return "budget_bound"
        if "artifact" in candidate:
            return "existing_artifact"
        if "implementation" in candidate or "guardrail" in candidate:
            return "guardrail_blocked"
        if status == "skipped":
            return "skipped_nonessential"
        if evidence_source == "checkpoint_eval":
            return "checkpoint_eval"
        if evidence_source in {"existing_logs", "existing_results", "mixed"} and status in {"ok", "partial"}:
            return "existing_artifact"
        if fidelity == "full" and status in {"ok", "partial"}:
            return "full_run_complete"
        if fidelity in {"smoke", "trend"} and observed_signals:
            return "early_stop_evidence"
        if status == "failed":
            return "runtime_failure"
        return None

    @staticmethod
    def _compute_execution_outcome(
        *,
        fidelity: str | None,
        status: str,
        evidence_source: str | None,
        override_args: list[str],
        metrics: dict[str, Any],
        observed_signals: list[str],
    ) -> str | None:
        if status not in {"ok", "partial"}:
            return None
        if fidelity == "full" and evidence_source == "fresh_run" and not override_args:
            return "FULLY_REPRODUCED"
        if fidelity in {"trend", "artifact"} and (metrics or observed_signals):
            return "TREND_SUPPORTED"
        if fidelity == "smoke":
            return "EXECUTABLE"
        return None

    # ------------------------------------------------------------------
    # Repo mutation guard
    # ------------------------------------------------------------------

    def _repo_guard_ignore_roots(self, repo_dir: Path) -> list[Path]:
        artifact_root = self.artifacts.run_root.resolve()
        repo_dir = repo_dir.resolve()
        return [artifact_root] if _path_is_relative_to(artifact_root, repo_dir) else []

    @staticmethod
    def _capture_repo_state(repo_dir: Path, *, ignore_roots: list[Path] | None = None) -> dict[str, str]:
        ignore_roots = [path.resolve() for path in (ignore_roots or [])]

        def should_ignore(path: Path) -> bool:
            resolved = path.resolve()
            return any(_path_is_relative_to(resolved, root) for root in ignore_roots)

        if (repo_dir / ".git").exists():
            proc = subprocess.run(
                ["git", "ls-files"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            files = [repo_dir / line.strip() for line in proc.stdout.splitlines() if line.strip()]
        else:
            files = [
                path for path in repo_dir.rglob("*")
                if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
            ]
        state: dict[str, str] = {}
        for path in files:
            if should_ignore(path):
                continue
            try:
                state[str(path.relative_to(repo_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:  # noqa: BLE001
                continue
        return state

    @classmethod
    def _detect_repo_mutation(
        cls,
        repo_dir: Path,
        baseline: dict[str, str],
        *,
        ignore_roots: list[Path] | None = None,
    ) -> list[str]:
        current = cls._capture_repo_state(repo_dir, ignore_roots=ignore_roots)
        changed = []
        for rel_path, digest in baseline.items():
            if current.get(rel_path) != digest:
                changed.append(rel_path)
        return changed

    @staticmethod
    def _is_runtime_artifact_mutation(rel_path: str) -> bool:
        """Return true for common training outputs that repos may track.

        Some older ML repos commit checkpoint/stat files, then overwrite those
        paths during training. That is undesirable, but it is evidence-producing
        runtime output rather than source/config mutation. Treating these as a
        warning prevents a completed reproduction from being silently discarded
        by the outer phase2 guard.
        """
        path = Path(rel_path)
        suffix = path.suffix.lower()
        if suffix in _RUNTIME_ARTIFACT_SUFFIXES:
            return True
        parts = {part.lower() for part in path.parts}
        stem = path.stem.lower()
        return bool(parts.intersection(_RUNTIME_ARTIFACT_NAME_TOKENS)) or any(
            token in stem for token in _RUNTIME_ARTIFACT_NAME_TOKENS
        )

    @classmethod
    def _partition_repo_mutations(cls, mutated_files: list[str]) -> tuple[list[str], list[str]]:
        artifact_mutations: list[str] = []
        blocking_mutations: list[str] = []
        for rel_path in mutated_files:
            if cls._is_runtime_artifact_mutation(rel_path):
                artifact_mutations.append(rel_path)
            else:
                blocking_mutations.append(rel_path)
        return artifact_mutations, blocking_mutations

    def _write_repo_mutation_guard(
        self,
        *,
        mutated_files: list[str],
        artifact_mutations: list[str],
        blocking_mutations: list[str],
    ) -> None:
        if not mutated_files:
            return
        reason_codes = []
        if artifact_mutations:
            reason_codes.append("REPO_RUNTIME_ARTIFACT_MUTATION_RECORDED")
        if blocking_mutations:
            reason_codes.append("SOURCE_MUTATION_DETECTED")
        self.artifacts.write_json(
            _REPO_MUTATION_GUARD_REL_PATH,
            {
                "schema_version": "repo_mutation_guard.v1",
                "status": "failed" if blocking_mutations else "warning",
                "mutated_files": mutated_files,
                "runtime_artifact_mutations": artifact_mutations,
                "blocking_mutations": blocking_mutations,
                "reason_codes": reason_codes,
            },
        )

    @staticmethod
    def _collect_descendant_pids(root_pid: int) -> list[int]:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            capture_output=True,
            text=True,
            check=False,
        )
        children_by_parent: dict[int, list[int]] = {}
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            pid, ppid = int(parts[0]), int(parts[1])
            children_by_parent.setdefault(ppid, []).append(pid)

        descendants: list[int] = []
        stack = list(children_by_parent.get(root_pid, []))
        while stack:
            pid = stack.pop()
            descendants.append(pid)
            stack.extend(children_by_parent.get(pid, []))
        return descendants

    @classmethod
    def _terminate_descendants(cls, root_pid: int) -> None:
        descendants = cls._collect_descendant_pids(root_pid)
        if not descendants:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in reversed(descendants):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    continue
                except PermissionError:
                    continue
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Logging + failures
    # ------------------------------------------------------------------

    def _append_activity(
        self,
        *,
        event: str,
        experiment_id: str | None,
        cwd: str,
        command: str,
        status: str,
        exit_code: int | None,
        duration_sec: float,
        artifacts: list[str],
        message: str,
    ) -> None:
        self.artifacts.append_jsonl(
            "execution/executor_outputs/executor_activity.jsonl",
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                "experiment_id": experiment_id,
                "cwd": cwd,
                "command": command,
                "status": status,
                "exit_code": exit_code,
                "duration_sec": round(duration_sec, 3),
                "artifacts": artifacts,
                "message": message,
            },
        )

    def _backfill_activity_from_runs(self, runs: list[dict[str, Any]]) -> None:
        for run in runs:
            experiment_id = str(run.get("experiment_id") or run.get("run_id") or "")
            command = str(run.get("command") or "")
            commands_attempted = [str(cmd) for cmd in run.get("commands_attempted", []) if str(cmd).strip()]
            self._append_activity(
                event="experiment_start",
                experiment_id=experiment_id,
                cwd=str(run.get("cwd") or "."),
                command=command,
                status="started",
                exit_code=None,
                duration_sec=0.0,
                artifacts=[],
                message=str(run.get("experiment_name") or experiment_id),
            )
            for attempted in commands_attempted or ([command] if command else []):
                self._append_activity(
                    event="command_end",
                    experiment_id=experiment_id,
                    cwd=str(run.get("cwd") or "."),
                    command=attempted,
                    status=str(run.get("status") or "unknown"),
                    exit_code=int(run.get("exit_code", 1)),
                    duration_sec=float(run.get("runtime_sec") or 0.0),
                    artifacts=[],
                    message="Recorded from executor_results.json",
                )
            for metric_name, value in (run.get("metrics") or {}).items():
                if str(metric_name).endswith("_all"):
                    continue
                self._append_activity(
                    event="metric_observed",
                    experiment_id=experiment_id,
                    cwd=str(run.get("cwd") or "."),
                    command=command,
                    status="ok",
                    exit_code=0,
                    duration_sec=0.0,
                    artifacts=[],
                    message=f"{metric_name}={value}",
                )
            for artifact_path in run.get("artifacts", []):
                self._append_activity(
                    event="artifact_recorded",
                    experiment_id=experiment_id,
                    cwd=str(run.get("cwd") or "."),
                    command=command,
                    status="ok",
                    exit_code=0,
                    duration_sec=0.0,
                    artifacts=[str(artifact_path)],
                    message="Artifact recorded from executor_results.json",
                )

    def _fail_fast(
        self,
        message: str,
        *,
        failure_code: str,
        stdout_tail: str = "",
        stderr_tail: str = "",
    ) -> ExecutionFailure:
        spec = classify_error_v2(stdout_tail, stderr_tail or message, 1, metrics={}, expected_metrics=[])
        return ExecutionFailure(
            attempt=1,
            stage="execution",
            step_failures=[
                StepFailure(
                    step_id="executor_session",
                    command="executor session",
                    exit_code=1,
                    error_type=spec.legacy_error_type,
                    error_message=message,
                    stdout_tail=stdout_tail[-2000:],
                    stderr_tail=stderr_tail[-2000:],
                    failure_code=failure_code,
                    failure_layer=spec.layer,
                    repair_strategy=spec.repair_strategy.value,
                    repair_action=spec.repair_action,
                    auto_repair_confidence=spec.auto_repair_confidence,
                )
            ],
            overall_error=message,
            is_dependency_issue=spec.layer == "dependency",
            reason_codes=[failure_code],
        )
