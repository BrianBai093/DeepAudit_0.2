from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2c.agents.phase2.collect_codex_outputs_newstyle import CollectCodexOutputsNewstyleAgent
from p2c.agents.phase2.prepare_sandbox_newstyle import PrepareSandboxNewstyleAgent
from p2c.agents.phase2.run_codex_exec_newstyle import RunCodexExecNewstyleAgent
from p2c.graph import run_phase_2
from p2c.io_artifacts import ArtifactManager
from p2c.runtime.base import RuntimeCommandResult


class DummyLLM:
    def chat_text(self, system: str, user: str) -> str:
        return ""

    def chat_json(self, schema, system: str, user: str):
        return {"notes": "", "reason_codes": []}


class FakeRuntime:
    backend_name = "e2b"

    def __init__(self) -> None:
        self.remote_files: dict[str, str] = {}
        self.uploaded_dirs: list[tuple[str, str, list[str] | None]] = []

    def upload_dir(self, local_dir: Path, remote_dir: str, exclude_globs: list[str] | None = None) -> None:
        self.uploaded_dirs.append((str(local_dir), remote_dir, exclude_globs))
        self.remote_files[f"{remote_dir}/.uploaded"] = "ok"

    def upload_file(self, local_file: Path, remote_path: str) -> None:
        self.remote_files[remote_path] = local_file.read_text(encoding="utf-8")

    def download_file(self, remote_path: str, local_file: Path) -> None:
        if remote_path not in self.remote_files:
            raise FileNotFoundError(remote_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text(self.remote_files[remote_path], encoding="utf-8")

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
        if 'test -n "$OPENAI_API_KEY"' in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "python3 -m pip --version" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="pip 24.0\n", stderr="")
        tool_paths = {
            "python": "/usr/bin/python",
            "python3": "/usr/bin/python3",
            "pip": "/usr/bin/pip",
            "pip3": "/usr/bin/pip3",
            "poetry": "/home/user/.local/bin/poetry",
            "uv": "/home/user/.local/bin/uv",
            "node": "/usr/bin/node",
            "npm": "/usr/bin/npm",
            "codex": "/usr/local/bin/codex",
            "apply_patch": "/workspace/bin/apply_patch",
        }
        for tool, path in tool_paths.items():
            if f"command -v {tool}" in command:
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout=path + "\n", stderr="")
            if f"{tool} --version" in command or (tool in {"python", "python3"} and f"{tool} -V" in command):
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout=f"{tool} version\n", stderr="")
        if "test -w" in command or "mkdir -p" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "python3 - <<'PY'" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="Linux\n6.8.0\n3.11.9\n", stderr="")
        return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")

    def read_text(self, remote_path: str) -> str:
        if remote_path not in self.remote_files:
            raise FileNotFoundError(remote_path)
        return self.remote_files[remote_path]

    def write_text(self, remote_path: str, content: str) -> None:
        self.remote_files[remote_path] = content


def make_artifacts(tmp_path: Path, run_id: str = "run_new") -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path, run_id)
    artifacts.ensure_tree()
    return artifacts


def install_runtime_monkeypatch(monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime) -> None:
    import p2c.agents.phase2.collect_codex_outputs_newstyle as collect_mod
    import p2c.agents.phase2.prepare_sandbox_newstyle as prepare_mod
    import p2c.agents.phase2.run_codex_exec_newstyle as run_mod

    monkeypatch.setattr(prepare_mod, "ensure_runtime", lambda ctx, artifacts: runtime)
    monkeypatch.setattr(run_mod, "ensure_runtime", lambda ctx, artifacts: runtime)
    monkeypatch.setattr(collect_mod, "ensure_runtime", lambda ctx, artifacts: runtime)


def test_graph_routes_to_newstyle_phase2() -> None:
    calls: list[str] = []

    class Stub:
        def __init__(self, name: str):
            self.name = name

        def run(self, ctx):
            calls.append(self.name)

    agents = {
        "prepare_sandbox": Stub("prepare_sandbox"),
        "run_codex_exec": Stub("run_codex_exec"),
        "collect_codex_outputs": Stub("collect_codex_outputs"),
        "prepare_sandbox_newstyle": Stub("prepare_sandbox_newstyle"),
        "run_codex_exec_newstyle": Stub("run_codex_exec_newstyle"),
        "collect_codex_outputs_newstyle": Stub("collect_codex_outputs_newstyle"),
    }
    run_phase_2({"phase2_style": "new"}, agents)

    assert calls == [
        "prepare_sandbox_newstyle",
        "run_codex_exec_newstyle",
        "collect_codex_outputs_newstyle",
    ]


def test_prepare_and_run_newstyle_single_exec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    repo_dir = tmp_path / "repo_root"
    code_dir = repo_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    artifacts.write_json(
        "task/task_spec.json",
        {
            "tasks": [
                {
                    "task_id": "task_01",
                    "entrypoint": "code/train.py",
                    "command": "python3 code/train.py",
                    "cwd": "code",
                    "runtime": "python",
                }
            ]
        },
    )
    ctx = {"repo_dir": str(repo_dir), "budget_minutes": 30}

    prepare = PrepareSandboxNewstyleAgent(llm=DummyLLM(), artifacts=artifacts, step_index=8, step_total=15)
    prepare.execute(ctx)

    calls: list[dict] = []

    def fake_bg_run(runtime_obj, **kwargs):
        calls.append(kwargs)
        summary = {
            "project_type": "python-cli",
            "dependency_steps": ["python3 -m pip install --user numpy"],
            "commands_run": ["python3 train.py"],
            "success_basis": "run",
            "execution_succeeded": True,
            "attempt_count": 1,
            "task_results": [
                {
                    "task_id": "task_01",
                    "planned_command": "python3 train.py",
                    "final_command": "python3 train.py",
                    "status": "ok",
                    "notes": "",
                }
            ],
            "remaining_blockers": [],
        }
        runtime_obj.write_text("/workspace/outputs/execution_summary.json", json.dumps(summary))
        runtime_obj.write_text("/workspace/outputs/codex_main.log", json.dumps(summary))
        runtime_obj.write_text("/workspace/outputs/codex_exec.log", "ok")
        runtime_obj.write_text("/workspace/outputs/patches.diff", "")
        return {
            "rc": 0,
            "timed_out": False,
            "polls": 1,
            "pid_path": "/workspace/outputs/codex_exec.pid",
            "exit_path": "/workspace/outputs/codex_exec.rc",
            "log_path": "/workspace/outputs/codex_main.log",
        }

    run_agent = RunCodexExecNewstyleAgent(llm=DummyLLM(), artifacts=artifacts, step_index=9, step_total=15)
    monkeypatch.setattr(run_agent.bg, "run", fake_bg_run)
    run_agent.execute(ctx)

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert "codex exec" in cmd
    assert "execution_summary.json" in cmd
    assert "task_run_results.json" not in cmd
    assert "claim_alignment.json" not in cmd
    assert "Read `/workspace/inputs/codex_execution_skill.md` first" in cmd


def test_collect_newstyle_accepts_summary_and_logs_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    runtime.write_text(
        "/workspace/outputs/execution_summary.json",
        json.dumps(
            {
                "project_type": "python-cli",
                "dependency_steps": [],
                "commands_run": ["python3 train.py"],
                "success_basis": "run",
                "execution_succeeded": True,
                "attempt_count": 1,
                "task_results": [],
                "remaining_blockers": [],
            }
        ),
    )
    runtime.write_text("/workspace/outputs/codex_main.log", "summary")
    runtime.write_text("/workspace/outputs/codex_exec.log", "exec")

    agent = CollectCodexOutputsNewstyleAgent(llm=DummyLLM(), artifacts=artifacts, step_index=10, step_total=15)
    result = agent.execute({"workspace_outputs_dir": "/workspace/outputs"})

    assert result["codex_outputs"]["execution_succeeded"] is True
    assert artifacts.path("execution/codex_outputs/execution_summary.json").exists()
