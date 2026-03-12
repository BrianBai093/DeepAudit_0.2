from __future__ import annotations

import json
import re
from pathlib import Path
import sys
import types

import pytest

from p2c.agents.phase2.collect_codex_outputs import CollectCodexOutputsAgent
from p2c.agents.phase2.codex_prompt_templates import build_autonomous_discovery_prompt
from p2c.agents.phase2.prepare_sandbox import PrepareSandboxAgent
from p2c.agents.phase2.run_codex_exec import RunCodexExecAgent
from p2c.io_artifacts import ArtifactManager
from p2c.runtime.base import RuntimeCommandResult
from p2c.runtime.e2b_runtime import E2BRuntime
from scripts.build_e2b_codex_template import _call_method, _call_template_build


class DummyLLM:
    def chat_text(self, system: str, user: str) -> str:
        return ""

    def chat_json(self, schema, system: str, user: str):
        return {"notes": "", "reason_codes": []}


class FakeRuntime:
    backend_name = "e2b"

    def __init__(self) -> None:
        self.remote_files: dict[str, str] = {}
        self.commands: list[tuple[str, str, int]] = []
        self.uploaded_dirs: list[tuple[str, str, list[str] | None]] = []
        self.uploaded_files: list[tuple[str, str]] = []

    def ensure_started(self) -> None:
        return

    def metadata(self) -> dict[str, str]:
        return {"backend": "fake-e2b"}

    def upload_dir(self, local_dir: Path, remote_dir: str, exclude_globs: list[str] | None = None) -> None:
        self.uploaded_dirs.append((str(local_dir), remote_dir, exclude_globs))
        self.remote_files[f"{remote_dir}/.uploaded"] = "ok"

    def upload_file(self, local_file: Path, remote_path: str) -> None:
        self.uploaded_files.append((str(local_file), remote_path))
        self.remote_files[remote_path] = local_file.read_text(encoding="utf-8")

    def download_file(self, remote_path: str, local_file: Path) -> None:
        if remote_path not in self.remote_files:
            raise FileNotFoundError(remote_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text(self.remote_files[remote_path], encoding="utf-8")

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
        self.commands.append((command, cwd, timeout_sec))

        if 'test -n "$OPENAI_API_KEY"' in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "python3 -m pip --version" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="pip 24.0\n", stderr="")
        if "command -v codex" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="/usr/local/bin/codex\n", stderr="")
        for tool, path in {
            "python": "/usr/bin/python",
            "python3": "/usr/bin/python3",
            "pip": "/usr/bin/pip",
            "pip3": "/usr/bin/pip3",
            "poetry": "/home/user/.local/bin/poetry",
            "uv": "/home/user/.local/bin/uv",
            "node": "/usr/bin/node",
            "npm": "/usr/bin/npm",
            "Rscript": "/usr/bin/Rscript",
            "apply_patch": "/workspace/bin/apply_patch",
        }.items():
            if f"command -v {tool}" in command:
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout=path + "\n", stderr="")
            if f"{tool} --version" in command or (tool in {"python", "python3"} and f"{tool} -V" in command):
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout=f"{tool} version\n", stderr="")
        if "test -w" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "python3 - <<'PY'" in command:
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=0,
                stdout="Linux\n6.8.0\n3.11.9\n8.0\n",
                stderr="",
            )

        file_match = re.search(r"test -f\s+(/[^'\"\s;]+)", command)
        if file_match:
            path = file_match.group(1)
            rc = 0 if path in self.remote_files else 1
            return RuntimeCommandResult(command=command, cwd=cwd, rc=rc, stdout="", stderr="")

        truncate_match = re.search(r": >\s+(/[^'\"\s;]+)", command)
        if truncate_match:
            self.remote_files[truncate_match.group(1)] = ""

        return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")

    def read_text(self, remote_path: str) -> str:
        if remote_path not in self.remote_files:
            raise FileNotFoundError(remote_path)
        return self.remote_files[remote_path]

    def write_text(self, remote_path: str, content: str) -> None:
        self.remote_files[remote_path] = content

    def close(self) -> None:
        return


def make_artifacts(tmp_path: Path, run_id: str = "run1") -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path, run_id)
    artifacts.ensure_tree()
    return artifacts


def write_task_spec(artifacts: ArtifactManager) -> dict:
    payload = {
        "tasks": [
            {
                "task_id": "task_01",
                "entrypoint": "train.py",
                "command": "python3 train.py",
                "cwd": ".",
                "runtime": "python",
                "timeout_class": "medium",
            }
        ],
        "constraints": {},
        "selection_notes": [],
        "reason_codes": [],
    }
    artifacts.write_json("task/task_spec.json", payload)
    return payload


def make_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("Run with python3 train.py\n", encoding="utf-8")
    (repo_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_dir / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    return repo_dir


def make_ctx(repo_dir: Path) -> dict:
    return {"repo_dir": str(repo_dir), "budget_minutes": 30}


def install_runtime_monkeypatch(monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime) -> None:
    import p2c.agents.phase2.collect_codex_outputs as collect_mod
    import p2c.agents.phase2.prepare_sandbox as prepare_mod
    import p2c.agents.phase2.run_codex_exec as run_mod

    monkeypatch.setattr(prepare_mod, "ensure_runtime", lambda ctx, artifacts: runtime)
    monkeypatch.setattr(run_mod, "ensure_runtime", lambda ctx, artifacts: runtime)
    monkeypatch.setattr(collect_mod, "ensure_runtime", lambda ctx, artifacts: runtime)


def test_prepare_sandbox_only_requires_task_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    repo_dir = make_repo(tmp_path)
    write_task_spec(artifacts)
    ctx = make_ctx(repo_dir)

    agent = PrepareSandboxAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=3)
    result = agent.execute(ctx)

    assert result["workspace"]["repo"] == "/workspace/repo"
    assert "/workspace/inputs/task_spec.json" in runtime.remote_files
    assert "/workspace/inputs/codex_execution_skill.md" in runtime.remote_files
    assert artifacts.path("execution/data_manifest.json").exists()


def test_discovery_prompt_requires_readme_data_download_steps() -> None:
    prompt = build_autonomous_discovery_prompt(
        repo_dir="/workspace/repo",
        outputs_dir="/workspace/outputs",
        skill_path="/workspace/inputs/codex_execution_skill.md",
    )

    assert "If the README contains explicit data download" in prompt
    assert "Do not skip documented download/setup steps." in prompt
    assert "documented data steps first and record the exact commands you used" in prompt
    assert "Read `/workspace/inputs/codex_execution_skill.md` first and follow it strictly." in prompt
    assert "If a command fails with `No module named X`" in prompt


def test_run_codex_exec_detects_r_requirement_from_repo(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path, run_id="repo_requires_r")
    agent = RunCodexExecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=2, step_total=3)
    repo_dir = tmp_path / "repo_r"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "script.R").write_text("#!/usr/bin/env Rscript\n", encoding="utf-8")

    assert agent._repo_requires_r(str(repo_dir)) is True


def test_prepare_sandbox_uploads_code_subdir_and_rewrites_task_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path, run_id="run_code_subdir")
    repo_dir = tmp_path / "repo_root"
    code_dir = repo_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    (code_dir / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
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
                    "timeout_class": "medium",
                }
            ],
            "entrypoints": [
                {
                    "path": "code/train.py",
                    "command": "python3 code/train.py",
                    "cwd": "code",
                    "runtime": "python",
                    "confidence": 0.9,
                    "evidence": "test",
                }
            ],
            "constraints": {"allowed_modification_scope": "Target/code"},
        },
    )
    ctx = make_ctx(repo_dir)

    agent = PrepareSandboxAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=3)
    agent.execute(ctx)

    assert runtime.uploaded_dirs[0][0].endswith("repo_root/code")
    uploaded_task_spec = json.loads(runtime.remote_files["/workspace/inputs/task_spec.json"])
    assert uploaded_task_spec["tasks"][0]["entrypoint"] == "train.py"
    assert uploaded_task_spec["tasks"][0]["command"] == "python3 train.py"
    assert uploaded_task_spec["tasks"][0]["cwd"] == "."
    assert uploaded_task_spec["constraints"]["allowed_modification_scope"] == "repo"


def test_run_codex_exec_uses_single_full_access_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    write_task_spec(artifacts)
    ctx = {
        "workspace_root": "/workspace",
        "workspace_repo_dir": "/workspace/repo",
        "workspace_outputs_dir": "/workspace/outputs",
        "workspace_inputs_dir": "/workspace/inputs",
        "budget_minutes": 30,
    }

    calls: list[dict] = []

    def fake_bg_run(runtime_obj, **kwargs):
        calls.append(kwargs)
        summary = {
            "project_type": "python",
            "dependency_steps": ["pip install -r requirements.txt"],
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
        runtime_obj.write_text("/workspace/outputs/codex_exec.log", "main log")
        runtime_obj.write_text("/workspace/outputs/codex_exec.stream.log", "stream")
        runtime_obj.write_text("/workspace/outputs/patches.diff", "")
        return {"rc": 0, "timed_out": False, "polls": 1, "log_path": "/workspace/outputs/codex_main.log"}

    agent = RunCodexExecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=2, step_total=3)
    monkeypatch.setattr(agent.bg, "run", fake_bg_run)

    agent.execute(ctx)

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert "codex exec" in cmd
    assert "--cd /workspace/repo" in cmd
    assert "--skip-git-repo-check" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "environment probe -> dependency install -> entrypoint discovery -> bounded retries" in cmd
    assert "Maximum 5 execution attempts total across the session." in cmd
    assert "Print only the same JSON object as your final stdout response." in cmd


def test_run_codex_exec_uses_sandbox_task_spec_for_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path, run_id="run_prompt_sandbox")
    artifacts.write_json(
        "execution/task_spec.sandbox.json",
        {
            "tasks": [
                {
                    "task_id": "task_01",
                    "entrypoint": "src/models/tlsh/predict.py",
                    "command": "python3 src/models/tlsh/predict.py",
                    "cwd": ".",
                    "runtime": "python",
                    "timeout_class": "long",
                }
            ],
            "constraints": {"allowed_modification_scope": "repo"},
        },
    )
    artifacts.write_json(
        "task/task_spec.json",
        {
            "tasks": [
                {
                    "task_id": "task_01",
                    "entrypoint": "code/src/models/tlsh/predict.py",
                    "command": "python3 code/src/models/tlsh/predict.py",
                    "cwd": "code",
                    "runtime": "python",
                    "timeout_class": "long",
                }
            ]
        },
    )
    ctx = {
        "workspace_root": "/workspace",
        "workspace_repo_dir": "/workspace/repo",
        "workspace_outputs_dir": "/workspace/outputs",
        "workspace_inputs_dir": "/workspace/inputs",
        "workspace_task_spec_remote": "/workspace/inputs/task_spec.json",
        "workspace_task_spec_local_artifact": "execution/task_spec.sandbox.json",
        "budget_minutes": 30,
    }

    calls: list[dict] = []

    def fake_bg_run(runtime_obj, **kwargs):
        calls.append(kwargs)
        summary = {
            "project_type": "python",
            "dependency_steps": [],
            "commands_run": [],
            "success_basis": "none",
            "execution_succeeded": False,
            "attempt_count": 1,
            "task_results": [],
            "remaining_blockers": [],
        }
        runtime_obj.write_text("/workspace/outputs/execution_summary.json", json.dumps(summary))
        runtime_obj.write_text("/workspace/outputs/codex_main.log", json.dumps(summary))
        runtime_obj.write_text("/workspace/outputs/codex_exec.log", "ok")
        return {"rc": 0, "timed_out": False, "polls": 1, "log_path": "/workspace/outputs/codex_main.log"}

    agent = RunCodexExecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=2, step_total=3)
    monkeypatch.setattr(agent.bg, "run", fake_bg_run)

    agent.execute(ctx)

    assert len(calls) >= 2
    execution_manifest = artifacts.read_json("execution/codex_outputs/run_manifest.json")
    first_run = execution_manifest["runs"][0]
    assert first_run["entrypoint"] == "src/models/tlsh/predict.py"
    assert first_run["command"] == "python3 src/models/tlsh/predict.py"
    assert "code/src/models/tlsh/predict.py" not in json.dumps(execution_manifest)


def test_run_codex_exec_retries_only_on_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    write_task_spec(artifacts)
    ctx = {
        "workspace_root": "/workspace",
        "workspace_repo_dir": "/workspace/repo",
        "workspace_outputs_dir": "/workspace/outputs",
        "workspace_inputs_dir": "/workspace/inputs",
        "budget_minutes": 30,
    }

    calls: list[dict] = []

    def fake_bg_run(runtime_obj, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            runtime_obj.write_text("/workspace/outputs/codex_main.log", "429 rate limit")
            runtime_obj.write_text("/workspace/outputs/codex_exec.log", "429 rate limit")
            return {"rc": 1, "timed_out": False, "polls": 1, "log_path": "/workspace/outputs/codex_main.log"}
        summary = {
            "project_type": "python",
            "dependency_steps": [],
            "commands_run": ["pytest"],
            "success_basis": "test",
            "execution_succeeded": True,
            "attempt_count": 2,
            "task_results": [],
            "remaining_blockers": [],
        }
        runtime_obj.write_text("/workspace/outputs/execution_summary.json", json.dumps(summary))
        runtime_obj.write_text("/workspace/outputs/codex_main.log", json.dumps(summary))
        runtime_obj.write_text("/workspace/outputs/codex_exec.log", "ok")
        return {"rc": 0, "timed_out": False, "polls": 1, "log_path": "/workspace/outputs/codex_main.log"}

    agent = RunCodexExecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=2, step_total=3)
    monkeypatch.setattr(agent.bg, "run", fake_bg_run)
    monkeypatch.setenv("P2C_CODEX_RATE_LIMIT_RETRIES", "1")
    monkeypatch.setenv("P2C_CODEX_RATE_LIMIT_BACKOFF_SEC", "0")

    agent.execute(ctx)

    assert len(calls) == 2


def test_collect_codex_outputs_accepts_summary_and_logs_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    runtime.write_text(
        "/workspace/outputs/execution_summary.json",
        json.dumps(
            {
                "project_type": "python",
                "dependency_steps": ["pip install -r requirements.txt"],
                "commands_run": ["python3 train.py"],
                "success_basis": "run",
                "execution_succeeded": True,
                "attempt_count": 1,
                "task_results": [],
                "remaining_blockers": [],
            }
        ),
    )
    runtime.write_text("/workspace/outputs/codex_main.log", "summary log")
    runtime.write_text("/workspace/outputs/codex_exec.log", "exec log")
    ctx = {"workspace_outputs_dir": "/workspace/outputs"}

    agent = CollectCodexOutputsAgent(llm=DummyLLM(), artifacts=artifacts, step_index=3, step_total=3)
    result = agent.execute(ctx)

    assert result["codex_outputs"]["execution_succeeded"] is True
    assert artifacts.path("execution/codex_outputs/execution_summary.json").exists()


def test_collect_codex_outputs_recovers_summary_from_main_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    summary = {
        "project_type": "python",
        "dependency_steps": [],
        "commands_run": ["pytest"],
        "success_basis": "test",
        "execution_succeeded": False,
        "attempt_count": 3,
        "task_results": [],
        "remaining_blockers": ["missing dataset"],
    }
    runtime.write_text("/workspace/outputs/codex_main.log", "prefix\n" + json.dumps(summary) + "\n")
    runtime.write_text("/workspace/outputs/codex_exec.log", "exec log")
    ctx = {"workspace_outputs_dir": "/workspace/outputs"}

    agent = CollectCodexOutputsAgent(llm=DummyLLM(), artifacts=artifacts, step_index=3, step_total=3)
    result = agent.execute(ctx)

    payload = json.loads(artifacts.path("execution/codex_outputs/execution_summary.json").read_text(encoding="utf-8"))
    assert result["codex_outputs"]["attempt_count"] == 3
    assert payload["remaining_blockers"] == ["missing dataset"]


def test_phase2_smoke_prepare_run_collect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    install_runtime_monkeypatch(monkeypatch, runtime)
    artifacts = make_artifacts(tmp_path)
    repo_dir = make_repo(tmp_path)
    write_task_spec(artifacts)
    ctx = make_ctx(repo_dir)

    prepare_agent = PrepareSandboxAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=3)
    prepare_agent.execute(ctx)

    run_agent = RunCodexExecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=2, step_total=3)

    def fake_bg_run(runtime_obj, **kwargs):
        summary = {
            "project_type": "python",
            "dependency_steps": ["pip install -r requirements.txt"],
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
        return {"rc": 0, "timed_out": False, "polls": 1, "log_path": "/workspace/outputs/codex_main.log"}

    monkeypatch.setattr(run_agent.bg, "run", fake_bg_run)
    run_agent.execute(ctx)

    collect_agent = CollectCodexOutputsAgent(llm=DummyLLM(), artifacts=artifacts, step_index=3, step_total=3)
    collect_agent.execute(ctx)

    assert artifacts.path("execution/codex_outputs/execution_summary.json").exists()
    assert artifacts.path("execution/codex_outputs/codex_main.log").exists()
    assert artifacts.path("execution/codex_outputs/codex_exec.log").exists()


def test_e2b_run_command_prefers_request_timeout_zero() -> None:
    calls: list[dict] = []

    class Commands:
        def run(self, **kwargs):
            calls.append(kwargs)
            return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    class Sandbox:
        commands = Commands()

    runtime = E2BRuntime(timeout_sec=3600)
    runtime._sandbox = Sandbox()
    runtime._sandbox_id = "fake"

    result = runtime.run_command("echo hi", cwd="/workspace", timeout_sec=30)

    assert result.rc == 0
    assert calls[0]["request_timeout"] == 0
    assert calls[0]["timeout"] == 30


def test_e2b_runtime_defaults_to_openai_codex_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("P2C_E2B_TEMPLATE", raising=False)
    runtime = E2BRuntime(timeout_sec=3600)
    assert runtime._template == "openai-codex"


def test_e2b_runtime_auto_builds_missing_template(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"create": 0, "build": 0}

    class FakeSandboxObj:
        sandbox_id = "sbx_123"
        commands = object()
        files = object()

    class FakeSandbox:
        @staticmethod
        def create(**kwargs):
            calls["create"] += 1
            if calls["create"] == 1:
                raise RuntimeError("404: template 'p2c-codex-toolchain' not found")
            return FakeSandboxObj()

    fake_mod = types.SimpleNamespace(Sandbox=FakeSandbox)
    monkeypatch.setitem(sys.modules, "e2b", fake_mod)
    monkeypatch.delenv("P2C_E2B_AUTO_BUILD_TEMPLATE", raising=False)
    monkeypatch.setenv("E2B_API_KEY", "test-key")

    runtime = E2BRuntime(timeout_sec=3600, template="p2c-codex-toolchain")

    def fake_build() -> None:
        calls["build"] += 1
        runtime._template_auto_built = True
        runtime._template_build_attempted = True

    monkeypatch.setattr(runtime, "_maybe_autobuild_template", fake_build)

    runtime.ensure_started()

    assert calls["build"] == 1
    assert calls["create"] == 2
    assert runtime.metadata()["template_auto_built"] is True


def test_template_builder_falls_back_to_alias_build_signature() -> None:
    calls: list[tuple[tuple[object, ...], dict]] = []

    class FakeTemplate:
        @staticmethod
        def build(*args, **kwargs):
            if len(args) == 2:
                raise TypeError("positional signature unsupported")
            calls.append((args, kwargs))
            return {"ok": True}

    template_obj = object()
    result = _call_template_build(
        FakeTemplate,
        template_obj,
        "p2c-codex-toolchain",
        cpu_count=2,
        memory_mb=2048,
        on_build_logs="logger",
    )

    assert result == {"ok": True}
    assert calls == [
        (
            (template_obj,),
            {
                "alias": "p2c-codex-toolchain",
                "cpu_count": 2,
                "memory_mb": 2048,
                "on_build_logs": "logger",
            },
        )
    ]


def test_template_builder_prefers_positional_build_signature() -> None:
    calls: list[tuple[tuple[object, ...], dict]] = []

    class FakeTemplate:
        @staticmethod
        def build(*args, **kwargs):
            calls.append((args, kwargs))
            return {"ok": True}

    template_obj = object()
    result = _call_template_build(
        FakeTemplate,
        template_obj,
        "p2c-codex-toolchain",
        cpu_count=2,
        memory_mb=2048,
        on_build_logs="logger",
    )

    assert result == {"ok": True}
    assert calls == [
        (
            (template_obj, "p2c-codex-toolchain"),
            {"cpu_count": 2, "memory_mb": 2048, "on_build_logs": "logger"},
        )
    ]


def test_template_builder_prefers_camelcase_npm_install_options() -> None:
    calls: list[tuple[tuple[object, ...], dict]] = []

    class FakeBuilder:
        def npmInstall(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    builder = FakeBuilder()
    out = _call_method(builder, ["npmInstall", "npm_install"], ["@openai/codex"], {"g": True})

    assert out is builder
    assert calls == [((["@openai/codex"], {"g": True}), {})]


def test_e2b_runtime_does_not_autobuild_default_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("P2C_E2B_TEMPLATE", raising=False)
    runtime = E2BRuntime(timeout_sec=3600)
    runtime._template_build_attempted = False

    runtime._maybe_autobuild_template()

    assert runtime._template == "openai-codex"
    assert runtime._template_build_attempted is False


def test_run_codex_exec_toolchain_probe_accepts_python_module_pip(tmp_path: Path) -> None:
    class ToolRuntime(FakeRuntime):
        def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
            tool_paths = {
                "python": "/usr/bin/python",
                "python3": "/usr/bin/python3",
                "poetry": "/home/user/.local/bin/poetry",
                "uv": "/home/user/.local/bin/uv",
                "node": "/usr/bin/node",
                "npm": "/usr/bin/npm",
                "codex": "/usr/local/bin/codex",
                "Rscript": "/usr/bin/Rscript",
                "apply_patch": "/workspace/bin/apply_patch",
            }
            if "python3 -m pip --version" in command:
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="pip 24.0 from /home/user/.local/lib\n", stderr="")
            for tool, path in tool_paths.items():
                if f"command -v {tool}" in command:
                    return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout=path + "\n", stderr="")
                if f"{tool} --version" in command or (tool in {"python", "python3"} and f"{tool} -V" in command):
                    return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout=f"{tool} version\n", stderr="")
            return super().run_command(command, cwd, timeout_sec)

    artifacts = make_artifacts(tmp_path, run_id="probe_ok")
    agent = RunCodexExecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=2, step_total=3)
    probe = agent._probe_toolchain(ToolRuntime(), workspace_root="/workspace", workspace_bin_dir="/workspace/bin")

    assert probe["paths"]["pip"] == "python3 -m pip"
    assert "TOOL_MISSING_PIP" not in probe["reason_codes"]
    assert agent._missing_required_tools(probe) == []


def test_run_codex_exec_detects_missing_required_tools(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path, run_id="probe_missing")
    agent = RunCodexExecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=2, step_total=3)
    probe = {
        "paths": {
            "python3": "/usr/bin/python3",
            "pip": "python3 -m pip",
            "poetry": None,
            "uv": None,
            "node": "/usr/bin/node",
            "npm": "/usr/bin/npm",
            "codex": "/usr/local/bin/codex",
            "Rscript": None,
        }
    }

    assert agent._missing_required_tools(probe) == ["poetry", "uv", "Rscript"]
