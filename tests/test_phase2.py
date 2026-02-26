from __future__ import annotations

import os
import re
import json
import sys
import types
from pathlib import Path

import pytest

from p2c.agents.base import AgentError
from p2c.agents.phase2.codex_prompt_templates import build_codex_main_prompt, build_codex_repair_prompt
from p2c.agents.phase2.collect_codex_outputs import CollectCodexOutputsAgent
from p2c.agents.phase2.prepare_sandbox import PrepareSandboxAgent
from p2c.agents.phase2.run_codex_exec import RunCodexExecAgent
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient
from p2c.runtime.base import RuntimeCommandResult
from p2c.runtime.local_runtime import LocalRuntime
from p2c.runtime.e2b_runtime import E2BRuntime


class FakeRuntime:
    backend_name = "e2b"

    def __init__(
        self,
        *,
        outputs_dir: str = "/workspace/outputs",
        repo_is_git: bool = False,
        always_fail_main: bool = False,
        all_runs_dependency_failed: bool = False,
        fail_main_with_pip_activity_only: bool = False,
        python_ok: bool = True,
        pip_available: bool = True,
        ensurepip_available: bool = True,
        required_modules: dict[str, bool] | None = None,
        ensurepip_makes_pip: bool = True,
        apt_makes_pip: bool = True,
        sudo_available: bool = True,
        has_requirements: bool = False,
        requirements_install_success: bool = True,
        compat_install_success: bool | None = None,
        main_rate_limit_failures: int = 0,
        template_name: str = "openai-codex",
    ) -> None:
        self.outputs_dir = outputs_dir
        self.repo_is_git = repo_is_git
        self.always_fail_main = always_fail_main
        self.all_runs_dependency_failed = all_runs_dependency_failed
        self.fail_main_with_pip_activity_only = fail_main_with_pip_activity_only
        self.python_ok = python_ok
        self.pip_available = pip_available
        self.ensurepip_available = ensurepip_available
        self.required_modules = required_modules or {"numpy": True}
        self.ensurepip_makes_pip = ensurepip_makes_pip
        self.apt_makes_pip = apt_makes_pip
        self.sudo_available = sudo_available
        self.has_requirements = has_requirements
        self.requirements_install_success = requirements_install_success
        self.compat_install_success = compat_install_success
        self.main_rate_limit_failures = main_rate_limit_failures
        self.template_name = template_name
        self.main_runs = 0
        self.files: dict[str, str] = {}
        self.commands: list[str] = []

    def ensure_started(self) -> None:
        return None

    def upload_dir(self, local_dir: Path, remote_dir: str, exclude_globs: list[str] | None = None) -> None:
        return None

    def upload_file(self, local_file: Path, remote_path: str) -> None:
        self.files[remote_path] = local_file.read_text(encoding="utf-8", errors="ignore")

    def download_file(self, remote_path: str, local_file: Path) -> None:
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text(self.files[remote_path], encoding="utf-8")

    def _write_success_outputs(self) -> None:
        self.files[f"{self.outputs_dir}/run_manifest.json"] = (
            '{"runs":[{"run_id":"r1","command":"python main.py","params":{},"cwd":"/workspace/repo",'
            '"exit_code":0,"status":"ok","runtime_sec":1.2,"stdout_tail":"ok","stderr_tail":"",'
            '"artifacts":[],"metrics":{"accuracy":0.9},"reason_codes":[]}],"reason_codes":[]}'
        )
        self.files[f"{self.outputs_dir}/claim_alignment.json"] = (
            '{"claims":[{"claim_id":"C1","required_metrics":["accuracy"],"source":["run_manifest"],'
            '"evaluable":"yes","reason":"found metric"}],"reason_codes":[]}'
        )
        self.files[f"{self.outputs_dir}/codex_worklog.jsonl"] = (
            '{"type":"run","ts":"2026-01-01T00:00:00Z","details":"run","result":"ok"}\n'
        )
        self.files[f"{self.outputs_dir}/patches.diff"] = "diff --git a/x b/x\n"
        self.files[f"{self.outputs_dir}/codex_main.log"] = "codex executed\n"
        self.files[f"{self.outputs_dir}/codex_exec.log"] = "codex executed\n"

    def _write_dependency_failed_outputs(self) -> None:
        self.files[f"{self.outputs_dir}/run_manifest.json"] = (
            '{"runs":[{"run_id":"r1","command":"python main.py","params":{},"cwd":"/workspace/repo",'
            '"exit_code":1,"status":"failed_dependency","runtime_sec":1.0,"stdout_tail":"",'
            '"stderr_tail":"No matching distribution found for pkg","artifacts":[],'
            '"metrics":{},"reason_codes":["DEPENDENCY_INSTALL_CONFLICT"]}],"reason_codes":["DEPENDENCY_INSTALL_CONFLICT"]}'
        )
        self.files[f"{self.outputs_dir}/claim_alignment.json"] = (
            '{"claims":[{"claim_id":"C1","required_metrics":["accuracy"],"source":["run_manifest"],'
            '"evaluable":"no","reason":"entrypoint unrunnable due to dependency conflict"}],"reason_codes":["DEPENDENCY_INSTALL_CONFLICT"]}'
        )
        self.files[f"{self.outputs_dir}/codex_worklog.jsonl"] = (
            '{"type":"install","ts":"2026-01-01T00:00:00Z","details":"pip install -r requirements.txt","result":"conflict"}\n'
        )
        self.files[f"{self.outputs_dir}/patches.diff"] = ""
        self.files[f"{self.outputs_dir}/codex_exec.log"] = (
            "pip install -r requirements.txt\n"
            "ERROR: Could not find a version that satisfies the requirement pkg\n"
            "ERROR: No matching distribution found for pkg\n"
        )
        self.files[f"{self.outputs_dir}/pip_install.log"] = (
            "Collecting pkg\n"
            "ERROR: Could not find a version that satisfies the requirement pkg\n"
            "ERROR: No matching distribution found for pkg\n"
        )

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
        self.commands.append(command)
        if "test -n \"$OPENAI_API_KEY\"" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "command -v codex" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "python3 -c \"import sys; print(sys.version)\"" in command:
            if self.python_ok:
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="3.11.9\n", stderr="")
            return RuntimeCommandResult(command=command, cwd=cwd, rc=1, stdout="", stderr="python3 missing")
        if "find_spec" in command and "ensurepip" not in command and "pip" in command:
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=0 if self.python_ok else 1,
                stdout=("1\n" if self.pip_available else "0\n"),
                stderr="",
            )
        if "find_spec" in command and "ensurepip" in command:
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=0 if self.python_ok else 1,
                stdout=("1\n" if self.ensurepip_available else "0\n"),
                stderr="",
            )
        if "find_spec" in command:
            m = re.search(r"find_spec\('([^']+)'\)", command)
            module_name = m.group(1) if m else ""
            if not module_name:
                for key in self.required_modules:
                    if key in command:
                        module_name = key
                        break
            ok = bool(self.required_modules.get(module_name, False))
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=0 if self.python_ok else 1,
                stdout=("1\n" if ok else "0\n"),
                stderr="",
            )
        if "python3 -m ensurepip --upgrade" in command:
            if self.ensurepip_makes_pip and self.ensurepip_available:
                self.pip_available = True
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "sudo -n true" in command:
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=0 if self.sudo_available else 1,
                stdout="",
                stderr="" if self.sudo_available else "sudo: a password is required",
            )
        if "apt-get install -y python3-pip" in command:
            if self.apt_makes_pip and (self.sudo_available or "sudo " not in command):
                self.pip_available = True
            rc = 0 if (self.sudo_available or "sudo " not in command) else 1
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=rc,
                stdout="",
                stderr="" if rc == 0 else "sudo unavailable",
            )
        if "apt-get update" in command:
            rc = 0 if (self.sudo_available or "sudo " not in command) else 1
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=rc,
                stdout="",
                stderr="" if rc == 0 else "sudo unavailable",
            )
        if "test -f requirements.txt" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0 if self.has_requirements else 1, stdout="", stderr="")
        if "python3 -m pip install -r requirements.txt" in command:
            if self.requirements_install_success:
                self.required_modules["numpy"] = True
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            self.files[f"{self.outputs_dir}/pip_install.log"] = "ERROR: ResolutionImpossible\n"
            self.files[f"{self.outputs_dir}/dependency_bootstrap.log"] = "ERROR: ResolutionImpossible\n"
            return RuntimeCommandResult(command=command, cwd=cwd, rc=1, stdout="", stderr="ResolutionImpossible")
        if "python3 -m pip install -r " in command and "requirements.compat.txt" in command:
            compat_ok = self.compat_install_success if self.compat_install_success is not None else self.requirements_install_success
            if compat_ok:
                self.required_modules["numpy"] = True
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            self.files[f"{self.outputs_dir}/pip_install.log"] = "ERROR: compat fallback failed\n"
            return RuntimeCommandResult(command=command, cwd=cwd, rc=1, stdout="", stderr="compat failed")
        if "python3 -m pip install -U pip setuptools wheel" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "codex exec" in command and "echo $! >" in command:
            self.main_runs += 1
            pid_paths = re.findall(r"(/[^'\" ;]*\.pid)", command)
            rc_paths = re.findall(r"(/[^'\" ;]*\.rc)", command)
            log_paths = re.findall(r"(/[^'\" ;]*\.log)", command)
            for p in pid_paths:
                self.files[p] = "123\n"

            if "codex_repair" in command:
                for p in rc_paths:
                    self.files[p] = "0"
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")

            if self.main_rate_limit_failures > 0:
                self.main_rate_limit_failures -= 1
                for p in rc_paths:
                    self.files[p] = "1"
                for lp in log_paths:
                    self.files[lp] = "stream disconnected ... rate limit reached ... retrying\n"
                self.files[f"{self.outputs_dir}/codex_exec.log"] = "stream disconnected ... rate limit reached ... retrying\n"
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            if self.always_fail_main:
                for p in rc_paths:
                    self.files[p] = "1"
                self.files[f"{self.outputs_dir}/codex_main.log"] = (
                    "pip install -r requirements.txt\n"
                    "ERROR: ResolutionImpossible\n"
                )
                self.files[f"{self.outputs_dir}/codex_exec.log"] = (
                    "pip install -r requirements.txt\n"
                    "ERROR: ResolutionImpossible\n"
                )
                self.files[f"{self.outputs_dir}/pip_install.log"] = (
                    "ERROR: ResolutionImpossible\n"
                    "because package A depends on B<1\n"
                )
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            if self.fail_main_with_pip_activity_only:
                for p in rc_paths:
                    self.files[p] = "1"
                self.files[f"{self.outputs_dir}/codex_main.log"] = (
                    "pip install -r requirements.txt\n"
                    "Collecting x\n"
                    "Installing collected packages: x\n"
                )
                self.files[f"{self.outputs_dir}/codex_exec.log"] = (
                    "pip install -r requirements.txt\n"
                    "Collecting x\n"
                    "Installing collected packages: x\n"
                )
                self.files[f"{self.outputs_dir}/pip_install.log"] = (
                    "pip install -r requirements.txt\n"
                    "Collecting x\n"
                    "Installing collected packages: x\n"
                )
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            if self.all_runs_dependency_failed:
                for p in rc_paths:
                    self.files[p] = "0"
                self._write_dependency_failed_outputs()
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            for p in rc_paths:
                self.files[p] = "0"
            self._write_success_outputs()
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "test -f" in command and ".rc" in command:
            rc = 1
            for p in self.files:
                if p.endswith(".rc") and p in command:
                    rc = 0
                    break
            return RuntimeCommandResult(command=command, cwd=cwd, rc=rc, stdout="", stderr="")
        if "python3 " in command and "main_a.py" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=1, stdout="", stderr="ModuleNotFoundError: No module named 'numpy'")
        if "python3 " in command and "main_b.py" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=1, stdout="", stderr="ModuleNotFoundError: No module named 'numpy'")
        return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")

    def read_text(self, remote_path: str) -> str:
        return self.files[remote_path]

    def write_text(self, remote_path: str, content: str) -> None:
        self.files[remote_path] = content

    def close(self) -> None:
        return None

    def metadata(self) -> dict:
        return {"backend": "e2b", "sandbox_id": "fake", "template": self.template_name}


class FakePrepareRuntime:
    backend_name = "e2b"

    def __init__(self) -> None:
        self.upload_dir_calls: list[dict] = []

    def ensure_started(self) -> None:
        return None

    def upload_dir(self, local_dir: Path, remote_dir: str, exclude_globs: list[str] | None = None) -> None:
        self.upload_dir_calls.append(
            {
                "local_dir": str(local_dir),
                "remote_dir": remote_dir,
                "exclude_globs": exclude_globs,
            }
        )

    def upload_file(self, local_file: Path, remote_path: str) -> None:
        return None

    def download_file(self, remote_path: str, local_file: Path) -> None:
        return None

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
        if "python3 - <<'PY'" in command:
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=0,
                stdout="Linux\n6.6\n3.11.9\n8.0\n",
                stderr="",
            )
        return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")

    def read_text(self, remote_path: str) -> str:
        return ""

    def write_text(self, remote_path: str, content: str) -> None:
        return None

    def close(self) -> None:
        return None

    def metadata(self) -> dict:
        return {"backend": "e2b", "sandbox_id": "fake-prepare"}


def _mk_artifacts(tmp_path: Path) -> ArtifactManager:
    manager = ArtifactManager(tmp_path / "artifacts", "run_test")
    manager.ensure_tree()
    return manager


def test_repair_prompt_interpolates_outputs_dir() -> None:
    prompt = build_codex_repair_prompt("/tmp/workspace/outputs")
    assert "{outputs_dir}" not in prompt
    assert "/tmp/workspace/outputs/run_manifest.json" in prompt


def test_codex_prompt_enforces_dependency_solver_contract() -> None:
    prompt = build_codex_main_prompt(
        max_self_heal_iters=2,
        repo_dir="/tmp/workspace/repo",
        inputs_task_spec="/tmp/workspace/inputs/task_spec.json",
        inputs_claims_ir="/tmp/workspace/inputs/claims_ir.json",
        outputs_dir="/tmp/workspace/outputs",
    )
    assert "dependency_solver" in prompt
    assert "pip_install.log" in prompt
    assert "dependency_solver.json" in prompt
    assert "reason_codes" in prompt


def test_run_codex_exec_uses_fixed_codex_flags(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    codex_cmds = [c for c in rt.commands if "codex exec" in c]
    assert codex_cmds
    assert any("--skip-git-repo-check" in c for c in codex_cmds)
    assert any("--dangerously-bypass-approvals-and-sandbox" in c for c in codex_cmds)
    assert any("gpt-5.1-codex-mini" in c for c in codex_cmds)
    assert "/tmp/workspace/outputs/run_manifest.json" in rt.files


def test_run_codex_exec_does_not_use_help_or_dynamic_add_dir(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
            "workspace_data_dir": "/tmp/workspace/data",
        }
    )

    codex_cmds = [c for c in rt.commands if "codex exec" in c]
    assert codex_cmds
    assert not any("codex exec --help" in c for c in rt.commands)
    assert not any("--add-dir" in c for c in codex_cmds)
    assert not any("--sandbox" in c for c in codex_cmds)
    assert any("codex_main.log" in c for c in rt.commands if "tee -a" in c)


def test_run_codex_exec_recovers_from_launcher_deadline_exceeded(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)

    class LaunchDeadlineRuntime(FakeRuntime):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self._raised_once = False

        def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
            if "codex exec" in command and "echo $! >" in command and not self._raised_once:
                self._raised_once = True
                pid_paths = re.findall(r"(/[^'\" ;]*\\.pid)", command)
                rc_paths = re.findall(r"(/[^'\" ;]*\\.rc)", command)
                for p in pid_paths:
                    self.files[p] = "123\n"
                for p in rc_paths:
                    self.files[p] = "0"
                self._write_success_outputs()
                raise RuntimeError("context deadline exceeded")
            return super().run_command(command, cwd, timeout_sec)

    rt = LaunchDeadlineRuntime(outputs_dir="/tmp/workspace/outputs")
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )
    manifest = json.loads(rt.read_text("/tmp/workspace/outputs/run_manifest.json"))
    assert manifest.get("runs")


def test_capability_gate_detects_missing_pip_and_ensurepip(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        required_modules={"numpy": False},
        has_requirements=True,
    )
    rt.files["/tmp/workspace/inputs/task_spec.json"] = '{"entrypoints":[{"path":"main_a.py","command":"python3 main_a.py"}]}'
    rt.files["/tmp/workspace/inputs/claims_ir.json"] = '{"claims":[{"claim_id":"c1","metric":"accuracy"}]}'
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_APT_ENABLE", "0")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    failure = artifacts.read_json("execution/codex_failure.json")
    snapshot = failure.get("capability_snapshot", {})
    assert snapshot.get("pip_available") is False
    assert snapshot.get("ensurepip_available") is False


def test_dependency_bootstrap_ensurepip_success_path(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=True,
        ensurepip_makes_pip=True,
        has_requirements=False,
        required_modules={"numpy": True},
    )
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )
    assert any("python3 -m ensurepip --upgrade" in c for c in rt.commands)


def test_dependency_bootstrap_apt_fallback_path(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        apt_makes_pip=True,
        has_requirements=False,
        required_modules={"numpy": True},
    )
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_APT_ENABLE", "1")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )
    assert any("apt-get install -y python3-pip" in c for c in rt.commands)


def test_e2b_runtime_uses_template_env_override(monkeypatch) -> None:
    calls: list[dict] = []

    class _DummyCommands:
        @staticmethod
        def run(**kwargs):
            return types.SimpleNamespace(exit_code=0, stdout="", stderr="")

    class _DummyFiles:
        store: dict[str, str] = {}

        @classmethod
        def read(cls, path: str) -> str:
            return cls.store.get(path, "")

        @classmethod
        def write(cls, path: str, content: str) -> None:
            cls.store[path] = content

    class _DummySandbox:
        sandbox_id = "sbx_test"
        commands = _DummyCommands()
        files = _DummyFiles()

        @classmethod
        def create(cls, **kwargs):
            calls.append(kwargs)
            return cls()

        def close(self) -> None:
            return None

    fake_e2b = types.ModuleType("e2b")
    fake_e2b.Sandbox = _DummySandbox
    monkeypatch.setitem(sys.modules, "e2b", fake_e2b)
    monkeypatch.setenv("E2B_API_KEY", "test-e2b-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("P2C_E2B_TEMPLATE", "custom-codex-template")

    rt = E2BRuntime(timeout_sec=1800)
    rt.ensure_started()
    meta = rt.metadata()

    assert calls
    assert calls[0].get("template") == "custom-codex-template"
    assert meta.get("template") == "custom-codex-template"
    rt.close()


def test_dependency_bootstrap_uses_sudo_when_available(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        sudo_available=True,
        apt_makes_pip=True,
        has_requirements=False,
        required_modules={"numpy": True},
    )
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_APT_ENABLE", "1")
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_RUNTIME_SUDO_ENABLE", "1")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )
    assert any("sudo -n true" in c for c in rt.commands)
    assert any("sudo apt-get install -y python3-pip" in c for c in rt.commands)


def test_dependency_bootstrap_marks_sudo_unavailable(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        sudo_available=False,
        apt_makes_pip=False,
        has_requirements=True,
        required_modules={"numpy": False},
    )
    rt.files["/tmp/workspace/inputs/task_spec.json"] = '{"entrypoints":[{"path":"main_a.py","command":"python3 main_a.py"}]}'
    rt.files["/tmp/workspace/inputs/claims_ir.json"] = '{"claims":[{"claim_id":"c1","metric":"accuracy"}]}'
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_APT_ENABLE", "1")
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_RUNTIME_SUDO_ENABLE", "1")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    failure = artifacts.read_json("execution/codex_failure.json")
    assert "DEP_BOOTSTRAP_SUDO_UNAVAILABLE" in failure.get("reason_codes", [])


def test_dependency_compat_fallback_generates_compat_requirements(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=True,
        ensurepip_available=True,
        has_requirements=True,
        requirements_install_success=False,
        compat_install_success=True,
        required_modules={"numpy": True},
    )
    rt.files["/tmp/workspace/repo/requirements.txt"] = (
        "tensorflow==1.15.4\nnumpy==1.13.3\nscikit_learn==0.19.1\nmatplotlib==2.1.0\n"
    )
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_COMPAT_MODE", "1")
    monkeypatch.setenv("P2C_DEP_COMPAT_PROFILE", "tf1_legacy")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    compat_text = rt.read_text("/tmp/workspace/outputs/requirements.compat.txt")
    solver = json.loads(rt.read_text("/tmp/workspace/outputs/dependency_solver.json"))
    assert "tensorflow==2.15.1" in compat_text
    assert "numpy==1.26.4" in compat_text
    assert solver.get("compat_replacements")
    assert solver.get("status") == "ready_with_compat_fallback"


def test_dependency_compat_fallback_reason_codes(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=True,
        ensurepip_available=True,
        has_requirements=True,
        requirements_install_success=False,
        compat_install_success=False,
        required_modules={"numpy": False},
    )
    rt.files["/tmp/workspace/repo/requirements.txt"] = "tensorflow_gpu==1.15.4\nnumpy==1.13.3\n"
    rt.files["/tmp/workspace/inputs/task_spec.json"] = '{"entrypoints":[{"path":"main_a.py","command":"python3 main_a.py"}]}'
    rt.files["/tmp/workspace/inputs/claims_ir.json"] = '{"claims":[{"claim_id":"c1","metric":"accuracy"}]}'
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_COMPAT_MODE", "1")
    monkeypatch.setenv("P2C_DEP_COMPAT_PROFILE", "tf1_legacy")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    failure = artifacts.read_json("execution/codex_failure.json")
    reasons = failure.get("reason_codes", [])
    assert "DEPENDENCY_COMPAT_FALLBACK_USED" in reasons
    assert "DEPENDENCY_COMPAT_FALLBACK_FAILED" in reasons


def test_reason_codes_are_deduplicated(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        sudo_available=False,
        apt_makes_pip=False,
        has_requirements=True,
        required_modules={"numpy": False},
    )
    rt.files["/tmp/workspace/inputs/task_spec.json"] = '{"entrypoints":[{"path":"main_a.py","command":"python3 main_a.py"}]}'
    rt.files["/tmp/workspace/inputs/claims_ir.json"] = '{"claims":[{"claim_id":"c1","metric":"accuracy"}]}'
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    reasons = artifacts.read_json("execution/codex_failure.json").get("reason_codes", [])
    assert len(reasons) == len(set(reasons))


def test_gate_fail_runs_each_entrypoint_once_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        required_modules={"numpy": False},
        has_requirements=True,
        apt_makes_pip=False,
    )
    rt.files["/tmp/workspace/inputs/task_spec.json"] = (
        '{"entrypoints":[{"path":"main_a.py","command":"python3 main_a.py"},'
        '{"path":"main_b.py","command":"python3 main_b.py"}]}'
    )
    rt.files["/tmp/workspace/inputs/claims_ir.json"] = '{"claims":[{"claim_id":"c1","metric":"accuracy"}]}'
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_APT_ENABLE", "0")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    manifest = rt.read_text("/tmp/workspace/outputs/run_manifest.json")
    assert '"run_id": "main_a.py"' in manifest
    assert '"run_id": "main_b.py"' in manifest


def test_gate_fail_skips_codex_main_command(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        required_modules={"numpy": False},
        has_requirements=True,
        apt_makes_pip=False,
    )
    rt.files["/tmp/workspace/inputs/task_spec.json"] = '{"entrypoints":[{"path":"main_a.py","command":"python3 main_a.py"}]}'
    rt.files["/tmp/workspace/inputs/claims_ir.json"] = '{"claims":[{"claim_id":"c1","metric":"accuracy"}]}'
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_APT_ENABLE", "0")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    assert not any("codex exec" in c for c in rt.commands)


def test_run_codex_exec_writes_failure_artifact_with_last_command_and_tails(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", always_fail_main=True)
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_ALLOW_GIT_INIT_FALLBACK", "1")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    payload = artifacts.read_json("execution/codex_failure.json")
    assert payload.get("stage") in {"main", "repair", "postcheck"}
    assert "codex exec" in payload.get("last_command", "")
    assert payload.get("exit_code") in {1, 124}
    assert isinstance(payload.get("capability_snapshot", {}), dict)
    assert isinstance(payload.get("dependency_bootstrap_trace", []), list)
    assert "ResolutionImpossible" in payload.get("codex_exec_log_tail", "") or "ResolutionImpossible" in payload.get(
        "pip_log_tail", ""
    )


def test_run_codex_exec_collects_pip_log_tail_on_install_failure(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", always_fail_main=True)
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_ALLOW_GIT_INIT_FALLBACK", "1")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    payload = artifacts.read_json("execution/codex_failure.json")
    assert "ResolutionImpossible" in payload.get("pip_log_tail", "")


def test_run_codex_exec_no_conflict_code_for_pip_activity_only(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", fail_main_with_pip_activity_only=True)
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_ALLOW_GIT_INIT_FALLBACK", "1")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    payload = artifacts.read_json("execution/codex_failure.json")
    reasons = payload.get("reason_codes", [])
    assert "DEPENDENCY_INSTALL_CONFLICT" not in reasons
    assert "DEPENDENCY_INSTALL_ACTIVITY_DETECTED" in reasons


def test_run_codex_exec_fails_when_all_entrypoints_unrunnable(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", all_runs_dependency_failed=True)
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    payload = artifacts.read_json("execution/codex_failure.json")
    assert "DEPENDENCY_UNRESOLVED" in payload.get("reason_codes", [])


def test_rate_limit_backoff_retries_main_then_fails(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", main_rate_limit_failures=20, has_requirements=False)
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_RATE_LIMIT_RETRIES", "2")
    monkeypatch.setenv("P2C_RATE_LIMIT_BACKOFF_SEC", "0")
    monkeypatch.setenv("P2C_RATE_LIMIT_BACKOFF_MULTIPLIER", "1")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    payload = artifacts.read_json("execution/codex_failure.json")
    reasons = payload.get("reason_codes", [])
    assert "CODEX_RATE_LIMIT_BACKOFF_EXHAUSTED" in reasons
    assert any("CODEX_RATE_LIMIT_BACKOFF_RETRY_1" == x for x in reasons)


def test_prompt_forbids_full_input_dump() -> None:
    prompt = build_codex_main_prompt(
        max_self_heal_iters=2,
        repo_dir="/tmp/workspace/repo",
        inputs_task_spec="/tmp/workspace/inputs/task_spec.json",
        inputs_claims_ir="/tmp/workspace/inputs/claims_ir.json",
        outputs_dir="/tmp/workspace/outputs",
    )
    assert "Do NOT print or dump full contents" in prompt
    assert "Only output compact summaries" in prompt or "only output compact summaries" in prompt


def test_fallback_outputs_contract_valid_when_dependency_unresolved(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        pip_available=False,
        ensurepip_available=False,
        required_modules={"numpy": False},
        has_requirements=True,
        apt_makes_pip=False,
    )
    rt.files["/tmp/workspace/inputs/task_spec.json"] = '{"entrypoints":[{"path":"main_a.py","command":"python3 main_a.py"}]}'
    rt.files["/tmp/workspace/inputs/claims_ir.json"] = '{"claims":[{"claim_id":"c1","metric":"accuracy"}]}'
    monkeypatch.setattr("p2c.agents.phase2.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_DEP_BOOTSTRAP_APT_ENABLE", "0")

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    with pytest.raises(AgentError):
        agent.run(
            {
                "workspace_root": "/tmp/workspace",
                "workspace_repo_dir": "/tmp/workspace/repo",
                "workspace_inputs_dir": "/tmp/workspace/inputs",
                "workspace_outputs_dir": "/tmp/workspace/outputs",
            }
        )

    run_manifest = artifacts.read_json("execution/codex_failure.json")
    assert "DEPENDENCY_UNRESOLVED" in run_manifest.get("reason_codes", [])
    assert json.loads(rt.read_text("/tmp/workspace/outputs/run_manifest.json")).get("runs")
    assert json.loads(rt.read_text("/tmp/workspace/outputs/claim_alignment.json")).get("claims")


def test_collect_codex_outputs_requires_workspace_ctx(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime()
    monkeypatch.setattr("p2c.agents.phase2.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

    agent = CollectCodexOutputsAgent(llm=LLMClient(), artifacts=artifacts, step_index=9, step_total=14)
    with pytest.raises(AgentError):
        agent.run({"workspace_outputs_dir": "/workspace/outputs"})


def test_collect_codex_outputs_does_not_call_git_commands(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    rt._write_success_outputs()
    rt.files["/tmp/workspace/outputs/codex_exec.log"] = "ok\n"
    monkeypatch.setattr("p2c.agents.phase2.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

    agent = CollectCodexOutputsAgent(llm=LLMClient(), artifacts=artifacts, step_index=9, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    assert not any("git " in c for c in rt.commands)


def test_collect_codex_outputs_writes_empty_patch_diff_when_missing(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    rt._write_success_outputs()
    rt.files.pop("/tmp/workspace/outputs/patches.diff", None)
    monkeypatch.setattr("p2c.agents.phase2.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

    agent = CollectCodexOutputsAgent(llm=LLMClient(), artifacts=artifacts, step_index=9, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    patch_text = artifacts.path("execution/codex_outputs/patches.diff").read_text(encoding="utf-8")
    assert patch_text == ""


def test_collect_codex_outputs_downloads_optional_dependency_logs_without_blocking(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    rt._write_success_outputs()
    rt.files["/tmp/workspace/outputs/dependency_solver.json"] = '{"steps":[{"name":"pip"}],"status":"ok","reason_codes":[]}'
    rt.files["/tmp/workspace/outputs/pip_install.log"] = "pip install -r requirements.txt\nok\n"
    rt.files["/tmp/workspace/outputs/capability_probe.json"] = '{"python_ok":true,"pip_available":true}'
    rt.files["/tmp/workspace/outputs/dependency_bootstrap.log"] = "bootstrap ok\n"
    monkeypatch.setattr("p2c.agents.phase2.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

    agent = CollectCodexOutputsAgent(llm=LLMClient(), artifacts=artifacts, step_index=9, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    dep_solver = artifacts.read_json("execution/codex_outputs/dependency_solver.json")
    pip_log = artifacts.path("execution/codex_outputs/pip_install.log").read_text(encoding="utf-8")
    cap_probe = artifacts.read_json("execution/codex_outputs/capability_probe.json")
    dep_bootstrap = artifacts.path("execution/codex_outputs/dependency_bootstrap.log").read_text(encoding="utf-8")
    assert dep_solver.get("status") == "ok"
    assert "pip install" in pip_log
    assert cap_probe.get("python_ok") is True
    assert "bootstrap ok" in dep_bootstrap


def test_repo_state_is_gitless_but_valid_schema(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    rt._write_success_outputs()
    monkeypatch.setattr("p2c.agents.phase2.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

    agent = CollectCodexOutputsAgent(llm=LLMClient(), artifacts=artifacts, step_index=9, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    repo_state = artifacts.read_json("execution/repo_state.json")
    assert repo_state.get("head") is None
    assert repo_state.get("branch") is None
    assert "NO_GIT_METADATA" in repo_state.get("reason_codes", [])


def test_upload_dir_excludes_git_by_default(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / ".git").mkdir(parents=True)
    (src / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (src / "main.py").write_text("print('ok')\n", encoding="utf-8")

    rt = LocalRuntime()
    rt.ensure_started()
    dst = tmp_path / "dst"
    rt.upload_dir(src, str(dst), exclude_globs=[".git", ".git/**"])

    assert (dst / "main.py").exists()
    assert not (dst / ".git").exists()


def test_upload_dir_can_include_git_with_flag(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / ".git").mkdir(parents=True)
    (src / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (src / "main.py").write_text("print('ok')\n", encoding="utf-8")

    rt = LocalRuntime()
    rt.ensure_started()
    dst = tmp_path / "dst"
    rt.upload_dir(src, str(dst), exclude_globs=None)

    assert (dst / "main.py").exists()
    assert (dst / ".git" / "config").exists()


def test_prepare_sandbox_excludes_git_by_default(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json("task/task_spec.json", {"entrypoints": []})
    artifacts.write_json("fingerprint/claims_ir.json", {"claims": []})
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")

    rt = FakePrepareRuntime()
    monkeypatch.setattr("p2c.agents.phase2.prepare_sandbox.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.delenv("P2C_INCLUDE_GIT", raising=False)

    agent = PrepareSandboxAgent(llm=LLMClient(), artifacts=artifacts, step_index=7, step_total=14)
    agent.run({"repo_dir": str(repo_dir)})

    assert rt.upload_dir_calls
    assert rt.upload_dir_calls[0]["exclude_globs"] == [".git", ".git/**"]
    run_log = artifacts.path("execution/run.log").read_text(encoding="utf-8")
    assert "repo upload mode=local_dir include_git=0" in run_log


def test_prepare_sandbox_can_include_git_with_flag(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json("task/task_spec.json", {"entrypoints": []})
    artifacts.write_json("fingerprint/claims_ir.json", {"claims": []})
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")

    rt = FakePrepareRuntime()
    monkeypatch.setattr("p2c.agents.phase2.prepare_sandbox.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_INCLUDE_GIT", "1")

    agent = PrepareSandboxAgent(llm=LLMClient(), artifacts=artifacts, step_index=7, step_total=14)
    agent.run({"repo_dir": str(repo_dir)})

    assert rt.upload_dir_calls
    assert rt.upload_dir_calls[0]["exclude_globs"] is None
