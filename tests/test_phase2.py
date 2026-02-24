from __future__ import annotations

import os
from pathlib import Path

import pytest

from p2c.agents.base import AgentError
from p2c.agents.codex_prompt_templates import build_codex_main_prompt, build_codex_repair_prompt
from p2c.agents.collect_codex_outputs import CollectCodexOutputsAgent
from p2c.agents.prepare_sandbox import PrepareSandboxAgent
from p2c.agents.run_codex_exec import RunCodexExecAgent
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient
from p2c.runtime.base import RuntimeCommandResult
from p2c.runtime.local_runtime import LocalRuntime


class FakeRuntime:
    backend_name = "e2b"

    def __init__(
        self,
        *,
        outputs_dir: str = "/workspace/outputs",
        help_text: str = "--approval-mode\n--skip-git-repo-check",
        repo_is_git: bool = False,
        fail_main_repo_check_once: bool = False,
        always_fail_main: bool = False,
        all_runs_dependency_failed: bool = False,
        fail_main_with_pip_activity_only: bool = False,
    ) -> None:
        self.outputs_dir = outputs_dir
        self.help_text = help_text
        self.repo_is_git = repo_is_git
        self.fail_main_repo_check_once = fail_main_repo_check_once
        self.always_fail_main = always_fail_main
        self.all_runs_dependency_failed = all_runs_dependency_failed
        self.fail_main_with_pip_activity_only = fail_main_with_pip_activity_only
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
        if "codex exec --help" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout=self.help_text, stderr="")
        if "git rev-parse --is-inside-work-tree" in command:
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0 if self.repo_is_git else 1, stdout="", stderr="")
        if "git init" in command:
            self.repo_is_git = True
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "codex_exec.pid" in command and "codex exec" in command:
            self.main_runs += 1
            self.files[f"{self.outputs_dir}/codex_exec.pid"] = "123\n"
            if self.fail_main_repo_check_once and self.main_runs == 1:
                self.files[f"{self.outputs_dir}/codex_exec.rc"] = "128"
                self.files[f"{self.outputs_dir}/codex_main.log"] = (
                    "Not inside a Git repo and --skip-git-repo-check was not specified.\n"
                )
                self.files[f"{self.outputs_dir}/codex_exec.log"] = (
                    "Not inside a Git repo and --skip-git-repo-check was not specified.\n"
                )
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            if self.always_fail_main:
                self.files[f"{self.outputs_dir}/codex_exec.rc"] = "1"
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
                self.files[f"{self.outputs_dir}/codex_exec.rc"] = "1"
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
                self.files[f"{self.outputs_dir}/codex_exec.rc"] = "0"
                self._write_dependency_failed_outputs()
                return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
            self.files[f"{self.outputs_dir}/codex_exec.rc"] = "0"
            self._write_success_outputs()
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "codex_repair.pid" in command and "codex exec" in command:
            self.files[f"{self.outputs_dir}/codex_repair.pid"] = "125\n"
            self.files[f"{self.outputs_dir}/codex_repair.rc"] = "0"
            return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")
        if "test -f" in command and "codex_exec.rc" in command:
            rc = 0 if f"{self.outputs_dir}/codex_exec.rc" in self.files else 1
            return RuntimeCommandResult(command=command, cwd=cwd, rc=rc, stdout="", stderr="")
        if "test -f" in command and "codex_repair.rc" in command:
            rc = 0 if f"{self.outputs_dir}/codex_repair.rc" in self.files else 1
            return RuntimeCommandResult(command=command, cwd=cwd, rc=rc, stdout="", stderr="")
        return RuntimeCommandResult(command=command, cwd=cwd, rc=0, stdout="", stderr="")

    def read_text(self, remote_path: str) -> str:
        return self.files[remote_path]

    def write_text(self, remote_path: str, content: str) -> None:
        self.files[remote_path] = content

    def close(self) -> None:
        return None

    def metadata(self) -> dict:
        return {"backend": "e2b", "sandbox_id": "fake"}


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
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", help_text="")
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

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


def test_run_codex_exec_does_not_use_git_init_fallback(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", help_text="", repo_is_git=False)
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    assert not any("git init" in c for c in rt.commands)
    assert "/tmp/workspace/outputs/run_manifest.json" in rt.files


def test_run_codex_exec_uses_repair_when_outputs_missing_due_to_repo_check(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(
        outputs_dir="/tmp/workspace/outputs",
        help_text="",
        repo_is_git=False,
        fail_main_repo_check_once=True,
    )
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

    agent = RunCodexExecAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=14)
    agent.run(
        {
            "workspace_root": "/tmp/workspace",
            "workspace_repo_dir": "/tmp/workspace/repo",
            "workspace_inputs_dir": "/tmp/workspace/inputs",
            "workspace_outputs_dir": "/tmp/workspace/outputs",
        }
    )

    # Depending on retry path, repair may run in first attempt.
    assert any("codex_repair.pid" in c for c in rt.commands) or len([c for c in rt.commands if "codex exec" in c]) >= 2
    assert not any("git init" in c for c in rt.commands)


def test_run_codex_exec_does_not_use_help_or_dynamic_add_dir(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", help_text="")
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

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


def test_run_codex_exec_writes_failure_artifact_with_last_command_and_tails(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", always_fail_main=True)
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
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
    assert "ResolutionImpossible" in payload.get("codex_exec_log_tail", "") or "ResolutionImpossible" in payload.get(
        "pip_log_tail", ""
    )


def test_run_codex_exec_collects_pip_log_tail_on_install_failure(tmp_path: Path, monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-key"
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs", always_fail_main=True)
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
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
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)
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
    monkeypatch.setattr("p2c.agents.run_codex_exec.ensure_runtime", lambda _ctx, _art: rt)

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


def test_collect_codex_outputs_requires_workspace_ctx(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime()
    monkeypatch.setattr("p2c.agents.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

    agent = CollectCodexOutputsAgent(llm=LLMClient(), artifacts=artifacts, step_index=9, step_total=14)
    with pytest.raises(AgentError):
        agent.run({"workspace_outputs_dir": "/workspace/outputs"})


def test_collect_codex_outputs_does_not_call_git_commands(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    rt._write_success_outputs()
    rt.files["/tmp/workspace/outputs/codex_exec.log"] = "ok\n"
    monkeypatch.setattr("p2c.agents.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

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
    monkeypatch.setattr("p2c.agents.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

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
    monkeypatch.setattr("p2c.agents.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

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
    assert dep_solver.get("status") == "ok"
    assert "pip install" in pip_log


def test_repo_state_is_gitless_but_valid_schema(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    rt = FakeRuntime(outputs_dir="/tmp/workspace/outputs")
    rt._write_success_outputs()
    monkeypatch.setattr("p2c.agents.collect_codex_outputs.ensure_runtime", lambda _ctx, _art: rt)

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
    monkeypatch.setattr("p2c.agents.prepare_sandbox.ensure_runtime", lambda _ctx, _art: rt)
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
    monkeypatch.setattr("p2c.agents.prepare_sandbox.ensure_runtime", lambda _ctx, _art: rt)
    monkeypatch.setenv("P2C_INCLUDE_GIT", "1")

    agent = PrepareSandboxAgent(llm=LLMClient(), artifacts=artifacts, step_index=7, step_total=14)
    agent.run({"repo_dir": str(repo_dir)})

    assert rt.upload_dir_calls
    assert rt.upload_dir_calls[0]["exclude_globs"] is None
