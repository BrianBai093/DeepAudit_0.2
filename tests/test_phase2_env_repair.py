from __future__ import annotations

from pathlib import Path
import subprocess

from p2c.agents.phase2.code_compat_agent import CodeCompatAgent
from p2c.agents.phase2.env_repair_agent import EnvRepairAgent
from p2c.agents.phase2.orchestrator import Phase2Orchestrator
from p2c.io_artifacts import ArtifactManager
from p2c.schemas import CodeCompatResult, EnvRepairResult, EnvSetupResult, ExecutorEnvSpec, RunManifestDoc


class DummyEnvMgr:
    env_name = "repair_env"
    backend = "venv"

    @staticmethod
    def env_path_actual() -> str:
        return "/tmp/p2c_venv_repair_env"


def make_artifacts(tmp_path: Path, run_id: str) -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path / "artifacts", run_id)
    artifacts.ensure_tree()
    return artifacts


class NativeFailToolAgent:
    cleanup_called = False
    run_called = False

    def build_env_spec(self, ctx):
        return ExecutorEnvSpec(
            env_name="native_fail_executor",
            python_version="3.8",
            native_environment_file="environment.yml",
        )

    def run(self, ctx):
        self.run_called = True
        return {
            "env_result": EnvSetupResult(
                env_name="native_fail_executor",
                python_version="3.8",
                install_commands=["conda env create -f environment.yml (ok=False)"],
                reason_codes=["NATIVE_CONDA_ENV_CREATE_FAILED"],
            )
        }

    def cleanup(self):
        self.cleanup_called = True


class DerivedFailToolAgent(NativeFailToolAgent):
    def build_env_spec(self, ctx):
        return ExecutorEnvSpec(env_name="derived_fail_executor", python_version="3.8")

    def run(self, ctx):
        self.run_called = True
        return {
            "env_result": EnvSetupResult(
                env_name="derived_fail_executor",
                python_version="3.8",
                install_commands=["create env (ok=False)"],
                reason_codes=["ENV_CREATE_FAILED"],
            )
        }


class SuccessfulRepairAgent:
    cleanup_called = False
    called = False
    env_manager = DummyEnvMgr()

    def run(self, ctx):
        self.called = True
        return {
            "env_repair_result": EnvRepairResult(
                status="success",
                selected_strategy="cpu_relaxed_py310",
                env_name="native_fail_executor",
                python_version="3.10",
                backend="venv",
                env_path="/tmp/p2c_venv_repair_env",
                validation_passed=True,
                reason_codes=["ENV_REPAIR_APPLIED"],
            ),
            "env_manager": self.env_manager,
        }

    def cleanup(self):
        self.cleanup_called = True


class SuccessfulCompatAgent:
    called = False

    def run(self, ctx):
        self.called = True
        return {
            "code_compat_result": CodeCompatResult(
                status="success",
                validation_passed=True,
                reason_codes=["CODE_COMPAT_NO_PATCH_NEEDED"],
            )
        }


class FailingCompatAgent(SuccessfulCompatAgent):
    def run(self, ctx):
        self.called = True
        return {
            "code_compat_result": CodeCompatResult(
                status="failed",
                validation_passed=False,
                reason_codes=["CODE_COMPAT_FAILED"],
            )
        }


class RecordingExecutorAgent:
    called = False
    env_mgr = None

    def run(self, ctx):
        self.called = True
        self.env_mgr = ctx.get("_p2_env_mgr")
        return {"success": True, "run_manifest": RunManifestDoc(runs=[], reason_codes=["EXECUTOR_AGENT_RUN"])}


def make_orchestrator(tmp_path: Path, tool_agent, repair_agent=None, compat_agent=None, executor_agent=None):
    artifacts = make_artifacts(tmp_path, "run")
    return Phase2Orchestrator(
        tool_agent=tool_agent,
        env_repair_agent=repair_agent,
        code_compat_agent=compat_agent,
        executor_agent=executor_agent or RecordingExecutorAgent(),
        llm=None,
        artifacts=artifacts,
        step_index=1,
        step_total=1,
    ), artifacts


def test_native_env_failure_enters_repair_branch_before_executor(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "environment.yml").write_text("name: old\ndependencies:\n  - python=3.8\n", encoding="utf-8")
    tool = NativeFailToolAgent()
    repair = SuccessfulRepairAgent()
    compat = SuccessfulCompatAgent()
    executor = RecordingExecutorAgent()
    orchestrator, _ = make_orchestrator(tmp_path, tool, repair, compat, executor)

    result = orchestrator.execute({"repo_dir": str(repo_dir), "run_id": "run", "budget_minutes": 5})

    assert result["status"] == "success"
    assert tool.run_called is True
    assert repair.called is True
    assert compat.called is True
    assert executor.called is True
    assert executor.env_mgr is repair.env_manager


def test_non_native_env_failure_does_not_enter_repair_branch(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    tool = DerivedFailToolAgent()
    repair = SuccessfulRepairAgent()
    compat = SuccessfulCompatAgent()
    executor = RecordingExecutorAgent()
    orchestrator, _ = make_orchestrator(tmp_path, tool, repair, compat, executor)

    result = orchestrator.execute({"repo_dir": str(repo_dir), "run_id": "run", "budget_minutes": 5})

    assert result["status"] == "failed"
    assert repair.called is False
    assert compat.called is False
    assert executor.called is False


def test_force_env_repair_skips_tool_agent_install(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "environment.yml").write_text("name: old\ndependencies:\n  - python=3.8\n", encoding="utf-8")
    tool = NativeFailToolAgent()
    repair = SuccessfulRepairAgent()
    compat = SuccessfulCompatAgent()
    executor = RecordingExecutorAgent()
    orchestrator, _ = make_orchestrator(tmp_path, tool, repair, compat, executor)

    result = orchestrator.execute(
        {
            "repo_dir": str(repo_dir),
            "run_id": "run",
            "budget_minutes": 5,
            "phase2_force_env_repair": True,
        }
    )

    assert result["status"] == "success"
    assert tool.run_called is False
    assert repair.called is True
    assert executor.called is True


def test_force_env_repair_without_native_env_file_fails_cleanly(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    tool = DerivedFailToolAgent()
    executor = RecordingExecutorAgent()
    artifacts = make_artifacts(tmp_path, "run")
    repair = EnvRepairAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    compat = SuccessfulCompatAgent()
    orchestrator = Phase2Orchestrator(
        tool_agent=tool,
        env_repair_agent=repair,
        code_compat_agent=compat,
        executor_agent=executor,
        llm=None,
        artifacts=artifacts,
        step_index=1,
        step_total=1,
    )

    result = orchestrator.execute(
        {
            "repo_dir": str(repo_dir),
            "run_id": "run",
            "budget_minutes": 5,
            "phase2_force_env_repair": True,
        }
    )

    assert result["status"] == "failed"
    assert "ENV_REPAIR_NO_NATIVE_ENV_FILE" in result["failures"][0]["reason_codes"]
    assert executor.called is False


def test_code_compat_patch_modifies_repo_and_writes_diff(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    target = repo_dir / "legacy.py"
    target.write_text("VALUE = 'old'\n", encoding="utf-8")
    artifacts = make_artifacts(tmp_path, "compat")
    artifacts.write_json(
        "execution/env_repair/repaired_environment_spec.json",
        {"env_name": "repair_env", "python_version": "3.10", "reason_codes": []},
    )

    class FailingThenPassingEnv:
        calls = 0

        def run_in_env(self, command, cwd=".", timeout_sec=120):
            self.calls += 1
            if self.calls == 1:
                return subprocess.CompletedProcess(command, 1, "", "ImportError: np.float is missing")
            return subprocess.CompletedProcess(command, 0, "ok", "")

    patch_text = """--- legacy.py
+++ legacy.py
@@ -1 +1 @@
-VALUE = 'old'
+VALUE = 'new'
"""

    def fake_request_patch(self, **kwargs):
        return patch_text, "updated legacy API", ["CODE_COMPAT_LLM_PATCH"]

    monkeypatch.setattr(CodeCompatAgent, "_request_patch", fake_request_patch)
    agent = CodeCompatAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    result = agent.execute(
        {
            "repo_dir": str(repo_dir),
            "_p2_env_mgr": FailingThenPassingEnv(),
            "_p2_env_repair_result": EnvRepairResult(status="success", validation_passed=True),
        }
    )

    assert result["code_compat_result"].status == "success"
    assert target.read_text(encoding="utf-8") == "VALUE = 'new'\n"
    assert result["code_compat_result"].changed_files == ["legacy.py"]
    assert artifacts.path("execution/code_compat/code_compat_patch.diff").is_file()


def test_code_compat_failure_blocks_executor(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "environment.yml").write_text("name: old\ndependencies:\n  - python=3.8\n", encoding="utf-8")
    tool = NativeFailToolAgent()
    repair = SuccessfulRepairAgent()
    compat = FailingCompatAgent()
    executor = RecordingExecutorAgent()
    orchestrator, _ = make_orchestrator(tmp_path, tool, repair, compat, executor)

    result = orchestrator.execute({"repo_dir": str(repo_dir), "run_id": "run", "budget_minutes": 5})

    assert result["status"] == "failed"
    assert compat.called is True
    assert executor.called is False
    assert "CODE_COMPAT_FAILED" in result["failures"][-1]["reason_codes"]
