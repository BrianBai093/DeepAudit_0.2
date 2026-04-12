from __future__ import annotations

from pathlib import Path

from p2c.agents.phase2.codex_executor import ClaudeResult, CodexExecutorAgent
from p2c.io_artifacts import ArtifactManager
from p2c.schemas import ExecutionPlan, ExecutionStep


class DummyEnvMgr:
    env_name = "test_env"


def test_executor_prompts_claude_with_absolute_outputs_dir(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    artifacts = ArtifactManager("artifacts", "run_abs_outputs")
    artifacts.ensure_tree()
    expected_outputs_dir = str(artifacts.path("execution/codex_outputs").resolve())
    seen: dict[str, str] = {}

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        seen["prompt"] = prompt
        return ClaudeResult(stdout="METRIC:accuracy=0.91\n", stderr="", returncode=0)

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute(
        {
            "repo_dir": str(repo_dir),
            "_p2_plan": ExecutionPlan(
                plan_id="p",
                env_name="test_env",
                execution_steps=[
                    ExecutionStep(
                        step_id="train",
                        description="run train",
                        command="python train.py",
                        expected_metrics=["accuracy"],
                    )
                ],
            ),
            "_p2_env_mgr": DummyEnvMgr(),
            "_p2_remaining_sec": 60,
        }
    )

    assert result["success"] is True
    assert expected_outputs_dir in seen["prompt"]
    assert "Target/code/artifacts" not in seen["prompt"]
