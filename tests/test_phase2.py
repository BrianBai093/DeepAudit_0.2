from __future__ import annotations

import os
from pathlib import Path

from p2c.agents.execute_and_heal import ExecuteAndHealAgent
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient


def _mk_artifacts(tmp_path: Path) -> ArtifactManager:
    manager = ArtifactManager(tmp_path / "artifacts", "run_test")
    manager.ensure_tree()
    return manager


def test_commands_jsonl_written(tmp_path: Path) -> None:
    os.environ["P2C_SKIP_MINI_SWE"] = "1"
    os.environ["P2C_RUNTIME_BACKEND"] = "local"
    artifacts = _mk_artifacts(tmp_path)
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    artifacts.write_json(
        "task/task_spec.json",
        {
            "goal": ["C1"],
            "constraints": {},
            "entrypoints": [
                {
                    "path": "inline",
                    "command": "python3 -c \"print('ok accuracy 0.90')\"",
                    "confidence": 1.0,
                    "evidence": "test",
                }
            ],
            "metric_observers": [{"name": "acc", "kind": "stdout_regex", "pattern": "accuracy"}],
            "run_matrix": [{"seed": 0, "timeout_sec": 30, "budget_minutes": 1}],
            "reason_codes": [],
        },
    )

    agent = ExecuteAndHealAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=12)
    agent.run({"repo_dir": str(repo_dir)})

    commands_path = artifacts.path("execution/commands.jsonl")
    assert commands_path.exists()
    assert commands_path.read_text(encoding="utf-8").strip() != ""


def test_patch_diff_exists_even_if_empty(tmp_path: Path) -> None:
    os.environ["P2C_SKIP_MINI_SWE"] = "1"
    os.environ["P2C_RUNTIME_BACKEND"] = "local"
    artifacts = _mk_artifacts(tmp_path)
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    artifacts.write_json(
        "task/task_spec.json",
        {
            "goal": [],
            "constraints": {},
            "entrypoints": [],
            "metric_observers": [],
            "run_matrix": [{"seed": 0, "timeout_sec": 5, "budget_minutes": 1}],
            "reason_codes": [],
        },
    )

    agent = ExecuteAndHealAgent(llm=LLMClient(), artifacts=artifacts, step_index=8, step_total=12)
    agent.run({"repo_dir": str(repo_dir)})

    patch_path = artifacts.path("execution/patch.diff")
    assert patch_path.exists()
