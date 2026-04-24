from __future__ import annotations

import json
from pathlib import Path

from p2c.agents.phase2.executor_agent import ExecutorAgent, ExecutorSessionResult
from p2c.io_artifacts import ArtifactManager


class DummyEnvMgr:
    env_name = "test_env"


def test_executor_prompt_uses_absolute_executor_outputs_dir(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("python train.py\n", encoding="utf-8")
    (repo_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")

    artifacts = ArtifactManager(tmp_path / "artifacts", "run_abs_outputs")
    artifacts.ensure_tree()
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "main run",
                    "description": "primary experiment",
                    "dataset": "MNIST",
                    "table_anchor": "Table 1",
                    "primary_metrics": ["accuracy"],
                    "is_primary": True,
                    "notes": None,
                }
            ],
            "claims": [],
            "reason_codes": [],
        },
    )
    artifacts.write_json("task/repo_analysis.json", {"dependency_profiles": [], "entrypoint_candidates": [], "reason_codes": []})
    artifacts.write_json("task/metric_contract.json", {"required_metrics": ["accuracy"], "parsers": [], "normalization": {}, "reason_codes": []})

    expected_outputs_dir = str(artifacts.path("execution/executor_outputs").resolve())
    seen: dict[str, str] = {}

    def fake_session(env_mgr, prompt, cwd, timeout_sec=600):
        seen["prompt"] = prompt
        outputs_dir = artifacts.path("execution/executor_outputs")
        stdout_log = "execution/executor_outputs/experiment_exp_01_stdout.log"
        stderr_log = "execution/executor_outputs/experiment_exp_01_stderr.log"
        narrative_log = "execution/executor_outputs/experiment_exp_01_narrative.log"
        artifacts.write_text(stdout_log, "METRIC:accuracy=0.91\n")
        artifacts.write_text(stderr_log, "")
        artifacts.write_text(narrative_log, "completed\n")
        artifacts.write_json(
            "execution/executor_outputs/executor_results.json",
            {
                "runs": [
                    {
                        "experiment_id": "exp_01",
                        "experiment_name": "main run",
                        "dataset": "MNIST",
                        "command": "python train.py",
                        "commands_attempted": ["python train.py"],
                        "cwd": ".",
                        "exit_code": 0,
                        "status": "ok",
                        "runtime_sec": 1.0,
                        "artifacts": [],
                        "metrics": {"accuracy": 0.91},
                        "notes": "done",
                        "logs": {
                            "stdout": stdout_log,
                            "stderr": stderr_log,
                            "narrative": narrative_log,
                            "activity": "execution/executor_outputs/executor_activity.jsonl",
                        },
                        "reason_codes": [],
                    }
                ]
            },
        )
        return ExecutorSessionResult(stdout="", stderr="", returncode=0, narrative="done")

    monkeypatch.setattr(ExecutorAgent, "_run_executor_session", staticmethod(fake_session))

    agent = ExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute(
        {
            "repo_dir": str(repo_dir),
            "_p2_env_mgr": DummyEnvMgr(),
            "_p2_remaining_sec": 60,
            "_p2_attempt": 1,
        }
    )

    assert result["success"] is True
    assert expected_outputs_dir in seen["prompt"]
    manifest = artifacts.read_json("execution/executor_outputs/run_manifest.json")
    assert manifest["runs"][0]["logs"]["stdout"] == "execution/executor_outputs/experiment_exp_01_stdout.log"
    assert "task_spec" not in seen["prompt"]
    assert "execution_plan" not in seen["prompt"]
    assert "claim_alignment" not in seen["prompt"]
