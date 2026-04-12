from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
from p2c.io_artifacts import ArtifactManager


def test_observe_metrics_preserves_unbounded_count_values(tmp_path: Path, monkeypatch) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_counts")
    artifacts.ensure_tree()
    artifacts.write_json(
        "task/metric_contract.json",
        {
            "required_metrics": ["fraud case counts"],
            "parsers": [],
            "normalization": {},
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/execution_plan.json",
        {
            "plan_id": "p1",
            "plan_version": 1,
            "python_version": "3.10",
            "execution_steps": [
                {
                    "step_id": "step_02_data_check",
                    "description": "check data",
                    "command": "python check_data.py",
                    "cwd": ".",
                    "timeout_sec": 60,
                    "depends_on": [],
                    "expected_metrics": ["fraud case counts"],
                    "is_setup": True,
                    "retry_on_failure": False,
                    "fallback_commands": [],
                    "required_artifacts": [],
                    "produced_artifacts": [],
                }
            ],
            "env_name": "test_env",
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/codex_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "step_02_data_check",
                    "command": "python check_data.py",
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "metrics": {"fraud case counts": 492},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )

    agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, "no key"))

    result = agent.execute({})

    records = result["metrics"]["records"]
    assert records[0]["metric_name"] == "fraud case counts"
    assert records[0]["value"] == 492.0


def test_observe_metrics_percent_normalizes_bounded_metrics(tmp_path: Path, monkeypatch) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_percent")
    artifacts.ensure_tree()
    artifacts.write_json(
        "task/metric_contract.json",
        {
            "required_metrics": ["accuracy"],
            "parsers": [],
            "normalization": {},
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/execution_plan.json",
        {
            "plan_id": "p1",
            "plan_version": 1,
            "python_version": "3.10",
            "execution_steps": [
                {
                    "step_id": "step_03_eval",
                    "description": "eval",
                    "command": "python eval.py",
                    "cwd": ".",
                    "timeout_sec": 60,
                    "depends_on": [],
                    "expected_metrics": ["accuracy"],
                    "is_setup": False,
                    "retry_on_failure": False,
                    "fallback_commands": [],
                    "required_artifacts": [],
                    "produced_artifacts": [],
                }
            ],
            "env_name": "test_env",
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/codex_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "step_03_eval",
                    "command": "python eval.py",
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "metrics": {"accuracy": 95.5},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )

    agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, "no key"))

    result = agent.execute({})

    assert result["metrics"]["records"][0]["value"] == 0.955
