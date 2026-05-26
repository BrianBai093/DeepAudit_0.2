from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
from p2c.io_artifacts import ArtifactManager


def test_observe_metrics_preserves_unbounded_values_from_manifest(tmp_path: Path, monkeypatch) -> None:
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
        "execution/executor_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_01",
                    "experiment_id": "exp_01",
                    "experiment_name": "data check",
                    "command": "python check_data.py",
                    "commands_attempted": ["python check_data.py"],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "metrics": {"fraud case counts": 492},
                    "logs": {},
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


def test_observe_metrics_reads_stdout_log_from_manifest_logs(tmp_path: Path, monkeypatch) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_percent")
    artifacts.ensure_tree()
    stdout_log = "execution/executor_outputs/experiment_exp_01_stdout.log"
    artifacts.write_text(stdout_log, "METRIC:accuracy=95.5\n")
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
        "execution/executor_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_01",
                    "experiment_id": "exp_01",
                    "experiment_name": "eval",
                    "command": "python eval.py",
                    "commands_attempted": ["python eval.py"],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "metrics": {},
                    "logs": {"stdout": stdout_log},
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


def test_observe_metrics_propagates_run_provenance(tmp_path: Path, monkeypatch) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_provenance")
    artifacts.ensure_tree()
    stdout_log = "execution/executor_outputs/experiment_exp_01_stdout.log"
    artifacts.write_text(stdout_log, "METRIC:accuracy=0.91\n")
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
        "execution/executor_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_01",
                    "experiment_id": "exp_01",
                    "experiment_name": "trend run",
                    "command": "python train.py --epochs 1",
                    "commands_attempted": ["python train.py --epochs 1"],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "partial",
                    "fidelity": "trend",
                    "execution_outcome": "TREND_SUPPORTED",
                    "evidence_source": "fresh_run",
                    "metrics": {},
                    "logs": {"stdout": stdout_log},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )

    agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, "no key"))

    result = agent.execute({})

    record = result["metrics"]["records"][0]
    assert record["run_id"] == "exp_01"
    assert record["experiment_id"] == "exp_01"
    assert record["fidelity"] == "trend"
    assert record["execution_outcome"] == "TREND_SUPPORTED"
    assert record["evidence_source"] == "fresh_run"
