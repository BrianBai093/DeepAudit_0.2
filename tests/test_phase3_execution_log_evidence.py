from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.execution_log_evidence import ExecutionLogEvidenceAgent
from p2c.io_artifacts import ArtifactManager


def _artifacts(tmp_path: Path) -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_logs")
    artifacts.ensure_tree()
    artifacts.path("execution/executor_outputs").mkdir(parents=True, exist_ok=True)
    return artifacts


def _log_by_name(result: dict, name: str) -> dict:
    return next(row for row in result["execution_log_evidence"]["logs"] if str(row.get("path", "")).endswith(name))


def test_parse_pepita_conv_log_extracts_100_epoch_curve(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    lines = []
    for epoch in range(1, 101):
        lines.append(f"[{epoch},  100] loss: {2.0 - epoch * 0.01:.4f}")
        lines.append(f"Test accuracy: {50.0 + epoch * 0.1:.2f} %")
    artifacts.write_text(
        "execution/executor_outputs/experiment_exp_02_full_CIFAR10_Conv_BP_stdout.log",
        "\n".join(lines),
    )

    result = ExecutionLogEvidenceAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1).execute({})
    log = _log_by_name(result, "experiment_exp_02_full_CIFAR10_Conv_BP_stdout.log")

    assert log["experiment_id"] == "exp_02"
    assert log["dataset"] == "cifar10"
    assert log["model_family"] == "conv"
    assert log["algorithm"] == "bp"
    test_curve = next(curve for curve in log["curves"] if curve["metric_name"] == "test_accuracy")
    loss_curve = next(curve for curve in log["curves"] if curve["metric_name"] == "loss")
    assert len(test_curve["points"]) == 100
    assert len(loss_curve["points"]) == 100
    assert log["metrics"][-1]["metric_name"] == "final_loss"
    assert "EXECUTED_CURVE" in log["reason_codes"]


def test_parse_fc_smoke_log_extracts_train_val_and_mean_test(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts.write_text(
        "execution/executor_outputs/experiment_exp_01_smoke_MNIST_FC_BP_stdout.log",
        "\n".join(
            [
                "Training accuracy = 93.5",
                "Validation accuracy = 91.25",
                "Mean test accuracy = [90.75]",
            ]
        ),
    )

    result = ExecutionLogEvidenceAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1).execute({})
    log = _log_by_name(result, "experiment_exp_01_smoke_MNIST_FC_BP_stdout.log")
    metrics = {row["metric_name"]: row["value"] for row in log["metrics"]}

    assert metrics["train_accuracy"] == 0.935
    assert metrics["val_accuracy"] == 0.9125
    assert metrics["mean_test_accuracy"] == 0.9075
    assert "SMOKE_ONLY" in log["reason_codes"]
    assert "EXECUTED_METRIC" in log["reason_codes"]


def test_parse_skipped_log_marks_skip_without_fabricating_metric(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts.write_text(
        "execution/executor_outputs/experiment_exp_03_full_CIFAR10_Conv_PEPITA_stdout.log",
        "exp_03 skipped: implementation entrypoint for PEPITA is unavailable.\n",
    )

    result = ExecutionLogEvidenceAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1).execute({})
    log = _log_by_name(result, "experiment_exp_03_full_CIFAR10_Conv_PEPITA_stdout.log")

    assert log["metrics"] == []
    assert log["curves"] == []
    assert "entrypoint" in log["skip_reason"]
    assert "SKIPPED_REASON" in log["reason_codes"]
