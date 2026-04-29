from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.align_evidence import AlignEvidenceAgent
from p2c.agents.phase3.claim_inputs import load_effective_claims_ir
from p2c.agents.phase3.execution_summary_evidence import ExecutionSummaryEvidenceAgent
from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
from p2c.io_artifacts import ArtifactManager
from p2c.schemas import MetricRecord


def make_artifacts(tmp_path: Path, run_id: str) -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path / "artifacts", run_id)
    artifacts.ensure_tree()
    return artifacts


def test_execution_summary_and_executor_results_override_guard_downgrade(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path, "effective")
    artifacts.write_json(
        "execution/executor_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_01",
                    "experiment_id": "exp_01",
                    "experiment_name": "table 1",
                    "command": "python train.py",
                    "commands_attempted": ["python train.py"],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "failed",
                    "fidelity": "smoke",
                    "evidence_source": "fresh_run",
                    "stop_reason": "guardrail_blocked",
                    "metrics": {},
                    "logs": {},
                    "reason_codes": ["COMMAND_NOT_OBSERVED", "UNTRACEABLE_METRICS"],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/executor_outputs/executor_results.json",
        {
            "runs": [
                {
                    "experiment_id": "exp_01",
                    "experiment_name": "table 1",
                    "command": "python train.py",
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "fidelity": "smoke+trend",
                    "evidence_source": "fresh_runs",
                    "metrics": {"accuracy": 0.98},
                    "reason_codes": [],
                }
            ]
        },
    )
    artifacts.write_text(
        "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md",
        """# Execution Summary

#### Exp_01: Table 1 Benchmark
**Status:** OK | **Fidelity:** Mixed (Smoke + Trend)

**Key Results:**
- Accuracy: 99.0% accuracy
""",
    )

    agent = ExecutionSummaryEvidenceAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({})

    run = result["effective_run_manifest"]["runs"][0]
    assert run["status"] == "ok"
    assert run["fidelity"] == "trend"
    assert run["evidence_source"] == "mixed"
    assert run["metrics"]["accuracy"] == 0.99
    assert "COMMAND_NOT_OBSERVED" not in run["reason_codes"]
    assert "SUMMARY_FINAL_PRIORITY" in run["reason_codes"]

    evidence = artifacts.read_json("results/execution_summary_evidence.json")
    assert evidence["summary_path"] == "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md"
    assert evidence["conflicts"]


def test_observe_metrics_reads_effective_manifest_and_summary_sources(tmp_path: Path, monkeypatch) -> None:
    artifacts = make_artifacts(tmp_path, "observe_effective")
    artifacts.write_json(
        "results/effective_run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_01",
                    "experiment_id": "exp_01",
                    "experiment_name": "table 1",
                    "command": "python train.py",
                    "commands_attempted": ["python train.py"],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "fidelity": "trend",
                    "execution_outcome": "TREND_SUPPORTED",
                    "evidence_source": "fresh_run",
                    "metrics": {"bp_mnist_trend_test_accuracy": 0.98},
                    "logs": {},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "results/execution_summary_evidence.json",
        {
            "summary_path": "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md",
            "summary_runs": [
                {
                    "run_id": "exp_01",
                    "experiment_id": "exp_01",
                    "fidelity": "trend",
                    "execution_outcome": "TREND_SUPPORTED",
                    "evidence_source": "mixed",
                    "metrics": {"accuracy": 0.99},
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "task/metric_contract.json",
        {"required_metrics": ["accuracy"], "parsers": [], "normalization": {}, "reason_codes": []},
    )

    agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, "no key"))

    result = agent.execute({})

    sources = {row["source"] for row in result["metrics"]["records"]}
    assert "results/effective_run_manifest.json:exp_01" in sources
    assert "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md:exp_01" in sources


def test_phase3_prefers_phase2_execution_package_over_raw_manifest(tmp_path: Path, monkeypatch) -> None:
    artifacts = make_artifacts(tmp_path, "package_priority")
    artifacts.write_json(
        "execution/executor_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_02",
                    "experiment_id": "exp_02",
                    "experiment_name": "conv table",
                    "command": "",
                    "cwd": ".",
                    "exit_code": 1,
                    "status": "failed",
                    "fidelity": "smoke",
                    "metrics": {},
                    "reason_codes": ["EXPERIMENT_RESULT_MISSING"],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/executor_outputs/phase2_execution_package.json",
        {
            "schema_version": "phase2_execution_package.v1",
            "source_files": {},
            "experiments": [
                {
                    "experiment_id": "exp_02",
                    "name": "Table 1 convolutional benchmark",
                    "aliases": ["exp_02_bp"],
                    "paper_target_refs": [],
                    "attempts": [
                        {
                            "attempt_id": "exp_02.bp.mnist.conv.full.100ep",
                            "experiment_id": "exp_02",
                            "source_experiment_id": "exp_02_bp",
                            "experiment_name": "Table 1 convolutional BP benchmark",
                            "scope": {"algorithm": "bp", "dataset": "mnist", "model_family": "conv", "epochs": 100},
                            "command": "python main_pytorch.py",
                            "commands_attempted": ["python main_pytorch.py"],
                            "cwd": ".",
                            "exit_code": 0,
                            "status": "ok",
                            "fidelity": "full",
                            "execution_outcome": "FULLY_REPRODUCED",
                            "evidence_source": "fresh_run",
                            "stop_reason": "full_run_complete",
                            "metrics": [
                                {
                                    "metric_name": "bp_mnist_conv_full_test_accuracy",
                                    "raw_metric_name": "test_accuracy",
                                    "value_ratio": 0.9055,
                                    "raw_value": 90.55,
                                    "unit": "ratio",
                                    "algorithm": "bp",
                                    "dataset": "mnist",
                                    "model_family": "conv",
                                    "fidelity": "full",
                                    "source_attempt_id": "exp_02.bp.mnist.conv.full.100ep",
                                }
                            ],
                            "logs": {},
                            "reason_codes": [],
                        }
                    ],
                    "best_attempts_by_scope": {"bp|mnist|conv|test_accuracy": "exp_02.bp.mnist.conv.full.100ep"},
                    "metrics": [],
                    "failures": [],
                    "logs": [],
                    "summary_for_llm": "full run produced accuracy",
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_text("execution/executor_outputs/PHASE2_RESULTS.md", "# Phase 2 Results\n")
    artifacts.write_json(
        "task/metric_contract.json",
        {"required_metrics": ["accuracy"], "parsers": [], "normalization": {}, "reason_codes": []},
    )

    evidence_agent = ExecutionSummaryEvidenceAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    evidence = evidence_agent.execute({})
    run = evidence["effective_run_manifest"]["runs"][0]
    assert run["run_id"] == "exp_02.bp.mnist.conv.full.100ep"
    assert run["status"] == "ok"
    assert run["metrics"]["bp_mnist_conv_full_test_accuracy"] == 0.9055
    assert evidence["execution_summary_evidence"]["conflicts"]

    metrics_agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(metrics_agent, "safe_chat_text", lambda system, user: (None, "no key"))
    result = metrics_agent.execute({})
    record = result["metrics"]["records"][0]
    assert record["source"] == "execution/executor_outputs/phase2_execution_package.json:exp_02.bp.mnist.conv.full.100ep"
    assert record["experiment_id"] == "exp_02"
    assert record["fidelity"] == "full"
    assert record["value"] == 0.9055


def test_execution_complete_table_overlays_canonical_conv_experiment(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path, "execution_complete")
    artifacts.write_json(
        "execution/executor_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_02",
                    "experiment_id": "exp_02",
                    "experiment_name": "Table 1 convolutional benchmark",
                    "command": "python main_pytorch.py",
                    "cwd": ".",
                    "exit_code": 1,
                    "status": "failed",
                    "fidelity": "smoke",
                    "metrics": {},
                    "reason_codes": ["EXPERIMENT_RESULT_MISSING"],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json("execution/executor_outputs/executor_results.json", {"runs": []})
    artifacts.write_text(
        "execution/executor_outputs/EXECUTION_COMPLETE.md",
        """# PEPITA Phase 2 Execution - COMPLETE

### Complete Results Table

| Experiment | Config | Fidelity | Epochs | Accuracy | Status | Time |
|-----------|--------|----------|--------|----------|--------|------|
| exp_02_bp | MNIST Conv BP | smoke | 1 | 91.41% | \\u2705 | 29.2s |
| exp_02_bp | MNIST Conv BP | full | 100 | 90.55% | \\u2705 | 2497.8s |
| exp_02_bp | CIFAR10 Conv BP | full | 100 | - | \\u274c | 5.8s |
""",
    )

    agent = ExecutionSummaryEvidenceAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({})

    evidence = result["execution_summary_evidence"]
    assert evidence["summary_path"] == "execution/executor_outputs/EXECUTION_COMPLETE.md"
    run = result["effective_run_manifest"]["runs"][0]
    assert run["run_id"] == "exp_02"
    assert run["status"] == "ok"
    assert run["fidelity"] == "full"
    assert abs(run["metrics"]["bp_mnist_conv_smoke_test_accuracy"] - 0.9141) < 1e-9
    assert abs(run["metrics"]["bp_mnist_conv_full_test_accuracy"] - 0.9055) < 1e-9


def test_observe_metrics_normalizes_scoped_percent_accuracy_values(tmp_path: Path, monkeypatch) -> None:
    artifacts = make_artifacts(tmp_path, "observe_percent")
    artifacts.write_json(
        "results/effective_run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_02",
                    "experiment_id": "exp_02",
                    "experiment_name": "table 1 conv",
                    "command": "python main_pytorch.py",
                    "commands_attempted": [],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "fidelity": "full",
                    "execution_outcome": "FULLY_REPRODUCED",
                    "evidence_source": "fresh_run",
                    "metrics": {"test_accuracy": 90.55, "bp_mnist_conv_full_test_accuracy": 90.55},
                    "logs": {},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json("results/execution_summary_evidence.json", {"summary_runs": [], "reason_codes": []})
    artifacts.write_json(
        "task/metric_contract.json",
        {"required_metrics": ["accuracy"], "parsers": [], "normalization": {}, "reason_codes": []},
    )

    agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, "no key"))

    result = agent.execute({})

    values = {row["metric_name"]: row["value"] for row in result["metrics"]["records"]}
    assert values["test_accuracy"] == 0.9055
    assert values["bp_mnist_conv_full_test_accuracy"] == 0.9055


def test_align_evidence_matches_effective_manifest_by_experiment_and_metric_suffix() -> None:
    claim = {
        "claim_id": "claim_01",
        "type": "result",
        "metric": "accuracy",
        "target": 0.98,
        "predicate": "accuracy = 0.98",
        "conditions": {"experiment_id": "exp_01"},
    }
    records = [
        MetricRecord(
            metric_name="bp_mnist_trend_test_accuracy",
            value=0.98,
            source="results/effective_run_manifest.json:exp_01",
            experiment_id="exp_01",
        ),
        MetricRecord(
            metric_name="accuracy",
            value=0.50,
            source="results/effective_run_manifest.json:exp_02",
            experiment_id="exp_02",
        ),
    ]
    candidate_runs = [
        type("Run", (), {"run_id": "exp_01", "experiment_id": "exp_01", "logs": None})()
    ]

    matched = AlignEvidenceAgent._match_records(
        claim=claim,
        candidate_runs=candidate_runs,
        records=records,
        ambiguous_metric=True,
    )

    assert [row.value for row in matched] == [0.98]


def test_phase3_effective_claims_normalize_stale_mean_std_targets(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path, "effective_claims")
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "experiments": [],
            "claims": [
                {
                    "claim_id": "claim_04",
                    "type": "result",
                    "predicate": "accuracy = 98.63±0.03",
                    "metric": "accuracy",
                    "target": 0.03,
                    "conditions": {"scope": "BP, fully connected models, MNIST"},
                    "tolerance_policy": {"abs_eps": 0.005, "rel_eps": 0.02},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )

    claims = load_effective_claims_ir(artifacts)
    claim = claims["claims"][0]

    assert abs(claim["target"] - 0.9863) < 1e-6
    assert abs(claim["tolerance_policy"]["std_eps"] - 0.0003) < 1e-9
    assert "MEAN_STD_TARGET_NORMALIZED" in claim["reason_codes"]
    assert artifacts.path("results/effective_claims_ir.json").is_file()


def test_align_evidence_uses_scope_algorithm_dataset_instead_of_nearest_generic_accuracy() -> None:
    claim = {
        "claim_id": "claim_04",
        "type": "result",
        "metric": "accuracy",
        "target": 0.9863,
        "predicate": "accuracy = 98.63±0.03",
        "conditions": {"experiment_id": "exp_01", "scope": "BP, fully connected models, MNIST"},
    }
    records = [
        MetricRecord(
            metric_name="dfa_mnist_smoke_test_accuracy",
            value=0.9196,
            source="results/effective_run_manifest.json:exp_01",
            run_id="exp_01",
            experiment_id="exp_01",
            fidelity="trend",
        ),
        MetricRecord(
            metric_name="accuracy",
            value=0.9860,
            source="results/effective_run_manifest.json:exp_01",
            run_id="exp_01",
            experiment_id="exp_01",
            fidelity="trend",
        ),
        MetricRecord(
            metric_name="bp_mnist_trend_test_accuracy",
            value=0.9792,
            source="results/effective_run_manifest.json:exp_01",
            run_id="exp_01",
            experiment_id="exp_01",
            fidelity="trend",
        ),
        MetricRecord(
            metric_name="bp_cifar10_trend_test_accuracy",
            value=0.5527,
            source="results/effective_run_manifest.json:exp_01",
            run_id="exp_01",
            experiment_id="exp_01",
            fidelity="trend",
        ),
    ]
    candidate_runs = [
        type(
            "Run",
            (),
            {
                "run_id": "exp_01",
                "experiment_id": "exp_01",
                "experiment_name": "Table 1 fully connected benchmark",
                "dataset": "MNIST",
                "command": "python main.py",
                "commands_attempted": ["python main.py --learn_type BP --mnist"],
                "logs": None,
            },
        )()
    ]

    matched = AlignEvidenceAgent._match_records(
        claim=claim,
        candidate_runs=candidate_runs,
        records=records,
        ambiguous_metric=True,
    )

    assert [row.metric_name for row in matched] == ["bp_mnist_trend_test_accuracy"]


def test_align_evidence_marks_unexecuted_scope_unmatched_instead_of_wrong_algorithm() -> None:
    claim = {
        "claim_id": "claim_05",
        "type": "result",
        "metric": "accuracy",
        "target": 0.5527,
        "predicate": "accuracy = 55.27±0.32",
        "conditions": {"experiment_id": "exp_01", "scope": "BP, fully connected models, CIFAR10"},
    }
    records = [
        MetricRecord(
            metric_name="dfa_mnist_smoke_test_accuracy",
            value=0.9196,
            source="results/effective_run_manifest.json:exp_01",
            run_id="exp_01",
            experiment_id="exp_01",
        ),
        MetricRecord(
            metric_name="bp_mnist_trend_test_accuracy",
            value=0.9792,
            source="results/effective_run_manifest.json:exp_01",
            run_id="exp_01",
            experiment_id="exp_01",
        ),
    ]
    candidate_runs = [
        type("Run", (), {"run_id": "exp_01", "experiment_id": "exp_01", "logs": None})()
    ]

    matched = AlignEvidenceAgent._match_records(
        claim=claim,
        candidate_runs=candidate_runs,
        records=records,
        ambiguous_metric=True,
    )

    assert matched == []


def test_align_evidence_uses_run_text_for_conv_scope_when_metric_is_compact() -> None:
    claim = {
        "claim_id": "claim_conv",
        "type": "result",
        "metric": "accuracy",
        "target": 0.9886,
        "predicate": "accuracy = 98.86±0.04",
        "conditions": {"experiment_id": "exp_02", "scope": "BP, convolutional models, MNIST"},
    }
    records = [
        MetricRecord(
            metric_name="bp_mnist_smoke_test_accuracy",
            value=0.9219,
            source="results/effective_run_manifest.json:exp_02",
            run_id="exp_02",
            experiment_id="exp_02",
        )
    ]
    candidate_runs = [
        type(
            "Run",
            (),
            {
                "run_id": "exp_02",
                "experiment_id": "exp_02",
                "experiment_name": "Table 1 convolutional benchmark",
                "dataset": "MNIST",
                "command": "python main_pytorch.py --model Net1conv1fcXL",
                "commands_attempted": [],
                "logs": None,
            },
        )()
    ]

    matched = AlignEvidenceAgent._match_records(
        claim=claim,
        candidate_runs=candidate_runs,
        records=records,
        ambiguous_metric=True,
    )

    assert [row.metric_name for row in matched] == ["bp_mnist_smoke_test_accuracy"]


def test_align_evidence_prefers_full_fidelity_before_numeric_closeness() -> None:
    claim = {
        "claim_id": "claim_conv",
        "type": "result",
        "metric": "accuracy",
        "target": 0.9886,
        "predicate": "accuracy = 98.86±0.04",
        "conditions": {"experiment_id": "exp_02", "scope": "BP, convolutional models, MNIST"},
    }
    records = [
        MetricRecord(
            metric_name="bp_mnist_conv_trend_test_accuracy",
            value=0.9194,
            source="results/effective_run_manifest.json:exp_02",
            run_id="exp_02",
            experiment_id="exp_02",
            fidelity="trend",
        ),
        MetricRecord(
            metric_name="bp_mnist_conv_full_test_accuracy",
            value=0.9055,
            source="results/effective_run_manifest.json:exp_02",
            run_id="exp_02",
            experiment_id="exp_02",
            fidelity="full",
            execution_outcome="FULLY_REPRODUCED",
        ),
    ]
    candidate_runs = [
        type(
            "Run",
            (),
            {
                "run_id": "exp_02",
                "experiment_id": "exp_02",
                "experiment_name": "Table 1 convolutional benchmark",
                "dataset": "MNIST",
                "command": "python main_pytorch.py --model Net1conv1fcXL",
                "commands_attempted": [],
                "logs": None,
            },
        )()
    ]

    matched = AlignEvidenceAgent._match_records(
        claim=claim,
        candidate_runs=candidate_runs,
        records=records,
        ambiguous_metric=True,
    )

    assert [row.metric_name for row in matched] == ["bp_mnist_conv_full_test_accuracy"]


def test_align_evidence_rejects_fc_scope_when_run_provenance_is_conv() -> None:
    claim = {
        "claim_id": "claim_fc",
        "type": "result",
        "metric": "accuracy",
        "target": 0.9863,
        "predicate": "accuracy = 98.63±0.03",
        "conditions": {"experiment_id": "exp_02", "scope": "BP, fully connected models, MNIST"},
    }
    records = [
        MetricRecord(
            metric_name="bp_mnist_smoke_test_accuracy",
            value=0.9219,
            source="results/effective_run_manifest.json:exp_02",
            run_id="exp_02",
            experiment_id="exp_02",
        )
    ]
    candidate_runs = [
        type(
            "Run",
            (),
            {
                "run_id": "exp_02",
                "experiment_id": "exp_02",
                "experiment_name": "Table 1 convolutional benchmark",
                "dataset": "MNIST",
                "command": "python main_pytorch.py --model Net1conv1fcXL",
                "commands_attempted": [],
                "logs": None,
            },
        )()
    ]

    matched = AlignEvidenceAgent._match_records(
        claim=claim,
        candidate_runs=candidate_runs,
        records=records,
        ambiguous_metric=True,
    )

    assert matched == []
