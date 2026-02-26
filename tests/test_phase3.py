from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.audit_report import AuditReportAgent
from p2c.agents.phase3.align_evidence import AlignEvidenceAgent
from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
from p2c.agents.phase3.verify_claims import evaluate_claim, VerifyClaimsAgent
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient
from p2c.schemas import MetricRecord


def _mk_artifacts(tmp_path: Path) -> ArtifactManager:
    manager = ArtifactManager(tmp_path / "artifacts", "run_test")
    manager.ensure_tree()
    return manager


def test_observe_metrics_from_run_manifest(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "execution/codex_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "r1",
                    "command": "python main.py",
                    "params": {},
                    "cwd": "/workspace/repo",
                    "exit_code": 0,
                    "status": "ok",
                    "runtime_sec": 1.0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "artifacts": [],
                    "metrics": {"accuracy": 83.3},
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json("task/metric_contract.json", {"required_metrics": ["accuracy"]})

    agent = ObserveMetricsAgent(llm=LLMClient(), artifacts=artifacts, step_index=11, step_total=14)
    agent.run({})

    metrics = artifacts.read_json("results/metrics.json")
    assert metrics["records"][0]["metric_name"] == "accuracy"
    assert metrics["records"][0]["value"] == 0.833


def test_align_and_verify_dual_track_outputs(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "type": "absolute",
                    "predicate": "acc 90%",
                    "metric": "accuracy",
                    "target": 0.9,
                    "baseline": None,
                    "conditions": {},
                    "aggregation": "best",
                    "evidence_set": [],
                    "tolerance_policy": {"abs_eps": 0.02, "rel_eps": 0.03},
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/codex_outputs/claim_alignment.json",
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "required_metrics": ["accuracy"],
                    "source": ["execution/codex_outputs/run_manifest.json:r1"],
                    "evaluable": "yes",
                    "reason": "metric present",
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "results/metrics.json",
        {
            "records": [
                {
                    "metric_name": "accuracy",
                    "value": 0.9,
                    "unit": "ratio",
                    "source": "x",
                    "parsed": True,
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )

    align = AlignEvidenceAgent(llm=LLMClient(), artifacts=artifacts, step_index=12, step_total=14)
    align.run({})

    verify = VerifyClaimsAgent(llm=LLMClient(), artifacts=artifacts, step_index=13, step_total=14)
    verify.run({})

    verdict = artifacts.read_json("results/verdict.json")
    eval_verdict = artifacts.read_json("results/evaluability_verdict.json")
    assert verdict["status"] in {"SUPPORTED", "PARTIALLY_SUPPORTED", "INCONCLUSIVE", "NOT_SUPPORTED"}
    assert "Evaluability=" in (verdict.get("summary") or "")
    assert eval_verdict["status"] in {"EVALUABLE", "PARTIAL", "NOT_EVALUABLE"}


def test_absolute_claim_verdict() -> None:
    claim = {
        "claim_id": "C1",
        "type": "absolute",
        "target": 0.8,
        "baseline": None,
        "tolerance_policy": {"abs_eps": 0.02, "rel_eps": 0.03},
    }
    supported = evaluate_claim(claim, [MetricRecord(metric_name="accuracy", value=0.81, source="x")])
    not_supported = evaluate_claim(claim, [MetricRecord(metric_name="accuracy", value=0.7, source="x")])

    assert supported.status == "SUPPORTED"
    assert not_supported.status == "NOT_SUPPORTED"


def test_inconclusive_when_missing_records() -> None:
    claim = {
        "claim_id": "C2",
        "type": "absolute",
        "target": 0.8,
        "baseline": None,
        "tolerance_policy": {"abs_eps": 0.02, "rel_eps": 0.03},
    }
    verdict = evaluate_claim(claim, [])
    assert verdict.status == "INCONCLUSIVE"


def test_audit_report_handles_missing_commit_branch(tmp_path: Path, monkeypatch) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json("execution/repo_state.json", {"head": None, "branch": None, "reason_codes": ["NO_GIT_METADATA"]})
    artifacts.write_json("execution/data_manifest.json", {"entries": [], "unresolved": False, "reason_codes": []})
    artifacts.write_json("results/metrics.json", {"records": [], "reason_codes": []})
    artifacts.write_json("results/verdict.json", {"status": "INCONCLUSIVE", "claim_verdicts": [], "reason_codes": [], "summary": "x"})
    artifacts.write_json(
        "results/evaluability_verdict.json",
        {"status": "NOT_EVALUABLE", "claim_rows": [], "reason_codes": [], "summary": "x"},
    )
    artifacts.write_json("task/task_spec.json", {"entrypoints": [], "reason_codes": []})

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr("p2c.agents.phase3.audit_report.subprocess.run", lambda *args, **kwargs: _Proc())

    agent = AuditReportAgent(llm=LLMClient(), artifacts=artifacts, step_index=14, step_total=14)
    agent.run({"run_id": "run_test", "repo_dir": "Target/code"})

    report = artifacts.path("results/report.md").read_text(encoding="utf-8")
    assert "N/A (gitless run)" in report
