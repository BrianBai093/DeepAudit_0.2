from __future__ import annotations

from p2c.agents.phase3.score_and_diagnose import ScoreAndDiagnoseAgent, _failure_entries
from p2c.agents.phase3.verify_claims import _fallback_evaluate
from p2c.schemas import MetricRecord


def test_failure_entries_accepts_list_and_legacy_dict() -> None:
    list_shape = [{"step_failures": [{"step_id": "s1"}]}]
    dict_shape = {"failures": [{"step_failures": [{"step_id": "s2"}]}]}

    assert _failure_entries([]) == []
    assert _failure_entries(list_shape) == list_shape
    assert _failure_entries(dict_shape) == dict_shape["failures"]


def test_classify_gaps_accepts_empty_failure_list() -> None:
    agent = ScoreAndDiagnoseAgent.__new__(ScoreAndDiagnoseAgent)

    gaps = agent._classify_gaps(
        verdict={"claim_verdicts": []},
        manifest={"runs": []},
        env_result={"validation_passed": True},
        failures=[],
    )

    assert gaps == []


def test_verify_claims_marks_reduced_and_artifact_evidence() -> None:
    claim = {
        "claim_id": "claim_01",
        "type": "result",
        "metric": "accuracy",
        "target": 0.9,
        "tolerance_policy": {"abs_eps": 0.05, "rel_eps": 0.05},
    }
    reduced = _fallback_evaluate(
        claim,
        [
            MetricRecord(
                metric_name="accuracy",
                value=0.89,
                source="execution/executor_outputs/run_manifest.json:exp_01",
                fidelity="trend",
                execution_outcome="TREND_SUPPORTED",
                evidence_source="fresh_run",
            )
        ],
    )
    artifact = _fallback_evaluate(
        claim,
        [
            MetricRecord(
                metric_name="accuracy",
                value=0.89,
                source="execution/executor_outputs/run_manifest.json:exp_02",
                fidelity="artifact",
                execution_outcome="TREND_SUPPORTED",
                evidence_source="existing_logs",
            )
        ],
    )
    full = _fallback_evaluate(
        claim,
        [
            MetricRecord(
                metric_name="accuracy",
                value=0.9,
                source="execution/executor_outputs/run_manifest.json:exp_03",
                fidelity="full",
                execution_outcome="FULLY_REPRODUCED",
                evidence_source="fresh_run",
            )
        ],
    )

    assert "REDUCED_FIDELITY_EVIDENCE" in reduced.reason_codes
    assert "ARTIFACT_BASED_EVIDENCE" in artifact.reason_codes
    assert "FULL_FIDELITY_EVIDENCE" in full.reason_codes


def test_score_execution_success_prefers_full_over_reduced_fidelity() -> None:
    agent = ScoreAndDiagnoseAgent.__new__(ScoreAndDiagnoseAgent)

    full_score = agent._score_execution_success(
        {"runs": [{"run_id": "exp_01", "status": "ok", "fidelity": "full", "execution_outcome": "FULLY_REPRODUCED"}]}
    )
    trend_score = agent._score_execution_success(
        {"runs": [{"run_id": "exp_01", "status": "ok", "fidelity": "trend", "execution_outcome": "TREND_SUPPORTED"}]}
    )
    smoke_score = agent._score_execution_success(
        {"runs": [{"run_id": "exp_01", "status": "ok", "fidelity": "smoke", "execution_outcome": "EXECUTABLE"}]}
    )

    assert full_score.score > trend_score.score > smoke_score.score


def test_compute_ecr_requires_full_fidelity_supported_claims() -> None:
    verdict = {
        "claim_verdicts": [
            {
                "claim_id": "claim_01",
                "status": "SUPPORTED",
                "reason_codes": ["MATCHED_METRIC", "FULL_FIDELITY_EVIDENCE"],
            }
        ]
    }
    manifest = {
        "runs": [
            {
                "run_id": "exp_01",
                "status": "ok",
                "execution_outcome": "FULLY_REPRODUCED",
                "reason_codes": [],
            }
        ]
    }
    claims_ir = {
        "claims": [
            {"claim_id": "claim_01", "type": "result"},
        ]
    }

    assert ScoreAndDiagnoseAgent._compute_ecr(verdict, manifest, 90, claims_ir)[0] is True

    reduced_verdict = {
        "claim_verdicts": [
            {
                "claim_id": "claim_01",
                "status": "SUPPORTED",
                "reason_codes": ["MATCHED_METRIC", "REDUCED_FIDELITY_EVIDENCE"],
            }
        ]
    }
    ok, reason = ScoreAndDiagnoseAgent._compute_ecr(reduced_verdict, manifest, 90, claims_ir)
    assert ok is False
    assert "lack full-fidelity evidence" in reason
