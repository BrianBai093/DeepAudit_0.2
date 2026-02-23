from __future__ import annotations

from p2c.agents.verify_claims import evaluate_claim
from p2c.schemas import MetricRecord


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
