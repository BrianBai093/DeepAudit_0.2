"""Tests for the claim context preservation pipeline (Phase 1 → Phase 2 → Phase 3).

Validates that:
1. Phase 1 carries table_anchor from fingerprint into claims_ir conditions
2. Phase 2 marks ambiguous claims as 'partial' instead of blindly 'yes'
3. Phase 3 align_evidence disambiguates same-named metrics using target proximity
4. Phase 3 verify_claims returns INCONCLUSIVE for unaligned ambiguous claims
"""

from __future__ import annotations


# ---- Phase 1: table_anchor propagation ----

def test_build_claims_ir_preserves_table_anchor():
    """visual_anchor from fingerprint should appear as table_anchor in conditions."""
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    fingerprint = {
        "claims": [
            {
                "id": "claim_09",
                "claim_type": "result",
                "fact": "accuracy = 0.9685",
                "scope": "from classification report in paper",
                "evidence_anchors": {
                    "text_anchor": "atomic_criteria[8]",
                    "visual_anchor": "Table 1",
                    "visual_data": {},
                },
                "reason_codes": ["TABLE_EXPANDED", "CLF_REPORT"],
            },
            {
                "id": "claim_10",
                "claim_type": "result",
                "fact": "accuracy = 0.7617",
                "scope": "from classification report in paper",
                "evidence_anchors": {
                    "text_anchor": "atomic_criteria[9]",
                    "visual_anchor": "Table 2",
                },
                "reason_codes": ["TABLE_EXPANDED", "CLF_REPORT"],
            },
        ],
        "reason_codes": [],
    }

    agent = BuildClaimsIRAgent.__new__(BuildClaimsIRAgent)
    claims, _ = agent._claims_from_fingerprint(fingerprint)

    assert len(claims) == 2
    assert claims[0].conditions.get("table_anchor") == "Table 1"
    assert claims[1].conditions.get("table_anchor") == "Table 2"
    # scope should still be there
    assert claims[0].conditions.get("scope") == "from classification report in paper"


def test_build_claims_ir_no_anchor_no_crash():
    """Claims without visual_anchor should not have table_anchor in conditions."""
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    fingerprint = {
        "claims": [
            {
                "id": "claim_05",
                "claim_type": "result",
                "fact": "accuracy > 99%",
                "scope": "MNIST dataset",
                "evidence_anchors": {
                    "text_anchor": "atomic_criteria[4]",
                    "visual_anchor": None,
                },
                "reason_codes": [],
            },
        ],
        "reason_codes": [],
    }

    agent = BuildClaimsIRAgent.__new__(BuildClaimsIRAgent)
    claims, _ = agent._claims_from_fingerprint(fingerprint)

    assert len(claims) == 1
    assert "table_anchor" not in claims[0].conditions
    assert claims[0].conditions.get("scope") == "MNIST dataset"


# ---- Phase 2: conservative claim_alignment ----

def test_claim_alignment_marks_ambiguous_as_partial():
    """When 3 claims all want 'accuracy' but only one value exists, mark partial."""
    from p2c.agents.phase2.result_extraction import build_claim_alignment
    from p2c.schemas import ClaimItem, ClaimsIR

    claims_ir = ClaimsIR(claims=[
        ClaimItem(claim_id="c1", type="result", predicate="acc=0.97", metric="accuracy", target=0.9685),
        ClaimItem(claim_id="c2", type="result", predicate="acc=0.76", metric="accuracy", target=0.7617),
        ClaimItem(claim_id="c3", type="result", predicate="acc=0.84", metric="accuracy", target=0.84),
    ])
    collected = {"accuracy": 0.9882}

    doc = build_claim_alignment(claims_ir, collected)

    for item in doc.claims:
        assert item.evaluable == "partial", f"{item.claim_id} should be partial, got {item.evaluable}"
        assert "deferred to Phase 3" in (item.reason or "")


def test_claim_alignment_unique_metric_stays_yes():
    """A single claim for a metric should remain evaluable='yes'."""
    from p2c.agents.phase2.result_extraction import build_claim_alignment
    from p2c.schemas import ClaimItem, ClaimsIR

    claims_ir = ClaimsIR(claims=[
        ClaimItem(claim_id="c1", type="result", predicate="loss=0.15", metric="loss", target=0.15),
        ClaimItem(claim_id="c2", type="config", predicate="epochs=100"),
    ])
    collected = {"loss": 0.1523}

    doc = build_claim_alignment(claims_ir, collected)

    loss_item = next(i for i in doc.claims if i.claim_id == "c1")
    config_item = next(i for i in doc.claims if i.claim_id == "c2")
    assert loss_item.evaluable == "yes"
    assert config_item.evaluable == "no"  # no metric for config


# ---- Phase 3: align_evidence disambiguation ----

def test_align_evidence_disambiguates_by_target():
    """When 3 claims want 'accuracy' with different targets, match by proximity."""
    from p2c.agents.phase3.align_evidence import AlignEvidenceAgent
    from p2c.schemas import MetricRecord

    records = [
        MetricRecord(metric_name="accuracy", value=0.9882, source="run1"),
        MetricRecord(metric_name="accuracy", value=0.9972, source="run2"),
    ]

    # Claim with target close to 0.9882 should match
    matched = AlignEvidenceAgent._match_records(
        metric_name="accuracy",
        target=0.9685,
        conditions={"table_anchor": "Table 1"},
        records=records,
        is_ambiguous=True,
    )
    # Should get the closest value(s) to 0.9685
    assert len(matched) >= 1
    values = [r.value for r in matched]
    assert 0.9882 in values  # closest to 0.9685

    # Claim with target=0.7617 — far from both records
    matched_far = AlignEvidenceAgent._match_records(
        metric_name="accuracy",
        target=0.7617,
        conditions={"table_anchor": "Table 2"},
        records=records,
        is_ambiguous=True,
    )
    # Should return nothing — no record is close to 0.7617
    assert len(matched_far) == 0, (
        f"Should not match any record for target=0.7617 (got {[r.value for r in matched_far]})"
    )


def test_align_evidence_non_ambiguous_returns_all():
    """When only 1 claim uses a metric, all matching records should be returned."""
    from p2c.agents.phase3.align_evidence import AlignEvidenceAgent
    from p2c.schemas import MetricRecord

    records = [
        MetricRecord(metric_name="loss", value=0.15, source="run1"),
        MetricRecord(metric_name="loss", value=0.12, source="run2"),
    ]

    matched = AlignEvidenceAgent._match_records(
        metric_name="loss",
        target=0.15,
        conditions={},
        records=records,
        is_ambiguous=False,
    )
    assert len(matched) == 2  # all records returned


# ---- Phase 3: verify_claims with missing_reason ----

def test_verify_inconclusive_for_ambiguous_unaligned():
    """Claims that can't be aligned should get INCONCLUSIVE with ALIGNMENT_AMBIGUOUS."""
    from p2c.agents.phase3.verify_claims import _fallback_evaluate as evaluate_claim

    claim = {
        "claim_id": "claim_10",
        "type": "result",
        "metric": "accuracy",
        "target": 0.7617,
        "tolerance_policy": {"abs_eps": 0.005, "rel_eps": 0.02},
    }

    verdict = evaluate_claim(
        claim,
        matched_records=[],
        missing_reason="Metric 'accuracy' was collected but could not be aligned to this specific claim (Table 2)",
    )

    assert verdict.status == "INCONCLUSIVE"
    assert "ALIGNMENT_AMBIGUOUS" in verdict.reason_codes


def test_verify_supported_when_matched():
    """A properly matched claim should still get SUPPORTED/NOT_SUPPORTED."""
    from p2c.agents.phase3.verify_claims import _fallback_evaluate as evaluate_claim
    from p2c.schemas import MetricRecord

    claim = {
        "claim_id": "claim_09",
        "type": "result",
        "metric": "accuracy",
        "target": 0.9685,
        "tolerance_policy": {"abs_eps": 0.005, "rel_eps": 0.02},
    }

    verdict = evaluate_claim(
        claim,
        matched_records=[MetricRecord(metric_name="accuracy", value=0.9882, source="run1")],
    )

    # 0.9882 vs 0.9685 = diff 0.0197, threshold = max(0.005, 0.02*0.9685) = 0.0194
    # 0.0197 > 0.0194 → NOT_SUPPORTED (just barely)
    assert verdict.status == "NOT_SUPPORTED"
    assert verdict.compared_value == 0.9882
    assert verdict.target_value == 0.9685
