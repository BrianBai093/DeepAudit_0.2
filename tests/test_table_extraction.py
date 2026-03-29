"""Tests for table metric extraction — verifies classification report tables
are handled correctly and implausible values (sample counts) are rejected."""

from __future__ import annotations

from p2c.agents.phase1.extract_fingerprint_atomic import ExtractFingerprintAtomicAgent


def _make_agent():
    """Minimal agent instance for calling static/instance methods."""

    class FakeArtifacts:
        def read_json(self, _):
            return {}

        def write_json(self, *_):
            pass

    class FakeLLM:
        def chat_text(self, *_):
            return ""

    return ExtractFingerprintAtomicAgent(
        llm=FakeLLM(), artifacts=FakeArtifacts(), step_index=1, step_total=1
    )


# ---- Classification report table (sklearn-style) ----

CLF_REPORT_HTML = """
<table>
<tr><th>class</th><th>precision</th><th>recall</th><th>f1-score</th><th>support</th></tr>
<tr><td>0</td><td>0.9702</td><td>0.9130</td><td>0.9408</td><td>1000</td></tr>
<tr><td>1</td><td>0.9239</td><td>0.9830</td><td>0.9525</td><td>1000</td></tr>
<tr><td>2</td><td>0.9607</td><td>0.9730</td><td>0.9668</td><td>1000</td></tr>
<tr><td>accuracy</td><td></td><td></td><td>0.9685</td><td>10000</td></tr>
<tr><td>macro avg</td><td>0.9692</td><td>0.9685</td><td>0.9684</td><td>10000</td></tr>
<tr><td>weighted avg</td><td>0.9692</td><td>0.9685</td><td>0.9684</td><td>10000</td></tr>
</table>
"""


def test_clf_report_rejects_support_column():
    """The 'support' column (sample counts like 10000) must NOT become metric values."""
    agent = _make_agent()
    unit = {"unit_id": "table_1", "text": CLF_REPORT_HTML, "type": "table_block"}
    accepted, rejected = agent._expand_table_unit(unit)

    # No accepted metric should have value >= 100 (those are sample counts)
    for item in accepted:
        assert item["metric_value"] <= 100.0, (
            f"Implausible metric value: {item['fact']} — likely a sample count"
        )

    # The accuracy row should produce a valid value around 0.9685
    accuracy_items = [a for a in accepted if "accuracy" in a.get("metric_name", "")]
    assert accuracy_items, "Should extract at least one accuracy metric from the clf report"

    for item in accuracy_items:
        assert 0.0 <= item["metric_value"] <= 1.0 or (0.0 <= item["metric_value"] <= 100.0), (
            f"Accuracy value {item['metric_value']} is out of plausible range"
        )


def test_clf_report_no_numeric_entity_names():
    """Entity/model names should NOT be numeric strings like '1000' or '10000'."""
    agent = _make_agent()
    unit = {"unit_id": "table_1", "text": CLF_REPORT_HTML, "type": "table_block"}
    accepted, _ = agent._expand_table_unit(unit)

    for item in accepted:
        entity = item.get("entity") or ""
        if entity:
            # Entity should not be a plain number (that would be a data value, not a model name)
            assert not entity.replace(".", "").isdigit(), (
                f"Entity '{entity}' looks like a number, not a model/column name"
            )


# ---- Standard comparison table (should still work) ----

STANDARD_TABLE_HTML = """
<table>
<tr><th>Method</th><th>Accuracy</th><th>F1</th></tr>
<tr><td>Baseline</td><td>92.3%</td><td>0.91</td></tr>
<tr><td>Ours</td><td>96.8%</td><td>0.95</td></tr>
</table>
"""


def test_standard_table_extraction():
    """Standard comparison tables (Method | Accuracy | F1) should still work correctly."""
    agent = _make_agent()
    unit = {"unit_id": "table_2", "text": STANDARD_TABLE_HTML, "type": "table_block"}
    accepted, rejected = agent._expand_table_unit(unit)

    # Should extract metrics from the data rows
    assert len(accepted) >= 2, f"Expected at least 2 metrics, got {len(accepted)}"

    facts = [a["fact"] for a in accepted]
    # Should have entries for Baseline and Ours
    baseline_facts = [f for f in facts if "Baseline" in f or "baseline" in f.lower()]
    ours_facts = [f for f in facts if "Ours" in f or "ours" in f.lower()]
    assert baseline_facts or ours_facts, f"Expected model-specific facts, got: {facts}"


# ---- Value plausibility ----

def test_plausibility_rejects_large_accuracy():
    """accuracy=10000 should be rejected as implausible."""
    assert not ExtractFingerprintAtomicAgent._is_plausible_metric_value(10000.0, "accuracy")
    assert not ExtractFingerprintAtomicAgent._is_plausible_metric_value(1024.0, "f1")
    assert not ExtractFingerprintAtomicAgent._is_plausible_metric_value(128.0, "precision")


def test_plausibility_accepts_valid_values():
    """Normal metric values should pass plausibility check."""
    assert ExtractFingerprintAtomicAgent._is_plausible_metric_value(96.85, "accuracy")
    assert ExtractFingerprintAtomicAgent._is_plausible_metric_value(0.9685, "accuracy")
    assert ExtractFingerprintAtomicAgent._is_plausible_metric_value(0.95, "f1")
    assert ExtractFingerprintAtomicAgent._is_plausible_metric_value(2.34, "loss")


# ---- build_claims_ir target extraction ----

def test_claims_ir_rejects_implausible_target():
    """build_claims_ir should not create target=10000 for accuracy."""
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    metric, target = BuildClaimsIRAgent._extract_metric_and_target("1000 accuracy = 10000")
    assert metric == "accuracy"
    assert target is None, f"Target should be None for implausible value, got {target}"


def test_claims_ir_extracts_valid_percentage():
    """Normal percentage targets should still work."""
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    metric, target = BuildClaimsIRAgent._extract_metric_and_target("accuracy = 96.85%")
    assert metric == "accuracy"
    assert abs(target - 0.9685) < 0.001


def test_claims_ir_extracts_valid_decimal():
    """Decimal ratio targets should still work."""
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    metric, target = BuildClaimsIRAgent._extract_metric_and_target("f1 = 0.95")
    assert metric == "f1"
    assert abs(target - 0.95) < 0.001
