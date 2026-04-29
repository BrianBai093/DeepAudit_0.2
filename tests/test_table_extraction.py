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


class MemoryArtifacts:
    def __init__(self, initial: dict[str, dict] | None = None) -> None:
        self.store = initial or {}

    def read_json(self, relative: str) -> dict:
        return self.store.get(relative, {})

    def write_json(self, relative: str, payload: dict) -> None:
        self.store[relative] = payload

    def append_text(self, relative: str, content: str) -> None:
        current = self.store.get(relative, {}).get("text", "")
        self.store[relative] = {"text": current + content}


class TableFallbackLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, *, schema, system: str, user: str):
        self.calls += 1
        assert "Table 1" in user
        return {
            "criteria": [
                {
                    "fact": "EP symmetric test error = 12.45%",
                    "scope": "CIFAR-10, Table 1, Squared Error loss, symmetric gradient estimate",
                    "facet": "metric_result",
                    "metric_name": "test error",
                    "metric_value": 12.45,
                    "metric_unit": "%",
                    "entity": "EP symmetric",
                    "table_anchor": "Table 1",
                }
            ]
        }

    def chat_text(self, *, system: str, user: str) -> str:
        return ""


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

PEPITA_TABLE_HTML = """
Table 1. Test accuracy [%] achieved by BP, FA, DRTP and PEPITA in the experiments.
<table>
<tr><td></td><td colspan="3">FULLY CONNECTED MODELS</td><td colspan="3">CONVOLUTIONAL MODELS</td></tr>
<tr><td></td><td>MNIST</td><td>CIFAR10</td><td>CIFAR100</td><td>MNIST</td><td>CIFAR10</td><td>CIFAR100</td></tr>
<tr><td>BP</td><td>98.63±0.03</td><td>55.27±0.32</td><td>27.58±0.09</td><td>98.86±0.04</td><td>64.99±0.32</td><td>34.20±0.20</td></tr>
<tr><td>PEPITA</td><td>98.01±0.09</td><td>52.57±0.36</td><td>24.91±0.22</td><td>98.29±0.13</td><td>56.33±1.35</td><td>27.56±0.60</td></tr>
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


def test_caption_metric_matrix_extracts_mean_std_cells_with_scope():
    agent = _make_agent()
    unit = {"unit_id": "table_pepita", "text": PEPITA_TABLE_HTML, "type": "table_block"}
    accepted, rejected = agent._expand_table_unit(unit)

    assert not rejected
    assert len(accepted) == 12

    bp_mnist = next(
        item
        for item in accepted
        if item["entity"] == "BP"
        and item["dataset_scope"] == "MNIST"
        and "fully connected" in item["scope"].lower()
    )
    assert bp_mnist["fact"] == "BP accuracy = 98.63±0.03"
    assert bp_mnist["metric_name"] == "accuracy"
    assert bp_mnist["metric_unit"] == "%"

    conv = next(
        item
        for item in accepted
        if item["entity"] == "PEPITA"
        and item["dataset_scope"] == "CIFAR10"
        and "convolutional" in item["scope"].lower()
    )
    assert conv["fact"] == "PEPITA accuracy = 56.33±1.35"


def test_table_block_llm_fallback_when_deterministic_parser_extracts_zero():
    table_html = """
    <table>
    <tr><td></td><td colspan="2">Equilibrium Propagation Error (%)</td></tr>
    <tr><td>Gradient Estimate</td><td>Test</td><td>Train</td></tr>
    <tr><td>Symmetric</td><td>12.45 (0.18)</td><td>7.83</td></tr>
    </table>
    """
    artifacts = MemoryArtifacts(
        {
            "fingerprint/guide_sentences.json": {
                "units": [
                    {
                        "unit_id": "t_0",
                        "type": "table_block",
                        "text": "Table 1: CIFAR-10 results.\n" + table_html,
                        "caption": "Table 1: CIFAR-10 results.",
                    }
                ],
                "selected_unit_ids": ["t_0"],
            },
            "fingerprint/visual_elements.json": {
                "elements": [
                    {
                        "element_id": "table_1",
                        "element_type": "table",
                        "visual_anchor": "Table 1",
                        "caption": "Table 1: CIFAR-10 results.",
                        "chart_type": "table",
                        "data_series": [],
                    }
                ]
            },
        }
    )
    llm = TableFallbackLLM()
    agent = ExtractFingerprintAtomicAgent(llm=llm, artifacts=artifacts, step_index=1, step_total=1)

    result = agent.execute({})

    criteria = result["atomic_criteria"]["criteria"]
    assert llm.calls == 1
    assert len(criteria) == 1
    assert criteria[0]["source_type"] == "llm_table_metric"
    assert criteria[0]["metric_name"] == "test error"
    assert criteria[0]["table_anchor"] == "Table 1"


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


def test_claims_ir_extracts_mean_as_target_and_std_as_tolerance():
    """mean±std should use the mean target, not the std value."""
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    parsed = BuildClaimsIRAgent._extract_metric_target_stats("accuracy = 98.63±0.03")

    assert parsed["metric"] == "accuracy"
    assert abs(parsed["target"] - 0.9863) < 1e-6
    assert abs(parsed["std_eps"] - 0.0003) < 1e-9
    assert "MEAN_STD_TARGET_NORMALIZED" in parsed["reason_codes"]


def test_claims_ir_extracts_valid_decimal():
    """Decimal ratio targets should still work."""
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    metric, target = BuildClaimsIRAgent._extract_metric_and_target("f1 = 0.95")
    assert metric == "f1"
    assert abs(target - 0.95) < 0.001


def test_fingerprint_filter_caps_claims_with_result_priority():
    from p2c.agents.phase1.extract_fingerprint_filter import ExtractFingerprintFilterAgent

    criteria = [
        {"facet": "execution_param", "fact": f"batch size {idx}", "scope": "setup"}
        for idx in range(4)
    ] + [
        {
            "facet": "metric_result",
            "fact": f"accuracy {90 + idx}%",
            "scope": "test",
            "metric_name": "accuracy",
            "metric_value": 90 + idx,
        }
        for idx in range(4)
    ]

    kept, dropped = ExtractFingerprintFilterAgent._limit_selected_indices(
        criteria,
        list(range(len(criteria))),
        max_claims=3,
    )

    assert dropped == 5
    assert len(kept) == 3
    assert all(criteria[idx]["facet"] == "metric_result" for idx in kept)
