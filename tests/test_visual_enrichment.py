from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent
from p2c.agents.phase1.enrich_claims_visual import EnrichClaimsVisualAgent
from p2c.agents.phase1.extract_fingerprint_filter import ExtractFingerprintFilterAgent
from p2c.agents.phase1.extract_visual_elements import resolve_existing_pdf_path
from p2c.schemas import AtomicCriterion


class DummyLLM:
    def chat_json(self, schema, system: str, user: str):
        return {}

    def chat_text(self, system: str, user: str) -> str:
        return ""


class VisualTableLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, schema, system: str, user: str):
        self.calls += 1
        assert "Table 1" in user
        return {
            "criteria": [
                {
                    "fact": "EP symmetric test error = 12.45%",
                    "scope": "CIFAR-10, Table 1",
                    "facet": "metric_result",
                    "metric_name": "test error",
                    "metric_value": 12.45,
                    "metric_unit": "%",
                    "entity": "EP symmetric",
                    "table_anchor": "Table 1",
                }
            ]
        }

    def chat_text(self, system: str, user: str) -> str:
        return ""


class MemoryArtifacts:
    def __init__(self, initial: dict[str, dict] | None = None) -> None:
        self.store = initial or {}

    def read_json(self, relative: str) -> dict:
        return self.store.get(relative, {})

    def write_json(self, relative: str, payload: dict):
        self.store[relative] = payload

    def append_text(self, relative: str, content: str):
        self.store[relative] = {"text": self.store.get(relative, {}).get("text", "") + content}


def test_resolve_existing_pdf_path_falls_back_to_current_origin_pdf(tmp_path: Path) -> None:
    paper_dir = tmp_path / "Target" / "paper"
    paper_dir.mkdir(parents=True)
    requested = paper_dir / "stale_origin.pdf"
    actual = paper_dir / "current_origin.pdf"
    actual.write_bytes(b"%PDF-1.4\n")

    original, resolved = resolve_existing_pdf_path(
        requested,
        {"paper_md_out": str(paper_dir / "full.md")},
    )

    assert original == requested
    assert resolved == actual


def test_enrich_claims_visual_uses_current_criteria_schema() -> None:
    artifacts = MemoryArtifacts(
        {
            "fingerprint/visual_elements.json": {
                "elements": [
                    {
                        "element_id": "fig_1",
                        "element_type": "figure",
                        "visual_anchor": "Figure 1",
                        "chart_type": "bar",
                        "axis_labels": {"x": "split", "y": "accuracy"},
                        "legend_entries": ["Model A"],
                        "data_series": [
                            {
                                "name": "Model A",
                                "values": [{"x": "test", "y": 0.95}],
                            }
                        ],
                    }
                ]
            },
            "fingerprint/atomic_criteria.json": {
                "criteria": [
                    {
                        "criterion": "<fact>accuracy = 0.90</fact> <scope>from Figure 1</scope>",
                        "fact": "accuracy = 0.90",
                        "scope": "from Figure 1",
                        "facet": "metric_result",
                        "source_type": "text_metric",
                        "metric_name": "accuracy",
                        "metric_value": 0.90,
                        "metric_unit": "ratio",
                        "reason_codes": [],
                    }
                ],
                "reason_codes": [],
            },
        }
    )
    agent = EnrichClaimsVisualAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=1)

    result = agent.execute({})

    criteria = artifacts.store["fingerprint/atomic_criteria.json"]["criteria"]
    visual_targets = artifacts.store["fingerprint/visual_targets.json"]["visual_targets"]
    assert result["enriched_count"] == 1
    assert result["new_from_figures"] == 0
    assert result["visual_targets"] == 1
    assert "accepted" not in artifacts.store["fingerprint/atomic_criteria.json"]
    assert criteria[0]["visual_data"]["element_id"] == "fig_1"
    assert len(criteria) == 1
    assert visual_targets[0]["element_id"] == "fig_1"
    assert visual_targets[0]["visual_anchor"] == "Figure 1"
    assert visual_targets[0]["series_names"] == ["Model A"]
    assert visual_targets[0]["metric_names"] == ["accuracy"]
    assert "FIGURE_POINTS_NOT_EXPANDED" in visual_targets[0]["reason_codes"]


def test_filter_propagates_visual_data_to_fingerprint() -> None:
    visual_data = {"element_id": "fig_1", "chart_type": "bar"}
    artifacts = MemoryArtifacts(
        {
            "fingerprint/atomic_criteria.json": {
                "criteria": [
                    {
                        "criterion": "<fact>accuracy = 0.95</fact> <scope>from Figure 1</scope>",
                        "fact": "accuracy = 0.95",
                        "scope": "from Figure 1",
                        "facet": "metric_result",
                        "source_type": "visual_metric",
                        "metric_name": "accuracy",
                        "metric_value": 0.95,
                        "metric_unit": "ratio",
                        "table_anchor": "Figure 1",
                        "visual_data": visual_data,
                        "reason_codes": ["VISUAL_FIGURE_EXTRACTED"],
                    }
                ]
            }
        }
    )
    agent = ExtractFingerprintFilterAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=1)

    agent.execute({})

    claim = artifacts.store["fingerprint/fingerprint.json"]["claims"][0]
    assert claim["evidence_anchors"]["visual_anchor"] == "Figure 1"
    assert claim["evidence_anchors"]["visual_data"] == {"element_id": "fig_1"}


def test_enrich_claims_visual_dedupes_against_metric_name_value() -> None:
    artifacts = MemoryArtifacts(
        {
            "fingerprint/visual_elements.json": {
                "elements": [
                    {
                        "element_id": "fig_1",
                        "element_type": "figure",
                        "visual_anchor": "Figure 1",
                        "chart_type": "bar",
                        "axis_labels": {"y": "accuracy"},
                        "data_series": [{"name": "Model A", "values": [{"x": "test", "y": 0.95}]}],
                    }
                ]
            },
            "fingerprint/atomic_criteria.json": {
                "criteria": [
                    {
                        "fact": "accuracy = 0.95",
                        "scope": "from text",
                        "facet": "metric_result",
                        "source_type": "text_metric",
                        "metric_name": "accuracy",
                        "metric_value": 0.95,
                        "reason_codes": [],
                    }
                ]
            },
        }
    )
    agent = EnrichClaimsVisualAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=1)

    result = agent.execute({})

    assert result["new_from_figures"] == 0
    assert len(artifacts.store["fingerprint/atomic_criteria.json"]["criteria"]) == 1


def test_build_claims_ir_links_visual_targets_to_claims() -> None:
    artifacts = MemoryArtifacts(
        {
            "fingerprint/fingerprint.json": {
                "claims": [
                    {
                        "id": "claim_01",
                        "claim_type": "result",
                        "fact": "accuracy = 0.95",
                        "scope": "from Figure 1",
                        "evidence_anchors": {
                            "text_anchor": "atomic_criteria[0]",
                            "visual_anchor": "Figure 1",
                            "visual_data": {"element_id": "fig_1", "chart_type": "bar"},
                        },
                        "reason_codes": [],
                    }
                ],
                "reason_codes": [],
            },
            "task/repo_analysis.json": {"entrypoint_candidates": [], "reason_codes": []},
            "fingerprint/visual_elements.json": {
                "elements": [
                    {
                        "element_id": "fig_1",
                        "element_type": "figure",
                        "page": 1,
                        "visual_anchor": "Figure 1",
                        "chart_type": "bar",
                        "associated_claim_ids": [],
                    }
                ],
                "reason_codes": [],
            },
            "fingerprint/visual_targets.json": {
                "visual_targets": [
                    {
                        "element_id": "fig_1",
                        "visual_anchor": "Figure 1",
                        "element_type": "figure",
                        "chart_type": "bar",
                        "caption": "",
                        "page": 1,
                        "reference_image_path": None,
                        "axis_labels": {},
                        "legend_entries": [],
                        "series_names": [],
                        "metric_names": [],
                        "model_names": [],
                        "sampling_strategy": None,
                        "semantic_summary": "",
                        "reconstruction_instructions": [],
                        "associated_claim_ids": [],
                        "reason_codes": ["OBJECT_LEVEL_VISUAL_TARGET"],
                    }
                ],
                "reason_codes": [],
            },
        }
    )
    agent = BuildClaimsIRAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=1)
    agent.safe_chat_json = lambda schema, system, user: (None, "offline")

    agent.execute({})

    claim = artifacts.store["fingerprint/claims_ir.json"]["claims"][0]
    visual_element = artifacts.store["fingerprint/visual_elements.json"]["elements"][0]
    visual_target = artifacts.store["fingerprint/visual_targets.json"]["visual_targets"][0]
    assert claim["conditions"]["visual_data"] == {"element_id": "fig_1"}
    assert visual_element["associated_claim_ids"] == ["claim_01"]
    assert visual_target["associated_claim_ids"] == ["claim_01"]


def test_visual_table_generates_criteria_even_when_atomic_is_empty() -> None:
    artifacts = MemoryArtifacts(
        {
            "fingerprint/visual_elements.json": {
                "elements": [
                    {
                        "element_id": "table_1",
                        "element_type": "table",
                        "visual_anchor": "Table 1",
                        "caption": "Table 1: Performance comparison on CIFAR-10.",
                        "chart_type": "table",
                        "data_series": [
                            {
                                "name": "Symmetric",
                                "values": [
                                    {
                                        "Gradient Estimate": "Symmetric",
                                        "EP Test Error (%)": "12.45 (0.18)",
                                        "EP Train Error (%)": "7.83",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            "fingerprint/atomic_criteria.json": {"criteria": [], "reason_codes": []},
        }
    )
    llm = VisualTableLLM()
    agent = EnrichClaimsVisualAgent(llm=llm, artifacts=artifacts, step_index=1, step_total=1)

    result = agent.execute({})

    criteria = artifacts.store["fingerprint/atomic_criteria.json"]["criteria"]
    assert llm.calls == 1
    assert result["new_from_tables"] == 1
    assert criteria[0]["source_type"] == "visual_table_metric"
    assert criteria[0]["metric_name"] == "test error"
    assert criteria[0]["visual_data"]["element_id"] == "table_1"


def test_visual_metric_is_valid_atomic_source_type() -> None:
    row = AtomicCriterion(
        criterion="<fact>accuracy = 0.95</fact> <scope>from Figure 1</scope>",
        fact="accuracy = 0.95",
        scope="from Figure 1",
        facet="metric_result",
        source_type="visual_metric",
        metric_name="accuracy",
        metric_value=0.95,
    )

    assert row.source_type == "visual_metric"


def test_visual_table_metric_is_valid_atomic_source_type() -> None:
    row = AtomicCriterion(
        criterion="<fact>test error = 12.45%</fact> <scope>from Table 1</scope>",
        fact="test error = 12.45%",
        scope="from Table 1",
        facet="metric_result",
        source_type="visual_table_metric",
        metric_name="test error",
        metric_value=12.45,
    )

    assert row.source_type == "visual_table_metric"
