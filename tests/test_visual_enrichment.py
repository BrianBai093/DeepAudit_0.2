from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.enrich_claims_visual import EnrichClaimsVisualAgent
from p2c.agents.phase1.extract_fingerprint_filter import ExtractFingerprintFilterAgent
from p2c.agents.phase1.extract_visual_elements import resolve_existing_pdf_path
from p2c.schemas import AtomicCriterion


class DummyLLM:
    def chat_json(self, schema, system: str, user: str):
        return {}

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
    assert result["enriched_count"] == 1
    assert result["new_from_figures"] == 1
    assert "accepted" not in artifacts.store["fingerprint/atomic_criteria.json"]
    assert criteria[0]["visual_data"]["element_id"] == "fig_1"
    assert criteria[1]["source_type"] == "visual_metric"
    assert criteria[1]["table_anchor"] == "Figure 1"


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
    assert claim["evidence_anchors"]["visual_data"] == visual_data


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
