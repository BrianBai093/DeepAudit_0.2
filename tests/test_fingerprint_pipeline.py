from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.extract_fingerprint_atomic import ExtractFingerprintAtomicAgent
from p2c.agents.phase1.extract_fingerprint_filter import ExtractFingerprintFilterAgent
from p2c.agents.phase1.extract_fingerprint_guide import ExtractFingerprintGuideAgent
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient
from p2c.schemas import Fingerprint


def _mk_artifacts(tmp_path: Path) -> ArtifactManager:
    manager = ArtifactManager(tmp_path / "artifacts", "run_test")
    manager.ensure_tree()
    return manager


def _sample_paper(tmp_path: Path) -> Path:
    p = tmp_path / "output" / "paper.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# Paper\n"
        "We use UCI heart disease dataset with a 66/34 train-test split. "
        "The optimizer is Adam with learning rate 0.001 and batch size 32. "
        "Future work may explore more hospitals.\n"
        "TABLE II TEST ACCURACIES FOR POOLED DATA AND VARIOUS LEARNING MODELS.\n"
        "<table><tr><td>Test</td><td>LR</td><td>SVM</td></tr><tr><td>Accuracy (%)</td><td>80.5</td><td>83.3</td></tr></table>\n",
        encoding="utf-8",
    )
    return p


def test_guide_selects_executable_units_only(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    paper = _sample_paper(tmp_path)

    agent = ExtractFingerprintGuideAgent(llm=LLMClient(), artifacts=artifacts, step_index=2, step_total=14)
    agent.run({"paper_md_out": str(paper)})

    payload = artifacts.read_json("fingerprint/guide_sentences.json")
    selected = set(payload.get("selected_unit_ids", []))
    units = {u["unit_id"]: u for u in payload.get("units", [])}

    assert selected
    assert any(units[uid]["type"] == "table_block" for uid in selected)
    # The "future work" sentence should be filtered out.
    assert not any("future work" in units[uid]["text"].lower() for uid in selected)


def test_guide_preserves_table_blocks(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    paper = _sample_paper(tmp_path)

    agent = ExtractFingerprintGuideAgent(llm=LLMClient(), artifacts=artifacts, step_index=2, step_total=14)
    agent.run({"paper_md_out": str(paper)})

    payload = artifacts.read_json("fingerprint/guide_sentences.json")
    units = payload.get("units", [])
    table_units = [u for u in units if u.get("type") == "table_block"]
    assert table_units
    selected = set(payload.get("selected_unit_ids", []))
    assert any(u.get("unit_id") in selected for u in table_units)


def test_atomic_outputs_fact_scope_and_structured_fields(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/guide_sentences.json",
        {
            "sentence_count": 2,
            "sentences": [
                "We use Adam optimizer with learning rate 0.001.",
                "Accuracy is 83.3% on test set.",
            ],
            "unit_count": 2,
            "units": [
                {
                    "unit_id": "s_0",
                    "type": "sentence",
                    "text": "We use Adam optimizer with learning rate 0.001.",
                    "origin_indices": [0],
                },
                {
                    "unit_id": "s_1",
                    "type": "sentence",
                    "text": "Accuracy is 83.3% on test set.",
                    "origin_indices": [1],
                },
            ],
            "selected_unit_ids": ["s_0", "s_1"],
            "selected_sentence_indices": [0, 1],
            "reason_codes": [],
        },
    )

    agent = ExtractFingerprintAtomicAgent(llm=LLMClient(), artifacts=artifacts, step_index=3, step_total=14)
    agent.run({})

    payload = artifacts.read_json("fingerprint/atomic_criteria.json")
    assert payload.get("criteria")
    for row in payload["criteria"]:
        assert row.get("criterion")
        assert row.get("fact")
        assert row.get("scope")
        assert row.get("facet")
        assert row.get("source_type")
    assert any(r.get("metric_name") == "accuracy" for r in payload["criteria"])


def test_atomic_rejects_malformed_numeric_noise(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/guide_sentences.json",
        {
            "sentence_count": 1,
            "sentences": ["top testing accuracy is 3 %$"],
            "unit_count": 1,
            "units": [
                {
                    "unit_id": "s_0",
                    "type": "sentence",
                    "text": "top testing accuracy is 3 %$",
                    "origin_indices": [0],
                }
            ],
            "selected_unit_ids": ["s_0"],
            "selected_sentence_indices": [0],
            "reason_codes": [],
        },
    )

    agent = ExtractFingerprintAtomicAgent(llm=LLMClient(), artifacts=artifacts, step_index=3, step_total=14)
    agent.run({})

    rejected = artifacts.read_json("fingerprint/atomic_rejected.json")
    assert rejected.get("rejected")
    assert any("MALFORMED_NUMERIC" in item.get("reason_codes", []) for item in rejected["rejected"])


def test_filter_strong_semantic_dedup_removes_paraphrase_duplicates(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    paper = _sample_paper(tmp_path)
    artifacts.write_json(
        "fingerprint/atomic_criteria.json",
        {
            "criteria": [
                {
                    "criterion": "<fact>accuracy = 83.3%</fact> <scope>test set</scope>",
                    "fact": "accuracy = 83.3%",
                    "scope": "test set",
                    "facet": "metric",
                    "source_type": "text_metric",
                    "metric_name": "accuracy",
                    "metric_value": 83.3,
                    "metric_unit": "%",
                    "entity": "SVM",
                    "comparator": None,
                    "dataset_scope": "test set",
                    "reason_codes": [],
                },
                {
                    "criterion": "<fact>test accuracy reaches 83.3%</fact> <scope>on test set</scope>",
                    "fact": "test accuracy reaches 83.3%",
                    "scope": "on test set",
                    "facet": "metric",
                    "source_type": "text_metric",
                    "metric_name": "accuracy",
                    "metric_value": 83.3,
                    "metric_unit": "%",
                    "entity": "SVM",
                    "comparator": None,
                    "dataset_scope": "test set",
                    "reason_codes": [],
                },
            ],
            "selected_unit_ids": ["s_0"],
            "reason_codes": [],
        },
    )

    agent = ExtractFingerprintFilterAgent(llm=LLMClient(), artifacts=artifacts, step_index=4, step_total=14)
    agent.run({"paper_md_out": str(paper)})

    payload = artifacts.read_json("fingerprint/fingerprint.json")
    assert len(payload["claims"]) == 1


def test_filter_preserves_all_table_metric_rows(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    paper = _sample_paper(tmp_path)
    artifacts.write_json(
        "fingerprint/atomic_criteria.json",
        {
            "criteria": [
                {
                    "criterion": "<fact>LR accuracy = 80.5%</fact> <scope>from table in paper</scope>",
                    "fact": "LR accuracy = 80.5%",
                    "scope": "from table in paper",
                    "facet": "metric",
                    "source_type": "table_metric",
                    "metric_name": "accuracy",
                    "metric_value": 80.5,
                    "metric_unit": "%",
                    "entity": "LR",
                    "comparator": None,
                    "dataset_scope": None,
                    "table_anchor": "Table II",
                    "reason_codes": ["TABLE_EXPANDED"],
                },
                {
                    "criterion": "<fact>SVM accuracy = 83.3%</fact> <scope>from table in paper</scope>",
                    "fact": "SVM accuracy = 83.3%",
                    "scope": "from table in paper",
                    "facet": "metric",
                    "source_type": "table_metric",
                    "metric_name": "accuracy",
                    "metric_value": 83.3,
                    "metric_unit": "%",
                    "entity": "SVM",
                    "comparator": None,
                    "dataset_scope": None,
                    "table_anchor": "Table II",
                    "reason_codes": ["TABLE_EXPANDED"],
                },
            ],
            "selected_unit_ids": ["t_0"],
            "reason_codes": [],
        },
    )

    agent = ExtractFingerprintFilterAgent(llm=LLMClient(), artifacts=artifacts, step_index=4, step_total=14)
    agent.run({"paper_md_out": str(paper)})

    payload = artifacts.read_json("fingerprint/fingerprint.json")
    assert len(payload["claims"]) == 2
    assert all(c.get("evidence_anchors", {}).get("visual_anchor") == "Table II" for c in payload["claims"])


def test_fingerprint_claim_type_distribution_is_reasonable(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    payload = {
        "fingerprint_id": "arXiv:2401.12345",
        "metadata": {
            "paper_id": "arXiv:2401.12345",
            "repository_url": "github.com/org/repo",
            "venue": "ICLR",
            "year": 2024,
        },
        "configurations": {
            "dataset_specs": [{"name": "UCI"}],
            "hyperparameters": {"learning_rate": 0.001},
            "model_arch": ["SVM"],
            "environment": {"framework": "sklearn"},
            "evaluation_metrics": ["Accuracy"],
        },
        "claims": [
            {
                "id": "claim_01",
                "claim_type": "Empirical",
                "fact": "Accuracy = 83.3%",
                "scope": "test set",
                "comparator": None,
                "verification_logic": "exact_match",
                "tolerance": {"abs": 0.005, "rel": 0.02, "text": "±0.5%"},
                "evidence_anchors": {
                    "text_anchor": "Section 3",
                    "visual_anchor": "Table 1",
                    "visual_data": {},
                },
                "reason_codes": [],
            }
        ],
        "reason_codes": [],
        "notes": None,
    }
    artifacts.write_json("fingerprint/fingerprint.json", payload)

    parsed = Fingerprint(**artifacts.read_json("fingerprint/fingerprint.json"))
    unknown = [c for c in parsed.claims if c.claim_type == "Unknown"]
    ratio = (len(unknown) / len(parsed.claims)) if parsed.claims else 0.0
    assert ratio < 0.1
