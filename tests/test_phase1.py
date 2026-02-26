from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent
from p2c.agents.phase1.compile_task_spec import CompileTaskSpecAgent
from p2c.agents.phase1.ingest_paper import IngestPaperAgent
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient
from p2c.schemas import ClaimsIR


def _mk_artifacts(tmp_path: Path) -> ArtifactManager:
    manager = ArtifactManager(tmp_path / "artifacts", "run_test")
    manager.ensure_tree()
    return manager


def test_ingest_replaces_images_and_writes_paper_md(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    full_md = tmp_path / "full.md"
    full_md.write_text("# Title\n\nhello\n\n![alt text](images/a.png)\n", encoding="utf-8")
    paper_md_out = tmp_path / "output" / "paper.md"

    agent = IngestPaperAgent(llm=LLMClient(), artifacts=artifacts, step_index=1, step_total=14)
    agent.run({"paper_md": str(full_md), "paper_md_out": str(paper_md_out)})

    assert paper_md_out.exists()
    text = paper_md_out.read_text(encoding="utf-8")
    assert "[ImageDescription]" in text
    assert "- source: images/a.png" in text
    assert "- alt: alt text" in text


def test_ingest_does_not_write_paper_text_or_citations(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    full_md = tmp_path / "full.md"
    full_md.write_text("# Title\n\n![x](img.png)\n", encoding="utf-8")
    paper_md_out = tmp_path / "output" / "paper.md"

    agent = IngestPaperAgent(llm=LLMClient(), artifacts=artifacts, step_index=1, step_total=14)
    agent.run({"paper_md": str(full_md), "paper_md_out": str(paper_md_out)})

    assert not artifacts.path("paper/paper_text.json").exists()
    assert not artifacts.path("paper/citations.json").exists()


def test_claims_ir_schema_valid(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/fingerprint.json",
        {
            "fingerprint_id": "fp1",
            "metadata": {
                "paper_id": "fp1",
                "repository_url": None,
                "venue": None,
                "year": 2024,
            },
            "configurations": {
                "dataset_specs": [],
                "hyperparameters": {},
                "model_arch": [],
                "environment": {},
                "evaluation_metrics": ["accuracy"],
            },
            "claims": [
                {
                    "id": "claim_01",
                    "claim_type": "Empirical",
                    "fact": "Accuracy = 83.3%",
                    "scope": "test set",
                    "comparator": None,
                    "verification_logic": "exact_match",
                    "tolerance": {"abs": 0.02, "rel": 0.03, "text": None},
                    "evidence_anchors": {
                        "text_anchor": "Section 3",
                        "visual_anchor": None,
                        "visual_data": {},
                    },
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
            "notes": None,
        },
    )

    agent = BuildClaimsIRAgent(llm=LLMClient(), artifacts=artifacts, step_index=5, step_total=14)
    agent.run({"paper_md_out": "unused.md"})

    payload = artifacts.read_json("fingerprint/claims_ir.json")
    parsed = ClaimsIR(**payload)
    assert parsed.claims
    assert all(c.claim_id for c in parsed.claims)
    assert parsed.reason_codes == ["SOURCE_FINGERPRINT_CLAIMS"]


def test_task_spec_has_entrypoints_and_observers(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "type": "absolute",
                    "predicate": "accuracy claim",
                    "metric": "accuracy",
                    "target": 0.8,
                    "baseline": None,
                    "conditions": {},
                    "aggregation": "best",
                    "evidence_set": ["paper"],
                    "tolerance_policy": {"abs_eps": 0.02, "rel_eps": 0.03},
                    "unverifiable_from_paper": False,
                    "reason_codes": [],
                    "notes": None,
                }
            ],
            "reason_codes": [],
        },
    )

    repo_dir = Path("Target/code").resolve()
    agent = CompileTaskSpecAgent(llm=LLMClient(), artifacts=artifacts, step_index=6, step_total=14)
    agent.run({"repo_dir": str(repo_dir), "budget_minutes": 5, "max_self_heal_iters": 2})

    task_spec = artifacts.read_json("task/task_spec.json")
    assert len(task_spec["entrypoints"]) <= 5
    assert len(task_spec["metric_observers"]) >= 1
