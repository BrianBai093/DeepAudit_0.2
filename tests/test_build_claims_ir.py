from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient


def _mk_artifacts(tmp_path: Path) -> ArtifactManager:
    manager = ArtifactManager(tmp_path / "artifacts", "run_test")
    manager.ensure_tree()
    return manager


def test_build_claims_ir_prioritizes_empirical_and_comparative(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/fingerprint.json",
        {
            "fingerprint_id": "fp1",
            "metadata": {"paper_id": "fp1", "repository_url": None, "venue": None, "year": 2024},
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
                    "tolerance": {"abs": 0.005, "rel": 0.02, "text": None},
                    "evidence_anchors": {"text_anchor": "atomic_criteria[1]", "visual_anchor": None, "visual_data": {}},
                    "reason_codes": [],
                },
                {
                    "id": "claim_02",
                    "claim_type": "Methodological",
                    "fact": "Dataset has 920 records across hospitals",
                    "scope": "dataset description",
                    "comparator": None,
                    "verification_logic": "exact_match",
                    "tolerance": {"abs": None, "rel": None, "text": None},
                    "evidence_anchors": {"text_anchor": "atomic_criteria[2]", "visual_anchor": None, "visual_data": {}},
                    "reason_codes": [],
                },
            ],
            "reason_codes": [],
            "notes": None,
        },
    )

    agent = BuildClaimsIRAgent(llm=LLMClient(), artifacts=artifacts, step_index=5, step_total=14)
    agent.run({"paper_md_out": "unused.md"})

    payload = artifacts.read_json("fingerprint/claims_ir.json")
    claim_ids = [c.get("claim_id") for c in payload.get("claims", [])]
    assert claim_ids == ["claim_01"]
    assert "NON_EXECUTABLE_METHOD_CLAIMS_SKIPPED" in payload.get("reason_codes", [])


def test_build_claims_ir_keeps_executable_methodological_claims(tmp_path: Path) -> None:
    artifacts = _mk_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/fingerprint.json",
        {
            "fingerprint_id": "fp2",
            "metadata": {"paper_id": "fp2", "repository_url": None, "venue": None, "year": 2024},
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
                    "claim_type": "Methodological",
                    "fact": "optimizer Adam with learning rate 0.001 and batch size 32",
                    "scope": "training",
                    "comparator": None,
                    "verification_logic": "exact_match",
                    "tolerance": {"abs": None, "rel": None, "text": None},
                    "evidence_anchors": {"text_anchor": "atomic_criteria[1]", "visual_anchor": None, "visual_data": {}},
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
    assert len(payload.get("claims", [])) == 1
    assert payload["claims"][0]["claim_id"] == "claim_01"
