from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.visual_to_repo_alignment import VisualToRepoAlignmentAgent
from p2c.io_artifacts import ArtifactManager


def _run_alignment(tmp_path: Path, element: dict) -> dict:
    repo_dir = tmp_path / "repo"
    figures_dir = repo_dir / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "roc_xgboost.png").write_bytes(b"not inspected by alignment")

    artifacts = ArtifactManager(tmp_path / "artifacts", "run_alignment")
    artifacts.ensure_tree()
    artifacts.write_json(
        "fingerprint/visual_elements.json",
        {"elements": [element], "page_count": 1, "reason_codes": []},
    )

    agent = VisualToRepoAlignmentAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({"repo_dir": str(repo_dir)})
    return result["visual_to_repo_alignment"]["alignments"][0]


def test_autoencoder_roc_does_not_match_xgboost_roc_artifact(tmp_path: Path) -> None:
    row = _run_alignment(
        tmp_path,
        {
            "element_id": "fig_2",
            "element_type": "figure",
            "page": 1,
            "chart_type": "line",
            "caption": "Figure 2. Auto Encoder AUROC Curve",
            "axis_labels": {"x": "False Positive Rate", "y": "True Positive Rate"},
            "model_names": ["Auto Encoder"],
            "data_series": [],
        },
    )

    assert row["status"] == "NO_MATCH"
    assert row["repo_artifact_path"] is None
    assert any("model mismatch" in reason for reason in row["mismatch_reasons"])


def test_xgboost_roc_matches_xgboost_roc_artifact(tmp_path: Path) -> None:
    row = _run_alignment(
        tmp_path,
        {
            "element_id": "fig_xgb",
            "element_type": "figure",
            "page": 1,
            "chart_type": "line",
            "caption": "XGBoost ROC curve",
            "axis_labels": {"x": "False Positive Rate", "y": "True Positive Rate"},
            "model_names": ["XGBoost"],
            "data_series": [],
        },
    )

    assert row["status"] == "MATCH"
    assert row["artifact_type"] == "image"
    assert row["matched_model_names"] == ["xgboost"]
    assert row["matched_metric_names"] == ["roc_auc"]


def test_sampling_strategy_is_required_for_strict_visual_match(tmp_path: Path) -> None:
    row = _run_alignment(
        tmp_path,
        {
            "element_id": "fig_under",
            "element_type": "figure",
            "page": 1,
            "chart_type": "line",
            "caption": "XGBoost ROC curve after under-sampling",
            "axis_labels": {"x": "False Positive Rate", "y": "True Positive Rate"},
            "model_names": ["XGBoost"],
            "sampling_strategy": "under-sampling",
            "data_series": [],
        },
    )

    assert row["status"] == "NO_MATCH"
    assert any("sampling mismatch" in reason for reason in row["mismatch_reasons"])
