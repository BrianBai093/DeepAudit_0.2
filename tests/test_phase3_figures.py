from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.reproduce_figures import ReproduceFiguresAgent
from p2c.io_artifacts import ArtifactManager


def test_reproduce_figures_writes_verdict_chart_with_absolute_save_path(tmp_path: Path) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_figures")
    artifacts.ensure_tree()
    artifacts.write_json(
        "results/verdict.json",
        {
            "status": "PARTIALLY_SUPPORTED",
            "claim_verdicts": [
                {
                    "claim_id": "claim_01",
                    "status": "SUPPORTED",
                    "detail": "ok",
                    "compared_value": 0.91,
                    "target_value": 0.9,
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json("results/metrics.json", {"records": [], "reason_codes": []})
    artifacts.write_json("fingerprint/visual_elements.json", {"elements": [], "reason_codes": []})

    agent = ReproduceFiguresAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({})

    fig = result["figures"]["figures"][0]
    assert fig["element_id"] == "verdict_comparison"
    assert fig["image_path"] == "results/figures/verdict_comparison.png"
    assert artifacts.path(fig["image_path"]).exists()
    assert artifacts.path(fig["image_path"]).stat().st_size > 0


def test_reproduce_figures_deterministic_visual_fallback_when_llm_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_dir = tmp_path / "repo"
    metrics_dir = repo_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "threshold_metrics.csv").write_text(
        "threshold,precision,recall,f1,tp,fp,fn,tn\n"
        "0.5,0.25,0.90,0.39,90,10,10,90\n"
        "0.9,0.50,0.80,0.62,80,4,20,96\n",
        encoding="utf-8",
    )

    artifacts = ArtifactManager(tmp_path / "artifacts", "run_visual_fallback")
    artifacts.ensure_tree()
    artifacts.write_json("results/verdict.json", {"status": "INCONCLUSIVE", "claim_verdicts": []})
    artifacts.write_json(
        "results/metrics.json",
        {
            "records": [
                {
                    "metric_name": "roc_auc",
                    "value": 0.98,
                    "source": "run_manifest",
                    "parsed": True,
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "claims": [
                {
                    "claim_id": "claim_02",
                    "type": "result",
                    "predicate": "true positive rate = 0.9",
                    "metric": "true positive rate",
                    "target": 0.9,
                    "conditions": {
                        "table_anchor": "Figure 2",
                        "visual_data": {"element_id": "fig_2"},
                    },
                }
            ],
            "experiments": [],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "fingerprint/visual_elements.json",
        {
            "elements": [
                {
                    "element_id": "fig_2",
                    "chart_type": "line",
                    "caption": "Figure 2. ROC curve",
                    "axis_labels": {"x": "False Positive Rate", "y": "True Positive Rate"},
                    "data_series": [
                        {
                            "name": "ROC curve",
                            "values": [
                                {"x": 0.0, "y": 0.0},
                                {"x": 0.1, "y": 0.9},
                                {"x": 1.0, "y": 1.0},
                            ],
                        }
                    ],
                }
            ],
            "reason_codes": [],
        },
    )

    agent = ReproduceFiguresAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, "no key"))

    result = agent.execute({"repo_dir": str(repo_dir)})
    figures = {row["element_id"]: row for row in result["figures"]["figures"]}

    assert figures["fig_2"]["image_path"] == "results/figures/fig_2.png"
    assert "DETERMINISTIC_FALLBACK" in figures["fig_2"]["reason_codes"]
    assert "Repo threshold sweep" in figures["fig_2"]["matplotlib_code"]
    assert "threshold_metrics.csv" in figures["fig_2"]["matplotlib_code"]
    assert artifacts.path(figures["fig_2"]["image_path"]).exists()
    assert artifacts.path(figures["fig_2"]["image_path"]).stat().st_size > 0
