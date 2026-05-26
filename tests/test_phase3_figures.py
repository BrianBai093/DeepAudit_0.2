from __future__ import annotations

import base64
from pathlib import Path

from p2c.agents.phase3.execution_log_evidence import ExecutionLogEvidenceAgent
from p2c.agents.phase3.reproduce_figures import ReproduceFiguresAgent, _normalize_plot_spec, _render_plot_spec
from p2c.io_artifacts import ArtifactManager


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _artifacts(tmp_path: Path, run_id: str = "run_figures") -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path / "artifacts", run_id)
    artifacts.ensure_tree()
    artifacts.write_json("fingerprint/claims_ir.json", {"claims": [], "experiments": [], "reason_codes": []})
    artifacts.write_json("results/verdict.json", {"status": "INCONCLUSIVE", "claim_verdicts": [], "reason_codes": []})
    artifacts.write_json("results/metrics.json", {"records": [], "reason_codes": []})
    artifacts.write_json("results/parsed_evidence.json", {"claim_evidence": [], "reason_codes": []})
    return artifacts


def _write_reference(artifacts: ArtifactManager, name: str) -> str:
    rel = f"fingerprint/visual_crops/{name}.png"
    artifacts.path(rel).parent.mkdir(parents=True, exist_ok=True)
    artifacts.path(rel).write_bytes(_ONE_PIXEL_PNG)
    return rel


def _write_phase2_package(artifacts: ArtifactManager, *, stdout_text: str = "") -> None:
    stdout_rel = "execution/executor_outputs/experiment_exp_01_stdout.log"
    artifacts.write_text(stdout_rel, stdout_text)
    artifacts.write_json(
        "execution/executor_outputs/phase2_execution_package.json",
        {
            "schema_version": "phase2_execution_package.v1",
            "source_files": {},
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "MNIST BP benchmark",
                    "table_anchor": "Figure 1",
                    "paper_target_refs": ["Figure 1"],
                    "attempts": [
                        {
                            "attempt_id": "exp_01.bp.mnist.fc",
                            "experiment_id": "exp_01",
                            "experiment_name": "MNIST BP benchmark",
                            "scope": {"algorithm": "bp", "dataset": "mnist", "model_family": "fc", "epochs": 2},
                            "status": "ok",
                            "fidelity": "trend",
                            "execution_outcome": "TREND_SUPPORTED",
                            "evidence_source": "fresh_run",
                            "stop_reason": "complete",
                            "metrics": [
                                {
                                    "metric_name": "bp_mnist_fc_test_accuracy",
                                    "raw_metric_name": "test_accuracy",
                                    "value": 0.91,
                                    "value_ratio": 0.91,
                                    "unit": "ratio",
                                    "algorithm": "bp",
                                    "dataset": "mnist",
                                    "model_family": "fc",
                                    "fidelity": "trend",
                                    "source_attempt_id": "exp_01.bp.mnist.fc",
                                }
                            ],
                            "logs": {"stdout": stdout_rel},
                            "artifacts": [stdout_rel],
                            "stdout_tail": stdout_text[-1000:],
                            "reason_codes": [],
                        }
                    ],
                    "metrics": [],
                    "failures": [],
                    "logs": [],
                    "summary_for_llm": "BP MNIST FC attempt produced accuracy evidence.",
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_text("execution/executor_outputs/PHASE2_RESULTS.md", "MNIST BP accuracy evidence.")


def _write_single_target(
    artifacts: ArtifactManager,
    *,
    element_id: str = "fig_curve",
    chart_type: str = "line",
    caption: str = "Figure 1. MNIST BP test accuracy curve.",
    model_names: list[str] | None = None,
    reference: str | None = None,
) -> None:
    reference = reference or _write_reference(artifacts, element_id)
    target = {
        "element_id": element_id,
        "visual_anchor": "Figure 1",
        "element_type": "figure",
        "chart_type": chart_type,
        "caption": caption,
        "reference_image_path": reference,
        "axis_labels": {"x": "Epoch", "y": "Accuracy"},
        "legend_entries": [],
        "series_names": ["test accuracy"],
        "metric_names": ["test accuracy"],
        "model_names": model_names or ["BP"],
        "semantic_summary": caption,
        "reconstruction_instructions": [],
        "associated_claim_ids": [],
        "reason_codes": [],
    }
    artifacts.write_json("fingerprint/visual_targets.json", {"visual_targets": [target], "reason_codes": []})
    artifacts.write_json(
        "fingerprint/visual_elements.json",
        {
            "elements": [
                {
                    "element_id": element_id,
                    "element_type": "figure",
                    "page": 1,
                    "chart_type": chart_type,
                    "caption": caption,
                    "crop_path": reference,
                    "axis_labels": {"x": "Epoch", "y": "Accuracy"},
                    "model_names": model_names or ["BP"],
                    "data_series": [],
                }
            ],
            "reason_codes": [],
        },
    )


def test_llm_plot_spec_renders_line_bar_and_table(tmp_path: Path, monkeypatch) -> None:
    specs = {
        "line": {
            "decision": "PLOT",
            "chart_type": "line",
            "title": "LLM line",
            "x_label": "Epoch",
            "y_label": "Accuracy",
            "series": [{"name": "BP", "x": [1, 2], "y": [0.8, 0.9], "source": "phase2", "style": {}}],
            "table": {"columns": [], "rows": [], "source": None},
            "unit": "ratio",
            "normalization": None,
            "evidence_sources": ["phase2"],
            "comparison_note": "Line rendered from LLM spec.",
            "confidence": 0.9,
            "reason_codes": ["LLM_SPEC"],
        },
        "bar": {
            "decision": "PLOT",
            "chart_type": "bar",
            "title": "LLM bar",
            "x_label": "Dataset",
            "y_label": "Accuracy",
            "series": [{"name": "BP", "x": ["MNIST"], "y": [0.9], "source": "phase2", "style": {}}],
            "table": {"columns": [], "rows": [], "source": None},
            "unit": "ratio",
            "normalization": None,
            "evidence_sources": ["phase2"],
            "comparison_note": "Bar rendered from LLM spec.",
            "confidence": 0.9,
            "reason_codes": ["LLM_SPEC"],
        },
        "table": {
            "decision": "PLOT",
            "chart_type": "table",
            "title": "LLM table",
            "x_label": "",
            "y_label": "",
            "series": [],
            "table": {"columns": ["Metric", "Value"], "rows": [["accuracy", "0.90"]], "source": "phase2"},
            "unit": None,
            "normalization": None,
            "evidence_sources": ["phase2"],
            "comparison_note": "Table rendered from LLM spec.",
            "confidence": 0.9,
            "reason_codes": ["LLM_SPEC"],
        },
    }
    for chart_type, spec in specs.items():
        artifacts = _artifacts(tmp_path, f"run_{chart_type}")
        _write_phase2_package(
            artifacts,
            stdout_text="[1,  10] loss: 1.0\nTesting...\nTest accuracy: 80 %\n[2,  10] loss: 0.9\nTesting...\nTest accuracy: 90 %\n",
        )
        _write_single_target(artifacts, element_id=f"fig_{chart_type}", chart_type=chart_type)
        agent = ReproduceFiguresAgent(llm=object(), artifacts=artifacts, step_index=1, step_total=1)
        monkeypatch.setattr(agent, "safe_chat_json", lambda schema, system, user, spec=spec: (spec, None))

        result = agent.execute({})
        figure = result["figures"]["figures"][0]

        assert figure["reproduction_status"] == "REPRODUCED"
        assert figure["image_path"] == f"results/figures/fig_{chart_type}_comparison.png"
        assert figure["reference_image_path"].endswith(f"fig_{chart_type}.png")
        assert "phase2" in figure["evidence_sources"]
        assert artifacts.path(figure["image_path"]).exists()
        assert artifacts.path(figure["reproduced_image_path"]).exists()


def test_bar_renderer_accepts_data_point_series(tmp_path: Path) -> None:
    output_path = tmp_path / "bar_from_data.png"
    spec = {
        "decision": "PLOT",
        "chart_type": "bar",
        "title": "Final loss",
        "x_label": "Run",
        "y_label": "Final loss",
        "series": [
            {"name": "run_a", "data": [{"x": "run_a", "y": 0.146}]},
            {"name": "run_b", "data": [{"x": "run_b", "y": 0.288}]},
            {"name": "run_c", "data": [{"x": "run_c", "y": 1.525}]},
        ],
        "table": {"columns": [], "rows": [], "source": None},
        "unit": "raw",
        "normalization": None,
        "evidence_sources": ["phase2"],
        "comparison_note": "Bar rendered from data point series.",
        "confidence": 0.9,
        "reason_codes": [],
    }

    assert _render_plot_spec(spec, output_path)

    import matplotlib.image as mpimg

    image = mpimg.imread(output_path)
    blue_pixels = (
        (image[..., 2] > 0.45)
        & (image[..., 0] < 0.35)
        & (image[..., 1] < 0.65)
    )
    assert int(blue_pixels.sum()) > 3000


def test_llm_codegen_is_primary_even_for_supported_plot_spec(tmp_path: Path, monkeypatch) -> None:
    artifacts = _artifacts(tmp_path, "run_codegen_primary")
    _write_phase2_package(artifacts, stdout_text="[1,  10] loss: 1.0\nTesting...\nTest accuracy: 80 %\n")
    _write_single_target(artifacts)
    calls = {"count": 0}

    def fake_chat_json(schema, system, user):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                {
                    "decision": "PLOT",
                    "chart_type": "line",
                    "title": "Supported line",
                    "x_label": "Epoch",
                    "y_label": "Accuracy",
                    "series": [{"name": "BP", "x": [1, 2], "y": [0.8, 0.9], "source": "phase2", "style": {}}],
                    "table": {"columns": [], "rows": [], "source": None},
                    "unit": "ratio",
                    "normalization": None,
                    "evidence_sources": ["phase2"],
                    "comparison_note": "Uses primary codegen.",
                    "confidence": 0.9,
                    "reason_codes": ["LLM_SPEC"],
                },
                None,
            )
        return (
            {
                "code": (
                    "import matplotlib.pyplot as plt\n"
                    "fig, ax = plt.subplots(figsize=(4, 3))\n"
                    "ax.plot(payload['plot_spec']['series'][0]['x'], payload['plot_spec']['series'][0]['y'], marker='o')\n"
                    "ax.set_title('codegen primary')\n"
                    "fig.savefig(output_path, dpi=120, bbox_inches='tight')\n"
                )
            },
            None,
        )

    agent = ReproduceFiguresAgent(llm=object(), artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_json", fake_chat_json)

    result = agent.execute({})
    figure = result["figures"]["figures"][0]

    assert figure["reproduction_status"] == "REPRODUCED"
    assert "LLM_CODEGEN_RENDERED" in figure["reason_codes"]
    assert "DETERMINISTIC_RENDERER_FALLBACK_RENDERED" not in figure["reason_codes"]
    assert figure["code_path"] == "results/figures/fig_curve_codegen_primary_audit.json"
    assert artifacts.path(figure["reproduced_image_path"]).exists()
    assert not artifacts.path("results/figures/_codegen_tmp").exists()


def test_numeric_chart_spec_repairs_missing_series_from_table_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "line_repaired_to_bar.png"
    spec = _normalize_plot_spec(
        {
            "decision": "PLOT",
            "chart_type": "line",
            "title": "Related loss",
            "x_label": "Epoch",
            "y_label": "Loss",
            "series": [{"name": "run_a"}, {"name": "run_b"}],
            "table": {
                "columns": ["series", "final_loss"],
                "rows": [
                    {"series": "run_a", "final_loss": 0.146},
                    {"series": "run_b", "final_loss": 1.525},
                ],
            },
            "unit": "raw",
            "normalization": None,
            "evidence_sources": ["phase2"],
            "comparison_note": "Table rows contain the numeric evidence.",
            "match_level": "RELATED",
            "matched_scope": {},
            "coverage_note": "",
            "confidence": 0.6,
            "reason_codes": [],
        }
    )

    assert spec["chart_type"] == "bar"
    assert "REPAIRED_SERIES_FROM_TABLE_ROWS" in spec["reason_codes"]
    assert _render_plot_spec(spec, output_path)


def test_result_scope_skips_diagram_and_labels_bp_for_pepita_as_related(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, "run_skip")
    _write_phase2_package(artifacts, stdout_text="[1,  10] loss: 1.0\nTesting...\nTest accuracy: 80 %\n")
    ref_diagram = _write_reference(artifacts, "fig_diagram")
    ref_pepita = _write_reference(artifacts, "fig_pepita")
    artifacts.write_json(
        "fingerprint/visual_targets.json",
        {
            "visual_targets": [
                {
                    "element_id": "fig_diagram",
                    "visual_anchor": "Figure 1",
                    "element_type": "figure",
                    "chart_type": "diagram",
                    "caption": "Figure 1. Method diagram.",
                    "reference_image_path": ref_diagram,
                    "model_names": ["BP"],
                    "metric_names": [],
                    "series_names": [],
                },
                {
                    "element_id": "fig_pepita",
                    "visual_anchor": "Figure 2",
                    "element_type": "figure",
                    "chart_type": "line",
                    "caption": "Figure 2. PEPITA test accuracy curve on MNIST.",
                    "reference_image_path": ref_pepita,
                    "model_names": ["PEPITA"],
                    "metric_names": ["accuracy"],
                    "series_names": ["PEPITA"],
                },
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json("fingerprint/visual_elements.json", {"elements": [], "reason_codes": []})

    agent = ReproduceFiguresAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({})

    figures = {row["element_id"]: row for row in result["figures"]["figures"]}
    assert "fig_pepita" in figures
    assert figures["fig_pepita"]["reproduction_status"] == "REPRODUCED"
    assert figures["fig_pepita"]["match_level"] == "RELATED"
    assert figures["fig_pepita"]["match_level"] != "EXACT"
    assert "Related evidence only" in figures["fig_pepita"]["comparison_notes"]
    assert artifacts.path(figures["fig_pepita"]["image_path"]).exists()
    skipped = {row["element_id"]: row for row in result["figures"]["skipped_targets"]}
    assert "fig_diagram" in skipped
    assert "SKIP_DIAGRAM" in skipped["fig_diagram"]["reason_codes"]


def test_llm_skip_is_overridden_when_related_evidence_can_be_plotted(tmp_path: Path, monkeypatch) -> None:
    artifacts = _artifacts(tmp_path, "run_llm_skip_override")
    _write_phase2_package(
        artifacts,
        stdout_text="[1,  10] loss: 1.0\nTesting...\nTest accuracy: 80 %\n[2,  10] loss: 0.9\nTesting...\nTest accuracy: 90 %\n",
    )
    _write_single_target(
        artifacts,
        element_id="fig_related",
        caption="Figure 2. PEPITA training accuracy curve on MNIST.",
        model_names=["PEPITA"],
    )
    skip_spec = {
        "decision": "SKIP",
        "chart_type": "text-panel",
        "title": "Skipped by planner",
        "x_label": "",
        "y_label": "",
        "series": [],
        "table": {"columns": [], "rows": [], "source": None},
        "unit": None,
        "normalization": None,
        "evidence_sources": ["phase2"],
        "comparison_note": "Planner found only adjacent evidence.",
        "match_level": "RELATED",
        "matched_scope": {"algorithm": "related"},
        "coverage_note": "BP evidence is adjacent to the PEPITA target.",
        "confidence": 0.2,
        "reason_codes": ["MISSING_FIGURE_SPECIFIC_CURVES"],
    }

    agent = ReproduceFiguresAgent(llm=object(), artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_json", lambda schema, system, user: (skip_spec, None))

    result = agent.execute({})
    figure = result["figures"]["figures"][0]

    assert figure["reproduction_status"] == "REPRODUCED"
    assert figure["match_level"] == "RELATED"
    assert "Related evidence only" in figure["comparison_notes"]
    assert "LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK" in figure["reason_codes"]
    assert "MISSING_FIGURE_SPECIFIC_CURVES" in figure["reason_codes"]
    assert artifacts.path(figure["image_path"]).exists()
    assert result["figures"]["skipped_targets"] == []


def test_log_evidence_generates_comparison_without_repo_figures_or_metrics(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, "run_log_only")
    artifacts.write_text(
        "execution/executor_outputs/experiment_exp_02_full_CIFAR10_Conv_BP_stdout.log",
        "[1,  10] loss: 1.5\nTest accuracy: 55 %\n[2,  10] loss: 1.2\nTest accuracy: 62 %\n",
    )
    ExecutionLogEvidenceAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1).execute({})
    _write_single_target(
        artifacts,
        element_id="fig_cifar_curve",
        caption="Figure 7. CIFAR10 Conv BP test accuracy curve.",
        model_names=["BP"],
    )

    agent = ReproduceFiguresAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({})
    figure = result["figures"]["figures"][0]

    assert figure["reproduction_status"] == "REPRODUCED"
    assert figure["match_level"] == "EXACT"
    assert "results/execution_log_evidence.json" in figure["evidence_sources"]
    assert "experiment_exp_02_full_CIFAR10_Conv_BP_stdout.log" in " ".join(figure["evidence_sources"])
    assert artifacts.path(figure["image_path"]).exists()


def test_deterministic_fallback_uses_matched_claim_verdicts_when_llm_unavailable(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, "run_deterministic")
    reference = _write_reference(artifacts, "table_1")
    artifacts.write_json(
        "fingerprint/visual_targets.json",
        {
            "visual_targets": [
                {
                    "element_id": "table_1",
                    "visual_anchor": "Table 1",
                    "element_type": "table",
                    "chart_type": "table",
                    "caption": "Table 1. Test accuracy results.",
                    "reference_image_path": reference,
                    "metric_names": ["accuracy"],
                    "model_names": ["BP"],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_json("fingerprint/visual_elements.json", {"elements": [], "reason_codes": []})
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "claims": [
                {
                    "claim_id": "claim_01",
                    "type": "result",
                    "predicate": "BP MNIST accuracy = 0.9",
                    "metric": "accuracy",
                    "target": 0.9,
                    "conditions": {"table_anchor": "Table 1"},
                }
            ],
            "experiments": [],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "results/verdict.json",
        {
            "status": "PARTIALLY_SUPPORTED",
            "claim_verdicts": [
                {
                    "claim_id": "claim_01",
                    "status": "SUPPORTED",
                    "detail": "matched",
                    "target_value": 0.9,
                    "compared_value": 0.91,
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )

    agent = ReproduceFiguresAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({})
    figure = result["figures"]["figures"][0]

    assert figure["reproduction_status"] == "REPRODUCED"
    assert figure["plot_spec"]["chart_type"] == "table"
    assert "DETERMINISTIC_VERDICT_TABLE_FALLBACK" in figure["reason_codes"]
    assert artifacts.path(figure["image_path"]).exists()


def test_codegen_fallback_renders_when_plot_spec_is_unsupported(tmp_path: Path, monkeypatch) -> None:
    artifacts = _artifacts(tmp_path, "run_codegen")
    _write_phase2_package(artifacts, stdout_text="[1,  10] loss: 1.0\nTesting...\nTest accuracy: 80 %\n")
    _write_single_target(artifacts)
    calls = {"count": 0}

    def fake_chat_json(schema, system, user):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                {
                    "decision": "PLOT",
                    "chart_type": "custom",
                    "title": "Custom",
                    "x_label": "",
                    "y_label": "",
                    "series": [],
                    "table": {"columns": [], "rows": [], "source": None},
                    "unit": None,
                    "normalization": None,
                    "evidence_sources": ["phase2"],
                    "comparison_note": "Uses codegen.",
                    "confidence": 0.8,
                    "reason_codes": ["UNSUPPORTED_FOR_RENDERER"],
                },
                None,
            )
        return (
            {
                "code": (
                    "import matplotlib.pyplot as plt\n"
                    "fig, ax = plt.subplots(figsize=(4, 3))\n"
                    "ax.plot([1, 2], [0.8, 0.9])\n"
                    "fig.savefig(output_path, dpi=120, bbox_inches='tight')\n"
                )
            },
            None,
        )

    agent = ReproduceFiguresAgent(llm=object(), artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_json", fake_chat_json)

    result = agent.execute({})
    figure = result["figures"]["figures"][0]

    assert figure["reproduction_status"] == "REPRODUCED"
    assert "LLM_CODEGEN_RENDERED" in figure["reason_codes"]
    assert figure["code_path"] == "results/figures/fig_curve_codegen_primary_audit.json"
    assert artifacts.path(figure["image_path"]).exists()
    assert artifacts.path(figure["code_path"]).exists()
    assert not artifacts.path("results/figures/_codegen_tmp").exists()


def test_codegen_rejects_dangerous_code_and_records_failure(tmp_path: Path, monkeypatch) -> None:
    artifacts = _artifacts(tmp_path, "run_dangerous_codegen")
    _write_phase2_package(artifacts, stdout_text="[1,  10] loss: 1.0\nTesting...\nTest accuracy: 80 %\n")
    _write_single_target(artifacts)
    calls = {"count": 0}

    def fake_chat_json(schema, system, user):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                {
                    "decision": "PLOT",
                    "chart_type": "custom",
                    "title": "Custom",
                    "x_label": "",
                    "y_label": "",
                    "series": [],
                    "table": {"columns": [], "rows": [], "source": None},
                    "unit": None,
                    "normalization": None,
                    "evidence_sources": ["phase2"],
                    "comparison_note": "Uses codegen.",
                    "confidence": 0.8,
                    "reason_codes": ["UNSUPPORTED_FOR_RENDERER"],
                },
                None,
            )
        return {"code": "import os\nos.system('echo unsafe')\n"}, None

    agent = ReproduceFiguresAgent(llm=object(), artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_json", fake_chat_json)

    result = agent.execute({})
    figure = result["figures"]["figures"][0]

    assert figure["reproduction_status"] == "FAILED"
    assert "CODEGEN_REJECTED" in figure["reason_codes"]
    assert figure["image_path"] == ""
