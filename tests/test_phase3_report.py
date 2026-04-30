from __future__ import annotations

from pathlib import Path

from p2c.agents.phase3.audit_report import AuditReportAgent, _build_report_prompt
from p2c.io_artifacts import ArtifactManager


def test_fallback_report_uses_claim_text_before_internal_id(tmp_path: Path, monkeypatch) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_report")
    artifacts.ensure_tree()
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "claims": [
                {
                    "claim_id": "claim_01",
                    "type": "config",
                    "predicate": "fraudulent:legit ratio = 1:1",
                    "conditions": {"experiment_id": "exp_01"},
                }
            ],
            "experiments": [],
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "results/verdict.json",
        {
            "status": "INCONCLUSIVE",
            "claim_verdicts": [
                {
                    "claim_id": "claim_01",
                    "status": "INCONCLUSIVE",
                    "detail": "Configuration claim requires direct code/config evidence.",
                    "compared_value": None,
                    "target_value": None,
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
            "summary": "Evaluated 1 claims.",
        },
    )
    artifacts.write_json(
        "results/evaluability_verdict.json",
        {"status": "PARTIAL", "claim_rows": [], "reason_codes": [], "summary": ""},
    )
    artifacts.write_json("results/metrics.json", {"records": [], "reason_codes": []})
    artifacts.write_json("execution/executor_outputs/run_manifest.json", {"runs": [], "reason_codes": []})
    artifacts.write_json(
        "results/reproduced_figures.json",
        {
            "figures": [
                {
                    "element_id": "verdict_comparison",
                    "image_path": "results/figures/verdict_comparison.png",
                    "comparison_notes": "Audit-only claim comparison",
                    "reproduction_status": "REPRODUCED",
                    "reason_codes": [],
                },
                {
                    "element_id": "fig_2",
                    "visual_anchor": "Figure 2",
                    "image_path": "results/figures/fig_2_comparison.png",
                    "comparison_notes": "Figure 2. ROC curve comparison",
                    "reproduction_status": "REPRODUCED",
                    "match_level": "RELATED",
                    "coverage_note": "BP evidence only.",
                    "evidence_sources": ["execution/executor_outputs/phase2_execution_package.json:exp_01"],
                    "reason_codes": [],
                }
            ],
            "skipped_targets": [
                {
                    "element_id": "fig_3",
                    "visual_anchor": "Figure 3",
                    "skip_reason": "No executable evidence.",
                    "evidence_sources": [],
                    "reason_codes": ["SKIP_NO_PHASE2_EVIDENCE"],
                }
            ],
            "reason_codes": [],
        },
    )

    agent = AuditReportAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, "no key"))

    agent.execute({"run_id": "run_report", "repo_dir": "/tmp/repo"})
    report = artifacts.path("results/report.md").read_text(encoding="utf-8")

    assert "**fraudulent:legit ratio = 1:1**" in report
    assert "**claim_01** [INCONCLUSIVE]" not in report
    assert "`claim_01, config, exp_01`" in report
    assert "![Figure 2. ROC curve comparison](figures/fig_2_comparison.png)" in report
    assert "](results/figures/fig_2.png)" not in report
    assert "verdict_comparison" not in report
    assert "1 visuals have only partial/related Phase2 evidence; 1 result-related visual targets lacked executable phase2 evidence and were skipped." in report


def test_report_prompt_includes_executor_activity_and_new_manifest(tmp_path: Path) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_prompt")
    artifacts.ensure_tree()
    artifacts.write_json("fingerprint/claims_ir.json", {"claims": [], "experiments": [], "reason_codes": []})
    artifacts.write_json(
        "execution/executor_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "exp_01",
                    "experiment_id": "exp_01",
                    "experiment_name": "fc table 1",
                    "command": "python train.py",
                    "commands_attempted": ["python train.py"],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "fidelity": "artifact",
                    "execution_outcome": "TREND_SUPPORTED",
                    "evidence_source": "existing_logs",
                    "stop_reason": "existing_artifact",
                    "runtime_sec": 1.0,
                    "stdout_tail": "accuracy=0.99\n",
                    "metrics": {"accuracy": 0.99},
                    "logs": {"activity": "execution/executor_outputs/executor_activity.jsonl"},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_text(
        "execution/executor_outputs/executor_activity.jsonl",
        '{"ts":"2026-04-22T00:00:00Z","event":"command_end","experiment_id":"exp_01","cwd":".","command":"python train.py","status":"ok","exit_code":0,"duration_sec":1.0,"artifacts":[],"message":"done"}\n',
    )
    artifacts.write_json(
        "execution/execution_failures.json",
        [
            {
                "attempt": 1,
                "plan_version": 1,
                "stage": "execution",
                "overall_error": "failed once",
                "step_failures": [],
            }
        ],
    )
    artifacts.write_text(
        "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md",
        "#### Exp_01: Table 1\n**Status:** OK | **Fidelity:** Artifact Evaluation\n",
    )
    artifacts.write_json(
        "results/execution_summary_evidence.json",
        {
            "summary_path": "execution/executor_outputs/EXECUTION_SUMMARY_FINAL.md",
            "summary_runs": [],
            "conflicts": [],
            "reason_codes": [],
        },
    )

    prompt = _build_report_prompt({"run_id": "run_prompt", "repo_dir": "/tmp/repo"}, artifacts)

    assert prompt.index("# EXECUTION SUMMARY FINAL") < prompt.index("# RUN MANIFEST")
    assert "same-origin" in prompt
    assert "highest priority" in prompt
    assert "# RUN MANIFEST" in prompt
    assert "# EXECUTOR ACTIVITY" in prompt
    assert "python train.py" in prompt
    assert "execution_outcome: TREND_SUPPORTED" in prompt
    assert "evidence_source: existing_logs" in prompt
    assert "# CLAIM ALIGNMENT" not in prompt
    assert "# EXECUTION PLAN" not in prompt


def test_report_prompt_uses_reproduced_metadata_without_code(tmp_path: Path) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_repro_prompt")
    artifacts.ensure_tree()
    artifacts.write_json("fingerprint/claims_ir.json", {"claims": [], "experiments": [], "reason_codes": []})
    artifacts.write_json("results/verdict.json", {"status": "INCONCLUSIVE", "claim_verdicts": [], "reason_codes": []})
    artifacts.write_json("results/evaluability_verdict.json", {"status": "PARTIAL", "claim_rows": [], "reason_codes": []})
    artifacts.write_json("execution/executor_outputs/run_manifest.json", {"runs": [], "reason_codes": []})
    artifacts.write_json(
        "results/reproduced_figures.json",
        {
            "figures": [
                {
                    "element_id": "verdict_comparison",
                    "image_path": "results/figures/verdict_comparison.png",
                    "comparison_notes": "Audit-only chart",
                    "reproduction_status": "REPRODUCED",
                    "matplotlib_code": "SHOULD_NOT_APPEAR",
                    "reason_codes": [],
                },
                {
                    "element_id": "table_1",
                    "visual_anchor": "Table 1",
                    "image_path": "results/figures/table_1_comparison.png",
                    "comparison_notes": "Table 1 comparison",
                    "reproduction_status": "REPRODUCED",
                    "match_level": "PARTIAL",
                    "matched_scope": {"evidence_algorithms": ["bp"]},
                    "coverage_note": "Missing target algorithm evidence for pepita.",
                    "evidence_sources": ["phase2"],
                    "matplotlib_code": "SHOULD_NOT_APPEAR",
                    "reason_codes": ["LLM_PLOT_SPEC_RENDERED"],
                },
                {
                    "element_id": "fig_failed",
                    "image_path": "",
                    "comparison_notes": "failed",
                    "reproduction_status": "FAILED",
                    "matplotlib_code": "SHOULD_NOT_APPEAR",
                    "reason_codes": [],
                },
            ],
            "skipped_targets": [],
            "reason_codes": [],
        },
    )

    prompt = _build_report_prompt({"run_id": "run_repro_prompt", "repo_dir": "/tmp/repo"}, artifacts)

    assert "table_1_comparison.png" in prompt
    assert '"match_level": "PARTIAL"' in prompt
    assert "Missing target algorithm evidence for pepita." in prompt
    assert "verdict_comparison.png" not in prompt
    assert "SHOULD_NOT_APPEAR" not in prompt
