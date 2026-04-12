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
    artifacts.write_json("execution/codex_outputs/run_manifest.json", {"runs": [], "reason_codes": []})
    artifacts.write_json(
        "results/reproduced_figures.json",
        {
            "figures": [
                {
                    "element_id": "fig_2",
                    "image_path": "results/figures/fig_2.png",
                    "comparison_notes": "Figure 2. ROC curve",
                    "reason_codes": [],
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
    assert "![Figure 2. ROC curve](figures/fig_2.png)" in report
    assert "](results/figures/fig_2.png)" not in report


def test_report_prompt_filters_metricless_inspection_metrics_and_reads_failure_list(tmp_path: Path) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_prompt")
    artifacts.ensure_tree()
    artifacts.write_json("fingerprint/claims_ir.json", {"claims": [], "experiments": [], "reason_codes": []})
    artifacts.write_json(
        "execution/execution_plan.json",
        {
            "plan_id": "p1",
            "plan_version": 1,
            "python_version": "3.10",
            "execution_steps": [
                {
                    "step_id": "step_01_repo_inspect",
                    "description": "read source",
                    "command": "python -c \"from pathlib import Path; print(Path('train.py').read_text())\"",
                    "cwd": ".",
                    "timeout_sec": 60,
                    "depends_on": [],
                    "expected_metrics": [],
                    "is_setup": True,
                    "retry_on_failure": False,
                    "fallback_commands": [],
                    "required_artifacts": [],
                    "produced_artifacts": [],
                }
            ],
            "env_name": "test_env",
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/codex_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "step_01_repo_inspect",
                    "command": "python calculate_metrics.py",
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "runtime_sec": 1.0,
                    "stdout_tail": "accuracy=0.99\nprecision=0.88\n",
                    "metrics": {"accuracy": 0.99},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
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

    prompt = _build_report_prompt({"run_id": "run_prompt", "repo_dir": "/tmp/repo"}, artifacts)

    assert '"accuracy": 0.99' not in prompt
    assert "accuracy=0.99" not in prompt
    assert "stdout_tail_note: omitted" in prompt
    assert "metrics_note: omitted" in prompt
    assert "# EXECUTION FAILURES" in prompt
    assert "failed once" in prompt
