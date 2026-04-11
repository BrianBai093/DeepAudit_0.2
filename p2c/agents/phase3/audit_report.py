from __future__ import annotations

import json
from pathlib import Path

from p2c.agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are a senior ML reproducibility auditor writing the final assessment report.

You will receive ALL artifacts from a paper reproduction pipeline including a pre-computed \
reproducibility score (0-100) with four dimensions and a gap diagnosis taxonomy.

Write a CONCISE reproducibility audit report in Markdown. Follow the structure EXACTLY:

## Report Structure (STRICT — follow this order and format)

### 1. Score Card (tables only, NO prose)

| Dimension | Score | Weight | Weighted | Key Evidence |
|-----------|-------|--------|----------|--------------|
| Environment | X/100 | 25% | ... | one-line evidence |
| Data Availability | X/100 | 25% | ... | one-line evidence |
| Execution Success | X/100 | 20% | ... | one-line evidence |
| Claim Match | X/100 | 30% | ... | one-line evidence |
| **Total** | | | **X/100** | |

**ECR (Executable-Claim Reproducible)**: ✅ True / ❌ False — one-line reason

### 2. Verdict Dashboard (table only, NO prose)

| Claim ID | Paper Value | Reproduced | Δ | Status |
|----------|-------------|------------|---|--------|
Use ✅ for SUPPORTED, ❌ for NOT_SUPPORTED, ⚠️ for INCONCLUSIVE.

### 3. Reproduced Figures
For each reproduced figure: embed ![caption](path) and add ONE sentence comparison note.
If no figures were generated, write "No figures reproduced."

### 4. Gap Diagnosis (table only)

| # | Category | Severity | Affected Claims | Description |
|---|----------|----------|-----------------|-------------|
Categories MUST be from: DATA_MISSING, PREPROCESS_UNSPECIFIED, CHECKPOINT_MISSING, \
ENVIRONMENT_UNDERDEFINED, ENTRYPOINT_UNCLEAR, NONDETERMINISM, COMPUTE_INFEASIBLE, RESULT_MISMATCH.

### 5. Experiment Coverage (table only)

| Experiment | Implemented? | Executed? | Result |
|------------|-------------|-----------|--------|

## Rules
- NO verbose paragraphs. Use tables and single-sentence bullets ONLY.
- Max 2 sentences per section OUTSIDE of tables.
- Every number MUST come from the provided artifacts.
- Do NOT fabricate metrics, file paths, or results.
- Do NOT repeat information across sections.
- Distinguish: not implemented vs executed-but-misaligned vs numerically-disagrees.
- Treat status="partial" as degraded success.
- Write in clear, professional English.
"""


def _build_report_prompt(ctx: dict, artifacts) -> str:
    """Assemble all Phase 1+2+3 artifacts into a single prompt."""
    sections = []

    # ── Paper claims + experiments ──────────────────────────────
    claims_ir = artifacts.read_json("fingerprint/claims_ir.json")

    experiments = claims_ir.get("experiments", [])
    if experiments:
        sections.append("# PAPER EXPERIMENTS (identified by LLM)")
        sections.append(json.dumps(experiments, indent=2, ensure_ascii=False)[:3000])

    sections.append("\n# PAPER CLAIMS (claims_ir.json)")
    sections.append(json.dumps(claims_ir.get("claims", []), indent=2, ensure_ascii=False)[:5000])

    # ── Fingerprint configurations ───────────────────────────────
    try:
        fp = artifacts.read_json("fingerprint/fingerprint.json")
        configs = fp.get("configurations", {})
        if configs:
            sections.append("\n# PAPER CONFIGURATIONS (from fingerprint)")
            sections.append(json.dumps(configs, indent=2, ensure_ascii=False)[:2000])
    except Exception:  # noqa: BLE001
        pass

    # ── Repo analysis ────────────────────────────────────────────
    try:
        repo = artifacts.read_json("task/repo_analysis.json")
        sections.append("\n# REPOSITORY ANALYSIS")
        sections.append(json.dumps(repo, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    # ── Execution plan ───────────────────────────────────────────
    try:
        plan = artifacts.read_json("execution/execution_plan.json")
        sections.append("\n# EXECUTION PLAN")
        # Only include key fields to save tokens
        plan_summary = {
            "plan_id": plan.get("plan_id"),
            "python_version": plan.get("python_version"),
            "steps": [
                {"step_id": s.get("step_id"), "description": s.get("description"), "command": s.get("command", "")[:200]}
                for s in plan.get("execution_steps", [])
            ],
            "compatibility_issues": plan.get("compatibility_issues", []),
        }
        sections.append(json.dumps(plan_summary, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    # ── Run manifest ─────────────────────────────────────────────
    manifest = artifacts.read_json("execution/codex_outputs/run_manifest.json")
    sections.append("\n# RUN MANIFEST (execution results)")
    for run in manifest.get("runs", []):
        sections.append(f"\n## Step: {run.get('run_id')}")
        sections.append(f"- status: {run.get('status')}")
        sections.append(f"- exit_code: {run.get('exit_code')}")
        sections.append(f"- runtime_sec: {run.get('runtime_sec')}")
        if run.get("params"):
            sections.append(f"- params: {json.dumps(run.get('params', {}), ensure_ascii=False)}")
        if run.get("reason_codes"):
            sections.append(f"- reason_codes: {json.dumps(run.get('reason_codes', []), ensure_ascii=False)}")
        if run.get("status") == "partial":
            sections.append("- partial_execution_note: primary command failed; fallback only validated artifacts or a reduced objective")
        sections.append(f"- metrics: {json.dumps(run.get('metrics', {}))}")
        # Include stdout tail for context (truncated)
        stdout = run.get("stdout_tail", "")
        if stdout:
            sections.append(f"- stdout_tail (last 500 chars): {stdout[-500:]}")

    # ── Execution failures ───────────────────────────────────────
    try:
        failures = artifacts.read_json("execution/execution_failures.json")
        if failures.get("failures"):
            sections.append("\n# EXECUTION FAILURES")
            sections.append(json.dumps(failures, indent=2, ensure_ascii=False)[:2000])
    except Exception:  # noqa: BLE001
        pass

    # ── Env setup ────────────────────────────────────────────────
    try:
        env = artifacts.read_json("execution/env_setup_result.json")
        sections.append("\n# ENVIRONMENT SETUP")
        env_summary = {
            "env_name": env.get("env_name"),
            "python_version": env.get("python_version"),
            "validation_passed": env.get("validation_passed"),
            "failed_packages": env.get("failed_packages", []),
        }
        sections.append(json.dumps(env_summary, indent=2, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass

    # ── Verdict (from verify_claims) ─────────────────────────────
    verdict = artifacts.read_json("results/verdict.json")
    sections.append("\n# CLAIM VERDICTS")
    sections.append(json.dumps(verdict, indent=2, ensure_ascii=False)[:4000])

    # ── Structured metrics + alignment ───────────────────────────
    try:
        metrics = artifacts.read_json("results/metrics.json")
        sections.append("\n# STRUCTURED METRICS")
        sections.append(json.dumps(metrics, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    try:
        alignment = artifacts.read_json("execution/codex_outputs/claim_alignment.json")
        sections.append("\n# CLAIM ALIGNMENT")
        sections.append(json.dumps(alignment, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    # ── Evaluability ─────────────────────────────────────────────
    eval_verdict = artifacts.read_json("results/evaluability_verdict.json")
    sections.append("\n# EVALUABILITY VERDICT")
    sections.append(json.dumps(eval_verdict, indent=2, ensure_ascii=False)[:2000])

    # ── Reproducibility Score (0-100) ───────────────────────────
    try:
        score = artifacts.read_json("results/reproducibility_score.json")
        sections.append("\n# REPRODUCIBILITY SCORE (0-100)")
        sections.append(json.dumps(score, indent=2, ensure_ascii=False)[:4000])
    except Exception:  # noqa: BLE001
        pass

    # ── Visual elements from PDF ────────────────────────────────
    try:
        ve = artifacts.read_json("fingerprint/visual_elements.json")
        if ve.get("elements"):
            sections.append("\n# VISUAL ELEMENTS FROM PAPER")
            sections.append(json.dumps(ve, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    # ── Reproduced figures ──────────────────────────────────────
    try:
        figs = artifacts.read_json("results/reproduced_figures.json")
        if figs.get("figures"):
            sections.append("\n# REPRODUCED FIGURES")
            for fig in figs["figures"]:
                if fig.get("image_path"):
                    sections.append(f"- {fig['element_id']}: ![{fig.get('comparison_notes', '')}]({fig['image_path']})")
    except Exception:  # noqa: BLE001
        pass

    # ── Context ──────────────────────────────────────────────────
    sections.append(f"\n# RUN CONTEXT")
    sections.append(f"- run_id: {ctx.get('run_id')}")
    sections.append(f"- repo_dir: {ctx.get('repo_dir')}")

    return "\n".join(sections)


class AuditReportAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="audit_report", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        user_prompt = _build_report_prompt(ctx, self.artifacts)

        # ── LLM generates the full report ────────────────────────
        report_text, llm_err = self.safe_chat_text(SYSTEM_PROMPT, user_prompt)

        if not report_text:
            self.log("PROGRESS", f"LLM unavailable ({llm_err}), generating fallback report")
            report_text = self._fallback_report(ctx)

        # ── Write report directly (no PictureToWords stripping on output) ──
        draft_path = self.artifacts.path("results/report.draft.md")
        draft_path.write_text(report_text, encoding="utf-8")
        self.artifacts.write_text("results/report.md", report_text)

        return {"report": "results/report.md"}

    def _fallback_report(self, ctx: dict) -> str:
        """Deterministic fallback when LLM is unavailable."""
        verdict = self.artifacts.read_json("results/verdict.json")
        eval_verdict = self.artifacts.read_json("results/evaluability_verdict.json")
        metrics = self.artifacts.read_json("results/metrics.json")
        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        manifest = self.artifacts.read_json("execution/codex_outputs/run_manifest.json")

        lines = [
            "# Reproducibility Audit Report",
            "",
            f"- run_id: `{ctx.get('run_id')}`",
            f"- repo_dir: `{ctx.get('repo_dir')}`",
            "",
            "## Executive Summary",
            "",
            f"Overall verdict: **{verdict.get('status', 'INCONCLUSIVE')}**",
            f"Evaluability: **{eval_verdict.get('status', 'NOT_EVALUABLE')}**",
            "",
            "*Note: LLM was unavailable. This is a simplified deterministic report.*",
            "",
            "## Claim Verdicts",
            "",
        ]

        for cv in verdict.get("claim_verdicts", []):
            claim = next(
                (c for c in claims_doc.get("claims", []) if c.get("claim_id") == cv.get("claim_id")),
                {},
            )
            lines.append(
                f"- **{cv.get('claim_id')}** [{cv.get('status')}]: "
                f"{claim.get('predicate', 'N/A')} — {cv.get('detail', '')}"
            )
            if cv.get("compared_value") is not None:
                lines.append(
                    f"  - Reproduced: {cv['compared_value']}, "
                    f"Target: {cv.get('target_value')}"
                )

        lines.extend([
            "",
            "## Execution Steps",
            "",
        ])
        for run in manifest.get("runs", []):
            lines.append(
                f"- **{run.get('run_id')}** [{run.get('status', 'unknown')}]: "
                f"command=`{run.get('command', '')}` exit_code={run.get('exit_code')}"
            )
            if run.get("status") == "partial":
                lines.append("  - primary command failed; fallback only validated artifacts or a reduced objective")
            if run.get("params"):
                lines.append(f"  - params: {json.dumps(run.get('params', {}), ensure_ascii=False)}")

        lines.extend([
            "",
            "## Metrics Collected",
            "",
        ])
        for rec in metrics.get("records", []):
            lines.append(
                f"- {rec.get('metric_name')} = {rec.get('value')} "
                f"(source: {rec.get('source')})"
            )

        return "\n".join(lines) + "\n"
