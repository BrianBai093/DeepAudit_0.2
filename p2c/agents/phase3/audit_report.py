from __future__ import annotations

import json
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.agents.phase3.claim_inputs import load_effective_claims_ir
from p2c.agents.phase3.execution_summary_evidence import SUMMARY_PRIORITY_NOTE, load_effective_run_manifest
from p2c.agents.phase2.result_extraction import is_static_inspection_command

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

| Claim | Paper Value | Reproduced | Δ | Status |
|-------|-------------|------------|---|--------|
Use ✅ for SUPPORTED, ❌ for NOT_SUPPORTED, ⚠️ for INCONCLUSIVE.
The Claim cell MUST use the concrete claim text first, with the internal claim_id in parentheses.
Example: "fraudulent:legit ratio = 1:1 (claim_01)".

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
The Result column MUST summarize status, fidelity, execution_outcome, evidence_source, and stop_reason.

## Rules
- NO verbose paragraphs. Use tables and single-sentence bullets ONLY.
- Max 2 sentences per section OUTSIDE of tables.
- Every number MUST come from the provided artifacts.
- Use the effective claims and effective run manifest supplied in the prompt. The effective claims
  normalize mean±std paper values so the mean is the target and std is tolerance context.
- If EXECUTION_SUMMARY_FINAL.md conflicts with lower-priority execution evidence, treat it as
  same-origin highest-priority evidence and mention conflicts rather than silently downgrading it.
- Do NOT fabricate metrics, file paths, or results.
- Do NOT repeat information across sections.
- Do NOT use bare claim IDs as the primary label; always show the concrete claim text first.
- Distinguish: not implemented vs executed-but-misaligned vs numerically-disagrees.
- Treat status="partial" as degraded success.
- Write in clear, professional English.
"""


def _claim_title(claim: dict, claim_id: str | None = None) -> str:
    """Human-readable claim label for reports."""
    predicate = str(claim.get("predicate") or "").strip()
    if predicate:
        return predicate
    metric = str(claim.get("metric") or "").strip()
    target = claim.get("target")
    if metric and target is not None:
        return f"{metric} = {target}"
    if metric:
        return metric
    return str(claim_id or "unknown claim")


def _claim_meta(claim: dict, claim_id: str | None = None) -> str:
    """Compact traceability metadata after the readable claim title."""
    parts = [str(claim_id)] if claim_id else []
    ctype = str(claim.get("type") or "").strip()
    if ctype:
        parts.append(ctype)
    conditions = claim.get("conditions", {})
    if isinstance(conditions, dict):
        experiment_id = str(conditions.get("experiment_id") or "").strip()
        if experiment_id:
            parts.append(experiment_id)
        table_anchor = str(conditions.get("table_anchor") or "").strip()
        if table_anchor:
            parts.append(table_anchor)
    return ", ".join(parts)


def _claim_label(claim: dict, claim_id: str | None = None) -> str:
    title = _claim_title(claim, claim_id)
    meta = _claim_meta(claim, claim_id)
    return f"{title} ({meta})" if meta else title


def _report_image_path(image_path: str) -> str:
    """Convert run-root-relative image paths for use from results/report.md."""
    path = str(image_path or "").strip()
    if path.startswith("results/"):
        return path[len("results/"):]
    return path


def _plan_step_map(plan: dict) -> dict[str, dict]:
    rows = plan.get("execution_steps", []) if isinstance(plan, dict) else []
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("step_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("step_id")
    }


def _metricless_report_step(run: dict, planned_step: dict | None) -> bool:
    """Return True for setup/static inspection runs whose metrics are untrusted."""
    planned_step = planned_step or {}
    expected = planned_step.get("expected_metrics") or []
    produced = planned_step.get("produced_artifacts") or []
    if expected or produced:
        return False
    if planned_step.get("is_setup"):
        return True
    planned_command = str(planned_step.get("command") or "")
    if planned_command and is_static_inspection_command(planned_command):
        return True
    return is_static_inspection_command(str(run.get("command") or ""))


def _run_metrics_for_report(run: dict, plan_steps: dict[str, dict]) -> dict:
    """Filter manifest metrics before they are shown to the report LLM."""
    metrics = run.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    planned_step = plan_steps.get(str(run.get("run_id") or ""))
    if _metricless_report_step(run, planned_step):
        return {}
    return metrics


def _failure_rows_for_report(payload) -> list:
    """Normalize execution_failures.json across list/current/legacy shapes."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("failures", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        if payload.get("step_failures"):
            return [payload]
    return []


def _build_report_prompt(ctx: dict, artifacts) -> str:
    """Assemble all Phase 1+2+3 artifacts into a single prompt."""
    sections = []

    # ── Paper claims + experiments ──────────────────────────────
    claims_ir = load_effective_claims_ir(artifacts)

    experiments = claims_ir.get("experiments", [])
    if experiments:
        sections.append("# PAPER EXPERIMENTS (identified by LLM)")
        sections.append(json.dumps(experiments, indent=2, ensure_ascii=False)[:3000])

    sections.append("\n# PAPER CLAIMS (effective_claims_ir.json)")
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

    # ── Highest-priority execution summary and effective manifest ──
    try:
        summary_evidence = artifacts.read_json("results/execution_summary_evidence.json")
        summary_path = summary_evidence.get("summary_path")
        if summary_path:
            summary_file = artifacts.path(summary_path)
            summary_text = summary_file.read_text(encoding="utf-8", errors="ignore") if summary_file.exists() else ""
            sections.append("\n# EXECUTION SUMMARY FINAL (same-origin, highest priority)")
            sections.append(SUMMARY_PRIORITY_NOTE)
            sections.append(summary_text[:6000])
    except Exception:  # noqa: BLE001
        pass

    manifest = load_effective_run_manifest(artifacts)
    sections.append("\n# RUN MANIFEST (effective Phase 3 execution results)")
    for run in manifest.get("runs", []):
        sections.append(f"\n## Experiment: {run.get('experiment_name') or run.get('run_id')}")
        sections.append(f"- status: {run.get('status')}")
        sections.append(f"- fidelity: {run.get('fidelity')}")
        sections.append(f"- execution_outcome: {run.get('execution_outcome')}")
        sections.append(f"- evidence_source: {run.get('evidence_source')}")
        sections.append(f"- stop_reason: {run.get('stop_reason')}")
        sections.append(f"- exit_code: {run.get('exit_code')}")
        sections.append(f"- runtime_sec: {run.get('runtime_sec')}")
        if run.get("params"):
            sections.append(f"- params: {json.dumps(run.get('params', {}), ensure_ascii=False)}")
        if run.get("reason_codes"):
            sections.append(f"- reason_codes: {json.dumps(run.get('reason_codes', []), ensure_ascii=False)}")
        sections.append(f"- commands_attempted: {json.dumps(run.get('commands_attempted', []), ensure_ascii=False)}")
        sections.append(f"- metrics: {json.dumps(run.get('metrics', {}), ensure_ascii=False)}")
        stdout = run.get("stdout_tail", "")
        if stdout:
            sections.append(f"- stdout_tail (last 500 chars): {stdout[-500:]}")
        logs = run.get("logs") or {}
        if logs:
            sections.append(f"- logs: {json.dumps(logs, ensure_ascii=False)}")

    try:
        raw_manifest = artifacts.read_json("execution/executor_outputs/run_manifest.json")
        if raw_manifest and raw_manifest != manifest:
            sections.append("\n# RAW RUN MANIFEST (lower-priority audit context)")
            sections.append(json.dumps(raw_manifest, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    # ── Execution failures ───────────────────────────────────────
    try:
        failures = artifacts.read_json("execution/execution_failures.json")
        failure_rows = _failure_rows_for_report(failures)
        if failure_rows:
            sections.append("\n# EXECUTION FAILURES")
            sections.append(json.dumps(failure_rows, indent=2, ensure_ascii=False)[:2000])
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

    # ── Structured metrics ───────────────────────────────────────
    try:
        metrics = artifacts.read_json("results/metrics.json")
        sections.append("\n# STRUCTURED METRICS")
        sections.append(json.dumps(metrics, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    try:
        activity = artifacts.path("execution/executor_outputs/executor_activity.jsonl")
        if activity.exists():
            sections.append("\n# EXECUTOR ACTIVITY")
            sections.append(activity.read_text(encoding="utf-8", errors="ignore")[:3000])
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

    try:
        visual_alignment = artifacts.read_json("results/visual_to_repo_alignment.json")
        if visual_alignment.get("alignments"):
            sections.append("\n# VISUAL TO REPO ALIGNMENT")
            sections.append(json.dumps(visual_alignment, indent=2, ensure_ascii=False)[:3000])
    except Exception:  # noqa: BLE001
        pass

    # ── Reproduced figures ──────────────────────────────────────
    try:
        figs = artifacts.read_json("results/reproduced_figures.json")
        if figs.get("figures"):
            sections.append("\n# REPRODUCED FIGURES")
            for fig in figs["figures"]:
                if fig.get("image_path"):
                    sections.append(
                        f"- {fig['element_id']}: "
                        f"![{fig.get('comparison_notes', '')}]({_report_image_path(fig['image_path'])})"
                    )
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
        claims_doc = load_effective_claims_ir(self.artifacts)
        manifest = load_effective_run_manifest(self.artifacts)
        reproduced_figures = self.artifacts.read_json("results/reproduced_figures.json")
        try:
            visual_alignment = self.artifacts.read_json("results/visual_to_repo_alignment.json")
        except Exception:  # noqa: BLE001
            visual_alignment = {}
        alignment_by_id = {
            str(row.get("element_id")): row
            for row in visual_alignment.get("alignments", [])
            if isinstance(row, dict) and row.get("element_id")
        }

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
            "## Reproduced Figures",
            "",
        ]

        figures = [
            fig for fig in reproduced_figures.get("figures", [])
            if fig.get("image_path")
        ]
        if figures:
            for fig in figures:
                image_path = _report_image_path(fig.get("image_path", ""))
                caption = fig.get("comparison_notes") or fig.get("element_id") or "reproduced figure"
                lines.append(f"![{caption}]({image_path})")
                lines.append("")
                element_id = str(fig.get("element_id") or "")
                alignment = alignment_by_id.get(element_id, {})
                alignment_note = ""
                if alignment:
                    status = alignment.get("status", "NO_MATCH")
                    reasons = alignment.get("mismatch_reasons") or []
                    reason = f" ({reasons[0]})" if reasons else ""
                    alignment_note = f" visual alignment={status}{reason}."
                lines.append(f"- {element_id}: {caption}.{alignment_note}")
                lines.append("")
        else:
            lines.extend(["No figures reproduced.", ""])

        lines.extend([
            "## Claim Verdicts",
            "",
        ])

        for cv in verdict.get("claim_verdicts", []):
            claim = next(
                (c for c in claims_doc.get("claims", []) if c.get("claim_id") == cv.get("claim_id")),
                {},
            )
            claim_id = cv.get("claim_id")
            lines.append(
                f"- **{_claim_title(claim, claim_id)}** "
                f"(`{_claim_meta(claim, claim_id)}`) [{cv.get('status')}]: "
                f"{cv.get('detail', '')}"
            )
            if cv.get("compared_value") is not None:
                lines.append(
                    f"  - Reproduced: {cv['compared_value']}, "
                    f"Target: {cv.get('target_value')}"
                )

        lines.extend([
            "",
            "## Execution Runs",
            "",
        ])
        for run in manifest.get("runs", []):
            run_id = str(run.get("run_id") or "")
            lines.append(
                f"- **{run_id}** [{run.get('status', 'unknown')}]: "
                f"exit_code={run.get('exit_code')}, "
                f"fidelity={run.get('fidelity')}, "
                f"outcome={run.get('execution_outcome')}, "
                f"evidence={run.get('evidence_source')}, "
                f"stop_reason={run.get('stop_reason')}"
            )
            if run.get("command"):
                lines.append(f"  - Primary command: `{run.get('command')}`")
            if run.get("commands_attempted"):
                lines.append(f"  - Commands attempted: {json.dumps(run.get('commands_attempted', []), ensure_ascii=False)}")
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
