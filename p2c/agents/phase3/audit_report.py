from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from p2c.agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are a senior ML reproducibility auditor writing the final assessment report.

You will receive ALL artifacts from a paper reproduction pipeline:
- Paper claims extracted from the paper (with experiment context)
- Repository analysis (what code/notebooks exist)
- Execution results (what ran, what metrics were produced)
- Claim verdicts (which claims were supported/not supported/inconclusive)
- Experiment coverage (which paper experiments are in the repo, which are missing)

Write a comprehensive reproducibility audit report in Markdown. The report MUST include:

## Structure

### 1. Executive Summary
- One-paragraph overview: what paper, what repo, what was the outcome
- **Reproducibility Score: X/10** (integer, based on criteria below)

### 2. Experiment Coverage
- Table listing each experiment from the paper
- For each: is it implemented in the repo? Was it executed? Result?
- Highlight missing experiments that the paper describes but the repo lacks

### 3. Result Verification
- For each result claim that could be verified:
  - Paper claimed value vs reproduced value
  - Within tolerance? Yes/No
  - Brief analysis of any discrepancy
- For claims that could NOT be verified: explain why

### 4. Configuration & Environment
- Were the reported configurations (dataset sizes, hyperparameters, etc.) \
consistent with what the code actually uses?
- Any environment issues encountered during execution?

### 5. Gaps & Concerns
- Missing experiments, unreproducible claims, code quality issues
- Anything suspicious or concerning about reproducibility

### 6. Scoring Breakdown
Explain the score using these criteria (each 0-2 points):
1. **Code Completeness** (0-2): Does the repo contain code for all paper experiments?
2. **Execution Success** (0-2): Did the code run without major issues?
3. **Result Accuracy** (0-2): Do reproduced results match paper claims?
4. **Documentation** (0-2): Is the repo well-documented for reproduction?
5. **Data Availability** (0-2): Are datasets accessible and properly referenced?

## Rules
- Be factual. Only cite values that appear in the artifacts.
- Do NOT fabricate metrics, file paths, or claim results.
- Use specific numbers and evidence for every assertion.
- If something is unclear, say so explicitly rather than guessing.
- Distinguish clearly between:
  1. repo not implemented,
  2. repo executed but cannot be aligned to the paper's exact experiment,
  3. repo executed and numerically disagrees with the paper.
- Treat `status="partial"` execution steps as degraded success: the primary command failed and a fallback only partially validated the step.
- If runnable entrypoints and successful execution evidence exist, do NOT describe the \
  repository as missing implementation unless the artifacts clearly prove that absence.
- If environment validation failed but failed_packages is empty and core steps ran, treat \
  that as a validation warning or probe mismatch, not as a hard execution failure.
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

        # ── Post-process: PictureToWords ─────────────────────────
        draft_path = self.artifacts.path("results/report.draft.md")
        draft_path.write_text(report_text, encoding="utf-8")

        picture_script = Path.cwd() / "PictureToWords.py"
        if picture_script.exists():
            cmd = [
                sys.executable,
                str(picture_script),
                "--input",
                str(draft_path),
                "--output",
                str(self.artifacts.path("results/report.md")),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.artifacts.append_text(
                "execution/run.log",
                f"\n$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n",
            )
            if proc.returncode != 0:
                self.artifacts.write_text(
                    "results/report.md",
                    draft_path.read_text(encoding="utf-8"),
                )
        else:
            self.artifacts.write_text(
                "results/report.md",
                draft_path.read_text(encoding="utf-8"),
            )

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
