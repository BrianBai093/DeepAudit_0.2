from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from p2c.agents.base import BaseAgent

SYSTEM_PROMPT = "You write concise audit summaries without fabrication."
USER_PROMPT_TEMPLATE = "Input: all artifacts. Output: results/report.md"


class AuditReportAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="audit_report", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        summary_text, _ = self.safe_chat_text(
            SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE + " Provide 3 bullet uncertainty points.",
        )

        repo_state = self.artifacts.read_json("execution/repo_state.json")
        manifest = self.artifacts.read_json("execution/data_manifest.json")
        metrics = self.artifacts.read_json("results/metrics.json")
        verdict = self.artifacts.read_json("results/verdict.json")
        eval_verdict = self.artifacts.read_json("results/evaluability_verdict.json")
        task_spec = self.artifacts.read_json("task/task_spec.json")

        report = []
        commit = repo_state.get("head")
        branch = repo_state.get("branch")
        commit_text = str(commit) if commit else "N/A (gitless run)"
        branch_text = str(branch) if branch else "N/A (gitless run)"
        report.append("# Paper2Code Audit Report")
        report.append("")
        report.append(f"- run_id: `{ctx['run_id']}`")
        report.append(f"- repo_dir: `{ctx['repo_dir']}`")
        report.append(f"- commit: `{commit_text}`")
        report.append(f"- branch: `{branch_text}`")
        report.append("")
        report.append("## Execution Trace")
        report.append("")
        report.append("- codex worklog: `execution/codex_outputs/codex_worklog.jsonl`")
        report.append("- codex run manifest: `execution/codex_outputs/run_manifest.json`")
        report.append("- run log: `execution/run.log`")
        report.append("")
        report.append("## Data Manifest")
        report.append("")
        report.append(f"- unresolved: `{manifest.get('unresolved')}`")
        report.append(f"- entries: `{len(manifest.get('entries', []))}`")
        report.append("")
        report.append("## Metric Summary")
        report.append("")
        for rec in metrics.get("records", []):
            report.append(
                f"- metric={rec.get('metric_name')} value={rec.get('value')} parsed={rec.get('parsed')} reason={rec.get('reason_codes', [])}"
            )
        report.append("")
        report.append("## Claim Verdicts")
        report.append("")
        report.append(f"- overall status: **{verdict.get('status', 'INCONCLUSIVE')}**")
        report.append(f"- evaluability status: **{eval_verdict.get('status', 'NOT_EVALUABLE')}**")
        for row in verdict.get("claim_verdicts", []):
            report.append(
                f"- {row.get('claim_id')}: {row.get('status')} ({row.get('detail')})"
            )
        report.append("")
        report.append("## Task Spec Snapshot")
        report.append("")
        tasks = task_spec.get("tasks", [])
        if isinstance(tasks, list) and tasks:
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                report.append(
                    f"- `{task.get('task_id')}`: `{task.get('command')}` "
                    f"(entrypoint `{task.get('entrypoint')}`, timeout `{task.get('timeout_class')}`)"
                )
        else:
            for ep in task_spec.get("entrypoints", []):
                report.append(f"- `{ep.get('command')}` from `{ep.get('path')}`")
        report.append("")
        report.append("## Uncertainty")
        report.append("")
        if summary_text:
            report.append(summary_text.strip())
        else:
            report.append("- LLM summary unavailable; report generated from deterministic artifacts.")

        draft_path = self.artifacts.path("results/report.draft.md")
        draft_path.write_text("\n".join(report) + "\n", encoding="utf-8")

        # Requirement: run PictureToWords before final report to textualize any markdown image references.
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
                self.artifacts.write_text("results/report.md", draft_path.read_text(encoding="utf-8"))
        else:
            self.artifacts.write_text("results/report.md", draft_path.read_text(encoding="utf-8"))

        return {"report": "results/report.md"}
