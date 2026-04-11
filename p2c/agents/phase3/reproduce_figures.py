"""ReproduceFiguresAgent — generates comparison charts (paper vs reproduced)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import ReproducedFigure, ReproducedFiguresDoc

# Disallowed imports in generated matplotlib code
_DANGEROUS_IMPORTS = re.compile(
    r"\bimport\s+(?:os|subprocess|socket|shutil|http|urllib|requests|sys)\b"
    r"|\bfrom\s+(?:os|subprocess|socket|shutil|http|urllib|requests|sys)\b"
)

MATPLOTLIB_SYSTEM_PROMPT = """\
You generate self-contained Python scripts using matplotlib to create comparison figures.
The script MUST:
- Only import matplotlib, numpy, and json (no os, subprocess, etc.)
- Save the figure to the exact path provided
- Use plt.tight_layout() and fig.savefig(path, dpi=150, bbox_inches='tight')
- Create a clear side-by-side comparison: paper values vs reproduced values
- Color-code: green (#2ecc71) for within tolerance, red (#e74c3c) for outside, gray (#95a5a6) for missing
- Include axis labels and a legend
- Print "FIGURE_SAVED" to stdout on success
"""


class ReproduceFiguresAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="reproduce_figures", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        # Always try to generate at least a verdict comparison chart
        verdict = self._safe_read("results/verdict.json")
        metrics = self._safe_read("results/metrics.json")
        visual_elements = self._safe_read("fingerprint/visual_elements.json")

        # Ensure figures directory exists
        figures_dir = self.artifacts.path("results/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        reproduced: list[ReproducedFigure] = []

        # 1. Generate verdict comparison bar chart (always, doesn't need PDF)
        verdict_fig = self._generate_verdict_chart(verdict, figures_dir)
        if verdict_fig:
            reproduced.append(verdict_fig)

        # 2. If visual elements exist, try to reproduce each figure
        elements = visual_elements.get("elements", [])
        chart_elements = [
            e for e in elements
            if e.get("chart_type") in ("bar", "line", "scatter")
            and e.get("data_series")
        ]

        for elem in chart_elements[:5]:  # Limit to 5 figures
            fig = self._reproduce_element(elem, verdict, metrics, figures_dir)
            if fig:
                reproduced.append(fig)

        doc = ReproducedFiguresDoc(
            figures=reproduced,
            reason_codes=["FIGURES_GENERATED"],
        )
        self.artifacts.write_json("results/reproduced_figures.json", doc.model_dump())
        self.log("DONE", f"Generated {len(reproduced)} figures")
        return {"figures": doc.model_dump()}

    def _safe_read(self, path: str) -> dict:
        try:
            return self.artifacts.read_json(path)
        except Exception:  # noqa: BLE001
            return {}

    def _generate_verdict_chart(
        self,
        verdict: dict,
        figures_dir: Path,
    ) -> ReproducedFigure | None:
        """Generate a bar chart comparing paper values vs reproduced values."""
        claim_verdicts = verdict.get("claim_verdicts", [])
        if not claim_verdicts:
            return None

        # Filter to claims with both target and reproduced values
        data_points = []
        for cv in claim_verdicts:
            target = cv.get("target_value")
            reproduced = cv.get("compared_value")
            if target is not None and reproduced is not None:
                data_points.append({
                    "claim_id": cv.get("claim_id", "?"),
                    "target": target,
                    "reproduced": reproduced,
                    "status": cv.get("status", "INCONCLUSIVE"),
                })

        if not data_points:
            return None

        output_path = str(figures_dir / "verdict_comparison.png")

        # Generate deterministic matplotlib code (no LLM needed)
        code = _build_verdict_chart_code(data_points, output_path)

        success = self._run_matplotlib(code)
        return ReproducedFigure(
            element_id="verdict_comparison",
            matplotlib_code=code,
            image_path="results/figures/verdict_comparison.png" if success else "",
            comparison_notes=f"Side-by-side comparison of {len(data_points)} claims",
            reason_codes=["DETERMINISTIC_CHART"] if success else ["CHART_FAILED"],
        )

    def _reproduce_element(
        self,
        elem: dict,
        verdict: dict,
        metrics: dict,
        figures_dir: Path,
    ) -> ReproducedFigure | None:
        """Use LLM to generate matplotlib code for a specific paper figure."""
        element_id = elem.get("element_id", "unknown")
        output_path = str(figures_dir / f"{element_id}.png")

        # Gather reproduced metrics
        metric_records = metrics.get("records", [])
        claim_verdicts = verdict.get("claim_verdicts", [])

        prompt = (
            f"Generate a matplotlib script to reproduce this paper figure.\n\n"
            f"Paper figure info:\n"
            f"- Chart type: {elem.get('chart_type')}\n"
            f"- Axis labels: {elem.get('axis_labels', {})}\n"
            f"- Legend: {elem.get('legend_entries', [])}\n"
            f"- Paper data: {elem.get('data_series', [])}\n\n"
            f"Reproduced metrics available:\n"
            f"{_format_metrics_for_prompt(metric_records, claim_verdicts)}\n\n"
            f"Save the figure to: {output_path}\n"
            f"Create TWO subplots side by side: 'Paper (claimed)' and 'Reproduced'."
        )

        text, err = self.safe_chat_text(MATPLOTLIB_SYSTEM_PROMPT, prompt)
        if not text:
            self.log("PROGRESS", f"LLM unavailable for {element_id}, skipping")
            return None

        # Extract Python code from response
        code = _extract_python_code(text)
        if not code:
            return None

        # Safety check
        if _DANGEROUS_IMPORTS.search(code):
            self.log("PROGRESS", f"Rejected unsafe code for {element_id}")
            return None

        success = self._run_matplotlib(code)
        return ReproducedFigure(
            element_id=element_id,
            matplotlib_code=code,
            image_path=f"results/figures/{element_id}.png" if success else "",
            comparison_notes=elem.get("caption", ""),
            reason_codes=["LLM_GENERATED"] if success else ["LLM_CHART_FAILED"],
        )

    def _run_matplotlib(self, code: str) -> bool:
        """Execute matplotlib code in a subprocess."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.artifacts.path(".")),
            )
            if proc.returncode == 0 and "FIGURE_SAVED" in proc.stdout:
                return True
            self.log("PROGRESS", f"matplotlib failed: {proc.stderr[:200]}")
            return False
        except Exception as e:  # noqa: BLE001
            self.log("PROGRESS", f"matplotlib execution error: {e}")
            return False


def _build_verdict_chart_code(data_points: list[dict], output_path: str) -> str:
    """Build deterministic matplotlib code for verdict comparison chart."""
    import json as _json
    data_json = _json.dumps(data_points)

    return f'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = json.loads({data_json!r})

claims = [d["claim_id"] for d in data]
targets = [d["target"] for d in data]
reproduced = [d["reproduced"] for d in data]
statuses = [d["status"] for d in data]

colors = []
for s in statuses:
    if s == "SUPPORTED":
        colors.append("#2ecc71")
    elif s == "NOT_SUPPORTED":
        colors.append("#e74c3c")
    else:
        colors.append("#95a5a6")

x = np.arange(len(claims))
width = 0.35

fig, ax = plt.subplots(figsize=(max(8, len(claims) * 1.2), 5))
bars1 = ax.bar(x - width/2, targets, width, label="Paper (claimed)", color="#3498db", alpha=0.8)
bars2 = ax.bar(x + width/2, reproduced, width, label="Reproduced", color=colors, alpha=0.8)

ax.set_ylabel("Value")
ax.set_title("Paper Claims vs Reproduced Results")
ax.set_xticks(x)
ax.set_xticklabels(claims, rotation=45, ha="right", fontsize=8)
ax.legend()

# Add status markers
for i, s in enumerate(statuses):
    marker = "\\u2705" if s == "SUPPORTED" else "\\u274c" if s == "NOT_SUPPORTED" else "\\u26a0\\ufe0f"
    ax.annotate(marker, (x[i], max(targets[i], reproduced[i])),
                textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10)

plt.tight_layout()
fig.savefig({output_path!r}, dpi=150, bbox_inches="tight")
print("FIGURE_SAVED")
'''


def _format_metrics_for_prompt(records: list[dict], verdicts: list[dict]) -> str:
    """Format available metrics for the LLM prompt."""
    lines = []
    for r in records[:20]:
        lines.append(f"  {r.get('metric_name')}: {r.get('value')} (source: {r.get('source', 'unknown')})")
    for cv in verdicts[:15]:
        if cv.get("compared_value") is not None:
            lines.append(
                f"  claim {cv.get('claim_id')}: target={cv.get('target_value')}, "
                f"reproduced={cv.get('compared_value')}, status={cv.get('status')}"
            )
    return "\n".join(lines) if lines else "  (no metrics available)"


def _extract_python_code(text: str) -> str | None:
    """Extract Python code block from LLM response."""
    # Try fenced code block first
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back to the entire response if it looks like code
    if "import matplotlib" in text or "plt.savefig" in text:
        return text.strip()
    return None
