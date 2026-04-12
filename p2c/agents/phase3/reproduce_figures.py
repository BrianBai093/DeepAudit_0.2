"""ReproduceFiguresAgent — generates comparison charts (paper vs reproduced)."""

from __future__ import annotations

import csv
import json
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
        claims_doc = self._safe_read("fingerprint/claims_ir.json")
        repo_figure_data = _load_repo_figure_data(ctx, metrics)

        # Ensure figures directory exists
        figures_dir = self.artifacts.path("results/figures").resolve()
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
            fig = self._reproduce_element(
                elem, verdict, metrics, claims_doc, repo_figure_data, figures_dir
            )
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

        output_path = str((figures_dir / "verdict_comparison.png").resolve())

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
        claims_doc: dict,
        repo_figure_data: dict,
        figures_dir: Path,
    ) -> ReproducedFigure | None:
        """Use LLM to generate matplotlib code for a specific paper figure."""
        element_id = elem.get("element_id", "unknown")
        output_path = str((figures_dir / f"{element_id}.png").resolve())

        # Gather reproduced metrics
        metric_records = metrics.get("records", [])
        claim_verdicts = verdict.get("claim_verdicts", [])
        matching_claims = _claims_for_element(element_id, claims_doc, claim_verdicts)

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
            self.log("PROGRESS", f"LLM unavailable for {element_id}, using deterministic fallback")
            return self._reproduce_element_deterministic(
                elem, matching_claims, metric_records, output_path, repo_figure_data
            )

        # Extract Python code from response
        code = _extract_python_code(text)
        if not code:
            return self._reproduce_element_deterministic(
                elem, matching_claims, metric_records, output_path, repo_figure_data
            )

        # Safety check
        if _DANGEROUS_IMPORTS.search(code):
            self.log("PROGRESS", f"Rejected unsafe code for {element_id}")
            return self._reproduce_element_deterministic(
                elem, matching_claims, metric_records, output_path, repo_figure_data
            )

        success = self._run_matplotlib(code)
        if not success:
            self.log("PROGRESS", f"LLM chart failed for {element_id}, using deterministic fallback")
            return self._reproduce_element_deterministic(
                elem, matching_claims, metric_records, output_path, repo_figure_data
            )
        return ReproducedFigure(
            element_id=element_id,
            matplotlib_code=code,
            image_path=f"results/figures/{element_id}.png",
            comparison_notes=elem.get("caption", ""),
            reason_codes=["LLM_GENERATED"],
        )

    def _reproduce_element_deterministic(
        self,
        elem: dict,
        matching_claims: list[dict],
        metric_records: list[dict],
        output_path: str,
        repo_figure_data: dict | None = None,
    ) -> ReproducedFigure | None:
        element_id = elem.get("element_id", "unknown")
        code = _build_visual_element_chart_code(
            elem=elem,
            matching_claims=matching_claims,
            metric_records=metric_records,
            repo_figure_data=repo_figure_data or {},
            output_path=output_path,
        )
        success = self._run_matplotlib(code)
        return ReproducedFigure(
            element_id=element_id,
            matplotlib_code=code,
            image_path=f"results/figures/{element_id}.png" if success else "",
            comparison_notes=(
                elem.get("caption", "")
                or "Paper figure with available reproduced evidence summary"
            ),
            reason_codes=["DETERMINISTIC_FALLBACK"] if success else ["DETERMINISTIC_CHART_FAILED"],
        )

    def _run_matplotlib(self, code: str) -> bool:
        """Execute matplotlib code in a subprocess."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.artifacts.path(".").resolve()),
            )
            if proc.returncode == 0 and "FIGURE_SAVED" in proc.stdout:
                return True
            diagnostic = (proc.stderr or proc.stdout or "")[-1000:]
            self.log("PROGRESS", f"matplotlib failed: {diagnostic}")
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

# Add status labels using ASCII text so headless servers do not need emoji fonts.
for i, s in enumerate(statuses):
    marker = "OK" if s == "SUPPORTED" else "DIFF" if s == "NOT_SUPPORTED" else "NA"
    color = "#2ecc71" if s == "SUPPORTED" else "#e74c3c" if s == "NOT_SUPPORTED" else "#95a5a6"
    ax.annotate(marker, (x[i], max(targets[i], reproduced[i])),
                textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8,
                color=color)

plt.tight_layout()
fig.savefig({output_path!r}, dpi=150, bbox_inches="tight")
print("FIGURE_SAVED")
'''


def _claims_for_element(
    element_id: str,
    claims_doc: dict[str, Any],
    claim_verdicts: list[dict],
) -> list[dict]:
    claims_by_id = {
        str(claim.get("claim_id")): claim
        for claim in claims_doc.get("claims", [])
        if isinstance(claim, dict) and claim.get("claim_id")
    }
    verdicts_by_id = {
        str(row.get("claim_id")): row
        for row in claim_verdicts
        if isinstance(row, dict) and row.get("claim_id")
    }
    rows: list[dict] = []
    for claim_id, claim in claims_by_id.items():
        conditions = claim.get("conditions", {})
        if not isinstance(conditions, dict):
            continue
        visual_data = conditions.get("visual_data", {})
        table_anchor = str(conditions.get("table_anchor") or "")
        visual_id = ""
        if isinstance(visual_data, dict):
            visual_id = str(visual_data.get("element_id") or "")
        if visual_id != element_id and table_anchor != element_id:
            continue
        verdict = verdicts_by_id.get(claim_id, {})
        rows.append({
            "claim_id": claim_id,
            "predicate": claim.get("predicate") or claim.get("metric") or claim_id,
            "target": verdict.get("target_value", claim.get("target")),
            "reproduced": verdict.get("compared_value"),
            "status": verdict.get("status", "INCONCLUSIVE"),
            "detail": verdict.get("detail", ""),
        })
    return rows


def _load_repo_figure_data(ctx: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Load repo-produced figure data that Phase 3 can plot deterministically."""
    repo_dir_raw = ctx.get("repo_dir")
    repo_dir = Path(str(repo_dir_raw)).expanduser() if repo_dir_raw else None
    if repo_dir and not repo_dir.is_absolute():
        repo_dir = repo_dir.resolve()

    roc_points: list[dict[str, float]] = []
    threshold_csv = repo_dir / "metrics" / "threshold_metrics.csv" if repo_dir else None
    if threshold_csv and threshold_csv.exists():
        try:
            with threshold_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    fp = _float_or_none(row.get("fp"))
                    tn = _float_or_none(row.get("tn"))
                    tp = _float_or_none(row.get("tp"))
                    fn = _float_or_none(row.get("fn"))
                    threshold = _float_or_none(row.get("threshold"))
                    if fp is None or tn is None or tp is None or fn is None:
                        continue
                    fpr_den = fp + tn
                    tpr_den = tp + fn
                    if fpr_den <= 0 or tpr_den <= 0:
                        continue
                    roc_points.append({
                        "x": fp / fpr_den,
                        "y": tp / tpr_den,
                        "threshold": threshold if threshold is not None else 0.0,
                    })
        except Exception:  # noqa: BLE001
            roc_points = []

    auc_values: list[float] = []
    for record in metrics.get("records", []):
        if not isinstance(record, dict):
            continue
        metric_name = str(record.get("metric_name") or "").lower()
        if metric_name != "roc_auc" and not metric_name.endswith("_roc_auc"):
            continue
        value = _float_or_none(record.get("value"))
        if value is None:
            continue
        if value not in auc_values:
            auc_values.append(value)

    roc_images: list[dict[str, str]] = []
    figures_dir = repo_dir / "figures" if repo_dir else None
    if figures_dir and figures_dir.exists():
        for image_path in figures_dir.glob("roc_*.png"):
            roc_images.append({
                "path": str(image_path.resolve()),
                "name": image_path.name,
            })

    def image_priority(row: dict[str, str]) -> tuple[int, str]:
        name = row.get("name", "").lower()
        if "xgboost" in name or "xgb" in name:
            return (0, name)
        if "random_forest" in name or "randomforest" in name:
            return (1, name)
        if "logistic" in name or "lr" in name:
            return (2, name)
        return (3, name)

    return {
        "repo_dir": str(repo_dir) if repo_dir else "",
        "threshold_metrics_csv": str(threshold_csv) if threshold_csv and threshold_csv.exists() else "",
        "roc_points": sorted(roc_points, key=lambda point: point["x"]),
        "roc_auc_values": auc_values,
        "roc_image_paths": sorted(roc_images, key=image_priority),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_visual_element_chart_code(
    *,
    elem: dict,
    matching_claims: list[dict],
    metric_records: list[dict],
    repo_figure_data: dict,
    output_path: str,
) -> str:
    """Build deterministic matplotlib code for a paper visual element."""
    payload = {
        "elem": elem,
        "claims": matching_claims[:16],
        "metrics": [
            {
                "metric_name": row.get("metric_name"),
                "value": row.get("value"),
                "source": row.get("source"),
            }
            for row in metric_records[:20]
            if isinstance(row, dict)
        ],
        "repo_figure_data": repo_figure_data,
        "output_path": output_path,
    }
    payload_json = json.dumps(payload, ensure_ascii=True)

    return f'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

payload = json.loads({payload_json!r})
elem = payload["elem"]
claims = payload["claims"]
metrics = payload["metrics"]
repo_figure_data = payload.get("repo_figure_data", {{}})
output_path = payload["output_path"]

GREEN = "#2ecc71"
RED = "#e74c3c"
GRAY = "#95a5a6"
BLUE = "#3498db"
ORANGE = "#f39c12"


def short(text, max_len=34):
    text = str(text or "")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def numeric_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def point_xy(points):
    xs = []
    ys = []
    labels = []
    for idx, point in enumerate(points or []):
        if not isinstance(point, dict):
            continue
        x = point.get("x", idx)
        y = point.get("y")
        y_val = numeric_or_none(y)
        if y_val is None:
            continue
        x_val = numeric_or_none(x)
        if x_val is None:
            labels.append(str(x))
            x_val = float(len(labels) - 1)
        xs.append(x_val)
        ys.append(y_val)
    return xs, ys, labels


def plot_paper(ax):
    chart_type = str(elem.get("chart_type") or "line").lower()
    data_series = elem.get("data_series") or []
    if not data_series:
        ax.text(0.5, 0.5, "No extracted paper data", ha="center", va="center")
        return

    if chart_type == "bar":
        series_count = max(1, len(data_series))
        width = min(0.8 / series_count, 0.35)
        any_labels = []
        for sidx, series in enumerate(data_series):
            xs, ys, labels = point_xy(series.get("values", []))
            if not xs:
                continue
            offset = (sidx - (series_count - 1) / 2.0) * width
            ax.bar(np.array(xs) + offset, ys, width=width, label=short(series.get("name", f"series {{sidx + 1}}")))
            if labels:
                any_labels = labels
        if any_labels:
            ax.set_xticks(np.arange(len(any_labels)))
            ax.set_xticklabels([short(x, 16) for x in any_labels], rotation=45, ha="right")
    else:
        for sidx, series in enumerate(data_series):
            xs, ys, _labels = point_xy(series.get("values", []))
            if not xs:
                continue
            label = short(series.get("name", f"series {{sidx + 1}}"))
            if chart_type == "scatter":
                ax.scatter(xs, ys, label=label, s=24)
            else:
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=label)

    axis_labels = elem.get("axis_labels") or {{}}
    ax.set_xlabel(axis_labels.get("x") or "x")
    ax.set_ylabel(axis_labels.get("y") or "value")
    ax.grid(True, alpha=0.3)
    if len(data_series) <= 10:
        ax.legend(fontsize=8, loc="best")


def plot_reproduced(ax):
    axis_labels = elem.get("axis_labels") or {{}}
    caption = str(elem.get("caption") or "")
    is_roc = (
        "false positive" in str(axis_labels.get("x", "")).lower()
        and "true positive" in str(axis_labels.get("y", "")).lower()
    ) or "aucroc" in caption.lower() or "roc" in caption.lower()
    repo_roc_points = repo_figure_data.get("roc_points") or []
    repo_roc_images = repo_figure_data.get("roc_image_paths") or []
    if is_roc and repo_roc_images:
        selected = repo_roc_images[0]
        try:
            img = mpimg.imread(selected["path"])
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title("Repo-generated ROC: " + short(selected.get("name"), 38))
            auc_values = repo_figure_data.get("roc_auc_values") or []
            if auc_values:
                auc_text = "ROC-AUC: " + ", ".join(f"{{float(v):.4f}}" for v in auc_values[:3])
                ax.text(0.03, 0.97, auc_text, ha="left", va="top",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRAY, alpha=0.9))
            return
        except Exception:
            pass

    if is_roc and repo_roc_points:
        xs = [float(point["x"]) for point in repo_roc_points]
        ys = [float(point["y"]) for point in repo_roc_points]
        thresholds = [point.get("threshold") for point in repo_roc_points]
        auc_values = repo_figure_data.get("roc_auc_values") or []
        auc_label = ""
        if auc_values:
            auc_label = " | ROC-AUC " + ", ".join(f"{{float(v):.4f}}" for v in auc_values[:3])
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, color=ORANGE,
                label="Repo threshold sweep" + auc_label)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2, label="Baseline")
        for idx, (xv, yv, thr) in enumerate(zip(xs, ys, thresholds)):
            if idx in {{0, len(xs) // 2, len(xs) - 1}}:
                ax.annotate(f"thr={{float(thr):.1f}}", (xv, yv),
                            textcoords="offset points", xytext=(5, -10), fontsize=7)
        ax.set_xlabel(axis_labels.get("x") or "False Positive Rate")
        ax.set_ylabel(axis_labels.get("y") or "True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        source = repo_figure_data.get("threshold_metrics_csv") or "repo threshold metrics"
        ax.text(0.03, 0.97, "Reproduced from\\n" + short(source, 42),
                ha="left", va="top", transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRAY, alpha=0.9))
        return

    comparable = [
        c for c in claims
        if c.get("target") is not None and c.get("reproduced") is not None
    ]
    if comparable:
        labels = [short(c.get("claim_id"), 18) for c in comparable]
        targets = [float(c.get("target")) for c in comparable]
        reproduced = [float(c.get("reproduced")) for c in comparable]
        colors = [
            GREEN if c.get("status") == "SUPPORTED" else RED if c.get("status") == "NOT_SUPPORTED" else GRAY
            for c in comparable
        ]
        x = np.arange(len(comparable))
        width = 0.35
        ax.bar(x - width / 2, targets, width, label="Paper target", color=BLUE, alpha=0.75)
        ax.bar(x + width / 2, reproduced, width, label="Reproduced", color=colors, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        return

    status_counts = {{"SUPPORTED": 0, "NOT_SUPPORTED": 0, "INCONCLUSIVE": 0}}
    for c in claims:
        status = str(c.get("status") or "INCONCLUSIVE")
        status_counts[status] = status_counts.get(status, 0) + 1

    text_lines = []
    if claims:
        text_lines.append("No aligned numeric reproduced series.")
        text_lines.append("")
        text_lines.append("Claim statuses:")
        for key in ["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]:
            if status_counts.get(key):
                text_lines.append(f"- {{key}}: {{status_counts[key]}}")
    else:
        text_lines.append("No claims were linked to this figure.")

    useful_metrics = []
    for row in metrics:
        name = str(row.get("metric_name") or "")
        value = row.get("value")
        if value is None:
            continue
        if len(useful_metrics) >= 8:
            break
        useful_metrics.append(f"{{short(name, 22)}} = {{value}}")
    if useful_metrics:
        text_lines.append("")
        text_lines.append("Available execution metrics:")
        text_lines.extend(f"- {{line}}" for line in useful_metrics)

    ax.text(
        0.03, 0.97, "\\n".join(text_lines),
        ha="left", va="top", transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRAY, alpha=0.95),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_paper(axes[0])
plot_reproduced(axes[1])

caption = elem.get("caption") or elem.get("element_id") or "Paper figure"
axes[0].set_title("Paper extracted")
axes[1].set_title("Reproduced evidence")
fig.suptitle(short(caption, 90), fontsize=13)
plt.tight_layout()
fig.savefig(output_path, dpi=150, bbox_inches="tight")
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
