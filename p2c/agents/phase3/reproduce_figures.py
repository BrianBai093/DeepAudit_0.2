"""ReproduceFiguresAgent — generates comparison charts (paper vs reproduced)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import ReproducedFigure, ReproducedFiguresDoc


class ReproduceFiguresAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="reproduce_figures", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        # Always try to generate at least a verdict comparison chart
        verdict = self._safe_read("results/verdict.json")
        metrics = self._safe_read("results/metrics.json")
        visual_elements = self._safe_read("fingerprint/visual_elements.json")
        claims_doc = self._safe_read("fingerprint/claims_ir.json")
        visual_alignment = self._safe_read("results/visual_to_repo_alignment.json")
        alignment_by_element = _alignment_by_element_id(visual_alignment)

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
            if _is_supported_visual_element(e)
        ]

        for elem in chart_elements:
            element_id = str(elem.get("element_id") or "")
            fig = self._reproduce_element(
                elem,
                verdict,
                metrics,
                claims_doc,
                alignment_by_element.get(element_id, _default_no_match_alignment(element_id)),
                figures_dir,
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
        alignment: dict,
        figures_dir: Path,
    ) -> ReproducedFigure | None:
        """Generate an alignment-controlled comparison chart for a paper visual."""
        element_id = elem.get("element_id", "unknown")
        output_path = str((figures_dir / f"{element_id}.png").resolve())

        # Gather reproduced metrics
        metric_records = metrics.get("records", [])
        claim_verdicts = verdict.get("claim_verdicts", [])
        matching_claims = _claims_for_element(element_id, claims_doc, claim_verdicts)
        return self._reproduce_element_deterministic(
            elem, matching_claims, metric_records, alignment, output_path,
        )

    def _reproduce_element_deterministic(
        self,
        elem: dict,
        matching_claims: list[dict],
        metric_records: list[dict],
        alignment: dict,
        output_path: str,
    ) -> ReproducedFigure | None:
        element_id = elem.get("element_id", "unknown")
        code = _build_visual_element_chart_code(
            elem=elem,
            matching_claims=matching_claims,
            metric_records=metric_records,
            alignment=alignment,
            output_path=output_path,
        )
        success = self._run_matplotlib(code)
        alignment_status = str(alignment.get("status") or "NO_MATCH")
        reasons = [
            str(reason)
            for reason in alignment.get("mismatch_reasons", [])
            if str(reason).strip()
        ]
        note = elem.get("caption") or "Paper figure with reproduced evidence summary"
        if alignment_status == "NO_MATCH":
            suffix = reasons[0] if reasons else "repo has no corresponding visual/data artifact"
            note = f"{note} | visual alignment: NO_MATCH ({suffix})"
        return ReproducedFigure(
            element_id=element_id,
            matplotlib_code=code,
            image_path=f"results/figures/{element_id}.png" if success else "",
            comparison_notes=note,
            reason_codes=_figure_reason_codes(alignment_status, success),
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


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _alignment_by_element_id(alignment_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = alignment_doc.get("alignments", []) if isinstance(alignment_doc, dict) else []
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("element_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("element_id")
    }


def _default_no_match_alignment(element_id: str) -> dict[str, Any]:
    return {
        "element_id": element_id,
        "status": "NO_MATCH",
        "repo_artifact_path": None,
        "artifact_type": None,
        "confidence": 0.0,
        "matched_model_names": [],
        "matched_sampling_strategy": None,
        "matched_metric_names": [],
        "mismatch_reasons": ["visual_to_repo_alignment.json has no row for this visual element"],
        "reason_codes": ["VISUAL_ALIGNMENT_MISSING"],
    }


def _is_supported_visual_element(elem: dict[str, Any]) -> bool:
    if not isinstance(elem, dict):
        return False
    chart_type = str(elem.get("chart_type") or "").lower()
    if chart_type in {"bar", "line", "scatter", "table", "heatmap"}:
        return True
    return str(elem.get("element_type") or "").lower() == "table"


def _figure_reason_codes(alignment_status: str, success: bool) -> list[str]:
    if not success:
        return ["DETERMINISTIC_CHART_FAILED"]
    codes = ["DETERMINISTIC_VISUAL_CHART"]
    if alignment_status == "MATCH":
        codes.append("VISUAL_ALIGNMENT_MATCH")
    else:
        codes.append("VISUAL_ALIGNMENT_NO_MATCH")
    return codes


def _build_visual_element_chart_code(
    *,
    elem: dict,
    matching_claims: list[dict],
    metric_records: list[dict],
    alignment: dict,
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
        "alignment": alignment,
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
alignment = payload.get("alignment", {{}})
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


def extract_table_rows():
    data_series = elem.get("data_series") or []
    rows = []
    for series in data_series:
        if not isinstance(series, dict):
            continue
        series_rows = series.get("rows")
        if isinstance(series_rows, list):
            rows.extend(row for row in series_rows if isinstance(row, dict))
        values = series.get("values")
        if isinstance(values, list):
            rows.extend(row for row in values if isinstance(row, dict))
    if not rows and isinstance(data_series, list):
        rows.extend(row for row in data_series if isinstance(row, dict))
    return rows[:16]


def render_table(ax, rows, title=None):
    ax.set_axis_off()
    if not rows:
        ax.text(0.5, 0.5, "No extracted table cells", ha="center", va="center", wrap=True)
        return
    columns = []
    for row in rows:
        for key in row.keys():
            key = str(key)
            if key not in columns and key not in {{"reason_codes"}}:
                columns.append(key)
        if len(columns) >= 7:
            break
    columns = columns[:7]
    cell_text = []
    for row in rows:
        cell_text.append([short(row.get(column, ""), 18) for column in columns])
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    if title:
        ax.set_title(title)


def extract_heatmap_matrix():
    for source in [elem] + list(elem.get("data_series") or []):
        if not isinstance(source, dict):
            continue
        matrix = source.get("matrix")
        if isinstance(matrix, list) and matrix and all(isinstance(row, list) for row in matrix):
            return matrix, source.get("x_labels") or source.get("columns") or [], source.get("y_labels") or source.get("rows_labels") or []
        cells = source.get("cells")
        if isinstance(cells, list):
            pivot = cell_matrix(cells)
            if pivot[0]:
                return pivot
        values = source.get("values")
        if isinstance(values, list):
            pivot = cell_matrix(values)
            if pivot[0]:
                return pivot
    return [], [], []


def cell_matrix(cells):
    xs = []
    ys = []
    values = {{}}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        x = cell.get("x")
        y = cell.get("y")
        value = numeric_or_none(cell.get("value", cell.get("z")))
        if x is None or y is None or value is None:
            continue
        x = str(x)
        y = str(y)
        if x not in xs:
            xs.append(x)
        if y not in ys:
            ys.append(y)
        values[(x, y)] = value
    if not xs or not ys:
        return [], [], []
    matrix = [[values.get((x, y), 0.0) for x in xs] for y in ys]
    return matrix, xs, ys


def plot_heatmap(ax):
    matrix, x_labels, y_labels = extract_heatmap_matrix()
    if not matrix:
        rows = extract_table_rows()
        render_table(ax, rows, "Paper extracted table")
        return
    arr = np.array(matrix, dtype=float)
    im = ax.imshow(arr, cmap="viridis")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels([short(x, 12) for x in x_labels], rotation=45, ha="right")
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels([short(y, 12) for y in y_labels])
    for row_idx in range(arr.shape[0]):
        for col_idx in range(arr.shape[1]):
            ax.text(col_idx, row_idx, f"{{arr[row_idx, col_idx]:.2g}}",
                    ha="center", va="center", color="white", fontsize=8)


def plot_paper(ax):
    chart_type = str(elem.get("chart_type") or "line").lower()
    if chart_type == "table" or str(elem.get("element_type") or "").lower() == "table":
        render_table(ax, extract_table_rows(), "Paper extracted table")
        return
    if chart_type == "heatmap":
        plot_heatmap(ax)
        return

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
    status = str(alignment.get("status") or "NO_MATCH")
    artifact_path = str(alignment.get("repo_artifact_path") or "")
    artifact_type = str(alignment.get("artifact_type") or "")
    if status != "MATCH":
        reasons = [
            str(reason)
            for reason in alignment.get("mismatch_reasons", [])
            if str(reason).strip()
        ]
        if not reasons:
            reasons = ["No strict visual-to-repo alignment was produced."]
        text_lines = [
            "No matching repo artifact/data for this paper visual.",
            "",
            "This figure is not replaced by another model or experiment.",
            "",
            "Reasons:",
        ]
        text_lines.extend("- " + short(reason, 70) for reason in reasons[:6])
        ax.text(
            0.04, 0.96, "\\n".join(text_lines),
            ha="left", va="top", transform=ax.transAxes, fontsize=9, wrap=True,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRAY, alpha=0.95),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        return

    if artifact_path and (
        artifact_type == "image"
        or artifact_path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ):
        try:
            img = mpimg.imread(artifact_path)
            ax.imshow(img)
            ax.set_axis_off()
            title_bits = ["Matched repo artifact"]
            models = alignment.get("matched_model_names") or []
            metrics = alignment.get("matched_metric_names") or []
            if models:
                title_bits.append(", ".join(str(x) for x in models[:3]))
            if metrics:
                title_bits.append(", ".join(str(x) for x in metrics[:3]))
            ax.set_title(short(" | ".join(title_bits), 70))
            return
        except Exception:
            text = "Matched repo image could not be rendered.\\n" + short(artifact_path, 80)
            ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            return

    if artifact_path:
        text_lines = [
            "Matched repo data artifact.",
            "",
            "Path:",
            short(artifact_path, 82),
        ]
        matched_metrics = alignment.get("matched_metric_names") or []
        matched_models = alignment.get("matched_model_names") or []
        if matched_models:
            text_lines.extend(["", "Model:", ", ".join(str(x) for x in matched_models)])
        if matched_metrics:
            text_lines.extend(["", "Metrics:", ", ".join(str(x) for x in matched_metrics)])
        ax.text(
            0.04, 0.96, "\\n".join(text_lines),
            ha="left", va="top", transform=ax.transAxes, fontsize=9, wrap=True,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRAY, alpha=0.95),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
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
