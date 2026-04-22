"""ExtractVisualElementsAgent — extracts figures and tables from paper PDF via vision API."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import VisualElement, VisualElementsDoc

SCAN_SYSTEM_PROMPT = (
    "You are analyzing pages from an academic paper PDF. "
    "Identify which pages contain figures (charts, plots, diagrams) or data tables. "
    "Return a JSON object: {\"figure_pages\": [1, 3, 5]} listing 1-indexed page numbers."
)

EXTRACT_SYSTEM_PROMPT = """\
You are an expert at reading academic paper figures and tables.
For each figure or table on the page(s), extract structured data.

Return a JSON object with this structure:
{
  "elements": [
    {
      "element_id": "fig_1",
      "element_type": "figure" or "table",
      "page": <int>,
      "caption": "full caption text",
      "chart_type": "bar" | "line" | "scatter" | "heatmap" | "table" | "diagram" | "other",
      "axis_labels": {"x": "label", "y": "label"},
      "legend_entries": ["series1", "series2"],
      "data_series": [
        {"name": "Method A", "values": [{"x": "Dataset1", "y": 0.95}, {"x": "Dataset2", "y": 0.91}]}
      ],
      "visual_anchor": "Figure 1",
      "bbox": {"x0": 0.10, "y0": 0.20, "x1": 0.90, "y1": 0.65},
      "x_axis_range": [0, 1],
      "y_axis_range": [0, 1],
      "series_semantics": [{"name": "Method A", "model": "AutoEncoder", "metric": "ROC-AUC"}],
      "model_names": ["AutoEncoder"],
      "sampling_strategy": "under-sampling" | "over-sampling" | "none" | null,
      "numeric_confidence": 0.85
    }
  ]
}

Rules:
- Extract APPROXIMATE numeric values from chart bars/lines/points as accurately as possible.
- bbox must be normalized page coordinates from 0 to 1 around the visual element.
- For tables: use data_series with one entry per row, values as [{column_header: cell_value}, ...].
- For classification-report tables, preserve row labels and column names in values.
- For heatmaps/confusion matrices, use data_series values containing matrix cells with row/column labels.
- If a figure is a diagram/architecture without numeric data, set chart_type="diagram" and data_series=[].
- Use the figure/table numbering from the paper (e.g., "Figure 3", "Table 2").
- Extract model_names and sampling_strategy from caption, legend, labels, or nearby title text.
- numeric_confidence is your confidence in the extracted numeric values, from 0.0 to 1.0.
- Only extract elements with reproducibility-relevant data (metrics, results, comparisons).
"""

_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "element_type": {"type": "string", "enum": ["figure", "table"]},
                    "page": {"type": "integer"},
                    "caption": {"type": "string"},
                    "chart_type": {"type": ["string", "null"]},
                    "axis_labels": {"type": "object"},
                    "legend_entries": {"type": "array", "items": {"type": "string"}},
                    "data_series": {"type": "array"},
                    "visual_anchor": {"type": "string"},
                    "bbox": {"type": "object"},
                    "x_axis_range": {"type": ["array", "null"]},
                    "y_axis_range": {"type": ["array", "null"]},
                    "series_semantics": {"type": "array"},
                    "model_names": {"type": "array", "items": {"type": "string"}},
                    "sampling_strategy": {"type": ["string", "null"]},
                    "numeric_confidence": {"type": ["number", "null"]},
                },
                "required": ["element_id", "element_type", "page"],
            },
        },
    },
    "required": ["elements"],
}


def resolve_existing_pdf_path(raw_pdf_path: str | Path, ctx: dict[str, Any]) -> tuple[Path, Path | None]:
    """Resolve a possibly stale PDF path to an existing paper PDF.

    MinerU-style paper folders often contain UUID-prefixed ``*_origin.pdf`` files.
    A rerun can leave ``--paper_pdf`` pointing at an older UUID while the current
    folder has exactly the PDF we need under a different UUID.
    """
    requested = Path(raw_pdf_path).expanduser()
    if requested.exists():
        return requested, requested

    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        path = path.expanduser()
        if path not in candidates:
            candidates.append(path)

    def add_pdf_candidates(directory: Path) -> None:
        if not directory.exists() or not directory.is_dir():
            return
        for pattern in ("*_origin.pdf", "*.pdf"):
            for path in sorted(directory.glob(pattern)):
                add_candidate(path)

    if requested.parent != Path("."):
        add_pdf_candidates(requested.parent)

    for key in ("paper_md_out", "paper_md"):
        raw = ctx.get(key)
        if not raw:
            continue
        paper_path = Path(str(raw)).expanduser()
        add_pdf_candidates(paper_path.parent)
        add_candidate(paper_path.parent.parent / "paper.pdf")

    repo_dir = ctx.get("repo_dir")
    if repo_dir:
        add_candidate(Path(str(repo_dir)).expanduser().parent / "paper.pdf")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return requested, candidate

    return requested, None


class ExtractVisualElementsAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="extract_visual_elements", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        pdf_path = ctx.get("paper_pdf")
        if not pdf_path:
            self.log("PROGRESS", "No paper_pdf provided, writing empty visual elements")
            doc = VisualElementsDoc(reason_codes=["NO_PDF_PROVIDED"])
            self.artifacts.write_json("fingerprint/visual_elements.json", doc.model_dump())
            return {"visual_elements": doc.model_dump()}

        requested_pdf_path, resolved_pdf_path = resolve_existing_pdf_path(pdf_path, ctx)
        if not resolved_pdf_path:
            self.log("PROGRESS", f"PDF not found: {requested_pdf_path}")
            doc = VisualElementsDoc(reason_codes=["PDF_NOT_FOUND"])
            self.artifacts.write_json("fingerprint/visual_elements.json", doc.model_dump())
            return {"visual_elements": doc.model_dump()}
        pdf_path = resolved_pdf_path
        if pdf_path != requested_pdf_path:
            ctx["paper_pdf"] = str(pdf_path)
            self.log("PROGRESS", f"PDF not found: {requested_pdf_path}; using fallback PDF: {pdf_path}")

        # Render PDF pages
        try:
            from p2c.utils.pdf_extract import pdf_to_page_images, write_page_image_assets
            pages = pdf_to_page_images(pdf_path, dpi=150)
            page_image_paths = write_page_image_assets(
                pages,
                self.artifacts.path("fingerprint/page_images"),
            )
        except ImportError:
            self.log("PROGRESS", "PyMuPDF not installed, skipping visual extraction")
            doc = VisualElementsDoc(reason_codes=["PYMUPDF_UNAVAILABLE"])
            self.artifacts.write_json("fingerprint/visual_elements.json", doc.model_dump())
            return {"visual_elements": doc.model_dump()}
        except Exception as e:  # noqa: BLE001
            self.log("PROGRESS", f"PDF rendering failed: {e}")
            doc = VisualElementsDoc(reason_codes=["PDF_RENDER_FAILED"])
            self.artifacts.write_json("fingerprint/visual_elements.json", doc.model_dump())
            return {"visual_elements": doc.model_dump()}

        total_pages = len(pages)
        self.log("PROGRESS", f"Rendered {total_pages} pages from PDF")

        # Pass 1: quick scan to find pages with figures/tables (low detail, saves tokens)
        figure_page_nums = self._scan_for_figure_pages(pages)
        if not figure_page_nums:
            self.log("PROGRESS", "No figure/table pages detected")
            doc = VisualElementsDoc(page_count=total_pages, reason_codes=["NO_FIGURES_DETECTED"])
            self.artifacts.write_json("fingerprint/visual_elements.json", doc.model_dump())
            return {"visual_elements": doc.model_dump()}

        self.log("PROGRESS", f"Found figures/tables on pages: {figure_page_nums}")

        # Pass 2: detailed extraction from figure pages
        figure_pages = [(pn, uri) for pn, uri in pages if pn in figure_page_nums]
        all_elements = self._extract_elements(figure_pages, page_image_paths=page_image_paths)

        doc = VisualElementsDoc(
            elements=all_elements,
            page_count=total_pages,
            reason_codes=["VISION_EXTRACTION_COMPLETE"],
        )
        self.artifacts.write_json("fingerprint/visual_elements.json", doc.model_dump())
        self.log("DONE", f"Extracted {len(all_elements)} visual elements from {len(figure_page_nums)} pages")
        return {"visual_elements": doc.model_dump()}

    def _scan_for_figure_pages(self, pages: list[tuple[int, str]]) -> set[int]:
        """Quick low-detail scan to identify pages containing figures or tables."""
        figure_pages: set[int] = set()

        # Send pages in batches of 4 at low detail
        batch_size = 4
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            images = [uri for _, uri in batch]
            page_nums = [pn for pn, _ in batch]

            user_text = (
                f"These are pages {page_nums} from an academic paper. "
                "Which pages contain figures (charts, plots, diagrams) or data tables with numbers? "
                "Return JSON: {\"figure_pages\": [page_numbers_here]}"
            )

            data, err = self.safe_chat_vision_json(
                schema={"type": "object", "properties": {"figure_pages": {"type": "array"}}, "required": ["figure_pages"]},
                system=SCAN_SYSTEM_PROMPT,
                user_text=user_text,
                images=images,
                detail="low",
            )
            if data and "figure_pages" in data:
                for pn in data["figure_pages"]:
                    if isinstance(pn, (int, float)):
                        figure_pages.add(int(pn))

        return figure_pages

    def _extract_elements(
        self,
        pages: list[tuple[int, str]],
        *,
        page_image_paths: dict[int, Path] | None = None,
    ) -> list[VisualElement]:
        """Detailed extraction of figures and tables from identified pages."""
        all_elements: list[VisualElement] = []

        # Process pages in pairs for detailed extraction
        batch_size = 2
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            images = [uri for _, uri in batch]
            page_nums = [pn for pn, _ in batch]

            user_text = (
                f"Extract all figures and tables with data from page(s) {page_nums}. "
                "Read approximate numeric values from any charts. "
                "For tables, extract all rows and columns."
            )

            data, err = self.safe_chat_vision_json(
                schema=_EXTRACT_SCHEMA,
                system=EXTRACT_SYSTEM_PROMPT,
                user_text=user_text,
                images=images,
                detail="high",
            )

            if not data:
                self.log("PROGRESS", f"Vision extraction failed for pages {page_nums}: {err}")
                continue

            for elem_dict in data.get("elements", []):
                try:
                    elem = VisualElement(
                        element_id=elem_dict.get("element_id", f"elem_{len(all_elements)}"),
                        element_type=elem_dict.get("element_type", "figure"),
                        page=elem_dict.get("page", page_nums[0]),
                        caption=elem_dict.get("caption", ""),
                        chart_type=elem_dict.get("chart_type"),
                        axis_labels=elem_dict.get("axis_labels", {}),
                        legend_entries=elem_dict.get("legend_entries", []),
                        data_series=elem_dict.get("data_series", []),
                        visual_anchor=elem_dict.get("visual_anchor", ""),
                        bbox=_normalize_bbox(elem_dict.get("bbox")),
                        x_axis_range=_axis_range(elem_dict.get("x_axis_range")),
                        y_axis_range=_axis_range(elem_dict.get("y_axis_range")),
                        series_semantics=[
                            row for row in elem_dict.get("series_semantics", [])
                            if isinstance(row, dict)
                        ],
                        model_names=[
                            str(name).strip()
                            for name in elem_dict.get("model_names", [])
                            if str(name).strip()
                        ],
                        sampling_strategy=elem_dict.get("sampling_strategy"),
                        numeric_confidence=_float_or_none(elem_dict.get("numeric_confidence")),
                        reason_codes=["VISION_EXTRACTED"],
                    )
                    self._attach_image_paths(elem, page_image_paths or {})
                    all_elements.append(elem)
                except Exception:  # noqa: BLE001
                    continue

        return all_elements

    def _attach_image_paths(self, elem: VisualElement, page_image_paths: dict[int, Path]) -> None:
        page_image = page_image_paths.get(int(elem.page))
        if not page_image:
            elem.reason_codes.append("PAGE_IMAGE_MISSING")
            return
        elem.raw_page_image = self._relative_artifact_path(page_image)
        if not elem.bbox:
            elem.reason_codes.append("BBOX_MISSING")
            return

        try:
            from p2c.utils.pdf_extract import crop_image_by_normalized_bbox
            crop_name = _safe_image_name(elem.element_id)
            crop_path = self.artifacts.path(f"fingerprint/visual_crops/{crop_name}.png")
            if crop_image_by_normalized_bbox(page_image, elem.bbox, crop_path):
                elem.crop_path = self._relative_artifact_path(crop_path)
            else:
                elem.reason_codes.append("CROP_FAILED")
        except Exception:  # noqa: BLE001
            elem.reason_codes.append("CROP_FAILED")

    def _relative_artifact_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.artifacts.run_root.resolve()).as_posix()
        except Exception:  # noqa: BLE001
            return path.as_posix()


def _safe_image_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "visual")).strip("_")
    return name or "visual"


def _normalize_bbox(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    aliases = {
        "x0": ("x0", "left", "xmin"),
        "y0": ("y0", "top", "ymin"),
        "x1": ("x1", "right", "xmax"),
        "y1": ("y1", "bottom", "ymax"),
    }
    out: dict[str, float] = {}
    for key, names in aliases.items():
        for name in names:
            if name in raw:
                value = _float_or_none(raw.get(name))
                if value is not None:
                    out[key] = value
                break
    if set(out) != {"x0", "y0", "x1", "y1"}:
        return {}
    if not (0.0 <= out["x0"] < out["x1"] <= 1.0 and 0.0 <= out["y0"] < out["y1"] <= 1.0):
        return {}
    return out


def _axis_range(raw: Any) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    a = _float_or_none(raw[0])
    b = _float_or_none(raw[1])
    if a is None or b is None:
        return None
    return [a, b]


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
