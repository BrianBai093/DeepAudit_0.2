"""ExtractVisualElementsAgent — extracts figures and tables from paper PDF via vision API."""

from __future__ import annotations

import json
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
      "visual_anchor": "Figure 1"
    }
  ]
}

Rules:
- Extract APPROXIMATE numeric values from chart bars/lines/points as accurately as possible.
- For tables: use data_series with one entry per row, values as [{column_header: cell_value}, ...].
- If a figure is a diagram/architecture without numeric data, set chart_type="diagram" and data_series=[].
- Use the figure/table numbering from the paper (e.g., "Figure 3", "Table 2").
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
                },
                "required": ["element_id", "element_type", "page"],
            },
        },
    },
    "required": ["elements"],
}


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

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.log("PROGRESS", f"PDF not found: {pdf_path}")
            doc = VisualElementsDoc(reason_codes=["PDF_NOT_FOUND"])
            self.artifacts.write_json("fingerprint/visual_elements.json", doc.model_dump())
            return {"visual_elements": doc.model_dump()}

        # Render PDF pages
        try:
            from p2c.utils.pdf_extract import pdf_to_page_images
            pages = pdf_to_page_images(pdf_path, dpi=150)
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
        all_elements = self._extract_elements(figure_pages)

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

    def _extract_elements(self, pages: list[tuple[int, str]]) -> list[VisualElement]:
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
                        reason_codes=["VISION_EXTRACTED"],
                    )
                    all_elements.append(elem)
                except Exception:  # noqa: BLE001
                    continue

        return all_elements
