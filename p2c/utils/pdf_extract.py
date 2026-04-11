"""PDF page rendering utility using PyMuPDF (fitz)."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def pdf_to_page_images(
    pdf_path: Path,
    dpi: int = 200,
) -> list[tuple[int, str]]:
    """Render each PDF page as a base64 PNG data URI.

    Returns a list of ``(page_number, data_uri)`` tuples (1-indexed pages).
    Requires ``PyMuPDF`` (``fitz``).
    """
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF visual extraction. "
            "Install with: pip install PyMuPDF"
        ) from e

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page at specified DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        png_bytes = pix.tobytes("png")

        # Encode as data URI
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"

        pages.append((page_num + 1, data_uri))  # 1-indexed

    doc.close()
    logger.info("Rendered %d pages from %s at %d DPI", len(pages), pdf_path.name, dpi)
    return pages


def page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF without rendering."""
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        return 0
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
