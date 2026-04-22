"""PDF page rendering utility using PyMuPDF (fitz)."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def data_uri_to_png_bytes(data_uri: str) -> bytes:
    """Decode a PNG data URI returned by pdf_to_page_images."""
    prefix = "data:image/png;base64,"
    if data_uri.startswith(prefix):
        return base64.b64decode(data_uri[len(prefix):])
    return base64.b64decode(data_uri)


def write_page_image_assets(
    pages: list[tuple[int, str]],
    output_dir: Path,
) -> dict[int, Path]:
    """Persist rendered page data URIs as PNG files keyed by 1-indexed page."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    for page_num, data_uri in pages:
        path = output_dir / f"page_{page_num:03d}.png"
        path.write_bytes(data_uri_to_png_bytes(data_uri))
        out[page_num] = path
    return out


def crop_image_by_normalized_bbox(
    image_path: Path,
    bbox: dict[str, float],
    output_path: Path,
) -> bool:
    """Crop an image using normalized x0/y0/x1/y1 coordinates."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        return False

    try:
        x0 = float(bbox.get("x0", 0.0))
        y0 = float(bbox.get("y0", 0.0))
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
    except (TypeError, ValueError):
        return False
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        width, height = img.size
        box = (
            max(0, min(width, int(round(x0 * width)))),
            max(0, min(height, int(round(y0 * height)))),
            max(0, min(width, int(round(x1 * width)))),
            max(0, min(height, int(round(y1 * height)))),
        )
        if box[0] >= box[2] or box[1] >= box[3]:
            return False
        img.crop(box).save(output_path)
    return True


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
