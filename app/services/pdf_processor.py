import logging
from dataclasses import dataclass
from typing import Iterator

import cv2
import fitz
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PdfPage:
    number: int
    text: str | None = None
    image: np.ndarray | None = None
    render_scale: float = 1.0
    error: str | None = None


def _render(page: fitz.Page, dpi: int, max_side: int) -> tuple[np.ndarray, float]:
    rect = page.rect
    longest_pt = max(rect.width, rect.height)
    scale = dpi / 72
    if longest_pt * scale > max_side:
        scale = max_side / longest_pt
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), scale


def iter_pdf_pages(content: bytes, *, dpi: int, max_side: int, min_text_chars: int) -> Iterator[PdfPage]:
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        for index in range(doc.page_count):
            number = index + 1
            try:
                page = doc.load_page(index)
                text = page.get_text("text") or ""
                if len(text.strip()) >= min_text_chars:
                    result = PdfPage(number, text=text)
                else:
                    image, scale = _render(page, dpi, max_side)
                    result = PdfPage(number, image=image, render_scale=scale)
            except Exception:
                logger.warning("Failed to read PDF page %d", number, exc_info=True)
                result = PdfPage(number, error="Page could not be read.")
            yield result
    finally:
        doc.close()