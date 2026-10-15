import logging
import math
import statistics
from dataclasses import dataclass, field

import numpy as np

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.pledge_slip import OcrLine, PageResult
from app.services.file_validator import ValidatedFile
from app.services.image_preprocessing import decode_image, enhance, map_points_back, resize_for_ocr, rotate_image
from app.services.ocr_service import OcrEngine, OcrService, RawLine
from app.services.pdf_processor import iter_pdf_pages

logger = logging.getLogger(__name__)

MIN_MEASURABLE_TEXT = 3
MIN_LINES_FOR_ORIENTATION = 5


@dataclass
class PageImage:
    image: np.ndarray
    scale: float


@dataclass
class PipelineResult:
    pages: list[PageResult] = field(default_factory=list)
    lines: list[OcrLine] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    images: dict[int, PageImage] = field(default_factory=dict)


def _text_score(raw: list[RawLine]) -> float:
    return sum((r.confidence or 0.0) * len(r.text.strip()) for r in raw)


def _measurable(raw: list[RawLine]) -> list[RawLine]:
    return [r for r in raw if r.box and len(r.box) == 4 and len(r.text.strip()) >= MIN_MEASURABLE_TEXT]


def _mostly_vertical(raw: list[RawLine]) -> bool:
    lines = _measurable(raw)
    if len(lines) < MIN_LINES_FOR_ORIENTATION:
        return False
    vertical = 0
    for r in lines:
        xs = [p[0] for p in r.box]
        ys = [p[1] for p in r.box]
        if (max(ys) - min(ys)) > 1.5 * (max(xs) - min(xs)):
            vertical += 1
    return vertical / len(lines) > 0.6


def _skew_angle(raw: list[RawLine]) -> float | None:
    angles: list[float] = []
    for r in _measurable(raw):
        (x0, y0), (x1, y1) = r.box[0], r.box[1]
        width = math.hypot(x1 - x0, y1 - y0)
        height = math.hypot(r.box[3][0] - x0, r.box[3][1] - y0)
        if width < 3 * max(height, 1.0):
            continue
        angles.append(math.degrees(math.atan2(y1 - y0, x1 - x0)))
    if len(angles) < MIN_LINES_FOR_ORIENTATION:
        return None
    angle = statistics.median(angles)
    return angle if 1.0 <= abs(angle) <= 20.0 else None


def _map_back(raw: list[RawLine], matrix: np.ndarray) -> list[RawLine]:
    return [RawLine(r.text, r.confidence, map_points_back(r.box, matrix) if r.box else None) for r in raw]


def _ocr_page(engine: OcrEngine, resized: np.ndarray, settings: Settings) -> tuple[list[RawLine], str | None, bool]:
    prepared = enhance(resized, settings.ocr_denoise)
    raw = engine.recognize(prepared)

    if settings.ocr_auto_rotate and _mostly_vertical(raw):
        best, best_score, best_angle, best_matrix = raw, _text_score(raw), 0, None
        for angle in (90, -90):
            rotated, matrix = rotate_image(prepared, angle)
            candidate = engine.recognize(rotated)
            score = _text_score(candidate)
            if score > best_score:
                best, best_score, best_angle, best_matrix = candidate, score, angle, matrix
        if best_matrix is not None:
            return _map_back(best, best_matrix), f"Page was rotated {best_angle} degrees for OCR.", False
        return raw, None, True

    if settings.ocr_deskew:
        angle = _skew_angle(raw)
        if angle is not None:
            rotated, matrix = rotate_image(prepared, angle)
            candidate = engine.recognize(rotated)
            if _text_score(candidate) > _text_score(raw):
                return _map_back(candidate, matrix), f"Corrected a {abs(angle):.1f} degree skew.", True

    return raw, None, True


def _ocr_lines(raw_lines: list[RawLine], page: int, start_index: int, scale: float) -> list[OcrLine]:
    out: list[OcrLine] = []
    for raw in raw_lines:
        text = raw.text.strip()
        if not text:
            continue
        box = [[round(x / scale, 1), round(y / scale, 1)] for x, y in raw.box] if raw.box else None
        confidence = round(raw.confidence, 4) if raw.confidence is not None else None
        out.append(
            OcrLine(page=page, index=start_index + len(out), text=text, confidence=confidence, box=box, source="ocr")
        )
    return out


def _text_lines(text: str, page: int, start_index: int) -> list[OcrLine]:
    out: list[OcrLine] = []
    for raw in text.splitlines():
        cleaned = " ".join(raw.split())
        if cleaned:
            out.append(OcrLine(page=page, index=start_index + len(out), text=cleaned, source="pdf_text"))
    return out


def _run_ocr(
    service: OcrService, image: np.ndarray, settings: Settings
) -> tuple[list[RawLine], float, np.ndarray, str | None, bool]:
    engine = service.get_engine()
    resized, scale = resize_for_ocr(image, settings.ocr_max_side, settings.ocr_min_side)
    raw, note, croppable = _ocr_page(engine, resized, settings)
    return raw, scale, resized, note, croppable


def _process_image(vfile: ValidatedFile, service: OcrService, settings: Settings, result: PipelineResult) -> None:
    try:
        image = decode_image(vfile.content)
        raw, scale, resized, note, croppable = _run_ocr(service, image, settings)
    except AppError:
        raise
    except Exception:
        logger.exception("OCR failed for image upload")
        raise AppError(500, "ocr_failed", "OCR processing failed. Please try again.")
    result.lines.extend(_ocr_lines(raw, 1, 0, scale))
    result.pages.append(PageResult(page=1, status="processed", source="ocr", message=note))
    if settings.ocr_refine and croppable:
        result.images[1] = PageImage(resized, scale)


def _process_pdf(vfile: ValidatedFile, service: OcrService, settings: Settings, result: PipelineResult) -> None:
    pages = iter_pdf_pages(
        vfile.content,
        dpi=settings.pdf_render_dpi,
        max_side=settings.ocr_max_side,
        min_text_chars=settings.pdf_min_text_chars,
    )
    for pdf_page in pages:
        n = pdf_page.number
        if pdf_page.error:
            result.pages.append(PageResult(page=n, status="failed", message=pdf_page.error))
            result.warnings.append(f"Page {n} could not be read.")
            continue
        if pdf_page.text is not None:
            result.lines.extend(_text_lines(pdf_page.text, n, len(result.lines)))
            result.pages.append(PageResult(page=n, status="processed", source="pdf_text"))
            continue
        try:
            raw, scale, resized, note, croppable = _run_ocr(service, pdf_page.image, settings)
        except AppError:
            raise
        except Exception:
            logger.exception("OCR failed on PDF page %d", n)
            result.pages.append(PageResult(page=n, status="failed", source="ocr", message="OCR failed on this page."))
            result.warnings.append(f"Page {n}: OCR failed.")
            continue
        total_scale = pdf_page.render_scale * scale
        result.lines.extend(_ocr_lines(raw, n, len(result.lines), total_scale))
        result.pages.append(PageResult(page=n, status="processed", source="ocr", message=note))
        if settings.ocr_refine and croppable:
            result.images[n] = PageImage(resized, total_scale)

    if not any(p.status == "processed" for p in result.pages):
        raise AppError(422, "processing_failed", "None of the PDF pages could be processed.")


def process_document(vfile: ValidatedFile, service: OcrService, settings: Settings) -> PipelineResult:
    result = PipelineResult()
    if vfile.kind == "pdf":
        _process_pdf(vfile, service, settings, result)
    else:
        _process_image(vfile, service, settings, result)
    if not result.lines:
        result.warnings.append("No text was detected in the document.")
    return result