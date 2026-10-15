import logging

import cv2
import numpy as np

from app.core.extraction_rules import FIELD_RULES, ITEM_FIELD_KINDS
from app.services.document_pipeline import PageImage
from app.services.field_extractor import Candidate, TableResult
from app.services.field_validator import Thresholds, parse_value
from app.services.ocr_service import HandwritingEngine, OcrEngine

logger = logging.getLogger(__name__)

KIND_BY_FIELD = {rule.name: rule.kind for rule in FIELD_RULES}
MIN_CROP_HEIGHT = 32
INK_SATURATION_MIN = 40
MIN_CONFIDENCE_GAIN = 0.05
MIN_REREAD_CONFIDENCE = 0.6


def _line_conf(cand: Candidate) -> float | None:
    confs = [line.confidence for line in cand.lines if line.confidence is not None]
    return min(confs) if confs else None


def _needs_refinement(kind: str, cand: Candidate, th: Thresholds, allow_text: bool) -> bool:
    if kind == "text" and not allow_text:
        return False
    if cand.ocr_confidence is not None or not cand.lines:
        return False
    if any(line.box is None for line in cand.lines):
        return False
    parsed = parse_value(kind, cand.raw)
    conf = _line_conf(cand)
    low = conf is not None and conf < th.min_ocr_confidence
    return parsed.error is not None or parsed.substituted or low


def _value_x_start(cand: Candidate, x0: float, x1: float) -> float | None:
    if cand.label is None or len(cand.lines) != 1 or cand.lines[0].index != cand.label.index:
        return None
    text = cand.label.text
    pos = text.rfind(cand.raw)
    if pos < 0:
        pos = max(0, len(text) - len(cand.raw))
    return x0 + (x1 - x0) * pos / max(1, len(text))


def crop_value(cand: Candidate, page_image: PageImage) -> np.ndarray | None:
    xs = [p[0] for line in cand.lines for p in line.box]
    ys = [p[1] for line in cand.lines for p in line.box]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    line_h = max(max(p[1] for p in line.box) - min(p[1] for p in line.box) for line in cand.lines)
    h = max(1.0, line_h)
    if y1 - y0 > 1.8 * h:
        return None

    start = _value_x_start(cand, x0, x1)
    if start is not None:
        x0 = max(x0, start - 0.2 * h)

    pad_x, pad_y = 0.6 * h, 0.35 * h
    s = page_image.scale
    height, width = page_image.image.shape[:2]
    X0 = max(0, int((x0 - pad_x) * s))
    X1 = min(width, int(round((x1 + pad_x) * s)))
    Y0 = max(0, int((y0 - pad_y) * s))
    Y1 = min(height, int(round((y1 + pad_y) * s)))
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return None
    return page_image.image[Y0:Y1, X0:X1].copy()


def crop_variants(crop: np.ndarray) -> list[np.ndarray]:
    if crop.shape[0] < MIN_CROP_HEIGHT:
        factor = MIN_CROP_HEIGHT / crop.shape[0]
        crop = cv2.resize(
            crop, (max(1, round(crop.shape[1] * factor)), MIN_CROP_HEIGHT), interpolation=cv2.INTER_CUBIC
        )
    variants = [crop]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
    darkest = np.ascontiguousarray(crop.min(axis=2))
    variants.append(cv2.cvtColor(darkest, cv2.COLOR_GRAY2BGR))
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    if int(saturation.max()) >= INK_SATURATION_MIN:
        ink = cv2.normalize(255 - saturation, None, 0, 255, cv2.NORM_MINMAX)
        variants.append(cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR))
    return variants


def _rank(kind: str, text: str) -> tuple[bool, bool]:
    parsed = parse_value(kind, text)
    return parsed.error is None, not parsed.substituted


def refine_candidate(
    kind: str,
    cand: Candidate,
    images: dict[int, PageImage],
    engine: OcrEngine,
    th: Thresholds,
    handwriting: HandwritingEngine | None = None,
) -> tuple[Candidate, bool]:
    if not _needs_refinement(kind, cand, th, allow_text=handwriting is not None):
        return cand, False
    page_image = images.get(cand.lines[0].page)
    if page_image is None:
        return cand, False
    crop = crop_value(cand, page_image)
    if crop is None:
        return cand, False

    variants = crop_variants(crop)
    jobs: list[tuple[object, np.ndarray, str]] = []
    if kind != "text":
        jobs.extend((engine, v, "ocr") for v in variants)
    if handwriting is not None:
        jobs.extend((handwriting, v, "handwriting") for v in (variants[0], variants[2]))

    original_conf = _line_conf(cand) or 0.0
    original_rank = _rank(kind, cand.raw)
    best_text, best_conf, best_rank, best_source = None, original_conf, original_rank, None

    for reader, variant, source in jobs:
        try:
            readings = reader.recognize_crop(variant)
        except Exception:
            logger.warning("Crop re-read failed (%s)", source, exc_info=True)
            continue
        for text, conf in readings:
            text = (text or "").strip()
            if not text or text == cand.raw or conf < MIN_REREAD_CONFIDENCE:
                continue
            rank = _rank(kind, text)
            if not rank[0] or rank < best_rank:
                continue
            if rank == original_rank and conf < original_conf + MIN_CONFIDENCE_GAIN:
                continue
            if rank == best_rank and conf <= best_conf:
                continue
            best_text, best_conf, best_rank, best_source = text, conf, rank, source

    if best_text is None:
        return cand, True
    how = "by the handwriting model" if best_source == "handwriting" else "from a cropped image region"
    return (
        Candidate(
            raw=best_text,
            lines=cand.lines,
            match_confidence=cand.match_confidence,
            label=cand.label,
            ocr_confidence=float(best_conf),
            note=f"Re-read {how}; first pass read '{cand.raw}'.",
        ),
        True,
    )


def refine_all(
    fields: dict[str, Candidate],
    table: TableResult,
    images: dict[int, PageImage],
    engine: OcrEngine,
    th: Thresholds,
    max_crops: int,
    handwriting: HandwritingEngine | None = None,
) -> int:
    used = 0

    def run(kind: str, cand: Candidate | None) -> Candidate | None:
        nonlocal used
        if cand is None or used >= max_crops:
            return cand
        new, did_crop = refine_candidate(kind, cand, images, engine, th, handwriting)
        used += int(did_crop)
        return new

    for name in list(fields):
        fields[name] = run(KIND_BY_FIELD.get(name, "text"), fields[name])
    for key in list(table.totals):
        table.totals[key] = run(ITEM_FIELD_KINDS.get(key, "text"), table.totals[key])
    for row in table.rows:
        row.description = run("text", row.description)
        for key in list(row.cells):
            row.cells[key] = run(ITEM_FIELD_KINDS.get(key, "text"), row.cells[key])
    return used