import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.schemas.pledge_slip import ExtractedData
from app.services.document_pipeline import PipelineResult, process_document
from app.services.field_extractor import extract_fields, extract_other_identifiers, extract_table
from app.services.field_validator import Thresholds, build_extracted_data
from app.services.file_validator import ValidatedFile
from app.services.ocr_service import OcrService
from app.services.value_refiner import refine_all

logger = logging.getLogger(__name__)


@dataclass
class SlipAnalysis:
    pipeline: PipelineResult
    extracted: ExtractedData
    warnings: list[str]


def analyze_document(vfile: ValidatedFile, service: OcrService, settings: Settings) -> SlipAnalysis:
    result = process_document(vfile, service, settings)
    th = Thresholds(settings.field_min_ocr_confidence, settings.field_min_match_confidence)
    table = extract_table(result.lines)
    fields = extract_fields(result.lines, exclude=table.header_indices)
    others = extract_other_identifiers(result.lines, exclude=table.header_indices)

    warnings = list(result.warnings)
    if settings.ocr_refine and result.images:
        try:
            used = refine_all(
                fields,
                table,
                result.images,
                service.get_engine(),
                th,
                settings.ocr_refine_max_crops,
                service.get_handwriting(),
            )
            logger.info("Re-read %d value region(s)", used)
        except Exception:
            logger.exception("Value re-read step failed")
            warnings.append("The value re-read step failed; first-pass OCR values were used.")
        if service.handwriting_status == "failed":
            warnings.append("The handwriting model is enabled but could not be loaded; it was skipped.")
    result.images.clear()

    extracted, extraction_warnings = build_extracted_data(fields, table, th, others)
    return SlipAnalysis(result, extracted, [*warnings, *extraction_warnings])