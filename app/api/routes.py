import logging
import time
import asyncio
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.errors import AppError
from app.schemas.common import ErrorResponse, HealthResponse
from app.schemas.pledge_slip import DocumentInfo, OcrResult, PledgeSlipResponse
from app.services.file_validator import read_upload_limited, validate_upload
from app.services.review_service import build_review
from app.services.slip_analyzer import analyze_document
from app.utils.request_context import get_request_id

logger = logging.getLogger(__name__)
router = APIRouter()

MULTIPART_OVERHEAD = 1024 * 1024
_waiting = 0
ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503)}


@router.get("/health", response_model=HealthResponse, tags=["system"], summary="Health check")
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    service = getattr(request.app.state, "ocr_service", None)
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        ocrInitialized=bool(service and service.ready),
        handwritingModel=service.handwriting_status if service else "disabled",
    )

@router.post(
    "/pledge-slips/upload",
    response_model=PledgeSlipResponse,
    responses=ERROR_RESPONSES,
    tags=["pledge-slips"],
    summary="Upload a pledge slip and extract its contents",
)
async def upload_pledge_slip(
    request: Request,
    file: UploadFile = File(..., description="JPG, JPEG, PNG or PDF, max 15 MB"),
) -> PledgeSlipResponse:
    settings = get_settings()
    started = time.perf_counter()
    global _waiting
    if _waiting >= settings.ocr_queue_max:
        await file.close()
        raise AppError(503, "server_busy", "The server is busy. Please try again in a minute.")

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > settings.max_upload_bytes + MULTIPART_OVERHEAD:
        await file.close()
        raise AppError(413, "file_too_large", f"File exceeds the {settings.max_upload_mb} MB limit.")

    try:
        content = await read_upload_limited(file, settings.max_upload_bytes)
    finally:
        await file.close()

    validated = await run_in_threadpool(
        validate_upload,
        file.filename,
        content,
        max_bytes=settings.max_upload_bytes,
        max_image_pixels=settings.max_image_pixels,
        max_pdf_pages=settings.max_pdf_pages,
    )

    service = request.app.state.ocr_service
    _waiting += 1
    try:
        await asyncio.wait_for(service.semaphore.acquire(), timeout=settings.ocr_queue_timeout)
    except asyncio.TimeoutError:
        raise AppError(503, "server_busy", "The server is busy. Please try again in a minute.")
    finally:
        _waiting -= 1
    try:
        analysis = await run_in_threadpool(analyze_document, validated, service, settings)
    finally:
        service.semaphore.release()

    result, extracted = analysis.pipeline, analysis.extracted
    review = build_review(extracted, settings.mandatory_field_list, analysis.warnings)

    response = PledgeSlipResponse(
        requestId=get_request_id(request),
        document=DocumentInfo(
            filename=validated.filename,
            fileType=validated.kind,
            pageCount=validated.page_count,
            pages=result.pages,
        ),
        extractedData=extracted,
        ocr=OcrResult(rawText="\n".join(line.text for line in result.lines), lines=result.lines),
        review=review,
    )
    logger.info(
        "Processed %s upload: %d page(s), %d line(s), %d item(s), review=%s in %.2fs",
        validated.kind,
        validated.page_count,
        len(result.lines),
        len(extracted.ornamentDetails),
        review.required,
        time.perf_counter() - started,
    )
    return response