import asyncio
import logging
import time

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.errors import AppError
from app.schemas.common import ErrorResponse, HealthResponse
from app.schemas.pledge_slip import PledgeSlipResponse
from app.services.file_validator import read_upload_limited, validate_upload
from app.services.job_queue import job_manager
from app.utils.request_context import get_request_id

logger = logging.getLogger(__name__)
router = APIRouter()

MULTIPART_OVERHEAD = 1024 * 1024
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
):
    settings = get_settings()
    started = time.perf_counter()

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

    job = job_manager.submit(validated, request.app.state.ocr_service)
    if job is None:
        raise AppError(503, "server_busy", "The queue is full. Please try again in a minute.")
    try:
        await asyncio.wait_for(job.event.wait(), timeout=600)
    except asyncio.TimeoutError:
        raise AppError(503, "server_busy", "The server is busy. Please try again in a minute.")
    if job.status == "failed":
        raise job.exception

    job.result["requestId"] = get_request_id(request)
    logger.info("Upload %s finished in %.2fs", validated.filename, time.perf_counter() - started)
    return job.result