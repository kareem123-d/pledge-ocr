import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.services.ocr_service import HandwritingEngine, OcrEngine, OcrService
from app.utils.request_context import RequestIdFilter, request_id_middleware

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def create_app(
    engine_factory: Callable[[], OcrEngine] | None = None,
    handwriting_factory: Callable[[], HandwritingEngine] | None = None,
) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = OcrService(settings, engine_factory, handwriting_factory)
        app.state.ocr_service = service
        if settings.ocr_preload:
            logger.info("Loading OCR engine")
            try:
                await run_in_threadpool(service.warm_up)
                logger.info("OCR engine ready")
            except Exception:
                logger.exception("OCR engine failed to load; it will be retried on the first upload")
        logger.info("Handwriting model: %s", service.handwriting_status)
        logger.info("Application started")
        yield
        logger.info("Application stopped")

    app = FastAPI(
        title="Loan Pledge Slip OCR API",
        version="1.0.0",
        description="Extracts fields from gold-loan pledge slips using local OCR.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.middleware("http")(request_id_middleware)

    register_exception_handlers(app)
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()