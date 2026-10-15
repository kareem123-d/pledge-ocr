import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.utils.request_context import get_request_id

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "requestId": get_request_id(request),
            "error": error,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.warning("AppError %s: %s", exc.code, exc.message)
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        details = [
            {
                "field": ".".join(str(p) for p in e.get("loc", [])),
                "message": e.get("msg", "Invalid value"),
            }
            for e in exc.errors()
        ]
        return _error_response(
            request, 422, "validation_error", "Request validation failed.", details
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        if exc.__cause__ is not None:
            logger.warning("HTTP %s caused by %r", exc.status_code, exc.__cause__, exc_info=exc.__cause__)
        return _error_response(
            request, exc.status_code, f"http_{exc.status_code}", str(exc.detail)
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return _error_response(
            request,
            500,
            "internal_error",
            "An unexpected error occurred. Please try again later.",
        )