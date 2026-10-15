from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    service: str
    ocrInitialized: bool
    handwritingModel: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    requestId: str
    error: ErrorDetail