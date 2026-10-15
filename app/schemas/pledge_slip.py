from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

FieldValue = str | int | float | None


class FieldStatus(str, Enum):
    EXTRACTED = "extracted"
    UNCERTAIN = "uncertain"
    MISSING = "missing"
    INVALID = "invalid"


class Evidence(BaseModel):
    page: int
    lineIndex: int
    text: str


class ExtractedField(BaseModel):
    value: FieldValue = None
    rawText: str | None = None
    unit: str | None = None
    status: FieldStatus = FieldStatus.MISSING
    ocrConfidence: float | None = Field(default=None, ge=0, le=1)
    matchConfidence: float | None = Field(default=None, ge=0, le=1)
    note: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

class IdentifierField(ExtractedField):
    label: str

class OrnamentItem(BaseModel):
    description: ExtractedField = Field(default_factory=ExtractedField)
    quantity: ExtractedField = Field(default_factory=ExtractedField)
    goldPurity: ExtractedField = Field(default_factory=ExtractedField)
    grossWeight: ExtractedField = Field(default_factory=ExtractedField)
    netWeight: ExtractedField = Field(default_factory=ExtractedField)


class ExtractedData(BaseModel):
    customerName: ExtractedField = Field(default_factory=ExtractedField)
    bankName: ExtractedField = Field(default_factory=ExtractedField)
    loanAccountNumber: ExtractedField = Field(default_factory=ExtractedField)
    pledgeNumber: ExtractedField = Field(default_factory=ExtractedField)
    loanAmount: ExtractedField = Field(default_factory=ExtractedField)
    loanDate: ExtractedField = Field(default_factory=ExtractedField)
    dueDate: ExtractedField = Field(default_factory=ExtractedField)
    branchName: ExtractedField = Field(default_factory=ExtractedField)
    phoneNumber: ExtractedField = Field(default_factory=ExtractedField)
    interestRate: ExtractedField = Field(default_factory=ExtractedField)
    tenure: ExtractedField = Field(default_factory=ExtractedField)
    netWeight: ExtractedField = Field(default_factory=ExtractedField)
    address: ExtractedField = Field(default_factory=ExtractedField)
    loanApplicationNumber: ExtractedField = Field(default_factory=ExtractedField)
    loanNumber: ExtractedField = Field(default_factory=ExtractedField)
    receiptNumber: ExtractedField = Field(default_factory=ExtractedField)
    customerId: ExtractedField = Field(default_factory=ExtractedField)
    transactionReference: ExtractedField = Field(default_factory=ExtractedField)
    packetNumber: ExtractedField = Field(default_factory=ExtractedField)
    branchCode: ExtractedField = Field(default_factory=ExtractedField)
    grossWeight: ExtractedField = Field(default_factory=ExtractedField)
    goldPurity: ExtractedField = Field(default_factory=ExtractedField)
    appraisedValue: ExtractedField = Field(default_factory=ExtractedField)
    ornamentDetails: list[OrnamentItem] = Field(default_factory=list)
    otherIdentifiers: list[IdentifierField] = Field(default_factory=list)


class OcrLine(BaseModel):
    page: int
    index: int
    text: str
    confidence: float | None = None
    box: list[list[float]] | None = None
    source: Literal["ocr", "pdf_text"]


class OcrResult(BaseModel):
    rawText: str = ""
    lines: list[OcrLine] = Field(default_factory=list)


class PageResult(BaseModel):
    page: int
    status: Literal["processed", "failed", "skipped"]
    source: Literal["ocr", "pdf_text"] | None = None
    message: str | None = None


class DocumentInfo(BaseModel):
    filename: str
    fileType: Literal["image", "pdf"]
    documentType: str = "gold_loan_pledge_slip"
    pageCount: int
    pages: list[PageResult] = Field(default_factory=list)


class ReviewInfo(BaseModel):
    required: bool
    missingMandatoryFields: list[str] = Field(default_factory=list)
    missingFields: list[str] = Field(default_factory=list)
    uncertainFields: list[str] = Field(default_factory=list)
    invalidFields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PledgeSlipResponse(BaseModel):
    success: Literal[True] = True
    requestId: str
    document: DocumentInfo
    extractedData: ExtractedData
    ocr: OcrResult
    review: ReviewInfo