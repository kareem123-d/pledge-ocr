import io
import os
import re
from dataclasses import dataclass

import fitz
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.errors import AppError

ALLOWED_EXTENSIONS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".pdf": "pdf"}
READ_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ValidatedFile:
    filename: str
    extension: str
    kind: str
    content: bytes
    page_count: int
    width: int | None = None
    height: int | None = None


def sanitize_filename(name: str | None) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return base[:120] or "upload"


def detect_format(content: bytes) -> str | None:
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content[:1024].find(b"%PDF-") != -1:
        return "pdf"
    return None


async def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise AppError(413, "file_too_large", f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_image(content: bytes, max_pixels: int) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
            if width * height > max_pixels:
                raise AppError(422, "image_too_large", "Image dimensions are too large to process.")
            img.verify()
        with Image.open(io.BytesIO(content)) as img:
            img.load()
    except AppError:
        raise
    except Image.DecompressionBombError:
        raise AppError(422, "image_too_large", "Image dimensions are too large to process.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise AppError(422, "invalid_image", "The image file is corrupted or unreadable.")
    if width < 1 or height < 1:
        raise AppError(422, "invalid_image", "The image has invalid dimensions.")
    return width, height


def _validate_pdf(content: bytes, max_pages: int) -> int:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception:
        raise AppError(422, "malformed_pdf", "The PDF file is corrupted or unreadable.")
    try:
        if doc.needs_pass:
            raise AppError(422, "encrypted_pdf", "Password-protected PDFs are not supported. Upload an unlocked copy.")
        page_count = doc.page_count
    except AppError:
        raise
    except Exception:
        raise AppError(422, "malformed_pdf", "The PDF file is corrupted or unreadable.")
    finally:
        doc.close()
    if page_count < 1:
        raise AppError(422, "malformed_pdf", "The PDF has no pages.")
    if page_count > max_pages:
        raise AppError(422, "too_many_pages", f"The PDF has {page_count} pages; the maximum is {max_pages}.")
    return page_count


def validate_upload(
    filename: str | None,
    content: bytes,
    *,
    max_bytes: int,
    max_image_pixels: int,
    max_pdf_pages: int,
) -> ValidatedFile:
    safe_name = sanitize_filename(filename)

    if not content:
        raise AppError(400, "empty_file", "The uploaded file is empty.")
    if len(content) > max_bytes:
        raise AppError(413, "file_too_large", f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.")

    extension = os.path.splitext(safe_name)[1].lower()
    expected = ALLOWED_EXTENSIONS.get(extension)
    if expected is None:
        raise AppError(415, "unsupported_file_type", "Only JPG, JPEG, PNG and PDF files are accepted.")

    detected = detect_format(content)
    if detected is None:
        raise AppError(415, "unsupported_file_type", "File content is not a valid JPG, PNG or PDF.")
    if detected != expected:
        raise AppError(415, "file_type_mismatch", "File extension does not match the file content.")

    if detected == "pdf":
        pages = _validate_pdf(content, max_pdf_pages)
        return ValidatedFile(safe_name, extension, "pdf", content, pages)

    width, height = _validate_image(content, max_image_pixels)
    return ValidatedFile(safe_name, extension, "image", content, 1, width, height)