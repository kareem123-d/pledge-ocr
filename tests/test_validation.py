import io

import fitz
import pytest
from PIL import Image

from app.core.errors import AppError
from app.services.file_validator import sanitize_filename, validate_upload

MB = 1024 * 1024
LIMITS = dict(max_bytes=15 * MB, max_image_pixels=40_000_000, max_pdf_pages=10)


def make_image(fmt: str = "PNG", size=(200, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format=fmt)
    return buf.getvalue()


def make_pdf(pages: int = 1, **save_kwargs) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Page {i + 1}")
    data = doc.tobytes(**save_kwargs)
    doc.close()
    return data


def expect_error(code: str, status: int, filename: str, content: bytes, **overrides):
    with pytest.raises(AppError) as exc:
        validate_upload(filename, content, **{**LIMITS, **overrides})
    assert exc.value.code == code
    assert exc.value.status_code == status


def test_valid_png():
    f = validate_upload("slip.png", make_image("PNG"), **LIMITS)
    assert f.kind == "image" and f.width == 200 and f.height == 100


def test_valid_jpeg():
    f = validate_upload("slip.JPG", make_image("JPEG"), **LIMITS)
    assert f.kind == "image" and f.extension == ".jpg"


def test_valid_pdf():
    f = validate_upload("slip.pdf", make_pdf(3), **LIMITS)
    assert f.kind == "pdf" and f.page_count == 3


def test_empty_file():
    expect_error("empty_file", 400, "slip.png", b"")


def test_oversized_file():
    expect_error("file_too_large", 413, "slip.png", make_image(), max_bytes=100)


def test_unsupported_extension():
    expect_error("unsupported_file_type", 415, "slip.txt", b"hello")


def test_unknown_content():
    expect_error("unsupported_file_type", 415, "slip.png", b"not an image at all")


def test_extension_content_mismatch():
    expect_error("file_type_mismatch", 415, "slip.pdf", make_image("PNG"))


def test_corrupted_image():
    expect_error("invalid_image", 422, "slip.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)


def test_truncated_image():
    data = make_image("PNG", (500, 500))
    expect_error("invalid_image", 422, "slip.png", data[: len(data) // 2])


def test_image_pixel_limit():
    expect_error("image_too_large", 422, "slip.png", make_image(size=(300, 300)), max_image_pixels=1000)


def test_malformed_pdf():
    expect_error("malformed_pdf", 422, "slip.pdf", b"%PDF-1.4\nthis is garbage")


def test_encrypted_pdf():
    data = make_pdf(1, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    expect_error("encrypted_pdf", 422, "slip.pdf", data)


def test_too_many_pages():
    expect_error("too_many_pages", 422, "slip.pdf", make_pdf(3), max_pdf_pages=2)


def test_sanitize_filename():
    assert sanitize_filename("..\\..\\evil<>name.png") == "evil__name.png"
    assert sanitize_filename("C:/Users/x/slip 1.jpg") == "slip_1.jpg"
    assert sanitize_filename(None) == "upload"
    assert sanitize_filename("....") == "upload"