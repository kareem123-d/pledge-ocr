import asyncio

import fitz
import httpx
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from helpers import FakeEngine, make_image_bytes, make_scanned_pdf, make_text_pdf

URL = "/api/v1/pledge-slips/upload"


def upload(client, name, content, mime="application/octet-stream"):
    return client.post(URL, files={"file": (name, content, mime)})


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["success"] is False
    assert body["requestId"]
    assert body["error"]["code"] == code
    assert r.headers["X-Request-ID"] == body["requestId"]


def test_valid_image_upload(client, fake_engine):
    r = upload(client, "slip.png", make_image_bytes(), "image/png")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["requestId"] == r.headers["X-Request-ID"]
    assert body["document"]["fileType"] == "image"
    assert body["document"]["pageCount"] == 1
    assert body["document"]["pages"][0] == {"page": 1, "status": "processed", "source": "ocr", "message": None}
    assert body["ocr"]["lines"][0]["text"] == "Customer Name: Test User"
    assert body["ocr"]["rawText"] == "Customer Name: Test User"
    assert body["extractedData"]["customerName"]["status"] == "extracted"
    assert body["extractedData"]["customerName"]["value"] == "Test User"
    assert body["review"]["required"] is True
    assert fake_engine.calls == 1


def test_text_pdf_uses_embedded_text(client, fake_engine):
    pdf = make_text_pdf("Customer Name: Ravi Kumar\nLoan Amount: 50000\nGross Weight: 12.5 g")
    r = upload(client, "slip.pdf", pdf, "application/pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert fake_engine.calls == 0
    assert body["document"]["pages"][0]["source"] == "pdf_text"
    texts = [line["text"] for line in body["ocr"]["lines"]]
    assert "Customer Name: Ravi Kumar" in texts
    assert all(line["source"] == "pdf_text" for line in body["ocr"]["lines"])


def test_scanned_pdf_runs_ocr_per_page(client, fake_engine):
    r = upload(client, "slip.pdf", make_scanned_pdf(2), "application/pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert fake_engine.calls == 2
    assert body["document"]["pageCount"] == 2
    assert {line["page"] for line in body["ocr"]["lines"]} == {1, 2}
    assert [line["index"] for line in body["ocr"]["lines"]] == [0, 1]


def test_pdf_partial_failure_is_reported():
    engine = FakeEngine(fail_on_calls={2})
    with TestClient(create_app(engine_factory=lambda: engine)) as c:
        r = upload(c, "slip.pdf", make_scanned_pdf(2), "application/pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [p["status"] for p in body["document"]["pages"]] == ["processed", "failed"]
    assert any("Page 2" in w for w in body["review"]["warnings"])
    assert {line["page"] for line in body["ocr"]["lines"]} == {1}
    assert "secret" not in r.text


def test_unsupported_type(client):
    assert_error(upload(client, "slip.txt", b"hello"), 415, "unsupported_file_type")


def test_empty_upload(client):
    assert_error(upload(client, "slip.png", b""), 400, "empty_file")


def test_missing_file_field(client):
    assert_error(client.post(URL), 422, "validation_error")


def test_oversized_upload(client):
    assert_error(upload(client, "slip.png", b"\x89PNG\r\n\x1a\n" + b"0" * (16 * 1024 * 1024)), 413, "file_too_large")


def test_invalid_image(client):
    assert_error(upload(client, "slip.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50), 422, "invalid_image")


def test_malformed_pdf(client):
    assert_error(upload(client, "slip.pdf", b"%PDF-1.4\ngarbage"), 422, "malformed_pdf")


def test_encrypted_pdf(client):
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    assert_error(upload(client, "slip.pdf", data), 422, "encrypted_pdf")


def test_ocr_failure_is_generic():
    engine = FakeEngine(fail_on_calls={1})
    with TestClient(create_app(engine_factory=lambda: engine)) as c:
        r = upload(c, "slip.png", make_image_bytes(), "image/png")
    assert_error(r, 500, "ocr_failed")
    assert "secret" not in r.text


def test_ocr_engine_unavailable():
    def broken():
        raise RuntimeError("model download failed")

    with TestClient(create_app(engine_factory=broken)) as c:
        r = upload(c, "slip.png", make_image_bytes(), "image/png")
    assert_error(r, 503, "ocr_unavailable")


def test_request_ids_are_unique(client):
    a = upload(client, "a.png", make_image_bytes(), "image/png").json()["requestId"]
    b = upload(client, "b.png", make_image_bytes(), "image/png").json()["requestId"]
    assert a != b


def test_trocr_disabled_by_default():
    assert Settings(_env_file=None).enable_trocr is False


async def test_concurrent_requests_are_isolated_and_limited():
    engine = FakeEngine(text_fn=lambda img: ["WHITE" if img.mean() > 127 else "BLACK"], delay=0.3)
    app = create_app(engine_factory=lambda: engine)
    files = [("white", "white"), ("black", "black"), ("white", "white"), ("black", "black")]
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            responses = await asyncio.gather(
                *[
                    ac.post(URL, files={"file": (f"{name}.png", make_image_bytes(color=color), "image/png")})
                    for name, color in files
                ]
            )
    assert all(r.status_code == 200 for r in responses)
    assert [r.json()["ocr"]["rawText"] for r in responses] == ["WHITE", "BLACK", "WHITE", "BLACK"]
    assert len({r.json()["requestId"] for r in responses}) == 4
    assert engine.max_active == 1