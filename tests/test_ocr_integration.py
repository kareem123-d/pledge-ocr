import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SAMPLES = Path(__file__).parent / "samples"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OCR_INTEGRATION") != "1",
    reason="Set RUN_OCR_INTEGRATION=1 to run real PaddleOCR on tests/samples",
)


def sample_files():
    if not SAMPLES.exists():
        return []
    return sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in EXTENSIONS)


@pytest.fixture(scope="module")
def real_client():
    with TestClient(create_app()) as c:
        yield c


@pytest.mark.parametrize("path", sample_files(), ids=lambda p: p.name)
def test_real_sample_returns_text(real_client, path):
    r = real_client.post("/api/v1/pledge-slips/upload", files={"file": (path.name, path.read_bytes())})
    assert r.status_code == 200, r.text
    assert r.json()["ocr"]["lines"]