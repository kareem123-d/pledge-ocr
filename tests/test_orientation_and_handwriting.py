import math

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.document_pipeline import _mostly_vertical, _skew_angle
from app.services.image_preprocessing import map_points_back, rotate_image
from app.services.ocr_service import RawLine
from helpers import FakeEngine, make_image_bytes

URL = "/api/v1/pledge-slips/upload"


def tilted_lines(angle_deg, n=6, conf=0.7):
    t = math.radians(angle_deg)
    lines = []
    for i in range(n):
        x0, y0, w, h = 50, 40 * i + 20, 300, 20
        p0 = [x0, y0]
        p1 = [x0 + w * math.cos(t), y0 + w * math.sin(t)]
        p3 = [x0 - h * math.sin(t), y0 + h * math.cos(t)]
        p2 = [p1[0] - h * math.sin(t), p1[1] + h * math.cos(t)]
        lines.append(RawLine(f"Printed line {i}", conf, [p0, p1, p2, p3]))
    return lines


def upload(client):
    return client.post(URL, files={"file": ("slip.png", make_image_bytes(), "image/png")})


def test_skew_angle_detected():
    assert abs(_skew_angle(tilted_lines(5)) - 5) < 0.1


def test_small_skew_is_ignored():
    assert _skew_angle(tilted_lines(0.5)) is None


def test_vertical_text_detected():
    lines = [RawLine(f"Line {i} text", 0.9, [[10 * i, 0], [10 * i + 20, 0], [10 * i + 20, 200], [10 * i, 200]]) for i in range(6)]
    assert _mostly_vertical(lines)


def test_rotate_and_map_back_roundtrip():
    img = np.full((200, 400, 3), 255, np.uint8)
    _, m = rotate_image(img, 30)
    x, y = 100.0, 50.0
    forward = [m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]]
    back = map_points_back([forward], m)[0]
    assert abs(back[0] - x) < 1e-6 and abs(back[1] - y) < 1e-6


class SkewEngine(FakeEngine):
    def recognize(self, image):
        with self._lock:
            self.calls += 1
            call = self.calls
        return tilted_lines(5, conf=0.7) if call == 1 else tilted_lines(0, conf=0.95)


def test_upload_corrects_skew():
    engine = SkewEngine()
    with TestClient(create_app(engine_factory=lambda: engine)) as c:
        r = upload(c)
    assert r.status_code == 200, r.text
    body = r.json()
    assert engine.calls == 2
    assert "skew" in body["document"]["pages"][0]["message"]
    assert all(line["confidence"] == 0.95 for line in body["ocr"]["lines"])


class FakeHandwriting:
    def __init__(self, reading):
        self.reading = reading
        self.calls = 0

    def recognize_crop(self, image):
        self.calls += 1
        return [self.reading]


def test_handwriting_model_rereads_amount():
    engine = FakeEngine(text_fn=lambda img: ["Amount of Loan: 00000"])
    hw = FakeHandwriting(("14,00,000", 0.9))
    with TestClient(create_app(engine_factory=lambda: engine, handwriting_factory=lambda: hw)) as c:
        r = upload(c)
        health = c.get("/api/v1/health").json()
    assert r.status_code == 200, r.text
    amount = r.json()["extractedData"]["loanAmount"]
    assert amount["value"] == 1400000
    assert amount["status"] == "uncertain"
    assert "handwriting model" in amount["note"]
    assert hw.calls >= 1
    assert health["handwritingModel"] == "loaded"


def test_handwriting_load_failure_does_not_break_upload():
    def broken():
        raise RuntimeError("torch is not installed")

    engine = FakeEngine(text_fn=lambda img: ["Amount of Loan: 00000"])
    with TestClient(create_app(engine_factory=lambda: engine, handwriting_factory=broken)) as c:
        r = upload(c)
        health = c.get("/api/v1/health").json()
    assert r.status_code == 200, r.text
    assert any("handwriting model" in w for w in r.json()["review"]["warnings"])
    assert health["handwritingModel"] == "failed"


def test_handwriting_disabled_by_default(client):
    assert client.get("/api/v1/health").json()["handwritingModel"] == "disabled"