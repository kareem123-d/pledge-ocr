import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.pledge_slip import FieldStatus, OcrLine
from app.services.document_pipeline import PageImage
from app.services.field_extractor import Candidate, TableResult
from app.services.field_validator import Thresholds, build_field
from app.services.value_refiner import crop_variants, refine_all, refine_candidate
from helpers import FakeEngine, make_image_bytes

TH = Thresholds()
IMAGES = {1: PageImage(image=np.full((300, 800, 3), 255, dtype=np.uint8), scale=1.0)}


def line(text, index=0, conf=0.95, box=(100, 100, 300, 120)):
    x0, y0, x1, y1 = box
    return OcrLine(
        page=1, index=index, text=text, confidence=conf,
        box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], source="ocr",
    )


def test_invalid_value_is_reread():
    engine = FakeEngine(crop_fn=lambda img: [("14,00,000", 0.95)])
    cand = Candidate("00000", [line("00000")], 0.85)
    new, used = refine_candidate("amount", cand, IMAGES, engine, TH)
    assert used
    assert new.raw == "14,00,000"
    field = build_field("amount", new, TH)
    assert field.value == 1400000
    assert field.status == FieldStatus.UNCERTAIN
    assert "00000" in field.note
    assert field.ocrConfidence == 0.95
    assert [e.text for e in field.evidence] == ["00000"]


def test_confident_valid_value_is_not_reread():
    engine = FakeEngine(crop_fn=lambda img: [("99999", 0.99)])
    cand = Candidate("50000", [line("50000", conf=0.98)], 0.85)
    new, used = refine_candidate("amount", cand, IMAGES, engine, TH)
    assert not used
    assert new is cand
    assert engine.crop_calls == 0


def test_unparseable_reread_does_not_replace():
    engine = FakeEngine(crop_fn=lambda img: [("garbage", 0.99)])
    cand = Candidate("H2", [line("H2", conf=0.89)], 0.85)
    new, used = refine_candidate("weight", cand, IMAGES, engine, TH)
    assert used
    assert new is cand


def test_small_confidence_gain_does_not_replace_parseable_value():
    engine = FakeEngine(crop_fn=lambda img: [("91", 0.73)])
    cand = Candidate("90", [line("90", conf=0.71)], 0.85)
    new, _ = refine_candidate("weight", cand, IMAGES, engine, TH)
    assert new is cand


def test_crop_budget_is_respected():
    engine = FakeEngine(crop_fn=lambda img: [("14,00,000", 0.95)])
    fields = {
        "loanAmount": Candidate("00000", [line("00000", 0)], 0.85),
        "appraisedValue": Candidate("00000", [line("00000", 1, box=(100, 150, 300, 170))], 0.85),
    }
    used = refine_all(fields, TableResult(), IMAGES, engine, TH, max_crops=1)
    assert used == 1
    assert fields["loanAmount"].raw == "14,00,000"
    assert fields["appraisedValue"].raw == "00000"


def test_colored_ink_variant_is_added():
    crop = np.full((20, 60, 3), 255, np.uint8)
    crop[5:15, 10:50] = (0, 0, 220)
    variants = crop_variants(crop)
    assert len(variants) == 4
    assert all(v.shape[0] >= 32 for v in variants)


def test_upload_rereads_unreadable_amount():
    engine = FakeEngine(
        text_fn=lambda img: ["Amount of Loan: 00000"],
        crop_fn=lambda img: [("14,00,000", 0.95)],
    )
    with TestClient(create_app(engine_factory=lambda: engine)) as c:
        r = c.post("/api/v1/pledge-slips/upload", files={"file": ("slip.png", make_image_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    amount = r.json()["extractedData"]["loanAmount"]
    assert amount["value"] == 1400000
    assert amount["status"] == "uncertain"
    assert "Re-read" in amount["note"]
    assert engine.crop_calls >= 1
def test_low_confidence_reread_is_rejected():
    engine = FakeEngine(crop_fn=lambda img: [("1", 0.18)])
    cand = Candidate("1/", [line("1/", conf=0.65)], 0.85)
    new, _ = refine_candidate("weight", cand, IMAGES, engine, TH)
    assert new is cand


def test_text_fields_are_not_reread():
    engine = FakeEngine(crop_fn=lambda img: [("Othersiaocd", 0.99)])
    cand = Candidate("Others LOCEC", [line("Others LOCEC", conf=0.64)], 0.85)
    new, used = refine_candidate("text", cand, IMAGES, engine, TH)
    assert new is cand
    assert not used