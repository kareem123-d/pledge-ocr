import io
import threading
import time

import fitz
from PIL import Image

from app.services.ocr_service import RawLine


class FakeEngine:
    def __init__(self, text_fn=None, fail_on_calls=(), delay: float = 0.0, crop_fn=None):
        self.text_fn = text_fn or (lambda image: ["Customer Name: Test User"])
        self.fail_on_calls = set(fail_on_calls)
        self.delay = delay
        self.crop_fn = crop_fn
        self.calls = 0
        self.crop_calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def recognize(self, image):
        with self._lock:
            self.calls += 1
            call = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if call in self.fail_on_calls:
                raise RuntimeError("engine exploded at C:\\secret\\model")
            return [
                RawLine(text=t, confidence=0.9, box=[[0, 0], [10, 0], [10, 10], [0, 10]])
                for t in self.text_fn(image)
            ]
        finally:
            with self._lock:
                self.active -= 1

    def recognize_crop(self, image):
        with self._lock:
            self.crop_calls += 1
        return list(self.crop_fn(image)) if self.crop_fn else []


def make_image_bytes(fmt: str = "PNG", size=(400, 200), color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def make_text_pdf(text: str) -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(pages: int = 1) -> bytes:
    img = make_image_bytes()
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, stream=img)
    data = doc.tobytes()
    doc.close()
    return data