import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import AppError

logger = logging.getLogger(__name__)


@dataclass
class RawLine:
    text: str
    confidence: float | None
    box: list[list[float]] | None


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> list[RawLine]: ...

    def recognize_crop(self, image: np.ndarray) -> list[tuple[str, float]]: ...


class HandwritingEngine(Protocol):
    def recognize_crop(self, image: np.ndarray) -> list[tuple[str, float]]: ...
def _preload_torch_if_installed() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return
    try:
        import torch  # noqa: F401
    except Exception:
        logger.warning("PyTorch is installed but failed to load; the handwriting model will be unavailable", exc_info=True)

class PaddleOcrEngine:
    def __init__(self, lang: str, use_angle_cls: bool):
        _preload_torch_if_installed()
        from paddleocr import PaddleOCR
        self._use_angle_cls = use_angle_cls
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        self._lock = threading.Lock()

    def recognize(self, image: np.ndarray) -> list[RawLine]:
        with self._lock:
            result = self._ocr.ocr(image, cls=self._use_angle_cls)
        lines: list[RawLine] = []
        for page in result or []:
            for item in page or []:
                box, (text, score) = item[0], item[1]
                lines.append(
                    RawLine(
                        text=str(text),
                        confidence=float(score),
                        box=[[float(p[0]), float(p[1])] for p in box],
                    )
                )
        return lines

    def recognize_crop(self, image: np.ndarray) -> list[tuple[str, float]]:
        with self._lock:
            result = self._ocr.ocr(image, det=False, cls=False)
        readings: list[tuple[str, float]] = []
        for page in result or []:
            for item in page or []:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                    readings.append((item[0], float(item[1])))
        return readings


def _load_trocr(model_name: str) -> HandwritingEngine:
    from app.services.handwriting_service import TrOcrEngine

    return TrOcrEngine(model_name)


class OcrService:
    def __init__(
        self,
        settings: Settings,
        engine_factory: Callable[[], OcrEngine] | None = None,
        handwriting_factory: Callable[[], HandwritingEngine] | None = None,
    ):
        self._factory = engine_factory or (
            lambda: PaddleOcrEngine(settings.ocr_lang, settings.ocr_use_angle_cls)
        )
        if handwriting_factory is not None:
            self._hw_factory = handwriting_factory
        elif settings.enable_trocr:
            self._hw_factory = lambda: _load_trocr(settings.trocr_model)
        else:
            self._hw_factory = None
        self._engine: OcrEngine | None = None
        self._handwriting: HandwritingEngine | None = None
        self._handwriting_failed = False
        self._init_lock = threading.Lock()
        self.semaphore = asyncio.Semaphore(max(1, settings.ocr_max_concurrency))

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def handwriting_status(self) -> str:
        if self._hw_factory is None:
            return "disabled"
        if self._handwriting is not None:
            return "loaded"
        if self._handwriting_failed:
            return "failed"
        return "not_loaded"

    def get_engine(self) -> OcrEngine:
        if self._engine is None:
            with self._init_lock:
                if self._engine is None:
                    try:
                        self._engine = self._factory()
                    except Exception:
                        logger.exception("Failed to initialize OCR engine")
                        raise AppError(
                            503, "ocr_unavailable", "The OCR engine is not available. Please try again later."
                        )
        return self._engine

    def get_handwriting(self) -> HandwritingEngine | None:
        if self._hw_factory is None or self._handwriting_failed:
            return None
        if self._handwriting is None:
            with self._init_lock:
                if self._handwriting is None and not self._handwriting_failed:
                    try:
                        self._handwriting = self._hw_factory()
                        logger.info("Handwriting model loaded")
                    except Exception:
                        logger.exception("Handwriting model failed to load; continuing without it")
                        self._handwriting_failed = True
        return self._handwriting

    def warm_up(self) -> None:
        self.get_engine()
        self.get_handwriting()