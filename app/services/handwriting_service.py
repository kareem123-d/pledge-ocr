import threading

import cv2
import numpy as np
from PIL import Image


class TrOcrEngine:
    def __init__(self, model_name: str):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._torch = torch
        self._processor = TrOCRProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self._model.eval()
        self._lock = threading.Lock()

    def recognize_crop(self, image: np.ndarray) -> list[tuple[str, float]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with self._lock, self._torch.no_grad():
            pixel_values = self._processor(images=pil, return_tensors="pt").pixel_values
            output = self._model.generate(
                pixel_values,
                max_new_tokens=32,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
            text = self._processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
            scores = self._model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        if not text or scores.numel() == 0:
            return []
        confidence = float(self._torch.exp(scores[0].mean()))
        return [(text, confidence)]