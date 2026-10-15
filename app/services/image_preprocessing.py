import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.config import Settings

LOW_CONTRAST_STD = 60.0


def decode_image(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as img:
        oriented = ImageOps.exif_transpose(img)
        rgb = np.asarray(oriented.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_for_ocr(image: np.ndarray, max_side: int, min_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side:
        scale = min_side / longest
    else:
        return image, 1.0
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation
    )
    return resized, scale


def enhance(image: np.ndarray, denoise: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    changed = False
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
        changed = True
    if float(gray.std()) < LOW_CONTRAST_STD:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        changed = True
    if not changed:
        return image
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def prepare_for_ocr(image: np.ndarray, settings: Settings) -> tuple[np.ndarray, float]:
    resized, scale = resize_for_ocr(image, settings.ocr_max_side, settings.ocr_min_side)
    return enhance(resized, settings.ocr_denoise), scale

def rotate_image(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    border = (255, 255, 255) if image.ndim == 3 else 255
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=border)
    return rotated, matrix


def map_points_back(points: list[list[float]], matrix: np.ndarray) -> list[list[float]]:
    inv = cv2.invertAffineTransform(matrix)
    return [
        [
            float(inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]),
            float(inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]),
        ]
        for x, y in points
    ]