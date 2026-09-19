"""Plate redaction operations."""

from enum import StrEnum

import cv2
import numpy as np

from plate_anonymizer.models import BoundingBox
from plate_anonymizer.utils.boxes import clip_box


class RedactionMode(StrEnum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    SOLID = "solid"


def redact(frame: np.ndarray, box: BoundingBox, mode: RedactionMode) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    b = clip_box(box, w, h)
    x1, y1, x2, y2 = map(int, (b.x1, b.y1, b.x2, b.y2))
    if x2 <= x1 or y2 <= y1:
        return out
    roi = out[y1:y2, x1:x2]
    if mode == RedactionMode.BLUR:
        kx = max(3, (roi.shape[1] // 3) | 1)
        ky = max(3, (roi.shape[0] // 3) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kx, ky), 0)
    elif mode == RedactionMode.PIXELATE:
        sw, sh = max(1, roi.shape[1] // 12), max(1, roi.shape[0] // 12)
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = cv2.resize(
            small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    elif mode == RedactionMode.SOLID:
        out[y1:y2, x1:x2] = 0
    else:
        raise ValueError(f"Unsupported redaction mode: {mode}")
    return out
