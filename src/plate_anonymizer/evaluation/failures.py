"""Utilities for exporting false-negative evidence."""

from pathlib import Path

import cv2
import numpy as np

from plate_anonymizer.models import BoundingBox
from plate_anonymizer.utils.boxes import clip_box


def save_failure_crop(
    frame: np.ndarray,
    ground_truth: BoundingBox,
    output_path: Path,
    context_padding: float = 1.0,
) -> None:
    height, width = frame.shape[:2]
    dx, dy = ground_truth.width * context_padding, ground_truth.height * context_padding
    box = clip_box(
        BoundingBox(
            ground_truth.x1 - dx,
            ground_truth.y1 - dy,
            ground_truth.x2 + dx,
            ground_truth.y2 + dy,
        ),
        width,
        height,
    )
    x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame[y1:y2, x1:x2]):
        raise RuntimeError(f"Could not write failure crop: {output_path}")
