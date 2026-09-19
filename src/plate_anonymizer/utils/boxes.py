"""Bounding-box geometry used by detection and evaluation."""

from plate_anonymizer.models import BoundingBox


def clip_box(box: BoundingBox, width: int, height: int) -> BoundingBox:
    return BoundingBox(
        x1=min(max(box.x1, 0.0), float(width)),
        y1=min(max(box.y1, 0.0), float(height)),
        x2=min(max(box.x2, 0.0), float(width)),
        y2=min(max(box.y2, 0.0), float(height)),
    )


def expand_box(box: BoundingBox, padding: float, width: int, height: int) -> BoundingBox:
    if padding < 0:
        raise ValueError("padding must be non-negative")
    dx = box.width * padding
    dy = box.height * padding
    return clip_box(
        BoundingBox(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy),
        width,
        height,
    )


def intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return w * h


def iou(a: BoundingBox, b: BoundingBox) -> float:
    intersection = intersection_area(a, b)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def coverage_ratio(prediction: BoundingBox, ground_truth: BoundingBox) -> float:
    """Fraction of the ground-truth plate covered by the redaction prediction."""
    if ground_truth.area <= 0:
        return 0.0
    return intersection_area(prediction, ground_truth) / ground_truth.area
