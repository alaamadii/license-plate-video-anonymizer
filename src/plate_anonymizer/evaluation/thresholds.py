"""Threshold sweeps for recall/precision trade-off analysis."""

from dataclasses import asdict

from plate_anonymizer.evaluation.metrics import match_boxes
from plate_anonymizer.models import BoundingBox


def threshold_sweep(
    predictions: list[tuple[BoundingBox, float]],
    ground_truth: list[BoundingBox],
    thresholds: list[float] | None = None,
    match_threshold: float = 0.5,
    metric: str = "iou",
) -> list[dict[str, float | int]]:
    thresholds = thresholds or [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    rows: list[dict[str, float | int]] = []
    for confidence in thresholds:
        boxes = [box for box, score in predictions if score >= confidence]
        result = match_boxes(boxes, ground_truth, match_threshold, metric)
        rows.append({
            "confidence": confidence,
            **asdict(result),
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        })
    return rows
