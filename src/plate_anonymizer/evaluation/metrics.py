"""Detection matching and privacy metrics."""

from dataclasses import dataclass
from typing import Callable

from plate_anonymizer.models import BoundingBox
from plate_anonymizer.utils.boxes import coverage_ratio, iou


@dataclass(frozen=True, slots=True)
class Metrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_boxes(
    predictions: list[BoundingBox],
    ground_truth: list[BoundingBox],
    threshold: float = 0.5,
    metric: str = "iou",
) -> Metrics:
    scorer: Callable[[BoundingBox, BoundingBox], float]
    if metric == "iou":
        scorer = iou
    elif metric == "coverage":
        scorer = coverage_ratio
    else:
        raise ValueError("metric must be 'iou' or 'coverage'")

    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            score = scorer(pred, gt)
            if score >= threshold:
                candidates.append((score, pi, gi))
    candidates.sort(reverse=True)

    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    for _, pi, gi in candidates:
        if pi not in used_predictions and gi not in used_ground_truth:
            used_predictions.add(pi)
            used_ground_truth.add(gi)

    tp = len(used_ground_truth)
    return Metrics(tp=tp, fp=len(predictions) - tp, fn=len(ground_truth) - tp)
