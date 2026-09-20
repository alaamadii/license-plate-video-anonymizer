"""Frame-aware evaluation and hard-case breakdown."""

from collections import defaultdict
from dataclasses import asdict

from plate_anonymizer.evaluation.dataset import Annotation
from plate_anonymizer.evaluation.metrics import Metrics, match_boxes
from plate_anonymizer.models import BoundingBox


def evaluate_frames(
    predictions: dict[int, list[BoundingBox]],
    annotations: list[Annotation],
    threshold: float = 0.5,
    metric: str = "iou",
) -> dict[str, object]:
    gt_by_frame: dict[int, list[BoundingBox]] = defaultdict(list)
    for annotation in annotations:
        gt_by_frame[annotation.frame_index].append(annotation.bbox)

    total = Metrics(0, 0, 0)
    for frame in set(gt_by_frame) | set(predictions):
        current = match_boxes(
            predictions.get(frame, []), gt_by_frame.get(frame, []), threshold, metric
        )
        total = Metrics(total.tp + current.tp, total.fp + current.fp, total.fn + current.fn)

    tags: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        for tag in annotation.tags:
            tags[tag].append(annotation)

    return {
        "overall": {
            **asdict(total),
            "precision": total.precision,
            "recall": total.recall,
            "f1": total.f1,
        },
        "ground_truth_instances": len(annotations),
        "tag_counts": {tag: len(items) for tag, items in sorted(tags.items())},
    }
