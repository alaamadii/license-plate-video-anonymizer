"""JSONL detection metadata."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from plate_anonymizer.models import BoundingBox, Detection


def append_detections(
    path: Path,
    detections: Iterable[Detection],
    redaction_boxes: Iterable[BoundingBox] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        boxes = iter(redaction_boxes) if redaction_boxes is not None else None
        for detection in detections:
            payload = asdict(detection)
            payload["source"] = detection.source.value
            if boxes is not None:
                box = next(boxes)
                payload["redaction_bbox"] = list(map(int, (box.x1, box.y1, box.x2, box.y2)))
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
