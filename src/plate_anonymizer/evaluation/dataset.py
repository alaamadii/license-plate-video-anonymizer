"""Simple frame-aware JSON evaluation format."""

import json
from dataclasses import dataclass
from pathlib import Path

from plate_anonymizer.models import BoundingBox


@dataclass(frozen=True, slots=True)
class Annotation:
    frame_index: int
    bbox: BoundingBox
    tags: tuple[str, ...] = ()


def load_annotations(path: Path) -> list[Annotation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Annotation(
            frame_index=int(item["frame_index"]),
            bbox=BoundingBox(*map(float, item["bbox"])),
            tags=tuple(item.get("tags", [])),
        )
        for item in raw
    ]
