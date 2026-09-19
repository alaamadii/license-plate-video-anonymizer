"""JSONL detection metadata."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from plate_anonymizer.models import Detection


def append_detections(path: Path, detections: Iterable[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for detection in detections:
            payload = asdict(detection)
            payload["source"] = detection.source.value
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
