"""Core domain models."""

from dataclasses import dataclass
from enum import StrEnum


class DetectionSource(StrEnum):
    DETECTOR = "detector"
    TRACKER = "tracker"
    INTERPOLATION = "interpolation"
    TEMPORAL_PROPAGATION = "temporal_propagation"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    frame_index: int
    timestamp_ms: float
    class_name: str = "license_plate"
    track_id: int | None = None
    source: DetectionSource = DetectionSource.DETECTOR
