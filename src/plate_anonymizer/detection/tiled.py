"""Overlapping tiled inference for small plates in high-resolution frames."""

from dataclasses import replace

import numpy as np

from plate_anonymizer.detection.base import BaseDetector
from plate_anonymizer.models import BoundingBox, Detection
from plate_anonymizer.utils.boxes import iou


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    if tile <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile must be positive and overlap must be in [0, 1)")
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    origins = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


def non_max_suppression(detections: list[Detection], threshold: float = 0.5) -> list[Detection]:
    remaining = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            candidate for candidate in remaining if iou(current.bbox, candidate.bbox) < threshold
        ]
    return kept


class TiledDetector(BaseDetector):
    def __init__(
        self,
        detector: BaseDetector,
        tile_size: int = 1280,
        overlap: float = 0.2,
        merge_iou: float = 0.5,
        include_full_frame: bool = True,
    ) -> None:
        self.detector = detector
        self.tile_size = tile_size
        self.overlap = overlap
        self.merge_iou = merge_iou
        self.include_full_frame = include_full_frame

    def detect(self, frame: np.ndarray, frame_index: int, timestamp_ms: float) -> list[Detection]:
        height, width = frame.shape[:2]
        detections = (
            self.detector.detect(frame, frame_index, timestamp_ms)
            if self.include_full_frame
            else []
        )
        for y in tile_origins(height, self.tile_size, self.overlap):
            for x in tile_origins(width, self.tile_size, self.overlap):
                tile = frame[
                    y : min(y + self.tile_size, height), x : min(x + self.tile_size, width)
                ]
                for detection in self.detector.detect(tile, frame_index, timestamp_ms):
                    box = detection.bbox
                    detections.append(
                        replace(
                            detection,
                            bbox=BoundingBox(box.x1 + x, box.y1 + y, box.x2 + x, box.y2 + y),
                        )
                    )
        return non_max_suppression(detections, self.merge_iou)
