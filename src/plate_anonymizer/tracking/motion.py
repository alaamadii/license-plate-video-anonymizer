"""Opt-in constant-velocity tracking with bounded, explicitly labelled propagation."""

import math
from dataclasses import dataclass, replace

from plate_anonymizer.models import BoundingBox, Detection, DetectionSource
from plate_anonymizer.utils.boxes import iou


@dataclass
class _Track:
    observed: Detection
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1

    def predict(self, index: int) -> BoundingBox:
        gap = index - self.observed.frame_index
        dx, dy = (v * gap for v in self.velocity)
        b = self.observed.bbox
        return BoundingBox(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy)


class MotionTracker:
    """Predictions never refresh the observation age or confirm a track.

    Two consecutive observations are required before propagation. Association uses
    predicted position, a center-distance gate and a size-change gate. It does not
    use image appearance and cannot guarantee identity through crossings/occlusion.
    """

    def __init__(self, max_gap: int = 6, min_hits: int = 2) -> None:
        if max_gap < 0 or min_hits < 2:
            raise ValueError("max_gap must be nonnegative and min_hits at least 2")
        self.max_gap, self.min_hits = max_gap, min_hits
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_index = -1

    def reset(self) -> None:
        """Drop motion history at a cut, without reusing IDs in the same video."""
        self._tracks.clear()

    def update(
        self, detections: list[Detection], frame_index: int, timestamp_ms: float
    ) -> list[Detection]:
        if frame_index <= self._last_index:
            raise ValueError("MotionTracker requires strictly increasing frame indices")
        self._last_index = frame_index
        self._tracks = {
            tid: track
            for tid, track in self._tracks.items()
            if frame_index - track.observed.frame_index <= self.max_gap + 1
        }
        candidates = []
        for tid, track in self._tracks.items():
            predicted = track.predict(frame_index)
            for di, detection in enumerate(detections):
                b = detection.bbox
                if min(b.width, b.height, predicted.width, predicted.height) <= 0:
                    continue
                size_ratio = max(
                    b.width / predicted.width,
                    predicted.width / b.width,
                    b.height / predicted.height,
                    predicted.height / b.height,
                )
                distance = math.hypot(
                    ((b.x1 + b.x2) - (predicted.x1 + predicted.x2)) / (2 * predicted.width),
                    ((b.y1 + b.y2) - (predicted.y1 + predicted.y2)) / (2 * predicted.height),
                )
                if size_ratio <= 1.8 and distance <= 0.8:
                    candidates.append((iou(predicted, b) - distance * 0.25, tid, di))
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, tid, di in sorted(candidates, reverse=True):
            if tid not in used_tracks and di not in assignments:
                assignments[di] = tid
                used_tracks.add(tid)

        output = []
        for di, detection in enumerate(detections):
            tid = assignments.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                tracked = replace(detection, track_id=tid)
                self._tracks[tid] = _Track(tracked)
            else:
                track = self._tracks[tid]
                old, b = track.observed, detection.bbox
                gap = frame_index - old.frame_index
                vx = ((b.x1 + b.x2) - (old.bbox.x1 + old.bbox.x2)) / (2 * gap)
                vy = ((b.y1 + b.y2) - (old.bbox.y1 + old.bbox.y2)) / (2 * gap)
                # Use the first measured displacement directly; smooth later updates.
                weight = 1.0 if track.hits == 1 else 0.5
                track.velocity = tuple(
                    weight * new + (1 - weight) * previous
                    for new, previous in zip((vx, vy), track.velocity, strict=True)
                )
                track.hits = track.hits + 1 if gap == 1 else 1
                tracked = replace(detection, track_id=tid)
                track.observed = tracked
            output.append(tracked)

        for track in self._tracks.values():
            gap = frame_index - track.observed.frame_index
            if not 1 <= gap <= self.max_gap or track.hits < self.min_hits:
                continue
            predicted = track.predict(frame_index)
            # Do not add an overlapping predicted mask on another current detection.
            if any(iou(predicted, detection.bbox) > 0.2 for detection in output):
                continue
            output.append(
                replace(
                    track.observed,
                    bbox=predicted,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    source=DetectionSource.TEMPORAL_PROPAGATION,
                    confidence=track.observed.confidence * (1 - gap / (self.max_gap + 1)),
                )
            )
        return output
