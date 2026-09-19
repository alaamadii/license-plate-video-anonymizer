"""Dependency-free IoU tracker used as a deterministic baseline."""

from dataclasses import replace

from plate_anonymizer.models import Detection
from plate_anonymizer.utils.boxes import iou


class IoUTracker:
    def __init__(self, match_iou: float = 0.3, max_age: int = 3) -> None:
        self.match_iou = match_iou
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, Detection] = {}

    def update(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        active = {
            tid: det
            for tid, det in self._tracks.items()
            if frame_index - det.frame_index <= self.max_age
        }
        assigned: set[int] = set()
        output: list[Detection] = []
        for detection in detections:
            best_id = None
            best_score = self.match_iou
            for tid, previous in active.items():
                if tid in assigned:
                    continue
                score = iou(detection.bbox, previous.bbox)
                if score >= best_score:
                    best_score, best_id = score, tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            assigned.add(best_id)
            tracked = replace(detection, track_id=best_id)
            self._tracks[best_id] = tracked
            output.append(tracked)
        self._tracks = {
            tid: det
            for tid, det in self._tracks.items()
            if frame_index - det.frame_index <= self.max_age
        }
        return output
