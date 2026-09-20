"""Bounded temporal buffering and short-gap recovery."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from plate_anonymizer.models import Detection
from plate_anonymizer.temporal.interpolation import interpolate_gap


@dataclass(slots=True)
class BufferedFrame:
    index: int
    timestamp_ms: float
    image: np.ndarray
    detections: list[Detection]


class TemporalBuffer:
    """Buffers a small number of frames so gaps can be filled before redaction."""

    def __init__(self, max_gap: int = 3) -> None:
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")
        self.max_gap = max_gap
        self._frames: deque[BufferedFrame] = deque()
        self._last_seen: dict[int, Detection] = {}

    def push(self, frame: BufferedFrame) -> list[BufferedFrame]:
        self._last_seen = {
            tid: detection
            for tid, detection in self._last_seen.items()
            if frame.index - detection.frame_index <= self.max_gap + 1
        }
        for detection in frame.detections:
            if detection.track_id is None:
                continue
            previous = self._last_seen.get(detection.track_id)
            if previous is not None:
                for recovered in interpolate_gap(previous, detection, self.max_gap):
                    for buffered in self._frames:
                        if buffered.index == recovered.frame_index:
                            buffered.detections.append(recovered)
                            break
            self._last_seen[detection.track_id] = detection

        self._frames.append(frame)
        ready: list[BufferedFrame] = []
        while len(self._frames) > self.max_gap + 1:
            ready.append(self._frames.popleft())
        return ready

    def flush(self) -> list[BufferedFrame]:
        ready = list(self._frames)
        self._frames.clear()
        self._last_seen.clear()
        return ready
