"""Detector interface."""

from abc import ABC, abstractmethod

import numpy as np

from plate_anonymizer.models import Detection


class BaseDetector(ABC):
    @abstractmethod
    def detect(
        self, frame: np.ndarray, frame_index: int, timestamp_ms: float
    ) -> list[Detection]:
        """Return license-plate detections for one frame."""
