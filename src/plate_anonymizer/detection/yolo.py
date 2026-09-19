"""Optional Ultralytics YOLO detector adapter.

Ultralytics is deliberately an optional dependency. Model weights are not bundled.
Users are responsible for selecting appropriately licensed license-plate weights.
"""

from pathlib import Path
from typing import Any

import numpy as np

from plate_anonymizer.detection.base import BaseDetector
from plate_anonymizer.models import BoundingBox, Detection


class YoloDetector(BaseDetector):
    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.15,
        iou: float = 0.5,
        device: str = "cpu",
        image_size: int = 1280,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'YOLO support requires the optional dependency: pip install -e ".[yolo]"'
            ) from exc
        self._model: Any = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.image_size = image_size

    def detect(
        self, frame: np.ndarray, frame_index: int, timestamp_ms: float
    ) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for xyxy, conf in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                strict=True,
            ):
                detections.append(
                    Detection(
                        bbox=BoundingBox(*map(float, xyxy)),
                        confidence=float(conf),
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
        return detections
