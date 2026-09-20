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
        plate_class_ids: list[int] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'YOLO support requires the optional dependency: pip install -e ".[yolo]"'
            ) from exc
        self._model: Any = YOLO(str(model_path))
        names = self._model.names
        if plate_class_ids is None:
            aliases = {"licenseplate", "licenceplate", "numberplate", "plate", "lp"}
            plate_class_ids = [
                int(index) for index, name in names.items()
                if "".join(c for c in name.lower() if c.isalnum()) in aliases
            ]
        if not plate_class_ids or any(index not in names for index in plate_class_ids):
            raise ValueError(
                "Model has no recognized license-plate class. Generic yolov8n.pt weights "
                "do not detect plates. Use plate-specific weights, or --plate-class-id "
                "for a custom plate label. "
                f"Model classes: {names}"
            )
        self.plate_class_ids = plate_class_ids
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
            classes=self.plate_class_ids,
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
