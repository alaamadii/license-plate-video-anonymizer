import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from plate_anonymizer.detection.yolo import YoloDetector


def fake_model(monkeypatch, names):
    model = Mock(names=names)
    model.predict.return_value = []
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model))
    return model


def test_rejects_general_object_detector(monkeypatch, tmp_path):
    fake_model(monkeypatch, {0: "person", 1: "car"})
    with pytest.raises(ValueError, match="no recognized license-plate class"):
        YoloDetector(tmp_path / "model.pt")


def test_filters_other_classes(monkeypatch, tmp_path):
    model = fake_model(monkeypatch, {0: "car", 1: "license_plate"})
    detector = YoloDetector(tmp_path / "model.pt")
    detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 0, 0)
    assert model.predict.call_args.kwargs["classes"] == [1]


def test_custom_class_id(monkeypatch, tmp_path):
    fake_model(monkeypatch, {0: "custom-label"})
    assert YoloDetector(tmp_path / "model.pt", plate_class_ids=[0]).plate_class_ids == [0]
    with pytest.raises(ValueError):
        YoloDetector(tmp_path / "model.pt", plate_class_ids=[1])
