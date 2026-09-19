import pytest

from plate_anonymizer.models import BoundingBox
from plate_anonymizer.utils.boxes import coverage_ratio, expand_box, iou


def test_iou_identical_boxes() -> None:
    box = BoundingBox(10, 10, 20, 20)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes() -> None:
    assert iou(BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)) == 0.0


def test_coverage_ratio_can_reward_conservative_redaction() -> None:
    ground_truth = BoundingBox(10, 10, 20, 20)
    prediction = BoundingBox(5, 5, 25, 25)
    assert coverage_ratio(prediction, ground_truth) == pytest.approx(1.0)


def test_expand_box_clips_to_frame() -> None:
    result = expand_box(BoundingBox(5, 5, 15, 15), 1.0, width=20, height=20)
    assert result == BoundingBox(0, 0, 20, 20)


def test_negative_padding_is_rejected() -> None:
    with pytest.raises(ValueError):
        expand_box(BoundingBox(1, 1, 2, 2), -0.1, 10, 10)
