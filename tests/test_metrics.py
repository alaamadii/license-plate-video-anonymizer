import pytest

from plate_anonymizer.evaluation.metrics import Metrics, match_boxes
from plate_anonymizer.models import BoundingBox


def test_metrics() -> None:
    m = Metrics(tp=99, fp=5, fn=1)
    assert m.recall == pytest.approx(0.99)
    assert m.precision == pytest.approx(99 / 104)


def test_matching_is_one_to_one() -> None:
    gt = [BoundingBox(0, 0, 10, 10)]
    preds = [BoundingBox(0, 0, 10, 10), BoundingBox(0, 0, 10, 10)]
    m = match_boxes(preds, gt)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)


def test_coverage_metric_accepts_conservative_box() -> None:
    gt = [BoundingBox(10, 10, 20, 20)]
    pred = [BoundingBox(0, 0, 30, 30)]
    assert match_boxes(pred, gt, 0.95, "coverage").recall == 1.0
