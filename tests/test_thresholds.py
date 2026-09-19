from plate_anonymizer.evaluation.thresholds import threshold_sweep
from plate_anonymizer.models import BoundingBox


def test_threshold_sweep_exposes_recall_tradeoff() -> None:
    gt = [BoundingBox(0, 0, 10, 10)]
    predictions = [(BoundingBox(0, 0, 10, 10), 0.1)]
    rows = threshold_sweep(predictions, gt, thresholds=[0.05, 0.2])
    assert rows[0]["recall"] == 1.0
    assert rows[1]["recall"] == 0.0
