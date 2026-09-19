from plate_anonymizer.models import BoundingBox, Detection, DetectionSource
from plate_anonymizer.temporal.interpolation import interpolate_gap


def test_interpolates_short_gap() -> None:
    a = Detection(BoundingBox(0, 0, 10, 10), 0.9, 1, 100, track_id=7)
    b = Detection(BoundingBox(20, 0, 30, 10), 0.8, 3, 300, track_id=7)
    result = interpolate_gap(a, b)
    assert len(result) == 1
    assert result[0].bbox == BoundingBox(10, 0, 20, 10)
    assert result[0].source == DetectionSource.INTERPOLATION
