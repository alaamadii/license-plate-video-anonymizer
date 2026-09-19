from plate_anonymizer.models import BoundingBox, Detection
from plate_anonymizer.tracking.simple import IoUTracker


def test_tracker_keeps_id_for_overlapping_plate() -> None:
    tracker = IoUTracker()
    first = tracker.update([Detection(BoundingBox(0, 0, 10, 10), 0.9, 0, 0)], 0)
    second = tracker.update([Detection(BoundingBox(1, 0, 11, 10), 0.9, 1, 33)], 1)
    assert first[0].track_id == second[0].track_id
