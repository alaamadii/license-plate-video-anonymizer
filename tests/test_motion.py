import pytest

from plate_anonymizer.models import BoundingBox, Detection, DetectionSource
from plate_anonymizer.tracking.motion import MotionTracker


def detection(index, x=10, width=20):
    return Detection(BoundingBox(x, 10, x + width, 20), 0.9, index, index * 33.3)


def test_tracks_motion_through_gap_without_iou_with_last_observation():
    tracker = MotionTracker(max_gap=6)
    first = tracker.update([detection(0)], 0, 0)[0]
    tracker.update([detection(1, 18)], 1, 33.3)
    for index in range(2, 7):
        result = tracker.update([], index, index * 33.3)
        assert len(result) == 1
        assert result[0].bbox.x1 == 10 + 8 * index
        assert result[0].source == DetectionSource.TEMPORAL_PROPAGATION
        assert result[0].frame_index == index
    reacquired = tracker.update([detection(7, 66)], 7, 233.1)[0]
    assert reacquired.track_id == first.track_id
    assert reacquired.source == DetectionSource.DETECTOR


def test_single_false_positive_is_not_propagated():
    tracker = MotionTracker()
    tracker.update([detection(0)], 0, 0)
    assert tracker.update([], 1, 33.3) == []


def test_predictions_expire_without_refreshing_observation_age():
    tracker = MotionTracker(max_gap=2)
    tracker.update([detection(0)], 0, 0)
    tracker.update([detection(1, 11)], 1, 33.3)
    assert tracker.update([], 2, 66.6)
    assert tracker.update([], 3, 99.9)
    assert tracker.update([], 4, 133.2) == []
    assert tracker.update([], 5, 166.5) == []


def test_reset_drops_history_without_reusing_ids():
    tracker = MotionTracker()
    tid = tracker.update([detection(0)], 0, 0)[0].track_id
    tracker.update([detection(1)], 1, 33.3)
    tracker.reset()
    assert tracker.update([], 2, 66.6) == []
    assert tracker.update([detection(3)], 3, 99.9)[0].track_id != tid


def test_large_box_jump_does_not_hijack_plate_track():
    tracker = MotionTracker()
    tid = tracker.update([detection(0)], 0, 0)[0].track_id
    tracker.update([detection(1)], 1, 33.3)
    result = tracker.update([detection(2, width=200)], 2, 66.6)
    direct = [d for d in result if d.source == DetectionSource.DETECTOR]
    assert direct[0].track_id != tid  # The direct false positive still needs detector work.


def test_assignments_are_one_to_one():
    tracker = MotionTracker()
    tracker.update([detection(0, 10), detection(0, 50)], 0, 0)
    result = tracker.update([detection(1, 12), detection(1, 48)], 1, 33.3)
    assert len(set(d.track_id for d in result)) == 2


def test_repeated_index_rejected():
    tracker = MotionTracker()
    tracker.update([], 0, 0)
    with pytest.raises(ValueError, match="increasing"):
        tracker.update([], 0, 0)
