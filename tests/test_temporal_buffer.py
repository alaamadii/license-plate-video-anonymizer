import numpy as np

from plate_anonymizer.models import BoundingBox, Detection
from plate_anonymizer.temporal.buffer import BufferedFrame, TemporalBuffer


def test_temporal_buffer_recovers_one_frame_gap() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    buffer = TemporalBuffer(max_gap=2)
    first = Detection(BoundingBox(0, 0, 4, 4), 0.9, 0, 0, track_id=1)
    last = Detection(BoundingBox(2, 0, 6, 4), 0.8, 2, 66, track_id=1)

    buffer.push(BufferedFrame(0, 0, image, [first]))
    buffer.push(BufferedFrame(1, 33, image, []))
    buffer.push(BufferedFrame(2, 66, image, [last]))
    frames = buffer.flush()

    assert len(frames[1].detections) == 1
    assert frames[1].detections[0].bbox == BoundingBox(1, 0, 5, 4)


def test_temporal_history_expires_during_long_videos() -> None:
    buffer = TemporalBuffer(max_gap=3)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    for index in range(100):
        detection = Detection(BoundingBox(0, 0, 4, 4), .9, index, index * 40, track_id=index)
        buffer.push(BufferedFrame(index, index * 40, image, [detection]))
    assert len(buffer._last_seen) <= 5
    buffer.flush()
    assert not buffer._last_seen
