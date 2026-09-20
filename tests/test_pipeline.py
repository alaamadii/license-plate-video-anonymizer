import json

import cv2
import numpy as np
import pytest

from plate_anonymizer.models import BoundingBox, Detection
from plate_anonymizer.pipeline import anonymize_video
from plate_anonymizer.redaction.core import RedactionMode


class GapDetector:
    def detect(self, frame, frame_index, timestamp_ms):
        if frame_index in {0, 4}:
            return [Detection(BoundingBox(10, 10, 30, 20), .9, frame_index, timestamp_ms)]
        return []


def test_video_roundtrip_and_maximum_gap(tmp_path):
    source, output, metadata = [tmp_path / p for p in ('in.mp4', 'out.mp4', 'out.jsonl')]
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 240, dtype=np.uint8))
    writer.release()
    progress = []
    assert anonymize_video(
        source, output, GapDetector(), metadata, redaction_mode=RedactionMode.SOLID,
        temporal_max_gap=3, progress=lambda *args: progress.append(args),
    ) == 5
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    assert len(rows) == 5  # All three missing frames recovered before encoding.
    assert all(row["redaction_bbox"] == [7, 8, 33, 21] for row in rows)
    cap = cv2.VideoCapture(str(output))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame[12:18, 12:28].max() < 15
        count += 1
    cap.release()
    assert count == 5
    assert progress[-1] == (5, 5, 2)


def test_rejects_path_collision_before_modifying_files(tmp_path):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'original')
    with pytest.raises(ValueError, match='different files'):
        anonymize_video(source, source, GapDetector(), tmp_path / 'meta.jsonl')
    assert source.read_bytes() == b'original'
