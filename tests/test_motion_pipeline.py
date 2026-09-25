import json

import cv2
import numpy as np

from plate_anonymizer.evaluation.video import evaluate_video
from plate_anonymizer.models import BoundingBox, Detection
from plate_anonymizer.pipeline import anonymize_video
from plate_anonymizer.redaction.core import RedactionMode


class MovingDetector:
    def detect(self, frame, frame_index, timestamp_ms):
        if 4 <= frame_index <= 9:
            return []
        x = 20 + 2 * frame_index
        return [Detection(BoundingBox(x, 20, x + 20, 30), 0.9, frame_index, timestamp_ms)]


def test_moving_six_frame_gap_is_measurably_recovered(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 64))
    labels = []
    for index in range(15):
        frame = np.full((64, 128, 3), 100, np.uint8)
        x = 20 + 2 * index
        frame[20:30, x : x + 20] = 240
        writer.write(frame)
        labels.append({"frame_index": index, "bbox": [x, 20, x + 20, 30], "track_id": "a"})
    writer.release()
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"evaluated_frames": list(range(15)), "annotations": labels}))
    results = {}
    for mode in ("iou", "motion"):
        video, metadata = tmp_path / f"{mode}.mp4", tmp_path / f"{mode}.jsonl"
        anonymize_video(
            source,
            video,
            MovingDetector(),
            metadata,
            redaction_mode=RedactionMode.SOLID,
            tracker_mode=mode,
        )
        results[mode] = evaluate_video(metadata, gt, "coverage", 0.95, "redaction_bbox")
        if mode == "motion":
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 7)
            ok, frame = cap.read()
            cap.release()
            assert ok and frame[22:28, 36:50].max() < 15
    assert results["iou"]["overall"]["fn"] == 6
    assert results["motion"]["overall"]["fn"] == 0
    assert results["motion"]["overall"]["fp"] == 0
    assert results["motion"]["continuity"]["tracks"]["a"]["longest_uncovered_run_frames"] == 0


def test_scene_change_prevents_ghost_mask(tmp_path):
    source = tmp_path / "cut.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 64))
    for i in range(8):
        writer.write(np.full((64, 128, 3), 30 if i < 4 else 220, np.uint8))
    writer.release()
    metadata = tmp_path / "out.jsonl"
    anonymize_video(source, tmp_path / "out.mp4", MovingDetector(), metadata, tracker_mode="motion")
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    assert all(row["frame_index"] < 4 for row in rows)


def test_prediction_leaving_frame_does_not_emit_empty_mask(tmp_path):
    class ExitingDetector:
        def detect(self, frame, frame_index, timestamp_ms):
            if frame_index > 1:
                return []
            x = 100 + frame_index * 10
            return [Detection(BoundingBox(x, 20, x + 18, 30), 0.9, frame_index, timestamp_ms)]

    source = tmp_path / "exit.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 64))
    for _ in range(8):
        writer.write(np.full((64, 128, 3), 100, np.uint8))
    writer.release()
    metadata = tmp_path / "out.jsonl"
    anonymize_video(
        source, tmp_path / "out.mp4", ExitingDetector(), metadata, tracker_mode="motion"
    )
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    assert max(row["frame_index"] for row in rows) == 2
    assert all(row["redaction_bbox"][0] < row["redaction_bbox"][2] for row in rows)
