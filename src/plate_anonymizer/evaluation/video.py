"""Evaluate pipeline JSONL on an explicitly reviewed set of frames from one video."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

from plate_anonymizer.evaluation.continuity import summarize_continuity
from plate_anonymizer.evaluation.failures import save_failure_crop
from plate_anonymizer.evaluation.metrics import Metrics, match_indices
from plate_anonymizer.models import BoundingBox


def _box(value) -> BoundingBox:
    values = [value[k] for k in ("x1", "y1", "x2", "y2")] if isinstance(value, dict) else value
    if len(values) != 4:
        raise ValueError("A bbox must contain four coordinates.")
    box = BoundingBox(*map(float, values))
    if not all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2)) or box.area <= 0:
        raise ValueError("Boxes must have finite coordinates and positive area.")
    return box


def _index(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("frame_index must be a nonnegative integer.")
    return value


def evaluate_video(
    predictions: Path,
    ground_truth: Path,
    metric: str = "iou",
    threshold: float = 0.5,
    box_field: str = "bbox",
    video: Path | None = None,
    failure_dir: Path | None = None,
) -> dict:
    """Only evaluated_frames enter metrics; empty reviewed frames count false positives."""
    if metric not in {"iou", "coverage"} or not 0 < threshold <= 1:
        raise ValueError("Use iou/coverage and a threshold greater than 0 and at most 1.")
    if box_field not in {"bbox", "redaction_bbox"}:
        raise ValueError("box_field must be bbox or redaction_bbox.")
    if failure_dir is not None and video is None:
        raise ValueError("--failure-dir requires --video.")
    manifest = json.loads(ground_truth.read_text(encoding="utf-8"))
    frames = [_index(i) for i in manifest["evaluated_frames"]]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("evaluated_frames must be nonempty and contain no duplicates.")
    reviewed = set(frames)
    annotations = defaultdict(list)
    track_ids = defaultdict(list)
    has_tracks = any("track_id" in item for item in manifest["annotations"])
    for item in manifest["annotations"]:
        index = _index(item["frame_index"])
        if index not in reviewed:
            raise ValueError("Annotation frame is missing from evaluated_frames.")
        annotations[index].append((_box(item["bbox"]), item.get("tags", [])))
        if has_tracks:
            track_id = item.get("track_id")
            if not isinstance(track_id, str) or not track_id.strip():
                raise ValueError("All annotations need nonempty string track_id for continuity.")
            if track_id in track_ids[index]:
                raise ValueError("A track_id may appear only once per frame.")
            track_ids[index].append(track_id)
    predicted = defaultdict(list)
    ignored = 0
    with predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            index = _index(item["frame_index"])
            if index not in reviewed:
                ignored += 1
                continue
            if box_field not in item:
                raise ValueError(f"Missing {box_field} in metadata; rerun anonymization.")
            predicted[index].append(_box(item[box_field]))
    tp = fp = fn = 0
    tags = defaultdict(lambda: {"tp": 0, "fn": 0})
    per_frame, failures = [], []
    tracks = defaultdict(list)
    for index in sorted(reviewed):
        gt = annotations[index]
        boxes = predicted[index]
        pairs = match_indices(boxes, [b for b, _ in gt], threshold, metric)
        matched = {gi for _, gi in pairs}
        counts = {"tp": len(pairs), "fp": len(boxes) - len(pairs), "fn": len(gt) - len(pairs)}
        tp += counts["tp"]
        fp += counts["fp"]
        fn += counts["fn"]
        per_frame.append({"frame_index": index, **counts})
        for gi, (box, item_tags) in enumerate(gt):
            if has_tracks:
                tracks[track_ids[index][gi]].append((index, gi in matched))
            for tag in set(item_tags):
                tags[tag]["tp" if gi in matched else "fn"] += 1
            if gi not in matched:
                failures.append(
                    {
                        "frame_index": index,
                        "annotation_index": gi,
                        "bbox": [box.x1, box.y1, box.x2, box.y2],
                        "tags": item_tags,
                    }
                )
    if failure_dir is not None:
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open evidence video: {video}")
            for failure in failures:
                cap.set(cv2.CAP_PROP_POS_FRAMES, failure["frame_index"])
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode evidence frame {failure['frame_index']}")
                name = f"frame_{failure['frame_index']:08d}_plate_{failure['annotation_index']}.jpg"
                save_failure_crop(frame, _box(failure["bbox"]), failure_dir / name)
                failure["crop"] = name
        finally:
            cap.release()
    result = Metrics(tp, fp, fn)
    return {
        "schema_version": 1,
        "metric": metric,
        "threshold": threshold,
        "box_field": box_field,
        "matching": "greedy_score_ordered_one_to_one",
        "evaluated_frame_count": len(reviewed),
        "ignored_prediction_rows": ignored,
        "overall": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        },
        "by_tag": {
            tag: {**counts, "recall": counts["tp"] / (counts["tp"] + counts["fn"])}
            for tag, counts in sorted(tags.items())
        },
        "per_frame": per_frame,
        "false_negatives": failures,
        "continuity": summarize_continuity(tracks) if has_tracks else None,
        "inputs": {
            "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "ground_truth_sha256": hashlib.sha256(ground_truth.read_bytes()).hexdigest(),
        },
    }
