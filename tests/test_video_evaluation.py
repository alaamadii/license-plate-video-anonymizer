import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from plate_anonymizer.cli import app
from plate_anonymizer.evaluation.video import evaluate_video


def inputs(tmp_path):
    gt, predictions = tmp_path / "gt.json", tmp_path / "pred.jsonl"
    gt.write_text(
        json.dumps(
            {
                "evaluated_frames": [0, 1, 2],
                "annotations": [
                    {"frame_index": 0, "bbox": [10, 10, 30, 20], "tags": ["small"]},
                    {"frame_index": 1, "bbox": [10, 10, 30, 20], "tags": ["night"]},
                ],
            }
        )
    )
    rows = [
        {
            "frame_index": i,
            "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 20},
            "redaction_bbox": [8, 8, 32, 22],
        }
        for i in (0, 2, 99)
    ]
    predictions.write_text("\n".join(map(json.dumps, rows)))
    return gt, predictions


def test_evaluation_matches_only_same_frame_and_counts_negatives(tmp_path):
    gt, predictions = inputs(tmp_path)
    result = evaluate_video(predictions, gt)
    assert result["overall"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert result["ignored_prediction_rows"] == 1
    assert result["by_tag"]["night"]["recall"] == 0
    assert result["by_tag"]["small"]["recall"] == 1
    assert result["false_negatives"][0]["frame_index"] == 1


def test_coverage_uses_rendered_boxes(tmp_path):
    gt, predictions = inputs(tmp_path)
    result = evaluate_video(predictions, gt, "coverage", 0.95, "redaction_bbox")
    assert result["overall"]["tp"] == 1
    predictions.write_text('{"frame_index":0,"bbox":[10,10,30,20]}')
    with pytest.raises(ValueError, match="Missing redaction_bbox"):
        evaluate_video(predictions, gt, "coverage", 0.95, "redaction_bbox")


def test_exports_actual_false_negative_crop(tmp_path):
    gt, predictions = inputs(tmp_path)
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((48, 64, 3), 200, np.uint8))
    writer.release()
    directory = tmp_path / "failures"
    result = evaluate_video(predictions, gt, video=video, failure_dir=directory)
    crop = directory / result["false_negatives"][0]["crop"]
    assert cv2.imread(str(crop)).size > 0


def test_cli_writes_report_and_protects_inputs(tmp_path):
    gt, predictions = inputs(tmp_path)
    report = tmp_path / "report.json"
    args = ["evaluate-video", "--predictions", str(predictions), "--ground-truth", str(gt)]
    result = CliRunner().invoke(app, [*args, "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert json.loads(report.read_text())["overall"]["fn"] == 1
    result = CliRunner().invoke(app, [*args, "--report", str(gt)])
    assert result.exit_code != 0


def test_invalid_annotation_frames_are_rejected(tmp_path):
    gt, predictions = inputs(tmp_path)
    manifest = json.loads(gt.read_text())
    manifest["evaluated_frames"] = [0]
    gt.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing from evaluated_frames"):
        evaluate_video(predictions, gt)
