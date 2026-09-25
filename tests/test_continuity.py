import json

import pytest

from plate_anonymizer.evaluation.continuity import summarize_continuity
from plate_anonymizer.evaluation.video import evaluate_video


def test_unreviewed_frames_do_not_invent_continuity():
    result = summarize_continuity({"a": [(0, True), (1, False), (2, False), (10, False)]})
    track = result["tracks"]["a"]
    assert track["uncovered_frames"] == 3
    assert track["longest_uncovered_run_frames"] == 2
    assert track["covered_to_uncovered_transitions"] == 1
    assert result["fully_protected_reviewed_tracks"] == 0


def test_continuity_uses_gt_identity_not_predicted_track_ids(tmp_path):
    gt, predictions = tmp_path / "gt.json", tmp_path / "pred.jsonl"
    labels = [{"frame_index": i, "bbox": [10, 10, 30, 20], "track_id": "car-a"} for i in range(4)]
    gt.write_text(json.dumps({"evaluated_frames": list(range(4)), "annotations": labels}))
    predictions.write_text(
        "\n".join(
            json.dumps(
                {
                    "frame_index": i,
                    "bbox": [10, 10, 30, 20],
                    "track_id": i + 99,
                }
            )
            for i in (0, 3)
        )
    )
    result = evaluate_video(predictions, gt)
    assert result["continuity"]["tracks"]["car-a"]["longest_uncovered_run_frames"] == 2
    labels[1].pop("track_id")
    gt.write_text(json.dumps({"evaluated_frames": list(range(4)), "annotations": labels}))
    with pytest.raises(ValueError, match="All annotations"):
        evaluate_video(predictions, gt)
