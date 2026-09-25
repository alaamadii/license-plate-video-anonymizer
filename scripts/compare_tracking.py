"""Replay identical recorded detections to isolate tracking changes on original video.

This does not rerun YOLO or measure detector improvements/inference throughput.
Use matching ORIGINAL footage, never the already-redacted result.
"""

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from plate_anonymizer.evaluation.video import evaluate_video
from plate_anonymizer.models import BoundingBox, Detection
from plate_anonymizer.pipeline import anonymize_video
from plate_anonymizer.redaction.core import RedactionMode
from plate_anonymizer.video.ffmpeg import mux_original_audio
from plate_anonymizer.video.io import VideoReader, VideoWriter


class RecordedDetector:
    def __init__(self, path: Path) -> None:
        self.frames = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["source"] != "detector":
                continue
            self.frames[item["frame_index"]].append(
                Detection(
                    BoundingBox(**item["bbox"]),
                    item["confidence"],
                    item["frame_index"],
                    item["timestamp_ms"],
                )
            )

    def detect(self, frame, frame_index, timestamp_ms):
        return self.frames.get(frame_index, [])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--motion-gap-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if not args.input.is_file() or not args.detections.is_file():
        parser.error("Original video and recorded detection metadata must exist.")
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve previous experiments.")
    detector = RecordedDetector(args.detections)
    with VideoReader(args.input) as reader:
        info = reader.info
        if detector.frames and max(detector.frames) >= info.frame_count:
            parser.error("Detection frame indices exceed input video length.")
    args.output_dir.mkdir(parents=True)
    report = {
        "experiment": "tracking_only_replay",
        "accuracy_measured": args.ground_truth is not None,
        "input_sha256": sha256(args.input),
        "detections_sha256": sha256(args.detections),
        "settings": {
            "iou_max_gap_frames": 3,
            "motion_max_gap_seconds": args.motion_gap_seconds,
            "box_padding": 0.15,
            "redaction": "solid",
        },
        "variants": {},
    }
    for mode in ("iou", "motion"):
        print(f"Rendering {mode} from recorded detector observations...", flush=True)
        video = args.output_dir / f"{mode}.mp4"
        metadata = video.with_suffix(".jsonl")
        count = anonymize_video(
            args.input,
            video,
            detector,
            metadata,
            redaction_mode=RedactionMode.SOLID,
            tracker_mode=mode,
            motion_max_gap_seconds=args.motion_gap_seconds,
        )
        rows = [json.loads(line) for line in metadata.read_text().splitlines()]
        summary = {"frames": count, "sources": dict(Counter(row["source"] for row in rows))}
        if args.ground_truth is not None:
            summary["evaluation"] = evaluate_video(
                metadata,
                args.ground_truth,
                metric="coverage",
                threshold=0.95,
                box_field="redaction_bbox",
            )
        report["variants"][mode] = summary
    # Keep each panel at source resolution. Text is outside the original image area.
    from plate_anonymizer.video.io import VideoInfo

    panel_info = VideoInfo(info.width * 2, info.height + 40, info.fps, info.frame_count)
    with tempfile.TemporaryDirectory(prefix="tracking-comparison-") as temporary:
        silent = Path(temporary) / "comparison.mp4"
        with VideoReader(args.output_dir / "iou.mp4") as before:
            with VideoReader(args.output_dir / "motion.mp4") as after:
                with VideoWriter(silent, panel_info) as writer:
                    for (_, _, left), (_, _, right) in zip(
                        before.frames(), after.frames(), strict=True
                    ):
                        title = np.zeros((40, info.width * 2, 3), dtype=np.uint8)
                        cv2.putText(
                            title,
                            "BEFORE: IoU",
                            (10, 27),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            title,
                            "AFTER: experimental motion",
                            (info.width + 10, 27),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        writer.write(np.vstack((title, np.hstack((left, right)))))
        mux_original_audio(silent, args.input, args.output_dir / "comparison.mp4")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comparison: {args.output_dir / 'comparison.mp4'}", flush=True)


if __name__ == "__main__":
    main()
