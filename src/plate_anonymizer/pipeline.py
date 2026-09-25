"""End-to-end detection, tracking, temporal recovery, and redaction pipeline."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from plate_anonymizer.detection.base import BaseDetector
from plate_anonymizer.metadata.writer import append_detections
from plate_anonymizer.redaction.core import RedactionMode, redact
from plate_anonymizer.temporal.buffer import BufferedFrame, TemporalBuffer
from plate_anonymizer.tracking.motion import MotionTracker
from plate_anonymizer.tracking.simple import IoUTracker
from plate_anonymizer.utils.boxes import clip_box, expand_box
from plate_anonymizer.video.io import VideoReader, VideoWriter


def _render(
    buffered: BufferedFrame,
    width: int,
    height: int,
    padding: float,
    mode: RedactionMode,
):
    rendered = buffered.image
    for detection in buffered.detections:
        rendered = redact(
            rendered,
            expand_box(detection.bbox, padding, width, height),
            mode,
        )
    return rendered


def anonymize_video(
    input_path: Path,
    output_path: Path,
    detector: BaseDetector,
    metadata_path: Path,
    box_padding: float = 0.15,
    redaction_mode: RedactionMode = RedactionMode.BLUR,
    tracking: bool = True,
    temporal_max_gap: int = 3,
    progress: Callable[[int, int, int], None] | None = None,
    tracker_mode: str = "iou",
    motion_max_gap_seconds: float = 0.2,
) -> int:
    """Run the privacy pipeline.

    A bounded frame buffer allows detections with stable track IDs to fill short
    detector gaps before those frames are irreversibly written.
    """
    if tracker_mode not in {"iou", "motion"} or not 0 <= motion_max_gap_seconds <= 1:
        raise ValueError("Use iou/motion tracking and a motion gap between 0 and 1 seconds.")
    paths = [path.resolve() for path in (input_path, output_path, metadata_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, output and metadata must be different files.")
    if metadata_path.exists():
        metadata_path.unlink()
    tracker = IoUTracker(max_age=temporal_max_gap + 1) if tracking else None
    temporal = TemporalBuffer(temporal_max_gap if tracking else 0)
    processed = 0
    detected = 0

    with VideoReader(input_path) as reader:
        motion = (
            MotionTracker(max_gap=int(reader.info.fps * motion_max_gap_seconds))
            if tracking and tracker_mode == "motion"
            else None
        )
        previous_gray = None
        with VideoWriter(output_path, reader.info) as writer:
            for frame_index, timestamp_ms, frame in reader.frames():
                detections = detector.detect(frame, frame_index, timestamp_ms)
                detected += len(detections)
                if progress is not None:
                    progress(frame_index + 1, reader.info.frame_count, detected)
                if motion is not None:
                    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
                    if previous_gray is not None:
                        change = np.abs(gray.astype(float) - previous_gray).mean()
                        if change > 45:
                            motion.reset()
                    previous_gray = gray
                    detections = motion.update(detections, frame_index, timestamp_ms)
                    detections = [
                        detection
                        for detection in detections
                        if clip_box(detection.bbox, reader.info.width, reader.info.height).area > 0
                    ]
                elif tracker is not None:
                    detections = tracker.update(detections, frame_index)

                ready = temporal.push(BufferedFrame(frame_index, timestamp_ms, frame, detections))
                for buffered in ready:
                    append_detections(
                        metadata_path,
                        buffered.detections,
                        (
                            expand_box(d.bbox, box_padding, reader.info.width, reader.info.height)
                            for d in buffered.detections
                        ),
                    )
                    writer.write(
                        _render(
                            buffered,
                            reader.info.width,
                            reader.info.height,
                            box_padding,
                            redaction_mode,
                        )
                    )
                    processed += 1

            for buffered in temporal.flush():
                append_detections(
                    metadata_path,
                    buffered.detections,
                    (
                        expand_box(d.bbox, box_padding, reader.info.width, reader.info.height)
                        for d in buffered.detections
                    ),
                )
                writer.write(
                    _render(
                        buffered,
                        reader.info.width,
                        reader.info.height,
                        box_padding,
                        redaction_mode,
                    )
                )
                processed += 1
    if processed == 0:
        raise RuntimeError("No video frames could be decoded from the input.")
    return processed
