"""End-to-end detection, tracking, temporal recovery, and redaction pipeline."""

from pathlib import Path

from plate_anonymizer.detection.base import BaseDetector
from plate_anonymizer.metadata.writer import append_detections
from plate_anonymizer.redaction.core import RedactionMode, redact
from plate_anonymizer.temporal.buffer import BufferedFrame, TemporalBuffer
from plate_anonymizer.tracking.simple import IoUTracker
from plate_anonymizer.utils.boxes import expand_box
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
) -> int:
    """Run the privacy pipeline.

    A bounded frame buffer allows detections with stable track IDs to fill short
    detector gaps before those frames are irreversibly written.
    """
    if metadata_path.exists():
        metadata_path.unlink()
    tracker = IoUTracker(max_age=max(temporal_max_gap, 1)) if tracking else None
    temporal = TemporalBuffer(temporal_max_gap if tracking else 0)
    processed = 0

    with VideoReader(input_path) as reader:
        with VideoWriter(output_path, reader.info) as writer:
            for frame_index, timestamp_ms, frame in reader.frames():
                detections = detector.detect(frame, frame_index, timestamp_ms)
                if tracker is not None:
                    detections = tracker.update(detections, frame_index)

                ready = temporal.push(
                    BufferedFrame(frame_index, timestamp_ms, frame, detections)
                )
                for buffered in ready:
                    append_detections(metadata_path, buffered.detections)
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
                append_detections(metadata_path, buffered.detections)
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
    return processed
