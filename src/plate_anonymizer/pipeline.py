"""End-to-end detection, tracking, temporal recovery, and redaction pipeline."""

from collections.abc import Callable
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
    progress: Callable[[int, int, int], None] | None = None,
) -> int:
    """Run the privacy pipeline.

    A bounded frame buffer allows detections with stable track IDs to fill short
    detector gaps before those frames are irreversibly written.
    """
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
        with VideoWriter(output_path, reader.info) as writer:
            for frame_index, timestamp_ms, frame in reader.frames():
                detections = detector.detect(frame, frame_index, timestamp_ms)
                detected += len(detections)
                if progress is not None:
                    progress(frame_index + 1, reader.info.frame_count, detected)
                if tracker is not None:
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
