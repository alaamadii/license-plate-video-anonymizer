"""End-to-end frame processing pipeline."""

from pathlib import Path

from plate_anonymizer.detection.base import BaseDetector
from plate_anonymizer.metadata.writer import append_detections
from plate_anonymizer.redaction.core import RedactionMode, redact
from plate_anonymizer.utils.boxes import expand_box
from plate_anonymizer.video.io import VideoReader, VideoWriter


def anonymize_video(
    input_path: Path,
    output_path: Path,
    detector: BaseDetector,
    metadata_path: Path,
    box_padding: float = 0.15,
    redaction_mode: RedactionMode = RedactionMode.BLUR,
) -> int:
    if metadata_path.exists():
        metadata_path.unlink()
    processed = 0
    with VideoReader(input_path) as reader:
        with VideoWriter(output_path, reader.info) as writer:
            for frame_index, timestamp_ms, frame in reader.frames():
                detections = detector.detect(frame, frame_index, timestamp_ms)
                rendered = frame
                for detection in detections:
                    box = expand_box(
                        detection.bbox, box_padding, reader.info.width, reader.info.height
                    )
                    rendered = redact(rendered, box, redaction_mode)
                append_detections(metadata_path, detections)
                writer.write(rendered)
                processed += 1
    return processed
