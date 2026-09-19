"""Short-gap linear interpolation for tracked plate boxes."""

from dataclasses import replace

from plate_anonymizer.models import BoundingBox, Detection, DetectionSource


def interpolate_gap(
    before: Detection, after: Detection, max_gap: int = 3
) -> list[Detection]:
    if before.track_id is None or before.track_id != after.track_id:
        return []
    gap = after.frame_index - before.frame_index - 1
    if gap <= 0 or gap > max_gap:
        return []
    output: list[Detection] = []
    for step in range(1, gap + 1):
        alpha = step / (gap + 1)
        a, b = before.bbox, after.bbox
        box = BoundingBox(
            a.x1 + (b.x1 - a.x1) * alpha,
            a.y1 + (b.y1 - a.y1) * alpha,
            a.x2 + (b.x2 - a.x2) * alpha,
            a.y2 + (b.y2 - a.y2) * alpha,
        )
        output.append(
            replace(
                before,
                bbox=box,
                confidence=min(before.confidence, after.confidence),
                frame_index=before.frame_index + step,
                timestamp_ms=before.timestamp_ms
                + (after.timestamp_ms - before.timestamp_ms) * alpha,
                source=DetectionSource.INTERPOLATION,
            )
        )
    return output
