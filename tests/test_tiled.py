from plate_anonymizer.detection.tiled import non_max_suppression, tile_origins
from plate_anonymizer.models import BoundingBox, Detection


def test_tile_origins_cover_trailing_edge() -> None:
    origins = tile_origins(3840, 1280, 0.2)
    assert origins[0] == 0
    assert origins[-1] == 2560


def test_nms_keeps_higher_confidence_duplicate() -> None:
    low = Detection(BoundingBox(0, 0, 10, 10), 0.5, 0, 0)
    high = Detection(BoundingBox(0, 0, 10, 10), 0.9, 0, 0)
    assert non_max_suppression([low, high]) == [high]
