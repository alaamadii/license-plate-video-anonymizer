# Annotation guide

Label every visible region identifiable as a vehicle license plate, whether or not
its characters are readable. Use a tight box around the visible extent. For an
occluded plate, label the visible plate region and tag partial/occluded; do not
invent the geometry of a completely hidden plate.

Use zero-based decoded frame indices and source-image pixel coordinates
[x1, y1, x2, y2]. Boxes must have positive area. Tag applicable conditions:
normal, small, very_small, motion_blur, partial, occluded, angled, night,
unreadable, edge_of_frame. Define size thresholds relative to the agreed source
resolution before annotation; record them in the dataset's own README.

Ambiguous objects go to adjudication, not automatic exclusion. A plate appearing
for one frame still counts. Review labels frame by frame after interpolation.

## Manifest for one video

```json
{
  "video_id": "session_01_clip_03",
  "evaluated_frames": [0, 1, 2],
  "annotations": [
    {"frame_index": 0, "bbox": [100, 200, 160, 222], "tags": ["small"]},
    {"frame_index": 1, "bbox": [103, 201, 164, 224], "tags": ["small", "partial"]}
  ]
}
```

Frame 2 was reviewed and contains no plate; a prediction there is a false positive.
Frames omitted from evaluated_frames are not scored. Every annotated frame must
be in evaluated_frames. Add source hashes, recording IDs and annotation policy
version as extra manifest fields for dataset management.

For future track metrics, retain stable vehicle/plate track IDs in the labelling
tool. The current evaluator scores frame instances, not track trajectories.
