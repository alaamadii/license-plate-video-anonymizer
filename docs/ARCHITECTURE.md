# Architecture and limits

## Implemented path

OpenCV decodes BGR frames. A plate-specific Ultralytics adapter predicts boxes;
optional overlapping tiles translate local coordinates to the source frame and
merge candidates with NMS. Full-frame inference uses a resized model input
(image-size 1280 by default), not an unscaled 4K tensor. Hybrid mode adds full-frame
predictions to tiled detections.

The deterministic IoU tracker assigns IDs. A bounded frame buffer interpolates
short gaps when the same track is detected again. Track history expires, so it
does not accumulate indefinitely in long recordings. This is short-gap recovery,
not optical flow or a trained motion model. No leading/trailing propagation or
scene-cut detector is implemented.

Expanded rectangles are rendered directly into pixels. JSONL records the original
box and integer clipped redaction_bbox used by the renderer. It also records
frame index, nominal timestamp, confidence, class, track ID and source
(detector/interpolation). No row is written for a frame with no detections.

The OpenCV writer produces a constant-FPS intermediate. With preserve-audio,
FFmpeg encodes MP4 as H.264/yuv420p and converts source audio to AAC. This requires
additional temporary disk space. Audio is not shortened to force a shorter video.

## Memory and scaling

Decoded frame storage is bounded by the temporal gap, not video length.
A 3840 x 2160 BGR frame is about 24.9 MB; a few buffered 4K frames are already
substantial before detector tensors, tiles and encoder buffers. One frame is
processed at a time; GPU batching, hardware encode/decode and restart checkpoints
are future work. Redaction currently copies frames per box and metadata opens a
file per emitted frame; profile these on crowded 4K footage before scaling.

## Known limits

- IoU association can fail for fast motion, occlusion or camera cuts.
- Interpolation cannot find never-detected plates and may cross an unrecognized cut.
- Nominal FPS timestamps do not preserve variable frame-rate timing.
- OpenCV read failure ends iteration; robust corrupted-input auditing is pending.
- HDR, colour metadata, rotation metadata and odd dimensions need dedicated tests.
- Long videos, 4K, H.265/MOV decoding and CUDA throughput are unvalidated locally.
- Solid masks obscure selected pixels; blur/pixelation require readability checks.
- Evaluation uses rectangular coverage and greedy assignment; it is not an OCR test.

See PROJECT_PLAN.md for the proposed production upgrades and acceptance gates.
