# Validation records

This document separates execution checks, controlled software tests and visual
observations. Street footage is not bundled with the repository. These runs do
not constitute a labelled accuracy benchmark.

## Street clip A

| Measurement | Result |
| --- | --- |
| Input | 1280 x 720, 25 FPS, 750 frames, 30 seconds |
| Settings | CPU; full-frame; image size 1280; confidence 0.15; padding 0.15 |
| Tracking and masks | IoU, three-frame interpolation gap, solid masks |
| Processing time | 371.81 seconds; approximately 2.02 FPS |
| Observations | 1,538 direct detections and 7 interpolated boxes |
| Output | All 750 frames decoded; H.264 video and AAC stereo audio |

Timing excludes model loading and the separate final encoding/audio pass.
The final render reused detector observations after correcting the tracking gap
boundary. Six sampled frames were reviewed; a shrub near frame 100 was
incorrectly masked. Detection counts are repeated observations, not unique plates.

## Street clip B

| Measurement | Result |
| --- | --- |
| Input | 848 x 478, 30 FPS, 1,800 frames, 60 seconds |
| Settings | CPU; full-frame; image size 1280; confidence 0.15; padding 0.15 |
| Tracking and masks | IoU, three-frame interpolation gap, solid masks |
| Processing time | 757.97 seconds; approximately 2.37 FPS |
| Observations | 4,871 direct detections and 357 interpolated boxes |
| Output | All 1,800 frames decoded; H.264 video and AAC stereo audio |

This timing includes the final audio/encoding pass and excludes model loading.
It is not directly comparable to clip A's timing scope. Visual review showed
missed plates, interruptions in coverage and a large false-positive mask around
50 seconds. Detection rows occurred on 1,748 frames, but this is not a recall
measurement: neither the number of visible plates nor correctness was labelled.

Both runs used the checkpoint described in [model provenance](../models/README.md).
CPU contention was uncontrolled and the CPU model was not recorded with the runs.
No GPU inference was used.

## Motion-tracking regression

A controlled generated sequence introduces a six-frame detection dropout while a
plate moves. Replaying identical observations leaves six uncovered frames with
the IoU baseline and zero with the experimental motion tracker. Separate tests
exercise expiration, one-off false detections, size changes, scene changes and
frame exits. These tests verify defined behavior, not real-world generalization.

## Single-frame detection probe

On a saved original-frame JPEG from clip B, full-frame inference returned two boxes.
Hybrid inference returned four, including the previously missed central white
BMW plate. Both used confidence 0.15 and image size 1280; hybrid added 512-pixel
tiles with 25% overlap. The first call included warm-up, so its timing cannot be
compared fairly with the second call. This observation does not establish
video-wide recall, precision or improvement in coverage continuity.

## Repository checks

The current 40-test suite passes in an isolated Windows/Python 3.13 environment.
Ruff and the synthetic CLI evaluation example also pass. The tracker comparison
script was exercised on a five-frame source extract through H.264 encoding.
The environment snapshot is documented in [setup](setup.md).

Hosted CI status is available in GitHub Actions. Docker execution, 4K throughput,
long-run stability, night performance, GPU cost and independent holdout accuracy
remain unvalidated. The original 28-test clean-install check is historical; the
expanded suite includes the motion-tracking tests.
