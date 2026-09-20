# Measured local demonstration

A privately supplied street clip was processed on 2026-09-20. The video and original
plate pixels are not bundled with this repository. This report is evidence of an
operating pipeline, not a labelled accuracy benchmark.

| Measurement | Result |
| --- | --- |
| Input | 1280 x 720, 25 FPS, 750 frames; 30 seconds of video |
| Inference | CPU, plate checkpoint documented in models/README.md |
| Settings | Full-frame, image size 1280, confidence 0.15, padding 0.15 |
| Redaction | Solid; tracking enabled; maximum interpolation gap 3 |
| Detection/redaction pass | 371.81 seconds; 750 / 371.81 = 2.017 FPS |
| Final boxes | 1,538 direct observations + 7 interpolated observations |
| Decode verification | 750 output frames; complete FFmpeg decode succeeded |
| Final streams | H.264 video, AAC stereo; container duration about 30.07 seconds |

Timing excludes model loading and the separate final encoding/audio pass. CPU
contention was not controlled; no GPU was used. Do not extrapolate this result to
4K or GPU pricing. The exact CPU model was not recorded in the original timed run.
The software snapshot is documented under constraints/.

The final render reused recorded detector observations after the tracking gap
boundary was corrected, then encoded H.264/AAC. Six sampled frames were inspected,
not all plate instances manually labelled. A shrub near frame 100 was incorrectly
masked. Every frame had at least one detection; this does not prove all plates
were found. Counts are repeated observations, not unique vehicles.

## What is not measured

Recall, precision, track-level privacy failures, 4K throughput, long-run stability,
night performance, and GPU cost remain unmeasured. The next milestone is a
representative labelled development/holdout dataset using EVALUATION_PROTOCOL.md.

The local output/ folder contains the video and validation artifacts for review.
It is ignored by Git. A public demo should use footage cleared for redistribution.

## Repository verification

After the evaluation workflow was added, 28 tests passed in both the development
environment and a clean Windows/Python 3.13 virtual environment installed with
the dev extra and recorded constraints. Ruff, pip check and the README synthetic
evaluation example passed in the clean environment. The Windows launcher was
also rerun on a five-frame extract with real inference and H.264/audio handling.
Hosted Linux/Windows CI and Docker execution remain unverified until run there.
