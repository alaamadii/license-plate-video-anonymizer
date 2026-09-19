# Production Validation Checklist

Before presenting this system as meeting a privacy SLA:

- Select license-plate-specific detector weights and record their source/license.
- Build a representative development set and a separate locked holdout.
- Annotate every visible plate according to the annotation guide.
- Include small, distant, blurred, partial, angled, occluded, night, unreadable, and edge cases.
- Tune confidence, image size, tiling, tracking, temporal gap, and padding only on development data.
- Run the locked holdout once the configuration is frozen.
- Report precision, recall, F1, IoU matching, and privacy coverage.
- Inspect every false negative and save evidence for review.
- Measure CPU/GPU throughput on named hardware.
- Calculate cost/video-hour using an explicitly supplied compute price.
- Verify output frame count, resolution, FPS, duration, and audio behavior.
- Verify the final media is permanently redacted rather than visually overlaid.
- Review model and dependency licensing for the deployment context.

A small demo video is useful for engineering validation but cannot establish a 99% recall claim.
