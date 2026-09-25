# Experimental motion tracking

The default IoU baseline is retained. Enable the experimental variant explicitly:

```bash
plate-anonymizer anonymize --input input/original.mp4 --output output/motion.mp4 --model models/plate_detector.pt --redaction solid --preserve-audio --tracker-mode motion --motion-max-gap-seconds 0.2
```

## Behavior and limits

Association uses predicted center position, distance and size-change gates.
Center velocity comes from real observations and is smoothed on updates.
Propagation requires two consecutive observations and stops after 0.2 seconds
by default. Predictions never renew observation age or confirm themselves.
Generated boxes are marked as temporal propagation; their decayed confidence
is a score, not a calibrated probability. Propagation holds box size constant.
A coarse downsampled luminance-change trigger clears motion history at large cuts.

This is constant-velocity tracking, not ByteTrack, optical flow, vehicle appearance
tracking or a newly trained detector. It can drift during acceleration, crossings
or occlusion. Repeated false detections can be propagated. Brief ghost masks can
remain after a car exits, until the age limit. The cut heuristic can miss subtle
cuts or trigger on lighting changes. Direct false-positive boxes are not removed.

## Reproduce a comparison

Use ORIGINAL footage and its matching cached JSONL. Only direct detector rows are
replayed; old interpolation is discarded. Use a new output directory:

```bash
python scripts/compare_tracking.py --input input/original.mp4 --detections output/baseline.jsonl --output-dir output/tracking-comparison --ground-truth data/labels.json
```

Outputs: two silent rendered videos, both metadata files, H.264/AAC side-by-side
comparison with source audio, input checksums and an evaluation report. Without
ground truth, only observation counts are reported. Replay times are not detector
inference timings. Frame counts alone cannot prove a cache belongs to a video;
retain the original/cache hashes and use matching recordings. Never substitute
an already-redacted video for the original.

## Continuity evaluation

Assign a nonempty string track ID to every ground-truth annotation, stable for each
physical plate. The existing evaluate-video command then reports uncovered frames,
longest consecutive uncovered run, coverage-loss transitions, and protection over
all reviewed frames per track. IDs come from labels, not predicted identities.
Unreviewed gaps break runs; sparse labels cannot establish full-track protection.
For mask continuity, use `--metric coverage --threshold 0.95 --box-field redaction_bbox`.

## Verification

A controlled moving-plate regression contains a six-frame detection dropout.
Identical observations leave six uncovered frames with IoU and zero with motion.
Tests also exercise expiration, single false detections, size jumps, scene changes,
association and real encoding. These are software tests, not street-video accuracy.

Real-world tracking gains have not been measured on a labelled holdout.
Detector ablations and independent validation are part of the project roadmap.

## Single-frame detector probe

On the saved original-frame JPEG from the second clip, full-frame inference
returned two boxes and hybrid inference returned four, including the previously
missed central white BMW plate. Both used the same checkpoint, confidence 0.15
and image size 1280; hybrid used 512-pixel tiles with 25% overlap plus full-frame
inference. This is a visual observation on one JPEG, not labelled recall/precision
or a video-wide improvement. Timing is not comparable because the first prediction
included model warm-up. Local JSON and preview images remain in ignored output/.
