# Roadmap and performance planning

The project develops detection quality, temporal coverage and repeatable evaluation
together. Changes are compared against a fixed baseline before defaults change.

## Current capabilities

Plate detection supports full-frame and overlapping-tile inference. The default
IoU tracker includes buffered interpolation. An optional constant-velocity tracker
adds bounded propagation and a coarse scene-change safeguard. The evaluation CLI
measures detection/mask matching and continuity over manually labelled tracks.

## Development priorities

| Area | Next work | Evidence required |
| --- | --- | --- |
| Dataset | Diverse recordings and reviewed plate tracks | Versioned development and independent holdout splits |
| Detection | Compare tile settings and fine-tune on hard cases | Recall and precision by condition |
| Temporal coverage | Compare motion models and appearance cues | Fewer uncovered frames without excessive ghost masks |
| Media reliability | Audit decode failures, timing and metadata | Codec, duration and long-run checks |
| Throughput | Profile inference, frame copies and encoding | Repeatable measurements on named hardware |
| Operations | Resume/retry and experiment manifests | Recoverable runs with traceable outputs |

Fine-tuning should use domain-representative plates and hard negative examples.
Separate recordings before tuning; adjacent frames of one vehicle must not leak
between development and holdout. Spatial and temporal padding should be selected
using mask coverage and over-redaction measurements.

## Performance measurement

Record complete decode, inference, tracking, redaction, encode and audio time.
Measure model warm-up separately. Record CPU/GPU, driver, library versions,
checkpoint hash, tile layout, peak memory and source codec/resolution/FPS.
Repeat runs and report median and range; test long recordings for memory growth.

GPU throughput has not been measured. Choose hardware through benchmark results
rather than extrapolating the CPU demonstrations. For cost planning:

```text
video hours per compute hour = video duration / processing duration
compute cost per video hour = compute hourly price / video hours per compute hour
```

For an explicitly hypothetical example, processing 30 FPS video at 15, 30 or
60 FPS gives 0.5, 1 or 2 video-hours per compute-hour. At an assumed USD 1/hour,
compute alone costs USD 2, 1 or 0.50 per video-hour. These are arithmetic scenarios,
not measured speeds or provider quotes; storage and transfer charges are separate.

```bash
plate-anonymizer benchmark --frames 108000 --source-fps 30 --processing-seconds 3600 --gpu-hour-cost 1.00
```

Replace the example values with a measured run and the actual all-in compute rate.
