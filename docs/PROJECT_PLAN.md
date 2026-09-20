# Production plan for street-video anonymization

## First architecture to benchmark

Start with a plate-specific detector on overlapping high-resolution tiles plus a
full-frame pass. Compare full vs tiled vs hybrid modes on the same development
clips, especially tiny and briefly visible plates. Keep the current IoU tracker
as an ablation baseline; evaluate a stronger motion-aware tracker, then optical
flow or neighboring-frame recovery for short misses. Reset temporal state at
scene cuts and cap propagation to avoid masking unrelated objects.

Fine-tune the detector on customer-domain hard cases and false positives after
establishing the baseline. Freeze a separate holdout before active learning.
Conservative spatial and temporal padding should be tuned against mask coverage
and over-redaction, not chosen only for visual appearance.

## Proposed milestones

| Stage | Reviewable output | Acceptance gate |
| --- | --- | --- |
| Discovery | Codec/FPS audit, representative sample, annotation policy | Agree privacy unit and matching rules |
| Baseline | Labels, baseline reports and missed-plate evidence | Reproduce metrics from a clean environment |
| Improvements | Fine-tuning and tracker/tiling ablations | Better development recall within precision budget |
| Acceptance | Locked holdout reports and human failure review | Agreed recall/precision and confidence support |
| Scale/handover | GPU timings, media checks, repeatable commands | Agreed cost/video-hour and long-run reliability |

This is a proposed sequence, not work already completed.

## GPU and cost plan

My first GPU benchmark candidate would be a single NVIDIA L4 with 24 GB VRAM,
for example a G2 instance. This is a proposed starting point for measurement,
not a claim that it is the cheapest option. Hardware reference:
[Google Cloud GPU specifications](https://docs.cloud.google.com/compute/docs/gpus).

Measure complete decode, inference, tracking, redaction, encode and audio time on
representative 4K/30 FPS clips after a separate warm-up. Record GPU/CPU model,
driver, PyTorch/CUDA versions, checkpoint hash, batch size, input size, tile layout,
peak RAM/VRAM and actual VM hourly price. Repeat at least three runs and report
median and range. Benchmark a long recording to check drift, memory and failures.

No GPU speed has been measured here. For budget sensitivity only, if end-to-end
throughput were 15, 30 or 60 FPS for 30 FPS video, one compute-hour would process
0.5, 1 or 2 video-hours. At a hypothetical all-in compute rate of USD 1/hour,
that would mean USD 2, 1 or 0.50 per video-hour, excluding storage/egress.
These are arithmetic scenarios, not expected benchmark results or provider quotes.

Use measured values with the calculator:

```bash
plate-anonymizer benchmark --frames 108000 --source-fps 30 --processing-seconds 3600 --gpu-hour-cost 1.00
```

The numbers above are hypothetical. GPU pricing must include the VM's CPU/RAM,
and speed estimates should follow the initial paid pilot, not precede evidence.
