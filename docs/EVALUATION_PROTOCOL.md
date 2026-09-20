# Evaluation protocol

## Acceptance definition

Agree the annotation and matching policy before labelling the locked holdout.
Targets: plate-instance recall >=99%, precision >=95%, with recall prioritized.
Report detector localization (IoU >=0.5) separately from mask coverage
(intersection(mask, visible plate) / visible plate area >=0.95). Coverage is a
geometric proxy: it does not prove that blurred characters are unreadable.
Use solid masking for the privacy acceptance run.

Report frame-instance recall and precision, per-condition recall, and missed-plate
evidence. Track-level protection (every visible frame of a plate protected) is an
additional production metric to implement; it is not yet produced by the CLI.
One long-lived parked car must not hide failures on briefly visible vehicles.

## Initial labelling budget

Start with a 10–15 minute discovery sample to estimate density and failure types.
As a planning budget, label 60–120 minutes for training/development and a separate
30–60 minutes for holdout, with contiguous clips sampled across camera sessions,
locations, traffic density, plate sizes, night/day, weather and motion.
These are starting allocations, not a statistical guarantee.

Aim initially for at least 1,000 distinct plate tracks across the dataset and
several hundred independent tracks in holdout, increasing collection if rare
conditions lack support. Annotate every plate-visible frame in selected clips
and explicitly review negative frames. Semi-automatic interpolation can speed
annotation, but humans must verify intermediate frames. Double-review ambiguous
cases and a random 10% audit, and adjudicate disagreements.

Split by recording/session/location before tuning. Never put adjacent frames from
the same track into different splits. Keep representative sampling separate from
a deliberately difficult stress set, and report both without mixing denominators.

## What would justify 99%?

For zero failures in n independent Bernoulli trials, the one-sided 95% exact
lower bound is 0.05^(1/n). With 299 successes it exceeds 0.99. This is an
illustration of sample requirements, not a justification for treating 299
adjacent video frames as independent.

Frames within tracks and sessions are correlated. Report raw TP/FP/FN and a
session- or track-clustered confidence interval, alongside condition-specific
counts. Collect more independent footage if the lower bound or rare-case support
is insufficient. A point estimate of 99% alone is not proof that population
recall is at least 99%. Clustered intervals and track aggregation are planned
analysis steps, not capabilities of the current CLI.

## Repeatable workflow

1. Version the reviewed-frame manifest and labels; preserve video IDs and hashes.
2. Tune weights, confidence, image size, tiles, padding and temporal settings on
   development only. Preserve each command, checkpoint hash and environment.
3. Freeze the configuration; run anonymization on each held-out clip.
4. Run evaluate-video once per clip for IoU using bbox, and for coverage using
   redaction_bbox. Sum TP/FP/FN across clips before calculating overall metrics;
   do not average clip recall percentages.
5. Inspect every missed-plate crop and false-positive frame. Never relabel an
   actual miss away to improve the score. Corrections need an audit trail.
6. After changes, rerun the same versioned development set. If holdout failures
   informed those changes, that holdout has become development data; obtain a
   fresh locked acceptance set.

## CLI semantics

The manifest explicitly lists evaluated_frames, including reviewed empty frames.
Predictions on other frames are ignored and counted as ignored_prediction_rows.
No cross-frame matching is allowed. Matches use greedy descending-score one-to-one
assignment; duplicates count as false positives. Results include matching rules,
threshold, box field, input SHA256 hashes, frame counts, per-frame counts, tag
recall and false negatives. Tag recall uses the same global frame assignment;
tags overlap and must not be summed. Precision with no predictions and recall
with no ground truth are reported as zero; inspect counts before interpretation.

One evaluation covers one video, with zero-based frame indices. Do not concatenate
multiple videos with overlapping indices. Historical JSONL without redaction_bbox
can be used for detector IoU; rerun anonymization to measure actual mask coverage.

Failure crops contain original plate pixels and stay in ignored local result
directories. The included synthetic example exists only to test this workflow.
