# Evaluation Protocol

The acceptance target is **recall >= 99%** and **precision >= 95%**, with recall prioritized. These are targets, not measured claims.

A defensible evaluation should use representative footage and a locked holdout set. Every visually identifiable plate that should be redacted should be annotated, including unreadable, partial, small, blurred, angled, occluded, night-time, and edge-of-frame examples.

Development thresholds must not be tuned on the final holdout. Evaluation will report conventional IoU matching as well as a privacy-oriented ground-truth coverage ratio, because a conservative redaction box can fully protect a plate despite imperfect IoU.

False negatives must be saved for direct inspection. Metrics should also be broken down by difficult-case tags. Dataset sufficiency should be reasoned about using the number and diversity of plate-visible instances/frames, not video minutes alone.
