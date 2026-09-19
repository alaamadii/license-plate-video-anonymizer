# Annotation Guide

Annotate every visible object that can reasonably be identified as a vehicle license plate and therefore requires anonymization. OCR readability is not required.

Use a tight bounding box around the visible plate extent. Add applicable tags: `normal`, `small`, `very_small`, `motion_blur`, `partial`, `occluded`, `angled`, `night`, `unreadable`, and `edge_of_frame`.

Do not silently discard ambiguous cases. Mark them for review so the evaluation policy can be applied consistently. A final production evaluation protocol should be agreed before labeling the locked holdout set.
