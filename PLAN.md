# Study plan

Written on 20 November 2026, after the pipeline ran end to end on one scene and before
the experiment matrix was run. It was not lodged with an outside registry, so it is a
statement of intent that a reader has to take on trust. Anything that changed after the
results came in is listed under "Amendments" at the bottom.

## Question

Novel view synthesis is normally scored on test cameras scattered among the training
cameras. A deployed system is asked for views from wherever the user goes. How much does
the usual test overstate what a radiance field delivers from a direction it was not
trained near, and does the image score tell us whether the geometry is still right?

## What is held fixed

- Three procedural scenes (`blocks`, `orbs`, `clutter`), 120 cameras each on a band of the
  upper hemisphere, 128 x 128 pixels, exact depth for every pixel.
- One model and one training recipe for every run (`FieldConfig` defaults in
  `src/viewgap/field.py`). No setting is tuned per design, per scene or per run.
- 24 test cameras and 96 training cameras in both designs.

## Designs compared

- `random`: 24 test cameras chosen at random.
- `cap`: the 24 cameras nearest a randomly chosen direction, so the test set is one
  contiguous patch of the hemisphere.

## Experiments

| | Runs | What varies |
|---|---|---|
| E1 | 18 | 3 scenes x 2 designs x 3 seeds, 96 training views, true poses |
| E2 | 18 | 3 scenes x 2 designs x {12, 24, 48} training views, seed 0, true poses |
| E3 | 6 | 3 scenes x 2 designs, seed 0, 96 views, poses recovered by structure from motion |

## Metrics

PSNR (primary), SSIM, mean absolute depth error on surface pixels, silhouette IoU. The
unit of analysis is the run. Differences between designs are paired within (scene, seed)
and reported with a bootstrap 95% interval over the nine pairs.

## Hypotheses

- **H1.** At 96 training views, `cap` scores lower PSNR than `random` by at least 1 dB,
  and the interval for the difference excludes zero.
- **H2.** Inside the `cap` runs, per-view PSNR falls as the angle to the nearest training
  camera grows (Spearman correlation negative in at least eight of nine runs).
- **H3.** Geometry suffers proportionally more than appearance: the ratio of depth error
  (cap over random) is larger than the ratio of image mean squared error.
- **H4.** Fewer training views widen the gap: the PSNR difference at 12 training views is
  larger than at 96.
- **H5.** Poses from structure from motion cost less than 1 dB of PSNR against true poses.

Each of these can fail, and a failed one will be reported as failed.

## Known limits, stated in advance

- Three synthetic scenes of one kind. The size of any effect will not transfer to real
  captures; the direction might.
- One small model. A stronger model could close or widen the gap.
- Nine pairs is a small sample, and three scenes is a smaller one. Per-scene numbers will
  be shown beside every pooled number.

## Amendments

None yet.
