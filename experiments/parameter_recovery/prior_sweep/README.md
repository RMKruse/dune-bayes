# Normal Prior/Smoothness Sweep

This folder contains the pre-registered first calibration-improvement sweep from
`CONTEXT.md`. It probes whether the ADR-0002 `PriorScale` machinery improves
Normal parameter-recovery effect-band calibration before broader architecture,
training, or posterior-family changes are considered.

Run a smoke check for one candidate:

```bash
uv run --extra experiments python experiments/parameter_recovery/run.py \
  experiments/parameter_recovery/prior_sweep/normal-fixed-0p3.yaml --smoke
```

Run a full candidate by omitting `--smoke`. Outputs go to
`experiments/parameter_recovery/runs/`, which is ignored until a run is manually
reviewed and promoted.

The first acceptance criterion is lower mean absolute coverage error across
Normal location and scale at nominal 50/80/90/95, while centered recovery remains
visually plausible, intercept coverage is not pathological, final loss does not
explode, ribbons do not become obviously over-wide everywhere, and
`metrics/prior_scale.json` shows an interpretable learned or sampled scale.
