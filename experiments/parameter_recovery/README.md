# Parameter recovery

This experiment implements GitHub #98 for the Normal, StudentT, and Gamma
families. Each `config-*.yaml` is the complete scientific specification for one
family: seed, truth, architecture, fit budget, posterior draws, calibration
levels, recovery band, and artifact destination.

Run one full simulation with, for example:

```bash
uv run --extra experiments python experiments/parameter_recovery/run.py \
  experiments/parameter_recovery/config-normal.yaml
```

Add `--smoke` for the bounded CI path. Outputs land beneath `runs/`; inspected
paper artifacts are promoted to `results/canonical-<family>/`.

## Statistical convention

The additive feature contribution is identified only up to a constant. Both
the configured truth and every coherent posterior effect draw are therefore
mean-centered over the evaluation grid before recovery ribbons and empirical
coverage are computed. These are raw additive-predictor effects (before the
family link). The absorbed level is evaluated separately as the model intercept
plus the draw's uncentered mean effect and written to `intercept_coverage.csv`.

Coverage at nominal 50/80/90/95 is measured and reported, not asserted to be
correct. Mean-field VI can under-cover as documented by ADR-0001; no post-hoc
recalibration or band inflation is applied.
