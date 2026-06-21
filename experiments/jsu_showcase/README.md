# Johnson's SU heteroscedastic-skew showcase

This experiment implements GitHub #100. One Bayesian shape function jointly
models all four Johnson's SU raw additive predictors: skew (γ), tailweight (δ),
location (ξ), and scale (λ). Both skew and linked scale vary with the covariate,
so the four-panel figure demonstrates epistemic credible ribbons beyond the
mean function supplied by earlier Bayesian NAM work.

Regenerate the canonical scratch run with:

```bash
uv run --extra experiments python experiments/jsu_showcase/run.py \
  experiments/jsu_showcase/config.yaml
```

Use `--smoke` for the bounded CI path. Scratch output is written beneath
`runs/`; inspected paper evidence is promoted to `results/canonical/`.

## Statistical convention

Shape functions are identified only up to a constant. The configured truth and
every coherent posterior effect draw are therefore centered over the evaluation
grid before plotting or coverage measurement. The absorbed level is assessed
separately in `intercept_coverage.csv`. Ribbons and coverage operate on raw
additive predictors; `recovery.npz` also records the linked truth so the known
heteroscedastic scale construction is directly auditable.

Coverage at nominal 50/80/90/95 is measured and reported, never asserted to be
correct. Mean-field VI can under-cover (ADR-0001), and no post-hoc calibration or
band inflation is applied.

## Canonical result

At the 90% nominal level, seed 10001 gives centered shape-function coverage of
0.960 (skew), 0.710 (tail), 0.680 (location), and 0.750 (scale). The figure
recovers the principal skew and location trends while showing materially wider
uncertainty over tailweight and scale—the intended uncertainty-over-shape
showcase rather than a claim of nominal mean-field calibration.
