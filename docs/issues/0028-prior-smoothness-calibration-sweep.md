# Issue 0028: Prior/smoothness calibration sweep

> Source: `CONTEXT.md` paper claim posture and ADR-0002 (`prior_scale` as the
> neural smoothing parameter). This is an experiment-scoped improvement branch,
> not a new inference-engine decision.

## What to learn

Test whether the existing `PriorScale` machinery improves epistemic effect-band
calibration before spending complexity on architecture sweeps or richer
posterior families.

The first experimental unit is Normal parameter recovery because it is small,
finite-variance, two-parameter, and already emits calibration,
intercept-coverage, recovery-figure, and prior-scale diagnostics.

## Pre-registered candidates

Run the five committed configs under `experiments/parameter_recovery/prior_sweep/`:

- fixed `prior_scale=0.3`
- fixed `prior_scale=1.0` baseline
- fixed `prior_scale=3.0`
- `prior='empirical_bayes'`, initialized at `prior_scale=1.0`
- `prior={'mode': 'hierarchical', 'hyperprior': 'inverse_gamma'}`, initialized
  at `prior_scale=1.0`

Half-Cauchy is deliberately deferred from the first pass because its
single-sample KL estimator adds noise before the cleaner inverse-gamma tier has
been interpreted.

## Acceptance criteria

- Mean absolute coverage error across Normal location and scale at nominal
  50/80/90/95 improves relative to the fixed-`1.0` baseline.
- Centered recovery remains visually plausible in `figures/recovery.pdf`.
- Intercept coverage is not pathological.
- Final loss does not explode.
- Ribbons do not become obviously over-wide everywhere.
- `metrics/prior_scale.json` reports an interpretable learned or sampled scale
  rather than collapse to an extreme.

## Seed and promotion policy

The first screening pass uses seed `9801`. No candidate may be promoted from one
seed. A candidate that beats the fixed-`1.0` baseline must be rerun against that
baseline on confirmatory seeds `9801`, `9811`, and `9821`.

All output stays under ignored `experiments/parameter_recovery/runs/` until the
screening and confirmatory checks have been reviewed. Only then may a run be
promoted to `results/` or added to the publication evidence manifest.

## Commands

Smoke-check one candidate:

```bash
uv run --extra experiments python experiments/parameter_recovery/run.py \
  experiments/parameter_recovery/prior_sweep/normal-empirical-bayes.yaml --smoke
```

Run a full candidate by omitting `--smoke`. Use
`experiments/publication/paper_results_explorer.ipynb` to compare available
sweep outputs and generate confirmatory seed configs after a screening winner is
chosen.

## Follow-up branch

If the prior/smoothness branch does not materially improve coverage, the first
richer-inference fallback is a last-layer richer posterior or low-rank covariance
path around `NeuralLinearMLP` or a linearized final layer. That branch is a
follow-up extension, not a v1 paper requirement.
