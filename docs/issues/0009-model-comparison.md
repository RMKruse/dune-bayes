# Issue 0009: Model comparison: WAIC / LOO / compare

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The arviz-backed model-comparison deep module. `to_inference_data()` exposes posterior + pointwise log-likelihood (from `LogLikSampler`) as an `InferenceData`. Ships `waic()`, `loo()` (PSIS-LOO, surfacing Pareto-k reliability warnings), and `compare()` (wrapper over `az.compare`). ELBO exposed as a secondary, biased evidence proxy. No literal Bayes Factors (ADR-0001). IC use `T_eval = 1000` by default.

## Acceptance criteria

- [ ] `to_inference_data()` returns an arviz `InferenceData` with pointwise log-likelihood
- [ ] `waic()` and `loo()` match a hand-computed arviz reference on a tiny fixture
- [ ] `loo()` surfaces the Pareto-k reliability warning at deliberately low T
- [ ] `compare()` ranks two fitted-model fixtures in the known order
- [ ] ELBO is retrievable as a secondary evidence proxy; no Bayes Factor is computed

## Blocked by

- Issue 0007 (LogLikSampler + MixtureSameFamily predictive)
