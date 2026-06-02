# Issue 0003: BayesianNAMLSS walking skeleton (fit with KL/N)

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The walking skeleton: a `BayesianNAMLSS` model class (same `AdditiveBaseModel`/`tf.keras.Model` lineage as `NAMLSS`) that trains a Bayesian additive formula end-to-end on the familiar `compile(loss=model.Loss)` / `fit()` surface. `Loss = mean-NLL + KL/N` (N from the `DataModule`; per-net variational KL rides `add_loss` auto-propagation, spike-verified through nested sub-models + `DistributionLambda`). Handles full or partial-Bayesian formulas (a deterministic net is a degenerate zero-variance contributor). `feature_dropout` defaults to 0 when Bayesian nets are present, fully configurable otherwise. `call()` stays a single stochastic pass. The deterministic `NAMLSS` is untouched.

## Acceptance criteria

- [ ] `BayesianNAMLSS(formula, data, family, ...)` builds for single- and multi-feature fully-Bayesian formulas
- [ ] `compile(loss=model.Loss)` + `fit()` trains to convergence on a toy regression (NLL decreases)
- [ ] KL/N (N = training-set size) appears in the total loss; the spike-proven propagation holds in the real model
- [ ] A partial-Bayesian formula (`BayesianMLP(x1) + MLP(x2)`) trains, the deterministic term contributing zero KL
- [ ] `feature_dropout` defaults to 0 when Bayesian nets are present and remains overridable

## Blocked by

- Issue 0002 (BayesianMLP shape function)
