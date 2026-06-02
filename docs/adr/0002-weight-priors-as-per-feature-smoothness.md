# 2. Weight priors as per-feature smoothness, with staged inference

Date: 2026-06-02

## Status

Accepted

## Context

Goal #2 (priors as principled regularization) and the "neural-BAMLSS" identity
hinge on how the weight priors of Bayesian shape functions are specified. A
classical GAM's smoothness penalty `λ βᵀ S β` is exactly a Gaussian prior on the
coefficients with variance ∝ 1/λ; estimating the smoothing parameter λ *is*
estimating the prior variance. So the prior scale on a feature net is the neural
analog of a per-feature smoothness/shrinkage parameter, and the design must
decide (1) the granularity of that scale, (2) how it is inferred, and (3) any
hyperprior.

## Decision

1. **Granularity — one prior-variance scalar per feature net.** This mirrors
   mgcv assigning one smoothing parameter λ per smooth term: one
   smoothness/shrinkage knob per additive term. Per-layer scales are available as
   an advanced override; a single global scale is rejected as too coarse.

2. **Inference — staged:**
   - *Default:* fixed Gaussian prior `N(0, σ²)`, with σ configurable per net via
     formula args (e.g. `BayesianMLP(x1; prior_scale=0.5)`).
   - *Opt-in (headline feature):* **empirical-Bayes** learning of the per-feature
     scale by optimizing it through the ELBO — the neural analog of mgcv's
     REML/ML smoothness selection. One point estimate of smoothness per feature,
     cheap and robust.
   - *Advanced tier:* full-hierarchical prior with a variational posterior over
     the scale. Default hyperprior = **half-Cauchy** (weakly informative,
     ARD-like shrink-to-zero that feeds feature relevance / goal 3, representable
     as a Normal–Inverse-Gamma scale mixture for VI). An **inverse-gamma** option
     is provided for BAMLSS-faithful replication (BAMLSS uses IG priors on
     variance components in its MCMC).

## Consequences

**Positive**

- Clean, interpretable mapping: one smoothness number per feature, matching GAM
  intuition and the existing per-feature additive structure.
- Staging keeps v1 tractable (fixed prior) while exposing the principled
  auto-smoothness path (empirical Bayes) and a fully-Bayesian tier.
- Half-Cauchy default doubles as soft feature selection, supporting model
  comparison (goal 3); IG mode preserves fidelity to the BAMLSS reference.
- Fits the existing formula-hyperparameter convention — no new user surface.

**Negative / accepted trade-offs**

- Three tiers = more code paths to implement and test than a single fixed prior.
- Empirical Bayes can over-shrink with very little data; full-hierarchical VI on
  the scale can be fragile (mitigated by non-centered / scale-mixture
  parameterization).
- Per-net scalar (vs per-layer) trades some flexibility for interpretability.

## Alternatives considered

- **Per-layer or per-weight scales** — more flexible but dissolve the
  one-smoothness-per-feature interpretation; offered only as an advanced override.
- **Fixed Gaussian only** — simplest, but no automatic smoothness selection;
  rejected as the sole option because it leaves goal 2 to manual tuning.
- **Full-hierarchical as the default** — most Bayesian but too fragile / slow to
  be the out-of-the-box behavior.
