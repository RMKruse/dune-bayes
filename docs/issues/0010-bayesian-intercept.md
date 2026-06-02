# Issue 0010: Bayesian intercept

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

A `BayesianIntercept` variational layer — one location per distributional parameter — with its own wide weakly-informative `Normal(0, σ_int²)` prior, deliberately decoupled from the per-feature `prior_scale` (a location must not be shrunk toward zero like a smoothness term). Bayesian by default; a `point` (deterministic, NAMpy `InterceptLayer`) fallback is retained. Rationale: effect plots mean-center each curve into the intercept, so the absorbed overall level — and its uncertainty — accumulates there.

## Acceptance criteria

- [ ] Intercept is variational by default with its own wide `Normal(0, σ_int²)` prior
- [ ] Intercept prior is independent of per-feature `prior_scale`
- [ ] `point` deterministic intercept fallback is selectable
- [ ] Intercept uncertainty shows up in the response-level predictive mean

## Blocked by

- Issue 0003 (BayesianNAMLSS skeleton)
