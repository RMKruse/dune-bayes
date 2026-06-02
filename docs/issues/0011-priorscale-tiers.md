# Issue 0011: PriorScale handle: empirical-Bayes + hierarchical tiers

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

Promote the fixed scalar from issue 0001 into the full `PriorScale` deep module (ADR-0002), resolving one per-feature prior-variance scalar (= one mgcv smoothing parameter λ; penalty ⇔ Gaussian prior) into a KL contribution across three tiers: fixed (default, per-net configurable in the formula); empirical-Bayes learned scale (neural analog of REML smoothness selection, opt-in); full-hierarchical with a hyperprior (half-Cauchy default, inverse-gamma for BAMLSS-faithful mode). Config is closure-free and serializable. The same scalar doubles as the categorical random-effect variance component (issue 0012).

## Acceptance criteria

- [ ] Per-net `prior_scale` is configurable in the formula and acts as the smoothing/shrinkage knob
- [ ] Empirical-Bayes tier learns the scale (opt-in) and is the REML analog
- [ ] Full-hierarchical tier supports half-Cauchy (default) and inverse-gamma hyperpriors
- [ ] Each tier produces the expected KL contribution against a reference
- [ ] Each tier serializes closure-free

## Blocked by

- Issue 0001 (VariationalDense atom)
