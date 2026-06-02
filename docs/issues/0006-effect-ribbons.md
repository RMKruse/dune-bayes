# Issue 0006: Centered epistemic effect ribbons

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The per-feature effect plot consuming `EffectSampler`: a centered, epistemic-only credible ribbon per feature. Each posterior-sampled curve is mean-centered over the data before quantiles, isolating shape uncertainty from the overall level. Default band 90%, configurable; centering is flag-able (this deviates from NAMpys uncentered default). Extends NAMpys existing visuals rather than rewriting them.

## Acceptance criteria

- [ ] Per-feature effect plot renders a credible ribbon (default 90%, configurable) from `EffectSampler` draws
- [ ] Ribbon is epistemic-only and curves are mean-centered before quantiles by default
- [ ] Centering is flag-able (uncentered mode available)
- [ ] Aleatoric uncertainty is NOT attributed to feature curves

## Blocked by

- Issue 0005 (EffectSampler workhorse)
