# Issue 0013: Interactions as joint Bayesian nets + surface plots

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

A Bayesian interaction `BayesianMLP(x1):BayesianMLP(x2)` resolves to a single joint Bayesian net over both inputs (the `:` syntax already does this — network name from the first term, the rest added as inputs — so no parser change and no new machinery). Only the plot changes: render a posterior-mean surface plus a separate epistemic-SD surface (or sliced 1D bands), since a 1D ribbon does not generalize to 2D.

## Acceptance criteria

- [ ] `BayesianMLP(x1):BayesianMLP(x2)` builds as a single joint variational net over both inputs (no parser change)
- [ ] `EffectSampler` handles the interaction term as-is
- [ ] Interaction renders as a posterior-mean surface plus a separate epistemic-SD surface (or sliced 1D bands)

## Blocked by

- Issue 0006 (Centered epistemic effect ribbons)
