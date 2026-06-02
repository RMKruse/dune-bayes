# Issue 0012: Categoricals as random effects

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

Make a categorical features first mapping (embedding/lookup → Dense) Bayesian under the per-feature `PriorScale`, giving a credible interval per level and shrinking rare levels toward the prior — partial pooling, the neural analog of a BAMLSS random effect (ADR-0005). The per-feature `prior_scale` doubles as the random-effect variance component. A point-embedding fallback exists but is not the default (it makes rare levels look falsely confident).

## Acceptance criteria

- [ ] Categorical first mapping is variational under the feature `PriorScale`
- [ ] Each level gets a credible interval; rare levels visibly shrink toward the prior
- [ ] `prior_scale` acts as the shared random-effect variance component across levels
- [ ] Point-embedding fallback is available but not the default

## Blocked by

- Issue 0006 (Centered epistemic effect ribbons)
- Issue 0011 (PriorScale handle)
