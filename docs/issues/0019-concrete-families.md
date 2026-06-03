# Issue 0019: Concrete distributional families

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

Add at least two concrete response families on top of the `BaseFamily` contract (issue 0018), **including one genuine 3-parameter shape family** so the package delivers on "location, scale **and** shape" rather than the 2-parameter Normal alone. Exact distributions are the implementer's choice (e.g. StudentT for location/scale/df, plus a positive-response family such as Gamma or LogNormal), provided one exposes a shape parameter beyond location and scale.

Each family wraps `torch.distributions`, routes scale/shape positivity through `softplus`, reports its `param_count`, and trains end-to-end inside `BayesianNAMLSS`, yielding per-feature effect curves on every distributional parameter.

## Acceptance criteria

- [ ] At least two families implement `BaseFamily`, one with `param_count >= 3` (a true shape parameter)
- [ ] A `BayesianNAMLSS` trains end-to-end with each new family on a toy dataset and produces a posterior predictive
- [ ] WAIC/LOO via the compare module run against a model using a non-Normal family
- [ ] Positivity/shape transforms follow the numerical rules (softplus, log-space `log_prob`)
- [ ] Boundary tests cover each family's `log_prob` against an independent `torch.distributions` reference and the per-parameter shape of effect samples

## Blocked by

- Issue 0018 (BaseFamily contract)
