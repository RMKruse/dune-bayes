# Issue 0018: BaseFamily contract

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

The family abstraction the PRD assumes ("families: `BaseFamily` + concretes") but which `dev` lacks — only a standalone `NormalFamily` exists. Introduce a `BaseFamily` contract (ABC or Protocol) capturing what every response family must expose: `param_count`, per-parameter link/transform functions (positivity via `softplus`, never `exp`/`clamp` — Numerical rule 1), and a `__call__(params) -> torch.distributions.Distribution` plus `log_prob`. Refactor `NormalFamily` to implement the contract with no behavior change.

This slice is the seam that makes "location, scale and shape" general rather than Normal-only; concrete additional families are issue 0019.

## Acceptance criteria

- [ ] `BaseFamily` defines the contract: `param_count`, link/transform application, and `__call__ -> Distribution`
- [ ] `NormalFamily` implements `BaseFamily` with identical observable behavior (existing tests stay green)
- [ ] Positivity transforms route through `softplus` per the numerical rules; `validate_args` follows the test-vs-hot-path convention
- [ ] A `BayesianNAMLSS` built with the refactored `NormalFamily` trains and produces a posterior predictive unchanged from before
- [ ] Boundary tests cover the contract surface (param_count, link application, log_prob against a torch.distributions reference)

## Blocked by

None - can start immediately.
