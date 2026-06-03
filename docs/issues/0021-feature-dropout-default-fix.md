# Issue 0021: Fix feature_dropout default no-op

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

Fix a defaulting bug in `BayesianNAMLSS`: when `feature_dropout` is left unset, both branches of the conditional currently resolve to `0.0`, so the "default to 0 when Bayesian nets are present, otherwise the NAMpy-style default" intent (user stories 42–43) is silently lost — the else-branch is dead.

Restore the intended behavior: `feature_dropout` defaults to `0.0` when any Bayesian net is present (the weight posterior is the sole stochasticity), and to the NAMpy deterministic default otherwise, while remaining fully configurable via the constructor argument (user story 43). When Bayesian nets are present, also strip/ignore internal per-layer dropout so epistemic uncertainty is not conflated with dropout noise (user story 41).

## Acceptance criteria

- [ ] With Bayesian nets present and `feature_dropout` unset, the effective rate is `0.0`
- [ ] With only deterministic nets and `feature_dropout` unset, the effective rate is the documented NAMpy-style default (not unconditionally `0.0`)
- [ ] An explicit `feature_dropout=` argument overrides the default in both cases
- [ ] Boundary tests cover all three paths (Bayesian-present default, deterministic default, explicit override)

## Blocked by

None - can start immediately.
