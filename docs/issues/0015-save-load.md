# Issue 0015: Save/load round-trip

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

End-to-end save/load for a fitted `BayesianNAMLSS`, resting on the closure-free `get_config` from issue 0001. The native `.keras` format and SavedModel both round-trip with variational weights intact; legacy H5 is explicitly unsupported (HDF5 weight-name collision across variational layers — spike-confirmed). A round-trip test guards the path.

## Acceptance criteria

- [ ] A fitted `BayesianNAMLSS` saves and loads via native `.keras` with `max|Δw| = 0`
- [ ] Same round-trip works via SavedModel
- [ ] Legacy H5 is documented/handled as unsupported (clear failure or skip, not silent corruption)
- [ ] Reloaded model runs a forward pass and reproduces architecture/hyperparameters

## Blocked by

- Issue 0003 (BayesianNAMLSS skeleton)
