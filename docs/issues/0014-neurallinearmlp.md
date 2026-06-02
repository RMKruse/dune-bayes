# Issue 0014: NeuralLinearMLP shape function

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The cheaper last-layer-only Bayesian companion: a deterministic hidden basis with a variational output layer (built from `VariationalDense`). More stable and faster than the fully-variational `BayesianMLP`; the fallback for the large-data end of the range and for quick baselines. Registered alongside the other shape functions and selectable per term.

## Acceptance criteria

- [ ] `NeuralLinearMLP` is registered and resolvable by name from a formula
- [ ] Hidden layers are deterministic; only the output layer is variational
- [ ] It trains within `BayesianNAMLSS` and contributes KL only from the output layer
- [ ] Effect bands from `EffectSampler` reflect last-layer-only uncertainty

## Blocked by

- Issue 0003 (BayesianNAMLSS skeleton)
