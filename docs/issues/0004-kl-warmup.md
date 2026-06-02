# Issue 0004: KL warm-up auto-injection

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

A `fit()` override on `BayesianNAMLSS` that auto-injects the KL warm-up callback (on by default), guarding against posterior collapse. The callback drives every `VariationalDense`s `kl_beta` as β = min(1, epoch / warmup_epochs) at epoch start, recursing through nested sub-models. Warm-up length configurable; the override is transparent to the normal `fit()` signature.

## Acceptance criteria

- [ ] `fit()` injects the warm-up callback automatically with no extra user setup
- [ ] `kl_beta` follows β: 0→1 over the configured warm-up epochs on every variational layer (including nested)
- [ ] Warm-up length is configurable and can be disabled
- [ ] A user-supplied `callbacks` list is preserved alongside the injected one

## Blocked by

- Issue 0003 (BayesianNAMLSS skeleton)
