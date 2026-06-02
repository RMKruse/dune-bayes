# Issue 0001: VariationalDense atom

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The single in-house variational primitive every Bayesian shape function is built from (ADR-0004), promoted from the verified `spikes/` seed into package code. A `Dense`-shaped Keras layer with a mean-field Normal weight posterior (`loc` + softplus `scale`), a prior set by serializable config (a `prior_scale` float, never a closure), KL emitted via `add_loss`, an internal flipout-style / local-reparameterization estimator flag for variance reduction, and a non-trainable `kl_beta` variable the warm-up callback (issue 0004) drives. `call()` is a single reparameterized stochastic pass.

This is a standalone, independently testable deep module — no model wiring yet.

## Acceptance criteria

- [ ] `VariationalDense(units, prior_scale, kl_divisor, flipout, activation, use_bias)` builds and runs, output shape `(batch, units)`
- [ ] KL is emitted via `add_loss` and equals the closed-form Gaussian–Gaussian KL against an independent hand-computed reference
- [ ] `kl_beta` gates the emitted KL (β=0 → 0, β=1 → full)
- [ ] flipout and vanilla estimators agree in expectation (mean over many draws) while differing in gradient variance
- [ ] `get_config`/`from_config` round-trip is closure-free (floats/strings/bools only) and reconstructs an equivalent layer

## Blocked by

None - can start immediately
