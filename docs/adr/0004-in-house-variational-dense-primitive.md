# 4. In-house `VariationalDense` primitive instead of raw TFP layers

Date: 2026-06-02

## Status

Accepted — both load-bearing claims verified by spike on the **then-target** stack
(TF 2.15.1 / Keras 2.15 / TFP 0.23.0 / Python 3.11), 2026-06-02. See
`spikes/` and the Verification section below.

**Amended by ADR-0006 (2026-06-03):** the core decision — an owned variational
atom rather than a stock library layer — **survives the backend switch to
PyTorch**, for the same reasons. What changes: the host framework (Keras `Layer` →
`nn.Module`), and serialization (`.keras`/SavedModel → `state_dict` + config dict),
which *dissolves* the H5/`.keras` fragility this ADR worked around — there is no
PyTorch equivalent of the weight-name collision. The TF/TFP spike results in the
Verification section below are now **historical**, but both load-bearing claims
have been **re-verified green on PyTorch** (torch 2.12 / Python 3.12): KL
aggregation across the nested NAMLSS graph via an explicit module-walk, and a
`state_dict` + config save/load round-trip with `max|Δw| = 0`. See the ported
`spikes/` and the PyTorch note in the Verification section.

## Context

Every Bayesian shape function (`BayesianMLP`, `NeuralLinearMLP`) is built from a
single atom: a `Dense` layer whose weights have a variational posterior. The
codebase already commits to `tfp` (families return `tfd.*` distributions; every
LSS head is a `tfp.layers.DistributionLambda`), but no variational layer exists
yet — so this primitive is chosen on a clean slate.

The two stock TFP options each conflict with decisions already locked:

- `tfp.layers.DenseVariational` — flexible priors via `make_prior_fn` /
  `make_posterior_fn` closures (what ADR-0002's per-feature `prior_scale`,
  empirical-Bayes, and half-Cauchy/IG tiers need), **but** higher gradient
  variance (one weight draw shared across the batch) and closures that do not
  serialize through Keras save/load.
- `tfp.layers.DenseFlipout` — low-variance gradients (flipout decorrelates draws
  within a minibatch), **but** a largely fixed prior that fights ADR-0002's
  hierarchical-scale design.

Both stock layers carry the documented Keras save/load fragility flagged as an
open risk in ADR-0003. Picking either raw primitive forces giving up something
already decided: prior flexibility (ADR-0002), gradient stability, or a working
round-trip.

## Decision

Ship a **thin in-house `VariationalDense`** Keras layer as the single atom, not
either raw TFP primitive.

- **Posterior:** mean-field Normal per weight (`loc` + softplus `scale`).
- **Prior:** specified by **serializable config** — a `prior_scale` float or a
  handle to the hierarchical scale from ADR-0002 — *not* a Python closure. This
  implements ADR-0002's prior design directly.
- **KL:** emitted via `add_loss`, riding the existing KL/N + warm-up machinery
  (ADR-0003) unchanged.
- **Estimator:** local-reparameterization / flipout-style draw available as an
  internal flag for variance reduction — Flipout's stability without inheriting
  its rigid prior, because we own the layer.
- **Serialization solved by design:** `get_config` / `from_config` round-trip the
  prior/posterior *hyperparameters* (floats, strings); the variational `loc` /
  `scale` save as ordinary Keras weights. Save/load becomes a supported, tested
  path rather than a flagged risk; a round-trip test guards it.

## Consequences

**Positive**

- The one combination neither stock layer offers: flexible per-feature/hierarchical
  priors (ADR-0002) **and** a variance-reduction option **and** working save/load.
- Closure-free `get_config` retires the ADR-0003 serialization risk.
- We control the reparameterization, so the flagship `BayesianMLP` can default to
  the low-variance estimator while keeping arbitrary priors.
- KL integrates with the existing `add_loss` path — no change to compile/fit.

**Negative / accepted trade-offs**

- ~80 lines of in-house layer code to own, test, and keep correct against TFP
  changes, instead of reusing a maintained primitive.
- The reparameterization/KL math is ours to get right (mitigated by a closed-form
  Gaussian–Gaussian KL and a unit test against a known reference).
- Re-derives functionality TFP already provides for the common (fixed-prior) case.

## Verification

> **PyTorch re-verification (ADR-0006, 2026-06-03).** The spikes have been ported to
> PyTorch and both claims pass: spike 1 — `collect_kl`'s module-walk gathers KL from
> all 6 nested `VariationalDense` modules through the sum and the `torch.distributions`
> family head, `beta` gates it (β=0 → 0, β=1 → restored), it scales exactly as KL/N
> (ratio = 256), and an optimizer step adds it to the NLL (gap = KL/N); spike 2 —
> a `config + state_dict` bundle reconstructs the model from config alone and
> round-trips every variational weight with `max|Δw| = 0`. The original TF/TFP
> verification below is retained as the historical record.

Scaffolded spikes (`spikes/`) reproduce the real NAMLSS graph shape (per-feature
sub-models → `Add` → `tfp.layers.DistributionLambda`) and assert the two claims on
the pinned target stack:

- **KL via `add_loss` (spike_kl_propagation.py):** ALL PASS. KL emitted inside
  `VariationalDense.call()` reaches the *outer* `model.losses` across the
  sub-model and `DistributionLambda` boundaries (6/6 layers), the warm-up `beta`
  gates it (β=0 → 0, β=1 → restored), it is weighted KL/N (ratio = N exactly), and
  `.fit()` adds it to the NLL. The KL/N + warm-up design needs no custom train
  loop.
- **Serialization (spike_serialization.py):** ALL PASS. `get_config` is
  closure-free and preserves hyperparameters; the **native `.keras`** format and
  **SavedModel** both round-trip with variational weights identical
  (`max|Δw| = 0`). **Legacy H5 fails** (HDF5 weight-name collision across the two
  variational layers) and is therefore *not* a supported save format — `.keras`
  (recommended) and SavedModel are. This narrows, but confirms, the claim that the
  in-house layer retires the `tfp.layers` save/load fragility.

## Alternatives considered

- **Raw `tfp.layers.DenseVariational`** — flexible priors, but high gradient
  variance and closure-based config that breaks Keras save/load. Rejected.
- **Raw `tfp.layers.DenseFlipout`** — low-variance gradients, but a rigid prior
  that conflicts with ADR-0002's hierarchical scales, plus the same save/load
  fragility. Rejected.
- **Wrap a stock layer and only override `get_config`** — does not fix the
  gradient-variance / prior-rigidity axis, only serialization. Insufficient.
