# 7. Local reparameterization for training; coherent weight draws for posterior sampling

Date: 2026-06-07

## Status

Accepted.

## Context

`VariationalDense` carries a variance-reduction estimator behind a flag named
`flipout`. The implementation (`_forward_flipout`) samples pre-activations from
the marginal `N(x @ loc, x² @ scale²)` with independent noise per batch row —
that is **local reparameterization** (Kingma, Salimans & Welling, 2015), *not*
flipout (Wen et al., 2018, sign-flipped pseudo-independent weight
perturbations). The flag defaults to off, so the de-facto training path has been
vanilla global reparameterization.

Two facts collide:

1. **Per-row independent noise is exactly what reduces gradient variance** in
   minibatch ELBO training — with one shared weight draw the noise is perfectly
   correlated across the batch and minibatch averaging cannot cancel it.
2. **Per-row independent noise destroys coherent function draws.** A posterior
   draw of a shape function must be one weight realization `W ~ q(W)` evaluated
   at every input. Under local reparameterization no `W` is ever materialized;
   each row's output is consistent with a *different* implicit weight draw.
   Per-point marginals survive, but everything joint — draw smoothness,
   monotonicity/peak statements, and crucially the `MixtureSameFamily`
   aleatoric/epistemic decomposition (which requires draw `t` to use the *same*
   weights across all observations and feature nets) — is silently corrupted.

The aleatoric/epistemic split is the package's core scientific claim, so the
sampling path must be provably coherent.

## Decision

- **Rename `flipout` → `local_reparam`** everywhere (layer flag, internals,
  configs, docs). Pre-v1, no API debt; the old name misstates the estimator and
  would be a methods-section error in the paper.
- **Local reparameterization is the training default** (`local_reparam=True`
  on `VariationalDense` as built by the shape functions): training only ever
  needs per-row marginals of the loss, and the variance reduction is free.
- **All posterior sampling uses vanilla coherent weight draws.**
  `EffectSampler.sample_effects`, `draw_predictive`, and `pointwise_log_lik`
  must never route through the local-reparam path: each of the `T` draws is one
  global weight realization applied to every input.
- **A test pins the boundary**: posterior-sampling entry points force the
  vanilla path (asserted directly), and a behavioral check verifies coherence
  within a draw (e.g. a linear shape function gives constant `f(x)/x` across a
  grid within one draw).

## Consequences

**Positive**

- Lower-variance ELBO gradients by default; faster, more stable training.
- The paper's methods section names the estimator correctly.
- The disentanglement claim rests on a provably coherent sampler, guarded by a
  test rather than convention.

**Negative / accepted trade-offs**

- Two forward paths to maintain in `VariationalDense`; the boundary between
  them is load-bearing and must stay tested.
- Local reparameterization assumes a factorized Gaussian posterior on weights —
  acceptable, since mean-field is the decided inference engine (ADR-0001).

## Alternatives considered

- **True flipout** — also variance-reducing while materializing
  pseudo-coherent weight perturbations; more code, weaker variance reduction
  than local reparam for mean-field Gaussians, and sampling would still use the
  vanilla path. Not worth the complexity.
- **Vanilla everywhere** — one code path, but pays full gradient variance in
  training for no statistical benefit.
- **Local reparam everywhere** — invalidates coherent function draws and the
  decomposition; ruled out (the core claim depends on it).
