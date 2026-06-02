# 3. Posterior-predictive representation, plotting, and comparison architecture

Date: 2026-06-02

## Status

Accepted. **Amended by ADR-0006 (2026-06-03):** the statistical contract below is
unchanged, but the backend is now **PyTorch**. Translate the TFP/Keras idioms as
follows: `tfd.MixtureSameFamily` → `torch.distributions.MixtureSameFamily`;
`compile/fit` + KL-via-`add_loss` → a thin trainer with explicit KL collection;
and the flagged "TFP variational-layer save/load is finicky" risk is **retired**
by `state_dict`. arviz is unaffected.

## Context

With variational weights, a single forward pass is one draw from the conditional,
not the predictive. The package needs a coherent contract for (a) the posterior
predictive, (b) uncertainty in the interpretability plots, and (c) model
comparison — all fed from posterior weight samples. The codebase is Keras
functional + `tfp` + `.fit()`, and the deterministic `NAMLSS` returns a dict
`{"output": p_y, "summed_output", **feature_preds}` from a single `call()`.

## Decision

- **Single sampling workhorse.** `sample_posterior_predictive(data, T)` runs `T`
  stochastic forward passes and returns, in one sweep: per-feature contribution
  samples, summed-predictor samples, and a pointwise log-likelihood matrix
  (`T × N`). `call()` remains a single stochastic pass (one MC ELBO sample/step).
- **Predictive = `tfd.MixtureSameFamily`** over the `T` weight-sampled family
  distributions (uniform mixture weights). Gives `.log_prob`, `.sample`, `.mean`,
  `.stddev`, quantiles for free; spread across components = epistemic, within a
  component = aleatoric.
- **Plots split by view.** Per-feature effect plots show a **centered,
  epistemic-only** credible ribbon (each sampled curve mean-centered before
  quantiles; default 90%, centering flag-able). Response-level plots
  (`plot_dist` / predicted-vs-actual) show the **full predictive band
  (epistemic + aleatoric)**; aleatoric is not attributed to single features.
- **Comparison on arviz.** `to_inference_data()` + **WAIC** and **PSIS-LOO**
  (Pareto-k diagnostics) + a `compare()` wrapper over `az.compare`. **ELBO** is a
  secondary biased evidence proxy. New dependency: **arviz**.
- **Dedicated `BayesianNAMLSS` model class** hosts all of the above and handles
  partial-Bayesian formulas; deterministic `NAMLSS` is unchanged. Training keeps
  the `compile(loss=model.Loss)`/`fit()` surface (KL auto via `add_loss`, warm-up
  auto-injected).

## Consequences

**Positive**

- One sampling pass serves prediction, interpretation, and comparison — no
  duplicated Monte-Carlo machinery.
- `MixtureSameFamily` keeps everything inside the existing TFP stack and yields a
  correct `log_prob` for the information criteria.
- arviz gives robust LOO with reliability diagnostics and a ready comparison
  table for the "handful of models" workflow.
- Familiar Keras training surface; low relearning cost.

**Negative / accepted trade-offs**

- New runtime dependency (arviz) and adoption of its `InferenceData` container.
- Prediction now costs `T×` forward passes; `T` is a speed/accuracy knob.
- `MixtureSameFamily` with large `T` has a memory/compute footprint at predict
  time.
- TFP variational-layer serialization (save/load) is known to be finicky — a
  flagged implementation risk, not yet designed.

## Alternatives considered

- **Raw sample tensors** as the predictive contract — maximally flexible but
  every consumer re-implements mixing/log-prob/quantiles. Rejected; raw
  per-feature samples are still exposed alongside the mixture.
- **Summary statistics only** — simplest to consume but discards the full
  predictive needed for honest `log_prob` / comparison. Rejected.
- **Hand-rolled WAIC/LOO** (no arviz) — avoids a dependency but reinvents PSIS and
  loses Pareto-k diagnostics. Rejected.
- **Extending `NAMLSS` polymorphically / a mixin** instead of a dedicated class —
  rejected for v1 to keep the deterministic path clean and the Bayesian behavior
  cohesive.
