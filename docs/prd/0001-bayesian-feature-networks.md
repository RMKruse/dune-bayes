# PRD 0001: neural-bamlss — Bayesian feature networks for distributional additive models

Status: Draft · Date: 2026-06-03

## Problem Statement

I fit interpretable distributional additive models (NAMLSS — Neural Additive Models for Location, Scale and Shape) with NAMpy. They give me per-feature effect curves on every parameter of the response distribution, but every effect is a single point estimate. On the small datasets I actually care about, I cannot tell which effects are well-determined and which are guesses, the models overfit without a principled regularizer, and when I have a handful of candidate formulas I have no defensible way to choose between them. NAMpy's deterministic feature networks give me no epistemic uncertainty, no principled prior-based regularization, and no Bayesian model-comparison criterion.

## Solution

A dedicated package, **neural-bamlss** — the neural analog of the BAMLSS R package — that reuses NAMpy's machinery (formula parser, `ShapeFunctionRegistry`, families, `DataModule`, plotting) but replaces the deterministic feature networks with **Bayesian** feature networks. Each feature's effect on each distributional parameter now carries a posterior, so I get:

1. **Credible intervals around every feature's shape function** (the headline payoff).
2. **Priors as principled regularization** that pay off most on small data, with the per-feature prior scale playing the role of an mgcv smoothing parameter λ.
3. **Principled model comparison** via WAIC / PSIS-LOO over a handful of candidate formulas, with the ELBO as a secondary evidence proxy.

I keep the familiar `compile(loss=model.Loss)` / `fit()` surface. I write a formula like `y ~ BayesianMLP(x1) + MLP(x2) + BayesianMLP(x3):BayesianMLP(x4)`, mixing Bayesian and deterministic terms per feature, and get banded effect plots, full predictive intervals, and `waic`/`loo`/`compare` methods.

## User Stories

1. As a modeller, I want each per-feature effect curve drawn with a credible ribbon, so that I can see which effects are well-determined and which are uncertain.
2. As a modeller, I want effect-plot ribbons that are epistemic-only and mean-centered, so that I read shape uncertainty in isolation from the overall level.
3. As a modeller, I want response-level plots to show the full predictive band (epistemic + aleatoric), so that I get a proper prediction interval rather than a shape band.
4. As a modeller, I want aleatoric uncertainty kept at the response level and not attributed to individual feature curves, so that effect plots are not misleadingly wide.
5. As a modeller on small data, I want priors acting as principled regularization, so that my model does not overfit and degrades gracefully where data is sparse.
6. As a modeller, I want each feature net to carry one prior-variance scalar that behaves like an mgcv smoothing parameter, so that I can reason about smoothness the way I do in a GAM.
7. As a modeller, I want to fix that prior scale per net in the formula, so that I can control regularization strength term by term.
8. As an advanced user, I want to opt into empirical-Bayes learning of the prior scale, so that I get the neural analog of REML smoothness selection without hand-tuning.
9. As an advanced user, I want a full-hierarchical prior tier (half-Cauchy default, inverse-gamma for BAMLSS-faithful mode), so that the smoothness is itself inferred under a hyperprior.
10. As a modeller comparing candidate formulas, I want a WAIC score per model, so that I can rank them on a defensible Bayesian criterion.
11. As a modeller, I want PSIS-LOO with Pareto-k reliability diagnostics, so that I know when the LOO estimate is untrustworthy.
12. As a modeller, I want a `compare()` wrapper over `az.compare`, so that I can rank several fitted models in one call.
13. As a modeller, I want the ELBO available as a secondary evidence proxy, so that I have a cheap (if biased) comparison number alongside WAIC/LOO.
14. As a modeller, I do NOT need literal marginal-likelihood Bayes Factors, so that I am not blocked on an intractable quantity for NN-sized models.
15. As a NAMpy user, I want to instantiate a dedicated `BayesianNAMLSS` class, so that posterior sampling, predictive, banded plotting, and IC methods live in one place while the deterministic `NAMLSS` stays untouched.
16. As a NAMpy user, I want to keep the `compile(loss=model.Loss)` / `fit()` surface, so that adopting Bayesian nets does not force me to learn a new training loop.
17. As a modeller, I want KL/N added to the loss automatically via `add_loss`, so that I do not hand-assemble the ELBO.
18. As a modeller, I want a KL warm-up (β: 0→1 over the first epochs) auto-injected by a `fit()` override and on by default, so that I am protected from posterior collapse without extra setup.
19. As a modeller, I want the warm-up length configurable, so that I can tune annealing for hard cases.
20. As a modeller, I want to mix Bayesian and deterministic terms in one formula (`y ~ BayesianMLP(x1) + MLP(x2)`), so that I can be Bayesian only where it matters.
21. As a modeller, I want a deterministic net to behave as a degenerate zero-variance contributor, so that partial-Bayesian formulas are well-defined.
22. As a modeller, I want to keep NAMpy's deterministic baselines (MLP, ResNet, …) registered, so that I can run direct Bayes-vs-deterministic effect comparisons and include a non-Bayesian baseline in WAIC/LOO.
23. As a modeller, I want a fully-variational `BayesianMLP` (every Dense → VariationalDense), so that uncertainty propagates through the function shape, not just a final rescaling.
24. As a modeller on larger data, I want a cheaper last-layer-only `NeuralLinearMLP`, so that I get a stable, fast Bayesian baseline.
25. As a modeller, I want a `sample_posterior_predictive(data, T)` capability, so that I can draw from the posterior predictive on new data.
26. As a modeller, I want the posterior predictive represented as a `tfd.MixtureSameFamily` over T weight-sampled family distributions, so that across-component spread is epistemic and within-component spread is aleatoric.
27. As a modeller, I want effect bands to come from posterior contribution samples, so that the ribbons reflect the actual weight posterior.
28. As a modeller, I want a sensible default number of posterior draws for plots (`T_predict = 200`), so that interactive plotting is fast with smooth-enough bands.
29. As a modeller, I want a higher default for information criteria (`T_eval = 1000`), so that WAIC/LOO are not biased/noisy from too few draws.
30. As a modeller, I want both T defaults overridable, so that I can trade speed for fidelity explicitly.
31. As a modeller, I want `loo()` to surface arviz's Pareto-k warning when T_eval was effectively too low, so that I am told when to raise it.
32. As a modeller, I want the variational intercept to be Bayesian by default with its own wide weakly-informative prior, so that the absorbed overall level carries uncertainty without being shrunk toward zero.
33. As a modeller, I want the intercept prior NOT tied to the per-feature prior_scale, so that a location term is not penalized like a smoothness term.
34. As a modeller, I want a `point` (deterministic) intercept fallback, so that I can drop intercept uncertainty when I do not want it.
35. As a modeller, I want `BayesianMLP(x1):BayesianMLP(x2)` to resolve to a single joint Bayesian net over both inputs, so that interactions need no new machinery.
36. As a modeller, I want an interaction rendered as a posterior-mean surface plus a separate epistemic-SD surface (or sliced 1D bands), so that I can read 2D effect uncertainty.
37. As a modeller, I want a categorical feature's first mapping (embedding/lookup → Dense) to be Bayesian under the per-feature prior_scale, so that I get a credible interval per level.
38. As a modeller, I want rare categorical levels to shrink toward the prior (partial pooling), so that the model behaves like a BAMLSS random effect and does not look falsely confident on thin levels.
39. As a modeller, I want the per-feature prior_scale to double as the categorical random-effect variance component, so that I have one coherent knob.
40. As a modeller, I want a point-embedding categorical fallback available, so that I can opt out of the random-effect treatment, even though it is not the default.
41. As a modeller, I want Bayesian shape functions to strip internal per-layer dropout, so that epistemic uncertainty is not conflated with dropout noise.
42. As a modeller, I want `feature_dropout` to default to 0 when Bayesian nets are present, so that the weight posterior is the only stochasticity by default.
43. As a modeller, I want dropout still fully configurable, so that I can build dropout-style Bayesian nets if I choose.
44. As a modeller, I want to save and load a fitted `BayesianNAMLSS`, so that I can persist and reuse models.
45. As a modeller, I want save/load to work via native `.keras` and SavedModel formats, so that I have a supported, tested round-trip (legacy H5 is not supported).
46. As a modeller, I want `call()` to remain a single stochastic pass (one MC ELBO sample per step), so that training cost stays close to the deterministic model.
47. As a maintainer, I want the variational primitive to be in-house rather than a raw TFP layer, so that I get flexible per-feature/hierarchical priors AND variance reduction AND working serialization in one atom.

## Implementation Decisions

**Reused from NAMpy, unchanged:** formula parser (`FormulaHandler.extract_formula_data`), `DataModule` (provides N = training-set size), families (`BaseFamily` + concretes; `param_count`, transforms, `__call__` → `tfd` distribution), `ShapeFunctionRegistry`, and the existing plotting entry points (extended, not rewritten). The deterministic `NAMLSS` and its shape functions stay in place as baselines.

**Modules to build:**

- **`VariationalDense` (deep module, ADR-0004).** The single atom every Bayesian shape function is built from. Mean-field Normal posterior per weight (`loc` + softplus `scale`); prior specified by serializable config (a `prior_scale` float or a hierarchical-scale handle, never a Python closure); KL emitted via `add_loss`; an internal flipout-style / local-reparameterization estimator flag for variance reduction; a `kl_beta` non-trainable variable the warm-up callback drives. Closure-free `get_config`/`from_config` round-trips hyperparameters (floats/strings); `loc`/`scale` persist as ordinary Keras weights. Interface: `VariationalDense(units, prior_scale|scale_handle, kl_divisor, flipout, activation, use_bias)`. Both the KL-via-`add_loss` propagation (through nested sub-models + `DistributionLambda`) and the save/load round-trip are spike-verified on the target stack (TF 2.15.1 / Keras 2.15 / TFP 0.23.0 / Python 3.11); supported formats are `.keras` and SavedModel.

- **`PriorScale` handle (deep module, ADR-0002).** Resolves one per-feature prior-variance scalar into a KL contribution across three tiers: (a) **fixed** scalar (default, configurable per net in the formula); (b) **empirical-Bayes** learned scale (neural analog of REML smoothness selection, opt-in); (c) **full-hierarchical** with a hyperprior (half-Cauchy default, inverse-gamma for BAMLSS-faithful mode). One prior-variance scalar per additive term = one mgcv λ (penalty ⇔ Gaussian prior duality). Config is closure-free and serializable. The same scalar doubles as the categorical random-effect variance component.

- **`EffectSampler` (deep module).** *Per the split-sampler decision:* the goal-1 workhorse. Given (model, data, T) it draws T weight samples and returns **per-feature contribution samples** only — the input to centered epistemic effect ribbons and interaction surfaces. Pure function; no log-likelihood or IC concerns. Default T = `T_predict` (200).

- **`LogLikSampler` (deep module).** *Per the split-sampler decision:* the goal-3 workhorse, separate from `EffectSampler`. Given (model, data, T) it returns **summed-predictor samples** and **pointwise log-likelihood samples**, and assembles the `tfd.MixtureSameFamily` posterior predictive (across-component = epistemic, within = aleatoric). Pure function; feeds the comparison module and response-level predictive bands. Default T = `T_eval` (1000) when called for IC, `T_predict` for predictive plots. Splitting it from `EffectSampler` keeps the cheap, frequently-run effect-band path independent from the expensive log-likelihood path that IC need at higher T.

  Rationale for the split (chosen over one combined sweep): the two paths have different default T, different callers (interactive plotting vs. rarely-run IC), and different outputs; coupling them forced effect plots to pay the log-likelihood cost or IC to run at plot-grade T. Two workhorses with a shared weight-sampling helper keep each interface narrow and independently testable.

- **Model-comparison module (deep module).** `to_inference_data()` exposes posterior + pointwise log-likelihood (from `LogLikSampler`) as an arviz `InferenceData`. Ships **`waic()`**, **`loo()`** (PSIS-LOO; surfaces Pareto-k reliability warnings), and **`compare()`** (wrapper over `az.compare`). ELBO exposed as a secondary, biased evidence proxy. No literal Bayes Factors (ADR-0001). Pure function of the log-likelihood samples.

- **Bayesian shape functions (registry entries).** `BayesianMLP` — fully variational (every `Dense` → `VariationalDense`). `NeuralLinearMLP` — deterministic hidden basis + variational output layer. Registered by name alongside the deterministic functions; selectable per term in the formula. Each implements the existing `ShapeFunction` contract (`forward` / `build` → `tf.keras.Model`, `output_dimension = family.param_count`). Categorical handling: the first mapping (embedding/lookup → Dense) is variational under the feature's `PriorScale`, giving per-level credible intervals and partial pooling (BAMLSS random-effect analog); a point-embedding fallback exists but is not the default. Internal per-layer dropout is stripped.

- **`BayesianIntercept` layer.** Variational intercept (one location per distributional parameter) with its own wide weakly-informative `Normal(0, σ_int²)` prior, deliberately decoupled from per-feature `prior_scale`. A `point` (deterministic, NAMpy `InterceptLayer`) fallback is retained. Default is Bayesian.

- **`BayesianNAMLSS` model class.** Subclasses the same `AdditiveBaseModel`/`tf.keras.Model` lineage as `NAMLSS`. Owns: a `fit()` override that auto-injects the KL warm-up callback (on by default, length configurable); the `Loss` = mean-NLL + KL/N (N from `DataModule`; per-net variational KL rides `add_loss` auto-propagation); `sample_posterior_predictive(data, T)` delegating to the two samplers; `waic`/`loo`/`compare`/`to_inference_data`; and banded `plot`/`plot_dist`. `call()` stays a single stochastic pass. `feature_dropout` defaults to 0 when Bayesian nets are present. Handles full or partial-Bayesian formulas (a deterministic net = zero-variance contributor).

- **KL warm-up callback.** Sets the `kl_beta` variable on every `VariationalDense` (recursing through nested sub-models) as β = min(1, epoch / warmup_epochs) at epoch start. Auto-injected by the `fit()` override.

- **Banded plotting (extends NAMpy visuals).** Per-feature effect plots: centered, epistemic-only credible ribbon (each posterior-sampled curve mean-centered before quantiles; default 90%, configurable; centering flag-able — deviates from NAMpy's uncentered default). Response-level plots: full predictive band = epistemic + aleatoric (default 90%). Interactions: posterior-mean surface + separate epistemic-SD surface (or sliced 1D bands).

**Interfaces / contracts:**

- Formula syntax unchanged; `:` already resolves to a single joint net over multiple inputs (network name from the first term) — so Bayesian interactions need no parser change.
- `EffectSampler(model, data, T) -> {feature_name: contribution_samples[T, n, param_count]}`.
- `LogLikSampler(model, data, T) -> {summed_samples[T, n, param_count], pointwise_loglik[T, n], predictive: tfd.MixtureSameFamily}`.
- Comparison module consumes `LogLikSampler` output → arviz `InferenceData` → `waic`/`loo`/`compare`.
- `VariationalDense.get_config()` returns only floats/strings/bools (closure-free).

## Testing Decisions

A good test here asserts **external behavior at a module boundary**, not internal wiring: given inputs to a module's public interface, the observable outputs (shapes, values against an independent reference, serialization round-trips, warnings) are correct. We do not assert on private attributes, layer internals, or exact RNG draws (reparameterization noise is not reproducible across distinct layer objects even under a global seed — the spikes already learned this and compare weights/shapes, not stochastic predictions). Prior art: the existing `spikes/` scripts (`spike_kl_propagation.py`, `spike_serialization.py`) establish the pattern — rebuild the real NAMLSS graph shape, assert claims, compare weights elementwise rather than predictions.

Modules to be unit-tested (all four selected):

- **`VariationalDense`** — closed-form Gaussian–Gaussian KL against an independent hand-computed reference; save/load round-trip on `.keras` and SavedModel with `max|Δw| = 0` (H5 explicitly expected to fail/skip); output shape `(batch, units)`; flipout vs. vanilla estimator agree in expectation (mean over many draws) while differing in gradient variance; `get_config` is closure-free.
- **`EffectSampler`** — output dict shape `[T, n, param_count]` per feature; posterior-mean of contributions is T-stable (mean converges, tightening CI with T) on a fixed toy posterior; centering produces zero-mean curves.
- **`LogLikSampler`** — shapes of `summed_samples` / `pointwise_loglik`; `MixtureSameFamily` decomposes correctly on a known toy posterior (across-component variance = injected epistemic, within = family aleatoric); pointwise log-lik matches direct family `log_prob` for a degenerate (single-draw) posterior.
- **Model comparison** — `waic`/`loo` match a hand-computed arviz reference on a tiny fixture of pointwise log-lik; `loo` surfaces the Pareto-k warning at deliberately low T; `compare` ranks two fixtures in the known order.
- **`PriorScale` handle** — each tier (fixed / empirical-Bayes / hierarchical) produces the expected KL contribution against a reference, and each is serializable closure-free.

(The two sampler workhorses are tested independently, consistent with the decision to split them.)

## Out of Scope

- Bayesian variants of NATTLSS / transformer-LSS — v1 covers `BayesianNAMLSS` only; these come later.
- Literal marginal-likelihood Bayes Factors (intractable for NN-sized models; ADR-0001).
- MCMC / non-variational inference engines — mean-field VI is the committed engine (ADR-0001).
- Legacy H5 serialization (weight-name collision; `.keras` and SavedModel are the supported paths).
- Rewriting NAMpy's deterministic `NAMLSS`, formula parser, families, or `DataModule` — these are reused as-is.
- A new training loop — the `compile`/`fit` surface is retained.

## Further Notes

- The two load-bearing TFP-behavior assumptions (KL via `add_loss` propagating through nested sub-models + `DistributionLambda`; closure-free `get_config` enabling save/load) are already empirically verified by the `spikes/` scripts on the exact pinned stack (TF 2.15.1 / Keras 2.15 / TFP 0.23.0 / Python 3.11). The production `VariationalDense` is a superset of the spike seed: it adds the flipout estimator and the `PriorScale` hierarchical handle.
- Target stack: TF ≤ 2.15.1, Keras < 3.0 (tf.keras legacy / Keras 2), TFP ≤ 0.23, Python 3.9–3.11. `uv`-managed `.venv-spike` already provisions Python 3.11.15 with the pinned versions.
- Governing decisions: ADR-0001 (mean-field VI, no Bayes Factors), ADR-0002 (per-feature/hierarchical priors ≈ smoothness), ADR-0003 (KL/N + warm-up on the compile/fit surface), ADR-0004 (in-house `VariationalDense`), ADR-0005 (categoricals as random effects, interactions as joint nets). Full glossary in `CONTEXT.md`.
