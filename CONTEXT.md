# Context: dune-bayes

**DUNE — Distributional Uncertainty in Neural-additive Estimation** builds on
the **NAMLSS** framework ([Thielmann et al., 2024](https://proceedings.mlr.press/v238/frederik-thielmann24a.html)
— Neural Additive Models for Location Scale and Shape) and makes it Bayesian.
It is being built by taking the `NAMpy` package (the deterministic NAMLSS
reference implementation) and replacing the deterministic feature networks
with **Bayesian** feature networks, so that each feature's effect on each
distributional parameter carries epistemic uncertainty. The Bayesian treatment
of distributional regression follows the spirit of the **BAMLSS** R package
(Bayesian Additive Models for Location, Scale and Shape).

> **Compute backend (ADR-0006).** The package targets **PyTorch**. NAMpy was
> written on TensorFlow / Keras 2 / TensorFlow-Probability (TFP); that stack is
> **legacy** and is being ported. The binding dependency was TFP (distributions,
> `DistributionLambda`, `MixtureSameFamily`), which is also the most stagnant part
> of the old stack — hence the switch. **JAX is the designated numerical future:**
> the deferred MCMC backend (ADR-0001) and performance work land on
> **NumPyro/BlackJAX**, slotting in behind the same distribution/inference seam.
> The statistical design below is backend-agnostic; where it names a TFP/Keras API
> (`tfd.*`, `DistributionLambda`, `add_loss`, `.keras` save), read the PyTorch
> equivalent (`torch.distributions.*`, a module returning a distribution, explicit
> KL collection, `state_dict`). See ADR-0006 for the full mapping.

## Goals (in priority order)

1. **Epistemic uncertainty on effects** — credible intervals around each
   feature's shape function / effect curve. This is the headline interpretability payoff.
2. **Priors as principled regularization** — especially valuable on small datasets.
3. **Principled model comparison** — a defensible Bayesian criterion across a
   handful of candidate formulas. WAIC / LOO-CV (and the ELBO as a secondary
   evidence proxy) are accepted; literal marginal-likelihood Bayes Factors are
   NOT a hard requirement (they are intractable for NN-sized models — see ADR-0001).

## Packaging

**dune-bayes is its own package**, reusing NAMpy's machinery by design — but the
reuse splits by backend (ADR-0006). NAMpy stays on TF/Keras/TFP and is **not
rewritten**:

- **Reused directly** (TF-free Python): the **formula parser** and the
  **`ShapeFunctionRegistry`** pattern carry no TF dependency.
- **Reimplemented in PyTorch** (cannot cross the backend boundary): **families**
  (`tfd.*` → `torch.distributions.*`), the **`DataModule`** (`tf.data` →
  `torch.utils.data` + numpy/sklearn preprocessing), **shape functions** (Keras →
  `nn.Module`), and **plotting** as needed.

The repo name reflects this new identity. The deterministic NAMpy shape functions
are retained as baselines (reached as the external TF package for comparison, not
ported into the PyTorch tree).

## Model class & training UX

The user instantiates a dedicated **`BayesianNAMLSS`** class (dune-bayes),
which owns posterior sampling, the `MixtureSameFamily` predictive, credible-band
plotting, and the WAIC/LOO methods. It handles **full or partial-Bayesian**
formulas (a deterministic net is a degenerate zero-variance contributor). The
deterministic `NAMLSS` stays untouched. v1 covers `BayesianNAMLSS`; Bayesian
variants of NATTLSS/transformer-LSS come later.

Training keeps the familiar **`compile(loss=model.loss)` / `fit()`** surface:
KL/N is added automatically via `add_loss`, and the KL warm-up callback is
auto-injected by a `fit()` override (on by default). New capabilities
(`sample_posterior_predictive`, `waic`/`loo`/`compare`, banded `plot`) are
separate methods. See ADR-0003.

## Model comparison

Built on **arviz**: `to_inference_data()` exposes posterior + pointwise
log-likelihood; ship **WAIC** and **PSIS-LOO** (with Pareto-k reliability
diagnostics) plus a `compare()` wrapper over `az.compare`. The **ELBO** is a
secondary, biased evidence proxy. No literal Bayes Factors (ADR-0001).

## Glossary

- **Shape function** — the per-feature network (today a small `tf.keras.Model`:
  MLP, ResNet, CubicSplineNet, …) that maps one feature (or an interaction) to a
  contribution for *each* distributional parameter. Output dimension equals the
  family's `param_count`. Registered by name and selected via the mgcv-style
  formula string, e.g. `y ~ MLP(x1) + MLP(x2):MLP(x3)`.
- **Bayesian shape function** — a shape function whose weights have a variational
  posterior instead of point values. Delivered as **new registry entries**
  (e.g. `BayesianMLP`) that live *alongside* the deterministic ones, selectable
  per-feature in the formula (`y ~ BayesianMLP(x1) + MLP(x2)`). Keeping the
  deterministic baselines is deliberate: it enables direct Bayes-vs-deterministic
  effect comparison and lets WAIC/LOO compare against a non-Bayesian baseline.
- **Family** — the response distribution (Normal, Poisson, Gamma, …) whose
  parameters the additive predictor targets. Supplies `param_count`, parameter
  link/transforms, and the log-likelihood. The family models **aleatoric**
  uncertainty; the Bayesian weights add **epistemic** uncertainty on top.
- **Aleatoric vs epistemic uncertainty** — aleatoric = irreducible noise in the
  response, captured by the family. Epistemic = uncertainty about the learned
  effects, captured by the posterior over weights. dune-bayes models both.
- **Inference engine** — the method that turns priors + likelihood into a
  posterior. Decided: **mean-field variational inference** (see ADR-0001).
- **ELBO** — evidence lower bound; the training objective = NLL + weight-KL/N.
  Doubles as a (biased) model-evidence proxy.
- **`VariationalDense`** — the single in-house atom every Bayesian shape function
  is built from: a thin **PyTorch `nn.Module`** with a mean-field Normal weight
  posterior (`loc` + softplus `scale`), a prior set by **serializable config** (a
  `prior_scale` float or a hierarchical-scale handle, never a closure), KL
  collected explicitly by walking the module tree, and an internal flipout-style
  estimator flag for variance reduction. Owning the atom (rather than a stock
  library layer) is what gives flexible per-feature/hierarchical priors (ADR-0002)
  **and** variance reduction **and** clean save/load (`state_dict` + config dict —
  no weight-name-collision failure mode). On the legacy TF stack this layer was a
  Keras `Layer` chosen over raw `tfp.layers.DenseVariational` / `DenseFlipout`, and
  the KL-via-`add_loss` and `.keras`/SavedModel save/load claims were
  spike-verified on TF 2.15.1 / Keras 2.15 / TFP 0.23; the spikes have since been
  **ported to PyTorch and both claims re-verified green** (module-walk KL across the
  NAMLSS graph; `state_dict` + config save/load with `max|Δw| = 0`; torch 2.12 /
  Python 3.12). See ADR-0004, ADR-0006, and `spikes/`.
- **`BayesianMLP`** — the flagship Bayesian shape function: **fully variational**
  (every `Dense` → `VariationalDense`), so uncertainty propagates through
  the function *shape*, not just a final rescaling. Serves goal 1 best; shape
  nets are tiny so the cost is acceptable.
- **`NeuralLinearMLP`** — last-layer-only Bayesian companion (deterministic
  hidden basis, variational output). Cheaper, more stable; the fallback for the
  large-data end of the scale range and for quick baselines.
- **Posterior predictive** — represented as a `tfd.MixtureSameFamily` over `T`
  weight-sampled family distributions. Spread *across* components = epistemic,
  spread *within* a component = aleatoric. Built by the
  `sample_posterior_predictive(data, T)` workhorse, which in one sweep yields:
  per-feature contribution samples (effect bands, goal 1), summed-predictor
  samples, and pointwise log-likelihood samples (WAIC/LOO, goal 3). `call()`
  stays a single stochastic pass (one MC sample of the ELBO per training step).
- **MC sample counts (`T_predict` / `T_eval`)** — the number of posterior weight
  draws is split by consumer, because information criteria are far more sensitive
  to low `T` than plots are. **`T_predict = 200`** is the default for
  `sample_posterior_predictive`, effect-band plots, and the `MixtureSameFamily`
  predictive (cheap, smooth-enough bands, fast interactive loop).
  **`T_eval = 1000`** is the internal default for `waic()` / `loo()` / `compare()`
  (the `log Σ exp` over draws and PSIS importance weights are biased/noisy at small
  `T`); IC are computed rarely, so the 5× cost is acceptable. Both are explicit,
  overridable arguments — the split is only about defaults. `loo()` surfaces
  arviz's Pareto-k / reliability warning when `T_eval` was effectively too low.
- **KL/N + warm-up** — training loss = mean-NLL + KL/N (N = training-set size,
  from the `DataModule`); per-feature-net variational KL rides along via Keras
  `add_loss` auto-propagation. A **KL warm-up** (β: 0→1 over the first epochs) is
  **on by default**, warm-up length configurable, to guard against posterior
  collapse.
- **Dropout interaction** — Bayesian shape functions **strip the internal
  per-layer dropout** (the weight posterior is the stochasticity) and **default
  `feature_dropout=0`** when Bayesian nets are present, so epistemic uncertainty
  isn't conflated with dropout noise. Both remain **fully user-configurable** for
  users who want dropout-style Bayesian nets anyway.
- **Effect plot vs response plot (band-by-view)** — *per-feature effect plots*
  show a **centered, epistemic-only** credible ribbon (each posterior-sampled
  curve mean-centered over the data before quantiles; isolates shape uncertainty;
  default 90%, configurable; centering is flag-able, deviates from NAMpy's
  uncentered default). *Response-level plots* (`plot_dist` / predicted-vs-actual)
  show the **full predictive band = epistemic + aleatoric** (the proper
  prediction interval; default 90%). Aleatoric is a response-level property and
  is deliberately NOT attributed to individual feature curves.
- **Bayesian intercept** — the global additive anchor (NAMpy's `InterceptLayer`,
  one bias per distributional parameter added to the summed predictor) is
  **variational by default**, with its **own wide / weakly-informative
  `Normal(0, σ_int²)` prior** — deliberately *not* tied to the per-feature
  `prior_scale` (it is a location, not a smoothness/shrinkage term, so it must not
  be shrunk toward zero). Rationale: because effect plots mean-center each
  posterior-sampled curve into the intercept (see *Effect plot vs response plot*),
  the intercept is where the absorbed overall level — and its uncertainty —
  accumulates; on small data that level-uncertainty is often the dominant
  epistemic term, so a point intercept would under-cover the response-level mean.
  A deterministic **`point`-intercept** fallback is retained.
- **Interactions** — the formula `:` syntax (e.g. `BayesianMLP(x1):BayesianMLP(x2)`)
  resolves to a **single joint net over multiple inputs**, not a product of nets
  (the network name comes from the first term; the rest are added as inputs —
  `nampy/formulas/formulas.py:223`). A Bayesian interaction therefore needs **no
  new machinery** — it is just a Bayesian shape function with a multi-feature
  input, and `sample_posterior_predictive` handles it as-is. Only the plot
  changes: an interaction renders as a posterior-mean **surface** plus a separate
  **epistemic-SD surface** (or sliced 1D bands), since a 1D ribbon doesn't
  generalize. See ADR-0005.
- **Categorical effect = random effect** — a categorical feature's first mapping
  (embedding / lookup → Dense) is **Bayesian**, like every other weight, under the
  per-feature `prior_scale`. This gives a **credible interval per level** and
  shrinks **rare levels toward the prior** — i.e. partial pooling, the neural
  analog of a **BAMLSS random effect** (shared variance-component / IG prior on
  categorical effects). The per-feature `prior_scale` thus doubles as the
  random-effect variance component. A point-embedding fallback exists but is not
  the default (it makes rare levels look falsely confident). See ADR-0005.
- **Prior scale ≈ smoothness** — each feature net carries **one** prior-variance
  scalar, the direct analog of one mgcv smoothing parameter λ per additive term
  (penalty ⇔ Gaussian prior). Fixed by default (configurable per net in the
  formula); opt-in **empirical-Bayes** learning of that scale is the neural
  analog of REML smoothness selection; full-hierarchical (half-Cauchy default,
  inverse-gamma for BAMLSS-faithful mode) is the advanced tier. See ADR-0002.
