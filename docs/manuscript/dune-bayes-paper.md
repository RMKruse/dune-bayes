# DUNE: Distributional Uncertainty In Neural-Additive Estimation

Status: manuscript scaffold for the first dune-bayes paper. The claim ledger
for this source lives at `docs/manuscript/claim-ledger.yaml` and resolves to the
promoted evidence manifest at `experiments/publication/evidence-manifest.yaml`.

## Introduction

DUNE provides per-feature, per-parameter epistemic uncertainty in
BayesianNAMLSS models and separates it from aleatoric uncertainty supplied by
the response family. The central thesis is that posterior draws over each shape
function make uncertainty about learned effects visible, while the family
distribution retains irreducible response noise; the response-level variance
decomposition then connects these two components through the law of total
variance.

The paper will frame this as uncertainty disentanglement for distributional
neural additive models, not as a predictive leaderboard claim. The main reader
promise is interpretability with uncertainty attached to each additive
contribution for each distributional parameter.

## Related Work

This section will position DUNE as a Bayesian extension of NAMLSS rather than a
new name for an unrelated model family. It should cover NAMLSS, Bayesian
additive models for location, scale, and shape, Bayesian neural additive models,
distributional regression, posterior predictive uncertainty, and model
comparison with WAIC and PSIS-LOO.

Comparator language should follow the publication evidence manifest: NAMpy
NAMLSS, LA-NAM, and BAMLSS are external comparators or documented exclusions,
not runtime dependencies. The benchmark discussion should distinguish
predictive scoring from DUNE's main uncertainty-structure contribution.

## Model And Methods

### Bayesian Distributional Additive Model

DUNE fits the package's `BayesianNAMLSS`, a Bayesian distributional extension of
NAMLSS. For observations `(x_i, y_i)`, a response family `F` supplies `K` family
parameters, and each additive term is a shape function that contributes to every
family parameter. The runtime implementation is PyTorch, with the deferred
JAX/NumPyro MCMC seam kept outside the v1 package backend (ADR-0006). On the
predictor scale,

```text
eta_{ik}(theta) = alpha_k(theta) + sum_{m=1}^M f_{mk}(x_{im}; theta_m),
    k = 1, ..., K,
y_i | theta ~ F(eta_i(theta)).
```

Here `f_m` may be a single-feature shape function, an interaction shape function
over several inputs, or a categorical effect. Bayesian shape functions replace
deterministic weights with variational weights, so epistemic uncertainty is
uncertainty over the learned shape effects and intercept, while aleatoric
uncertainty remains the variance implied by the response family.

### Mean-Field Variational Objective

The v1 inference engine is mean-field variational inference (ADR-0001). Each
`VariationalDense` weight or bias has a Normal variational factor,

```text
q_phi(theta) = product_l Normal(mu_l, softplus(rho_l)^2),
```

and the corresponding prior term is a serializable Gaussian prior, optionally
with a per-feature prior-scale handle (ADR-0002 and ADR-0004). The negative ELBO
optimized during training is

```text
L(phi) =
    (1 / |B|) sum_{i in B} -log p_F(y_i | eta_i(theta_phi))
    + beta / N * KL(q_phi(theta) || p(theta)).
```

`N` is the full training-set size, not the minibatch size, so the KL/N scaling is
stable under minibatching. `beta` is the KL warm-up factor used during early
epochs; after warm-up the objective is the usual negative ELBO with the
mean-field posterior and prior terms above.

### Training Draws Versus Posterior Draws

Training uses local reparameterization (ADR-0007): for a variational dense
layer, the minibatch pre-activation is sampled from its marginal Normal
distribution using per-row noise. This reduces ELBO-gradient variance, but it
does not materialize a single weight matrix shared across the data.

Posterior predictive, effect, and log-likelihood sampling therefore use
coherent global posterior weight draws instead. For draw `t`, one global
`theta^(t) ~ q_phi(theta)` is applied to all observations, all feature shape
functions, and all family parameters before the next draw is taken. This
coherence is required for smooth effect draws and for the epistemic/aleatoric
interpretation of the posterior predictive mixture.

### Posterior Predictive Mixture And Variance Decomposition

For `T` coherent posterior draws, DUNE represents the posterior predictive as a
uniform mixture of family distributions,

```text
p(y_i | D) ~= (1 / T) sum_{t=1}^T p_F(y_i | eta_i(theta^(t))).
```

Spread across mixture components is epistemic uncertainty because it comes from
different posterior weight draws. Spread within each component is aleatoric
uncertainty because it is the response-family variance conditional on one draw.
The paper's uncertainty-disentanglement metric is the law of total variance:

```text
Var(y_i | D)
  ~= E_t[ Var_F(y_i | theta^(t)) ]
     + Var_t[ E_F(y_i | theta^(t)) ].
```

The first term is the aleatoric component and the second is the epistemic
component. The implementation computes this variance decomposition from each
draw's family mean and family variance, surfacing infinite family variance as
infinite rather than clamping it.

### Effect Ribbons And Response-Level Bands

Effect ribbons summarize posterior draws of a single shape function for one
family parameter. Each draw is centered over the plotting data before quantiles
are taken, so effect centering removes arbitrary additive level and leaves an
epistemic-only ribbon around the shape. Because centering moves level
uncertainty into the additive anchor, intercept coverage is assessed separately
in simulation evidence.

Response-level bands answer a different question. They are full predictive
intervals drawn from the posterior predictive mixture, so they include both
epistemic spread across posterior components and aleatoric spread within each
family component. Aleatoric response noise is deliberately not attributed to
individual feature curves.

### Families And Numerical Stability

Family parameterizations follow the package contract: Normal, Gamma, Student-t,
Johnson's SU, Negative Binomial, and Beta families expose a fixed
`param_count`, predictor-to-parameter links, `log_prob`, and defined mean and
variance behavior for the decomposition. Positive scale, rate, concentration,
and dispersion parameters use `softplus(x) + EPS`; the additive `EPS` is the
numerical floor that prevents bare softplus underflow at extreme negative
pre-link values. The method does not use `exp`, positive `transform_to`, or
post-hoc clamps for learned positive quantities.

All likelihood and information-criterion accumulation stays in log-space:
training uses `log_prob`, WAIC uses `logsumexp` across posterior draws, and
pointwise log-likelihood matrices are accumulated in float64 for evaluation.
Finite extreme pre-link checks at `+/-1e4` are part of the documented numerical
contract (ADR-0008).

### Evaluation And Exclusions

Model comparison is predictive and posterior-simulation based (ADR-0003). WAIC
uses the WAIC2 expected log pointwise predictive density with log-space
accumulation; PSIS-LOO is computed through ArviZ and reports Pareto-`k`
reliability diagnostics; `compare()` ranks candidate formulas by the LOO result.
CRPS scores predictive samples, PIT calibration evaluates predictive CDF values
(with randomized PIT for discrete responses when requested), and coverage
reports central credible-band coverage for effect or parameter draws in the
simulation studies.

The ELBO is retained only as a biased secondary evidence proxy. The paper computes
no Bayes factors: literal Bayes factors would require tractable marginal
likelihood estimates that are out of scope for the v1 mean-field neural model.

## Experiments

The experiment section is organized by the manuscript claim ledger, not by
scratch run folders. Simulation evidence covers the central disentanglement
demonstration, parameter-recovery studies across families, and the Johnson's SU
showcase. Validation evidence covers VI versus NUTS as a limitation study for
mean-field narrowness. Real-data benchmark evidence covers the UCI
characterization panel and comparator policy.

Planned manuscript outputs:

- Figure: central variance decomposition in dense/noisy versus sparse/quiet
  regions.
- Figure/table: per-family, per-parameter epistemic bands and coverage summaries.
- Table/figure: VI-versus-NUTS band-width and interval comparison.
- Table: UCI real-data characterization with comparator exclusions stated.

### UCI Benchmark Characterization

The UCI benchmark is a ten-dataset characterization panel, not a predictive
ranking exercise.
The promoted full evidence lives at
`experiments/uci_benchmark/results/canonical` and is the source for
`tables/benchmark-comparator-panel__comparison.csv` plus the per-dataset NLL,
CRPS, calibration, and variance-decomposition tables. The characterization table
is generated from promoted canonical evidence, not the NAMLSS smoke tracer and
not scratch runs under `experiments/uci_benchmark/runs`. This is not a claim of
universal predictive dominance.

The canonical benchmark gate declares these dataset/family pairs: autompg
(normal), concrete (normal), energy (normal), kin8nm (normal), naval (beta after
the open-unit response transform), power (normal), protein (normal), wine
(normal), yacht (normal), and bike (negative binomial). Every comparison row is
scored through the shared predictive scoring metrics NLL, CRPS, and PIT
calibration. Those metrics characterize real-data predictive fit and calibration;
the paper's main uncertainty-structure contribution remains per-feature,
per-parameter epistemic shape-function bands and the response-level
epistemic/aleatoric variance decomposition.

The promoted canonical comparators are the full BayesianNAMLSS implementation in
dune-bayes, a BayesNAM-style degenerate configuration of the same package, a
plain MLP, and a deep ensemble. The BayesNAM-style row is not a canonical
external implementation; it is an in-package location-only variational baseline
that makes the mean-only comparison visible under the same scoring harness.
Table rows should preserve the uncertainty scope labels
`distributional_parameter_bands`, `mean_only_variational_location`, and
`predictive_only` so readers do not mistake a predictive-only baseline for a
distributional shape-function method.

External NAMpy/NAMLSS, LA-NAM, and BAMLSS/R are documented exclusions for this
submission rather than promoted canonical evidence. NAMpy/NAMLSS remains
optional TensorFlow-era process-boundary code; the promoted live NAMLSS smoke
tracer proves the boundary, but it is not cited as a paper benchmark result.
LA-NAM was not enabled for the canonical run. BAMLSS/R fixtures are not
available for the full ten-dataset panel, beyond the Auto MPG fixture that tests
the scoring seam. These exclusions should be explained in reviewer-facing
language rather than hidden in table footnotes.

Every number, figure, or table should be generated from promoted artifacts under
`experiments/*/results/canonical` through the paper artifact builder, never from
`experiments/*/runs` scratch output.

## Limitations

The paper should be explicit that mean-field VI can under-cover posterior
uncertainty; the validation-only NUTS study quantifies this limitation but does
not make NUTS a shipped inference backend. The benchmark panel characterizes
real-data behavior and comparator context, but it should not claim universal
predictive dominance unless the promoted table supports that claim.

Other limitations to keep visible: no Bayes factors, no post-hoc recalibration
or band inflation, no runtime TensorFlow dependency, no shipped HMC backend, and
optional external comparator gaps documented as exclusions when canonical
evidence is not promoted.

## Reproducibility

The reproducibility spine is the promoted evidence manifest, benchmark
publication gate, paper artifact builder, and bounded audit. Each major claim in
`docs/manuscript/claim-ledger.yaml` resolves to promoted evidence, evidence
class, intended table or figure output, and a limitation note.

The bounded audit should report package checks, smoke experiment paths,
publication evidence validation, benchmark gate status, artifact assembly, and
dependency readiness. Full canonical reruns remain manual and should be labelled
as such in the paper artifact.

## Citation And Artifact Notes

The citable artifact should point to the release metadata, citation file, DOI
deposit plan, promoted canonical results, reviewer evidence appendix, and
generated paper artifacts. Citation text should remain pending until the
author-approved preprint or paper citation exists.

The manuscript should cite the archived release artifact, not transient local
state. When the artifact is frozen, the manuscript, release tag, DOI-backed
archive, evidence manifest, and generated figures/tables should refer to the
same commit.
