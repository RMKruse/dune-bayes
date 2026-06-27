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

The model is the package's `BayesianNAMLSS`: an additive distributional model
whose shape function terms contribute to every response-family parameter. The
paper should reuse the project vocabulary directly: a shape function is the
per-feature or interaction network, epistemic uncertainty is uncertainty over
its learned effect, aleatoric uncertainty is family variance, and variance
decomposition is computed from posterior predictive draws.

Methods terminology is ADR-backed. Mean-field VI is the v1 inference engine
(ADR-0001), per-feature prior scales act as smoothness controls (ADR-0002),
posterior predictive sampling and WAIC/PSIS-LOO follow the comparison
architecture in ADR-0003, `VariationalDense` is the in-house variational atom
(ADR-0004), the runtime backend is PyTorch with JAX/NumPyro deferred behind the
future seam (ADR-0006), and training uses local reparameterization while
posterior sampling uses coherent global weight draws (ADR-0007).

Numerical stability should be described as part of the method, not as an
implementation footnote. Family scale and variance links use `softplus(x) +
EPS`, log-likelihood and information-criterion accumulation stay in log-space,
and finite extreme pre-link checks are part of the reproducibility contract
recorded in ADR-0008 and the project numerical rules.

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
