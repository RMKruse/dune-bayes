# Comparator Scope Decision

GitHub issue: #145
Status: accepted for the first DUNE paper submission

## Decision

Do not require full external NAMLSS, LA-NAM, or BAMLSS/R comparator evidence
before the first submission. Narrow the benchmark claim to the promoted
canonical UCI panel and its documented exclusions.

The UCI panel is characterization evidence unless promoted results support a
stronger predictive claim. Under this decision, the benchmark table may describe
how DUNE behaves on the ten-dataset UCI panel against the scored canonical
comparators, but it must not claim universal predictive dominance.

## Manuscript Benchmark Wording

Use this wording, or a tighter edit with the same scope:

> On a ten-dataset UCI characterization panel, DUNE is evaluated against the
> promoted canonical comparators: the full BayesianNAMLSS implementation,
> a BayesNAM-style degenerate configuration of the same package, a plain MLP,
> and a deep ensemble. These results characterize predictive fit, calibration,
> and uncertainty decomposition under a shared scoring harness; they are not a
> claim of universal predictive dominance. External NAMpy/NAMLSS, LA-NAM, and
> BAMLSS/R comparators are documented exclusions for this submission rather than
> promoted canonical evidence.

## Excluded External Comparators

- NAMpy/NAMLSS: the TensorFlow-era NAMpy/NAMLSS comparator is not promoted as
  full canonical evidence because its optional environment remains outside the
  PyTorch package runtime. The live process boundary is covered by the promoted
  NAMLSS smoke tracer, but smoke evidence is not used as a paper benchmark
  result.
- LA-NAM: LA-NAM is not promoted as full canonical evidence because its optional
  third-party environment was not enabled for the canonical run. The dependency
  instructions remain documented for future reruns, but the first submission
  does not cite LA-NAM scores.
- BAMLSS/R: BAMLSS/R is not promoted as full canonical evidence because
  maintainer-produced fixtures are not available for the full ten-dataset panel.
  The committed Auto MPG fixture keeps the scoring seam tested, but it is not
  generalized into a paper-wide BAMLSS/R comparison.

## Runtime Boundary

TensorFlow-era NAMLSS, LA-NAM, BAMLSS/R, JAX, and NumPyro remain experiments-only
or future-backend concerns; they do not become package runtime dependencies for
`dune_bayes`. The paper may discuss those systems as related work, validation
context, optional external comparators, or future directions, but package
installation and import remain PyTorch-only for the submitted artifact.

## Follow-Up Issues

No first-submission follow-up implementation issues are required under this
decision. If reviewer feedback or a later paper revision requires full external
comparator evidence, create one implementation issue per comparator and name the
dataset scope explicitly:

- NAMpy/NAMLSS on the ten-dataset UCI panel.
- LA-NAM on the ten-dataset UCI panel, with pinned source and license status.
- BAMLSS/R on the ten-dataset UCI panel, with seeded fixtures or a committed
  regeneration script.
