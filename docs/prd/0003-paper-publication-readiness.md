# PRD 0003: Paper publication readiness — evidence freeze, release, submission

Status: Draft · Date: 2026-06-23

This PRD assumes PRD 0001 and PRD 0002 have been implemented or otherwise
handled. Its scope is the final publication layer: turning dune-bayes from a
paper-grade implementation into a reviewer-auditable, reproducible, citable
research artifact.

## Problem Statement

I have implemented the package capabilities, numerical hardening, metrics,
families, experiments, and baseline harness needed for a dune-bayes paper. The
remaining risk is no longer primarily statistical machinery; it is publication
readiness. I need to turn a broad implementation into a frozen evidence package
whose claims, figures, tables, configs, seeds, comparator runs, and release
metadata are all traceable.

Without this final layer, the paper can still fail in review for avoidable
reasons: a table may be assembled from an unpromoted scratch run, a figure may
not be regenerable from the committed config, a comparator may only have a smoke
artifact where the paper claims a full-panel result, the exact software version
may be unclear, CI may not run automatically on the public repo, or the README
may still describe the project as pre-paper research software. The core
scientific story — uncertainty disentanglement in distributional neural additive
models — needs an audit trail as careful as the ELBO and family numerics.

## Solution

Create a paper-publication readiness track for dune-bayes. The track freezes the
paper claims, validates every claim against promoted canonical artifacts, builds
paper tables and figures from those artifacts, runs a clean reproducibility
audit, promotes the full benchmark/comparator panel, and prepares a citable
research-software release.

The result is a submission-ready bundle:

- a claim-to-evidence ledger for the paper's main scientific statements;
- a single artifact build path for all paper-facing figures and tables;
- full canonical experiment results for simulations, HMC validation, and the
  UCI/comparator panel;
- an explicit reproducibility report from a clean environment;
- release metadata, citation metadata, and documentation updated for the paper;
- a reviewer-facing appendix or reproducibility note explaining exactly how the
  evidence was generated.

This PRD deliberately does not reopen the inference engine, variational family,
family tiers, metrics, or baseline design decisions. Those are governed by the
existing ADRs and PRDs. This track is about evidence discipline, publication
packaging, and reviewer trust.

## User Stories

1. As a paper author, I want the paper's main claims frozen before final runs,
   so that the results are interpreted against pre-declared evidence targets.
2. As a paper author, I want every major claim linked to a promoted canonical
   artifact, so that reviewers can trace statements back to data.
3. As a paper author, I want the disentanglement claim linked to the variance
   decomposition experiment, so that the paper's central result is directly
   auditable.
4. As a paper author, I want the per-feature, per-parameter epistemic-band claim
   linked to parameter-recovery and Johnson's SU evidence, so that the novel
   contribution is supported beyond a mean-only example.
5. As a paper author, I want the mean-field VI limitation linked to the
   VI-vs-NUTS agreement study, so that under-coverage is quantified rather than
   hand-waved.
6. As a paper author, I want the UCI/comparator claim linked to full-panel
   canonical runs, so that benchmark statements do not depend on smoke
   artifacts.
7. As a paper author, I want the original NAMLSS comparator run live through the
   common harness on shared splits, so that the Bayesian-vs-deterministic
   comparison is not based on printed historical numbers.
8. As a paper author, I want the LA-NAM comparator either enabled and scored or
   explicitly marked unavailable with a documented reason, so that the closest
   Bayesian additive baseline is handled honestly.
9. As a paper author, I want BAMLSS fixtures generated, committed, and scored
   where the paper claims BAMLSS comparison, so that the stats-world baseline is
   reproducible.
10. As a paper author, I want the BayesNAM-style degenerate dune-bayes baseline
    clearly labeled, so that readers do not mistake it for a canonical external
    implementation.
11. As a paper author, I want plain MLP and deep ensemble sanity floors in the
    final comparison table, so that conventional neural baselines are visible.
12. As a paper author, I want all comparator models scored by the same metrics
    package, so that NLL, CRPS, and calibration are not confounded by
    model-specific evaluation code.
13. As a paper author, I want all paper tables generated from machine-readable
    result files, so that numbers are never copied by hand.
14. As a paper author, I want all paper figures generated from promoted
    artifacts, so that final figures can be regenerated after review comments.
15. As a paper author, I want figure captions to name the uncertainty component
    being shown, so that epistemic effect bands are not confused with response
    predictive intervals.
16. As a paper author, I want coverage tables to state that coverage is measured
    rather than asserted correct, so that the mean-field VI narrowness doctrine
    is preserved.
17. As a paper author, I want centered effect recovery and intercept coverage
    reported separately, so that additive non-identifiability does not pollute
    coverage claims.
18. As a paper author, I want the law-of-total-variance decomposition reported
    with both aleatoric and epistemic components, so that the thesis is expressed
    in the same terms as the package metric.
19. As a paper author, I want heavy-tail infinite-variance behavior documented
    where relevant, so that the paper does not imply clamped aleatoric variance.
20. As a paper author, I want all family parameterizations stated consistently,
    so that Johnson's SU, Beta, NegativeBinomial, Gamma, StudentT, and Normal
    results are interpretable.
21. As a paper author, I want the methods section assembled from ADR-backed
    decisions, so that the text cannot drift from the implementation.
22. As a paper author, I want a release tag tied to the submitted paper version,
    so that citations resolve to the exact code used for results.
23. As a paper author, I want archived code and promoted results deposited with a
    DOI, so that the artifact is citable and stable.
24. As a paper author, I want citation metadata added before release, so that
    GitHub, Zenodo, and users know how to cite dune-bayes.
25. As a paper author, I want the README citation section updated after the
    preprint exists, so that the repo no longer says there is no dune-bayes
    paper.
26. As a maintainer, I want CI enabled on public push and pull request events,
    so that the public repository advertises the same correctness discipline as
    the paper.
27. As a maintainer, I want experiment smoke tests to keep running separately
    from full experiments, so that CI remains bounded while experiment CLIs do
    not rot.
28. As a maintainer, I want a clean reproducibility report from a fresh clone, so
    that I know a reviewer can rebuild the environment.
29. As a maintainer, I want the lockfile and dependency groups checked against
    the paper instructions, so that the reproducibility path is not missing
    optional experiment dependencies.
30. As a maintainer, I want full canonical runs kept separate from scratch runs,
    so that accidental unreviewed artifacts are not cited.
31. As a maintainer, I want every promoted run to include config, seed, run
    metadata, metrics, arrays, and figures as applicable, so that artifacts are
    complete.
32. As a maintainer, I want an evidence validator to fail on missing or stale
    canonical artifacts, so that publication readiness is mechanically checked.
33. As a maintainer, I want the validator to distinguish smoke artifacts from
    paper evidence, so that benchmark claims cannot accidentally cite CI-scale
    runs.
34. As a maintainer, I want artifact hashes or timestamps recorded in a manifest,
    so that later edits to figures or metrics are visible.
35. As a maintainer, I want the paper artifact builder to produce deterministic
    output paths, so that LaTeX, Markdown, or submission files can reference
    stable filenames.
36. As a maintainer, I want a small summary table of all experiments and their
    canonical status, so that release readiness can be reviewed quickly.
37. As a maintainer, I want optional comparator prerequisites documented, so that
    TensorFlow-era NAMLSS, LA-NAM, and BAMLSS do not leak into the package
    runtime.
38. As a maintainer, I want unavailable optional baselines to fail loudly when
    enabled, so that silent omission cannot change comparison tables.
39. As a maintainer, I want the public package API documentation to match the
    paper examples, so that readers can reproduce the headline workflow.
40. As a maintainer, I want no new TensorFlow runtime imports in dune-bayes, so
    that the PyTorch backend decision remains intact.
41. As a reviewer, I want to know which results are simulation evidence and which
    are real-data benchmark evidence, so that I can judge the claims properly.
42. As a reviewer, I want to see the exact seeds and split protocol, so that I can
    evaluate robustness and reproducibility.
43. As a reviewer, I want comparison metrics defined once, so that I can trust
    that CRPS, NLL, and PIT calibration mean the same thing for every model.
44. As a reviewer, I want clear statements of what mean-field VI can and cannot
    claim, so that the uncertainty bands are not oversold.
45. As a reviewer, I want the HMC validation to be scoped as validation-only, so
    that I do not expect a shipped MCMC backend.
46. As a reader, I want a short tutorial that reproduces the core formula-fit-band
    workflow, so that I can try the method without reading the whole paper first.
47. As a reader, I want the difference between effect ribbons and response bands
    documented, so that I interpret epistemic and aleatoric uncertainty
    correctly.
48. As a reader, I want the release notes to identify the paper artifact version,
    so that I can distinguish research-code evolution from the submitted result.
49. As a future contributor, I want publication-only scripts isolated from the
    package runtime, so that paper assembly does not become a package dependency.
50. As a future contributor, I want deferred nice-to-haves recorded separately, so
    that the first paper does not grow into an open-ended methods expansion.

## Implementation Decisions

- This PRD creates a publication-readiness layer. It does not change the
  statistical model, inference engine, family contract, posterior-sampling
  contract, or metric definitions unless a reproducibility audit exposes a bug.
- The primary deep module is an evidence manifest and validator. The manifest
  records each paper claim, its canonical experiment artifact, the expected
  artifact class, whether the artifact is smoke or full evidence, and the
  minimal metadata needed to audit it. The validator exposes a narrow interface:
  load the manifest, inspect the promoted artifacts, and return a pass/fail
  readiness report with actionable missing items.
- A paper artifact builder consumes promoted results and writes the final
  submission tables and figures. It does not refit models and does not read
  scratch runs. Its interface is deliberately simple: consume canonical evidence,
  write derived artifacts, and emit provenance.
- The full UCI/comparator panel is treated as publication-blocking if the paper
  makes benchmark claims. Smoke runs remain useful CI tracers, but they are not
  acceptable paper evidence except when explicitly labeled as smoke-only
  validation.
- The comparator harness remains the single scoring route for NLL, CRPS, and PIT
  calibration. Optional external baselines run in isolated environments or from
  committed fixtures, preserving the PyTorch/Python 3.12 runtime boundary.
- Canonical promotion remains deliberate: scratch output is inspected first, then
  promoted with complete config, run metadata, metrics, arrays, and figures where
  applicable.
- The reproducibility audit is a scripted workflow, not a prose checklist. It
  verifies the core package checks, experiment smoke checks, manifest validation,
  and at least selected canonical artifact regeneration from a clean environment.
- Release readiness includes non-code metadata: citation metadata, README status
  and citation updates, release notes, artifact DOI preparation, and a tag tied
  to the submitted paper version.
- CI activation is part of publication readiness. Manual-only CI is acceptable
  while the repo is private, but public release requires push and pull-request
  triggers for the core checks and experiment smokes.
- The paper methods section is assembled from the ADR-backed register:
  mean-field VI, KL/N, local reparameterization for training, coherent global
  posterior draws for prediction, positivity links with explicit floors,
  law-of-total-variance decomposition, fair CRPS, PIT, coverage, WAIC/LOO, and
  validation-only HMC.
- The paper limitations section is not optional. It must preserve the documented
  mean-field VI narrowness, no post-hoc band inflation, no shipped HMC backend,
  and no claims of literal marginal-likelihood Bayes factors.
- Deferred nice-to-haves are recorded outside the publication-blocking checklist.
  New methods are not added before submission unless they directly fix a
  correctness, reproducibility, or reviewer-blocking evidence problem.

## Testing Decisions

- Good tests for this PRD assert publication-facing behavior at boundaries:
  manifests validate, paper artifacts are built from promoted evidence, smoke
  artifacts cannot satisfy full-evidence claims, release metadata exists, and CI
  configuration contains the expected public triggers when release mode is
  enabled.
- The evidence manifest validator is tested with small synthetic manifests and
  temporary artifact trees. Tests cover present artifacts, missing artifacts,
  stale or malformed metadata, smoke-vs-full mismatches, and helpful failure
  messages.
- The paper artifact builder is tested against minimal fixture result files. It
  should produce stable table and figure filenames, write provenance, and fail if
  asked to read scratch artifacts.
- The benchmark publication gate is tested with fixture comparison tables. It
  should detect whether every claimed comparator/dataset/metric combination has
  a full canonical result or an explicit documented exclusion.
- The reproducibility audit workflow is tested in bounded mode. It should run
  core checks and experiment smokes without invoking full expensive experiments.
- Documentation and release metadata tests check external behavior: citation
  metadata parses, README status matches release mode, and the paper artifact
  version is recorded.
- Prior art for these tests already exists in the repository's experiment
  convention tests, experiment smoke tests, canonical-evidence tests, tooling
  contract tests, and boundary-behavior doctrine from the numerical package
  tests.
- Full experiment reruns remain manual and opt-in. CI validates smoke paths,
  manifests, and artifact assembly; it does not run the full paper benchmark
  panel on every pull request.

## Out of Scope

- New inference engines, including a shipped HMC/NUTS backend.
- A JAX runtime port of the package.
- First-party Laplace approximation.
- Post-hoc conformal calibration, band inflation, or recalibration.
- New family tiers such as zero-inflated, hurdle, GEV, skew-t, LogNormal, or
  Weibull unless the already-defined paper experiments require them.
- New shape-function architectures or formula syntax extensions.
- Reopening mean-field VI, KL scaling, posterior sampling, family link, or
  variance-decomposition design decisions without a discovered correctness bug.
- Chasing RMSE as a headline benchmark metric.
- Making full expensive experiment runs mandatory in normal CI.
- Publishing to package indexes unless the release plan explicitly decides to do
  so.

## Further Notes

- Treat this PRD as a release train, not as another modeling feature wave. The
  success criterion is that a reviewer can connect every major paper statement
  to an exact config, seed, artifact, and code release.
- The UCI/comparator panel deserves special attention before submission. Smoke
  results prove the harness seam; they do not by themselves satisfy a full
  benchmark claim.
- The first paper should stay centered on uncertainty disentanglement:
  per-feature, per-parameter epistemic uncertainty separated from aleatoric
  family uncertainty in distributional neural additive models.
- Nice-to-haves worth considering after the paper artifact is frozen include
  prior-tier ablations, posterior-draw sensitivity, runtime scaling, richer
  baseline panels, tutorial notebooks, a public documentation site, and future
  work on richer variational families or JAX-backed performance.
