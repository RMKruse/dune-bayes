# PRD 0002: Paper hardening — correctness, disentanglement, experiments

Status: Draft · Date: 2026-06-07

Distilled from the 2026-06-07 grilling session over the maintainer's hardening
brief. Decision record: ADR-0007 (estimator/sampling split) and the session
trail ADR-0008; glossary updates in `CONTEXT.md` (variance decomposition,
family tiers, coverage evaluation, HMC validation); process updates in
`CLAUDE.md` (link floors, layout tiers, mypy gating, non-skippable tests).

## Problem Statement

I have a working prototype of a Bayesian distributional NAM, and I am writing
a research paper whose thesis is **uncertainty disentanglement**: cleanly
separating, per feature and per distribution parameter, epistemic uncertainty
(posterior over weights — "how confident am I that this shape function is
correct?") from aleatoric uncertainty (the fitted family's spread — "how much
does the outcome inherently vary here?"). The prototype's ELBO, sampling
paths, and numerics have been built slice by slice but never verified as a
paper-grade foundation: the variance-reduction estimator is misnamed and its
boundary with posterior sampling is unguarded, link functions underflow at
extreme pre-link values, the KL claims per prior tier have no verification
tests, the calibration/disentanglement metrics that constitute the paper's
empirical core do not exist, there are no reproducible experiments, and there
are no baselines. A wrong KL term or an unstable log-likelihood invalidates
the entire paper.

## Solution

Harden the existing mean-field VI path (never replace it) in five ordered
waves, working strictly top-down by paper risk:

1. **Wave 1 — Correctness & stability of the ELBO and sampling.** Rename the
   misnamed `flipout` estimator to `local_reparam`, make it the training
   default, and pin (by test) that all posterior sampling uses coherent global
   weight draws. Verify the KL claims of every prior tier against what is
   actually claimable. Floor every positivity link (`softplus + EPS`) and gate
   every family on extreme pre-link inputs. Gradcheck the deterministic
   numerical atoms; assert no NaN gradients on an ill-conditioned batch.
   Settle the determinism question by experiment and amend the losing
   document.
2. **Wave 2 — Disentanglement & calibration metrics.** A tested
   `metrics` package module: the law-of-total-variance decomposition (the
   thesis in code form), sample-based fair CRPS, (randomized) PIT, and
   per-parameter quantile coverage.
3. **Wave 3 — Families.** Johnson's SU (custom distribution, scipy
   parameterization, closed-form moments), NegativeBinomial, Beta — one at a
   time, each behind the correctness gate.
4. **Wave 4 — Experiments.** A new `experiments/` tier (configs + seeds +
   logged artifacts, smoke-tested in CI): parameter-recovery and
   disentanglement simulations, the JSU heteroscedastic-skew showcase, and a
   VI-vs-NUTS agreement study on a NumPyro reimplementation of one small
   instance.
5. **Wave 5 — Baselines.** A common evaluation harness scoring NAMLSS (live
   NAMpy re-run), LA-NAM (wrapped), BayesNAM-style (labeled degenerate config
   of dune-bayes), BAMLSS (committed R script + fixtures), and an MLP / deep
   ensemble sanity floor on identical splits.

If a change does not improve correctness, stability, the
uncertainty-separation story, or the experiments demonstrating it, it is not
in scope.

## User Stories

1. As a paper author, I want the training-time variance-reduction estimator
   correctly named `local_reparam`, so that my methods section does not
   misstate the algorithm as flipout.
2. As a paper author, I want local reparameterization on by default during
   training, so that ELBO gradients have lower variance at no statistical
   cost.
3. As a paper author, I want every posterior-sampling entry point to use
   coherent global weight draws, so that each of my T draws is one function
   and the mixture decomposition is valid.
4. As a paper author, I want a test that pins the estimator/sampling boundary,
   so that no future change silently routes sampling through per-row noise.
5. As a paper author, I want the analytic Gaussian weight-KL verified against
   an MC estimate for the fixed and empirical-Bayes tiers, so that the ELBO's
   largest term is independently confirmed.
6. As a paper author, I want the closed-form inverse-gamma scale-KL verified
   against an MC estimate, so that the BAMLSS-faithful tier is confirmed.
7. As a paper author, I want the half-Cauchy single-sample MC KL verified as
   unbiased against 1-D quadrature ground truth, so that the default
   hierarchical tier's estimator is honest about what it is.
8. As a paper author, I want an explicit KL/N-scaling test on a hand-derived
   toy model, so that the prior's weight in the objective is provably right.
9. As a paper author, I want β-warm-up tests (β=0 ⇒ pure NLL, β=1 ⇒ full
   ELBO, monotone schedule reaching 1), so that annealing is verified.
10. As a modeller, I want every positivity link to be `softplus(x) + EPS`, so
    that no finite pre-link input can produce a zero scale and a poisoned
    `log_prob`.
11. As a modeller, I want every registered family to pass an extreme pre-link
    (±1e4) finite-`log_prob` gate, so that training cannot NaN from a wild
    early-epoch predictor.
12. As a paper author, I want gradcheck on the deterministic numerical atoms
    (Gaussian KL, IG scale-KL, every family's `log_prob` through its links,
    near boundaries, float64), so that gradient correctness is verified where
    hand-written math lives.
13. As a paper author, I want a no-NaN-gradient test on a deliberately
    ill-conditioned batch, so that the full stochastic ELBO survives the worst
    realistic step.
14. As a paper author, I want a determinism test under a re-seed protocol
    (seed → build → fit, twice, identical trajectories), so that the
    reproducibility constraint is settled by evidence and the losing document
    amended.
15. As an experimenter, I want `seed_everything(seed, deterministic=False)`
    with an opt-in deterministic-algorithms flag, so that experiments get
    bitwise reproducibility while interactive use stays fast.
16. As a paper author, I want a generic law-of-total-variance decomposition
    (aleatoric = E_θ[Var(y|θ)], epistemic = Var_θ[E(y|θ)]) computed from each
    draw's family mean/variance, so that the paper's core claim exists as
    tested code.
17. As a paper author, I want the decomposition verified on a synthetic case
    where both components are known by construction (dense+noisy region vs
    sparse+quiet region), so that the disentanglement claim is rock solid.
18. As a modeller, I want infinite aleatoric variance surfaced honestly
    (`inf` plus a warning naming the cause), so that heavy-tail truth is never
    clamped into a plausible-looking lie.
19. As a modeller, I want `StudentTFamily(df_min=...)` (default keeping
    df > 1), so that finite-variance experiments can pin df > 2 explicitly.
20. As a modeller, I want every family to guarantee defined (or documented)
    mean/variance on its distribution, so that the decomposition works
    generically for any registered family.
21. As a paper author, I want per-feature, per-parameter epistemic credible
    bands — including scale and shape parameters, so that I provide what
    mean-only Bayesian NAMs cannot.
22. As a paper author, I want a sample-based fair CRPS estimator (sort-based,
    float64) tested against the analytic Gaussian CRPS, so that all models and
    families are scored by one proper rule.
23. As a paper author, I want PIT histograms / reliability diagrams with
    randomized PIT for discrete families, so that calibration claims survive
    count data.
24. As a paper author, I want per-parameter quantile coverage at nominal
    50/80/90/95, measured and reported — never asserted as "correct", so that
    mean-field narrowness is quantified rather than denied.
25. As a paper author, I want recovery comparisons on centered truth vs
    centered posterior draws with intercept coverage assessed separately, so
    that a non-identifiable level constant cannot corrupt coverage numbers.
26. As a modeller, I want a Johnson's SU family (custom distribution, scipy
    `johnsonsu` parameterization, closed-form moments, reference-tested
    against scipy for log-prob and moments), so that epistemic uncertainty
    over skew/kurtosis shape functions is demonstrable.
27. As a modeller, I want a NegativeBinomial family, so that the variance
    decomposition is proven on discrete support.
28. As a modeller, I want a Beta family, so that bounded-response benchmark
    datasets are servable.
29. As a modeller, I want each new family to ship with reference log-prob
    tests, the extreme-value gate, a parameter-recovery fit test, and a
    docstring pinning its parameterization, so that families are added behind
    a stability gate, never batch-added.
30. As an experimenter, I want an `experiments/` tier (one subdir per
    experiment, `config.yaml` + `run.py`, gitignored `runs/`, committed
    canonical `results/`), so that every paper artifact is regenerable from a
    config and a seed.
31. As an experimenter, I want every experiment to expose `--smoke` and CI to
    run all smokes, so that experiments cannot silently rot.
32. As a paper author, I want a parameter-recovery simulation per family with
    per-parameter calibration tables, so that credible-band quality is
    evidenced where ground truth is known.
33. As a paper author, I want the disentanglement centerpiece figure
    (epistemic↑ where data is sparse, aleatoric↑ where noise is high) clean
    and scriptable, so that the paper's central claim is one command away.
34. As a paper author, I want the JSU heteroscedastic-skew study showing
    uncertainty over scale and shape shape-functions, so that I demonstrate
    what no prior Bayesian-NAM work provides.
35. As a paper author, I want a VI-vs-NUTS agreement study on one small
    fixed-prior instance via a NumPyro reimplementation, with a shared
    log-joint fixture proving the two model definitions identical, so that VI
    quality is measured against a gold standard without shipping an MCMC
    engine.
36. As a paper author, I want the VI-vs-NUTS band-width ratio reported, so
    that observed under-coverage is attributed to mean-field shrinkage with
    evidence.
37. As a paper author, I want the standard UCI regression panel including
    count-data and bounded-response members, scored on NLL, CRPS, and
    calibration, so that the package shows competitive likelihood and superior
    uncertainty rather than chasing RMSE.
38. As a paper author, I want the original NAMLSS experiments re-run live
    through NAMpy inside the common harness on identical splits and
    preprocessing, so that the Bayesian version's added value is measured
    without printed-number confounds (I will supply the original experiment
    code).
39. As a paper author, I want LA-NAM wrapped from the authors' code (pinned
    git dependency in the experiments extra, license permitting; fixtures
    otherwise), so that the closest competitor is represented faithfully.
40. As a paper author, I want a BayesNAM-style baseline as a clearly labeled
    degenerate configuration of dune-bayes (location-only Bayesian, learned
    homoscedastic scale), so that the mean-only-variational comparison exists
    despite no canonical public implementation.
41. As a paper author, I want BAMLSS results produced by a committed, seeded R
    script writing per-observation predictive fixtures, so that the
    stats-world reference is reproducible without an rpy2 bridge.
42. As a paper author, I want a plain MLP and deep ensemble sanity floor in
    the harness, so that reviewers see the conventional lower bound.
43. As a maintainer, I want every baseline behind one adapter scored by
    `dune_bayes.metrics` on the same splits, so that comparison tables are
    confound-free by construction.
44. As a maintainer, I want WAIC/PSIS-LOO comparison kept on the existing
    arviz spine (float64), so that model-comparison claims stay
    reference-tested.
45. As a maintainer, I want scipy as an explicit runtime dependency, so that
    eval-time CDFs (StudentT, JSU) are computed by a trusted reference rather
    than hand-rolled incomplete-beta math.
46. As a maintainer, I want mypy CI-gating with `disallow_untyped_defs` on the
    package (not `--strict`), so that the public-signature typing rule is
    mechanized without paying torch's strict-mode tax.
47. As a maintainer, I want correctness tests unskippable and slow suites
    (`hmc`, `experiment` markers) opt-in, so that the core gates run in every
    CI job while the suite stays fast.
48. As a maintainer, I want decision-shaped math recorded in ADRs and
    derivation-shaped math beside the code (no separate DECISIONS.md), so that
    the methods section assembles from one non-drifting register.

## Implementation Decisions

All decisions below were settled in the 2026-06-07 grilling session; the
brief's wording loses wherever it conflicts.

1. **Estimator rename + split (ADR-0007).** `flipout` → `local_reparam`
   everywhere (the implementation is Kingma-style local reparameterization,
   not flipout). Training default ON; `EffectSampler`, `draw_predictive`, and
   `pointwise_log_lik` always use vanilla coherent global weight draws. A test
   pins the boundary (path assertion + behavioral coherence check on a linear
   shape function). No true-flipout implementation.
2. **Prior-tier KL verification is four tests, not three.** Fixed: analytic
   Gaussian KL vs MC. Empirical-Bayes: same plus gradient-reaches-scale
   assertion. Hierarchical-IG: closed-form scale-KL vs MC. Hierarchical
   half-Cauchy: single-sample MC estimator's mean vs 1-D quadrature ground
   truth (there is no analytic KL to verify — the honest claim is
   unbiasedness). Single-sample MC KL stays in the training loop.
3. **No behavior change to ELBO scaling or warm-up** — KL/N (full-data N) and
   the linear β warm-up already exist; Wave 1 adds the verification tests the
   brief demands (hand-derived toy-model KL scaling, β endpoints, monotone
   schedule).
4. **Links: explicit per-family `softplus(x) + EPS`; `transform_to` rejected**
   (its positive transform is `ExpTransform`, which overflows and violates the
   softplus-never-exp rule). Documented consequence: minimum representable
   scale is EPS. The ±1e4 finite-`log_prob` gate is a parametrized per-family
   test. Bounded-domain link conventions are deferred until a family needs
   one.
5. **Gradcheck re-scoped to deterministic atoms** (Gaussian KL, IG scale-KL,
   each family's `log_prob` through its links at interior and near-boundary
   points, float64, `validate_args=True`). No full-ELBO gradcheck (stochastic
   objective makes finite differences meaningless). The ill-conditioned-batch
   no-NaN-gradient test covers the full model.
6. **Determinism settled by experiment.** Re-seed protocol test
   (seed → build → fit ×2, bit-identical trajectories, CPU). GREEN ⇒ weaken
   the CLAUDE.md caveat to its true scope (one RNG stream, no re-seed);
   RED ⇒ hunt the uncontrolled RNG source before the experiments layer.
   `seed_everything` grows `deterministic=False` (opt-in
   `torch.use_deterministic_algorithms`).
7. **Family contract extension.** The decomposition consumes `dist.mean` /
   `dist.variance`; every family guarantees these defined-or-documented, with
   a per-family conformance test. `StudentTFamily` gains `df_min` (default
   preserves df > 1; df ≤ 2 yields truthful `inf` aleatoric variance plus a
   cause-naming warning — never clamped).
8. **Family tiers.** Core: Normal, StudentT, Gamma (exist) + Johnson's SU +
   NegativeBinomial + Beta (promoted: benchmark panel includes
   bounded-response data). On-demand: LogNormal, Weibull. Deferred:
   zero-inflated, GEV/skew-t. One family per slice, each behind the gate.
9. **Johnson's SU build route.** Custom `Distribution` subclass (log_prob,
   reparameterized rsample via inverse transform of a Normal draw, closed-form
   mean/variance with a documented small-δ validity note), pinned to the scipy
   `johnsonsu` parameterization (z = γ + δ·arcsinh((y−ξ)/λ)) so scipy is a
   zero-translation reference. `TransformedDistribution` rejected (no
   moments).
10. **Metrics live in the package** (`dune_bayes.metrics`): variance
    decomposition, fair sample-based CRPS (sort-based O(M log M), float64),
    PIT (randomized for discrete support, seeded), per-parameter quantile
    coverage. WAIC/LOO stay in `compare/`. scipy becomes an explicit runtime
    dependency (eval-time CDFs).
11. **`experiments/` is a third top-level tier** — in-repo, not packaged, not
    importable. One subdir per experiment with `config.yaml` + `run.py`;
    `runs/` gitignored; canonical artifacts promoted deliberately to committed
    `results/`; `--smoke` mode run by CI; orchestration-only helpers in
    `experiments/_harness/`. Tier rule: statistical capability → package;
    orchestration → experiments; throwaway → spikes.
12. **HMC is validation-only, route NumPyro/JAX** (consistent with the
    ADR-0001/0006 seam): one small fixed-prior instance reimplemented in
    ~50 lines of JAX inside `experiments/`; JAX/NumPyro live only in an
    `experiments` optional-dependency group. A shared fixture asserts torch
    and JAX log-joints agree before bands are compared. Agreement criterion
    pre-registered: VI bands inside HMC bands with matching centers;
    hierarchical priors out of scope (funnel geometry). No first-party Laplace.
13. **Coverage doctrine.** Empirical coverage is measured/reported per
    parameter at nominal 50/80/90/95 — never asserted "correct" (mean-field
    under-coverage is an ADR-0001 documented property; the VI-vs-NUTS
    band-width ratio quantifies it). No post-hoc band
    inflation/recalibration in v1. Recovery sims compare centered truth vs
    centered draws; intercept coverage assessed separately.
14. **Baselines integration.** Common adapter → per-observation predictive →
    scored by `dune_bayes.metrics` on identical splits. NAMLSS: live NAMpy
    re-run inside the harness (maintainer supplies original experiment code;
    published numbers are never the comparison basis). LA-NAM: pinned git
    dependency in the experiments extra pending license, else fixtures.
    BayesNAM: labeled degenerate dune-bayes config (no canonical public
    implementation exists). BAMLSS: committed seeded R script producing
    fixtures (no rpy2). MLP + deep ensemble: minimal in-harness torch.
15. **Benchmark panel.** Standard UCI regression suite as used by
    NAM/NAMLSS/LA-NAM, including its count-data and bounded-response members;
    metrics are NLL, CRPS, calibration (not RMSE-chasing).
16. **Engineering.** mypy: `disallow_untyped_defs` on the package, CI-gating
    once green, `--strict` explicitly not a pre-v1 target. Coverage threshold
    rejected (boundary-behavior doctrine reaffirmed). Correctness tests
    unskippable; `hmc`/`experiment` markers opt-in. No DECISIONS.md — ADRs +
    code-adjacent derivations remain the single register.

## Testing Decisions

- The existing doctrine holds: assert **external behavior at module
  boundaries** — values against an independent reference, shapes, round-trips,
  warnings; never private internals or single stochastic draws. The four
  archetypes (closed-form, round-trip, shape, MC-convergence) cover every new
  numerical claim; tolerances stay explicit and commented.
- **Wave 1 is almost entirely tests**: the four prior-tier KL tests (analytic
  vs MC; quadrature unbiasedness), the toy-model KL/N scaling test, β-endpoint
  and monotonicity tests, the estimator/sampling boundary test (path assertion
  + linear-shape coherence), per-family ±1e4 gates, per-atom gradchecks,
  ill-conditioned-batch NaN gate, and the re-seed determinism test.
- **Independent references per claim**: hand-derived KL values; 1-D quadrature
  for the half-Cauchy KL; `scipy.stats` for every family log-prob and for JSU
  moments; analytic Gaussian CRPS for the CRPS estimator; arviz for WAIC/LOO
  (existing); the torch-vs-JAX log-joint fixture for the HMC study.
- **The disentanglement test is the paper's claim in code form**: a synthetic
  construction where aleatoric and epistemic components are known by
  construction (dense+noisy vs sparse+quiet regions), asserted with
  MC-convergence tolerances, plus a discrete-support variant once NegBin
  lands.
- **Modules under test**: `layers` (estimator boundary), `priors` (KL tiers),
  `families` (every family: reference log-prob, gate, conformance, recovery
  fit), `metrics` (all four metric groups), `model` (scaling, warm-up, NaN
  gate, determinism), `sampling` (coherence, convergence). Experiments are
  exercised via `--smoke` in CI, not unit-tested.
- Prior art: `tests/priors/` closed-form IG-KL tests, `tests/compare/` arviz
  reference tests, `tests/layers/test_variational_dense.py` KL spike
  migrations — the new tests follow those patterns.
- Correctness tests are unskippable; slow suites are opt-in markers.

## Out of Scope

- **First-party Laplace approximation** (LA-NAM is a baseline, not a feature).
- **A general HMC/MCMC engine** — the NumPyro study validates; it does not
  ship an inference backend. Hierarchical priors under NUTS likewise out.
- **Post-hoc calibration / band inflation** (conformal-style) — clean
  follow-up work, not a v1 foundation change.
- **Zero-inflated / hurdle / GEV / skew-t families** (Tier C, deferred);
  LogNormal/Weibull until an experiment needs them.
- **mypy `--strict`**, a numeric coverage threshold, and `DECISIONS.md`.
- **JAX runtime port** of the package (ADR-0006 staging stands; perf headroom
  measured in the T-sweep spike, vectorized sweeps already landed).
- **New shape-function architectures, formula-syntax extensions, or any
  feature** that does not serve correctness, stability, disentanglement, or
  the experiments.

## Further Notes

- Wave ordering is risk ordering: do not start baselines while the ELBO is
  suspect; the `local_reparam` rename goes first because it touches every file
  and gets cheaper never.
- The maintainer is a main author of the original NAMLSS paper and will drop
  the original experiment code into the repo; until then §4.2-style continuity
  work assumes the live-re-run design only.
- The numbering foot-gun stands: GitHub issue # = slice # + 1 for the original
  PRD; new slices get whatever numbers the tracker assigns — always
  cross-check.
