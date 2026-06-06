# 8. Paper-hardening decision record (grilling session, 2026-06-07)

Date: 2026-06-07

## Status

Accepted. Decision record for **PRD 0002 (#84)**; the estimator/sampling split
it references is its own ADR (**ADR-0007**).

## Context

The maintainer brought a hardening brief: turn the working prototype into a
correct, numerically stable, well-tested foundation for a research paper whose
thesis is **uncertainty disentanglement** (per-feature, per-parameter
separation of epistemic and aleatoric uncertainty). The brief was stress-tested
question-by-question against the codebase, CONTEXT.md, CLAUDE.md, and
ADRs 0001–0006. Several of the brief's instructions conflicted with the
repo's documented rules or with mathematical reality; each conflict was
resolved explicitly. This ADR records the decisions and the full artifact
trail so the paper's methods section and future sessions have one register.

## Decisions

1. **Estimator** — `flipout` renamed `local_reparam` (it *is* local
   reparameterization); training default ON; all posterior sampling uses
   coherent global weight draws, pinned by test. → **ADR-0007**.
2. **Prior-tier KL verification re-scoped to four tests** — fixed and
   empirical-Bayes tiers: analytic Gaussian KL vs MC; hierarchical-IG:
   closed-form vs MC; hierarchical half-Cauchy: **no analytic KL exists** —
   verify the single-sample MC estimator's unbiasedness against 1-D
   quadrature. Single-sample MC KL stays in the training loop.
3. **Variance decomposition contract** — generic law-of-total-variance from
   each draw's `dist.mean`/`dist.variance`; every family guarantees
   defined-or-documented moments (conformance-tested);
   `StudentTFamily(df_min=…)` (default keeps df > 1); infinite aleatoric
   variance surfaced honestly (`inf` + cause-naming warning), never clamped.
4. **Links** — explicit per-family `softplus(x) + EPS`;
   `transform_to`/`ExpTransform` rejected (overflow; violates rule 1); bare
   softplus rejected (float32 underflow to exact 0 near pre-link −104);
   per-family ±1e4 finite-`log_prob` gate. Consequence: minimum representable
   scale = EPS.
5. **Determinism settled by experiment** — re-seed protocol test
   (`seed_everything → build → fit` ×2, bit-identical trajectories); the
   losing document (brief's claim or CLAUDE.md's caveat) is amended on the
   evidence. `seed_everything` gains opt-in `deterministic=` flag.
6. **Gradcheck re-scoped to deterministic atoms** (Gaussian KL, IG scale-KL,
   family `log_prob` through links, near boundaries, float64) — full-ELBO
   gradcheck is ill-posed (stochastic objective). Ill-conditioned-batch
   no-NaN-gradient gate kept for the full model.
7. **Family tiers** — core: Normal, StudentT, Gamma, Johnson's SU,
   NegativeBinomial, Beta (promoted: bounded-response benchmark data);
   on-demand: LogNormal, Weibull; deferred: zero-inflated, GEV/skew-t. One
   family per slice, behind the gate, never batch-added ("8+" scoped down by
   the brief's own gating principle).
8. **Johnson's SU build route** — custom `Distribution` subclass (log_prob,
   reparameterized rsample, closed-form moments), pinned to the **scipy
   `johnsonsu` parameterization** (z = γ + δ·arcsinh((y−ξ)/λ)) for
   zero-translation reference tests. `TransformedDistribution` rejected (no
   moments).
9. **HMC validation route 2** — NumPyro/JAX reimplementation of one small
   fixed-prior instance in `experiments/` (consistent with the ADR-0001/0006
   seam); JAX only in the `experiments` extra; shared log-joint equivalence
   fixture; pre-registered criterion: VI bands inside HMC bands, matching
   centers; band-width ratio quantifies mean-field shrinkage. No first-party
   Laplace; no shipped MCMC engine.
10. **Metrics live in the package** (`dune_bayes.metrics`): decomposition,
    fair sample-based CRPS (sort-based, float64, vs analytic Gaussian
    reference), PIT (randomized for discrete, seeded), per-parameter quantile
    coverage. scipy becomes an explicit runtime dependency (eval-time CDFs;
    torch StudentT has no `.cdf`).
11. **`experiments/` is a third top-level tier** — config + seed + artifacts;
    `runs/` gitignored, canonical `results/` committed; `--smoke` in CI;
    orchestration in `_harness/`. Tier rule: statistical capability → package;
    orchestration → experiments; throwaway → spikes.
12. **NAMLSS comparability via live re-run** — NAMpy executed inside the
    common harness on identical splits/preprocessing; published numbers never
    the comparison basis. The maintainer (a main author of the NAMLSS paper)
    supplies the original experiment code as a sanity cross-check. Benchmark
    panel: standard UCI suite incl. count and bounded-response members.
13. **Baselines** — common adapter scored by `dune_bayes.metrics` on shared
    splits. LA-NAM: pinned git dep pending license (else fixtures). BayesNAM:
    labeled degenerate dune-bayes config (no canonical implementation exists).
    BAMLSS: committed seeded R script + fixtures, no rpy2. MLP + deep ensemble
    in-harness.
14. **Coverage doctrine** — empirical coverage measured/reported per parameter
    (50/80/90/95), never asserted "correct" (mean-field narrowness is an
    ADR-0001 documented property, quantified by the HMC band-width ratio); no
    post-hoc band inflation in v1; recovery compares centered truth vs
    centered draws, intercept coverage separate.
15. **Engineering** — mypy `disallow_untyped_defs` + CI gating, not
    `--strict`; no coverage threshold (boundary doctrine reaffirmed);
    correctness tests unskippable, `hmc`/`experiment` markers opt-in; **no
    DECISIONS.md** — ADRs + code-adjacent derivations remain the single
    register.

## Artifact trail

**Created (uncommitted at session end; commits only on explicit instruction):**

- `docs/adr/0007-local-reparam-training-coherent-draws-sampling.md`
- `docs/adr/0008-paper-hardening-decision-record.md` (this file)
- `docs/prd/0002-paper-hardening.md`

**Edited:**

- `CONTEXT.md` — glossary entries added/updated: *VariationalDense* (estimator
  naming + sampling split), *Variance decomposition (disentanglement)*,
  *Family* (link floor, gate, tiers, JSU parameterization), *Inference engine*
  (HMC validation scope), *Coverage evaluation (simulations)*.
- `CLAUDE.md` — numerical rule 1 (softplus+EPS floor, transform_to ban,
  ±1e4 gate); layout (`metrics/` module, `experiments/` tier + tier rule);
  tooling (mypy gating decision); testing (coverage-threshold rejection
  reaffirmed; unskippable correctness tests + opt-in slow markers).

**Published to GitHub (RMKruse/dune-bayes):**

- PRD: **#84**.
- Slices: **#85–#108** (24 tracer-bullet slices; #104, #105, #107 are HITL —
  maintainer code drop, license call, local R run).
- Wave epics with subtask lists: **#109** (Wave 1 — ELBO & sampling
  correctness: #85–#90), **#110** (Wave 2 — metrics: #91–#93), **#111**
  (Wave 3 — families: #94–#96), **#112** (Wave 4 — experiments: #97–#102),
  **#113** (Wave 5 — baselines + tooling: #103–#108).
- Labels created: `comp: metrics`, `comp: experiments`.

## Consequences

- Wave ordering is risk ordering; later waves do not start while the ELBO is
  suspect. The rename (#85) goes first.
- The brief itself is superseded by PRD 0002 wherever they differ; this ADR is
  the conflict ledger.
- The paper's methods section can be assembled from ADRs 0001–0008 plus
  code-adjacent derivations — by design, nothing methods-relevant lives
  anywhere else.
