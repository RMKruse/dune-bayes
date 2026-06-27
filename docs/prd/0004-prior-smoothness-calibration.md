# PRD 0004: Prior/smoothness calibration for epistemic effect bands

Status: Draft · Date: 2026-06-27 · GitHub: #158

This PRD follows the paper-results grilling session after the first
publication-readiness evidence review. It assumes the core package, metrics,
families, experiment harness, and paper artifact path from PRDs 0001-0003
exist. It deliberately narrows the next improvement branch to the existing
ADR-0002 prior/smoothness machinery before considering architecture sweeps or
richer posterior families.

## Problem Statement

I have evidence that the current DUNE approach is scientifically useful but not
automatically well calibrated. The paper-results explorer makes the situation
plain: mean-field VI can produce narrow epistemic effect bands, some simulation
coverage gaps are large, VI-vs-NUTS comparisons quantify shrinkage, and real-data
benchmarks do not support a predictive-leaderboard claim.

The project therefore needs a disciplined calibration-improvement path that does
not hide uncertainty problems with post-hoc band inflation. I need to know
whether the already-designed `prior_scale`/`PriorScale` machinery can improve
per-feature, per-parameter epistemic-band coverage before I spend effort on
larger architectural sweeps, richer posterior families, or a second methods
paper. Without this path, the paper risks either overselling mean-field VI or
wandering into open-ended tuning without pre-declared acceptance criteria.

## Solution

Create a prior/smoothness calibration track focused first on Normal parameter
recovery. The track uses the existing interpretation of `prior_scale` as the
neural smoothing parameter: fixed prior-scale sensitivity, empirical-Bayes
smoothness selection, and the hierarchical inverse-gamma prior tier are tested
against the current fixed `1.0` baseline.

The first pass screens five pre-registered candidates under one seed. A
candidate only becomes paper evidence if it beats the baseline under a declared
coverage criterion and then survives a baseline-versus-candidate confirmatory
run on three pre-declared seeds. All outputs stay in ignored scratch runs until
manual review. Promotion to canonical results and publication evidence happens
only after the confirmatory pass.

This solution keeps the paper claim posture honest: DUNE is framed as a
methodologically transparent distributional Bayesian NAM framework that exposes
epistemic and aleatoric uncertainty and measures calibration limitations, not as
a universal predictive-performance winner. Predictive NLL/CRPS remain guardrails
and context, but the primary optimization target for this PRD is epistemic
effect-band calibration.

## User Stories

1. As a paper author, I want calibration improvement to mean model/inference
   improvement rather than post-hoc band inflation, so that the paper does not
   hide mean-field VI limitations.
2. As a paper author, I want the first improvement branch to use existing
   prior/smoothness machinery, so that the next experiment tests an ADR-backed
   design before inventing a new inference method.
3. As a paper author, I want Normal parameter recovery as the first experimental
   unit, so that I can reason about finite-variance two-parameter behavior before
   adding family-specific complications.
4. As a paper author, I want fixed `prior_scale` sensitivity at small, baseline,
   and large values, so that I can see whether smoothness strength controls
   effect-band coverage.
5. As a paper author, I want an empirical-Bayes prior-scale candidate, so that I
   can test the neural analog of REML smoothness selection.
6. As a paper author, I want a hierarchical inverse-gamma candidate, so that I
   can test the BAMLSS-faithful variance-component tier with a cleaner KL signal
   than the half-Cauchy first pass.
7. As a paper author, I want half-Cauchy deferred from the first sweep, so that
   single-sample KL noise does not obscure the initial interpretation.
8. As a paper author, I want the candidate set pre-registered, so that the sweep
   cannot become a cherry-picked search after seeing results.
9. As a paper author, I want a declared acceptance criterion based on mean
   absolute coverage error, so that “better calibration” is measurable.
10. As a paper author, I want centered recovery plots reviewed alongside
    coverage tables, so that improved coverage does not merely mean uselessly
    wide ribbons.
11. As a paper author, I want intercept coverage reviewed separately, so that
    additive non-identifiability remains handled honestly.
12. As a paper author, I want final loss monitored, so that apparent calibration
    gains are not accepted when training has failed.
13. As a paper author, I want learned or sampled prior-scale diagnostics emitted
    with every run, so that smoothness behavior is interpretable.
14. As a paper author, I want screening on one seed but promotion blocked by
    three confirmatory seeds, so that one lucky RNG draw cannot become paper
    evidence.
15. As a paper author, I want the confirmatory seeds fixed before seeing the
    winner, so that the robustness check is not selected post hoc.
16. As a paper author, I want all sweep output to remain under ignored scratch
    runs first, so that unreviewed artifacts cannot leak into paper evidence.
17. As a paper author, I want promotion to results and manifest updates gated by
    manual review, so that canonical evidence remains deliberate.
18. As a paper author, I want calibration-improving simulation settings kept
    separate from benchmark settings when needed, so that predictive performance
    is not silently traded away.
19. As a paper author, I want benchmark NLL/CRPS treated as guardrails, so that
    the paper remains honest about real-data behavior without changing its
    central claim.
20. As a paper author, I want the paper-results notebook to discover sweep
    outputs automatically, so that I can manually inspect performance without
    copying numbers.
21. As a paper author, I want the notebook to generate confirmatory seed configs
    only after a screening candidate is chosen, so that the workflow stays tidy.
22. As a maintainer, I want the experiment runner to accept prior-tier specs from
    experiment configuration, so that the public shape-function API and
    experiment layer stay aligned.
23. As a maintainer, I want the experiment runner to keep fixed-prior behavior
    backward-compatible, so that existing canonical results remain interpretable.
24. As a maintainer, I want the sweep configs committed but their outputs
    ignored, so that the plan is reproducible without promoting scratch results.
25. As a maintainer, I want smoke tests for the new prior candidates, so that the
    experiment contract does not rot.
26. As a maintainer, I want prior-scale diagnostics covered by tests, so that the
    acceptance criterion has machine-readable evidence.
27. As a maintainer, I want the issue document to record the sweep shape, so that
    future work can be broken into implementation tickets without rereading the
    conversation.
28. As a reviewer, I want to see that calibration limitations are measured before
    any claims are strengthened, so that I can trust the uncertainty story.
29. As a reviewer, I want to know whether prior/smoothness tuning improves
    epistemic bands, so that the claimed Bayesian regularization is supported by
    evidence.
30. As a reviewer, I want richer posterior inference kept out of v1 scope unless
    needed, so that the first paper does not blur into an unbounded methods
    expansion.
31. As a future contributor, I want the richer-inference fallback named but
    deferred, so that the next branch is clear if prior/smoothness does not work.
32. As a future contributor, I want the fallback to start with last-layer
    richer covariance rather than full-network flows, so that implementation
    risk stays proportional to the calibration problem.
33. As a future contributor, I want this PRD to preserve the terminology of
    shape functions, epistemic uncertainty, aleatoric uncertainty, prior-scale
    smoothness, and coverage evaluation, so that the work remains aligned with
    the domain glossary.

## Implementation Decisions

- The paper claim posture is narrowed. The first paper is not a
  predictive-leaderboard claim; it is a methodological-transparency claim about
  per-feature, per-parameter epistemic uncertainty, response-family aleatoric
  uncertainty, and measured limitations of mean-field VI.
- Calibration improvement means improving the model or inference setup. Detached
  post-hoc band inflation, conformal recalibration, or cosmetic interval
  correction remain outside the method.
- The primary deep module for this PRD is the parameter-recovery experiment
  runner. It exposes a simple configuration surface: family, truth,
  architecture, training budget, prior-scale tier, draw count, calibration
  levels, and artifact destination. Internally it handles model construction,
  fitting, effect sampling, centered coverage, intercept coverage, plots, arrays,
  and prior-scale diagnostics.
- The experiment runner accepts an optional prior-tier specification in the
  architecture configuration and forwards it to the Bayesian shape function. If
  the prior spec is omitted, existing fixed-prior behavior remains unchanged.
- Every run writes a prior-scale diagnostic artifact. Fixed candidates report
  the configured scale; empirical-Bayes candidates report the learned positive
  scale; hierarchical candidates report posterior log-scale location,
  log-scale spread, median, and mean.
- The first sweep is scoped to Normal parameter recovery. This is the cleanest
  sandbox for understanding effect-band coverage because it avoids heavy-tail,
  positive-support, bounded-support, and count-family complications.
- The pre-registered candidate set is exactly five candidates: fixed
  `prior_scale` values `0.3`, `1.0`, and `3.0`; empirical-Bayes initialized at
  `1.0`; and hierarchical inverse-gamma initialized at `1.0`.
- The fixed `1.0` candidate is the baseline. All conclusions are relative to
  that baseline, not relative to an untracked historical run.
- The first screening seed is `9801`. A screening winner must be compared
  against the fixed `1.0` baseline on confirmatory seeds `9801`, `9811`, and
  `9821` before any promotion.
- Mean absolute coverage error across Normal location and scale at nominal
  50/80/90/95 is the first acceptance metric. It is not sufficient by itself:
  recovery plots, intercept coverage, final loss, ribbon width, and prior-scale
  diagnostics remain required review inputs.
- Calibration-improving settings may remain separate from benchmark settings.
  A setting that helps simulation effect-band coverage but hurts NLL/CRPS may
  support the uncertainty claim, but it does not become the package default or
  the benchmark configuration unless predictive guardrails also pass.
- The paper-results notebook becomes the manual investigation surface. It reads
  promoted artifacts by default, lists pre-registered sweep commands, summarizes
  available scratch sweep outputs, and can materialize confirmatory seed configs
  after a screening candidate is selected.
- Scratch outputs remain ignored. Committed configuration files define what to
  run; generated metrics, arrays, figures, and confirmatory config material stay
  in the ignored experiment run area until reviewed.
- Promotion to canonical results and publication evidence is explicitly
  separate from running the sweep. The evidence manifest is updated only after
  the screening and confirmatory checks justify promotion.
- The first richer-inference fallback, if prior/smoothness does not materially
  improve coverage, is a last-layer richer posterior or low-rank covariance path
  around a last-layer Bayesian shape function or linearized final layer.
- Richer posterior inference is follow-up scope, not a v1 paper requirement.
  The first paper may honestly report mean-field VI limitations and the
  disciplined prior/smoothness attempt without shipping a richer posterior
  family.
- No new ADR is required for this PRD. The work exercises ADR-0002 and the
  existing coverage doctrine; it does not change a hard-to-reverse architectural
  decision.

## Testing Decisions

- Tests should assert external experiment behavior at module boundaries:
  artifacts are written, calibration tables have the expected shape, prior-scale
  diagnostics expose the expected fields, configs remain pre-registered, and
  smoke runs complete. They should not assert private optimizer trajectories or
  single stochastic draws.
- The parameter-recovery experiment runner is tested with smoke runs. Good tests
  verify that recovery figures, calibration tables, intercept-coverage tables,
  arrays, run metadata, and prior-scale diagnostics are produced.
- The prior/smoothness sweep configuration set is tested as a public contract.
  A good test verifies that the first sweep contains exactly the five
  pre-registered Normal candidates and that their artifact roots point to
  scratch output.
- Empirical-Bayes and hierarchical inverse-gamma candidates receive smoke tests
  that verify prior-scale diagnostics are written and contain positive,
  interpretable scale summaries.
- Existing deterministic regeneration tests should include the new prior-scale
  diagnostics for fixed-prior runs, because paper-facing artifacts are expected
  to be reproducible from config and seed.
- Prior art for these tests is the existing experiment-boundary suite for
  parameter recovery, which checks smoke artifacts, family calibration grids,
  reproducibility from config and seed, centered truth/draws, and intercept
  coverage.
- The notebook is validation-checked by executing it top-to-bottom. It should
  not run full experiments by default; optional local sweep execution remains
  opt-in.
- Full candidate runs are not CI-gating. CI should run smoke contracts; full
  canonical and confirmatory runs remain manual and review-gated.

## Out of Scope

- Post-hoc conformal calibration, band inflation, or detached interval
  recalibration.
- Changing the default package prior, benchmark configuration, or paper evidence
  manifest based on a single screening run.
- Promoting any scratch run into canonical results without manual review and the
  confirmatory seed check.
- Treating real-data benchmarks as a universal predictive-dominance claim.
- Full-network normalizing flows, full-covariance posterior inference, or a
  shipped MCMC backend for v1.
- Reopening the mean-field VI decision, family tiers, positivity-link rules,
  variance-decomposition contract, or posterior predictive representation.
- Adding half-Cauchy hierarchical sweeps to the first pass.
- Running or scoring optional external baselines as part of this calibration
  branch unless a later benchmark-specific issue requests it.
- Committing generated scratch metrics, arrays, figures, or notebook-generated
  confirmatory configs before promotion.

## Further Notes

- This PRD builds on the existing domain language in the glossary: shape
  function, Bayesian shape function, prior scale as smoothness, epistemic
  uncertainty, aleatoric uncertainty, posterior predictive, coverage evaluation,
  and variance decomposition.
- The plan intentionally preserves the no-post-hoc-recalibration doctrine while
  creating room to improve calibration through the model/inference setup.
- The initial implementation work already established the runner hook,
  pre-registered configs, prior-scale diagnostics, focused experiment tests, and
  notebook support. Remaining work is to run the full screening candidates,
  inspect outputs, choose whether any candidate deserves confirmatory seeds, and
  decide whether to promote new evidence.
- If no prior/smoothness candidate materially improves coverage, the next PRD or
  issue should focus on a last-layer richer posterior or low-rank covariance
  fallback, not on broad unstructured tuning.
