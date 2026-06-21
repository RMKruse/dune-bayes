# AGENTS.md — working rules for dune-bayes

Operational guidance for working in this repo. The **what** and **why** of the
design live in `CONTEXT.md` (glossary), `docs/adr/` (decisions), and
`docs/prd/` + `docs/issues/` (scope). This file covers the **how**: numerical
conventions, coding rules, and the GitHub workflow. When this file and an ADR
disagree on a decision, the ADR wins for *design* and this file wins for
*process* — surface the conflict rather than guessing.

## Two non-negotiable principles

1. **Mathematical correctness over cleverness.** A correct, plainly-written
   numerical implementation always beats a clever or terse one. If a closed form
   exists, use it; do not Monte-Carlo-estimate what can be derived.
2. **Numerical stability is non-negotiable.** It is a hard requirement, not a
   nice-to-have. Code that is fast but unstable is wrong. See *Numerical rules*.

---

## Project shape

- **`dune-bayes`** (DUNE — Distributional Uncertainty in Neural-additive
  Estimation) builds on the NAMLSS framework (Thielmann et al., 2024):
  NAMpy's deterministic feature networks made **Bayesian** (mean-field VI),
  in the spirit of the BAMLSS R package. Full picture in `CONTEXT.md`.
- **Backend is PyTorch** (ADR-0006). JAX/NumPyro is the designated numerical
  future + deferred MCMC backend, behind a thin distribution/inference seam.
- **Layout — `src/` layout, fresh `dune_bayes` namespace:**
  ```
  src/dune_bayes/
    layers/    # VariationalDense (ADR-0004), BayesianIntercept
    shapes/    # BayesianMLP, NeuralLinearMLP (registry entries)
    priors/    # PriorScale handle: fixed / empirical-Bayes / hierarchical (ADR-0002)
    sampling/  # EffectSampler, LogLikSampler (split workhorses)
    families/  # torch.distributions wrappers (param_count, links, log_prob)
    data/      # DataModule (torch.utils.data + numpy/sklearn preprocessing)
    compare/   # to_inference_data, waic / loo / compare (arviz, ADR-0003)
    metrics/   # CRPS, PIT/reliability, coverage, variance decomposition (tested package code)
    formula/   # mgcv-style formula parser — REIMPLEMENTED, not reused (see below)
    model.py   # BayesianNAMLSS
  tests/       # mirrors src/dune_bayes/
  docs/        # CONTEXT.md, adr/, prd/, issues/
  NAMpy/       # TF baseline — REFERENCE + external comparator only; never imported
  spikes/      # throwaway verification scripts (NOT package code, NOT in CI)
  experiments/ # paper evidence: one subdir per experiment (config.yaml + run.py);
               # runs/ gitignored, canonical results/ committed; --smoke flag runs in CI;
               # orchestration in experiments/_harness/, never in the package.
               # Tier rule: statistical capability → src; orchestration → experiments;
               # throwaway → spikes. JAX/NumPyro allowed here only (experiments extra).
  pyproject.toml
  ```
- Import as `import dune_bayes as db`; install editable with
  `uv pip install -e .` against `src/dune_bayes`.
- **NAMpy is reference-only.** Read `NAMpy/` to see how something was done on the
  old TF stack, but **do not import, vendor, or port its code**. The formula
  parser and `ShapeFunctionRegistry` are **reimplemented from scratch** in
  `dune_bayes/`. This reverses ADR-0006's original "reuse NAMpy's TF-free
  Python directly" wording — see the amendment note in
  `docs/adr/0006-pytorch-compute-backend-jax-future.md`.
- **No TensorFlow anywhere in `dune_bayes`.** TF / TFP must never be importable
  from the runtime namespace; a test enforces this (see *Dependencies*).

---

## Tooling

- **Python 3.12 only.** Do not add 3.9–3.11 compatibility shims — the old TF-era
  `≤3.11` cap is gone (ADR-0006).
- **`uv`** manages environments and dependencies. `.venv-torch` is the working env
  (torch 2.12 / Python 3.12).
- **`pyproject.toml`** with a PEP 621 `[project]` table is the single source of
  packaging truth. No `setup.py` (that lives only in the legacy `NAMpy/`).
- **`ruff`** is the only formatter + linter (replaces black / isort / flake8).
  **Line length 88.**
- **`mypy`** type-checks; type hints on **all public signatures**, mechanized via
  `disallow_untyped_defs = true` for `src/dune_bayes`. Once that is green, mypy is
  **CI-gating**. Full `--strict` is deliberately NOT the target pre-v1 (torch's
  typing makes it a tax with no numerical-correctness payoff).
- **`.pre-commit-config.yaml`** runs ruff (format + lint).

---

## Numerical rules

The operational form of "numerical stability is non-negotiable." New numerical
code must follow all of these; a violation is a bug, not a style nit.

1. **Positivity via `softplus`, never `exp` or `clamp`.** Every variance / scale
   parameter is `scale = softplus(rho)` (matches `spikes/variational_dense.py`,
   `_RHO_INIT = -3.0`). Family scale params (Normal σ, Gamma rate, …) go through
   softplus too — **as `softplus(x) + EPS`** (bare softplus underflows to exact
   0 near pre-link −104 in float32, poisoning `log_prob`). Never
   `transform_to(constraints.positive)` — that is `ExpTransform`, which
   overflows. Every family must pass an extreme pre-link (±1e4)
   finite-`log_prob` gate test.
2. **Stay in log-space.** Use `log_prob`, never `log(prob)`. Use `logsumexp` /
   `logaddexp` for the `T`-draw mixture and WAIC accumulation — never a hand-rolled
   `log(sum(exp(...)))`.
3. **No bare `log` / division / `sqrt` on learned quantities without a floor —
   but prefer reformulation over epsilon.** Reach for `log1p`, `expm1`,
   `logaddexp` first. When an epsilon is genuinely unavoidable, use the single
   named constant `EPS` (`1e-6` for float32, `1e-12` for float64), never a magic
   literal.
4. **Closed-form over Monte-Carlo whenever a closed form exists.** The
   Gaussian–Gaussian KL is analytic (it is in the spike) — never MC-estimate it.
5. **KL is never silently dropped.** Every `VariationalDense` stashes its KL each
   `forward()`; `collect_kl` must reach **all** of them (the spike asserts 6/6). A
   Bayesian module contributing zero KL is a bug, not an optimization.
6. **`validate_args=False` in the training hot path** (speed), **`True` in test
   fixtures** (catch invalid distribution params). Never validate in the loop.
7. **Every numerical claim has a test against an independent reference** —
   hand-computed KL, arviz reference for WAIC/LOO, etc. See *Testing*.

### Dtype, device, seeding

- **dtype: float32 by default** (torch default, 2× faster). **Cast to float64 only
  for log-likelihood / WAIC / LOO accumulation** (the `logsumexp`-over-draws and
  Pareto-k fits are where float32 bites) — do this inside `LogLikSampler` /
  `compare`. Everything else (forward pass, training) stays float32.
- **device: CPU by default; CUDA opt-in via an explicit `device=` argument.** This
  code runs on varied machines with different GPUs, so never assume a device.
  **MPS is allowed but unsupported** (distribution / `MixtureSameFamily` ops are
  flaky on MPS) — do not promise it.
- **seeding:** one `seed_everything(seed, deterministic=False)` helper seeds
  torch / numpy / Python `random`; `deterministic=True` opt-ins
  `torch.use_deterministic_algorithms` (slower — use in determinism tests and
  experiment configs). `sample_posterior_predictive` takes an optional `seed` /
  `generator` for reproducible bands. **The full re-seed protocol
  (`seed_everything → build → fit`) is exactly reproducible on CPU** — verified
  by experiment in `tests/model/test_reseed_determinism.py` (issue #90). The
  narrower caveat that remains: **two models built back-to-back within one RNG
  stream (no re-seed between them) do not draw identical noise** — each build
  advances the global stream, so don't compare freshly-built objects without
  re-seeding in between. Tests must respect this (see *Testing*).

---

## Testing

Philosophy (from the PRD): assert **external behavior at module boundaries** —
shapes, values against an independent reference, serialization round-trips,
warnings. **Never** assert private internals, layer wiring, or a single stochastic
draw.

- **`pytest`.** Tests in top-level `tests/`, mirroring `src/dune_bayes/`
  (`tests/layers/test_variational_dense.py`, …).
- **Four reference-test archetypes**, one per numerical claim:
  1. **Closed-form** — e.g. KL against a hand-computed Gaussian–Gaussian value.
  2. **Round-trip** — save/load with `max|Δw| == 0` (exact equality).
  3. **Shape** — `(batch, units)`, `[T, n, param_count]`, etc.
  4. **MC-convergence** — posterior-mean stabilizes as `T` grows, asserted with
     `pytest.approx(..., rel=…)` under a fixed seed (never a single-draw assert).
- **Tolerances are explicit and commented** — every `approx` / `atol` says *why*
  that value (MC noise vs float error). No magic tolerances.
- **The two ADR-0004 load-bearing claims now live in `tests/`**, not the spikes:
  (a) module-walk KL reaches every `VariationalDense`; (b) `state_dict` + config
  round-trip with `max|Δw| == 0`. These migrate into
  `tests/layers/test_variational_dense.py` when issue 0001 (#2) is built.
- **Spikes are throwaway and stay OUT of CI.** `spikes/` proved the design
  pre-build; once the real tests exist, the spikes may rot.
- **Coverage is tracked but not gated** pre-v1 — the boundary-behavior doctrine
  makes line-coverage a poor proxy. The real bar: a new public module ships with a
  boundary test. (Reaffirmed 2026-06-07 against a proposed numeric threshold.)
- **Correctness tests are never skippable** — no `skipif`/`xfail` on numerical
  correctness tests; they run unconditionally in every CI job. Slow suites
  (`hmc`, `experiment` markers) are opt-**in** via `-m`, so the core suite stays
  fast without ever excusing the correctness gates.
- **"Done" = the issue's acceptance criteria each have a corresponding test**, and
  the test names trace back to them.

---

## Docstrings & comments

- **Google-style docstrings** (`Args:` / `Returns:` / `Raises:`) on every public
  class and function. Trivial private helpers get a one-liner, not theater.
- **ADR + issue cross-reference is MANDATORY** in every decision-bearing module's
  docstring, e.g.
  `"""VariationalDense — the variational atom (ADR-0004, issue 0001)."""`. This is
  what keeps a 20-issue / 6-ADR design navigable from the code.
- **Comments explain *why*, never *what*.** A math-justification comment is
  required beside any non-obvious numerical step (why softplus, why this init, why
  log-space). `spikes/variational_dense.py` is the exemplar.
- **Use `CONTEXT.md` glossary spellings exactly** for domain terms (epistemic vs
  aleatoric, shape function, `prior_scale`-as-smoothness) so code and design
  language stay aligned.

---

## GitHub workflow

> **Hard rules — these override any default behavior:**
> - **Never commit, ever, unless explicitly told to in that moment.** Approval to
>   commit one thing is not standing approval. No autonomous commits.
> - **Commits are always authored as the user's account (`RMKruse`)** — never the
>   Codex identity.
> - **No `Co-Authored-By` trailer** on commits. (Overrides the default harness
>   trailer convention.)

- **Repo:** `git@github.com:RMKruse/dune-bayes.git`. Issues #1–#20 published:
  PRD = #1, slices = #2–#16, epics = #17–#20.
- **⚠️ Numbering foot-gun: `GitHub# = slice# + 1`.** Slice 0001 → issue **#2**.
  Always double-check which number you mean.
- **Branch per issue**, named off the **GitHub issue number**:
  `issue-0002-variational-dense` (zero-padded issue #, kebab summary). So slice
  0001 → issue #2 → branch `issue-0002-variational-dense`.
- **Integration target is the `dev` branch**, not `main`. PRs target `dev`;
  `dev` merges to `main` at release points. (Create `dev` only when instructed —
  it is the standing integration branch.)
- **Commits: conventional-ish, imperative mood**, matching the existing log
  (`Add …`, `Fix …`, `Port …`). Subject ≤ ~72 chars. No mandatory `feat:` / `fix:`
  prefixes.
- **One PR per slice, targeting `dev`, squash-merged** to keep `dev` linear. PR
  title references the issue; body **closes it** (`Closes #2`).
- **PR body always states:** the acceptance criteria, **how each was met**, and
  **what was tested**.
- **`dev` (and `main`) stay releasable** — slices cut end-to-end (atom → shape
  function → model → output → tests), so each merged slice keeps the package
  importable.

---

## Dependencies

- **Runtime (all hard, non-optional):** `torch` (≥2.12), `arviz` (comparison
  spine, Goal 3), `numpy`, `scikit-learn` (preprocessing), `matplotlib` (banded
  plots, Goal 1). `torch.distributions` is bundled with torch.
- **Pinning: lower-bound floors in `pyproject.toml`** (`torch>=2.12`, …), not hard
  pins — track fixes pre-v1.
- **Commit `uv.lock`** for reproducible contributor environments (floors for
  consumers, exact lock for devs).
- **`arviz` and `matplotlib` are core, not extras** — WAIC/LOO (Goal 3) and banded
  plots (Goal 1) are headline features.
- **Not v1 deps:** Pyro (kept as an option, not required — ADR-0006);
  TensorFlow / TFP (**must never be importable** from `dune_bayes`).
- **Enforce no-TF mechanically:** a test asserts `tensorflow` and
  `tensorflow_probability` are not importable from the `dune_bayes` namespace.

---

## Pointers

- Glossary & statistical contract: `CONTEXT.md`
- Decisions: `docs/adr/0001`…`0006`
- Scope & acceptance criteria: `docs/prd/0001-bayesian-feature-networks.md`,
  `docs/issues/`
- Verified design claims (historical, PyTorch): `spikes/`
