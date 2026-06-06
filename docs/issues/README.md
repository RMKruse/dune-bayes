# dune-bayes issues

> **Backend superseded (ADR-0006, 2026-06-03).** These slices were written against
> the TensorFlow / Keras 2 / TFP stack; the package now targets **PyTorch** (JAX/
> NumPyro is the numerical future). The *behaviour* in each acceptance criterion
> stands, but TF/TFP API nouns translate per ADR-0006: `add_loss` → explicit
> module-walk KL; `tfd.*` → `torch.distributions.*`; `tf.keras.Model` →
> `nn.Module`; `.keras`/SavedModel/H5 → `state_dict`. Issue 0001 (`VariationalDense`
> atom) and 0015 (save/load) change the most; the spike-verified claims they cite
> are historical and owed re-verification on PyTorch.

Tracer-bullet vertical slices for the PRD in `docs/prd/0001-bayesian-feature-networks.md`.
Each slice cuts end-to-end (atom → shape function → model → output → tests) and is
verifiable on its own. All slices are AFK (every design decision is locked in ADR-0001…0006).

Issues are also published on GitHub (`RMKruse/dune-bayes`): PRD = #1, slices = #2–#16,
epics = #17–#20. GitHub number = `slice number + 1`.

## Epics

| Epic | GitHub | PRD goal | Encompasses (slice / GitHub#) |
|------|--------|----------|-------------------------------|
| Foundation — Bayesian networks & training surface | #17 | foundation | 0001/#2, 0002/#3, 0003/#4, 0004/#5, 0014/#15, 0015/#16 |
| Epistemic uncertainty & posterior predictive | #18 | Goal 1 | 0005/#6, 0006/#7, 0007/#8, 0008/#9, 0010/#11, 0013/#14 |
| Priors as regularization & random effects | #19 | Goal 2 | 0011/#12, 0012/#13 |
| Principled model comparison | #20 | Goal 3 | 0009/#10 |

## Slices

| # | GitHub | Slice | Epic | Blocked by |
|---|--------|-------|------|-----------|
| 0001 | #2 | VariationalDense atom | Foundation | — |
| 0002 | #3 | BayesianMLP shape function | Foundation | 0001 |
| 0003 | #4 | BayesianNAMLSS walking skeleton (fit + KL/N) | Foundation | 0002 |
| 0004 | #5 | KL warm-up auto-injection | Foundation | 0003 |
| 0005 | #6 | EffectSampler workhorse | Goal 1 | 0003 |
| 0006 | #7 | Centered epistemic effect ribbons | Goal 1 | 0005 |
| 0007 | #8 | LogLikSampler + MixtureSameFamily predictive | Goal 1 | 0003 |
| 0008 | #9 | Response-level predictive bands | Goal 1 | 0007 |
| 0009 | #10 | Model comparison: WAIC / LOO / compare | Goal 3 | 0007 |
| 0010 | #11 | Bayesian intercept | Goal 1 | 0003 |
| 0011 | #12 | PriorScale handle: EB + hierarchical tiers | Goal 2 | 0001 |
| 0012 | #13 | Categoricals as random effects | Goal 2 | 0006, 0011 |
| 0013 | #14 | Interactions as joint Bayesian nets + surfaces | Goal 1 | 0006 |
| 0014 | #15 | NeuralLinearMLP shape function | Foundation | 0003 |
| 0015 | #16 | Save/load round-trip | Foundation | 0003 |

Dependency root is **0001**; the skeleton **0003** unblocks most downstream work.

## Gap-fill slices (post-implementation review of `dev`, 2026-06-03)

> **Numbering note:** the `GitHub# = slice# + 1` rule applied only to the original
> batch. Epics took #17–#20 and PRs/merges consumed later numbers, so these slices
> carry their **actual** GitHub numbers below — do not infer them from the slice #.

These close gaps between the PRD's stated scope and what shipped on `dev`: the
formula-string surface, multiple response families, registered deterministic
baselines, and a `feature_dropout` defaulting bug.

| # | GitHub | Slice | Epic | Blocked by |
|---|--------|-------|------|-----------|
| 0016 | #37 | Formula-string parser — additive terms | Foundation | — |
| 0017 | #41 | Formula-string parser — interaction terms | Foundation | 0016 |
| 0018 | #38 | BaseFamily contract | Foundation | — |
| 0019 | #42 | Concrete distributional families | Foundation | 0018 |
| 0020 | #39 | Deterministic baseline shape functions (MLP, ResNet) | Foundation | — |
| 0021 | #40 | Fix feature_dropout default no-op (bug) | Foundation | — |

## DataModule slices (gap-fill, 2026-06-04)

The `data/` component promised in the package layout (CLAUDE.md) was never built:
the PRD said "reuse NAMpy's `DataModule`", but the ADR-0006 amendment made NAMpy
reference-only, leaving the gap. These slices reimplement it: pandas DataFrame →
model-ready tensors, sklearn-style preprocessing, and the N-for-KL/N convenience.
pandas becomes an explicit runtime dependency (floor pin) with slice 0022.

| # | GitHub | Slice | Epic | Blocked by |
|---|--------|-------|------|-----------|
| 0022 | #49 | DataModule walking skeleton: tabular data → tensors + N | Foundation | — |
| 0023 | #50 | Numeric preprocessing: train-fit scaling + inverse for plot axes | Foundation | 0022 |
| 0024 | #51 | Categorical encoding feeding random effects | Goal 2 | 0022 |
| 0025 | #52 | DataModule state round-trip | Foundation | 0023, 0024 |
| 0026 | #53 | Minibatching via torch.utils.data (KL/N stays full-data N) | Foundation | 0022 |

Dependency root is **0022** (labeled `blocking` on GitHub); 0023/0024 parallelize
after it.
