# neural-bamlss issues

Tracer-bullet vertical slices for the PRD in `docs/prd/0001-bayesian-feature-networks.md`.
Each slice cuts end-to-end (atom → shape function → model → output → tests) and is
verifiable on its own. All slices are AFK (every design decision is locked in ADR-0001…0005).

Issues are also published on GitHub (`RMKruse/neural-bamlss`): PRD = #1, slices = #2–#16,
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
