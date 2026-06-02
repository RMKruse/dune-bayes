# Epic: Foundation — Bayesian feature networks & training surface

> GitHub: #17 · Source: PRD `docs/prd/0001-bayesian-feature-networks.md` (#1)

The Bayesian feature-network machinery and the training / persistence surface every other
epic builds on. Delivers a `BayesianNAMLSS` that trains a Bayesian additive formula
end-to-end on the familiar `compile`/`fit` surface, the shape-function library, and a
supported save/load path.

## Encompasses

- 0001 / #2 — VariationalDense atom
- 0002 / #3 — BayesianMLP shape function
- 0003 / #4 — BayesianNAMLSS walking skeleton (fit + KL/N)
- 0004 / #5 — KL warm-up auto-injection
- 0014 / #15 — NeuralLinearMLP shape function
- 0015 / #16 — Save/load round-trip
