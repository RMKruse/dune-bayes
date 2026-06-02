# 1. Mean-field variational inference as the inference engine

Date: 2026-06-02

## Status

Accepted

## Context

neural-bamlss replaces NAMpy's deterministic feature networks with Bayesian ones.
The user's goals, in priority order, are: (1) epistemic uncertainty / credible
intervals on each feature's effect; (2) priors acting as principled
regularization, most valuable on small data; (3) a principled Bayesian
model-comparison criterion across a handful of candidate formulas. The user also
raised **Bayes Factors** as a desired comparison tool.

Constraints discovered:

- The codebase is built entirely on the Keras functional API, `tfp.layers`, and
  Keras `.fit()`. The NAMLSS forward pass sums per-feature `Dense`-based nets and
  pipes the sum through `tfp.layers.DistributionLambda`; loss is the NLL
  `-y_hat.log_prob(y)`.
- Target datasets **vary a lot** in size (hundreds to >50k rows).
- Only **a handful** of models are compared in a typical workflow.
- True Bayes Factors require the marginal likelihood `p(y|M) = ∫ p(y|θ)p(θ)dθ`,
  which is intractable and prior-hypersensitive for NN-sized parameter spaces.

Candidate inference methods: mean-field VI (`tfp` variational layers),
full-covariance / normalizing-flow VI, MC-Dropout, deep ensembles, MCMC/HMC.

## Decision

Use **mean-field variational inference** via `tfp` variational Dense layers as
the single inference engine for v1.

- Shape-function `Dense` layers become variational (e.g.
  `tfp.layers.DenseVariational` / `DenseFlipout`).
- Loss = NLL + (weight-KL / N) = negative ELBO.
- Credible intervals on effects come from Monte-Carlo sampling of weights at
  prediction time.
- Model comparison uses **WAIC / PSIS-LOO**, with the **ELBO as a secondary
  (biased) evidence proxy**. Literal marginal-likelihood Bayes Factors are out of
  scope for v1.
- Priors and the likelihood are kept **inference-agnostic** so an MCMC backend
  can be added later for the small-data, gold-standard regime without reworking
  the model definition.

## Consequences

**Positive**

- The only method that degrades gracefully across the full data-scale range
  (minibatched VI is scale-invariant; the codebase already minibatches).
- Drop-in for the existing `forward()` methods and Keras `.fit()` loop — minimal
  architectural disruption.
- Satisfies all three priority goals: explicit priors (goal 2), sampled credible
  intervals (goal 1), tractable WAIC/LOO + ELBO (goal 3).
- Keeps a clean seam for a future MCMC backend.

**Negative / accepted trade-offs**

- Mean-field underestimates posterior uncertainty → credible bands on effects
  will tend to be too narrow.
- The ELBO is a biased estimate of the log evidence.
- True Bayes Factors are foregone; if ever required, they need the deferred MCMC
  backend plus bridge/thermodynamic integration.

## Alternatives considered

- **MCMC/HMC** — gold-standard posterior and true marginal likelihoods, but
  hostile to the Keras `.fit()` architecture and infeasible on the large end of
  the data range. Deferred behind the prior/likelihood seam.
- **Full-covariance / flow VI** — richer posteriors, more faithful uncertainty,
  but more complexity and slower training; can be a later upgrade of the same VI
  engine.
- **MC-Dropout** — nearly free (dropout already present) but only implicit
  priors (fails goal 2) and poor for model evidence.
- **Deep ensembles** — good epistemic uncertainty but no explicit priors and no
  evidence/Bayes-factor story.
