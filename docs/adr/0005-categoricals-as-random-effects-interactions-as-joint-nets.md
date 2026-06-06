# 5. Bayesian categorical embeddings as random effects; interactions as joint nets

Date: 2026-06-02

## Status

Accepted

## Context

Two feature types beyond plain continuous inputs need a defined Bayesian
treatment: **interactions** and **categoricals**.

- *Interactions:* the formula parser already resolves `Net(x1):Net(x2)` to a
  **single joint net over multiple inputs** — the network name is taken from the
  first term and the remaining features are added as inputs
  (`nampy/formulas/formulas.py:223`). There is no product-of-nets construct.
- *Categoricals:* a categorical feature passes through a first mapping (embedding
  table / lookup → Dense) before the shape net. The open question is whether that
  first mapping carries a weight posterior or stays a point estimate.

The user's priority goals are epistemic uncertainty on effects (goal 1) and
priors as regularization on small data (goal 2). The hard case is a categorical
with many levels and few rows per level.

## Decision

- **Interactions need no new machinery.** An interaction term is just a Bayesian
  shape function (e.g. `BayesianMLP`) consuming multiple inputs via the existing
  `:` joint-net semantics. `sample_posterior_predictive` handles it unchanged. The
  only consequence is plotting: a 1D ribbon does not generalize, so an interaction
  renders as a posterior-mean **surface** plus a separate **epistemic-SD surface**
  (or sliced 1D bands at fixed values of the other feature).

- **Categorical first mapping is Bayesian.** The embedding/category weights get a
  weight posterior under the per-feature `prior_scale` (ADR-0002), like every
  other `VariationalDense` weight (ADR-0004). This yields a **credible interval
  per level**, and **rare levels shrink toward the prior** — i.e. partial pooling.
  This is explicitly framed and documented as the neural analog of a **BAMLSS
  random effect** (BAMLSS uses a shared variance-component / IG prior on
  categorical effects in its MCMC). A point-embedding-with-Bayesian-head option is
  rejected as the default because it makes rare levels look as confident as common
  ones.

## Consequences

**Positive**

- Per-level credible intervals and automatic partial pooling — a direct fidelity
  win for the "neural analog of BAMLSS" identity and a strong small-data behavior (goal 2).
- Interactions cost nothing new architecturally; they reuse the joint-net path and
  the single sampling workhorse.
- Uniform story: every weight in the model, including embeddings, is a
  `VariationalDense`-style variational weight under the same prior machinery.

**Negative / accepted trade-offs**

- A Bayesian embedding table is more parameters to fit variationally than a point
  embedding; high-cardinality categoricals raise cost and can be harder to train
  (mitigated by the shared per-feature prior acting as the pooling variance).
- Interaction surfaces need dedicated 2D plotting (mean surface + SD surface /
  slices) rather than reusing the 1D ribbon code.
- The per-feature `prior_scale` now doubles as the random-effect variance
  component for categoricals; users tuning it affect pooling strength, which must
  be documented.

## Alternatives considered

- **Point embedding, Bayesian downstream only** — cheaper and simpler, but rare
  levels look falsely confident (no per-level epistemic uncertainty, no pooling).
  Rejected as default.
- **Bayesian embedding without the random-effect framing** — same mechanics but
  treats it as just another Bayesian layer; rejected because the partial-pooling /
  variance-component interpretation is the point of fidelity to BAMLSS and guides
  how users set `prior_scale`.
- **A dedicated random-effect term type** separate from embeddings — more explicit
  but adds formula surface; deferred, since the Bayesian embedding already
  delivers the behavior.
