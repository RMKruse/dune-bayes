# Issue 0017: Formula-string parser — interaction terms

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

Extend the formula parser (issue 0016) so a `:`-joined term such as `BayesianMLP(x1):BayesianMLP(x2)` resolves to a **single joint Bayesian net over both inputs**, with the network name taken from the first term (ADR-0005). The joint net already exists at the object level (issue 0013 / GitHub #14); this slice is the parser path that lets a user express it in a formula string.

The parser must build one shape function whose input is the concatenation/pairing of the named features and register it under a combined key, so `BayesianNAMLSS` treats it as one additive contributor. Per-term kwargs on the joint term (e.g. `prior_scale`) apply to the joint net.

## Acceptance criteria

- [ ] `Net(x1):Net(x2)` parses to a single joint shape function over both inputs (name from the first term, per ADR-0005)
- [ ] A formula mixing additive and interaction terms (`y ~ BayesianMLP(x1) + BayesianMLP(x2):BayesianMLP(x3)`) constructs and trains end-to-end
- [ ] The joint term contributes a single KL term and a single entry in `EffectSampler`/interaction-surface output
- [ ] Per-term kwargs on the interaction term are forwarded to the joint net
- [ ] Boundary tests cover joint-term parsing, the combined-key contract, and mixed additive+interaction formulas

## Blocked by

- Issue 0016 (Formula-string parser — additive terms)
