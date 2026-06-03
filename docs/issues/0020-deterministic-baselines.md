# Issue 0020: Deterministic baseline shape functions (MLP, ResNet)

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

Register pure-deterministic baseline shape functions so users can run direct Bayes-vs-deterministic effect comparisons and include a non-Bayesian baseline in WAIC/LOO (user stories 21, 22). `dev` currently registers only `BayesianMLP` and `NeuralLinearMLP` (the latter still variational in its output layer), so a partial-Bayesian formula has no actual deterministic net to drop in.

Add deterministic `MLP` and `ResNet` shape functions (plain `nn.Linear` stacks, no `VariationalDense`) registered in `ShapeFunctionRegistry`. They implement the shape-function contract `(batch, in_features) -> (batch, param_count)` and behave as **zero-variance / zero-KL contributors** inside `BayesianNAMLSS`, so `collect_kl` ignores them and the model is well-defined for fully- or partially-deterministic formulas.

## Acceptance criteria

- [ ] `MLP` and `ResNet` deterministic shape functions are registered and resolvable by name
- [ ] Each contributes zero KL inside `BayesianNAMLSS` (`collect_kl` sees nothing from them)
- [ ] A partial-Bayesian model (e.g. `BayesianMLP(x1) + MLP(x2)`) trains end-to-end
- [ ] A fully-deterministic model is comparable against a Bayesian one via WAIC/LOO
- [ ] Effect samples from a deterministic net show (near-)zero epistemic spread
- [ ] Boundary tests cover registration, the zero-KL contract, and output shape

## Blocked by

None - can start immediately.
