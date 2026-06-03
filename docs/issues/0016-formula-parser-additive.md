# Issue 0016: Formula-string parser — additive terms

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

The user-facing formula surface the PRD's Solution section promises but which `dev` currently lacks: a parser that turns a string like `y ~ BayesianMLP(x1) + MLP(x2) + NeuralLinearMLP(x3)` into the `dict[str, nn.Module]` formula that `BayesianNAMLSS` already consumes, and a way to construct a model directly from that string.

Scope is **additive terms only** (`+`-separated single-input terms); interaction terms (`:`) are issue 0017. Each term names a registered shape function and its input feature, with optional per-term keyword arguments (e.g. `BayesianMLP(x1, prior_scale=0.5)`, `NeuralLinearMLP(x2, hidden_dims=(32, 32))`) so regularization strength is controllable term-by-term (user story 7). The response name left of `~` is captured. Unknown shape-function names raise a clear error referencing the registry.

Per CLAUDE.md / ADR-0006 this is **reimplemented from scratch in PyTorch**, not ported or imported from NAMpy's TF `FormulaHandler`.

## Acceptance criteria

- [ ] A formula string with `+`-separated terms parses into `{feature_name: shape_function_instance}` resolvable via `ShapeFunctionRegistry`
- [ ] Per-term keyword arguments (e.g. `prior_scale`, `hidden_dims`) are parsed and forwarded to the shape-function constructor
- [ ] A `BayesianNAMLSS` can be constructed from a formula string and trains end-to-end on a toy dataset
- [ ] Mixing Bayesian and deterministic terms in one formula is supported (deterministic terms contribute zero KL)
- [ ] An unknown shape-function name raises a clear, actionable error
- [ ] Boundary tests cover: term parsing, kwarg forwarding, response-name capture, and the unknown-name error

## Blocked by

None - can start immediately.
