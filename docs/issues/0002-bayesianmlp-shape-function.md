# Issue 0002: BayesianMLP shape function

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The flagship fully-variational shape function, registered alongside the deterministic nets. Every `Dense` becomes a `VariationalDense`, so uncertainty propagates through the function shape, not just a final rescaling. Implements the existing `ShapeFunction` contract (`forward`/`build` → `tf.keras.Model`, `output_dimension = family.param_count`). Internal per-layer dropout is stripped (the weight posterior is the stochasticity). Selectable per term in the formula, e.g. `y ~ BayesianMLP(x1) + MLP(x2)`.

## Acceptance criteria

- [ ] `BayesianMLP` is registered in `ShapeFunctionRegistry` and resolvable by name from a formula
- [ ] `build()` returns a `tf.keras.Model` mapping one feature to `(batch, param_count)`
- [ ] The built model carries the variational KL in `model.losses`
- [ ] Internal per-layer dropout is absent
- [ ] A formula mixing `BayesianMLP` and a deterministic net parses and builds

## Blocked by

- Issue 0001 (VariationalDense atom)
