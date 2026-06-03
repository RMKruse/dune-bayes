# Issue 0022: DataModule walking skeleton — tabular data → model-ready tensors + N

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

A `DataModule` in a new `data` component: constructed from a pandas DataFrame plus the response-column name, it produces the model-ready inputs the package already consumes — the per-feature tensor dict and the target tensor — and exposes `n_obs` (training-set size N). The formula-string and fit surfaces accept a DataModule, so KL/N is wired from the data rather than hand-passed: building a model from formula + DataModule yields the documented mean-NLL + KL/N objective without the user ever supplying `n_obs`.

No preprocessing yet — raw numeric columns pass through as float32 tensors of shape (n, 1). This is the tracer bullet the preprocessing slices (numeric scaling, categorical coding) build on.

pandas becomes an explicit runtime dependency (lower-bound floor; it is already present transitively via arviz).

## Acceptance criteria

- [ ] A DataModule built from a DataFrame + response name yields a feature dict (float32, shape `(n, 1)` per feature) and target tensor matching the model's existing fit contract
- [ ] `n_obs` equals the training-set size and is exposed; building a model from formula + DataModule auto-wires the KL divisor (KL/N) with no explicit `n_obs` argument
- [ ] End-to-end: a BayesianNAMLSS built from formula + DataModule fits on toy data; loss history is finite and the KL term reflects N
- [ ] pandas declared as a runtime dependency (floor pin) in `pyproject.toml`
- [ ] Boundary tests cover shapes, dtype, and `n_obs` (no private-internals asserts)

## Blocked by

None - can start immediately.
