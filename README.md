# DUNE — Distributional Uncertainty in Neural-additive Estimation

Bayesian additive models for location, scale, and shape — neural shape
functions with mean-field variational inference, in PyTorch. Distributed
as the `dune-bayes` package (`import dune_bayes`).

`dune-bayes` is the neural analog of the
[BAMLSS](https://cran.r-project.org/package=bamlss) R package. Each feature
gets its own small neural network (a *shape function*) that contributes to
**every parameter** of a response distribution (location *and* scale *and*
shape), and the network weights carry a variational posterior — so every
feature effect comes with a credible band, not just a point curve.

Three goals, in priority order:

1. **Epistemic uncertainty on effects** — credible intervals around each
   feature's shape function, separated from the family's aleatoric noise.
2. **Priors as principled regularization** — the per-feature `prior_scale` is
   the direct analog of an mgcv smoothing parameter (fixed, empirical-Bayes,
   or fully hierarchical).
3. **Principled model comparison** — WAIC and PSIS-LOO (via
   [arviz](https://python.arviz.org)) across candidate formulas, including
   deterministic baselines.

## Status

Pre-v1 research software (`0.1.0.dev0`). The statistical design is settled
(see [`docs/adr/`](docs/adr/)) and the core path — formula → fit → effect
bands → WAIC/LOO — works and is tested, but the API may still change without
deprecation. Python 3.12+, CPU by default, CUDA opt-in.

## Install

Not yet on PyPI. Install from a clone (we use [uv](https://docs.astral.sh/uv/)):

```sh
git clone git@github.com:RMKruse/dune-bayes.git
cd dune-bayes
uv pip install -e .
```

## Quick example: fit → plot → compare

```python
import torch
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.families import NormalFamily
from dune_bayes.sampling import sample_effects
from dune_bayes.plotting import plot_effect_ribbon
from dune_bayes.compare import compare

# Toy data: nonlinear effect on the mean, Gaussian noise.
n = 500
x1, x2 = torch.rand(n, 1), torch.rand(n, 1)
y = torch.sin(6 * x1).squeeze(-1) + 0.5 * x2.squeeze(-1) + 0.1 * torch.randn(n)
X = {"x1": x1, "x2": x2}

# mgcv-style formula; one shape function per term, Bayesian or deterministic.
model = BayesianNAMLSS.from_formula(
    "y ~ BayesianMLP(x1) + NeuralLinearMLP(x2)", family=NormalFamily(), n_obs=n
)
model.fit(X, y, epochs=200)

# Per-feature effect bands: T posterior draws -> centered 90% credible ribbon.
draws = sample_effects(model, X, T=200)        # {feature: Tensor[T, n, param_count]}
plot_effect_ribbon(draws["x1"], x1.squeeze(-1))

# Bayes-vs-deterministic comparison, ranked by PSIS-LOO.
baseline = BayesianNAMLSS.from_formula(
    "y ~ MLP(x1) + MLP(x2)", family=NormalFamily(), n_obs=n
)
baseline.fit(X, y, epochs=200)
print(compare({"bayesian": model, "deterministic": baseline}, X, y))
```

Shape functions available in formulas: `BayesianMLP` (fully variational),
`NeuralLinearMLP` (deterministic hidden layers, variational output — cheaper),
and the deterministic baselines `MLP` and `ResNet`. Interactions are joint
nets over multiple inputs (`BayesianMLP(x1):BayesianMLP(x2)`); categorical
features become Bayesian embeddings — the neural analog of a BAMLSS random
effect, with partial pooling of rare levels. Families shipped so far:
`NormalFamily`, `StudentTFamily`, `GammaFamily`.

## Documentation

- [`CONTEXT.md`](CONTEXT.md) — the statistical contract and glossary (start here).
- [`docs/adr/`](docs/adr/) — the six design decisions (inference engine,
  priors-as-smoothness, predictive/comparison architecture, the
  `VariationalDense` atom, categoricals & interactions, PyTorch backend).
- [`docs/prd/`](docs/prd/) and [`docs/issues/`](docs/issues/) — scope and
  acceptance criteria.

## Citing

There is no dune-bayes paper yet. If you use the package, please cite the
repository, and the BAMLSS framework it builds on:

> Umlauf, N., Klein, N., & Zeileis, A. (2018). BAMLSS: Bayesian additive
> models for location, scale, and shape (and beyond). *Journal of
> Computational and Graphical Statistics*, 27(3), 612–627.

## License

[MIT](LICENSE)
