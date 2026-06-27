# Reader Workflow: Formula, Fit, Bands, Compare

This short tutorial mirrors the paper-facing DUNE workflow with the package API:
write a formula, fit a Bayesian additive distributional model, inspect an
epistemic effect ribbon for one shape function, then draw response-level
posterior predictive bands. The example is deliberately small so the code can
run as a documentation smoke test; use larger data, more epochs, and the default
draw counts for analysis.

The important distinction is:

- An effect ribbon is per-feature. It summarizes epistemic uncertainty in one
  learned shape function, usually centered so the curve describes shape rather
  than the model intercept.
- A response band is per-observation. It summarizes the posterior predictive
  distribution and therefore includes both epistemic uncertainty from posterior
  weight draws and aleatoric uncertainty from the response family.
- A variance decomposition reports that same split numerically:
  aleatoric uncertainty is the average family variance, while epistemic
  uncertainty is the variance across posterior component means.

## Minimal Runnable Example

```python
# docs-smoke: reader-workflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dune_bayes.families import NormalFamily
from dune_bayes.metrics import variance_decomposition
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.plotting import plot_dist, plot_effect_ribbon, predictive_quantiles
from dune_bayes.sampling import draw_predictive, sample_effects
from dune_bayes.utils import seed_everything

seed_everything(11)

n = 24
x1 = torch.linspace(-1.0, 1.0, n).unsqueeze(-1)
x2 = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
y = (
    torch.sin(3.0 * x1).squeeze(-1)
    + 0.4 * x2.squeeze(-1)
    + 0.08 * torch.randn(n)
)
X = {"x1": x1, "x2": x2}

family = NormalFamily()
model = BayesianNAMLSS.from_formula(
    "y ~ BayesianMLP(x1, hidden_dims=(4,))"
    " + NeuralLinearMLP(x2, hidden_dims=(4,))",
    family=family,
    n_obs=n,
)
model.fit(X, y, epochs=2, lr=0.01, warmup_epochs=1)

# Per-feature epistemic uncertainty: one tensor per shape function.
effects = sample_effects(model, X, T=6)
effect_draws = effects["x1"]
effect_ax = plot_effect_ribbon(
    effect_draws,
    x1.squeeze(-1),
    credible_interval=0.90,
    feature_name="x1",
)

# Response-level posterior predictive uncertainty: epistemic plus aleatoric.
draws = draw_predictive(model, X, T=6)
predictive_band = predictive_quantiles(
    draws.predictive,
    credible_interval=0.90,
    n_samples=64,
    seed=13,
)
response_ax = plot_dist(draws.predictive, y, n_samples=64, seed=13)

# Numeric split of the response-level predictive variance.
variance = variance_decomposition(model, draws.summed_samples)

plt.close(effect_ax.figure)
plt.close(response_ax.figure)
```

The formula creates two shape functions: a fully variational `BayesianMLP` for
`x1` and a cheaper `NeuralLinearMLP` for `x2`. Both contribute additively to the
Normal family's location and scale parameters. The effect ribbon for `x1` is
therefore a posterior summary of that feature's contribution to the selected
distributional parameter, not a prediction interval for future observations.
The same formula-fit-band workflow is supported for the shipped paper-artifact
families: `NormalFamily`, `GammaFamily`, `StudentTFamily`, `JohnsonSUFamily`,
`NegativeBinomialFamily`, and `BetaFamily`; each family sets the output width
through its `param_count` and applies its own parameter links.

`draw_predictive` uses coherent posterior weight draws to build a
`MixtureSameFamily` posterior predictive distribution. `plot_dist` and
`predictive_quantiles` operate on that response-level distribution, so their
bands are wider whenever the family has meaningful aleatoric uncertainty. The
`variance_decomposition` output exposes the law-of-total-variance split as
`aleatoric`, `epistemic`, and `total`.

## Model Comparison

After fitting candidate formulas, rank them with the comparison API:

```python
# illustrative: run after fitting two or more candidate formulas.
from dune_bayes.compare import compare

ranking = compare({"candidate_a": model, "candidate_b": baseline}, X, y)
print(ranking)
```

For a real comparison, fit at least two candidate models and use the default
posterior draw counts. The tiny tutorial run above is only a reproducible API
check, not evidence for a paper claim.
