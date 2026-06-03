"""Model comparison: WAIC / LOO / compare — arviz-backed (issue 0009 / GitHub #10).

Design:
  - to_inference_data() packages LogLikSampler output into an arviz DataTree
    (arviz 1.x InferenceData) with the (chain, draw, obs) shape convention.
  - waic() implements WAIC2 from first principles (Vehtari et al. 2017, eq. 12)
    because arviz 1.x dropped az.waic; uses logsumexp for numerical stability.
  - loo() delegates to az.loo (PSIS-LOO) and surfaces Pareto-k warnings.
  - compare() delegates to az.compare (stacking weights by default).
  - elbo() returns the negative ELBO-loss as a biased evidence proxy.
    No Bayes Factor is computed (ADR-0001).
  - pointwise_loglik stays float64 throughout (CLAUDE.md dtype rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import torch

from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.sampling.log_lik_sampler import T_EVAL, LogLikSampler


@dataclass
class WaicData:
    """Results from a WAIC computation (issue 0009 / GitHub #10).

    Attributes:
        elpd: Log-scale expected log-pointwise predictive density (higher is better).
            elpd = lppd - p_waic.
        p: Effective number of parameters (p_waic = sum_i var_T(ll_i_t)).
        se: Standard error of elpd: sqrt(n * var_i(lppd_i - p_waic_i)).
        n_data_points: Number of observations n.
        n_samples: Number of posterior draws T used.
    """

    elpd: float
    p: float
    se: float
    n_data_points: int
    n_samples: int


def to_inference_data(
    model: BayesianNAMLSS,
    X: dict[str, torch.Tensor],
    y: torch.Tensor,
    T: int = T_EVAL,
    var_name: str = "y",
) -> Any:
    """Convert model + data to an arviz DataTree with pointwise log-likelihood.

    Runs T posterior draws via LogLikSampler and packages the resulting
    (T, n) float64 log-likelihood matrix into arviz's (chain, draw, obs) layout.
    Suitable as input to az.loo / az.compare.

    Args:
        model: Fitted BayesianNAMLSS instance.
        X: Feature dict {name: Tensor[n, in_features]}.
        y: Response tensor of shape (n,).
        T: Number of posterior weight draws. Defaults to T_EVAL=1000.
        var_name: Name for the log-likelihood variable (default 'y').

    Returns:
        xr.DataTree with 'posterior' (dummy) and 'log_likelihood' groups.
        log_likelihood[var_name] has shape (chain=1, draw=T, obs=n) float64.
    """
    sampler = LogLikSampler()
    result = sampler(model, X, y, T=T)
    # (T, n) → (1, T, n) for arviz (chain, draw, obs) convention.
    ll_np = result.pointwise_loglik.numpy()[np.newaxis, ...]  # (1, T, n) float64
    # az.loo requires a posterior group to compute the relative MCMC efficiency.
    return az.from_dict(
        {
            "posterior": {"dummy": np.zeros((1, T))},
            "log_likelihood": {var_name: ll_np},
        }
    )


def _waic_from_loglik(ll: torch.Tensor) -> WaicData:
    """Compute WAIC2 from a (T, n) float64 log-likelihood tensor.

    WAIC2 formula (Vehtari et al. 2017, eq. 12):
        lppd_i  = logsumexp_T(ll_i_t) - log(T)      [log-scale; numerical rule 2]
        p_waic_i = var_T(ll_i_t)                      [biased variance, correction=0]
        elpd_waic = sum_i(lppd_i - p_waic_i)

    Biased variance (correction=0) is standard: the WAIC2 formula divides by T,
    not T-1 (Vehtari et al. 2017).
    """
    T_val = ll.shape[0]
    n_val = ll.shape[1]
    log_T = torch.log(torch.tensor(float(T_val), dtype=torch.float64))
    # logsumexp over T draws stays in log-space (numerical rule 2).
    lppd_i = torch.logsumexp(ll, dim=0) - log_T  # (n,)
    p_waic_i = ll.var(dim=0, correction=0)  # (n,) biased variance
    elpd_i = lppd_i - p_waic_i  # (n,)
    elpd = float(elpd_i.sum())
    p = float(p_waic_i.sum())
    se = float(torch.sqrt(torch.tensor(n_val, dtype=torch.float64) * elpd_i.var()))
    return WaicData(
        elpd=elpd, p=p, se=se, n_data_points=int(n_val), n_samples=int(T_val)
    )


def waic(
    model: BayesianNAMLSS,
    X: dict[str, torch.Tensor],
    y: torch.Tensor,
    T: int = T_EVAL,
) -> WaicData:
    """Compute WAIC (WAIC2) from T posterior draws.

    arviz 1.x dropped az.waic; this implements the WAIC2 formula from first
    principles using logsumexp for numerical stability (CLAUDE.md rule 2).

    Args:
        model: Fitted BayesianNAMLSS instance.
        X: Feature dict {name: Tensor[n, in_features]}.
        y: Response tensor of shape (n,).
        T: Posterior draws. Defaults to T_EVAL=1000.

    Returns:
        WaicData with elpd (higher is better), effective p, SE, and metadata.
    """
    sampler = LogLikSampler()
    result = sampler(model, X, y, T=T)
    return _waic_from_loglik(result.pointwise_loglik)


def loo(
    model: BayesianNAMLSS,
    X: dict[str, torch.Tensor],
    y: torch.Tensor,
    T: int = T_EVAL,
    var_name: str = "y",
) -> Any:
    """Compute PSIS-LOO via arviz.

    Pareto-k reliability warnings are surfaced by arviz when T is too small
    for accurate importance weighting (k > 0.5 for any observation).
    Rule of thumb: T >= 4 * n_obs for reliable PSIS-LOO.

    Args:
        model: Fitted BayesianNAMLSS instance.
        X: Feature dict {name: Tensor[n, in_features]}.
        y: Response tensor of shape (n,).
        T: Posterior draws. Defaults to T_EVAL=1000.
        var_name: Log-likelihood variable name.

    Returns:
        arviz ELPDData with elpd, se, p, and Pareto-k diagnostics.

    Raises:
        ValueError: If the model has no stochastic weight posterior (i.e. no
            VariationalDense layers), PSIS cannot fit a Pareto tail to identical
            importance weights. Use waic() for deterministic baselines; only pass
            Bayesian models (BayesianMLP / NeuralLinearMLP) to loo() and compare().
    """
    idata = to_inference_data(model, X, y, T=T, var_name=var_name)
    return az.loo(idata, var_name=var_name)


def compare(
    models: dict[str, BayesianNAMLSS],
    X: dict[str, torch.Tensor],
    y: torch.Tensor,
    T: int = T_EVAL,
    var_name: str = "y",
) -> Any:
    """Rank models by PSIS-LOO ELPD using az.compare.

    Each model's LOO is computed independently, then az.compare ranks them
    best-to-worst (highest ELPD = index 0).

    Args:
        models: Dict mapping model name → fitted BayesianNAMLSS.
        X: Feature dict shared by all models.
        y: Response tensor shared by all models.
        T: Posterior draws per model. Defaults to T_EVAL=1000.
        var_name: Log-likelihood variable name.

    Returns:
        pandas DataFrame from az.compare, ordered best-to-worst by ELPD.
    """
    loo_results = {
        name: loo(m, X, y, T=T, var_name=var_name) for name, m in models.items()
    }
    return az.compare(loo_results)


def elbo(
    model: BayesianNAMLSS,
    X: dict[str, torch.Tensor],
    y: torch.Tensor,
) -> float:
    """Return the ELBO as a biased secondary evidence proxy.

    ELBO = -(mean_NLL + KL/N) is a lower bound on log-evidence. It is biased
    (depends on N, KL weight, and warm-up schedule) and should NOT be used as
    a primary model-comparison criterion — use loo() or compare() instead.

    No Bayes Factor is computed (ADR-0001, issue 0009 / GitHub #10).

    Args:
        model: Fitted BayesianNAMLSS instance.
        X: Feature dict {name: Tensor[n, in_features]}.
        y: Response tensor of shape (n,).

    Returns:
        Scalar float. Higher ELBO = better evidence lower bound.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            loss = model.Loss(X, y)
        return float(-loss)
    finally:
        if was_training:
            model.train()
