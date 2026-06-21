"""Shared fixed model for the HMC agreement study (ADR-0008, GitHub #101).

The study deliberately uses a tiny two-feature Normal distributional model:
each additive feature has one linear coefficient for location and raw scale,
plus a two-parameter intercept.  Keeping this definition here makes NumPyro a
validation-only experiment dependency and leaves ``dune_bayes`` unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch

from dune_bayes.families import NormalFamily
from dune_bayes.utils import EPS

jax.config.update("jax_enable_x64", True)

FEATURE_PRIOR_SCALE = 1.0
INTERCEPT_PRIOR_SCALE = 10.0

Data = Mapping[str, np.ndarray]
Parameters = Mapping[str, np.ndarray]


def numpyro_model(data: Data) -> None:
    """Define the fixed model for NUTS using NumPyro.

    Args:
        data: Arrays ``x1``, ``x2``, and ``y`` with one value per observation.
    """
    x1_weight = numpyro.sample(
        "x1_weight",
        dist.Normal(jnp.zeros(2), FEATURE_PRIOR_SCALE).to_event(1),
    )
    x2_weight = numpyro.sample(
        "x2_weight",
        dist.Normal(jnp.zeros(2), FEATURE_PRIOR_SCALE).to_event(1),
    )
    intercept = numpyro.sample(
        "intercept",
        dist.Normal(jnp.zeros(2), INTERCEPT_PRIOR_SCALE).to_event(1),
    )
    x1 = jnp.asarray(data["x1"], dtype=jnp.float64)
    x2 = jnp.asarray(data["x2"], dtype=jnp.float64)
    predictor = x1[:, None] * x1_weight + x2[:, None] * x2_weight + intercept
    scale = jax.nn.softplus(predictor[:, 1]) + EPS
    numpyro.sample(
        "y",
        dist.Normal(predictor[:, 0], scale),
        obs=jnp.asarray(data["y"], dtype=jnp.float64),
    )


def torch_log_joint(data: Data, parameters: Parameters) -> float:
    """Evaluate the fixed model's Torch log joint in float64.

    Args:
        data: Arrays ``x1``, ``x2``, and ``y`` with one value per observation.
        parameters: Two-vector arrays ``x1_weight``, ``x2_weight``, and
            ``intercept`` ordered as location then raw scale.

    Returns:
        Scalar log likelihood plus fixed Normal weight-prior log density.
    """
    tensors = {
        name: torch.as_tensor(value, dtype=torch.float64)
        for name, value in parameters.items()
    }
    x1 = torch.as_tensor(data["x1"], dtype=torch.float64)
    x2 = torch.as_tensor(data["x2"], dtype=torch.float64)
    y = torch.as_tensor(data["y"], dtype=torch.float64)
    predictor = (
        x1[:, None] * tensors["x1_weight"]
        + x2[:, None] * tensors["x2_weight"]
        + tensors["intercept"]
    )
    log_likelihood = NormalFamily(validate_args=True)(predictor).log_prob(y).sum()
    feature_prior = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(FEATURE_PRIOR_SCALE, dtype=torch.float64),
        validate_args=True,
    )
    intercept_prior = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(INTERCEPT_PRIOR_SCALE, dtype=torch.float64),
        validate_args=True,
    )
    log_prior = feature_prior.log_prob(tensors["x1_weight"]).sum()
    log_prior = log_prior + feature_prior.log_prob(tensors["x2_weight"]).sum()
    log_prior = log_prior + intercept_prior.log_prob(tensors["intercept"]).sum()
    return float(log_likelihood + log_prior)


def jax_log_joint(data: Data, parameters: Parameters) -> float:
    """Independently evaluate the fixed model's JAX log joint in float64.

    Args:
        data: Arrays ``x1``, ``x2``, and ``y`` with one value per observation.
        parameters: Two-vector arrays ``x1_weight``, ``x2_weight``, and
            ``intercept`` ordered as location then raw scale.

    Returns:
        Scalar log likelihood plus fixed Normal weight-prior log density.
    """
    x1 = jnp.asarray(data["x1"], dtype=jnp.float64)
    x2 = jnp.asarray(data["x2"], dtype=jnp.float64)
    y = jnp.asarray(data["y"], dtype=jnp.float64)
    x1_weight = jnp.asarray(parameters["x1_weight"], dtype=jnp.float64)
    x2_weight = jnp.asarray(parameters["x2_weight"], dtype=jnp.float64)
    intercept = jnp.asarray(parameters["intercept"], dtype=jnp.float64)
    predictor = x1[:, None] * x1_weight + x2[:, None] * x2_weight + intercept
    # Match NormalFamily's stable positivity link exactly; the likelihood is
    # otherwise evaluated independently by JAX.
    scale = jax.nn.softplus(predictor[:, 1]) + EPS
    log_likelihood = jsp.stats.norm.logpdf(y, predictor[:, 0], scale).sum()
    log_prior = jsp.stats.norm.logpdf(
        x1_weight, loc=0.0, scale=FEATURE_PRIOR_SCALE
    ).sum()
    log_prior += jsp.stats.norm.logpdf(
        x2_weight, loc=0.0, scale=FEATURE_PRIOR_SCALE
    ).sum()
    log_prior += jsp.stats.norm.logpdf(
        intercept, loc=0.0, scale=INTERCEPT_PRIOR_SCALE
    ).sum()
    return float(log_likelihood + log_prior)
