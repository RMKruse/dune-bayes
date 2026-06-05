"""Predictive draw + log-likelihood scoring (issue 0007 / GitHub #8, #68).

Two pure functions split from the former fused ``LogLikSampler`` (the PRD's
goal-3 workhorse name): ``draw_predictive`` runs T stochastic forward passes
and assembles the MixtureSameFamily posterior predictive; ``pointwise_log_lik``
scores an observed response against already-drawn predictor samples.

Design:
  - Splitting from sample_effects keeps the cheap per-feature path independent
    from the expensive log-likelihood path (CONTEXT.md, ADR-0003).
  - Drawing and scoring are separate jobs (GitHub #68): the predictive needs
    no response, so sample_posterior_predictive never touches log_prob — the
    former dummy-y hack violated the Gamma family's support.
  - pointwise_log_lik is float64: logsumexp-over-draws for WAIC/LOO bites in
    float32 (numerical rule, CLAUDE.md dtype section).
  - MixtureSameFamily encodes the epistemic/aleatoric split: variance across
    components = epistemic uncertainty; within each component = family aleatoric.
  - T_eval = 1000 is the default for information-criterion runs; T_predict = 200
    for predictive plots (CONTEXT.md "MC sample counts").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_bamlss.model import BayesianNAMLSS

T_PREDICT: int = 200
T_EVAL: int = 1000


@dataclass
class PredictiveDraws:
    """Outputs from a single draw_predictive call.

    Attributes:
        summed_samples: Stacked summed-predictor draws, shape (T, n, param_count).
        predictive: Uniform MixtureSameFamily over the T weight-sampled family
            distributions.  Spread across components = epistemic; within = aleatoric.
    """

    summed_samples: torch.Tensor
    predictive: torch.distributions.MixtureSameFamily


def draw_predictive(
    model: BayesianNAMLSS,
    X: dict[str, torch.Tensor],
    T: int = T_PREDICT,
) -> PredictiveDraws:
    """Draw T posterior weight samples; return summed predictor and predictive.

    Args:
        model: Fitted BayesianNAMLSS instance.
        X: Feature dict {name: Tensor[n, in_features]} — the same dict
            forward() accepts: interaction terms ("x1:x2") are supplied as
            per-feature entries and concatenated by model.predict_params.
        T: Number of independent posterior weight draws. Defaults to T_PREDICT.

    Returns:
        PredictiveDraws with summed_samples (T, n, param_count) and a
        MixtureSameFamily predictive with batch_shape (n,).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # T independent forward passes → summed predictor for each draw.
            # predict_params is the single owner of predictor assembly
            # (interaction keys, dropout — inert under eval(); issue 0060).
            summed_samples = torch.stack(
                [model.predict_params(X) for _ in range(T)], dim=0
            )  # (T, n, param_count)

            # Build MixtureSameFamily (ADR-0003).
            # Permute (T, n, param_count) → (n, T, param_count) so the family
            # sees T as the last batch dimension — required by MixtureSameFamily
            # which indexes mixture components on the last batch axis.
            params_batched = summed_samples.permute(1, 0, 2)  # (n, T, param_count)
            component_dist = model.family(params_batched)  # batch_shape (n, T)

            n = int(summed_samples.shape[1])
            # Uniform mixture over T components; stays float32 (forward path).
            mix_dist = torch.distributions.Categorical(
                logits=torch.zeros(n, T, dtype=torch.float32)
            )
            predictive = torch.distributions.MixtureSameFamily(mix_dist, component_dist)

            return PredictiveDraws(
                summed_samples=summed_samples,
                predictive=predictive,
            )
    finally:
        if was_training:
            model.train()


def pointwise_log_lik(
    model: BayesianNAMLSS,
    summed_samples: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Score a response against already-drawn predictor samples.

    The scoring half of the split workhorse (GitHub #68): consumes the
    summed_samples produced by draw_predictive, so WAIC/LOO runs draw once
    and score once instead of re-running T forward passes per criterion.

    Args:
        model: BayesianNAMLSS instance — only its family is used here.
        summed_samples: Summed-predictor draws, shape (T, n, param_count),
            as returned by draw_predictive.
        y: Response tensor of shape (n,).

    Returns:
        Per-observation, per-draw log-likelihood, shape (T, n) float64.
        Float64 is required for numerically stable WAIC/LOO logsumexp
        accumulation (CLAUDE.md dtype rule).
    """
    with torch.no_grad():
        # Pointwise log-likelihood in float64 (numerical rule: logsumexp).
        T = int(summed_samples.shape[0])
        loglik_list: list[torch.Tensor] = []
        for t in range(T):
            dist_t = model.family(summed_samples[t])
            ll = dist_t.log_prob(y).to(torch.float64)  # (n,) float64
            loglik_list.append(ll)
        return torch.stack(loglik_list, dim=0)  # (T, n) float64
