"""Predictive draw + log-likelihood scoring (issue 0007 / GitHub #8, #68).

Two pure functions split from the former fused ``LogLikSampler`` (the PRD's
goal-3 workhorse name): ``draw_predictive`` draws T posterior weight samples
and assembles the MixtureSameFamily posterior predictive; ``pointwise_log_lik``
scores an observed response against already-drawn predictor samples.

Design:
  - Splitting from sample_effects keeps the cheap per-feature path independent
    from the expensive log-likelihood path (CONTEXT.md, ADR-0003).
  - The T draws are batched, not looped (issue 0027 / GitHub #80): inputs
    expand along a leading sample dimension (fresh per-slice noise in every
    variational layer) and draw_predictive chunks the dispatches over T
    (sampling/chunking.py); pointwise_log_lik scores all draws in one
    broadcast family dispatch.
  - Drawing and scoring are separate jobs (GitHub #68): the predictive needs
    no response, so sample_posterior_predictive never touches log_prob — the
    former dummy-y hack violated the Gamma family's support.
  - pointwise_log_lik is float64: logsumexp-over-draws for WAIC/LOO bites in
    float32 (numerical rule, CLAUDE.md dtype section).
  - MixtureSameFamily encodes the epistemic/aleatoric split: variance across
    components = epistemic uncertainty; within each component = family aleatoric.
    The split is valid only for coherent draws (ADR-0007, issue #85): each of
    the T components must be ONE global weight realization — eval_mode forces
    every variational layer onto the vanilla path, so the training-only
    local-reparameterization estimator can never corrupt the decomposition.
  - T_eval = 1000 is the default for information-criterion runs; T_predict = 200
    for predictive plots (CONTEXT.md "MC sample counts").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dune_bayes.model import BayesianNAMLSS
from dune_bayes.sampling.chunking import _chunk_sizes
from dune_bayes.utils import eval_mode

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
    chunk_size: int | None = None,
) -> PredictiveDraws:
    """Draw T posterior weight samples; return summed predictor and predictive.

    Args:
        model: Fitted BayesianNAMLSS instance.
        X: Feature dict {name: Tensor[n, in_features]} — the same dict
            forward() accepts: interaction terms ("x1:x2") are supplied as
            per-feature entries and concatenated by model.predict_params.
        T: Number of independent posterior weight draws. Defaults to T_PREDICT.
        chunk_size: Max draws batched per dispatch — the internal memory knob
            (issue 0027). None (default) derives it from a fixed row budget so
            small batches sweep all T draws in one dispatch while large n
            stays near the per-draw loop's memory profile.

    Returns:
        PredictiveDraws with summed_samples (T, n, param_count) and a
        MixtureSameFamily predictive with batch_shape (n,).
    """
    with eval_mode(model), torch.no_grad():
        # Batched draws (issue 0027): inputs expand (zero-copy views) along a
        # leading sample dimension; every variational layer draws fresh noise
        # per slice, so one predict_params dispatch yields `size` independent
        # draws. predict_params stays the single owner of predictor assembly
        # (interaction keys, dropout — inert under eval(); issue 0060).
        n = int(next(iter(X.values())).shape[0])
        chunks = []
        for size in _chunk_sizes(T, n, chunk_size):
            X_s = {k: v.expand(size, *v.shape) for k, v in X.items()}
            chunks.append(model.predict_params(X_s))
        summed_samples = torch.cat(chunks, dim=0)  # (T, n, param_count)

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
        # One broadcast dispatch over all T draws (issue 0027 / GitHub #80):
        # families build on torch.distributions, whose links and log_prob are
        # elementwise, so a (T, n, param_count) input yields batch_shape (T, n)
        # and y (n,) broadcasts across the draw axis — identical arithmetic to
        # the former per-draw loop. Scoring is elementwise (FLOP-bound), so no
        # T-chunking is needed: peak memory is a few (T, n) temporaries, the
        # same order as the (T, n) float64 result itself.
        dist = model.family(summed_samples)  # batch_shape (T, n)
        # Float64 for WAIC/LOO logsumexp accumulation (numerical rule).
        ll: torch.Tensor = dist.log_prob(y)  # (T, n)
        return ll.to(torch.float64)
