"""LogLikSampler — log-likelihood + predictive workhorse (issue 0007 / GitHub #8).

Runs T stochastic forward passes and returns summed-predictor samples,
pointwise log-likelihood, and a MixtureSameFamily posterior predictive.

Design:
  - Splitting from EffectSampler keeps the cheap per-feature path independent
    from the expensive log-likelihood path (CONTEXT.md, ADR-0003).
  - pointwise_loglik is float64: logsumexp-over-draws for WAIC/LOO bites in
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
class LogLikResult:
    """Outputs from a single LogLikSampler call.

    Attributes:
        summed_samples: Stacked summed-predictor draws, shape (T, n, param_count).
        pointwise_loglik: Per-observation, per-draw log-likelihood in float64,
            shape (T, n).  Float64 is required for numerically stable WAIC/LOO
            logsumexp accumulation.
        predictive: Uniform MixtureSameFamily over the T weight-sampled family
            distributions.  Spread across components = epistemic; within = aleatoric.
    """

    summed_samples: torch.Tensor
    pointwise_loglik: torch.Tensor
    predictive: torch.distributions.MixtureSameFamily


class LogLikSampler:
    """Callable that runs T posterior weight draws and returns the full predictive.

    Args:
        None — instantiate once, call with different (model, X, y, T) tuples.

    Example::

        sampler = LogLikSampler()
        result = sampler(model, X, y)         # T defaults to T_predict = 200
        result = sampler(model, X, y, T=1000) # IC run
        result.summed_samples   # (T, n, param_count)
        result.pointwise_loglik # (T, n) float64
        result.predictive       # MixtureSameFamily, batch_shape (n,)
    """

    T_predict: int = T_PREDICT
    T_eval: int = T_EVAL

    def __call__(
        self,
        model: BayesianNAMLSS,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        T: int = T_PREDICT,
    ) -> LogLikResult:
        """Draw T posterior weight samples; return summed predictor, loglik, predictive.

        Args:
            model: Fitted BayesianNAMLSS instance.
            X: Feature dict {name: Tensor[n, in_features]}.
            y: Response tensor of shape (n,).
            T: Number of independent posterior weight draws. Defaults to T_predict.

        Returns:
            LogLikResult with summed_samples (T, n, param_count), pointwise_loglik
            (T, n) float64, and a MixtureSameFamily predictive with batch_shape (n,).
        """
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                # T independent forward passes → summed predictor for each draw.
                summed_list: list[torch.Tensor] = []
                for _ in range(T):
                    contribs = [
                        model.nets[name](X[name]) for name in model.feature_names
                    ]
                    # Sum contributions; stack→sum over the feature dim.
                    summed = torch.stack(contribs, dim=0).sum(dim=0)  # (n, param_count)
                    summed_list.append(summed)

                summed_samples = torch.stack(summed_list, dim=0)  # (T, n, param_count)

                # Pointwise log-likelihood in float64 (numerical rule: logsumexp).
                loglik_list: list[torch.Tensor] = []
                for t in range(T):
                    dist_t = model.family(summed_samples[t])
                    ll = dist_t.log_prob(y).to(torch.float64)  # (n,) float64
                    loglik_list.append(ll)

                pointwise_loglik = torch.stack(loglik_list, dim=0)  # (T, n) float64

                # Build MixtureSameFamily (ADR-0003).
                # Permute (T, n, param_count) → (n, T, param_count) so the family
                # sees T as the last batch dimension — required by MixtureSameFamily
                # which indexes mixture components on the last batch axis.
                params_batched = summed_samples.permute(1, 0, 2)  # (n, T, param_count)
                component_dist = model.family(params_batched)  # batch_shape (n, T)

                n = int(y.shape[0])
                # Uniform mixture over T components; stays float32 (forward path).
                mix_dist = torch.distributions.Categorical(
                    logits=torch.zeros(n, T, dtype=torch.float32)
                )
                predictive = torch.distributions.MixtureSameFamily(
                    mix_dist, component_dist
                )

                return LogLikResult(
                    summed_samples=summed_samples,
                    pointwise_loglik=pointwise_loglik,
                    predictive=predictive,
                )
        finally:
            if was_training:
                model.train()
