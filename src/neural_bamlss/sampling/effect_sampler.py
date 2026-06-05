"""sample_effects — per-feature contribution workhorse (issue 0005 / GitHub #6, #68).

Draws T posterior weight samples and returns per-feature contribution tensors.
Pure function of (model, data, T) — the PRD's ``EffectSampler``, demoted from
a stateless class to a plain function (GitHub #68): no mutation of model
parameters or mode.

Design:
  - Each BayesianMLP forward() draws fresh weights via reparameterization.
    Calling a net T times under torch.no_grad() gives T independent samples.
  - Per-feature contributions (not the summed predictor) are returned for
    epistemic ribbon plots (CONTEXT.md "Effect plot vs response plot").
  - T_predict = 200: default posterior-draw count (CONTEXT.md "MC sample counts").
  - eval() is set for sampling; training mode is restored via finally.
"""

from __future__ import annotations

import torch

from neural_bamlss.model import BayesianNAMLSS

T_PREDICT: int = 200


def sample_effects(
    model: BayesianNAMLSS,
    data: dict[str, torch.Tensor],
    T: int = T_PREDICT,
) -> dict[str, torch.Tensor]:
    """Draw T posterior weight samples; return per-feature contribution tensors.

    Args:
        model: Fitted BayesianNAMLSS instance. Its posterior is defined by
            the variational parameters (loc, rho) of every VariationalDense.
        data: Feature dict {name: Tensor[n, in_features]}, one entry per
            feature in model.feature_names.  Unlike forward()/draw_predictive,
            interaction terms are keyed by the full term name ("x1:x2") and
            their value is the **pre-concatenated** (n, 2) grid tensor —
            callers supply per-term grids, so no concatenation happens here
            (issue 0060).
        T: Number of independent posterior weight draws. Defaults to
            T_PREDICT (200).

    Returns:
        Dict mapping feature name → Tensor[T, n, param_count] of float32.
        Each slice [t, :, :] is one posterior weight sample's contribution.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            samples: dict[str, torch.Tensor] = {}
            for name in model.feature_names:
                net = model.nets[name]
                x = data[name]
                # Each net(x) samples fresh weights via reparameterization
                # (VariationalDense.forward); T calls → T independent draws.
                draws = torch.stack([net(x) for _ in range(T)], dim=0)
                samples[name] = draws
        return samples
    finally:
        if was_training:
            model.train()
