"""sample_effects — per-feature contribution workhorse (issue 0005 / GitHub #6, #68).

Draws T posterior weight samples and returns per-feature contribution tensors.
Pure function of (model, data, T) — the PRD's ``EffectSampler``, demoted from
a stateless class to a plain function (GitHub #68): no mutation of model
parameters or mode.

Design:
  - The T draws are batched, not looped (issue 0027 / GitHub #80): inputs are
    expanded along a leading sample dimension and each variational layer draws
    fresh per-slice noise (sample-dimension reparameterization), so per-draw
    cost approaches the batched lower bound in the dispatch-bound small-batch
    regime (~6x on plotting grids).
  - Chunked over T: a full (T, n, hidden) activation would dwarf the loop's
    peak memory at large n. _chunk_sizes caps each dispatch at a row budget,
    keeping peak memory within a small factor of the per-draw loop's.
  - Per-feature contributions (not the summed predictor) are returned for
    epistemic ribbon plots (CONTEXT.md "Effect plot vs response plot").
  - T_predict = 200: default posterior-draw count (CONTEXT.md "MC sample counts").
  - eval() is set for sampling; the caller's mode is restored via eval_mode.
"""

from __future__ import annotations

import torch

from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.sampling.chunking import _chunk_sizes
from neural_bamlss.utils import eval_mode

T_PREDICT: int = 200


def sample_effects(
    model: BayesianNAMLSS,
    data: dict[str, torch.Tensor],
    T: int = T_PREDICT,
    chunk_size: int | None = None,
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
        chunk_size: Max draws batched per dispatch — the internal memory knob
            (issue 0027). None (default) derives it from a fixed row budget so
            small grids batch all T draws while large n stays near the
            per-draw loop's memory profile.

    Returns:
        Dict mapping feature name → Tensor[T, n, param_count] of float32.
        Each slice [t, :, :] is one posterior weight sample's contribution.
    """
    with eval_mode(model), torch.no_grad():
        samples: dict[str, torch.Tensor] = {}
        for name in model.feature_names:
            net = model.nets[name]
            x = data[name]
            # Batched draws (issue 0027): expand is a zero-copy view; each
            # variational layer draws fresh noise per sample slice, so one
            # dispatch yields `size` independent posterior draws.
            chunks = [
                net(x.expand(size, *x.shape))
                for size in _chunk_sizes(T, int(x.shape[0]), chunk_size)
            ]
            samples[name] = torch.cat(chunks, dim=0)
    return samples
