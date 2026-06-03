"""Posterior sampling workhorses (issue 0005, issue 0007 / GitHub #6, #8).

EffectSampler:  per-feature contribution samples → effect ribbons (issue 0005).
LogLikSampler:  pointwise log-likelihood samples → WAIC/LOO (issue 0007, future).
"""

from neural_bamlss.sampling.effect_sampler import T_PREDICT, EffectSampler

__all__ = ["EffectSampler", "T_PREDICT"]
