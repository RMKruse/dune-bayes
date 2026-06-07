"""BaseFamily contract for dune-bayes response families (issue 0018 / GitHub #38).

Every response family must expose:
  - param_count (int): number of raw network outputs consumed.
  - __call__(params) -> Distribution: apply per-parameter links, return distribution.
  - log_prob(params, y) -> Tensor: pointwise log-likelihood (concrete on the
    base — it is always ``self(params).log_prob(y)``; families implement only
    ``__call__``).

Positivity links must be ``softplus(x) + EPS`` (numerical rule 1; GitHub #88):
bare softplus underflows to exactly 0.0 near pre-link −104 in float32, which
poisons ``log_prob``. Consequence: the minimum representable scale (or rate /
concentration) is EPS. ``transform_to(constraints.positive)`` is explicitly
rejected — it is ExpTransform, which overflows.
validate_args follows the test-vs-hot-path convention (numerical rule 6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseFamily(ABC):
    """Abstract base for response-distribution families (ADR-0006, issue 0018).

    Subclasses must set ``param_count: int`` as a class attribute and implement
    ``__call__``; ``log_prob`` is inherited.

    Every positive distribution parameter is linked as ``softplus(x) + EPS``
    (numerical rule 1), so the smallest scale a family can represent is EPS —
    the price of a finite ``log_prob`` at arbitrarily negative pre-link values
    (the ±1e4 gate test in ``tests/families/test_link_floor_gate.py``).
    """

    param_count: int

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Only enforce on concrete (non-abstract) subclasses.
        if not getattr(cls, "__abstractmethods__", None) and not isinstance(
            getattr(cls, "param_count", None), int
        ):
            raise TypeError(
                f"{cls.__name__} must define 'param_count: int' as a class attribute"
            )

    @abstractmethod
    def __call__(self, params: torch.Tensor) -> torch.distributions.Distribution:
        """Apply per-parameter link functions and return a distribution.

        Args:
            params: Raw network output, shape (batch, param_count).

        Returns:
            A torch.distributions.Distribution with batch_shape (batch,).
        """

    def log_prob(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pointwise log-likelihood.

        Stays in log-space via the distribution's ``log_prob`` (numerical
        rule 2). Concrete here because it is identical for every family.

        Args:
            params: Raw network output, shape (batch, param_count).
            y: Observed responses, shape (batch,).

        Returns:
            Tensor of shape (batch,) with pointwise log p(y | params).
        """
        return self(params).log_prob(y)
