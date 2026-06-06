"""BaseFamily contract for neural-BAMLSS response families (issue 0018 / GitHub #38).

Every response family must expose:
  - param_count (int): number of raw network outputs consumed.
  - __call__(params) -> Distribution: apply per-parameter links, return distribution.
  - log_prob(params, y) -> Tensor: pointwise log-likelihood (concrete on the
    base — it is always ``self(params).log_prob(y)``; families implement only
    ``__call__``).

Positivity transforms must route through softplus (numerical rule 1).
validate_args follows the test-vs-hot-path convention (numerical rule 6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseFamily(ABC):
    """Abstract base for response-distribution families (ADR-0006, issue 0018).

    Subclasses must set ``param_count: int`` as a class attribute and implement
    ``__call__``; ``log_prob`` is inherited.
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
