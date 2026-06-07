"""Cross-cutting helpers: EPS constant, activation registry, eval_mode,
tensor coercion, and seed_everything (CLAUDE.md, GitHub #62, #90).

The re-seed protocol (seed_everything → build → fit) is exactly reproducible
on CPU (verified in tests/model/test_reseed_determinism.py, GitHub #90). The
remaining caveat: two models built back-to-back within one RNG stream — no
re-seed between them — do not draw identical noise, because each build
advances the global stream.
"""

import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The single named epsilon floor for float32 (numerical rule 3): used only
# where a floor is genuinely unavoidable (softplus outputs that must stay
# strictly positive, constant-feature scale denominators). float64
# accumulations would use 1e-12, but the only float64 path (log-lik / WAIC)
# never needs a floor — it stays in log-space.
EPS: float = 1e-6

# The single activation registry shared by every layer / shape function that
# takes an activation name (VariationalDense, NeuralLinearMLP, DeterministicMLP).
_ACTIVATIONS: dict[str | None, Callable[[torch.Tensor], torch.Tensor] | None] = {
    None: None,
    "linear": None,
    "relu": F.relu,
    "tanh": torch.tanh,
}


def resolve_activation(
    name: str | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Map an activation name to its callable; None means identity.

    Args:
        name: One of {None, "linear", "relu", "tanh"}.

    Returns:
        The activation callable, or None for the identity (None / "linear").

    Raises:
        ValueError: If ``name`` is not a known activation.
    """
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; choose from {set(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]


@contextmanager
def eval_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Put ``model`` in eval() for the block; restore the prior mode on exit.

    Sampling and scoring paths must run in eval() (dropout inert, no KL-side
    effects intended for training) without clobbering the caller's mode.

    Args:
        model: Module whose training flag is toggled.

    Yields:
        The same model, now in eval mode.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Coerce a tensor or array-like to a numpy array (plot-input shim)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed the global torch, numpy, and Python ``random`` generators.

    The flag always sets torch's deterministic-algorithms mode (on or off) so
    the global state after a call is fully explicit, never inherited.

    Args:
        seed: Integer seed applied to every global generator.
        deterministic: When True, additionally enable
            ``torch.use_deterministic_algorithms`` — bit-exact run-to-run
            reproducibility at the cost of speed (and errors on ops with no
            deterministic kernel). Default False keeps the fast kernels;
            seeding alone already suffices on CPU (GitHub #90).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)
