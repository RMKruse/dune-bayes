"""Tests for seed_everything (GitHub #62).

Reference-test archetype (CLAUDE.md): MC-convergence/reproducibility at the
RNG-stream boundary — re-seeding reproduces the identical draw sequence from
each global generator. Deliberately NOT tested: draw equality across two
freshly-built model objects (CLAUDE.md seeding rule forbids relying on it).
"""

import random

import numpy as np
import torch

from neural_bamlss.utils import seed_everything


def test_torch_stream_reproducible_after_reseed():
    """Re-seeding with the same seed replays the identical torch sequence."""
    seed_everything(42)
    first = torch.randn(8)
    seed_everything(42)
    second = torch.randn(8)
    # Same generator state => bit-identical draws; exact equality, no tolerance.
    assert torch.equal(first, second)


def test_numpy_stream_reproducible_after_reseed():
    """Re-seeding with the same seed replays the identical numpy sequence."""
    seed_everything(42)
    first = np.random.randn(8)
    seed_everything(42)
    second = np.random.randn(8)
    # Same generator state => bit-identical draws; exact equality, no tolerance.
    assert np.array_equal(first, second)


def test_python_random_stream_reproducible_after_reseed():
    """Re-seeding with the same seed replays the identical random sequence."""
    seed_everything(42)
    first = [random.random() for _ in range(8)]
    seed_everything(42)
    second = [random.random() for _ in range(8)]
    # Same generator state => bit-identical draws; exact equality, no tolerance.
    assert first == second
