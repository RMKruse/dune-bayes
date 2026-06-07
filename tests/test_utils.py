"""Tests for seed_everything (GitHub #62, #90).

Reference-test archetype (CLAUDE.md): MC-convergence/reproducibility at the
RNG-stream boundary — re-seeding reproduces the identical draw sequence from
each global generator. The full re-seed protocol (seed → build → fit) lives in
``tests/model/test_reseed_determinism.py`` (GitHub #90).
"""

import random

import numpy as np
import torch

from dune_bayes.utils import seed_everything


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


def test_deterministic_flag_opts_into_deterministic_algorithms():
    """deterministic=True turns torch's deterministic-algorithms mode on;
    the default leaves it off (speed — the documented trade-off, GitHub #90)."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(42)
        assert not torch.are_deterministic_algorithms_enabled()
        seed_everything(42, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        # Global toggle — restore so other tests see the default mode.
        torch.use_deterministic_algorithms(was_enabled)
