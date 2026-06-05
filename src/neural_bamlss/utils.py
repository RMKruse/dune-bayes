"""seed_everything — one helper for global RNG seeding (CLAUDE.md, GitHub #62).

Reproducibility holds only within one model object, never across two
freshly-built ones (CLAUDE.md seeding rule) — this helper makes a single
seed → build → sample sequence deterministic, nothing more.
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the global torch, numpy, and Python ``random`` generators.

    Args:
        seed: Integer seed applied to every global generator.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
