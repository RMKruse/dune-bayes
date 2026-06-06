"""Cross-cutting helpers: EPS constant and seed_everything (CLAUDE.md, GitHub #62).

Reproducibility holds only within one model object, never across two
freshly-built ones (CLAUDE.md seeding rule) — seed_everything makes a single
seed → build → sample sequence deterministic, nothing more.
"""

import random

import numpy as np
import torch

# The single named epsilon floor for float32 (numerical rule 3): used only
# where a floor is genuinely unavoidable (softplus outputs that must stay
# strictly positive, constant-feature scale denominators). float64
# accumulations would use 1e-12, but the only float64 path (log-lik / WAIC)
# never needs a floor — it stays in log-space.
EPS: float = 1e-6


def seed_everything(seed: int) -> None:
    """Seed the global torch, numpy, and Python ``random`` generators.

    Args:
        seed: Integer seed applied to every global generator.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
