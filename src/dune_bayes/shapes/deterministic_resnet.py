"""Fully-deterministic ResNet shape function (issue 0020 / GitHub #39).

DeterministicResNet: stacked residual blocks of plain nn.Linear layers — no
VariationalDense. Contributes zero KL inside BayesianNAMLSS. Useful as a
Bayes-vs-deterministic baseline in WAIC/LOO comparisons (user stories 21, 22).

Architecture:
  in_proj  : Linear(in_features, hidden_dim)
  blocks   : num_blocks × ResBlock(hidden_dim)
             each block: relu(W2(relu(W1(x))) + x)  — skip adds the input
  out_proj : Linear(hidden_dim, param_count, bias=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Single residual block: two linear layers with an identity skip connection.

    Args:
        hidden_dim: Width of both linear layers (input and output share the same
            dimension so the skip addition is dimension-free, no projection needed).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear2(F.relu(self.linear1(x))) + x)


class DeterministicResNet(nn.Module):
    """Fully-deterministic ResNet shape function for dune-bayes.

    Maps a feature tensor (batch, in_features) to (batch, param_count) via an
    input projection, stacked residual blocks, and a linear output head. All
    layers are plain nn.Linear — collect_kl() returns 0.0 for this module.

    Args:
        in_features: Input dimension.
        param_count: Output dimension — equals the family's param_count.
        hidden_dim: Width of the residual blocks. Defaults to 64.
        num_blocks: Number of ResBlock layers stacked. Defaults to 2.
    """

    def __init__(
        self,
        in_features: int,
        param_count: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.param_count = int(param_count)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        self.in_proj = nn.Linear(self.in_features, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(self.hidden_dim) for _ in range(self.num_blocks)]
        )
        # Output layer: no bias (intercept handled by BayesianIntercept).
        self.out_proj = nn.Linear(self.hidden_dim, self.param_count, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, in_features) → (batch, param_count).

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, param_count).
        """
        x = F.relu(self.in_proj(x))
        for block in self.blocks:
            x = block(x)
        # nn.Module.__call__ is untyped in torch's stubs; anchor the return type.
        out: torch.Tensor = self.out_proj(x)
        return out

    # ── serialization ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free config (ints only)."""
        return {
            "in_features": self.in_features,
            "param_count": self.param_count,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
        }

    @classmethod
    def from_config(cls, config: dict) -> "DeterministicResNet":
        """Reconstruct a DeterministicResNet from a get_config() dict.

        Args:
            config: Dict as returned by get_config().

        Returns:
            New DeterministicResNet with the same architecture.
        """
        return cls(**config)
