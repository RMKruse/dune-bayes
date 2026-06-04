"""DataModule — tabular data → model-ready tensors + N (issue 0022 / GitHub #49).

Walking skeleton of the data component: a DataFrame plus the response-column
name becomes the per-feature tensor dict and target tensor the model already
consumes, with n_obs exposed so KL/N is wired from the data rather than
hand-passed. No preprocessing yet — raw numeric columns pass through as
float32 tensors of shape (n, 1); the preprocessing slices (0023, 0024) build
on this.
"""

from __future__ import annotations

import pandas as pd
import torch


class DataModule:
    """Model-ready view of a tabular dataset.

    Args:
        df: Source DataFrame; every column except ``response`` becomes a
            feature.
        response: Name of the response column in ``df``.

    Raises:
        ValueError: If ``response`` is not a column of ``df``.
    """

    def __init__(self, df: pd.DataFrame, response: str) -> None:
        if response not in df.columns:
            raise ValueError(
                f"Response column {response!r} not found in DataFrame "
                f"(columns: {list(df.columns)})."
            )
        self.response = response
        # torch.tensor (not as_tensor): pandas ≥3 hands out read-only views,
        # and a tensor must own writable memory, not alias the DataFrame.
        # float32 throughout — the model's working dtype (CLAUDE.md dtype rule).
        self.features: dict[str, torch.Tensor] = {
            name: torch.tensor(df[name].to_numpy(), dtype=torch.float32).unsqueeze(-1)
            for name in df.columns
            if name != response
        }
        # (n,), not (n, 1) — dist.log_prob(y) requires the target's shape to
        # match the family batch_shape.
        self.target: torch.Tensor = torch.tensor(
            df[response].to_numpy(), dtype=torch.float32
        )
        # Training-set size N — the KL/N divisor source (ADR-0001).
        self.n_obs: int = len(df)
