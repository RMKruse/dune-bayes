"""DataModule — tabular data → model-ready tensors + N (issue 0022 / GitHub #49).

Walking skeleton of the data component: a DataFrame plus the response-column
name becomes the per-feature tensor dict and target tensor the model already
consumes, with n_obs exposed so KL/N is wired from the data rather than
hand-passed. Numeric preprocessing (issue 0023 / GitHub #50) is opt-in via
``numeric_scaling``: standardize by default, min-max per feature on request.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch

# Single named epsilon constant (CLAUDE.md numerical rules — never bare divisions).
_EPS_F32: float = 1e-6


class _FeatureScaler:
    """Fit-once, apply-many scaler for a single numeric feature.

    Two methods: "standard" (mean/std) and "minmax" (min/max).
    Constant features are handled with an EPS floor on the denominator
    rather than a bare division (CLAUDE.md numerical rule 3).
    """

    def __init__(self, method: Literal["standard", "minmax"]) -> None:
        self.method = method
        self._loc: float = 0.0  # mean for standard; min for minmax
        self._scale: float = 1.0  # std for standard; range for minmax

    def fit(self, arr: np.ndarray) -> None:
        if self.method == "standard":
            self._loc = float(arr.mean())
            # EPS floor: constant features get scale=1 (pass-through after centering)
            self._scale = float(max(arr.std(), _EPS_F32))
        else:
            self._loc = float(arr.min())
            self._scale = float(max(arr.max() - arr.min(), _EPS_F32))

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self._loc) / self._scale

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self._scale + self._loc


class DataModule:
    """Model-ready view of a tabular dataset.

    Args:
        df: Source DataFrame; every column except ``response`` becomes a
            feature.
        response: Name of the response column in ``df``.
        numeric_scaling: Per-feature scaling method.  ``None`` (default) means
            no preprocessing — raw float32 pass-through.  Pass a dict to
            enable scaling: features listed in the dict get the specified
            method (``"standard"`` or ``"minmax"``); features *not* listed
            get ``"standard"`` (the default method).  An empty dict ``{}``
            therefore standardizes all features.

    Raises:
        ValueError: If ``response`` is not a column of ``df``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        response: str,
        numeric_scaling: dict[str, Literal["standard", "minmax"]] | None = None,
    ) -> None:
        if response not in df.columns:
            raise ValueError(
                f"Response column {response!r} not found in DataFrame "
                f"(columns: {list(df.columns)})."
            )
        self.response = response
        feature_names = [name for name in df.columns if name != response]

        if numeric_scaling is not None:
            # Fit one scaler per feature on this (training) data.
            self._scalers: dict[str, _FeatureScaler] = {}
            for name in feature_names:
                method = numeric_scaling.get(name, "standard")
                scaler = _FeatureScaler(method)
                scaler.fit(df[name].to_numpy(dtype=np.float32))
                self._scalers[name] = scaler

            self.features: dict[str, torch.Tensor] = {
                name: torch.tensor(
                    self._scalers[name].transform(df[name].to_numpy(dtype=np.float32)),
                    dtype=torch.float32,
                ).unsqueeze(-1)
                for name in feature_names
            }
        else:
            self._scalers = {}
            # torch.tensor (not as_tensor): pandas ≥3 hands out read-only views,
            # and a tensor must own writable memory, not alias the DataFrame.
            self.features = {
                name: torch.tensor(df[name].to_numpy(), dtype=torch.float32).unsqueeze(
                    -1
                )
                for name in feature_names
            }

        # (n,), not (n, 1) — dist.log_prob(y) requires the target's shape to
        # match the family batch_shape.
        self.target: torch.Tensor = torch.tensor(
            df[response].to_numpy(), dtype=torch.float32
        )
        # Training-set size N — the KL/N divisor source (ADR-0001).
        self.n_obs: int = len(df)

    def transform(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Apply fitted scaling statistics to new data without refitting.

        Args:
            df: New DataFrame containing the same feature columns used at fit
                time.

        Returns:
            Dict mapping feature name → scaled float32 tensor of shape ``(n, 1)``.

        Raises:
            RuntimeError: If no scaling statistics have been fitted (i.e.
                ``numeric_scaling`` was ``None`` at construction time).
            KeyError: If a fitted feature column is missing from ``df``.
        """
        if not self._scalers:
            raise RuntimeError(
                "No scaling statistics fitted. Pass numeric_scaling= to __init__."
            )
        return {
            name: torch.tensor(
                scaler.transform(df[name].to_numpy(dtype=np.float32)),
                dtype=torch.float32,
            ).unsqueeze(-1)
            for name, scaler in self._scalers.items()
        }

    def inverse_transform(
        self, feature_name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Recover original feature scale from a scaled tensor.

        Intended for plot axes: a regular grid scaled for the model is
        inverted back to the units the modeller actually thinks in
        (issue 0023, user story 1).

        Args:
            feature_name: Name of the feature to invert.
            tensor: Scaled values, shape ``(n,)`` or ``(n, 1)``.

        Returns:
            Original-scale float32 tensor with the same shape as ``tensor``.

        Raises:
            KeyError: If ``feature_name`` was not fitted.
            RuntimeError: If no scaling statistics have been fitted.
        """
        if not self._scalers:
            raise RuntimeError(
                "No scaling statistics fitted. Pass numeric_scaling= to __init__."
            )
        if feature_name not in self._scalers:
            raise KeyError(
                f"No scaler fitted for feature {feature_name!r}. "
                f"Fitted features: {list(self._scalers)}."
            )
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        return torch.tensor(
            self._scalers[feature_name].inverse_transform(arr),
            dtype=torch.float32,
        )
