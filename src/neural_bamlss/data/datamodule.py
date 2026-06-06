"""DataModule — tabular data → model-ready tensors + N.

Issue 0022–0026 / GitHub #49–#53.

Walking skeleton of the data component: a DataFrame plus the response-column
name becomes the per-feature tensor dict and target tensor the model already
consumes, with n_obs exposed so KL/N is wired from the data rather than
hand-passed. Numeric preprocessing (issue 0023 / GitHub #50) is opt-in via
``numeric_scaling``: standardize by default, min-max per feature on request.
Categorical encoding (issue 0024 / GitHub #51) is automatic: object/category
dtype columns are integer-coded 0..K-1 as torch.long tensors for
BayesianEmbedding.  Minibatching (issue 0026 / GitHub #53) is provided via
``dataloader()``: a seedable DataLoader over (feature-dict, target) batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from neural_bamlss.utils import EPS


class _FeatureDictDataset(torch.utils.data.Dataset):
    """torch.utils.data.Dataset over a feature dict + target tensor.

    Each item is a (feature_dict_row, target_scalar) pair. The feature dict
    preserves the same keys as DataModule.features; each value is a 1-D slice
    along the first (observation) dimension.
    """

    def __init__(
        self,
        features: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> None:
        self._features = features
        self._target = target

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {k: v[idx] for k, v in self._features.items()}, self._target[idx]


def _is_categorical(series: pd.Series) -> bool:
    """Return True for string or pandas Categorical dtype columns.

    Uses dtype.kind == 'O' (works for both numpy object dtype and pandas
    StringDtype in pandas ≥3, where dtype == object is False for strings).
    """
    return series.dtype.kind == "O" or hasattr(series.dtype, "categories")


class _CategoricalEncoder:
    """Fit-once, apply-many encoder for a single categorical feature.

    Levels are sorted for a deterministic, stable code assignment.
    Unseen levels at transform time raise ValueError by default — a silent
    new index would give BayesianEmbedding a code the posterior never saw.
    """

    def __init__(self) -> None:
        self._categories: list[str] = []
        self._mapping: dict[str, int] = {}

    def fit(self, arr: np.ndarray) -> None:
        # Sorted so the code assignment is deterministic regardless of row order.
        self._categories = sorted(set(arr.tolist()))
        self._mapping = {cat: i for i, cat in enumerate(self._categories)}

    def transform(self, arr: np.ndarray) -> np.ndarray:
        # Categories are sorted at fit time, so searchsorted recovers each
        # value's code vectorized; a mismatch after clipping flags an unseen
        # level (values sorting past the last category clip back onto it).
        cats = np.asarray(self._categories)
        vals = np.asarray(arr)
        idx = np.searchsorted(cats, vals)
        clipped = np.minimum(idx, len(cats) - 1)
        unseen = cats[clipped] != vals
        if unseen.any():
            # .tolist() yields a plain Python str so the !r repr stays clean.
            val = vals[unseen].tolist()[0]
            raise ValueError(
                f"Unseen category {val!r} in transform; "
                f"known levels: {self._categories}."
            )
        return idx.astype(np.int64)

    @property
    def num_levels(self) -> int:
        return len(self._categories)


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
            self._scale = float(max(arr.std(), EPS))
        else:
            self._loc = float(arr.min())
            self._scale = float(max(arr.max() - arr.min(), EPS))

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

    Notes:
        Categorical features (``object`` or ``category`` dtype) are
        automatically encoded as ``torch.long`` codes ``0..K-1`` and stored
        alongside numeric features in ``self.features``.  The level count
        per categorical feature is available via ``num_levels``.
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

        cat_names = [n for n in feature_names if _is_categorical(df[n])]
        num_names = [n for n in feature_names if not _is_categorical(df[n])]

        # Fit categorical encoders (issue 0024 / GitHub #51).
        self._encoders: dict[str, _CategoricalEncoder] = {}
        for name in cat_names:
            enc = _CategoricalEncoder()
            enc.fit(df[name].to_numpy())
            self._encoders[name] = enc

        if numeric_scaling is not None:
            # Fit one scaler per numeric feature on this (training) data.
            self._scalers: dict[str, _FeatureScaler] = {}
            for name in num_names:
                method = numeric_scaling.get(name, "standard")
                scaler = _FeatureScaler(method)
                scaler.fit(df[name].to_numpy(dtype=np.float32))
                self._scalers[name] = scaler

            numeric_tensors: dict[str, torch.Tensor] = {
                name: torch.tensor(
                    self._scalers[name].transform(df[name].to_numpy(dtype=np.float32)),
                    dtype=torch.float32,
                ).unsqueeze(-1)
                for name in num_names
            }
        else:
            self._scalers = {}
            # torch.tensor (not as_tensor): pandas ≥3 hands out read-only views,
            # and a tensor must own writable memory, not alias the DataFrame.
            numeric_tensors = {
                name: torch.tensor(df[name].to_numpy(), dtype=torch.float32).unsqueeze(
                    -1
                )
                for name in num_names
            }

        categorical_tensors: dict[str, torch.Tensor] = {
            name: torch.tensor(
                self._encoders[name].transform(df[name].to_numpy()), dtype=torch.long
            ).unsqueeze(-1)
            for name in cat_names
        }

        self.features: dict[str, torch.Tensor] = {
            **numeric_tensors,
            **categorical_tensors,
        }

        # (n,), not (n, 1) — dist.log_prob(y) requires the target's shape to
        # match the family batch_shape.
        self.target: torch.Tensor = torch.tensor(
            df[response].to_numpy(), dtype=torch.float32
        )
        # Training-set size N — the KL/N divisor source (ADR-0001).
        self.n_obs: int = len(df)

    @property
    def num_levels(self) -> dict[str, int]:
        """Level count per categorical feature, for sizing BayesianEmbedding."""
        return {name: enc.num_levels for name, enc in self._encoders.items()}

    def dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.DataLoader:
        """Return a DataLoader that yields (feature-dict, target) minibatches.

        The KL divisor must remain ``n_obs`` (full training-set size), not the
        batch size. This method only produces the batches; the model's ``fit()``
        is responsible for keeping ``kl_divisor`` fixed at N (issue 0026).

        Args:
            batch_size: Number of observations per batch.  The final batch may
                be smaller than ``batch_size`` (``drop_last=False``).
            shuffle: Whether to shuffle observations each epoch (default True).
            generator: Optional seeded ``torch.Generator`` for reproducible
                shuffling within one model object (consistent with the
                one-model-object reproducibility rule in CLAUDE.md).

        Returns:
            A ``torch.utils.data.DataLoader`` whose items are
            ``(feature_dict, target)`` tuples — the same structure as
            ``self.features`` and ``self.target``, sliced along dim 0.
        """
        dataset = _FeatureDictDataset(self.features, self.target)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            drop_last=False,
        )

    def transform(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Apply fitted preprocessing to new data without refitting.

        Applies numeric scalers and categorical encoders fitted at construction
        time.  Unseen category levels raise ``ValueError`` immediately.

        Args:
            df: New DataFrame containing the feature columns used at fit time.

        Returns:
            Dict mapping feature name → tensor of shape ``(n, 1)``.
            Numeric features are float32; categorical features are torch.long.

        Raises:
            RuntimeError: If no preprocessing was fitted at construction time.
            KeyError: If a fitted feature column is missing from ``df``.
            ValueError: If a categorical column contains an unseen level.
        """
        if not self._scalers and not self._encoders:
            raise RuntimeError(
                "No preprocessing fitted. Pass numeric_scaling= or include "
                "categorical columns to __init__."
            )
        result: dict[str, torch.Tensor] = {}
        for name, scaler in self._scalers.items():
            result[name] = torch.tensor(
                scaler.transform(df[name].to_numpy(dtype=np.float32)),
                dtype=torch.float32,
            ).unsqueeze(-1)
        for name, enc in self._encoders.items():
            result[name] = torch.tensor(
                enc.transform(df[name].to_numpy()), dtype=torch.long
            ).unsqueeze(-1)
        return result

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

    # ── serialization (issue 0025 / GitHub #52) ───────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Return a closure-free dict of all fitted preprocessing state.

        The dict contains only plain Python scalars, strings, and lists —
        no class instances — so it survives ``torch.save`` / ``torch.load``
        without being coupled to the current class definitions.

        Returns:
            Dict with keys: ``response``, ``n_obs``, ``scalers``
            (per-feature ``{method, loc, scale}``), ``encoders``
            (per-feature ``{categories: list[str]}``).
        """
        scalers_state = {
            name: {"method": s.method, "loc": s._loc, "scale": s._scale}
            for name, s in self._scalers.items()
        }
        encoders_state = {
            name: {"categories": list(e._categories)}
            for name, e in self._encoders.items()
        }
        return {
            "response": self.response,
            "n_obs": self.n_obs,
            "scalers": scalers_state,
            "encoders": encoders_state,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> DataModule:
        """Reconstruct a DataModule from a saved state dict without refitting.

        Only the preprocessing state is restored.  ``features`` and
        ``target`` tensors are **not** restored — use ``transform(df)`` on
        fresh data instead of accessing ``dm.features`` directly.

        Args:
            state: Dict produced by ``get_state()``.

        Returns:
            A DataModule whose ``transform()`` is identical to the original.
        """
        # Bypass __init__ so we never touch a DataFrame.
        obj = object.__new__(cls)
        obj.response = state["response"]
        obj.n_obs = state["n_obs"]
        obj.features = {}
        obj.target = torch.empty(0)

        obj._scalers = {}
        for name, s in state["scalers"].items():
            scaler = _FeatureScaler(s["method"])
            scaler._loc = s["loc"]
            scaler._scale = s["scale"]
            obj._scalers[name] = scaler

        obj._encoders = {}
        for name, e in state["encoders"].items():
            enc = _CategoricalEncoder()
            enc._categories = list(e["categories"])
            enc._mapping = {cat: i for i, cat in enumerate(enc._categories)}
            obj._encoders[name] = enc

        return obj

    def save_state(self, path: Path | str) -> None:
        """Serialize fitted preprocessing state to ``path`` via ``torch.save``.

        Args:
            path: Destination file path (e.g. ``"dm_state.pt"``).
        """
        torch.save(self.get_state(), Path(path))

    @classmethod
    def load_state(cls, path: Path | str) -> DataModule:
        """Load a DataModule from a state file written by ``save_state()``.

        Args:
            path: Path to the state file.

        Returns:
            Reconstructed DataModule; identical transform behaviour to the
            original.
        """
        state = torch.load(Path(path), weights_only=True)
        return cls.from_state(state)
