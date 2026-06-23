"""Dense/noisy versus sparse/quiet simulation (ADR-0008, GitHub #99)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

_cache_root = Path(tempfile.gettempdir()) / "dune-bayes-experiments-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_harness = importlib.import_module("experiments._harness")
ArtifactPaths = _harness.ArtifactPaths
run_experiment = _harness.run_experiment

from dune_bayes.families import NormalFamily  # noqa: E402
from dune_bayes.metrics import variance_decomposition  # noqa: E402
from dune_bayes.model import BayesianNAMLSS  # noqa: E402
from dune_bayes.sampling import draw_predictive  # noqa: E402
from dune_bayes.shapes import BayesianMLP  # noqa: E402
from dune_bayes.utils import EPS  # noqa: E402


def _truth(config: Mapping[str, Any], x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return location, scale, and raw Normal-family predictors."""
    truth = config["truth"]
    location = float(truth["location_amplitude"]) * torch.sin(torch.pi * x)
    width = float(truth["transition_width"])
    dense_weight = torch.sigmoid(-x / width)
    scale = (
        float(truth["quiet_scale"])
        + (float(truth["noisy_scale"]) - float(truth["quiet_scale"])) * dense_weight
    )
    # Inverse-softplus makes the simulated scale exactly known by construction.
    raw_scale = torch.log(torch.expm1(scale))
    return location, scale, torch.stack((location, raw_scale), dim=-1)


def _sample_covariate(config: Mapping[str, Any], n: int) -> torch.Tensor:
    """Sample abundant left-region and sparse right-region observations."""
    data = config["data"]
    n_dense = round(n * float(data["dense_fraction"]))
    n_sparse = n - n_dense
    dense = torch.empty(n_dense).uniform_(*map(float, data["dense_region"]))
    sparse = torch.empty(n_sparse).uniform_(*map(float, data["sparse_region"]))
    return torch.cat((dense, sparse)).sort().values


def _regional_mean(values: torch.Tensor, grid: torch.Tensor, *, sparse: bool) -> float:
    """Average a curve on one side of the pre-registered region boundary."""
    mask = grid > 0.0 if sparse else grid < 0.0
    return float(values[mask].mean())


def _plot(
    path: Path,
    x: torch.Tensor,
    y: torch.Tensor,
    grid: torch.Tensor,
    truth_location: torch.Tensor,
    truth_aleatoric: torch.Tensor,
    aleatoric: torch.Tensor,
    epistemic: torch.Tensor,
) -> None:
    """Render the paper-facing design and decomposition figure."""
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    axes[0].scatter(x.numpy(), y.numpy(), s=8, alpha=0.3, label="observations")
    axes[0].plot(grid.numpy(), truth_location.numpy(), "k--", label="true mean")
    axes[0].axvspan(-1.0, 0.0, alpha=0.08, color="tab:orange", label="dense + noisy")
    axes[0].axvspan(0.0, 1.0, alpha=0.08, color="tab:blue", label="sparse + quiet")
    axes[0].set(ylabel="response", title="Simulation design")
    axes[0].legend(ncol=2, fontsize=8, loc="upper right")
    axes[1].plot(
        grid.numpy(), aleatoric.numpy(), color="tab:orange", label="estimated aleatoric"
    )
    axes[1].plot(
        grid.numpy(), epistemic.numpy(), color="tab:blue", label="estimated epistemic"
    )
    axes[1].plot(
        grid.numpy(),
        truth_aleatoric.numpy(),
        "--",
        color="tab:orange",
        label="true noise variance",
    )
    axes[1].axvline(0.0, color="0.5", linewidth=0.8)
    axes[1].set(
        xlabel="covariate x",
        ylabel="variance",
        title="Predictive variance decomposition",
    )
    axes[1].legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def _run(config: Mapping[str, Any], paths: ArtifactPaths, smoke: bool) -> None:
    """Fit and decompose the configured synthetic Normal regression."""
    n = min(int(config["data"]["n"]), 48) if smoke else int(config["data"]["n"])
    grid_size = (
        min(int(config["data"]["grid_size"]), 64)
        if smoke
        else int(config["data"]["grid_size"])
    )
    draws_count = min(int(config["draws"]), 32) if smoke else int(config["draws"])
    epochs = (
        min(int(config["training"]["epochs"]), 120)
        if smoke
        else int(config["training"]["epochs"])
    )
    x = _sample_covariate(config, n)
    truth_location, _, raw_params = _truth(config, x)
    family = NormalFamily()
    y = family(raw_params).sample()
    hidden_dims = [int(width) for width in config["architecture"]["hidden_dims"]]
    model = BayesianNAMLSS(
        formula={
            "x": BayesianMLP(
                1,
                family.param_count,
                hidden_dims=hidden_dims,
                prior_scale=float(config["architecture"]["prior_scale"]),
                kl_divisor=n,
                activation=str(config["architecture"]["activation"]),
            )
        },
        family=family,
        n_obs=n,
    )
    features = {"x": x.unsqueeze(-1)}
    history = model.fit(
        features,
        y,
        epochs=epochs,
        lr=float(config["training"]["learning_rate"]),
        warmup_epochs=min(int(config["training"]["warmup_epochs"]), epochs),
    )
    grid = torch.linspace(-1.0, 1.0, grid_size)
    truth_grid_location, truth_grid_scale, _ = _truth(config, grid)
    draws = draw_predictive(model, {"x": grid.unsqueeze(-1)}, T=draws_count)
    components = variance_decomposition(model, draws.summed_samples)
    metrics = {
        "aleatoric_dense": _regional_mean(components.aleatoric, grid, sparse=False),
        "aleatoric_sparse": _regional_mean(components.aleatoric, grid, sparse=True),
        "draws": draws_count,
        "epochs": epochs,
        "epistemic_dense": _regional_mean(components.epistemic, grid, sparse=False),
        "epistemic_sparse": _regional_mean(components.epistemic, grid, sparse=True),
        "final_loss": history["loss"][-1],
        "n": n,
    }
    # Ratios encode the pre-registered directional claims directly in the run
    # log. EPS is the package-wide float32 floor for learned denominators.
    metrics["epistemic_sparse_to_dense_ratio"] = metrics["epistemic_sparse"] / (
        metrics["epistemic_dense"] + EPS
    )
    metrics["aleatoric_dense_to_sparse_ratio"] = metrics["aleatoric_dense"] / (
        metrics["aleatoric_sparse"] + EPS
    )
    (paths.metrics / "regional_components.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez(
        paths.arrays / "decomposition.npz",
        x=x.numpy(),
        y=y.numpy(),
        grid=grid.numpy(),
        truth_location=truth_grid_location.numpy(),
        truth_aleatoric=truth_grid_scale.square().numpy(),
        summed_samples=draws.summed_samples.numpy(),
        aleatoric=components.aleatoric.numpy(),
        epistemic=components.epistemic.numpy(),
        total=components.total.numpy(),
    )
    _plot(
        paths.figures / "disentanglement.pdf",
        x,
        y,
        grid,
        truth_grid_location,
        truth_grid_scale.square(),
        components.aleatoric,
        components.epistemic,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the disentanglement simulation from one complete config.

    Args:
        argv: Optional CLI arguments; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run_experiment(args.config, smoke=args.smoke, experiment=_run)


if __name__ == "__main__":
    main()
