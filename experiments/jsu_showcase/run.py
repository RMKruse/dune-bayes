"""Johnson's SU heteroscedastic-skew showcase (ADR-0008, GitHub #100)."""

from __future__ import annotations

import argparse
import csv
import importlib
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

from dune_bayes.families import JohnsonSUFamily  # noqa: E402
from dune_bayes.metrics import quantile_coverage  # noqa: E402
from dune_bayes.model import BayesianNAMLSS  # noqa: E402
from dune_bayes.plotting import plot_effect_ribbon  # noqa: E402
from dune_bayes.sampling import sample_effects  # noqa: E402
from dune_bayes.shapes import BayesianMLP  # noqa: E402


def _truth(config: Mapping[str, Any], x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return centered raw-predictor effects, intercept, and linked parameters."""
    columns = []
    for specification in config["truth"]["effects"]:
        kind = str(specification["kind"])
        if kind == "linear":
            basis = x
        else:
            phase = float(specification["frequency"]) * torch.pi * x
            basis = torch.sin(phase) if kind == "sin" else torch.cos(phase)
        columns.append(float(specification["amplitude"]) * basis)
    effects = torch.stack(columns, dim=-1)
    effects = effects - effects.mean(dim=0, keepdim=True)
    intercept = torch.tensor(config["truth"]["intercept"], dtype=torch.float32)
    raw_parameters = effects + intercept
    distribution = JohnsonSUFamily()(raw_parameters)
    linked = torch.stack(
        (
            distribution.skew,
            distribution.tail,
            distribution.loc,
            distribution.scale,
        ),
        dim=-1,
    )
    return effects, intercept, raw_parameters, linked


def _write_coverage(
    path: Path,
    parameter_names: list[str],
    levels: tuple[float, ...],
    coverage: torch.Tensor,
) -> None:
    """Write long-form centered shape-function coverage."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("parameter", "nominal", "empirical_coverage"),
            lineterminator="\n",
        )
        writer.writeheader()
        for parameter_index, parameter in enumerate(parameter_names):
            for level_index, level in enumerate(levels):
                writer.writerow(
                    {
                        "parameter": parameter,
                        "nominal": level,
                        "empirical_coverage": float(
                            coverage[parameter_index, level_index]
                        ),
                    }
                )


def _plot_ribbons(
    path: Path,
    grid: torch.Tensor,
    truth: torch.Tensor,
    draws: torch.Tensor,
    parameter_names: list[str],
    credible_level: float,
) -> None:
    """Render one centered epistemic effect ribbon for every JSU parameter."""
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, parameter_names, strict=True)):
        plot_effect_ribbon(
            draws,
            grid,
            credible_interval=credible_level,
            center=True,
            param_idx=index,
            ax=axis,
            feature_name="x",
        )
        axis.plot(grid.numpy(), truth[:, index].numpy(), "k--", label="truth")
        axis.set(title=name, ylabel=f"centered raw {name}")
    axes.flat[0].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def _run(config: Mapping[str, Any], paths: ArtifactPaths, smoke: bool) -> None:
    """Fit the configured JSU simulation and emit effect-recovery evidence."""
    n = min(int(config["data"]["n"]), 48) if smoke else int(config["data"]["n"])
    grid_size = (
        min(int(config["data"]["grid_size"]), 48)
        if smoke
        else int(config["data"]["grid_size"])
    )
    draws_count = min(int(config["draws"]), 32) if smoke else int(config["draws"])
    epochs = (
        min(int(config["training"]["epochs"]), 4)
        if smoke
        else int(config["training"]["epochs"])
    )
    x = torch.linspace(-1.0, 1.0, n)
    _, _, raw_parameters, _ = _truth(config, x)
    family = JohnsonSUFamily()
    y = family(raw_parameters).sample()
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
    model.fit(
        {"x": x.unsqueeze(-1)},
        y,
        epochs=epochs,
        lr=float(config["training"]["learning_rate"]),
        warmup_epochs=min(int(config["training"]["warmup_epochs"]), epochs),
    )

    grid = torch.linspace(-1.0, 1.0, grid_size)
    truth, intercept_truth, _, linked_truth = _truth(config, grid)
    effect_draws = sample_effects(model, {"x": grid.unsqueeze(-1)}, T=draws_count)["x"]
    centered_draws = effect_draws - effect_draws.mean(dim=1, keepdim=True)
    levels = tuple(float(level) for level in config["coverage"]["levels"])
    coverage = quantile_coverage(centered_draws, truth, levels=levels).coverage
    with torch.no_grad():
        # Shape-function levels are non-identifiable. Assess the model intercept
        # plus the uncentered effect mean as the separately absorbed level.
        intercept_draws = model.intercept(n_samples=draws_count) + effect_draws.mean(
            dim=1
        )
    intercept_coverage = quantile_coverage(
        intercept_draws.unsqueeze(1), intercept_truth.unsqueeze(0), levels=levels
    ).coverage
    parameter_names = [str(name) for name in config["parameter_names"]]
    _write_coverage(paths.metrics / "coverage.csv", parameter_names, levels, coverage)
    _write_coverage(
        paths.metrics / "intercept_coverage.csv",
        parameter_names,
        levels,
        intercept_coverage,
    )
    _plot_ribbons(
        paths.figures / "effect_ribbons.pdf",
        grid,
        truth,
        effect_draws,
        parameter_names,
        float(config["ribbons"]["credible_level"]),
    )
    np.savez(
        paths.arrays / "recovery.npz",
        grid=grid.numpy(),
        truth=truth.numpy(),
        centered_draws=centered_draws.numpy(),
        intercept_truth=intercept_truth.numpy(),
        intercept_draws=intercept_draws.numpy(),
        linked_truth=linked_truth.numpy(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the JSU showcase from one complete config.

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
