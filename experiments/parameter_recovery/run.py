"""Parameter-recovery simulations (ADR-0001/0008, GitHub #98)."""

from __future__ import annotations

import argparse
import csv
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

from dune_bayes.families import GammaFamily, NormalFamily, StudentTFamily  # noqa: E402
from dune_bayes.metrics import quantile_coverage  # noqa: E402
from dune_bayes.model import BayesianNAMLSS  # noqa: E402
from dune_bayes.sampling import sample_effects  # noqa: E402
from dune_bayes.shapes import BayesianMLP  # noqa: E402


def _truth(
    config: Mapping[str, Any], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered raw-predictor effects and their separate intercept."""
    columns = []
    for specification in config["truth"]["effects"]:
        phase = float(specification["frequency"]) * torch.pi * x
        basis = torch.sin(phase) if specification["kind"] == "sin" else torch.cos(phase)
        columns.append(float(specification["amplitude"]) * basis)
    effects = torch.stack(columns, dim=-1)
    effects = effects - effects.mean(dim=0, keepdim=True)
    return effects, torch.tensor(config["truth"]["intercept"], dtype=torch.float32)


def _family(family_name: str) -> NormalFamily | StudentTFamily | GammaFamily:
    """Construct the configured core continuous family."""
    if family_name == "normal":
        return NormalFamily()
    if family_name == "student_t":
        # Finite variance is part of the experiment contract (CONTEXT.md).
        return StudentTFamily(df_min=2.0)
    if family_name == "gamma":
        return GammaFamily()
    raise ValueError(f"unsupported family: {family_name}")


def _write_calibration(
    path: Path,
    parameter_names: list[str],
    levels: tuple[float, ...],
    coverage: torch.Tensor,
) -> None:
    """Write the long-form per-parameter calibration table."""
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


def _write_prior_scale_diagnostics(
    path: Path,
    model: BayesianNAMLSS,
    architecture: Mapping[str, Any],
) -> None:
    """Write the prior/smoothness state used by the recovery run."""
    shape = model.nets["x"]
    handle = getattr(shape, "prior_scale_handle", None)
    diagnostics: dict[str, Any] = {
        "prior": architecture.get("prior"),
        "configured_prior_scale": float(architecture["prior_scale"]),
    }

    if handle is None:
        diagnostics.update(
            {
                "mode": "fixed",
                "scale": float(architecture["prior_scale"]),
            }
        )
    elif handle.mode == "empirical_bayes":
        diagnostics.update(
            {
                "mode": handle.mode,
                "initial_scale": handle.initial_scale,
                "scale": float(torch.nn.functional.softplus(handle.rho).detach()),
            }
        )
    else:
        sigma = torch.nn.functional.softplus(handle.rho_s).detach()
        loc = handle.loc_s.detach()
        diagnostics.update(
            {
                "mode": handle.mode,
                "hyperprior": handle.hyperprior,
                "initial_scale": handle.initial_scale,
                "log_scale_loc": float(loc),
                "log_scale_sigma": float(sigma),
                "scale_median": float(torch.exp(loc)),
                "scale_mean": float(torch.exp(loc + 0.5 * sigma.pow(2))),
            }
        )

    path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")


def _plot_recovery(
    path: Path,
    x: torch.Tensor,
    truth: torch.Tensor,
    draws: torch.Tensor,
    parameter_names: list[str],
    credible_level: float,
) -> None:
    """Plot centered truth against posterior epistemic ribbons."""
    figure, axes = plt.subplots(
        len(parameter_names), 1, figsize=(6, 2.6 * len(parameter_names)), sharex=True
    )
    axes_array = np.atleast_1d(axes)
    x_np = x.detach().cpu().numpy()
    for index, (axis, name) in enumerate(zip(axes_array, parameter_names, strict=True)):
        values = draws[..., index].detach().cpu().numpy()
        tail = (1.0 - credible_level) / 2.0
        lo, median, hi = np.quantile(values, (tail, 0.5, 1.0 - tail), axis=0)
        axis.fill_between(
            x_np,
            lo,
            hi,
            alpha=0.25,
            label=f"{credible_level:.0%} epistemic band",
        )
        axis.plot(x_np, median, label="posterior median")
        axis.plot(x_np, truth[:, index].cpu().numpy(), "k--", label="truth")
        axis.set(ylabel=f"centered {name}")
    axes_array[0].legend(loc="best")
    axes_array[-1].set_xlabel("x")
    figure.tight_layout()
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def _plot_calibration(
    path: Path,
    parameter_names: list[str],
    levels: tuple[float, ...],
    coverage: torch.Tensor,
) -> None:
    """Plot empirical against nominal coverage for every parameter."""
    figure, axes = plt.subplots(
        1, len(parameter_names), figsize=(3.2 * len(parameter_names), 3.0)
    )
    axes_array = np.atleast_1d(axes)
    for index, (axis, name) in enumerate(zip(axes_array, parameter_names, strict=True)):
        axis.plot(levels, levels, "k--", label="nominal")
        axis.plot(
            levels,
            coverage[index].detach().cpu().numpy(),
            marker="o",
            label="measured",
        )
        axis.set(
            title=name,
            xlabel="nominal coverage",
            ylabel="empirical coverage",
            xlim=(0.45, 1.0),
            ylim=(0.0, 1.0),
        )
    axes_array[0].legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def _run(config: Mapping[str, Any], paths: ArtifactPaths, smoke: bool) -> None:
    """Fit one configured family and emit recovery evidence."""
    family_name = str(config["family"])
    n = min(int(config["data"]["n"]), 32) if smoke else int(config["data"]["n"])
    draws_count = min(int(config["draws"]), 32) if smoke else int(config["draws"])
    epochs = (
        min(int(config["training"]["epochs"]), 4)
        if smoke
        else int(config["training"]["epochs"])
    )
    x = torch.linspace(-1.0, 1.0, n)
    truth, intercept_truth = _truth(config, x)
    raw_params = truth + intercept_truth
    family = _family(family_name)
    y = family(raw_params).sample()
    architecture = config["architecture"]
    hidden_dims = [int(width) for width in architecture["hidden_dims"]]
    model = BayesianNAMLSS(
        formula={
            "x": BayesianMLP(
                1,
                family.param_count,
                hidden_dims=hidden_dims,
                prior_scale=float(architecture["prior_scale"]),
                prior=architecture.get("prior"),
                kl_divisor=n,
                activation=str(architecture["activation"]),
            )
        },
        family=family,
        n_obs=n,
    )
    features = {"x": x.unsqueeze(-1)}
    model.fit(
        features,
        y,
        epochs=epochs,
        lr=float(config["training"]["learning_rate"]),
        warmup_epochs=min(int(config["training"]["warmup_epochs"]), epochs),
    )
    effect_draws = sample_effects(model, features, T=draws_count)["x"]
    centered_draws = effect_draws - effect_draws.mean(dim=1, keepdim=True)
    levels = tuple(float(level) for level in config["calibration"]["levels"])
    coverage = quantile_coverage(centered_draws, truth, levels=levels).coverage
    with torch.no_grad():
        # Shape-function levels are non-identifiable. The separately assessed
        # level is the model intercept plus the mean effect absorbed into it.
        intercept_draws = model.intercept(n_samples=draws_count) + effect_draws.mean(
            dim=1
        )
    intercept_coverage = quantile_coverage(
        intercept_draws.unsqueeze(1), intercept_truth.unsqueeze(0), levels=levels
    ).coverage

    parameter_names = [str(name) for name in config["parameter_names"]]
    _write_calibration(
        paths.metrics / "calibration.csv", parameter_names, levels, coverage
    )
    _write_calibration(
        paths.metrics / "intercept_coverage.csv",
        parameter_names,
        levels,
        intercept_coverage,
    )
    _write_prior_scale_diagnostics(
        paths.metrics / "prior_scale.json", model, architecture
    )
    _plot_recovery(
        paths.figures / "recovery.pdf",
        x,
        truth,
        centered_draws,
        parameter_names,
        float(config["recovery"]["credible_level"]),
    )
    _plot_calibration(
        paths.figures / "calibration.pdf", parameter_names, levels, coverage
    )
    np.savez(
        paths.arrays / "recovery.npz",
        x=x.numpy(),
        truth=truth.numpy(),
        centered_draws=centered_draws.numpy(),
        intercept_truth=intercept_truth.numpy(),
        intercept_draws=intercept_draws.numpy(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one family config through the recovery simulation.

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
