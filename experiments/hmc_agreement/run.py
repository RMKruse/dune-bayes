"""VI-versus-NUTS validation experiment (ADR-0001/0008, GitHub #101)."""

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

import jax
import numpy as np
import torch
import torch.nn.functional as F
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

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

_fixed_model = importlib.import_module("experiments.hmc_agreement.model")

from dune_bayes.families import NormalFamily  # noqa: E402
from dune_bayes.model import BayesianNAMLSS  # noqa: E402
from dune_bayes.shapes import BayesianMLP  # noqa: E402
from dune_bayes.utils import EPS  # noqa: E402

_PARAMETER_LABELS = ("loc", "raw_scale")


def _generate_data(config: Mapping[str, Any], *, smoke: bool) -> dict[str, np.ndarray]:
    """Generate the seeded synthetic distributional-regression instance."""
    data_config = config["data"]
    n = min(int(data_config["n"]), 24) if smoke else int(data_config["n"])
    rng = np.random.default_rng(int(data_config["noise_seed"]))
    correlation = float(data_config["feature_correlation"])
    x1 = rng.normal(size=n)
    x2 = correlation * x1 + np.sqrt(1.0 - correlation**2) * rng.normal(size=n)
    truth = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in data_config["truth"].items()
    }
    predictor = (
        x1[:, None] * truth["x1_weight"]
        + x2[:, None] * truth["x2_weight"]
        + truth["intercept"]
    )
    scale = np.logaddexp(0.0, predictor[:, 1]) + EPS
    y = rng.normal(predictor[:, 0], scale)
    return {"x1": x1, "x2": x2, "y": y}


def _fit_vi(
    config: Mapping[str, Any], data: Mapping[str, np.ndarray], *, smoke: bool
) -> BayesianNAMLSS:
    """Fit the matching mean-field dune-bayes model."""
    architecture = config["architecture"]
    n = int(data["y"].shape[0])
    formula = {
        name: BayesianMLP(
            in_features=1,
            param_count=2,
            hidden_dims=list(architecture["hidden_dims"]),
            prior_scale=float(architecture["prior_scale"]),
            kl_divisor=float(n),
            validate_args=False,
        )
        for name in ("x1", "x2")
    }
    model = BayesianNAMLSS(
        formula=formula,
        family=NormalFamily(),
        n_obs=n,
        feature_dropout=0.0,
    )
    training = config["training"]
    epochs = min(int(training["epochs"]), 20) if smoke else int(training["epochs"])
    model.fit(
        {
            "x1": torch.as_tensor(data["x1"], dtype=torch.float32)[:, None],
            "x2": torch.as_tensor(data["x2"], dtype=torch.float32)[:, None],
        },
        torch.as_tensor(data["y"], dtype=torch.float32),
        epochs=epochs,
        lr=float(training["lr"]),
        warmup_epochs=min(int(training["warmup_epochs"]), epochs),
    )
    return model


def _vi_samples(model: BayesianNAMLSS, draws: int, seed: int) -> dict[str, np.ndarray]:
    """Draw directly from the fitted mean-field parameter posterior."""
    generator = torch.Generator().manual_seed(seed)
    samples: dict[str, np.ndarray] = {}
    for name in ("x1", "x2"):
        layer = model.nets[name].layers[-1]
        loc = layer.kernel_loc.detach()
        scale = F.softplus(layer.kernel_rho.detach())
        noise = torch.randn(draws, *loc.shape, generator=generator)
        samples[f"{name}_weight"] = (loc + scale * noise).squeeze(1).numpy()
    intercept_scale = F.softplus(model.intercept.rho.detach())
    intercept_noise = torch.randn(draws, model.intercept.units, generator=generator)
    samples["intercept"] = (
        model.intercept.loc.detach() + intercept_scale * intercept_noise
    ).numpy()
    return samples


def _run_nuts(
    config: Mapping[str, Any], data: Mapping[str, np.ndarray], *, smoke: bool
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    """Run NUTS and reduce convergence diagnostics for the run log."""
    nuts_config = config["nuts"]
    warmup = (
        min(int(nuts_config["warmup"]), 20) if smoke else int(nuts_config["warmup"])
    )
    samples = (
        min(int(nuts_config["samples"]), 20) if smoke else int(nuts_config["samples"])
    )
    chains = 2 if smoke else int(nuts_config["chains"])
    mcmc = MCMC(
        NUTS(
            _fixed_model.numpyro_model,
            target_accept_prob=float(nuts_config["target_accept_prob"]),
        ),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(int(config["seed"])), data=data)
    grouped = mcmc.get_samples(group_by_chain=True)
    diagnostic_table = summary(grouped, group_by_chain=True)
    r_hats = np.concatenate(
        [
            np.asarray(values["r_hat"]).reshape(-1)
            for values in diagnostic_table.values()
        ]
    )
    ess = np.concatenate(
        [
            np.asarray(values["n_eff"]).reshape(-1)
            for values in diagnostic_table.values()
        ]
    )
    divergences = int(
        np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum()
    )
    diagnostics: dict[str, int | float] = {
        "chains": chains,
        "warmup": warmup,
        "samples_per_chain": samples,
        "r_hat_max": float(np.nanmax(r_hats)),
        "ess_bulk_min": float(np.nanmin(ess)),
        "divergences": divergences,
    }
    flattened = {
        name: np.asarray(value).reshape(chains * samples, 2)
        for name, value in grouped.items()
    }
    return flattened, diagnostics


def _interval(values: np.ndarray, credible_mass: float) -> tuple[np.ndarray, ...]:
    """Return lower, median, and upper arrays along the draw dimension."""
    tail = (1.0 - credible_mass) / 2.0
    lower, center, upper = np.quantile(values, [tail, 0.5, 1.0 - tail], axis=0)
    return lower, center, upper


def _write_parameter_intervals(
    paths: ArtifactPaths,
    vi: Mapping[str, np.ndarray],
    nuts: Mapping[str, np.ndarray],
    credible_mass: float,
) -> None:
    """Write one VI/NUTS interval row per scalar model parameter."""
    rows: list[dict[str, str | float]] = []
    for name in ("x1_weight", "x2_weight", "intercept"):
        vi_interval = _interval(vi[name], credible_mass)
        nuts_interval = _interval(nuts[name], credible_mass)
        for index, label in enumerate(_PARAMETER_LABELS):
            rows.append(
                {
                    "parameter": f"{name}[{label}]",
                    "vi_lower": float(vi_interval[0][index]),
                    "vi_center": float(vi_interval[1][index]),
                    "vi_upper": float(vi_interval[2][index]),
                    "nuts_lower": float(nuts_interval[0][index]),
                    "nuts_center": float(nuts_interval[1][index]),
                    "nuts_upper": float(nuts_interval[2][index]),
                }
            )
    destination = paths.metrics / "parameter_intervals.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _effect_bands(
    samples: Mapping[str, np.ndarray], grid: np.ndarray, credible_mass: float
) -> dict[str, np.ndarray]:
    """Compute centered epistemic bands for each feature contribution."""
    bands: dict[str, np.ndarray] = {}
    for feature in ("x1", "x2"):
        effects = samples[f"{feature}_weight"][:, None, :] * grid[None, :, None]
        effects = effects - effects.mean(axis=1, keepdims=True)
        lower, center, upper = _interval(effects, credible_mass)
        bands[f"{feature}_lower"] = lower
        bands[f"{feature}_center"] = center
        bands[f"{feature}_upper"] = upper
    return bands


def _write_band_artifacts(
    paths: ArtifactPaths,
    grid: np.ndarray,
    vi_bands: Mapping[str, np.ndarray],
    nuts_bands: Mapping[str, np.ndarray],
    credible_mass: float,
) -> None:
    """Write effect arrays, comparison metrics, and the overlay figure."""
    arrays: dict[str, np.ndarray] = {"grid": grid}
    arrays.update({f"vi_{key}": value for key, value in vi_bands.items()})
    arrays.update({f"nuts_{key}": value for key, value in nuts_bands.items()})
    np.savez(paths.arrays / "posterior_bands.npz", **arrays)

    ratio_rows: list[dict[str, str | float]] = []
    figure, axes = plt.subplots(2, 2, figsize=(8, 6))
    for row, feature in enumerate(("x1", "x2")):
        vi_lower = vi_bands[f"{feature}_lower"]
        vi_center = vi_bands[f"{feature}_center"]
        vi_upper = vi_bands[f"{feature}_upper"]
        nuts_lower = nuts_bands[f"{feature}_lower"]
        nuts_center = nuts_bands[f"{feature}_center"]
        nuts_upper = nuts_bands[f"{feature}_upper"]
        for column, parameter in enumerate(_PARAMETER_LABELS):
            vi_width = vi_upper[:, column] - vi_lower[:, column]
            nuts_width = nuts_upper[:, column] - nuts_lower[:, column]
            usable = nuts_width > np.finfo(np.float64).eps
            ratio = vi_width[usable] / nuts_width[usable]
            center_difference = (
                np.abs(vi_center[usable, column] - nuts_center[usable, column])
                / nuts_width[usable]
            )
            inside = (vi_lower[:, column] >= nuts_lower[:, column]) & (
                vi_upper[:, column] <= nuts_upper[:, column]
            )
            ratio_rows.append(
                {
                    "feature": feature,
                    "distribution_parameter": parameter,
                    "median_vi_to_nuts_width_ratio": float(np.median(ratio)),
                    "vi_inside_nuts_fraction": float(np.mean(inside[usable])),
                    "median_normalized_center_difference": float(
                        np.median(center_difference)
                    ),
                }
            )
            axis = axes[row, column]
            axis.fill_between(
                grid,
                nuts_lower[:, column],
                nuts_upper[:, column],
                color="C0",
                alpha=0.2,
                label=f"NUTS {credible_mass:.0%}" if row == column == 0 else None,
            )
            axis.fill_between(
                grid,
                vi_lower[:, column],
                vi_upper[:, column],
                color="C1",
                alpha=0.35,
                label=f"VI {credible_mass:.0%}" if row == column == 0 else None,
            )
            axis.plot(grid, nuts_center[:, column], color="C0", linewidth=1)
            axis.plot(grid, vi_center[:, column], color="C1", linewidth=1)
            axis.set(
                title=f"{feature}: {parameter.replace('_', ' ')}",
                xlabel=feature,
                ylabel="effect",
            )
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(
        paths.figures / "vi_vs_nuts.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)

    destination = paths.metrics / "band_width_ratios.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ratio_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ratio_rows)


def _run(config: Mapping[str, Any], paths: ArtifactPaths, smoke: bool) -> None:
    """Execute the fixed VI-versus-NUTS comparison."""
    data = _generate_data(config, smoke=smoke)
    truth = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in config["data"]["truth"].items()
    }
    torch_joint = _fixed_model.torch_log_joint(data, truth)
    jax_joint = _fixed_model.jax_log_joint(data, truth)
    if not np.isclose(torch_joint, jax_joint, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            f"Torch/JAX log-joint mismatch: {torch_joint} versus {jax_joint}"
        )

    vi_model = _fit_vi(config, data, smoke=smoke)
    comparison = config["comparison"]
    vi_draws = (
        min(int(comparison["vi_draws"]), 40) if smoke else int(comparison["vi_draws"])
    )
    vi = _vi_samples(vi_model, vi_draws, int(config["seed"]) + 1)
    nuts, diagnostics = _run_nuts(config, data, smoke=smoke)
    credible_mass = float(comparison["credible_mass"])
    grid_points = (
        min(int(comparison["grid_points"]), 11)
        if smoke
        else int(comparison["grid_points"])
    )
    grid = np.linspace(-1.2, 1.2, grid_points)

    _write_parameter_intervals(paths, vi, nuts, credible_mass)
    _write_band_artifacts(
        paths,
        grid,
        _effect_bands(vi, grid, credible_mass),
        _effect_bands(nuts, grid, credible_mass),
        credible_mass,
    )
    (paths.metrics / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _enrich_run_log(paths: ArtifactPaths) -> None:
    """Put NUTS convergence diagnostics in the conventional root run log."""
    run_path = paths.root / "run.json"
    run_log = json.loads(run_path.read_text(encoding="utf-8"))
    run_log["nuts"] = json.loads(
        (paths.metrics / "diagnostics.json").read_text(encoding="utf-8")
    )
    run_path.write_text(
        json.dumps(run_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the HMC agreement experiment from a complete config.

    Args:
        argv: Optional CLI arguments; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    paths = run_experiment(args.config, smoke=args.smoke, experiment=_run)
    _enrich_run_log(paths)


if __name__ == "__main__":
    main()
