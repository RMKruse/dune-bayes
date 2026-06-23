"""Trivial end-to-end experiment scaffold (ADR-0008, GitHub #97)."""

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

# CI and sandboxed runners may have read-only home directories. Keep plotting
# caches in a writable process-independent location so smoke runs remain cheap.
_cache_root = Path(tempfile.gettempdir()) / "dune-bayes-experiments-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

# Direct ``python experiments/<name>/run.py`` execution puts only this file's
# directory on sys.path; expose the repository-local, deliberately un-packaged
# harness without adding experiments to the dune_bayes runtime namespace.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_harness = importlib.import_module("experiments._harness")
ArtifactPaths = _harness.ArtifactPaths
run_experiment = _harness.run_experiment


def _run(config: Mapping[str, Any], paths: ArtifactPaths, smoke: bool) -> None:
    draws = min(int(config["draws"]), 8) if smoke else int(config["draws"])
    n = min(int(config["data"]["n"]), 16) if smoke else int(config["data"]["n"])
    epochs = 1 if smoke else int(config["training"]["epochs"])
    samples = np.random.normal(size=(draws, n))
    mean = samples.mean(axis=0)

    np.savez(paths.arrays / "samples.npz", samples=samples)
    (paths.metrics / "metrics.json").write_text(
        json.dumps(
            {
                "draws": draws,
                "epochs": epochs,
                "n": n,
                "sample_mean": float(samples.mean()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(4, 2.5))
    axis.plot(mean)
    axis.set(xlabel="observation", ylabel="posterior mean")
    figure.tight_layout()
    # Wall-clock PDF metadata would make a seeded regeneration byte-different.
    figure.savefig(
        paths.figures / "sample_mean.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the walking-skeleton experiment from one complete config.

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
