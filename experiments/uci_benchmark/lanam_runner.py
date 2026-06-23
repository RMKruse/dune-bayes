"""External LA-NAM runner for the UCI benchmark (ADR-0008, GitHub #105).

This script is intentionally outside ``src/dune_bayes`` and is meant to run
under the experiments extra with the pinned upstream LA-NAM dependency.  It
consumes the common harness's already-split, already-preprocessed arrays, fits
the authors' mean-only Laplace-approximated NAM regressor, and writes
per-observation predictive samples/log-density/CDF for scoring by
``dune_bayes.metrics``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EPS = 1e-6


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--family",
        choices=["normal", "negative_binomial", "beta"],
        required=True,
        help=(
            "Shared-panel response family. LA-NAM remains mean-only; all "
            "families are scored as Gaussian mean predictives."
        ),
    )
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--predictive-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    del args.draws

    import pandas as pd
    import torch
    from laplace_skorch.regressor import LaplaceAdditiveNetRegressor
    from scipy.stats import norm

    torch.manual_seed(args.seed)
    payload = np.load(args.input)
    train_features = payload["train_features"].astype(np.float32)
    train_target = payload["train_target"].astype(np.float32)
    test_features = payload["test_features"].astype(np.float32)
    test_target = payload["test_target"].astype(np.float32)

    columns = [f"x{index}" for index in range(train_features.shape[1])]
    train_frame = pd.DataFrame(train_features, columns=columns, dtype=np.float32)
    test_frame = pd.DataFrame(test_features, columns=columns, dtype=np.float32)

    net = LaplaceAdditiveNetRegressor(
        lr=args.learning_rate,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device="cpu",
        verbose=0,
    )
    net.fit(train_frame, train_target)
    mean, std = net.predict(test_frame, return_std=True)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), EPS)

    rng = np.random.default_rng(args.seed)
    samples = rng.normal(
        loc=mean,
        scale=std,
        size=(args.predictive_samples, test_target.shape[0]),
    )
    log_density = norm.logpdf(test_target.astype(np.float64), loc=mean, scale=std)
    cdf = norm.cdf(test_target.astype(np.float64), loc=mean, scale=std)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        samples=samples,
        log_density=log_density,
        cdf=cdf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
