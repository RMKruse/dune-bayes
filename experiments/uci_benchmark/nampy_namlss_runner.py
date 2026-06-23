"""External Normal-NAMLSS runner for the UCI benchmark (ADR-0008, GitHub #104).

This script is intentionally outside ``src/dune_bayes`` and is meant to run
under a separate TensorFlow/Keras/TFP interpreter.  It consumes the common
harness's already-split, already-preprocessed arrays, fits the deterministic
NAMLSS graph shape from the original paper code, and writes per-observation
predictive samples/log-density/CDF for scoring by ``dune_bayes.metrics``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np


def _install_optional_paper_stubs() -> None:
    """Stub paper-only comparator imports that NAMLSS itself never touches."""
    if "xgboost" not in sys.modules:
        xgboost = types.ModuleType("xgboost")

        class _UnusedXGB:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("XGBoost is not required for the NAMLSS runner.")

        xgboost.XGBClassifier = _UnusedXGB
        xgboost.XGBRegressor = _UnusedXGB
        sys.modules["xgboost"] = xgboost
    if "interpret" not in sys.modules:
        interpret = types.ModuleType("interpret")
        glassbox = types.ModuleType("interpret.glassbox")

        class _UnusedEBM:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("InterpretML is not required for the NAMLSS runner.")

        glassbox.ExplainableBoostingClassifier = _UnusedEBM
        glassbox.ExplainableBoostingRegressor = _UnusedEBM
        interpret.glassbox = glassbox
        sys.modules["interpret"] = interpret
        sys.modules["interpret.glassbox"] = glassbox


def _load_paper_module(paper_code_dir: Path) -> Any:
    """Import the supplied paper script after isolating optional dependencies."""
    _install_optional_paper_stubs()
    sys.path.insert(0, str(paper_code_dir))
    return importlib.import_module("NeurIPS_Benchmarks")


def _build_normal_namlss(
    paper: Any,
    *,
    n_features: int,
    learning_rate: float,
) -> tuple[Any, Any]:
    """Build the original paper's two-head additive Normal NAMLSS graph."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    inputs = [tf.keras.Input(shape=(1,)) for _ in range(2 * n_features)]
    loc_models = [paper.built_DNN(inputs[index]) for index in range(n_features)]
    scale_models = [
        paper.define_models_scale(inputs[index + n_features])
        for index in range(n_features)
    ]
    raw_loc = tf.keras.layers.Add()([model.output for model in loc_models])
    raw_scale = tf.keras.layers.Add()([model.output for model in scale_models])
    raw_params = tf.keras.layers.Concatenate()([raw_loc, raw_scale])
    predictive = tfp.layers.DistributionLambda(
        lambda value: tfp.distributions.Normal(
            loc=paper.LINEAR(value[:, 0]),
            scale=tf.math.softplus(value[:, 1]),
        )
    )(raw_params)
    model = tf.keras.Model(
        inputs=[model.input for model in [*loc_models, *scale_models]],
        outputs=predictive,
    )
    parameter_model = tf.keras.Model(inputs=model.inputs, outputs=raw_params)

    def nll(y_true: Any, y_hat: Any) -> Any:
        return -y_hat.log_prob(y_true)

    model.compile(
        loss=nll,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )
    return model, parameter_model


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-code-dir", type=Path, required=True)
    parser.add_argument("--family", choices=["normal"], required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--predictive-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    # Keep the live comparator CPU-bound by default.  Users can override this
    # outside the package when deliberately validating GPU parity.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import tensorflow as tf
    import tensorflow_probability as tfp

    tf.keras.utils.set_random_seed(args.seed)
    with suppress(Exception):
        tf.config.experimental.enable_op_determinism()
    # Older TF builds may not expose this hook; the seed still fixes the
    # stochastic path for the smoke-scale validation we run here.

    payload = np.load(args.input)
    train_features = payload["train_features"].astype(np.float32)
    train_target = payload["train_target"].astype(np.float32)
    test_features = payload["test_features"].astype(np.float32)
    test_target = payload["test_target"].astype(np.float32)

    paper = _load_paper_module(args.paper_code_dir)
    model, parameter_model = _build_normal_namlss(
        paper,
        n_features=train_features.shape[1],
        learning_rate=args.learning_rate,
    )
    train_inputs = [
        train_features[:, index] for index in range(train_features.shape[1])
    ] * 2
    test_inputs = [
        test_features[:, index] for index in range(test_features.shape[1])
    ] * 2
    model.fit(
        train_inputs,
        train_target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )

    raw_params = parameter_model.predict(test_inputs, verbose=0)
    loc = paper.LINEAR(raw_params[:, 0])
    scale = tf.math.softplus(raw_params[:, 1])
    distribution = tfp.distributions.Normal(loc=loc, scale=scale)
    samples = distribution.sample(args.predictive_samples, seed=args.seed).numpy()
    log_density = distribution.log_prob(test_target).numpy()
    cdf = distribution.cdf(test_target).numpy()

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
