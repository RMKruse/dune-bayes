"""Spike 2 — does a VariationalDense model round-trip through save/load?

ADR-0004 claims the in-house layer retires the save/load fragility that sinks raw
`tfp.layers.DenseVariational` / `DenseFlipout` (their prior/posterior closures
don't serialize). The claim rests on a closure-free `get_config` that stores only
floats/strings, with the variational loc/scale saved as ordinary Keras weights.

This script builds the same nested-sub-model + DistributionLambda model as spike 1
and asserts, for each supported save format:

  [A] Layer-level get_config/from_config rebuilds the layer with identical prior
      hyperparameters and no closures.
  [B] A full model save -> load reconstructs without needing to re-supply the
      architecture, only custom_objects.
  [C] Every variational weight (loc/rho for kernel and bias) survives the
      round-trip elementwise, and the reloaded model runs a forward pass with the
      right output shape. (We compare *weights*, not stochastic predictions —
      reparameterization noise is not reproducible across two distinct layer
      objects, so a prediction match would be an unreliable test.)

Formats tried: Keras V3 (`.keras`), legacy H5 (`.h5`), and SavedModel dir. Each is
attempted independently; a format that the installed TF/TFP version cannot handle
is reported as a SKIP with the error, not a hard failure — the point of the spike
is to discover which format(s) work here.

Run:  python spike_serialization.py
Exit code 0 = at least one format fully round-trips and config checks pass.
"""

import os
import sys
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from variational_dense import VariationalDense, CUSTOM_OBJECTS, set_kl_beta

tfd = tfp.distributions
PARAM_COUNT = 2
N = 128


def normal_from_params(t):
    loc = t[..., 0]
    scale = 1e-3 + tf.math.softplus(t[..., 1])
    return tfd.Normal(loc=loc, scale=scale)


def make_feature_net(name, prior_scale=0.7):
    inp = tf.keras.Input(shape=(1,), name=f"{name}_in")
    h = VariationalDense(8, prior_scale=prior_scale, kl_divisor=N, activation="relu")(inp)
    out = VariationalDense(PARAM_COUNT, prior_scale=prior_scale, kl_divisor=N)(h)
    return tf.keras.Model(inp, out, name=name)


def build_model(n_features=2):
    inputs = {f"x{i}": tf.keras.Input(shape=(1,), name=f"x{i}") for i in range(n_features)}
    nets = [make_feature_net(f"feat{i}") for i in range(n_features)]
    contribs = [nets[i](inputs[f"x{i}"]) for i in range(n_features)]
    summed = tf.keras.layers.Add(name="summed_output")(contribs)
    # Return the raw params (not the distribution) so predictions are a plain
    # tensor we can compare numerically across save/load.
    return tf.keras.Model(inputs=inputs, outputs=summed)


def weights_match(model_a, model_b, atol=1e-6):
    """Elementwise comparison of every trainable+non-trainable weight, in order."""
    wa, wb = model_a.get_weights(), model_b.get_weights()
    if len(wa) != len(wb):
        return False, f"weight count {len(wa)} != {len(wb)}"
    worst = 0.0
    for a, b in zip(wa, wb):
        if a.shape != b.shape:
            return False, f"shape {a.shape} != {b.shape}"
        worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.0)
    return worst <= atol, f"max|Δw|={worst:.2e} over {len(wa)} arrays"


def toy_X(n_features=2):
    rng = np.random.default_rng(7)
    return {f"x{i}": rng.normal(size=(N, 1)).astype("float32") for i in range(n_features)}


def main():
    failures = []
    any_format_ok = False

    def check(name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}{' :: ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    # --- [A] layer config round-trips with no closures ---
    layer = VariationalDense(4, prior_scale=0.33, kl_divisor=99.0, activation="relu")
    layer.build((None, 5))
    cfg = layer.get_config()
    closure_free = all(
        not callable(v) for v in cfg.values()
    )
    rebuilt = VariationalDense.from_config(cfg)
    check(
        "A: get_config/from_config is closure-free and preserves hyperparams",
        closure_free
        and rebuilt.prior_scale == 0.33
        and rebuilt.kl_divisor == 99.0
        and rebuilt.units == 4,
        f"prior_scale={rebuilt.prior_scale}, kl_divisor={rebuilt.kl_divisor}",
    )

    # --- [B]/[C] full-model save -> load per format ---
    X = toy_X()
    model = build_model()
    set_kl_beta(model, 1.0)
    _ = model(X)  # build

    tmp = tempfile.mkdtemp(prefix="vd_spike_")
    formats = [
        ("keras_v3", os.path.join(tmp, "m.keras")),
        ("h5", os.path.join(tmp, "m.h5")),
        ("saved_model", os.path.join(tmp, "sm")),
    ]

    for fmt, path in formats:
        try:
            model.save(path)
            loaded = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
            # [C] weights survive elementwise...
            ok_w, detail_w = weights_match(model, loaded)
            # ...and the reloaded model actually runs with the right shape.
            out = loaded(X)
            ok_run = tuple(out.shape) == (N, PARAM_COUNT)
            ok = ok_w and ok_run
            check(
                f"B/C[{fmt}]: save->load reconstructs, weights survive, runs",
                ok,
                f"{detail_w}; out_shape={tuple(out.shape)}",
            )
            any_format_ok = any_format_ok or ok
        except Exception as exc:  # noqa: BLE001 — we want to report, not crash
            print(f"[SKIP] B/C[{fmt}]: format not usable here :: {type(exc).__name__}: {exc}")

    print()
    if failures or not any_format_ok:
        msg = []
        if failures:
            msg.append(f"{len(failures)} hard FAILURE(S): {failures}")
        if not any_format_ok:
            msg.append("no save format fully round-tripped")
        print("SPIKE 2 RESULT: " + "; ".join(msg))
        return 1
    print("SPIKE 2 RESULT: ALL PASS — closure-free config round-trips save/load.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
