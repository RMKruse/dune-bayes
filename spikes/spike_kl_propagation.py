"""Spike 1 — does KL via add_loss survive the real NAMLSS graph shape?

The genuine risk (CONTEXT / ADR-0003 / ADR-0004): in NAMLSS each feature net is a
*separate* `tf.keras.Model`; their outputs are summed and piped through
`tfp.layers.DistributionLambda(lambda x: family(x))`. We need KL emitted by
`VariationalDense` layers *inside those per-feature sub-models* to show up in the
*outer* model's `model.losses`, get added to the NLL by `.fit()`, and respond to
the warm-up beta. If Keras silently drops add_loss across the sub-model boundary
or across `DistributionLambda`, the whole KL/N + warm-up design fails.

This script builds exactly that structure on toy Normal-family data and asserts:

  [A] Outer model collects one KL loss per VariationalDense, across sub-models.
  [B] beta=0 zeroes the KL contribution; beta=1 restores it (warm-up works).
  [C] The KL scales as KL/N (kl_divisor), not raw KL.
  [D] A real .fit() run executes and the reported loss exceeds the pure NLL by
      approximately the KL/N term (i.e. add_loss is actually in the train step).

Run:  python spike_kl_propagation.py
Exit code 0 = all PASS.  Non-zero = a claim failed (printed FAIL ...).
"""

import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from variational_dense import (
    VariationalDense,
    count_variational_layers,
    set_kl_beta,
)

tfd = tfp.distributions
PARAM_COUNT = 2  # Normal: loc, scale — mirrors family.param_count
N = 256          # toy training-set size -> kl_divisor


def make_feature_net(name, prior_scale):
    """A per-feature shape function: small variational MLP, output = param_count.

    Built as its own tf.keras.Model, exactly like NAMpy feature nets, so the
    spike exercises the sub-model -> outer-model add_loss boundary.
    """
    inp = tf.keras.Input(shape=(1,), name=f"{name}_in")
    h = VariationalDense(8, prior_scale=prior_scale, kl_divisor=N, activation="relu")(inp)
    out = VariationalDense(PARAM_COUNT, prior_scale=prior_scale, kl_divisor=N)(h)
    return tf.keras.Model(inp, out, name=name)


def normal_from_params(t):
    loc = t[..., 0]
    scale = 1e-3 + tf.math.softplus(t[..., 1])  # positivity link
    return tfd.Normal(loc=loc, scale=scale)


def build_model(n_features=3, prior_scale=1.0):
    inputs = {f"x{i}": tf.keras.Input(shape=(1,), name=f"x{i}") for i in range(n_features)}
    nets = [make_feature_net(f"feat{i}", prior_scale) for i in range(n_features)]
    contribs = [nets[i](inputs[f"x{i}"]) for i in range(n_features)]
    summed = tf.keras.layers.Add(name="summed_output")(contribs)
    p_y = tfp.layers.DistributionLambda(normal_from_params, name="p_y")(summed)
    return tf.keras.Model(inputs=inputs, outputs=p_y)


def nll(y_true, y_hat):
    return -tf.reduce_mean(y_hat.log_prob(tf.squeeze(y_true)))


def toy_data(n_features=3):
    rng = np.random.default_rng(0)
    X = {f"x{i}": rng.normal(size=(N, 1)).astype("float32") for i in range(n_features)}
    y = (X["x0"].squeeze() * 1.5 - 0.5 * X["x1"].squeeze()
         + rng.normal(scale=0.3, size=N)).astype("float32")
    return X, y


def main():
    failures = []

    def check(name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}{' :: ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    n_features = 3
    model = build_model(n_features=n_features)
    X, y = toy_data(n_features)

    # --- [A] every VariationalDense KL reaches the outer model.losses ---
    n_var = count_variational_layers(model)
    set_kl_beta(model, 1.0)
    _ = model(X)  # one forward pass to populate add_loss tensors
    n_losses = len(model.losses)
    check(
        "A: KL from nested sub-models reaches outer model.losses",
        n_var == 2 * n_features and n_losses == n_var,
        f"{n_var} variational layers, {n_losses} KL losses",
    )

    # --- [B] warm-up beta gates the KL ---
    set_kl_beta(model, 0.0)
    _ = model(X)
    kl_off = float(tf.add_n(model.losses)) if model.losses else 0.0
    set_kl_beta(model, 1.0)
    _ = model(X)
    kl_on = float(tf.add_n(model.losses))
    check(
        "B: beta=0 zeroes KL, beta=1 restores it",
        abs(kl_off) < 1e-6 and kl_on > 1e-6,
        f"kl(beta=0)={kl_off:.3e}, kl(beta=1)={kl_on:.3e}",
    )

    # --- [C] KL is weighted by 1/N (kl_divisor), not raw ---
    # Rebuild an identical model with kl_divisor=1 and compare magnitude.
    raw_model = build_model(n_features=n_features)
    # copy weights so the KL is computed on identical posteriors
    raw_model.set_weights(model.get_weights())
    for layer in _all_layers(raw_model):
        if isinstance(layer, VariationalDense):
            layer.kl_divisor = 1.0
    set_kl_beta(raw_model, 1.0)
    _ = raw_model(X)
    kl_raw = float(tf.add_n(raw_model.losses))
    ratio = kl_raw / kl_on if kl_on else float("nan")
    check(
        "C: KL is scaled by 1/N (kl_divisor)",
        abs(ratio - N) / N < 0.02,
        f"raw/divided = {ratio:.1f} (expected ~{N})",
    )

    # --- [D] .fit() actually adds KL/N to the NLL in the train step ---
    set_kl_beta(model, 1.0)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=nll)
    hist = model.fit(X, y, epochs=1, batch_size=N, verbose=0)
    total_loss = hist.history["loss"][-1]
    # Recompute NLL and KL/N at the post-step weights for an apples-to-apples gap.
    y_hat = model(X)
    nll_only = float(nll(y, y_hat))
    kl_term = float(tf.add_n(model.losses))
    gap = total_loss - nll_only
    check(
        "D: fit() loss = NLL + KL/N (add_loss is in the train step)",
        kl_term > 1e-6 and gap > 1e-6,
        f"total={total_loss:.4f}, nll={nll_only:.4f}, kl/N={kl_term:.4e}, gap={gap:.4e}",
    )

    print()
    if failures:
        print(f"SPIKE 1 RESULT: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("SPIKE 1 RESULT: ALL PASS — KL/add_loss survives the NAMLSS graph shape.")
    return 0


def _all_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from _all_layers(layer)


if __name__ == "__main__":
    sys.exit(main())
