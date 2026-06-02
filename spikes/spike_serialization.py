"""Spike 2 — does a VariationalDense model round-trip through save/load?

PyTorch port (ADR-0006). ADR-0004 claimed an in-house layer retires the save/load
fragility that sinks raw `tfp.layers.DenseVariational` / `DenseFlipout` (their
prior/posterior closures don't serialize). On TF the claim rested on a closure-free
`get_config` plus the `.keras`/SavedModel format matrix (and legacy H5 *failing* on
a weight-name collision). On PyTorch the claim is far cleaner and has no format
matrix at all: a config dict (only ints/floats/strings — no closures) plus a
`state_dict`, saved together with `torch.save` and reconstructed without re-supplying
the architecture. There is no H5-style weight-name-collision failure mode to dodge.

This script builds the same nested-sub-module + family-head model as spike 1 and
asserts:

  [A] Module-level get_config/from_config rebuilds the module with identical prior
      hyperparameters and no closures.
  [B] A full model save -> load reconstructs from the saved config alone (no
      architecture re-supplied in code), then loads weights.
  [C] Every variational weight (loc/rho for kernel and bias) survives the
      round-trip elementwise, and the reloaded model runs a forward pass with the
      right output shape. (We compare *weights*, not stochastic predictions —
      reparameterization noise is not reproducible across two module objects, so a
      prediction match would be an unreliable test.)

Run:  python spike_serialization.py
Exit code 0 = the round-trip and config checks pass.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

from variational_dense import VariationalDense, set_kl_beta

PARAM_COUNT = 2
N = 128


class FeatureNet(nn.Module):
    def __init__(self, prior_scale=0.7):
        super().__init__()
        self.h = VariationalDense(1, 8, prior_scale=prior_scale, kl_divisor=N, activation="relu")
        self.out = VariationalDense(8, PARAM_COUNT, prior_scale=prior_scale, kl_divisor=N)

    def forward(self, x):
        return self.out(self.h(x))


class SummedModel(nn.Module):
    """Per-feature nets -> sum of raw params (no distribution head, so predictions
    are a plain tensor we could compare numerically). Carries its own config so the
    save bundle can reconstruct it without the architecture being re-supplied."""

    def __init__(self, n_features=2, prior_scale=0.7):
        super().__init__()
        self.n_features = n_features
        self.prior_scale = prior_scale
        self.nets = nn.ModuleList(FeatureNet(prior_scale) for _ in range(n_features))

    def forward(self, X):
        contribs = [self.nets[i](X[f"x{i}"]) for i in range(self.n_features)]
        return torch.stack(contribs, dim=0).sum(dim=0)

    def get_config(self):
        return {"n_features": self.n_features, "prior_scale": self.prior_scale}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def save_model(model, path):
    """The single save path (ADR-0006): config dict + state_dict, one file."""
    torch.save({"config": model.get_config(), "state_dict": model.state_dict()}, path)


def load_model(path):
    """Reconstruct from the saved config alone, then load weights."""
    bundle = torch.load(path, weights_only=True)
    model = SummedModel.from_config(bundle["config"])
    model.load_state_dict(bundle["state_dict"])
    return model


def weights_match(model_a, model_b, atol=1e-6):
    """Elementwise comparison of every parameter + buffer, by name."""
    sa, sb = model_a.state_dict(), model_b.state_dict()
    if sa.keys() != sb.keys():
        return False, f"key mismatch {set(sa) ^ set(sb)}"
    worst = 0.0
    for k in sa:
        a, b = sa[k], sb[k]
        if a.shape != b.shape:
            return False, f"{k} shape {tuple(a.shape)} != {tuple(b.shape)}"
        worst = max(worst, float((a - b).abs().max()) if a.numel() else 0.0)
    return worst <= atol, f"max|Δw|={worst:.2e} over {len(sa)} tensors"


def toy_X(n_features=2):
    g = torch.Generator().manual_seed(7)
    return {f"x{i}": torch.randn(N, 1, generator=g) for i in range(n_features)}


def main():
    torch.manual_seed(0)
    failures = []

    def check(name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}{' :: ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    # --- [A] module config round-trips with no closures ---
    layer = VariationalDense(5, 4, prior_scale=0.33, kl_divisor=99.0, activation="relu")
    cfg = layer.get_config()
    closure_free = all(not callable(v) for v in cfg.values())
    rebuilt = VariationalDense.from_config(cfg)
    check(
        "A: get_config/from_config is closure-free and preserves hyperparams",
        closure_free
        and rebuilt.prior_scale == 0.33
        and rebuilt.kl_divisor == 99.0
        and rebuilt.units == 4
        and rebuilt.in_features == 5,
        f"prior_scale={rebuilt.prior_scale}, kl_divisor={rebuilt.kl_divisor}",
    )

    # --- [B]/[C] full-model save -> load (single torch path) ---
    X = toy_X()
    model = SummedModel()
    set_kl_beta(model, 1.0)
    _ = model(X)  # exercise a forward pass before saving

    tmp = tempfile.mkdtemp(prefix="vd_spike_")
    path = os.path.join(tmp, "m.pt")
    try:
        save_model(model, path)
        loaded = load_model(path)  # [B] reconstructs from saved config alone
        ok_w, detail_w = weights_match(model, loaded)  # [C] weights survive
        out = loaded(X)
        ok_run = tuple(out.shape) == (N, PARAM_COUNT)
        check(
            "B/C: save->load reconstructs from config, weights survive, runs",
            ok_w and ok_run,
            f"{detail_w}; out_shape={tuple(out.shape)}",
        )
    except Exception as exc:  # noqa: BLE001 — report, don't crash
        check("B/C: save->load", False, f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"SPIKE 2 RESULT: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("SPIKE 2 RESULT: ALL PASS — config + state_dict round-trips save/load.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
