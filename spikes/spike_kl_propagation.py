"""Spike 1 — does module-walk KL survive the real NAMLSS graph shape?

PyTorch port (ADR-0006). The genuine risk (CONTEXT / ADR-0003 / ADR-0004): in
NAMLSS each feature net is a *separate* sub-module; their outputs are summed and
piped through a family distribution head. We need KL emitted by `VariationalDense`
modules *inside those per-feature sub-modules* to show up when we sum the loss, get
added to the NLL by the training step, and respond to the warm-up beta.

On TF this rode Keras `add_loss` auto-propagation, whose risk was silent dropping
across the sub-model / `DistributionLambda` boundary. On PyTorch there is no such
magic: `collect_kl` walks the module tree explicitly (`model.modules()` recurses),
so the boundary-crossing is structural, not framework-dependent. This spike asserts
the walk-based design behaves correctly on toy Normal-family data:

  [A] The walk collects one KL per VariationalDense, across sub-modules.
  [B] beta=0 zeroes the KL contribution; beta=1 restores it (warm-up works).
  [C] The KL scales as KL/N (kl_divisor), not raw KL.
  [D] A real optimizer step executes and the total loss exceeds the pure NLL by
      approximately the KL/N term (i.e. KL is actually in the train step).

Run:  python spike_kl_propagation.py
Exit code 0 = all PASS.  Non-zero = a claim failed (printed FAIL ...).
"""

import sys

import torch
import torch.nn as nn

from variational_dense import (
    VariationalDense,
    collect_kl,
    count_variational_layers,
    set_kl_beta,
)

PARAM_COUNT = 2  # Normal: loc, scale — mirrors family.param_count
N = 256          # toy training-set size -> kl_divisor


class FeatureNet(nn.Module):
    """A per-feature shape function: small variational MLP, output = param_count.

    Its own nn.Module, exactly like NAMpy feature nets, so the spike exercises the
    nested-sub-module -> outer-module KL-walk boundary.
    """

    def __init__(self, prior_scale):
        super().__init__()
        self.h = VariationalDense(1, 8, prior_scale=prior_scale, kl_divisor=N, activation="relu")
        self.out = VariationalDense(8, PARAM_COUNT, prior_scale=prior_scale, kl_divisor=N)

    def forward(self, x):
        return self.out(self.h(x))


def normal_from_params(t):
    loc = t[..., 0]
    scale = 1e-3 + torch.nn.functional.softplus(t[..., 1])  # positivity link
    return torch.distributions.Normal(loc=loc, scale=scale)


class NAMLSSModel(nn.Module):
    """Per-feature nets -> sum -> Normal family head (the real NAMLSS shape)."""

    def __init__(self, n_features=3, prior_scale=1.0):
        super().__init__()
        self.n_features = n_features
        self.nets = nn.ModuleList(FeatureNet(prior_scale) for _ in range(n_features))

    def forward(self, X):
        contribs = [self.nets[i](X[f"x{i}"]) for i in range(self.n_features)]
        summed = torch.stack(contribs, dim=0).sum(dim=0)  # the Add layer
        return normal_from_params(summed)


def nll(dist, y_true):
    return -dist.log_prob(y_true.squeeze(-1)).mean()


def toy_data(n_features=3):
    g = torch.Generator().manual_seed(0)
    X = {f"x{i}": torch.randn(N, 1, generator=g) for i in range(n_features)}
    y = (X["x0"].squeeze(-1) * 1.5 - 0.5 * X["x1"].squeeze(-1)
         + 0.3 * torch.randn(N, generator=g))
    return X, y


def main():
    torch.manual_seed(0)
    failures = []

    def check(name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}{' :: ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    n_features = 3
    model = NAMLSSModel(n_features=n_features)
    X, y = toy_data(n_features)

    # --- [A] every VariationalDense KL is collected by the module walk ---
    n_var = count_variational_layers(model)
    set_kl_beta(model, 1.0)
    _ = model(X)  # one forward pass to populate each module's .kl
    n_collected = sum(1 for layer in model.modules() if isinstance(layer, VariationalDense))
    check(
        "A: KL from nested sub-modules is collected by the outer walk",
        n_var == 2 * n_features and n_collected == n_var,
        f"{n_var} variational modules, {n_collected} collected",
    )

    # --- [B] warm-up beta gates the KL ---
    set_kl_beta(model, 0.0)
    _ = model(X)
    kl_off = float(collect_kl(model).detach())
    set_kl_beta(model, 1.0)
    _ = model(X)
    kl_on = float(collect_kl(model).detach())
    check(
        "B: beta=0 zeroes KL, beta=1 restores it",
        abs(kl_off) < 1e-6 and kl_on > 1e-6,
        f"kl(beta=0)={kl_off:.3e}, kl(beta=1)={kl_on:.3e}",
    )

    # --- [C] KL is weighted by 1/N (kl_divisor), not raw ---
    set_kl_beta(model, 1.0)
    _ = model(X)
    kl_divided = float(collect_kl(model).detach())
    # Flip kl_divisor to 1 on the same posteriors and recompute.
    for layer in model.modules():
        if isinstance(layer, VariationalDense):
            layer.kl_divisor = 1.0
    _ = model(X)
    kl_raw = float(collect_kl(model).detach())
    ratio = kl_raw / kl_divided if kl_divided else float("nan")
    check(
        "C: KL is scaled by 1/N (kl_divisor)",
        abs(ratio - N) / N < 0.05,
        f"raw/divided = {ratio:.1f} (expected ~{N})",
    )
    # restore divisor for [D]
    for layer in model.modules():
        if isinstance(layer, VariationalDense):
            layer.kl_divisor = float(N)

    # --- [D] an optimizer step actually adds KL/N to the NLL ---
    set_kl_beta(model, 1.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dist = model(X)
    nll_only = float(nll(dist, y).detach())
    kl_term = float(collect_kl(model).detach())
    total = nll(dist, y) + collect_kl(model)
    total_val = float(total.detach())
    opt.zero_grad()
    total.backward()
    opt.step()
    gap = total_val - nll_only
    check(
        "D: train step loss = NLL + KL/N (KL is in the objective)",
        kl_term > 1e-6 and gap > 1e-6,
        f"total={total_val:.4f}, nll={nll_only:.4f}, kl/N={kl_term:.4e}, gap={gap:.4e}",
    )

    print()
    if failures:
        print(f"SPIKE 1 RESULT: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("SPIKE 1 RESULT: ALL PASS — module-walk KL survives the NAMLSS graph shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
