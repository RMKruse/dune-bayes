"""Minimal in-house variational Dense module — the spike seed for ADR-0004.

PyTorch port (ADR-0006). This is intentionally small and self-contained: just
enough of the real `VariationalDense` to verify the two load-bearing claims the
design rests on, now on the PyTorch backend:

  1. KL collected by a module-walk inside `call()`/`forward()` aggregates from
     nested per-feature sub-modules, through a summation, through the family
     distribution head, into a single training loss (so the existing KL/N + warm-up
     machinery can work). Verified by `spike_kl_propagation.py`.

  2. A config dict + `state_dict` round-trips through save/load with the variational
     weights intact (so save/load is a supported path, not a flagged risk).
     Verified by `spike_serialization.py`.

Posterior: mean-field Normal per weight (loc + softplus(rho) scale).
Prior:     N(0, prior_scale**2), specified by a *serializable float* (no closure).
KL:        closed-form Gaussian-Gaussian, scaled by (beta / kl_divisor) where
           `beta` is the warm-up annealing buffer and `kl_divisor` is N
           (training-set size) for the KL/N weighting.

NOT production code. The real module (ADR-0004) adds the flipout-style
variance-reduction estimator and the hierarchical-scale handle (ADR-0002); those
are out of scope for what this spike needs to prove.

Why this is *simpler* than the TF original (ADR-0006): there is no `add_loss`
auto-propagation to rely on — each module stashes its KL on a buffer during
`forward()` and a plain `nn.Module` walk (`collect_kl`) sums them. And there is no
`.keras`/SavedModel/H5 format matrix — save/load is one path: a config dict plus
`state_dict`, with no weight-name-collision failure mode.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus(rho) keeps scale strictly positive; this initial value gives an
# initial posterior scale of ~softplus(-3) ~= 0.049, i.e. a tight-ish start.
_RHO_INIT = -3.0

# Activation registry: keep it tiny and serializable (string <-> callable).
_ACTIVATIONS = {None: None, "linear": None, "relu": F.relu, "tanh": torch.tanh}


class VariationalDense(nn.Module):
    """Dense module with a mean-field Gaussian weight posterior.

    Args:
        in_features: input dimension. (PyTorch builds eagerly, so unlike the Keras
            original the input dim is given at construction, not inferred in build.)
        units: output dimension.
        prior_scale: std of the N(0, prior_scale**2) weight prior (a plain float,
            so it serializes — this is the whole point of the module vs. raw TFP).
        kl_divisor: denominator for the KL term (set to N for KL/N weighting).
        activation: optional activation name applied after the affine map.
        use_bias: whether to include a variational bias.
    """

    def __init__(
        self,
        in_features,
        units,
        prior_scale=1.0,
        kl_divisor=1.0,
        activation=None,
        use_bias=True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.units = int(units)
        self.prior_scale = float(prior_scale)
        self.kl_divisor = float(kl_divisor)
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        self.activation = activation
        self.use_bias = bool(use_bias)

        # Mean-field Normal posterior parameters (loc + softplus(rho) scale).
        self.kernel_loc = nn.Parameter(torch.empty(self.in_features, self.units))
        self.kernel_rho = nn.Parameter(torch.full((self.in_features, self.units), _RHO_INIT))
        nn.init.xavier_normal_(self.kernel_loc)
        if self.use_bias:
            self.bias_loc = nn.Parameter(torch.zeros(self.units))
            self.bias_rho = nn.Parameter(torch.full((self.units,), _RHO_INIT))
        else:
            self.register_parameter("bias_loc", None)
            self.register_parameter("bias_rho", None)

        # Warm-up annealing factor beta in [0, 1]; a non-trainable buffer (moves
        # with .to(device), saved in state_dict, but no gradient). Lives on the
        # module so the KL stashed in forward() always reflects the current beta.
        self.register_buffer("beta", torch.tensor(1.0))
        # Most-recent KL contribution, stashed each forward() for collect_kl().
        # Not a buffer: it is a live autograd tensor, recomputed every pass.
        self.kl = torch.zeros(())

    @staticmethod
    def _gaussian_kl(loc, scale, prior_scale):
        """Closed-form KL[ N(loc, scale^2) || N(0, prior_scale^2) ], summed."""
        return torch.sum(
            math.log(prior_scale) - torch.log(scale)
            + (scale.pow(2) + loc.pow(2)) / (2.0 * prior_scale**2)
            - 0.5
        )

    def forward(self, inputs):
        kernel_scale = F.softplus(self.kernel_rho)
        # Reparameterized sample of the kernel.
        kernel = self.kernel_loc + kernel_scale * torch.randn_like(self.kernel_loc)
        outputs = inputs @ kernel

        kl = self._gaussian_kl(self.kernel_loc, kernel_scale, self.prior_scale)

        if self.use_bias:
            bias_scale = F.softplus(self.bias_rho)
            bias = self.bias_loc + bias_scale * torch.randn_like(self.bias_loc)
            outputs = outputs + bias
            kl = kl + self._gaussian_kl(self.bias_loc, bias_scale, self.prior_scale)

        # KL/N, annealed by the warm-up beta. Stashed on the module (not added to a
        # global loss as in Keras) so collect_kl() can sum it after a forward pass;
        # tracks both `beta` updates and the current posterior each forward pass.
        self.kl = self.beta * kl / self.kl_divisor

        act = _ACTIVATIONS[self.activation]
        if act is not None:
            outputs = act(outputs)
        return outputs

    def get_config(self):
        """Closure-free config — only ints/floats/strings, so save/load round-trips."""
        return {
            "in_features": self.in_features,
            "units": self.units,
            "prior_scale": self.prior_scale,
            "kl_divisor": self.kl_divisor,
            "activation": self.activation,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def set_kl_beta(model, beta):
    """Set the warm-up annealing factor on every VariationalDense in a model."""
    for layer in _iter_variational_layers(model):
        layer.beta.fill_(float(beta))


def collect_kl(model):
    """Sum the most-recent KL contribution of every VariationalDense.

    Call after a forward pass. This is the PyTorch replacement for Keras
    `add_loss` auto-propagation: an explicit module-walk, which trivially crosses
    nested-sub-module and distribution-head boundaries because `model.modules()`
    recurses through the whole tree.
    """
    total = torch.zeros(())
    for layer in _iter_variational_layers(model):
        total = total + layer.kl
    return total


def _iter_variational_layers(model):
    """Yield every VariationalDense in the module tree (recurses sub-modules)."""
    for module in model.modules():
        if isinstance(module, VariationalDense):
            yield module


def count_variational_layers(model):
    return sum(1 for _ in _iter_variational_layers(model))


class KLWarmup:
    """Linearly anneal beta: 0 -> 1 over the first `warmup_epochs` epochs.

    A plain helper (no Keras callback in PyTorch): call `on_epoch_begin(epoch)` at
    the top of each epoch from the training loop.
    """

    def __init__(self, warmup_epochs):
        self.warmup_epochs = max(int(warmup_epochs), 1)

    def on_epoch_begin(self, model, epoch):
        beta = min(1.0, epoch / self.warmup_epochs)
        set_kl_beta(model, beta)


if __name__ == "__main__":
    # Smoke check: a single module builds, runs, and registers exactly one KL.
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    layer = VariationalDense(3, 2, prior_scale=0.5, kl_divisor=100.0)
    y = layer(x)
    assert y.shape == (4, 2), y.shape
    assert layer.kl.item() > 0.0, layer.kl
    print("variational_dense smoke check OK:", float(layer.kl.detach()))
    print("torch", torch.__version__)
