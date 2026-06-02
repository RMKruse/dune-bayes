"""Minimal in-house variational Dense layer — the spike seed for ADR-0004.

This is intentionally small and self-contained: just enough of the real
`VariationalDense` to verify the two load-bearing TFP claims the design rests on:

  1. KL emitted via `add_loss` inside `call()` propagates from nested per-feature
     sub-models, through a summation, through `tfp.layers.DistributionLambda`, up
     to the outer model's `model.losses` (so the existing KL/N + warm-up machinery
     can work). Verified by `spike_kl_propagation.py`.

  2. A closure-free `get_config` / `from_config` round-trips through Keras
     save/load with the variational weights intact (so save/load is a supported
     path, not a flagged risk). Verified by `spike_serialization.py`.

Posterior: mean-field Normal per weight (loc + softplus(rho) scale).
Prior:     N(0, prior_scale**2), specified by a *serializable float* (no closure).
KL:        closed-form Gaussian-Gaussian, scaled by (beta / kl_divisor) where
           `beta` is the warm-up annealing variable and `kl_divisor` is N
           (training-set size) for the KL/N weighting.

NOT production code. The real layer (ADR-0004) adds the flipout-style
variance-reduction estimator and the hierarchical-scale handle (ADR-0002); those
are out of scope for what this spike needs to prove.
"""

import numpy as np
import tensorflow as tf

# softplus(rho) keeps scale strictly positive; this initial value gives an
# initial posterior scale of ~softplus(-3) ~= 0.049, i.e. a tight-ish start.
_RHO_INIT = -3.0


class VariationalDense(tf.keras.layers.Layer):
    """Dense layer with a mean-field Gaussian weight posterior.

    Args:
        units: output dimension.
        prior_scale: std of the N(0, prior_scale**2) weight prior (a plain float,
            so it serializes — this is the whole point of the layer vs. raw TFP).
        kl_divisor: denominator for the KL term (set to N for KL/N weighting).
        activation: optional Keras activation applied after the affine map.
        use_bias: whether to include a variational bias.
    """

    def __init__(
        self,
        units,
        prior_scale=1.0,
        kl_divisor=1.0,
        activation=None,
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.prior_scale = float(prior_scale)
        self.kl_divisor = float(kl_divisor)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = bool(use_bias)
        # Warm-up annealing factor beta in [0, 1]; non-trainable, driven by a
        # callback. Lives on the layer so the KL added in call() always reflects
        # the current beta.
        self.beta = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="kl_beta")

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.kernel_loc = self.add_weight(
            name="kernel_loc",
            shape=(in_dim, self.units),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )
        self.kernel_rho = self.add_weight(
            name="kernel_rho",
            shape=(in_dim, self.units),
            initializer=tf.keras.initializers.Constant(_RHO_INIT),
            trainable=True,
        )
        if self.use_bias:
            self.bias_loc = self.add_weight(
                name="bias_loc",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
            self.bias_rho = self.add_weight(
                name="bias_rho",
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(_RHO_INIT),
                trainable=True,
            )
        super().build(input_shape)

    @staticmethod
    def _gaussian_kl(loc, scale, prior_scale):
        """Closed-form KL[ N(loc, scale^2) || N(0, prior_scale^2) ], summed."""
        prior_scale = tf.cast(prior_scale, loc.dtype)
        return tf.reduce_sum(
            tf.math.log(prior_scale / scale)
            + (tf.square(scale) + tf.square(loc)) / (2.0 * tf.square(prior_scale))
            - 0.5
        )

    def call(self, inputs, training=None):
        kernel_scale = tf.math.softplus(self.kernel_rho)
        # Reparameterized sample of the kernel.
        kernel = self.kernel_loc + kernel_scale * tf.random.normal(
            tf.shape(self.kernel_loc), dtype=self.kernel_loc.dtype
        )
        outputs = tf.matmul(inputs, kernel)

        kl = self._gaussian_kl(self.kernel_loc, kernel_scale, self.prior_scale)

        if self.use_bias:
            bias_scale = tf.math.softplus(self.bias_rho)
            bias = self.bias_loc + bias_scale * tf.random.normal(
                tf.shape(self.bias_loc), dtype=self.bias_loc.dtype
            )
            outputs = outputs + bias
            kl += self._gaussian_kl(self.bias_loc, bias_scale, self.prior_scale)

        # KL/N, annealed by the warm-up beta. Added inside call() so it tracks
        # both `beta` updates and the current posterior each forward pass.
        self.add_loss(self.beta * kl / self.kl_divisor)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        """Closure-free config — only floats/strings, so save/load round-trips."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "prior_scale": self.prior_scale,
                "kl_divisor": self.kl_divisor,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
            }
        )
        return config


def set_kl_beta(model, beta):
    """Set the warm-up annealing factor on every VariationalDense in a model."""
    for layer in _iter_variational_layers(model):
        layer.beta.assign(float(beta))


def _iter_variational_layers(model):
    for layer in getattr(model, "layers", []):
        if isinstance(layer, VariationalDense):
            yield layer
        # Recurse into nested sub-models / layers-with-layers.
        if hasattr(layer, "layers"):
            yield from _iter_variational_layers(layer)


class KLWarmupCallback(tf.keras.callbacks.Callback):
    """Linearly anneal beta: 0 -> 1 over the first `warmup_epochs` epochs."""

    def __init__(self, warmup_epochs):
        super().__init__()
        self.warmup_epochs = max(int(warmup_epochs), 1)

    def on_epoch_begin(self, epoch, logs=None):
        beta = min(1.0, epoch / self.warmup_epochs)
        set_kl_beta(self.model, beta)


# Custom-objects bundle for save/load (mirrors how the real package will register
# these with the Keras serialization machinery).
CUSTOM_OBJECTS = {"VariationalDense": VariationalDense}


def count_variational_layers(model):
    return sum(1 for _ in _iter_variational_layers(model))


if __name__ == "__main__":
    # Smoke check: a single layer builds, runs, and registers exactly one KL loss.
    x = tf.random.normal((4, 3))
    layer = VariationalDense(2, prior_scale=0.5, kl_divisor=100.0)
    y = layer(x)
    assert y.shape == (4, 2), y.shape
    assert len(layer.losses) == 1, layer.losses
    print("variational_dense smoke check OK:", float(layer.losses[0]))
    print("numpy", np.__version__, "tf", tf.__version__)
