"""PriorScale handle — three-tier prior variance (ADR-0002, issue 0011 / GitHub #12).

Maps one per-feature prior-variance scalar — the neural analog of one mgcv
smoothing parameter λ (penalty ⇔ Gaussian prior, ADR-0002) — across three tiers:

  fixed:          constant scale; KL contribution = 0.
  empirical_bayes: learned point estimate (REML analog); KL = 0.
  hierarchical:   log-normal variational posterior over the scale, with KL
                  from the chosen hyperprior.

The same scalar doubles as the categorical random-effect variance component
(issue 0012 / GitHub #13). Config is closure-free and serializable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus(-3) ≈ 0.049 — tight initial posterior scale (matches VariationalDense).
_RHO_INIT = -3.0


def _softplus_inv(y: float) -> float:
    """Numerically stable inverse of softplus: log(exp(y) − 1) for y > 0."""
    return math.log(math.expm1(y))


class PriorScale(nn.Module):
    """Three-tier prior variance handle (ADR-0002, issue 0011 / GitHub #12).

    Each feature net carries exactly one PriorScale — the neural analog of mgcv's
    per-feature smoothing parameter.  Config is closure-free and serializable.

    Tiers:
        fixed:          constant buffer, no KL.
        empirical_bayes: learnable point estimate (REML analog), no KL.
        hierarchical:   log-normal q(s) with KL vs. the chosen hyperprior:
                        "half_cauchy" — weakly informative, ARD-like shrinkage
                                        (default; no closed-form KL, uses one
                                        reparameterized MC sample per call).
                        "inverse_gamma" — BAMLSS-faithful IG variance component
                                          (closed-form KL, see derivation below).

    Args:
        mode: ``"fixed"`` | ``"empirical_bayes"`` | ``"hierarchical"``.
        scale: Initial / fixed scale value (positive float). Default 1.0.
        hyperprior: Relevant only for ``"hierarchical"`` mode.
            ``"half_cauchy"`` or ``"inverse_gamma"``. Default ``"half_cauchy"``.
        tau: Half-Cauchy scale τ > 0 (hierarchical + half_cauchy). Default 1.0.
        alpha0: InvGamma shape α₀ > 0 (hierarchical + inverse_gamma). Default 1.0.
        beta0: InvGamma rate β₀ > 0 (hierarchical + inverse_gamma). Default 1.0.
        validate_args: Passed to torch.distributions constructors. False in the
            training hot path; True in test fixtures (numerical rule 6).
    """

    def __init__(
        self,
        mode: str = "fixed",
        scale: float = 1.0,
        hyperprior: str = "half_cauchy",
        tau: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        validate_args: bool = False,
    ) -> None:
        super().__init__()
        _valid_modes = ("fixed", "empirical_bayes", "hierarchical")
        if mode not in _valid_modes:
            raise ValueError(f"unknown mode {mode!r}; choose from {_valid_modes}")
        _valid_hps = ("half_cauchy", "inverse_gamma")
        if hyperprior not in _valid_hps:
            raise ValueError(
                f"unknown hyperprior {hyperprior!r}; choose from {_valid_hps}"
            )

        self.mode = mode
        self.initial_scale = float(scale)
        self.hyperprior = hyperprior
        self.tau = float(tau)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.validate_args = bool(validate_args)

        if mode == "fixed":
            # Non-trainable constant — moves with .to(device), saved in state_dict.
            self.register_buffer("_scale_buf", torch.tensor(float(scale)))

        elif mode == "empirical_bayes":
            # scale = softplus(rho); init rho = softplus_inv(initial_scale) so the
            # starting scale exactly equals the user-specified value.
            self.rho = nn.Parameter(torch.tensor(_softplus_inv(float(scale))))

        else:  # hierarchical
            # q(log s) = N(loc_s, softplus(rho_s)²) — reparameterizable log-normal
            # posterior.  Initialized so the mode of q is near initial_scale.
            self.loc_s = nn.Parameter(torch.tensor(math.log(float(scale))))
            self.rho_s = nn.Parameter(torch.tensor(_RHO_INIT))

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self) -> torch.Tensor:
        """Return the current prior scale as a positive scalar tensor.

        Fixed:         returns the constant buffer value.
        Empirical-Bayes: returns softplus(rho) — a differentiable point estimate.
        Hierarchical:  returns exp(loc_s + sigma_s · ε), ε ~ N(0,1) — a
                       reparameterized sample from the log-normal posterior.
        """
        if self.mode == "fixed":
            return self._scale_buf
        if self.mode == "empirical_bayes":
            return F.softplus(self.rho)
        # hierarchical
        sigma_s = F.softplus(self.rho_s)
        eps = torch.randn((), device=self.loc_s.device, dtype=self.loc_s.dtype)
        return torch.exp(self.loc_s + sigma_s * eps)

    # ── KL ────────────────────────────────────────────────────────────────────

    def kl(self) -> torch.Tensor:
        """KL contribution from the scale's hyperprior.

        Fixed / empirical_bayes:
            Returns zero — no hyperprior, no KL term.

        Hierarchical inverse_gamma:
            Closed-form KL[LogNormal(loc_s, sigma_s²) ‖ sqrt(InvGamma(α₀, β₀))].

            Derivation: p(s) = 2β₀^α₀/Γ(α₀) · s^(−2α₀−1) · exp(−β₀/s²)
            (the distribution of sqrt(Y) when Y ~ InvGamma(α₀, β₀)).
            Using E_q[log s] = loc_s and E_q[1/s²] = exp(−2·loc_s + 2·sigma_s²)
            under q(s) = LogNormal(loc_s, sigma_s²):

              KL = 2α₀·loc_s + β₀·exp(−2·loc_s + 2·sigma_s²)
                   − ½·log(2π·sigma_s²) − ½ − log2
                   − α₀·log(β₀) + lgamma(α₀)

        Hierarchical half_cauchy:
            No closed form for KL[LogNormal ‖ HalfCauchy]; uses one reparameterized
            sample s = exp(loc_s + sigma_s·ε) to estimate log q(s) − log p(s).
            This is the standard reparameterization gradient estimator for the KL.

        Returns:
            Scalar tensor (with autograd).
        """
        if self.mode in ("fixed", "empirical_bayes"):
            return torch.zeros(())

        mu = self.loc_s
        sigma_s = F.softplus(self.rho_s)

        if self.hyperprior == "inverse_gamma":
            # Closed-form — pure float32 arithmetic, no MC noise.
            kl = (
                2.0 * self.alpha0 * mu
                + self.beta0 * torch.exp(-2.0 * mu + 2.0 * sigma_s.pow(2))
                - 0.5 * torch.log(2.0 * math.pi * sigma_s.pow(2))
                - 0.5
                - math.log(2.0)
                - self.alpha0 * math.log(self.beta0)
                + torch.lgamma(
                    torch.tensor(self.alpha0, dtype=mu.dtype, device=mu.device)
                )
            )
            return kl

        # half_cauchy — single reparameterized sample.
        eps = torch.randn((), device=mu.device, dtype=mu.dtype)
        log_s = mu + sigma_s * eps
        s = torch.exp(log_s)

        # log q(s): LogNormal(mu, sigma_s²) density in s-space.
        log_q = (
            -0.5 * torch.log(2.0 * math.pi * sigma_s.pow(2))
            - (log_s - mu).pow(2) / (2.0 * sigma_s.pow(2))
            - log_s  # Jacobian d(log s)/ds = 1/s → extra −log s in the density
        )

        # log p(s): HalfCauchy(0, tau) density.
        log_p = (
            math.log(2.0 / math.pi)
            - math.log(self.tau)
            - torch.log(1.0 + (s / self.tau).pow(2))
        )

        return log_q - log_p

    # ── serialisation ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free config (ints, floats, strings, bools only)."""
        return {
            "mode": self.mode,
            "scale": self.initial_scale,
            "hyperprior": self.hyperprior,
            "tau": self.tau,
            "alpha0": self.alpha0,
            "beta0": self.beta0,
        }

    @classmethod
    def from_config(cls, config: dict) -> "PriorScale":
        return cls(**config)
