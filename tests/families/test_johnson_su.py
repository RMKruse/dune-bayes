"""Boundary tests for the JohnsonSU distribution + family (issue #94).

Reference: ``scipy.stats.johnsonsu`` — the distribution is pinned to scipy's
parameterization ``z = γ + δ·arcsinh((y − ξ)/λ)`` with (γ, δ, ξ, λ) mapping
to scipy's (a, b, loc, scale), so scipy is a zero-translation reference
(closed-form reference-test archetype, CLAUDE.md).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from scipy import stats

from dune_bayes.families import JohnsonSUFamily
from dune_bayes.families.johnson_su import JohnsonSU


class TestJohnsonSULogProb:
    def test_log_prob_matches_scipy_at_moderate_params(self):
        """log_prob == scipy johnsonsu.logpdf — same params, zero translation.

        float64 + tight rtol: both sides are closed-form evaluations of the
        same density, so agreement is to float error, not MC noise.
        """
        skew = torch.tensor([0.0, -1.5, 2.0], dtype=torch.float64)
        tail = torch.tensor([1.0, 0.8, 2.5], dtype=torch.float64)
        loc = torch.tensor([0.0, 3.0, -2.0], dtype=torch.float64)
        scale = torch.tensor([1.0, 0.5, 4.0], dtype=torch.float64)
        y = torch.tensor([0.3, 2.0, -5.0], dtype=torch.float64)

        dist = JohnsonSU(skew, tail, loc, scale, validate_args=True)
        result = dist.log_prob(y)

        expected = stats.johnsonsu.logpdf(
            y.numpy(),
            a=skew.numpy(),
            b=tail.numpy(),
            loc=loc.numpy(),
            scale=scale.numpy(),
        )
        torch.testing.assert_close(
            result, torch.from_numpy(expected), rtol=1e-12, atol=1e-12
        )

    def test_log_prob_matches_scipy_in_tails_and_near_boundary(self):
        """log_prob == scipy across deep tails and near-boundary δ, λ (#94 AC1).

        Cartesian grid: δ down to the EPS link floor (1e-6) and up to 100,
        λ down to 1e-6, responses out to ±1e3 standard deviations from loc.
        atol 1e-9: pure float64 closed-form on both sides; the residual is
        accumulation order in asinh/log1p, far below 1e-9 in practice.
        """
        skew_v = [-5.0, 0.0, 5.0]
        tail_v = [1e-6, 0.05, 1.0, 100.0]
        scale_v = [1e-6, 1.0, 50.0]
        y_v = [-1e3, -1.0, 0.0, 1.0, 1e3]

        grid = torch.cartesian_prod(
            *(
                torch.tensor(v, dtype=torch.float64)
                for v in (skew_v, tail_v, scale_v, y_v)
            )
        )
        skew, tail, scale, y = grid.unbind(-1)
        loc = torch.zeros_like(skew)

        dist = JohnsonSU(skew, tail, loc, scale, validate_args=True)
        result = dist.log_prob(y)

        expected = torch.from_numpy(
            stats.johnsonsu.logpdf(
                y.numpy(),
                a=skew.numpy(),
                b=tail.numpy(),
                loc=loc.numpy(),
                scale=scale.numpy(),
            )
        )
        assert torch.isfinite(result).all()
        # scipy 1.17's johnsonsu has no log-space _logpdf — it computes
        # log(pdf), which underflows to -inf once the density drops below
        # float64 min (log ≈ −745). Our log-space form (numerical rule 2)
        # stays finite there, so scipy is only a reference where ITS value
        # is finite; at its -inf rows we assert ours sits below the
        # underflow threshold, i.e. the disagreement is scipy's underflow,
        # not a density mismatch.
        scipy_ok = torch.isfinite(expected)
        torch.testing.assert_close(
            result[scipy_ok], expected[scipy_ok], rtol=1e-9, atol=1e-9
        )
        log_float64_min = math.log(torch.finfo(torch.float64).tiny)  # ≈ −708
        assert (result[~scipy_ok] < log_float64_min).all()


class TestJohnsonSUMoments:
    def test_mean_and_variance_match_scipy(self):
        """Closed-form mean/variance == scipy johnsonsu.stats (#94 AC2).

        Covers zero / negative / positive skew and δ from the
        moment-explosion edge (0.3 → exp(δ⁻²) ≈ 6.7e4) to heavy floor
        (δ = 10). rtol 1e-10: both sides closed-form in float64; the residual
        is expm1/exp accumulation, orders below 1e-10.
        """
        skew = torch.tensor([0.0, -1.5, 2.0, 0.7], dtype=torch.float64)
        tail = torch.tensor([1.0, 0.3, 2.5, 10.0], dtype=torch.float64)
        loc = torch.tensor([0.0, 3.0, -2.0, 100.0], dtype=torch.float64)
        scale = torch.tensor([1.0, 0.5, 4.0, 0.01], dtype=torch.float64)

        dist = JohnsonSU(skew, tail, loc, scale, validate_args=True)

        mean_ref, var_ref = stats.johnsonsu.stats(
            a=skew.numpy(),
            b=tail.numpy(),
            loc=loc.numpy(),
            scale=scale.numpy(),
            moments="mv",
        )
        torch.testing.assert_close(
            dist.mean, torch.from_numpy(mean_ref), rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            dist.variance, torch.from_numpy(var_ref), rtol=1e-10, atol=1e-10
        )

    def test_moments_overflow_to_inf_never_nan_in_float32(self):
        """Small-δ / large-skew overflow is honest ±inf, never NaN (#94 note).

        The exp(δ⁻²) term explodes for small δ (documented validity note):
        at δ = EPS the true moments are ~e^(5e11) — finite mathematically,
        unrepresentable in any float. The naive mean form exp(δ⁻²/2)·sinh(γ/δ)
        gives inf·0 = NaN at γ = 0; the implementation must yield 0 there
        (sinh(0) = 0 exactly) and ±inf at γ ≠ 0, never NaN — NaN would poison
        the variance decomposition (#91 gate).
        """
        eps = 1e-6  # the family link's EPS floor for δ, λ
        skew = torch.tensor([0.0, 3.0, -3.0, 1e4], dtype=torch.float32)
        tail = torch.full((4,), eps, dtype=torch.float32)
        loc = torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        scale = torch.ones(4, dtype=torch.float32)

        dist = JohnsonSU(skew, tail, loc, scale, validate_args=True)

        mean, variance = dist.mean, dist.variance
        assert not torch.isnan(mean).any()
        assert not torch.isnan(variance).any()
        # γ = 0: sinh(0) = 0 kills the exploding factor exactly → mean = loc.
        torch.testing.assert_close(mean[0], torch.tensor(5.0))
        # γ ≠ 0: overflow surfaces as honest signed infinity (sign = −sign γ).
        assert torch.isneginf(mean[1]) and torch.isposinf(mean[2])
        assert torch.isneginf(mean[3])
        # Variance overflows to +inf — never negative, never -inf.
        assert torch.isposinf(variance).all()


class TestJohnsonSURSample:
    # Moderate params: δ = 2 keeps exp(δ⁻²) mild so analytic moments are
    # well-conditioned and sample moments converge at the usual √N rate.
    SKEW, TAIL, LOC, SCALE = 1.0, 2.0, -1.0, 1.5

    def _dist(self) -> JohnsonSU:
        t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
        return JohnsonSU(
            t(self.SKEW),
            t(self.TAIL),
            t(self.LOC),
            t(self.SCALE),
            validate_args=True,
        )

    def test_rsample_moments_converge_to_analytic(self):
        """Sample mean/variance → closed-form mean/variance (#94 AC3).

        MC-convergence archetype: N = 400_000 draws under a fixed seed.
        SE(mean) = σ/√N ≈ 0.0015, so atol 0.01 on the mean is > 6 SE;
        rel 0.02 on the variance is ~6 SE of the sample variance (kurtosis
        of JSU at δ = 2 is mild, ≈ 4). MC noise, not float error.
        """
        torch.manual_seed(94)
        dist = self._dist()
        draws = dist.rsample((400_000,))

        assert draws.shape == (400_000,)
        torch.testing.assert_close(draws.mean(), dist.mean, rtol=0.0, atol=0.01)
        torch.testing.assert_close(draws.var(), dist.variance, rtol=0.02, atol=0.0)

    def test_rsample_is_reparameterized_and_differentiable(self):
        """Gradients flow through rsample to all four parameters (#94 AC3).

        The draw is an exact inverse transform of a Normal draw, so it is
        differentiable in (γ, δ, ξ, λ); has_rsample must advertise it.
        """
        assert JohnsonSU.has_rsample
        params = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in (self.SKEW, self.TAIL, self.LOC, self.SCALE)
        ]
        dist = JohnsonSU(*params, validate_args=True)
        torch.manual_seed(7)
        dist.rsample((64,)).sum().backward()
        for p in params:
            assert p.grad is not None
            assert torch.isfinite(p.grad)
            assert p.grad != 0.0


class TestJohnsonSUCdf:
    def test_cdf_matches_scipy(self):
        """cdf == scipy johnsonsu.cdf — closed form Φ(z) on both sides.

        Off-center y and asymmetric params exercise each of the four
        parameter mappings. rtol 1e-12: both sides are erf evaluations of
        the same z-score, float64 closed form (no MC noise).
        """
        skew = torch.tensor([0.0, -1.5, 2.0], dtype=torch.float64)
        tail = torch.tensor([1.0, 0.8, 2.5], dtype=torch.float64)
        loc = torch.tensor([0.0, 3.0, -2.0], dtype=torch.float64)
        scale = torch.tensor([1.0, 0.5, 4.0], dtype=torch.float64)
        y = torch.tensor([0.3, 2.0, -5.0], dtype=torch.float64)

        dist = JohnsonSU(skew, tail, loc, scale, validate_args=True)
        result = dist.cdf(y)

        expected = stats.johnsonsu.cdf(
            y.numpy(),
            a=skew.numpy(),
            b=tail.numpy(),
            loc=loc.numpy(),
            scale=scale.numpy(),
        )
        torch.testing.assert_close(
            result, torch.from_numpy(expected), rtol=1e-12, atol=1e-12
        )


class TestJohnsonSUFamilyContract:
    """Boundary tests for JohnsonSUFamily (issue #94, family side).

    Column order is the scipy order (γ, δ, ξ, λ); links identity /
    softplus+EPS / identity / softplus+EPS per the issue.
    """

    @pytest.fixture
    def family(self):
        return JohnsonSUFamily(validate_args=True)

    @pytest.fixture
    def params(self):
        torch.manual_seed(0)
        # (batch=5, param_count=4); cols: raw γ, raw δ, raw ξ, raw λ
        return torch.randn(5, 4)

    def test_param_count(self):
        assert JohnsonSUFamily.param_count == 4

    def test_call_returns_johnson_su_distribution(self, family, params):
        dist = family(params)
        assert isinstance(dist, JohnsonSU)
        assert dist.batch_shape == (5,)

    def test_links_match_issue_spec(self, family, params):
        """identity(γ) / softplus+EPS(δ) / identity(ξ) / softplus+EPS(λ)."""
        dist = family(params)
        eps = 1e-6  # EPS for float32, utils.EPS
        torch.testing.assert_close(dist.skew, params[..., 0])
        torch.testing.assert_close(dist.tail, F.softplus(params[..., 1]) + eps)
        torch.testing.assert_close(dist.loc, params[..., 2])
        torch.testing.assert_close(dist.scale, F.softplus(params[..., 3]) + eps)

    def test_family_log_prob_matches_scipy_through_links(self, family, params):
        """family.log_prob == scipy logpdf of the linked parameters.

        End-to-end through the links — this is the value WAIC/LOO will
        accumulate. atol 1e-5: float32 forward vs scipy float64 reference.
        """
        torch.manual_seed(1)
        y = torch.randn(5)
        dist = family(params)
        expected = stats.johnsonsu.logpdf(
            y.numpy(),
            a=dist.skew.numpy(),
            b=dist.tail.numpy(),
            loc=dist.loc.numpy(),
            scale=dist.scale.numpy(),
        )
        torch.testing.assert_close(
            family.log_prob(params, y).to(torch.float64),
            torch.from_numpy(expected),
            rtol=0.0,
            atol=1e-5,
        )

    def test_cdf_is_float64_and_matches_scipy(self, family, params):
        """BaseFamily.cdf contract: float64 out, scipy-checked (PIT feed)."""
        torch.manual_seed(2)
        y = torch.randn(5)
        result = family.cdf(params, y)
        dist = family(params)
        expected = stats.johnsonsu.cdf(
            y.numpy(),
            a=dist.skew.numpy(),
            b=dist.tail.numpy(),
            loc=dist.loc.numpy(),
            scale=dist.scale.numpy(),
        )
        assert result.dtype == torch.float64
        # float32 z-score against scipy's float64: erf compresses input
        # error, 1e-6 absolute headroom is ample.
        torch.testing.assert_close(
            result, torch.from_numpy(expected), rtol=0.0, atol=1e-6
        )


class TestJohnsonSUParameterRecovery:
    def test_mle_recovers_all_four_parameters(self):
        """MLE through family.log_prob recovers (γ, δ, ξ, λ) (#94 AC5).

        Simulate n = 4000 JSU draws at known parameters, fit the four raw
        pre-link values by Adam on the mean negative log-likelihood — the
        exact gradient path BayesianNAMLSS trains through. Tolerances are
        generous (atol 0.2): they cover MLE sampling error at n = 4000
        (γ and δ are strongly correlated in the JSU Fisher information, so
        their SEs are the widest) plus residual optimizer error — this is a
        recovery test, not a precision benchmark.
        """
        truth = {"skew": -1.0, "tail": 1.5, "loc": 2.0, "scale": 0.8}
        g = torch.manual_seed(940)
        y = JohnsonSU(
            torch.tensor(truth["skew"]),
            torch.tensor(truth["tail"]),
            torch.tensor(truth["loc"]),
            torch.tensor(truth["scale"]),
        ).sample((4000,))
        del g

        family = JohnsonSUFamily()
        raw = torch.zeros(4, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=0.05)
        for _ in range(800):
            optimizer.zero_grad()
            nll = -family.log_prob(raw.expand(y.shape[0], 4), y).mean()
            nll.backward()
            optimizer.step()

        fitted = family(raw.detach().unsqueeze(0))
        recovered = {
            "skew": fitted.skew.item(),
            "tail": fitted.tail.item(),
            "loc": fitted.loc.item(),
            "scale": fitted.scale.item(),
        }
        for name, true_value in truth.items():
            assert recovered[name] == pytest.approx(true_value, abs=0.2), (
                f"{name}: recovered {recovered[name]:.3f}, true {true_value}"
            )
