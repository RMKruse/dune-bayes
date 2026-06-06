"""Boundary tests for GammaFamily (issue 0019 / GitHub #42)."""

import pytest
import torch
import torch.nn.functional as F

from neural_bamlss.families import GammaFamily


@pytest.fixture
def family():
    return GammaFamily(validate_args=True)


@pytest.fixture
def params():
    torch.manual_seed(0)
    # (batch=4, param_count=2); cols: raw_concentration, raw_rate
    return torch.randn(4, 2)


@pytest.fixture
def y():
    # Gamma responses must be strictly positive
    torch.manual_seed(2)
    return torch.rand(4) + 0.1


class TestGammaFamilyContract:
    def test_param_count(self):
        assert GammaFamily.param_count == 2

    def test_call_returns_gamma_distribution(self, family, params):
        dist = family(params)
        assert isinstance(dist, torch.distributions.Gamma)

    def test_concentration_softplus_link(self, family, params):
        dist = family(params)
        expected = F.softplus(params[..., 0]) + 1e-6
        torch.testing.assert_close(dist.concentration, expected)

    def test_rate_softplus_link(self, family, params):
        dist = family(params)
        expected = F.softplus(params[..., 1]) + 1e-6
        torch.testing.assert_close(dist.rate, expected)

    def test_log_prob_matches_torch_reference(self, family, params, y):
        concentration = F.softplus(params[..., 0]) + 1e-6
        rate = F.softplus(params[..., 1]) + 1e-6
        ref_dist = torch.distributions.Gamma(concentration=concentration, rate=rate)
        reference = ref_dist.log_prob(y)
        torch.testing.assert_close(family.log_prob(params, y), reference)

    def test_validate_args_rejects_non_positive_y(self):
        family_strict = GammaFamily(validate_args=True)
        params = torch.tensor([[1.0, 1.0]])
        y_neg = torch.tensor([-1.0])
        with pytest.raises(ValueError):
            family_strict.log_prob(params, y_neg)
