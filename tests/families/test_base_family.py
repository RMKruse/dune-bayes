"""Boundary tests for BaseFamily contract (issue 0018 / GitHub #38)."""

import pytest
import torch
import torch.nn.functional as F

from neural_bamlss.families import BaseFamily, NormalFamily


@pytest.fixture
def family():
    return NormalFamily(validate_args=True)


@pytest.fixture
def params():
    torch.manual_seed(0)
    # (batch=4, param_count=2); col 1 is the raw scale before softplus
    return torch.randn(4, 2)


@pytest.fixture
def y(params):
    torch.manual_seed(1)
    return torch.randn(params.shape[0])


class TestNormalFamilyIsBaseFamily:
    """NormalFamily satisfies the BaseFamily contract."""

    def test_normal_family_is_base_family(self):
        family = NormalFamily()
        assert isinstance(family, BaseFamily)

    def test_param_count(self, family):
        assert family.param_count == 2

    def test_call_returns_distribution(self, family, params):
        dist = family(params)
        assert isinstance(dist, torch.distributions.Distribution)

    def test_identity_link_for_loc(self, family, params):
        dist = family(params)
        torch.testing.assert_close(dist.loc, params[..., 0])

    def test_softplus_link_for_scale(self, family, params):
        dist = family(params)
        expected = F.softplus(params[..., 1]) + 1e-6
        torch.testing.assert_close(dist.scale, expected)

    def test_log_prob_matches_reference(self, family, params, y):
        # Reference: compute directly from NormalFamily internals to avoid circular.
        loc = params[..., 0]
        scale = F.softplus(params[..., 1]) + 1e-6
        reference = torch.distributions.Normal(loc, scale).log_prob(y)
        torch.testing.assert_close(family.log_prob(params, y), reference)

    def test_validate_args_rejects_invalid_scale(self):
        # validate_args=True must catch NaN distribution parameters.
        family_strict = NormalFamily(validate_args=True)
        nan_params = torch.tensor([[float("nan"), float("nan")]])
        with pytest.raises(Exception):
            family_strict(nan_params).log_prob(torch.tensor([0.0]))


class TestBaseFamilyContractEnforcement:
    """__init_subclass__ enforces param_count at class-definition time."""

    def test_subclass_without_param_count_raises(self):
        with pytest.raises(TypeError, match="param_count"):

            class BadFamily(BaseFamily):
                def __call__(self, params):
                    return torch.distributions.Normal(params[..., 0], params[..., 1])

                def log_prob(self, params, y):
                    return self(params).log_prob(y)

    def test_concrete_subclass_with_param_count_ok(self):
        class GoodFamily(BaseFamily):
            param_count = 1

            def __call__(self, params):
                ones = torch.ones_like(params[..., 0])
                return torch.distributions.Normal(params[..., 0], ones)

            def log_prob(self, params, y):
                return self(params).log_prob(y)

        # Must instantiate without error.
        gf = GoodFamily()
        assert gf.param_count == 1
