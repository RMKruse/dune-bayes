"""Boundary tests for StudentTFamily (issue 0019 / GitHub #42)."""

import pytest
import torch
import torch.nn.functional as F

from neural_bamlss.families import StudentTFamily


@pytest.fixture
def family():
    return StudentTFamily(validate_args=True)


@pytest.fixture
def params():
    torch.manual_seed(0)
    # (batch=4, param_count=3); cols: raw_loc, raw_scale, raw_df
    return torch.randn(4, 3)


@pytest.fixture
def y(params):
    torch.manual_seed(1)
    return torch.randn(params.shape[0])


class TestStudentTFamilyContract:
    def test_param_count(self):
        assert StudentTFamily.param_count == 3

    def test_call_returns_student_t_distribution(self, family, params):
        dist = family(params)
        assert isinstance(dist, torch.distributions.StudentT)

    def test_loc_identity_link(self, family, params):
        dist = family(params)
        torch.testing.assert_close(dist.loc, params[..., 0])

    def test_scale_softplus_link(self, family, params):
        dist = family(params)
        expected = F.softplus(params[..., 1]) + 1e-6
        torch.testing.assert_close(dist.scale, expected)

    def test_df_softplus_plus_one_link(self, family, params):
        # df > 1 guarantees finite variance
        dist = family(params)
        expected = F.softplus(params[..., 2]) + 1.0
        torch.testing.assert_close(dist.df, expected)
        assert (dist.df > 1.0).all()

    def test_log_prob_matches_torch_reference(self, family, params, y):
        loc = params[..., 0]
        scale = F.softplus(params[..., 1]) + 1e-6
        df = F.softplus(params[..., 2]) + 1.0
        ref_dist = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
        reference = ref_dist.log_prob(y)
        torch.testing.assert_close(family.log_prob(params, y), reference)

    def test_validate_args_rejects_nan_params(self):
        family_strict = StudentTFamily(validate_args=True)
        nan_params = torch.tensor([[float("nan"), float("nan"), float("nan")]])
        with pytest.raises(ValueError):
            family_strict(nan_params).log_prob(torch.tensor([0.0]))
