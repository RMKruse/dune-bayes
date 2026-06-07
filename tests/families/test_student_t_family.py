"""Boundary tests for StudentTFamily (issue 0019 / GitHub #42)."""

import pytest
import torch
import torch.nn.functional as F

from dune_bayes.families import StudentTFamily


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
        # df > 1 STRICTLY guarantees a defined mean (variance needs df > 2;
        # see df_min). EPS floor: bare softplus underflows to exact 0.0 in
        # float32, and torch's StudentT mean is NaN at df == 1 (GitHub #91).
        dist = family(params)
        expected = F.softplus(params[..., 2]) + 1e-6 + 1.0
        torch.testing.assert_close(dist.df, expected)
        assert (dist.df > 1.0).all()

    def test_log_prob_matches_torch_reference(self, family, params, y):
        loc = params[..., 0]
        scale = F.softplus(params[..., 1]) + 1e-6
        df = F.softplus(params[..., 2]) + 1e-6 + 1.0
        ref_dist = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
        reference = ref_dist.log_prob(y)
        torch.testing.assert_close(family.log_prob(params, y), reference)

    def test_df_min_pins_variance_finite(self, params):
        """df_min=2.0 forces df > 2, so dist.variance is finite for ANY pre-link.

        df > 1 (the default) only guarantees a finite mean; the variance of a
        StudentT is finite iff df > 2 (issue 0091 / GitHub #91).
        """
        family = StudentTFamily(df_min=2.0, validate_args=True)
        dist = family(params)
        expected_df = F.softplus(params[..., 2]) + 1e-6 + 2.0
        torch.testing.assert_close(dist.df, expected_df)
        assert (dist.df > 2.0).all()
        assert torch.isfinite(dist.variance).all()

    def test_df_min_default_preserves_original_link(self, params):
        """Default df_min=1.0 reproduces the pre-#91 link exactly (max|Δ| == 0)."""
        dist_default = StudentTFamily(validate_args=True)(params)
        dist_explicit = StudentTFamily(df_min=1.0, validate_args=True)(params)
        torch.testing.assert_close(dist_default.df, dist_explicit.df, rtol=0, atol=0)

    def test_validate_args_rejects_nan_params(self):
        family_strict = StudentTFamily(validate_args=True)
        nan_params = torch.tensor([[float("nan"), float("nan"), float("nan")]])
        with pytest.raises(ValueError):
            family_strict(nan_params).log_prob(torch.tensor([0.0]))
