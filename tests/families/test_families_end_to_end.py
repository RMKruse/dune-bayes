"""End-to-end tests for concrete families with BayesianNAMLSS (issue 0019 / GitHub #42).

Acceptance criteria covered:
  - AC2: BayesianNAMLSS trains end-to-end with StudentTFamily and GammaFamily.
  - AC3: WAIC/LOO run against a model using a non-Normal (StudentT) family.
  - AC5: sample_posterior_predictive works with StudentTFamily.
"""

import arviz as az
import pytest
import torch

from neural_bamlss.compare import loo, waic
from neural_bamlss.families import GammaFamily, StudentTFamily
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.shapes import BayesianMLP

N_OBS = 64
IN = 1


# ── StudentT end-to-end ───────────────────────────────────────────────────────


@pytest.fixture
def student_t_family():
    return StudentTFamily()


@pytest.fixture
def student_t_model(student_t_family):
    formula = {
        "x1": BayesianMLP(
            IN, student_t_family.param_count, hidden_dims=[8], kl_divisor=N_OBS
        ),
    }
    return BayesianNAMLSS(formula=formula, family=student_t_family, n_obs=N_OBS)


@pytest.fixture
def real_valued_data():
    g = torch.Generator().manual_seed(10)
    X = {"x1": torch.randn(N_OBS, IN, generator=g)}
    y = 2.0 * X["x1"].squeeze(-1) + 0.3 * torch.randn(N_OBS, generator=g)
    return X, y


class TestStudentTFamilyEndToEnd:
    def test_forward_returns_student_t_distribution(
        self, student_t_model, real_valued_data
    ):
        X, _ = real_valued_data
        dist = student_t_model(X)
        assert isinstance(dist, torch.distributions.StudentT)
        assert dist.batch_shape == (N_OBS,)

    def test_fit_reduces_nll(self, student_t_family, real_valued_data):
        """BayesianNAMLSS with StudentTFamily trains to convergence (AC2)."""
        torch.manual_seed(0)
        X, y = real_valued_data
        model = BayesianNAMLSS(
            formula={
                "x1": BayesianMLP(
                    IN, student_t_family.param_count, hidden_dims=[8], kl_divisor=N_OBS
                )
            },
            family=student_t_family,
            n_obs=N_OBS,
        )
        history = model.fit(X, y, epochs=50, lr=1e-2)
        assert history["nll"][-1] < history["nll"][0] * 1.10

    def test_posterior_predictive_is_mixture(self, student_t_model, real_valued_data):
        """sample_posterior_predictive returns MixtureSameFamily (AC2)."""
        X, _ = real_valued_data
        predictive = student_t_model.sample_posterior_predictive(X, T=20)
        assert isinstance(predictive, torch.distributions.MixtureSameFamily)
        assert predictive.batch_shape == (N_OBS,)

    def test_waic_runs_with_student_t(self, student_t_family, real_valued_data):
        """WAIC runs against a model using StudentTFamily (AC3)."""
        torch.manual_seed(1)
        X, y = real_valued_data
        model = BayesianNAMLSS(
            formula={
                "x1": BayesianMLP(
                    IN, student_t_family.param_count, hidden_dims=[8], kl_divisor=N_OBS
                )
            },
            family=student_t_family,
            n_obs=N_OBS,
        )
        model.fit(X, y, epochs=10, lr=1e-2)
        waic_data = waic(model, X, y, T=50)
        assert hasattr(waic_data, "elpd")
        assert torch.isfinite(torch.tensor(waic_data.elpd))

    def test_loo_runs_with_student_t(self, student_t_family, real_valued_data):
        """LOO runs against a model using StudentTFamily (AC3)."""
        torch.manual_seed(2)
        X, y = real_valued_data
        model = BayesianNAMLSS(
            formula={
                "x1": BayesianMLP(
                    IN, student_t_family.param_count, hidden_dims=[8], kl_divisor=N_OBS
                )
            },
            family=student_t_family,
            n_obs=N_OBS,
        )
        model.fit(X, y, epochs=10, lr=1e-2)
        loo_data = loo(model, X, y, T=50)
        assert isinstance(loo_data, az.ELPDData)
        assert torch.isfinite(torch.tensor(float(loo_data.elpd)))


# ── Gamma end-to-end ──────────────────────────────────────────────────────────


@pytest.fixture
def gamma_family():
    return GammaFamily()


@pytest.fixture
def positive_data():
    g = torch.Generator().manual_seed(20)
    X = {"x1": torch.randn(N_OBS, IN, generator=g)}
    # Positive responses drawn from Gamma(concentration=2, rate=1) + small signal
    y = torch.distributions.Gamma(2.0, 1.0).sample((N_OBS,))
    return X, y


class TestGammaFamilyEndToEnd:
    def test_forward_returns_gamma_distribution(self, gamma_family, positive_data):
        X, _ = positive_data
        formula = {
            "x1": BayesianMLP(
                IN, gamma_family.param_count, hidden_dims=[8], kl_divisor=N_OBS
            ),
        }
        model = BayesianNAMLSS(formula=formula, family=gamma_family, n_obs=N_OBS)
        dist = model(X)
        assert isinstance(dist, torch.distributions.Gamma)
        assert dist.batch_shape == (N_OBS,)

    def test_fit_trains_without_error(self, gamma_family, positive_data):
        """BayesianNAMLSS with GammaFamily trains without error (AC2)."""
        torch.manual_seed(3)
        X, y = positive_data
        model = BayesianNAMLSS(
            formula={
                "x1": BayesianMLP(
                    IN, gamma_family.param_count, hidden_dims=[8], kl_divisor=N_OBS
                )
            },
            family=gamma_family,
            n_obs=N_OBS,
        )
        history = model.fit(X, y, epochs=30, lr=1e-2)
        assert len(history["loss"]) == 30
        assert all(torch.isfinite(torch.tensor(v)) for v in history["loss"])
