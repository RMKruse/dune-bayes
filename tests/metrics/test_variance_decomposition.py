"""Tests for the variance decomposition (disentanglement) — issue 0091 / GitHub #91.

The package's core scientific claim (CONTEXT.md glossary: "Variance
decomposition (disentanglement)"): the law-of-total-variance split of the
posterior predictive into aleatoric = E_θ[Var(y|θ)] and
epistemic = Var_θ[E(y|θ)], computed generically from each posterior draw's
``dist.mean`` / ``dist.variance`` for any registered family.

Reference-test archetypes (CLAUDE.md):
  - Shape:     aleatoric / epistemic / total are (n,).
  - Reference: aleatoric + epistemic == MixtureSameFamily.variance — torch's
               own mixture-moment code path is the independent reference for
               the law of total variance.
"""

import warnings

import pytest
import torch

from dune_bayes.families import GammaFamily, NormalFamily, StudentTFamily
from dune_bayes.metrics import variance_decomposition
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.sampling import draw_predictive
from dune_bayes.shapes import BayesianMLP
from dune_bayes.utils import seed_everything

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32
IN = 1

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def family():
    return NormalFamily(validate_args=True)


@pytest.fixture
def model(family):
    seed_everything(0)
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def data_x():
    g = torch.Generator().manual_seed(0)
    return {"x1": torch.randn(N_OBS, IN, generator=g)}


# ── 1: Shape + law of total variance — tracer bullet ──────────────────────────


def test_decomposition_shapes_and_law_of_total_variance(model, data_x):
    """aleatoric/epistemic/total are (n,); their sum is the predictive variance.

    The independent reference is torch's ``MixtureSameFamily.variance``, which
    computes the same law-of-total-variance split through its own code path.
    Both sides are float32 evaluations of the same quantity on the SAME draws
    (no MC noise), so the tolerance is float32 round-off only.
    """
    seed_everything(1)
    draws = draw_predictive(model, data_x, T=50)
    decomp = variance_decomposition(model, draws.summed_samples)

    assert decomp.aleatoric.shape == (N_OBS,)
    assert decomp.epistemic.shape == (N_OBS,)
    assert decomp.total.shape == (N_OBS,)

    # Law of total variance: total == aleatoric + epistemic by definition...
    torch.testing.assert_close(decomp.total, decomp.aleatoric + decomp.epistemic)
    # ...and it must equal the mixture predictive's variance (the independent
    # reference). rtol covers float32 summation-order differences between the
    # two implementations; no statistical noise enters (identical draws).
    torch.testing.assert_close(
        decomp.total, draws.predictive.variance, rtol=1e-4, atol=1e-6
    )
    # Both components are non-negative variances.
    assert (decomp.aleatoric >= 0).all()
    assert (decomp.epistemic >= 0).all()


# ── 2: Generic over families — no family-specific branches ────────────────────


@pytest.mark.parametrize(
    "any_family",
    [
        NormalFamily(validate_args=True),
        GammaFamily(validate_args=True),
        # df_min=2.0 pins df > 2 so every draw has finite variance (#91);
        # the honest-inf path for df ≤ 2 is tested separately below.
        StudentTFamily(validate_args=True, df_min=2.0),
    ],
    ids=lambda f: type(f).__name__,
)
def test_decomposition_is_generic_over_families(any_family, data_x):
    """One code path serves Normal, Gamma and StudentT (acceptance criterion).

    Same independent reference as the tracer: torch's MixtureSameFamily
    moments on identical draws — float32 round-off tolerance, no MC noise.
    """
    seed_everything(2)
    formula = {
        "x1": BayesianMLP(
            IN, any_family.param_count, hidden_dims=[8], kl_divisor=N_OBS
        ),
    }
    fam_model = BayesianNAMLSS(formula=formula, family=any_family, n_obs=N_OBS)
    draws = draw_predictive(fam_model, data_x, T=50)
    decomp = variance_decomposition(fam_model, draws.summed_samples)

    assert torch.isfinite(decomp.total).all()
    torch.testing.assert_close(
        decomp.total, draws.predictive.variance, rtol=1e-4, atol=1e-6
    )


# ── 3: Honest-inf path — StudentT df ≤ 2 (acceptance criterion) ───────────────


def _student_t_model() -> BayesianNAMLSS:
    """Minimal StudentT model — variance_decomposition only uses its family."""
    family = StudentTFamily(validate_args=True)  # default df_min=1.0 (df > 1)
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


def test_studentt_df_le_2_yields_inf_and_cause_naming_warning():
    """Draws with df ≤ 2 → aleatoric is honestly inf + a counting warning.

    Synthetic coherent draws (T=4, n=2): at observation 0, draws 0–1 land in
    the 1 < df ≤ 2 regime (raw df pre-link −5 → df ≈ 1.007, truly infinite
    variance); draws 2–3 and all of observation 1 use raw df 5 → df ≈ 6
    (finite). So 2 of 4 draws offend, and exactly observation 0's aleatoric
    must be inf — never clamped.
    """
    model = _student_t_model()
    raw = torch.zeros(4, 2, 3)  # (T, n, param_count); loc 0, scale softplus(0)
    raw[..., 2] = 5.0  # df ≈ 6 everywhere → finite variance
    raw[:2, 0, 2] = -5.0  # obs 0, draws 0–1: df ≈ 1.007 ≤ 2 → infinite variance

    with pytest.warns(RuntimeWarning, match=r"2 of 4 .*draws.*StudentTFamily"):
        decomp = variance_decomposition(model, raw)

    assert torch.isposinf(decomp.aleatoric[0])  # inf surfaced, not clamped
    assert torch.isfinite(decomp.aleatoric[1])
    assert torch.isposinf(decomp.total[0])  # inf propagates through the LOTV sum
    assert torch.isfinite(decomp.epistemic).all()  # means stay finite (df > 1)


def test_no_warning_when_all_draws_have_finite_variance():
    """The honest-inf warning must not fire on a fully-finite decomposition."""
    model = _student_t_model()
    raw = torch.zeros(4, 2, 3)
    raw[..., 2] = 5.0  # df ≈ 6 everywhere → finite variance
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        variance_decomposition(model, raw)


# ── 4: Disentanglement — the synthetic two-region construction ────────────────


def test_disentanglement_two_region_synthetic():
    """Epistemic and aleatoric separate where each is known by construction.

    Region A (x ∈ [-2, -1]): 600 observations, noise σ = 1.0 — abundant data
    pins the effects (epistemic small) while the family must absorb the noise
    (aleatoric large, ≈ σ² = 1.0).
    Region B (x ∈ [1, 2]): 20 observations (30× sparser), noise σ = 0.3 — the
    posterior over weights stays wide (epistemic large relative to A) while
    the response is quiet (aleatoric small relative to A).

    Construction notes (tuned empirically across seeds before fixing one):
      - σ_B = 0.3, not 0.1: local epistemic variance scales like σ²/n_local,
        so an ultra-quiet sparse region would pin the function HARDER than
        the noisy dense one (σ_B²/n_B must exceed σ_A²/n_A for the sparse
        region to dominate epistemically: 0.09/20 ≈ 13 × (1.0/600)).
      - No within-B "epistemic > aleatoric" assert: with 20 points the
        σ-head's posterior stays wide, inflating E_θ[σ(x)²] above the true
        0.09 — that upward bias IS sparse-region epistemic honesty, so only
        cross-region orderings are constructed truths here.

    Tolerances are MC-convergence margins, not exact values (CLAUDE.md
    archetype 4): mean-field VI is approximate and T = 200 draws carry MC
    noise, so we assert region-mean ORDERING at ratios ≥ 2.5× — under the
    fixed re-seed protocol (#90) the run is deterministic and the observed
    ratios are ≈ 6× (aleatoric) and ≈ 7× (epistemic), leaving > 2× headroom
    for numerically-benign implementation changes; across 6 probe seeds the
    orderings held at ≥ 1.7× / ≥ 3.3×.
    """
    seed_everything(0)
    n_a, n_b = 600, 20
    n_total = n_a + n_b
    g = torch.Generator().manual_seed(0)
    x_a = torch.rand(n_a, 1, generator=g) - 2.0  # dense:  x ∈ [-2, -1]
    # Sparse but covering: evenly spread so σ(x) is identified throughout B
    # (clumped points would leave the σ-head unconstrained at test points).
    x_b = torch.linspace(1.0, 2.0, n_b).unsqueeze(-1)
    x = torch.cat([x_a, x_b])
    # Smooth signal + region-dependent noise: σ=1.0 in A, σ=0.3 in B.
    sigma = torch.where(x.squeeze(-1) < 0, 1.0, 0.3)
    y = torch.sin(x.squeeze(-1)) + sigma * torch.randn(n_total, generator=g)

    family = NormalFamily()
    model = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(
                IN, family.param_count, hidden_dims=[16], kl_divisor=n_total
            )
        },
        family=family,
        n_obs=n_total,
    )
    # 1500 epochs: the σ-head needs the long tail to learn region B's low
    # noise from only 20 likelihood terms against the KL pull (~0.7 s CPU).
    model.fit({"x1": x}, y, epochs=1500, lr=1e-2)

    # Decompose at region interiors (no boundary/extrapolation effects).
    x_test_a = torch.linspace(-1.8, -1.2, 16).unsqueeze(-1)
    x_test_b = torch.linspace(1.2, 1.8, 16).unsqueeze(-1)
    x_test = {"x1": torch.cat([x_test_a, x_test_b])}
    draws = draw_predictive(model, x_test, T=200)
    decomp = variance_decomposition(model, draws.summed_samples)

    alea_a = decomp.aleatoric[:16].mean()
    alea_b = decomp.aleatoric[16:].mean()
    epi_a = decomp.epistemic[:16].mean()
    epi_b = decomp.epistemic[16:].mean()

    # Aleatoric tracks the noise regions (true ratio 11×; observed ≈ 6×).
    assert alea_a > 2.5 * alea_b, f"aleatoric A={alea_a:.4f} !>> B={alea_b:.4f}"
    # The dense-region aleatoric must be the right ORDER (σ²=1): [0.3, 3] is a
    # generous VI-calibration band — failing it means the split leaked, not
    # that the optimizer was unlucky.
    assert 0.3 < float(alea_a) < 3.0, f"aleatoric A={alea_a:.4f} not ≈ 1.0"
    # Epistemic tracks data density (observed ≈ 7×).
    assert epi_b > 2.5 * epi_a, f"epistemic B={epi_b:.6f} !>> A={epi_a:.6f}"
    # Dense+noisy region: irreducible noise dominates effect uncertainty.
    assert alea_a > epi_a, "dense+noisy region: aleatoric must dominate"
