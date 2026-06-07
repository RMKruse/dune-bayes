"""Tests for per-parameter quantile coverage — issue 0093 / GitHub #93.

Empirical coverage of central credible bands at nominal 50/80/90/95, per
distribution parameter (PRD 0002 §24). Coverage doctrine (CONTEXT.md): the
function MEASURES and REPORTS — mean-field under-coverage is an expected,
documented property, so nothing here asserts coverage is "correct"; the tests
assert the measurement itself against fixtures whose coverage is known by
construction.

Reference-test archetype (CLAUDE.md): closed-form — deterministic draw grids
make the expected coverage exactly computable, no MC noise.
"""

import pytest
import torch

from dune_bayes.metrics import quantile_coverage

# ── 1: Known-by-construction coverage, per parameter (criterion 3) ────────────


def test_coverage_known_by_construction():
    """Coverage table is exact on a deterministic grid fixture, per parameter.

    Construction: every cell's T draws are the uniform grid 0..T−1, so the
    central level-L band is [(T−1)·(1−L)/2, (T−1)·(1+L)/2] exactly (torch's
    default linear-interpolation quantile on a uniform grid). Parameter 0
    places truth_i at grid fraction i/(n−1) — covered iff that fraction lies
    inside [(1−L)/2, (1+L)/2], so expected coverage is a COUNT the test
    computes independently. Parameter 1 places truth outside the draw range
    entirely → coverage exactly 0 at every level, proving the table really is
    per-parameter. All quantities are exact → equality tolerance is float
    round-off only (atol 1e-12).
    """
    t, n = 1_001, 1_001
    levels = (0.5, 0.8, 0.9, 0.95)
    grid = torch.arange(t, dtype=torch.float64)  # 0 .. T−1
    samples = grid.reshape(t, 1, 1).expand(t, n, 2).clone()

    fractions = torch.arange(n, dtype=torch.float64) / (n - 1)  # i/(n−1)
    truth = torch.stack(
        [
            fractions * (t - 1),  # param 0: sweeps the whole grid
            torch.full((n,), -10.0),  # param 1: below every draw → never covered
        ],
        dim=-1,
    )

    result = quantile_coverage(samples, truth, levels=levels)

    assert result.levels == levels
    assert result.coverage.shape == (2, len(levels))
    for j, level in enumerate(levels):
        lo, hi = (1 - level) / 2, (1 + level) / 2
        expected = ((fractions >= lo) & (fractions <= hi)).double().mean()
        torch.testing.assert_close(
            result.coverage[0, j], expected, rtol=0.0, atol=1e-12
        )
    torch.testing.assert_close(
        result.coverage[1], torch.zeros(len(levels), dtype=torch.float64)
    )


# ── 2: Exchangeable-draws fixture — empirical ≈ nominal (MC archetype) ────────


def test_coverage_near_nominal_for_exchangeable_truth():
    """When truth is one more draw from the sampled posterior, coverage ≈ L.

    Construction: samples and truth all i.i.d. N(0, 1) per (obs, param) cell
    — truth is exchangeable with the draws, so the central level-L interval
    covers it with probability L (up to O(1/T) order-statistic bias ≈ 0.003
    at T = 400). MC tolerance: coverage is a binomial mean over n = 4_000
    cells, SE = √(L(1−L)/n) ≤ 0.0079; atol = 0.04 ≈ 5 SE + the finite-T
    bias, under the fixed seed.
    """
    g = torch.Generator().manual_seed(8)
    t, n, p = 400, 4_000, 2
    samples = torch.randn(t, n, p, generator=g, dtype=torch.float64)
    truth = torch.randn(n, p, generator=g, dtype=torch.float64)

    result = quantile_coverage(samples, truth)

    assert result.levels == (0.5, 0.8, 0.9, 0.95)
    for j, level in enumerate(result.levels):
        for param in range(p):
            assert result.coverage[param, j].item() == pytest.approx(level, abs=0.04), (
                f"param {param}, level {level}: {result.coverage[param, j]:.3f}"
            )
