"""Unit tests for the two-step CVI QP calibration (src/cvi_calibration.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.cvi_calibration import calibrate_surface, prepare_expiries
from src.arbitrage_checks import validate_surface


def test_prepare_expiries_splits_bid_only_ask_only_and_mid(synthetic_quotes_table) -> None:
    expiries = prepare_expiries(synthetic_quotes_table)
    assert len(expiries) == synthetic_quotes_table["expiry_id"].nunique()
    for e in expiries:
        # Fully two-sided synthetic quotes: every strike goes to the mid bucket.
        assert len(e.k_mid) > 0
        assert np.all(e.v_ask_mid >= e.v_bid_mid)


def test_calibration_runs_and_sorts_expiries_by_maturity(synthetic_quotes_table) -> None:
    result = calibrate_surface(synthetic_quotes_table, n_knots=10)
    maturities = [e.T for e in result.expiries]
    assert maturities == sorted(maturities)
    assert result.n_knots == 10
    assert len(result.params) == len(result.expiries)
    assert result.n_variables == 10 * len(result.expiries)


def test_calibrated_chi_squared_is_finite_and_non_negative(synthetic_quotes_table) -> None:
    result = calibrate_surface(synthetic_quotes_table, n_knots=12)
    assert result.chi_squared_avg >= 0
    assert np.isfinite(result.chi_squared_avg)
    for value in result.chi_squared_per_expiry.values():
        assert value >= 0 or np.isnan(value)


def test_calibration_is_free_of_static_arbitrage_on_synthetic_data(synthetic_quotes_table) -> None:
    result = calibrate_surface(synthetic_quotes_table, n_knots=12)
    report = validate_surface(result)
    assert report.calendar_spread_violations == 0
    assert report.butterfly_violations == 0
    assert report.min_pdf >= -1e-6


def test_edge_knot_convexities_are_zero_by_construction(synthetic_quotes_table) -> None:
    """Linear-extrapolation constraint c(z_0) = c(z_{n-1}) = 0 (paper, Non-
    Arbitrage-Related Constraints): those are structural zeros, not free
    variables, so the fitted slope must be constant beyond the edge knots."""
    result = calibrate_surface(synthetic_quotes_table, n_knots=10)
    basis = result.bases[0]
    z0, zn = basis.knots[0], basis.knots[-1]
    _, Vp, _ = basis.eval_rows(np.array([z0 - 1.0, z0 - 2.0, zn + 1.0, zn + 2.0]))
    slope = Vp @ result.params[0]
    assert slope[0] == pytest.approx(slope[1], abs=1e-8)
    assert slope[2] == pytest.approx(slope[3], abs=1e-8)


def test_more_knots_do_not_increase_chi_squared(synthetic_quotes_table) -> None:
    """More free parameters (within the regularized QP) should fit at least
    as well in the least-squares sense (paper, Chi-Squared vs Knots study)."""
    chi2_few = calibrate_surface(synthetic_quotes_table, n_knots=7).chi_squared_avg
    chi2_many = calibrate_surface(synthetic_quotes_table, n_knots=20).chi_squared_avg
    assert chi2_many <= chi2_few + 1e-6


def test_calibration_rejects_empty_quotes_table(synthetic_quotes_table) -> None:
    with pytest.raises(ValueError):
        calibrate_surface(synthetic_quotes_table.iloc[0:0], n_knots=10)
