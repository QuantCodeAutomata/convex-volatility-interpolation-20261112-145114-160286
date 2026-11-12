"""Unit tests for static-arbitrage diagnostics (src/arbitrage_checks.py)."""

from __future__ import annotations

import numpy as np

from src.arbitrage_checks import check_butterfly, check_calendar_spread, validate_surface
from src.cvi_calibration import calibrate_surface


def test_validate_surface_returns_consistent_report(synthetic_quotes_table) -> None:
    result = calibrate_surface(synthetic_quotes_table, n_knots=12)
    report = validate_surface(result)
    cal_violations, max_decrease = check_calendar_spread(result)
    fly_violations, min_pdf = check_butterfly(result)
    assert report.calendar_spread_violations == cal_violations
    assert report.butterfly_violations == fly_violations
    assert report.min_pdf == min_pdf
    assert report.max_total_variance_decrease == max_decrease


def test_calendar_spread_check_flags_a_decreasing_total_variance(synthetic_quotes_table) -> None:
    """A surface deliberately broken (later expiry's variance halved) must be
    flagged by the calendar-spread check (paper, No-Calendar-Spread
    Constraint: total variance must be non-decreasing in T)."""
    result = calibrate_surface(synthetic_quotes_table, n_knots=10)
    broken = result.params.copy()
    broken[-1] = broken[-1] * 0.1  # shrink the longest expiry's variance sharply
    result.params = broken
    violations, max_decrease = check_calendar_spread(result)
    assert violations > 0
    assert max_decrease > 0


def test_calendar_spread_check_passes_on_fitted_surface(synthetic_quotes_table) -> None:
    result = calibrate_surface(synthetic_quotes_table, n_knots=12)
    violations, max_decrease = check_calendar_spread(result)
    assert violations == 0
    assert max_decrease <= 1e-8


def test_min_pdf_is_finite_number(synthetic_quotes_table) -> None:
    result = calibrate_surface(synthetic_quotes_table, n_knots=12)
    report = validate_surface(result)
    assert np.isfinite(report.min_pdf)
