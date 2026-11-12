"""Static-arbitrage diagnostics for a fitted CVI surface.

Paper, "Putting It All Together", step 10: "Validate: check no calendar
spread arbitrage, no butterfly arbitrage, positive PDF." These checks
re-evaluate the fitted surface on a dense strike grid, independently of the
constraints used during calibration, so that a solver tolerance issue would
still show up here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.cvi_calibration import CVIFitResult


@dataclass
class ArbitrageReport:
    calendar_spread_violations: int
    butterfly_violations: int
    min_pdf: float
    max_total_variance_decrease: float  # should be <= 0 (non-increasing is a violation)


def _total_variance_on_common_strikes(result: CVIFitResult, j: int, strikes: np.ndarray) -> np.ndarray:
    e = result.expiries[j]
    z = e.z(np.log(strikes / e.F))
    v = result.evaluate(j, z)
    return v * e.T


def check_calendar_spread(result: CVIFitResult, n_strikes: int = 50) -> tuple[int, float]:
    """Total variance w(K, T) = v(K, T) * T must be non-decreasing in T at
    fixed strike K (paper, "No-Calendar-Spread-Arbitrage Constraint")."""
    violations = 0
    max_decrease = -np.inf
    for j in range(len(result.expiries) - 1):
        e_j = result.expiries[j]
        strikes = e_j.F * np.exp(np.linspace(-0.3, 0.3, n_strikes))
        w_j = _total_variance_on_common_strikes(result, j, strikes)
        w_j1 = _total_variance_on_common_strikes(result, j + 1, strikes)
        decrease = w_j - w_j1
        violations += int(np.sum(decrease > 1e-8))
        max_decrease = max(max_decrease, float(np.max(decrease)))
    return violations, (max_decrease if np.isfinite(max_decrease) else 0.0)


def _implied_pdf(result: CVIFitResult, j: int, strikes: np.ndarray) -> np.ndarray:
    """Breeden-Litzenberger density from the fitted Black-76 call prices,
    via a centered finite difference of d^2 C / dK^2."""
    from src.black_scholes import black76_price

    e = result.expiries[j]
    dk = strikes[1] - strikes[0]
    prices = np.empty_like(strikes)
    for i, k in enumerate(strikes):
        z = e.z(np.log(k / e.F))
        v = max(float(result.evaluate(j, np.array([z]))[0]), 1e-12)
        vol = np.sqrt(v)
        prices[i] = black76_price(e.F, k, e.T, vol, discount=1.0, is_call=True)
    second_derivative = (prices[2:] - 2 * prices[1:-1] + prices[:-2]) / dk ** 2
    return second_derivative


def check_butterfly(result: CVIFitResult, n_strikes: int = 200) -> tuple[int, float]:
    """The risk-neutral density (second strike-derivative of the call price)
    must be non-negative everywhere (paper, "No-Butterfly-Arbitrage
    Constraint")."""
    violations = 0
    min_pdf = np.inf
    for j in range(len(result.expiries)):
        e = result.expiries[j]
        # Uniform grid in K (not in log-moneyness): the central second
        # difference below assumes a constant step dk.
        strikes = np.linspace(e.F * np.exp(-0.5), e.F * np.exp(0.5), n_strikes)
        pdf = _implied_pdf(result, j, strikes)
        violations += int(np.sum(pdf < -1e-6))
        min_pdf = min(min_pdf, float(np.min(pdf)))
    return violations, min_pdf


def validate_surface(result: CVIFitResult) -> ArbitrageReport:
    """Run every static-arbitrage check of paper step 10 on a fitted surface."""
    cal_violations, max_decrease = check_calendar_spread(result)
    fly_violations, min_pdf = check_butterfly(result)
    return ArbitrageReport(
        calendar_spread_violations=cal_violations,
        butterfly_violations=fly_violations,
        min_pdf=min_pdf,
        max_total_variance_decrease=max_decrease,
    )
