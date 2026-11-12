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


def _total_variance_at_log_moneyness(result: CVIFitResult, j: int, k_grid: np.ndarray) -> np.ndarray:
    """w(k, T_j) = v(z(k), T_j) * T_j at fixed log-forward-moneyness k = log(K/F)."""
    e = result.expiries[j]
    z = e.z(k_grid)
    v = result.evaluate(j, z)
    return v * e.T


def check_calendar_spread(result: CVIFitResult, n_strikes: int = 50) -> tuple[int, float]:
    """Total variance w(k, T) = v(k, T) * T must be non-decreasing in T at
    fixed log-forward-moneyness k = log(K/F) (paper, "No-Calendar-Spread-
    Arbitrage Constraint"). Comparing at fixed k, not at a fixed absolute
    strike K, is what the calibration constraint itself enforces (it compares
    expiry T_j at strike K_i with expiry T_{j+1} at the forward-adjusted
    strike K_i * F_{T_{j+1}} / F_{T_j}, i.e. the same k): forwards differ
    across expiries, so fixed-K would not test the enforced constraint.

    The interior discretized constraint ("Putting It All Together", step 5d)
    is only imposed where the forward-adjusted strike falls inside BOTH
    expiries' knot domain; beyond it only the weaker tail-slope version (step
    5d, tail case) applies. The check below mirrors that scope: it uses the
    log-moneyness range shared by the two expiries' fitted domains, not an
    arbitrary wide band that would partly probe the unconstrained linear
    extrapolation and report spurious violations there.
    """
    violations = 0
    max_decrease = -np.inf
    for j in range(len(result.expiries) - 1):
        e_j, e_j1 = result.expiries[j], result.expiries[j + 1]
        z_max_j = result.bases[j].knots[-1]
        z_max_j1 = result.bases[j + 1].knots[-1]
        k_max_j = z_max_j * e_j.sigma_star * np.sqrt(e_j.T)
        k_max_j1 = z_max_j1 * e_j1.sigma_star * np.sqrt(e_j1.T)
        k_max = 0.95 * min(k_max_j, k_max_j1)  # stay inside both domains
        if k_max <= 0:
            continue
        k_grid = np.linspace(-k_max, k_max, n_strikes)
        w_j = _total_variance_at_log_moneyness(result, j, k_grid)
        w_j1 = _total_variance_at_log_moneyness(result, j + 1, k_grid)
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
    Constraint").

    The linearized butterfly constraint (step 5f/5g) is only imposed on the
    discretized interior strikes and at the two edge knots; it is not imposed
    deep into the linear-extrapolation tail beyond the last knot, where only
    Lee's tail-slope bound (step 5e) applies. Probing an arbitrarily wide
    strike band (e.g. the full +/-50% log-moneyness range) would partly
    evaluate the unconstrained region and report violations the calibration
    never claims to rule out there. The check below stays inside each
    expiry's own fitted knot domain, matching the scope of step 5f/5g.
    """
    violations = 0
    min_pdf = np.inf
    for j in range(len(result.expiries)):
        e = result.expiries[j]
        z_max = result.bases[j].knots[-1]
        k_max = 0.95 * z_max * e.sigma_star * np.sqrt(e.T)
        if k_max <= 0:
            continue
        # Uniform grid in K (not in log-moneyness): the central second
        # difference below assumes a constant step dk.
        strikes = np.linspace(e.F * np.exp(-k_max), e.F * np.exp(k_max), n_strikes)
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
