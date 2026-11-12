"""Black-76 forward pricer, vega, and implied-volatility solver.

Index and equity options are converted to implied variance through the
Black-76 model on the forward price F_T, discounted at a flat short-term
rate (src/config.DISCOUNT_RATE; the paper does not specify a discounting
convention). Custom implementation: no bundled option-pricing library in
the environment reproduces Black-76 with a user-supplied discount factor,
so the formulas (Black, 1976) are implemented directly with scipy for the
normal CDF and the implied-volatility root find.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76_price(forward: float, strike: float, t: float, vol: float,
                   discount: float, is_call: bool) -> float:
    """Black-76 price of a European option on a forward.

    Parameters
    ----------
    forward : forward price F_T.
    strike : strike K.
    t : time to expiry in years.
    vol : volatility (annualized).
    discount : discount factor applied to the undiscounted Black-76 payoff.
    is_call : True for a call, False for a put.
    """
    if t <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return discount * intrinsic
    sqrt_t = np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * vol ** 2 * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if is_call:
        price = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        price = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    return discount * price


def black76_vega(forward: float, strike: float, t: float, vol: float, discount: float) -> float:
    """Black-76 vega (price sensitivity to a unit change in volatility)."""
    if t <= 0 or vol <= 0:
        return 0.0
    sqrt_t = np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * vol ** 2 * t) / (vol * sqrt_t)
    return discount * forward * norm.pdf(d1) * sqrt_t


def implied_vol(price: float, forward: float, strike: float, t: float,
                 discount: float, is_call: bool,
                 lo: float = 1e-4, hi: float = 5.0) -> float:
    """Black-76 implied volatility recovered from an option price via Brent's method.

    Raises ValueError if the price lies outside the no-arbitrage bounds
    implied by vol in (lo, hi), which the caller should treat as a bad quote.
    """
    intrinsic = discount * (max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0))
    if price <= intrinsic:
        raise ValueError("price at or below discounted intrinsic value")

    def objective(vol: float) -> float:
        return black76_price(forward, strike, t, vol, discount, is_call) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0 or f_hi < 0:
        raise ValueError("price outside the Black-76 bracket [lo, hi] for this (F, K, T)")
    return brentq(objective, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200)
