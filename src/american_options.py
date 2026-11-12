"""De-Americanization of single-name equity option quotes.

The paper notes that American quotes on single-name equities must be
de-Americanized before entering the CVI fit, and suggests "a tree-based or
finite-difference method" without giving further detail (paper,
"Implementation", de-Americanization remark). Custom implementation: no
bundled library in the environment prices American options on a forward with
a user-chosen discount rate, so a standard Cox-Ross-Rubinstein binomial tree
is implemented directly and used both to price American options and, via
root-finding on the volatility, to recover the equivalent European-style
implied volatility (the vol that, together with Black-76, reproduces the
quoted American price through the tree).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def american_binomial_price(forward: float, strike: float, t: float, vol: float,
                             discount: float, is_call: bool, n_steps: int = 300) -> float:
    """CRR binomial price of an American option on a forward/future.

    The forward price is driftless under the pricing measure used here (the
    T-forward measure for an option on a forward contract), so each step has
    up/down factors u = exp(vol*sqrt(dt)), d = 1/u, and the discounted
    risk-neutral probability p = (1 - d) / (u - d); every node is discounted
    at the flat rate implied by `discount` (= exp(-r*t)) and checked against
    the immediate-exercise payoff.
    """
    if t <= 0 or vol <= 0:
        return max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    dt = t / n_steps
    r = -np.log(discount) / t
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    disc_step = np.exp(-r * dt)
    p = (1.0 - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError("binomial probability out of (0, 1); check vol/dt")

    j = np.arange(n_steps + 1)
    prices = forward * u ** (n_steps - j) * d ** j
    values = np.maximum(prices - strike, 0.0) if is_call else np.maximum(strike - prices, 0.0)

    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        prices = forward * u ** (step - j) * d ** j
        continuation = disc_step * (p * values[:step + 1] + (1 - p) * values[1:step + 2])
        intrinsic = np.maximum(prices - strike, 0.0) if is_call else np.maximum(strike - prices, 0.0)
        values = np.maximum(continuation, intrinsic)
    return float(values[0])


def american_implied_vol(price: float, forward: float, strike: float, t: float,
                          discount: float, is_call: bool,
                          lo: float = 1e-4, hi: float = 5.0, n_steps: int = 200) -> float:
    """Equivalent (de-Americanized) implied volatility recovered via Brent's method."""
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    if price <= intrinsic:
        raise ValueError("price at or below intrinsic value")

    def objective(vol: float) -> float:
        return american_binomial_price(forward, strike, t, vol, discount, is_call, n_steps) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0 or f_hi < 0:
        raise ValueError("price outside the binomial-tree bracket [lo, hi] for this (F, K, T)")
    return brentq(objective, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=100)
