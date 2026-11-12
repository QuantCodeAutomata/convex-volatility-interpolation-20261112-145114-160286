"""Unit tests for de-Americanization (src/american_options.py)."""

from __future__ import annotations

import pytest

from src.american_options import american_binomial_price, american_implied_vol
from src.black_scholes import black76_price


def test_american_call_on_non_dividend_forward_equals_european() -> None:
    """With no early-exercise premium source here (a forward, not a spot with
    dividends), an American call price should match its European counterpart."""
    forward, strike, t, vol, discount = 100.0, 95.0, 0.5, 0.25, 0.98
    american = american_binomial_price(forward, strike, t, vol, discount, is_call=True, n_steps=600)
    european = black76_price(forward, strike, t, vol, discount, is_call=True)
    assert american == pytest.approx(european, rel=1.5e-2)


def test_american_put_price_exceeds_or_equals_intrinsic() -> None:
    forward, strike, t, vol, discount = 100.0, 120.0, 1.0, 0.3, 0.95
    price = american_binomial_price(forward, strike, t, vol, discount, is_call=False)
    assert price >= max(strike - forward, 0.0)


def test_american_implied_vol_round_trips_binomial_price() -> None:
    forward, strike, t, vol, discount = 100.0, 90.0, 0.4, 0.35, 0.97
    price = american_binomial_price(forward, strike, t, vol, discount, is_call=False, n_steps=250)
    recovered = american_implied_vol(price, forward, strike, t, discount, is_call=False, n_steps=250)
    assert recovered == pytest.approx(vol, abs=5e-3)


def test_american_implied_vol_rejects_below_intrinsic_price() -> None:
    with pytest.raises(ValueError):
        american_implied_vol(price=1.0, forward=100.0, strike=50.0, t=1.0, discount=1.0, is_call=True)
