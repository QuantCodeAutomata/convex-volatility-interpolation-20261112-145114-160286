"""Unit tests for the Black-76 pricer / implied-vol solver (src/black_scholes.py)."""

from __future__ import annotations

import pytest

from src.black_scholes import black76_price, black76_vega, implied_vol


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("vol", [0.1, 0.25, 0.6])
def test_implied_vol_round_trips_black76_price(is_call: bool, vol: float) -> None:
    forward, strike, t, discount = 100.0, 105.0, 0.75, 0.97
    price = black76_price(forward, strike, t, vol, discount, is_call)
    recovered = implied_vol(price, forward, strike, t, discount, is_call)
    assert recovered == pytest.approx(vol, abs=1e-6)


def test_vega_is_positive_and_peaks_near_the_money() -> None:
    forward, t, discount = 100.0, 1.0, 0.95
    vega_atm = black76_vega(forward, 100.0, t, 0.2, discount)
    vega_otm = black76_vega(forward, 150.0, t, 0.2, discount)
    assert vega_atm > 0 and vega_otm > 0
    assert vega_atm > vega_otm


def test_implied_vol_rejects_price_below_intrinsic() -> None:
    with pytest.raises(ValueError):
        implied_vol(price=0.0, forward=100.0, strike=80.0, t=1.0, discount=1.0, is_call=True)


def test_black76_price_zero_maturity_is_intrinsic() -> None:
    assert black76_price(100.0, 90.0, 0.0, 0.2, 1.0, True) == pytest.approx(10.0)
    assert black76_price(100.0, 110.0, 0.0, 0.2, 1.0, False) == pytest.approx(10.0)
