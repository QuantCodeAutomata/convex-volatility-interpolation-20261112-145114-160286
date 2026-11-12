"""Shared pytest fixtures: synthetic quote tables and a cached real-data slice."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from data.data_loader import load_quotes_table


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test loads a real options slice through data/data_loader.py"
    )


def synthetic_quotes(n_expiries: int = 3, n_strikes: int = 15, forward: float = 100.0,
                      seed: int = 0) -> pd.DataFrame:
    """A small, arbitrage-free-by-construction synthetic quote table.

    Each expiry has a flat-plus-skew smile sigma(z) = sigma_* (1 + a*z + b*z^2)
    around the forward, quoted with a fixed +/-0.002 vol bid/ask half-spread.
    Used by src/ unit tests instead of real market data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    maturities = np.linspace(0.1, 1.5, n_expiries)
    for T in maturities:
        sigma_star = float(0.18 + 0.03 * rng.random())
        strikes = forward * np.exp(np.linspace(-0.45, 0.45, n_strikes))
        for k in strikes:
            z = np.log(k / forward) / (sigma_star * np.sqrt(T))
            vol = sigma_star * (1.0 + 0.08 * z + 0.015 * z ** 2)
            vol = max(vol, 0.03)
            rows.append({
                "expiry_id": f"T{T:.4f}", "T": T, "F": forward, "sigma_star": sigma_star,
                "strike": k, "bid_iv": vol - 0.002, "ask_iv": vol + 0.002,
                "vega_bid": 0.3, "vega_ask": 0.3,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_quotes_table() -> pd.DataFrame:
    return synthetic_quotes()


@pytest.fixture(scope="session")
def real_quotes_slice(request: pytest.FixtureRequest) -> pd.DataFrame:
    """A small real SPX options slice, cached across test runs."""
    if not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("MASSIVE_API_KEY is not set; skipping market-data test")
    cache_dir = request.config.cache.mkdir("market_data_slice")
    cache_file = cache_dir / "spx_slice.json"
    if cache_file.exists():
        return pd.read_json(cache_file)
    table = load_quotes_table("I:SPX", "2026-11-11", max_expiries=2, n_strikes=10)
    table.to_json(cache_file)
    return table
