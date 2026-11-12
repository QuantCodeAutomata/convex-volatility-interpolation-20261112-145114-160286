"""Tests for data/data_loader.py: synthetic edge cases plus a real-data slice."""

from __future__ import annotations

import pytest

from data.data_loader import load_quotes_table


@pytest.mark.market_data
def test_real_quotes_slice_has_expected_columns_and_signs(real_quotes_slice) -> None:
    df = real_quotes_slice
    assert not df.empty
    expected_cols = {"expiry_id", "T", "F", "sigma_star", "strike", "bid_iv", "ask_iv",
                      "vega_bid", "vega_ask"}
    assert expected_cols.issubset(df.columns)
    assert (df["T"] > 0).all()
    assert (df["F"] > 0).all()
    assert (df["strike"] > 0).all()
    quoted_both = df["bid_iv"].notna() & df["ask_iv"].notna()
    assert (df.loc[quoted_both, "ask_iv"] >= df.loc[quoted_both, "bid_iv"]).all()


@pytest.mark.market_data
def test_real_quotes_slice_respects_requested_expiry_count(real_quotes_slice) -> None:
    assert real_quotes_slice["expiry_id"].nunique() <= 2


@pytest.mark.market_data
def test_load_quotes_table_rejects_unknown_underlying() -> None:
    with pytest.raises(Exception):
        load_quotes_table("I:DOES_NOT_EXIST", "2026-11-11", max_expiries=1, n_strikes=5)
