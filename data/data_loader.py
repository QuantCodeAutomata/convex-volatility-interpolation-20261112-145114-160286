"""Market option quotes for the CVI calibration, loaded from Massive REST.

The CVI fit needs, for one underlying, a table of bid/ask-quoted options
across strikes and expiries plus each expiry's forward price. Bars (OHLCV)
are not used anywhere in this replication: the paper's inputs are option
quotes, which the QuantX Dataloader does not serve, so they are loaded
directly from the Massive REST options endpoints
(``/v3/snapshot/options/{underlying}`` for the chain, ``/v3/quotes/{ticker}``
for NBBO bid/ask), as the data the Dataloader does not serve.

Forward prices are not returned by the API and are instead recovered from
put-call parity on the near-the-money strikes of each expiry (paper,
"Pre-computation": "forward price F_Tj for each expiry"), and the ATM anchor
volatility sigma_* is the implied volatility of the option whose strike is
closest to that forward (paper leaves the exact anchor-estimation method
unspecified).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.american_options import american_implied_vol
from src.black_scholes import black76_vega, implied_vol
from src.config import DISCOUNT_RATE, MASSIVE_API_ROOT

RETRY_STATUSES = {429, 502, 503, 504}


class MassiveDataError(RuntimeError):
    """A failed Massive REST request or a response missing required fields."""


def _headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise MassiveDataError("MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {key}"}


def _get(url: str, params: dict | None = None) -> dict:
    headers = _headers()
    for attempt in range(4):
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code not in RETRY_STATUSES:
            break
        time.sleep(1.5 * (attempt + 1))
    if response.status_code != 200:
        raise MassiveDataError(f"GET {url}: HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def fetch_option_chain(underlying: str, expiration_gte: str, expiration_lte: str,
                        strike_gte: float | None = None, strike_lte: float | None = None,
                        max_contracts: int = 2000) -> list[dict]:
    """Paginate /v3/snapshot/options/{underlying}, returning contract details.

    Each item has keys: ticker, strike, contract_type ("call"/"put"),
    exercise_style, expiration_date, close (the option's own last daily
    close, used only to estimate a forward/spot ballpark before quotes are
    fetched).
    """
    params = {
        "limit": 250,
        "order": "asc",
        "sort": "strike_price",
        "expiration_date.gte": expiration_gte,
        "expiration_date.lte": expiration_lte,
    }
    if strike_gte is not None:
        params["strike_price.gte"] = strike_gte
    if strike_lte is not None:
        params["strike_price.lte"] = strike_lte
    url = f"{MASSIVE_API_ROOT}/v3/snapshot/options/{underlying}"
    contracts: list[dict] = []
    while url and len(contracts) < max_contracts:
        data = _get(url, params)
        results = data.get("results", [])
        for item in results:
            details = item.get("details", {})
            contracts.append({
                "ticker": details.get("ticker"),
                "strike": details.get("strike_price"),
                "contract_type": details.get("contract_type"),
                "exercise_style": details.get("exercise_style"),
                "expiration_date": details.get("expiration_date"),
                "close": (item.get("day") or {}).get("close"),
            })
        url = data.get("next_url")
        params = None  # next_url already carries the query string
    if not contracts:
        raise MassiveDataError(f"empty option chain for {underlying} in [{expiration_gte}, {expiration_lte}]")
    return contracts


def fetch_recent_quote(ticker: str, n: int = 25) -> tuple[float, float] | None:
    """Most recent two-sided (bid, ask) quote among the last `n` NBBO ticks.

    The very latest tick is sometimes a stale zero-quote print; scanning the
    last `n` ticks for the first one with both sides positive gives a usable
    two-sided market.
    """
    url = f"{MASSIVE_API_ROOT}/v3/quotes/{ticker}"
    data = _get(url, {"limit": n, "order": "desc", "sort": "timestamp"})
    for quote in data.get("results", []):
        bid, ask = quote.get("bid_price", 0.0), quote.get("ask_price", 0.0)
        if bid and ask and ask > bid > 0:
            return float(bid), float(ask)
    return None


@dataclass
class ExpiryQuotes:
    expiration_date: str
    T: float
    forward: float
    sigma_star: float
    rows: pd.DataFrame  # strike, bid_iv, ask_iv, vega_bid, vega_ask


def _nearest_crossing_forward(contracts: list[dict]) -> float:
    """Rough forward estimate from the call/put day-close crossing strike.

    Used only to center the strike window before any live quote is
    requested; the final forward used in the fit comes from put-call parity
    on live quotes in `build_expiry_quotes`.
    """
    by_strike: dict[float, dict[str, float]] = {}
    for c in contracts:
        if c["close"] is None:
            continue
        by_strike.setdefault(c["strike"], {})[c["contract_type"]] = c["close"]
    pairs = sorted(
        (k, v["call"] - v["put"]) for k, v in by_strike.items() if "call" in v and "put" in v
    )
    if not pairs:
        raise MassiveDataError("no call/put pair with both closes to estimate a forward")
    for (k0, d0), (k1, d1) in zip(pairs, pairs[1:]):
        if d0 >= 0 >= d1:
            if d0 == d1:
                return k0
            w = d0 / (d0 - d1)
            return k0 + w * (k1 - k0)
    return min(pairs, key=lambda p: abs(p[1]))[0]


def build_expiry_quotes(underlying: str, expiration_date: str, valuation_date: pd.Timestamp,
                         n_strikes: int, discount_rate: float = DISCOUNT_RATE,
                         max_workers: int = 8) -> ExpiryQuotes | None:
    """Build one expiry's quote table: forward, ATM anchor vol, and the
    per-strike bid/ask implied variances used by `src.cvi_calibration`.

    Only the out-of-the-money side (call if K >= F, put otherwise) is kept
    for each strike, matching the paper's "tightest quotes ... from both
    calls and puts" selection; a handful of near-the-money strikes are
    queried on both sides to recover the forward from put-call parity.
    """
    t = (pd.Timestamp(expiration_date) - valuation_date).days / 365.0
    if t <= 1.0 / 365.0:
        return None
    discount = np.exp(-discount_rate * t)

    wide = fetch_option_chain(underlying, expiration_date, expiration_date, max_contracts=4000)
    rough_forward = _nearest_crossing_forward(wide)
    strikes_sorted = sorted({c["strike"] for c in wide})
    if len(strikes_sorted) < n_strikes:
        return None

    by_strike_type: dict[tuple[float, str], str] = {
        (c["strike"], c["contract_type"]): c["ticker"] for c in wide
    }
    exercise_style = wide[0]["exercise_style"] or "european"
    is_american = exercise_style == "american"

    near_idx = int(np.argmin([abs(k - rough_forward) for k in strikes_sorted]))
    near_strikes = strikes_sorted[max(0, near_idx - 2): near_idx + 3]

    step = max(len(strikes_sorted) // n_strikes, 1)
    lo = max(0, near_idx - step * n_strikes // 2)
    window = strikes_sorted[lo: lo + step * n_strikes: step][:n_strikes]
    selected_strikes = sorted(set(window) | set(near_strikes))

    def quote_both(strike: float) -> dict:
        out = {"strike": strike}
        for side in ("call", "put"):
            ticker = by_strike_type.get((strike, side))
            if ticker is None:
                continue
            q = fetch_recent_quote(ticker)
            if q is not None:
                out[side] = q
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        quoted_near = list(pool.map(quote_both, near_strikes))

    pairs = [(q["strike"], q["call"], q["put"]) for q in quoted_near if "call" in q and "put" in q]
    if len(pairs) < 2:
        return None
    strikes_arr = np.array([p[0] for p in pairs])
    diffs = np.array([0.5 * (p[1][0] + p[1][1]) - 0.5 * (p[2][0] + p[2][1]) for p in pairs])
    # C_mid - P_mid = discount * (F - K)  =>  linear regression in K.
    slope, intercept = np.polyfit(strikes_arr, diffs, 1)
    forward = float(intercept / discount - slope / discount * 0) if False else None
    # Solve discount*(F-K) = intercept + slope*K  =>  -discount = slope, discount*F = intercept
    forward = float(intercept / discount + strikes_arr.mean() * 0)
    forward = float(intercept / discount) if abs(slope + discount) < 0.5 * discount else float(
        -intercept / slope
    )

    def quote_one(strike: float) -> dict | None:
        side = "call" if strike >= forward else "put"
        ticker = by_strike_type.get((strike, side))
        if ticker is None:
            return None
        cached = next((q for q in quoted_near if q["strike"] == strike and side in q), None)
        bid_ask = cached[side] if cached else fetch_recent_quote(ticker)
        if bid_ask is None:
            return None
        return {"strike": strike, "side": side, "bid": bid_ask[0], "ask": bid_ask[1]}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        quoted = [q for q in pool.map(quote_one, selected_strikes) if q is not None]
    if len(quoted) < 4:
        return None

    iv_fn = american_implied_vol if is_american else implied_vol
    records = []
    for q in quoted:
        is_call = q["side"] == "call"
        try:
            bid_iv = iv_fn(q["bid"], forward, q["strike"], t, discount, is_call)
            ask_iv = iv_fn(q["ask"], forward, q["strike"], t, discount, is_call)
        except ValueError:
            continue
        vega_bid = black76_vega(forward, q["strike"], t, bid_iv, discount)
        vega_ask = black76_vega(forward, q["strike"], t, ask_iv, discount)
        records.append({
            "strike": q["strike"], "bid_iv": bid_iv, "ask_iv": ask_iv,
            "vega_bid": vega_bid, "vega_ask": vega_ask,
        })
    if len(records) < 4:
        return None
    rows = pd.DataFrame(records).sort_values("strike").reset_index(drop=True)
    atm_row = rows.iloc[(rows["strike"] - forward).abs().argsort().iloc[0]]
    sigma_star = 0.5 * (atm_row["bid_iv"] + atm_row["ask_iv"])

    return ExpiryQuotes(expiration_date=expiration_date, T=t, forward=forward,
                         sigma_star=float(sigma_star), rows=rows)


def load_quotes_table(underlying: str, valuation_date: str, max_expiries: int,
                       n_strikes: int, discount_rate: float = DISCOUNT_RATE) -> pd.DataFrame:
    """Tidy quotes table for `src.cvi_calibration.calibrate_surface`.

    Columns: expiry_id, T, F, sigma_star, strike, bid_iv, ask_iv, vega_bid,
    vega_ask. Expiries with fewer than 4 usable two-sided quotes are
    skipped; the realized expiry count may be below `max_expiries`.
    """
    val_date = pd.Timestamp(valuation_date)
    far = (val_date + pd.Timedelta(days=760)).strftime("%Y-%m-%d")
    near = (val_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    chain = fetch_option_chain(underlying, near, far, max_contracts=250)
    expirations = sorted({c["expiration_date"] for c in chain})

    frames = []
    for expiration_date in expirations:
        if len(frames) >= max_expiries:
            break
        expiry = build_expiry_quotes(underlying, expiration_date, val_date, n_strikes, discount_rate)
        if expiry is None:
            continue
        rows = expiry.rows.copy()
        rows["expiry_id"] = expiry.expiration_date
        rows["T"] = expiry.T
        rows["F"] = expiry.forward
        rows["sigma_star"] = expiry.sigma_star
        frames.append(rows)
    if not frames:
        raise MassiveDataError(f"no usable expiry for {underlying} (0 of {len(expirations)} expiries)")
    table = pd.concat(frames, ignore_index=True)
    return table[["expiry_id", "T", "F", "sigma_star", "strike", "bid_iv", "ask_iv", "vega_bid", "vega_ask"]]
