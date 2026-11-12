# Convex Volatility Interpolation (CVI)

Replication of selected results from *Convex Volatility Interpolation*,
Fabrice Deschâtres (Volptima), author version of the article published in
*Cutting Edge, Risk*, February 2026.

This repository implements the CVI arbitrage-free implied-variance surface
parameterization and its two-step convex quadratic-program (QP) calibration,
and reproduces three of the paper's numerical experiments:

1. **exp_1 — CVI volatility surface calibration via convex optimization**:
   a single-underlying fit (objective, constraints, two-step calibration,
   static-arbitrage validation).
2. **exp_2 — Calibration-time benchmarking with the Clarabel solver**: how
   solve time, and the sparsity of the QP matrices `P` and `A`, scale with
   the number of knots per expiry.
3. **exp_3 — Chi-squared fit quality vs number of knots**: how the
   least-squares chi-squared statistic of the fit depends on the number of
   knots per expiry, across several underlyings.

## Repository layout

```
src/     CVI spline basis, Black-Scholes/76 helpers, de-Americanization,
         the two-step QP calibration, and static-arbitrage diagnostics
exp/     exp_1.py, exp_2.py, exp_3.py — the three experiment entry points
data/    data_loader.py — real option-quote loading through Massive REST
tests/   pytest unit tests (synthetic fixtures) and a market-data test
results/ RESULTS.md plus every table, plot and metric file the experiments save
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # includes the real-data slice test
```

## Running the experiments

```bash
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
```

Each experiment accepts a `--full` flag that selects the paper's full
configuration (see `src/config.py` for the paper vs. executed parameter
pairs, and `results/RESULTS.md` for the executed configuration and the
reasons it was reduced).

## Data source

Option quotes (bid/ask implied volatilities, forward prices) are loaded
through `data/data_loader.py` from the Massive REST API (options chain
snapshot and reference endpoints); OHLCV bars are not used by these
experiments. See `results/RESULTS.md` for the underlyings actually used and
every deviation from the paper's data.

## Reference

Fabrice Deschâtres, "Convex Volatility Interpolation", author version of the
article published in *Cutting Edge, Risk*, February 2026.
