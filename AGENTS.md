# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A replication of selected numerical experiments from Fabrice Deschâtres,
"Convex Volatility Interpolation" (author version of the article published
in *Cutting Edge, Risk*, February 2026): the CVI arbitrage-free
implied-variance surface parameterization (cubic spline in normalized
log-moneyness, `n+2` free parameters per expiry — ATM variance, ATM skew,
`n` anchored convexities) and its two-step convex quadratic-program (QP)
calibration to market option quotes, solved with CVXPY and the Clarabel
solver.

## Layout

```
src/            flat library modules: spline parameterization, Black-76
                 pricing and implied volatility, American-option
                 de-Americanization, the QP calibration, static-arbitrage
                 diagnostics, and config.py (all experiment parameters as
                 module-level UPPER_CASE variables)
exp/            exp_1.py (single-surface calibration + arbitrage checks),
                 exp_2.py (Clarabel timing benchmark vs number of knots),
                 exp_3.py (chi-squared vs number of knots across underlyings)
tests/          pytest tests for every src/ module and for the data loader
                 (market-data tests are marked and skipped without an API key)
data/           data_loader.py: option-quote loading from the Massive REST
                 API, implied-variance conversion, forward and ATM-anchor
                 estimation
results/        RESULTS.md plus every table/plot each exp/ script saves
                 (overwritten on rerun — numbers in RESULTS.md must be
                 checked against the regenerated files and updated by hand)
```

`src/`, `exp/` and `data/` each contain an empty `__init__.py`. Experiments
run as `python -m exp.<name>` and tests as `python -m pytest` from the
repository root; no `sys.path` edits in code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY
```

Only `MASSIVE_API_KEY` is used by this repository (option-quote data comes
from Massive's options endpoints; `DATALOADER_API_KEY` is kept in
`.env.example` for consistency with the market-data convention but is not
called by any module here, since the Dataloader serves OHLCV bars, not
option quotes).

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # includes the real-data slice tests
```

Unit tests of `src/` run on synthetic fixtures (`tests/conftest.py`) and
check the paper's properties — signs and directions of the ATM skew and
convexities, absence of look-ahead between calibration iterations, spline
continuity at knots, QP feasibility, and the arbitrage diagnostics on both
an arbitrage-free and a deliberately broken synthetic surface — not the
paper's reported numbers. Tests marked `market_data` load a small, cached
slice of a real option chain through `data/data_loader.py` and are skipped
automatically when `MASSIVE_API_KEY` is not set.

## Experiments

```bash
python -m exp.exp_1            # reduced run (see results/RESULTS.md)
python -m exp.exp_1 --full     # paper's full expiry count
python -m exp.exp_2
python -m exp.exp_2 --full
python -m exp.exp_3
python -m exp.exp_3 --full
```

`reproduce.sh` runs the unit tests and all three experiments in their
reduced (default) configuration.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables, each with a one-line comment naming its paper
  section/equation or stating that the paper does not specify it. Where a
  parameter was reduced to fit the time available for this replication, the
  paper value and the executed value are two adjacent variables
  (`..._PAPER` / `..._RUN`), and each experiment's `--full` flag selects the
  paper values.
- All data access goes through `data/data_loader.py`; no module under
  `src/` or `exp/` calls an external API directly.
- `MASSIVE_API_KEY` (and `DATALOADER_API_KEY`, unused here) are read from
  the environment via `python-dotenv` and never written to code, logs, or
  results.
- After rerunning an experiment, the files it writes under `results/` are
  overwritten; `results/RESULTS.md` is not regenerated automatically; its
  numbers and verdicts must be checked against the new files and updated by
  hand.
- Deviations from the paper (data source, universe, reduced run sizes,
  assumed constants) are recorded only in `results/RESULTS.md`, not in code
  comments or config values beyond the one-line parameter source note.
