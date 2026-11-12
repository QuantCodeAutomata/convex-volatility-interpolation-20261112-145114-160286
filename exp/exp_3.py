"""exp_3: chi-squared fit quality vs number of knots per expiry.

For each underlying, fits the CVI surface at every knot count in a range and
records the least-squares penalty (chi-squared statistic, averaged across
expiries) to find the knot count at which the fit quality plateaus (paper,
"Numerical Results", chi-squared-vs-knots figure). Run with
``python -m exp.exp_3 [--full]``.
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.data_loader import load_quotes_table
from src.config import (
    EXP3_KNOT_RANGE_PAPER,
    EXP3_KNOT_RANGE_RUN,
    EXP3_MAX_EXPIRIES_RUN,
    EXP3_STRIKES_PER_EXPIRY_RUN,
    EXP3_UNDERLYINGS_PAPER,
    EXP3_UNDERLYINGS_RUN,
    VALUATION_DATE,
)
from src.cvi_calibration import calibrate_surface

RESULTS_DIR = "results"


def run(full: bool = False) -> pd.DataFrame:
    """Compute chi-squared vs knot count for every underlying and save results."""
    underlyings = EXP3_UNDERLYINGS_PAPER if full else EXP3_UNDERLYINGS_RUN
    knot_range = EXP3_KNOT_RANGE_PAPER if full else EXP3_KNOT_RANGE_RUN
    rows = []
    for underlying in underlyings:
        print(f"[exp_3] loading {underlying} quotes as of {VALUATION_DATE} "
              f"(max_expiries={EXP3_MAX_EXPIRIES_RUN})")
        try:
            quotes = load_quotes_table(
                underlying, VALUATION_DATE, max_expiries=EXP3_MAX_EXPIRIES_RUN,
                n_strikes=EXP3_STRIKES_PER_EXPIRY_RUN,
            )
        except Exception as exc:  # pragma: no cover - network/entitlement failure
            print(f"[exp_3]   {underlying}: failed to load ({exc}); skipped")
            continue
        n_expiries = quotes["expiry_id"].nunique()
        print(f"[exp_3] {underlying}: {len(quotes)} quotes, {n_expiries} expiries")

        for n_knots in knot_range:
            t0 = time.time()
            try:
                result = calibrate_surface(quotes, n_knots=n_knots)
            except RuntimeError as exc:
                # Too few knots can leave the QP infeasible for a sparse quote
                # set (not enough free parameters to satisfy every linear
                # no-arbitrage constraint); skip this point rather than
                # fabricate a value.
                print(f"[exp_3]   {underlying} n_knots={n_knots}: {exc}; skipped")
                continue
            rows.append({
                "asset": underlying,
                "n_expiries": int(n_expiries),
                "n_knots": n_knots,
                "chi_squared_avg": result.chi_squared_avg,
                "solve_time_seconds": time.time() - t0,
            })
        print(f"[exp_3]   knots {knot_range[0]}-{knot_range[-1]} done")

    table = pd.DataFrame(rows)
    table.to_csv(f"{RESULTS_DIR}/exp3_chi2_vs_knots.csv", index=False)
    _plot(table)
    print("[exp_3] saved results/exp3_chi2_vs_knots.csv and results/exp3_chi2_vs_knots.png")
    return table


def _plot(table: pd.DataFrame) -> None:
    """Chi-squared (log scale) vs number of knots, one line per underlying."""
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for asset, group in table.groupby("asset"):
        group = group.sort_values("n_knots")
        ax.plot(group["n_knots"], group["chi_squared_avg"], marker="o", label=asset)
    ax.set_yscale("log")
    ax.set_xlabel("knots per expiry")
    ax.set_ylabel("chi-squared (average across expiries, log scale)")
    ax.set_title("CVI fit quality vs number of knots")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_chi2_vs_knots.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's full underlying list and knot range")
    args = parser.parse_args()
    run(full=args.full)
