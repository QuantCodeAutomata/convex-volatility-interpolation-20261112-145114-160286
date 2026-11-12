"""exp_2: CVI calibration-time benchmarking with the Clarabel QP solver.

Measures how the two-step calibration time and the sparse objective/
constraint matrix sizes scale with the number of knots per expiry (paper,
"Numerical Results", calibration-time table). Run with
``python -m exp.exp_2 [--full]``.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.data_loader import load_quotes_table
from src.config import (
    EXP2_KNOT_COUNTS_PAPER,
    EXP2_MAX_EXPIRIES_PAPER,
    EXP2_MAX_EXPIRIES_RUN,
    EXP2_STRIKES_PER_EXPIRY_RUN,
    EXP2_UNDERLYINGS,
    VALUATION_DATE,
)
from src.cvi_calibration import calibrate_surface

RESULTS_DIR = "results"


def run(full: bool = False) -> pd.DataFrame:
    """Benchmark calibration time across knot counts and save a results table."""
    max_expiries = EXP2_MAX_EXPIRIES_PAPER if full else EXP2_MAX_EXPIRIES_RUN
    rows = []
    for underlying in EXP2_UNDERLYINGS:
        print(f"[exp_2] loading {underlying} quotes as of {VALUATION_DATE} "
              f"(max_expiries={max_expiries})")
        t0 = time.time()
        quotes = load_quotes_table(
            underlying, VALUATION_DATE, max_expiries=max_expiries,
            n_strikes=EXP2_STRIKES_PER_EXPIRY_RUN,
        )
        load_time = time.time() - t0
        n_expiries = quotes["expiry_id"].nunique()
        print(f"[exp_2] {underlying}: {len(quotes)} quotes, {n_expiries} expiries, "
              f"loaded in {load_time:.1f}s")

        for n_knots in EXP2_KNOT_COUNTS_PAPER:
            result = calibrate_surface(quotes, n_knots=n_knots)
            rows.append({
                "asset": underlying,
                "n_expiries": int(n_expiries),
                "n_quotes": int(len(quotes)),
                "n_knots": n_knots,
                "n_variables": result.n_variables,
                "p_nnz": result.p_nnz,
                "a_nnz": result.a_nnz,
                "n_constraints_final": result.n_constraints_final,
                "calc_time_seconds": result.solve_time_total,
                "iteration_times_seconds": json.dumps(result.iteration_times),
            })
            print(f"[exp_2]   n_knots={n_knots:3d}  "
                  f"calc_time={result.solve_time_total:.4f}s  "
                  f"P_nnz={result.p_nnz}  A_nnz={result.a_nnz}")

    table = pd.DataFrame(rows)
    table.to_csv(f"{RESULTS_DIR}/exp2_timing.csv", index=False)
    _plot(table)
    print("[exp_2] saved results/exp2_timing.csv and results/exp2_timing.png")
    return table


def _plot(table: pd.DataFrame) -> None:
    """Calibration time vs number of knots per expiry, one line per asset."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for asset, group in table.groupby("asset"):
        group = group.sort_values("n_knots")
        ax.plot(group["n_knots"], group["calc_time_seconds"], marker="o", label=asset)
    ax.set_xlabel("knots per expiry")
    ax.set_ylabel("total calibration time (s)")
    ax.set_title("CVI calibration time vs number of knots (Clarabel solver)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_timing.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's full expiry count instead of the reduced run")
    args = parser.parse_args()
    run(full=args.full)
