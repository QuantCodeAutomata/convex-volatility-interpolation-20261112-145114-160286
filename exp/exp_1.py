"""exp_1: CVI volatility-surface calibration via convex optimization.

Fits one underlying's full implied-variance surface with the two-step CVI
quadratic program of ``src.cvi_calibration.calibrate_surface`` and checks
that the result is free of static arbitrage (paper, "Putting It All
Together", steps 7-10). Run with ``python -m exp.exp_1 [--full]``.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import load_quotes_table
from src.arbitrage_checks import validate_surface
from src.config import (
    EXP1_MAX_EXPIRIES_PAPER,
    EXP1_MAX_EXPIRIES_RUN,
    EXP1_N_KNOTS,
    EXP1_STRIKES_PER_EXPIRY,
    EXP1_UNDERLYING,
    VALUATION_DATE,
)
from src.cvi_calibration import calibrate_surface

RESULTS_DIR = "results"


def run(full: bool = False) -> dict:
    """Run the exp_1 calibration and save its artifacts to `results/`."""
    max_expiries = EXP1_MAX_EXPIRIES_PAPER if full else EXP1_MAX_EXPIRIES_RUN

    print(f"[exp_1] loading {EXP1_UNDERLYING} quotes as of {VALUATION_DATE} "
          f"(max_expiries={max_expiries}, strikes/expiry={EXP1_STRIKES_PER_EXPIRY})")
    t0 = time.time()
    quotes = load_quotes_table(
        EXP1_UNDERLYING, VALUATION_DATE, max_expiries=max_expiries,
        n_strikes=EXP1_STRIKES_PER_EXPIRY,
    )
    load_time = time.time() - t0
    n_expiries = quotes["expiry_id"].nunique()
    print(f"[exp_1] loaded {len(quotes)} quotes across {n_expiries} expiries "
          f"in {load_time:.1f}s")

    result = calibrate_surface(quotes, n_knots=EXP1_N_KNOTS)
    report = validate_surface(result)

    print(f"[exp_1] solve_time_total={result.solve_time_total:.4f}s "
          f"chi2_avg={result.chi_squared_avg:.4f} "
          f"calendar_violations={report.calendar_spread_violations} "
          f"butterfly_violations={report.butterfly_violations} "
          f"min_pdf={report.min_pdf:.6g}")

    summary = {
        "underlying": EXP1_UNDERLYING,
        "valuation_date": VALUATION_DATE,
        "n_expiries": int(n_expiries),
        "n_quotes": int(len(quotes)),
        "n_knots": EXP1_N_KNOTS,
        "n_variables": result.n_variables,
        "n_constraints_final": result.n_constraints_final,
        "p_nnz": result.p_nnz,
        "a_nnz": result.a_nnz,
        "load_time_seconds": load_time,
        "solve_time_total_seconds": result.solve_time_total,
        "iteration_times_seconds": result.iteration_times,
        "chi_squared_avg": result.chi_squared_avg,
        "chi_squared_per_expiry": result.chi_squared_per_expiry,
        "calendar_spread_violations": report.calendar_spread_violations,
        "butterfly_violations": report.butterfly_violations,
        "min_implied_pdf": report.min_pdf,
        "max_total_variance_decrease": report.max_total_variance_decrease,
        "full_configuration": full,
    }
    with open(f"{RESULTS_DIR}/exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _plot_smiles(result)
    print("[exp_1] saved results/exp1_summary.json and results/exp1_smiles.png")
    return summary


def _plot_smiles(result) -> None:
    """Plot fitted implied-volatility smiles against bid/ask mid quotes."""
    n_show = min(6, len(result.expiries))
    idx = np.linspace(0, len(result.expiries) - 1, n_show).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    for ax, j in zip(axes.flat, idx):
        e = result.expiries[j]
        if len(e.k_mid):
            z_mid = e.z(e.k_mid)
            ax.scatter(z_mid, np.sqrt(e.v_mid), s=12, color="black", label="mid IV")
        z_grid = np.linspace(result.bases[j].knots[0], result.bases[j].knots[-1], 200)
        v_fit = result.evaluate(j, z_grid)
        ax.plot(z_grid, np.sqrt(np.clip(v_fit, 0, None)), color="C0", label="CVI fit")
        ax.set_title(f"T={e.T:.3f}y  chi2={result.chi_squared_per_expiry[e.expiry_id]:.3g}")
        ax.set_xlabel("normalized log-moneyness z")
        ax.set_ylabel("implied vol")
        ax.legend(fontsize=8)
    fig.suptitle(f"CVI fit: {result.expiries[0].expiry_id if result.expiries else ''} "
                 f"({len(result.expiries)} expiries, n_knots={result.n_knots})")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_smiles.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's full expiry count instead of the reduced run")
    args = parser.parse_args()
    run(full=args.full)
