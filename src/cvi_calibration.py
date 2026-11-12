"""CVI volatility-surface calibration: the convex QP of the paper.

Builds and solves, through CVXPY, the quadratic program whose objective is
the sum of the Least Squares, Above Ask, Below Bid and Strike Regularization
penalties and whose constraints enforce linear extrapolation, positivity of
variance, no-calendar-spread, Lee's tail-slope bounds and no-butterfly
arbitrage (paper, "Objective Function" and "Non-Arbitrage-Related/Related
Constraints"). CVXPY (Diamond and Boyd, 2016) is the modeling language and
Clarabel (Goulart and Chen, 2024) the QP solver, as specified by the paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from src.config import (
    LEE_MARGIN,
    N_BUTTERFLY_ITERATIONS,
    N_DISCRETIZATION_STRIKES,
    QP_SOLVER,
    STRIKE_REG_LAMBDA,
)
from src.cvi_spline import SplineBasis, build_knots


@dataclass
class ExpiryData:
    """One expiry's market quotes, already converted to implied variance."""

    expiry_id: str
    T: float
    F: float
    sigma_star: float
    k_mid: np.ndarray
    v_mid: np.ndarray
    v_bid_mid: np.ndarray
    v_ask_mid: np.ndarray
    k_ask_only: np.ndarray
    v_ask_only: np.ndarray
    vega_ask_only: np.ndarray
    k_bid_only: np.ndarray
    v_bid_only: np.ndarray
    vega_bid_only: np.ndarray

    @property
    def v_star(self) -> float:
        return self.sigma_star ** 2

    def z(self, k: np.ndarray) -> np.ndarray:
        return k / (self.sigma_star * np.sqrt(self.T))


def prepare_expiries(quotes: pd.DataFrame) -> list[ExpiryData]:
    """Group a tidy quotes table into one ExpiryData per expiry, sorted by T.

    `quotes` has one row per (expiry, strike) with columns: expiry_id, T, F,
    sigma_star, strike, bid_iv, ask_iv (NaN where the side is not quoted).
    Variance is sigma^2; `k` is log-forward-moneyness log(K/F).
    """
    expiries = []
    for expiry_id, grp in quotes.groupby("expiry_id", sort=False):
        grp = grp.sort_values("strike")
        T, F, sigma_star = grp["T"].iloc[0], grp["F"].iloc[0], grp["sigma_star"].iloc[0]
        k = np.log(grp["strike"].to_numpy() / F)
        v_bid = grp["bid_iv"].to_numpy() ** 2
        v_ask = grp["ask_iv"].to_numpy() ** 2
        has_bid = ~np.isnan(v_bid)
        has_ask = ~np.isnan(v_ask)
        both = has_bid & has_ask
        ask_only = has_ask & ~has_bid
        bid_only = has_bid & ~has_ask
        v_mid = 0.5 * (v_bid[both] + v_ask[both])
        expiries.append(ExpiryData(
            expiry_id=str(expiry_id), T=float(T), F=float(F), sigma_star=float(sigma_star),
            k_mid=k[both], v_mid=v_mid, v_bid_mid=v_bid[both], v_ask_mid=v_ask[both],
            k_ask_only=k[ask_only], v_ask_only=v_ask[ask_only],
            vega_ask_only=grp["vega_ask"].to_numpy()[ask_only],
            k_bid_only=k[bid_only], v_bid_only=v_bid[bid_only],
            vega_bid_only=grp["vega_bid"].to_numpy()[bid_only],
        ))
    expiries.sort(key=lambda e: e.T)
    return expiries


@dataclass
class CVIFitResult:
    expiries: list[ExpiryData]
    bases: list[SplineBasis]
    params: list[np.ndarray]           # fitted x per expiry (length n)
    chi_squared_per_expiry: dict
    chi_squared_avg: float
    solve_time_total: float
    iteration_times: list
    n_variables: int
    n_constraints_final: int
    p_nnz: int
    a_nnz: int
    n_knots: int

    def evaluate(self, expiry_index: int, z: np.ndarray) -> np.ndarray:
        V, _, _ = self.bases[expiry_index].eval_rows(z)
        return V @ self.params[expiry_index]


def _edge_butterfly_rhs(v: float, v_star: float, T: float, k: float) -> float:
    """Closed-form RHS of (3)/(4) at an edge knot, as a function of v."""
    sqrt_vT = np.sqrt(v * T)
    denom = k ** 2 - (v ** 2 * T ** 2) / 4 - v * T
    return 2 * v * np.sqrt(T) * (k - np.sign(k) * sqrt_vT * np.sqrt(1 + v * T / 4)) / (np.sqrt(v_star) * denom)


def calibrate_surface(
    quotes: pd.DataFrame,
    n_knots: int,
    *,
    lambda_reg: float = STRIKE_REG_LAMBDA,
    n_discretization: int = N_DISCRETIZATION_STRIKES,
    n_butterfly_iterations: int = N_BUTTERFLY_ITERATIONS,
    lee_margin: float = LEE_MARGIN,
    solver: str = QP_SOLVER,
) -> CVIFitResult:
    """Two-step CVI calibration of a full volatility surface.

    quotes: tidy DataFrame, one row per (expiry, strike), columns
        expiry_id, T, F, sigma_star, strike, bid_iv, ask_iv (ask_iv/bid_iv
        may be NaN when only one side is quoted).
    n_knots: number of spline knots per expiry (same for every expiry).
    Returns a CVIFitResult with the fitted parameters, the chi-squared
    statistic per expiry, timing and QP problem-size diagnostics.
    """
    expiries = prepare_expiries(quotes)
    m = len(expiries)
    if m == 0:
        raise ValueError("no expiries found in the quotes table")
    bases = []
    knots_list = []
    for e in expiries:
        knots, atm_index = build_knots(n_knots)
        bases.append(SplineBasis(knots, atm_index))
        knots_list.append(knots)

    x_vars = [cp.Variable(n_knots, name=f"x_{j}") for j in range(m)]
    z_grid = np.linspace(-1.0, 1.0, n_discretization) * knots_list[0][-1]

    # Pre-compute, per expiry, the basis rows needed for the objective and
    # for the discretized/tail constraints.
    rows_mid = []   # (V, z) at mid-quoted strikes
    rows_ask = []
    rows_bid = []
    rows_grid = []  # (V, Vp, Vpp) on the common discretized z grid
    rows_edges = []  # (V0, Vp0, Vpp0, Vn, Vpn, Vppn) at z0, z_{n-1}
    for e, basis in zip(expiries, bases):
        z_mid = e.z(e.k_mid)
        z_ask = e.z(e.k_ask_only)
        z_bid = e.z(e.k_bid_only)
        rows_mid.append(basis.eval_rows(z_mid) if len(z_mid) else (np.zeros((0, n_knots)),) * 3)
        rows_ask.append(basis.eval_rows(z_ask) if len(z_ask) else (np.zeros((0, n_knots)),) * 3)
        rows_bid.append(basis.eval_rows(z_bid) if len(z_bid) else (np.zeros((0, n_knots)),) * 3)
        rows_grid.append(basis.eval_rows(z_grid))
        rows_edges.append(basis.eval_rows(np.array([basis.knots[0], basis.knots[-1]])))

    # ------------------------------------------------------------------
    # Objective (built once; independent of the butterfly iteration)
    # ------------------------------------------------------------------
    ls_terms, reg_terms = [], []
    above_ask_slacks, below_bid_slacks = [], []
    above_ask_terms, below_bid_terms = [], []
    for j, e in enumerate(expiries):
        Vm, _, _ = rows_mid[j]
        z0, zn = bases[j].knots[0], bases[j].knots[-1]
        in_range_mid = (e.z(e.k_mid) >= z0) & (e.z(e.k_mid) <= zn) if len(e.k_mid) else np.array([], dtype=bool)
        if Vm.shape[0] and in_range_mid.any():
            Vm_in = Vm[in_range_mid]
            v_mid_in = e.v_mid[in_range_mid]
            w = 1.0 / (e.v_ask_mid[in_range_mid] - e.v_bid_mid[in_range_mid]) ** 2
            n_mid = max(int(in_range_mid.sum()), 1)
            ls_terms.append(cp.sum(cp.multiply(w, cp.square(Vm_in @ x_vars[j] - v_mid_in))) / n_mid)
            q_j = float(np.sum(w))
        else:
            q_j = 0.0

        Va, _, _ = rows_ask[j]
        if Va.shape[0]:
            vega_sum = e.vega_ask_only.sum()
            w_ask = q_j * e.vega_ask_only / vega_sum if vega_sum > 0 else np.zeros_like(e.vega_ask_only)
            t = cp.Variable(Va.shape[0], nonneg=True)
            above_ask_slacks.append((t, Va, e.v_ask_only))
            n_ask = max(Va.shape[0], 1)
            above_ask_terms.append(cp.sum(cp.multiply(w_ask, cp.square(t))) / n_ask)

        Vb, _, _ = rows_bid[j]
        z_bid_all = e.z(e.k_bid_only)
        in_range_bid = (z_bid_all >= z0) & (z_bid_all <= zn) if len(z_bid_all) else np.array([], dtype=bool)
        if Vb.shape[0] and in_range_bid.any():
            Vb_in, v_bid_in, vega_bid_in = Vb[in_range_bid], e.v_bid_only[in_range_bid], e.vega_bid_only[in_range_bid]
            vega_sum = vega_bid_in.sum()
            w_bid = q_j * vega_bid_in / vega_sum if vega_sum > 0 else np.zeros_like(vega_bid_in)
            t = cp.Variable(Vb_in.shape[0], nonneg=True)
            below_bid_slacks.append((t, Vb_in, v_bid_in))
            n_bid = max(Vb_in.shape[0], 1)
            below_bid_terms.append(cp.sum(cp.multiply(w_bid, cp.square(t))) / n_bid)

        # Strike-regularization: total variation of convexities c_0..c_{n-1}
        # (c_0 = c_{n-1} = 0 structurally; interior c_i = x_vars[j][2 + i - 1]).
        c_full = cp.hstack([0.0, x_vars[j][2:], 0.0])
        reg_terms.append(cp.sum(cp.abs(c_full[1:] - c_full[:-1])))

    objective = cp.sum(ls_terms) if ls_terms else 0
    if above_ask_terms:
        objective = objective + cp.sum(above_ask_terms)
    if below_bid_terms:
        objective = objective + cp.sum(below_bid_terms)
    objective = objective + lambda_reg * cp.sum(reg_terms)

    base_constraints = []
    ai = 0
    for j, e in enumerate(expiries):
        Va, _, _ = rows_ask[j]
        if Va.shape[0]:
            t, V, v_ask = above_ask_slacks[ai]
            base_constraints.append(t >= V @ x_vars[j] - v_ask)
            ai += 1
    bi = 0
    for j, e in enumerate(expiries):
        z_bid_all = e.z(e.k_bid_only)
        z0, zn = bases[j].knots[0], bases[j].knots[-1]
        in_range_bid = (z_bid_all >= z0) & (z_bid_all <= zn) if len(z_bid_all) else np.array([], dtype=bool)
        if in_range_bid.any():
            t, V, v_bid = below_bid_slacks[bi]
            base_constraints.append(V @ x_vars[j] - t <= v_bid)
            bi += 1

    return _solve_two_step(
        expiries, bases, x_vars, objective, base_constraints, rows_grid, rows_edges, z_grid,
        n_knots, n_butterfly_iterations, lee_margin, solver,
    )


def _solve_two_step(expiries, bases, x_vars, objective, base_constraints, rows_grid, rows_edges,
                     z_grid, n_knots, n_butterfly_iterations, lee_margin, solver) -> CVIFitResult:
    m = len(expiries)
    x_prev = None
    iteration_times = []
    problem = None
    for iteration in range(n_butterfly_iterations):
        constraints = list(base_constraints)

        # Tail positivity dv/dz(z0) <= 0, dv/dz(z_{n-1}) >= 0 (all expiries).
        for j in range(m):
            _, Vp, _ = rows_edges[j]
            constraints += [Vp[0] @ x_vars[j] <= 0, Vp[1] @ x_vars[j] >= 0]

        # Positivity of the variance on the discretized grid, first expiry only.
        Vg0, _, _ = rows_grid[0]
        constraints.append(Vg0 @ x_vars[0] >= 0)

        # Lee's tail-slope bounds (all expiries), enforced with a small margin.
        for j, e in enumerate(expiries):
            v_star, T = e.v_star, e.T
            bound = 2.0 / np.sqrt(v_star * T)
            _, Vp, _ = rows_edges[j]
            constraints.append(Vp[0] @ x_vars[j] >= v_star * (-bound + lee_margin))
            constraints.append(Vp[1] @ x_vars[j] <= v_star * (bound - lee_margin))

        # No-calendar-spread arbitrage: interior discretized + tail version.
        for j in range(m - 1):
            e_j, e_j1 = expiries[j], expiries[j + 1]
            Vg_j, _, _ = rows_grid[j]
            k_grid = z_grid * e_j.sigma_star * np.sqrt(e_j.T)
            z_j1 = k_grid / (e_j1.sigma_star * np.sqrt(e_j1.T))
            in_range = (z_j1 >= bases[j + 1].knots[0]) & (z_j1 <= bases[j + 1].knots[-1])
            if in_range.any():
                V_j1, _, _ = bases[j + 1].eval_rows(z_j1[in_range])
                constraints.append(
                    Vg_j[in_range] @ x_vars[j] * e_j.T <= V_j1 @ x_vars[j + 1] * e_j1.T
                )
            Vp_j, Vp_j1 = rows_edges[j][1], rows_edges[j + 1][1]
            scale_j = np.sqrt(e_j.T / e_j.v_star)
            scale_j1 = np.sqrt(e_j1.T / e_j1.v_star)
            constraints.append(Vp_j[0] @ x_vars[j] * scale_j >= Vp_j1[0] @ x_vars[j + 1] * scale_j1)
            constraints.append(Vp_j[1] @ x_vars[j] * scale_j <= Vp_j1[1] @ x_vars[j + 1] * scale_j1)

        # No-butterfly-arbitrage, linearized around the previous iteration.
        if iteration > 0 and x_prev is not None:
            for j, e in enumerate(expiries):
                v_star, T = e.v_star, e.T
                Vg, Vpg, Vppg = rows_grid[j]
                k_grid = z_grid * v_star ** 0.5 * np.sqrt(T)
                v_ref = Vg @ x_prev[j]
                s_ref = (Vpg @ x_prev[j]) / v_star
                valid = v_ref > 1e-10
                for idx in np.where(valid)[0]:
                    k, vr, sr = k_grid[idx], v_ref[idx], s_ref[idx]
                    d1r = (-k + vr * T / 2) / np.sqrt(vr * T)
                    d2r = (-k - vr * T / 2) / np.sqrt(vr * T)
                    beta0 = (v_star / (2 * vr) * sr ** 2 + np.sqrt(v_star * T) * sr
                             - 2 * (1 + d1r * np.sqrt(v_star / vr) * sr + d1r * d2r * sr ** 2 * v_star / (4 * vr)))
                    beta1 = (v_star / vr * sr + np.sqrt(v_star * T)
                             - 2 * (d1r * np.sqrt(v_star / vr) + d1r * d2r * sr * v_star / (2 * vr)))
                    beta2 = (sr / vr) * (-v_star / (2 * vr) * sr
                                         - 2 * (k / (vr * T)) * (np.sqrt(v_star * T) - (k / 2) * sr * v_star / vr))
                    c_expr = Vppg[idx] @ x_vars[j] / v_star
                    s_expr = Vpg[idx] @ x_vars[j] / v_star
                    v_expr = Vg[idx] @ x_vars[j]
                    constraints.append(c_expr >= beta0 + beta1 * (s_expr - sr) + beta2 * (v_expr - vr))

                # No-butterfly at the edge knots z0, z_{n-1}: linearized (3)-(4).
                knots = bases[j].knots
                k0, kn = knots[0] * e.sigma_star * np.sqrt(T), knots[-1] * e.sigma_star * np.sqrt(T)
                v_ref0 = rows_edges[j][0][0] @ x_prev[j]
                v_refn = rows_edges[j][0][1] @ x_prev[j]
                h = 1e-6 * v_star
                for k_edge, v_ref_edge, side in ((k0, v_ref0, "lo"), (kn, v_refn, "hi")):
                    if v_ref_edge <= h:
                        continue
                    rhs = _edge_butterfly_rhs(v_ref_edge, v_star, T, k_edge)
                    d_rhs = (_edge_butterfly_rhs(v_ref_edge + h, v_star, T, k_edge)
                             - _edge_butterfly_rhs(v_ref_edge - h, v_star, T, k_edge)) / (2 * h)
                    row = 0 if side == "lo" else 1
                    v_expr = rows_edges[j][0][row] @ x_vars[j]
                    s_expr = rows_edges[j][1][row] @ x_vars[j] / v_star
                    linear_rhs = rhs + d_rhs * (v_expr - v_ref_edge)
                    if side == "hi":
                        constraints.append(s_expr <= linear_rhs - lee_margin)
                    else:
                        constraints.append(s_expr >= linear_rhs + lee_margin)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        start = time.time()
        problem.solve(solver=solver)
        iteration_times.append(time.time() - start)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP solve failed at iteration {iteration}: status={problem.status}")
        x_prev = [x.value.copy() for x in x_vars]

    try:
        data = problem.get_problem_data(solver)[0]
        p_nnz = int(data["P"].nnz) if data.get("P") is not None else 0
        a_nnz = int(data["A"].nnz) if data.get("A") is not None else 0
        n_constraints_final = data["A"].shape[0] if data.get("A") is not None else 0
    except Exception:
        p_nnz, a_nnz, n_constraints_final = 0, 0, 0

    chi2_per_expiry = {}
    for j, e in enumerate(expiries):
        if len(e.k_mid) == 0:
            chi2_per_expiry[e.expiry_id] = float("nan")
            continue
        Vm, _, _ = bases[j].eval_rows(e.z(e.k_mid))
        v_fit = Vm @ x_prev[j]
        w = 1.0 / (e.v_ask_mid - e.v_bid_mid) ** 2
        chi2_per_expiry[e.expiry_id] = float(np.sum(w * (v_fit - e.v_mid) ** 2) / max(len(e.k_mid), 1))
    chi2_avg = float(np.nanmean(list(chi2_per_expiry.values())))

    return CVIFitResult(
        expiries=expiries, bases=bases, params=x_prev,
        chi_squared_per_expiry=chi2_per_expiry, chi_squared_avg=chi2_avg,
        solve_time_total=float(np.sum(iteration_times)), iteration_times=iteration_times,
        n_variables=sum(x.size for x in x_vars), n_constraints_final=n_constraints_final,
        p_nnz=p_nnz, a_nnz=a_nnz, n_knots=n_knots,
    )
