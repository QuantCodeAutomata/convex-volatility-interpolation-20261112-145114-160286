"""CVI cubic-spline parameterization (paper, "The CVI Parameterization").

For each expiry, the variance v(z) is a cubic spline in normalized
log-moneyness z, specified by n + 2 numbers: v(0) (ATM variance), dv/dz(0)
(ATM skew), and the n second derivatives (convexities) at the knots
z_0 < ... < z_{n-1}, with z_l = 0 at one knot. Linear extrapolation beyond
the edge knots forces the edge convexities to zero, leaving n free
parameters per expiry: x = [v(0), dv/dz(0), M_1, ..., M_{n-2}] (paper,
"there are n free parameters per expiry").

A standard natural-cubic-spline recursion (second derivatives piecewise
linear between knots) gives the knot values and slopes, and therefore the
spline itself, as an exact *linear* function of x. This is the linear
transformation the paper refers to when it says the cubic-spline
parameters and the n + 2 B-spline weights "are linked by a known linear
transformation and can thus be used interchangeably"; this implementation
works directly in cubic-spline parameter space, which is linearly
equivalent, instead of separately building a B-spline basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import KNOT_DENSITY, Z_MAX, Z_MIN


def build_knots(n_knots: int, z_min: float = Z_MIN, z_max: float = Z_MAX,
                 density: float = KNOT_DENSITY) -> tuple[np.ndarray, int]:
    """Knots z_0 < ... < z_{n-1} with one knot exactly at zero (ATM knot).

    Knots are placed with a sinh transform of evenly spaced points, which
    increases knot density near the money (density > 0) as the paper
    recommends, without specifying an exact placement rule.
    """
    if n_knots < 3:
        raise ValueError("at least 3 knots are required (2 edges + ATM)")
    n_neg = n_knots // 2
    n_pos = n_knots - 1 - n_neg

    def sinh_side(count: int, limit: float) -> np.ndarray:
        if count == 0:
            return np.array([])
        t = np.linspace(1.0 / count, 1.0, count)
        return limit * np.sinh(density * t) / np.sinh(density)

    z_neg = -sinh_side(n_neg, -z_min)[::-1]
    z_pos = sinh_side(n_pos, z_max)
    knots = np.concatenate([z_neg, [0.0], z_pos])
    assert np.all(np.diff(knots) > 0), "knots must be strictly increasing"
    assert knots[n_neg] == 0.0
    return knots, n_neg


@dataclass
class SplineBasis:
    """Linear map from the n free CVI parameters to the spline and its
    first two derivatives, evaluated at arbitrary points z."""

    knots: np.ndarray
    atm_index: int

    def __post_init__(self) -> None:
        n = len(self.knots)
        self.n = n
        # Row i of `value_rows`/`slope_rows` holds the length-n coefficient
        # vector expressing the knot value y_i / slope y'_i as a linear
        # function of x = [v0, s0, c_1, ..., c_{n-2}].
        value_rows = np.zeros((n, n))
        slope_rows = np.zeros((n, n))
        convexity_rows = np.zeros((n, n))  # M_i rows; zero at the edges
        for i in range(1, n - 1):
            convexity_rows[i, 2 + (i - 1)] = 1.0

        l = self.atm_index
        value_rows[l, 0] = 1.0
        slope_rows[l, 1] = 1.0

        for i in range(l, n - 1):
            h = self.knots[i + 1] - self.knots[i]
            value_rows[i + 1] = (
                value_rows[i] + h * slope_rows[i]
                + h ** 2 / 6.0 * (2 * convexity_rows[i] + convexity_rows[i + 1])
            )
            slope_rows[i + 1] = slope_rows[i] + h / 2.0 * (convexity_rows[i] + convexity_rows[i + 1])

        for i in range(l, 0, -1):
            h = self.knots[i] - self.knots[i - 1]
            slope_rows[i - 1] = slope_rows[i] - h / 2.0 * (convexity_rows[i - 1] + convexity_rows[i])
            value_rows[i - 1] = (
                value_rows[i] - h * slope_rows[i - 1]
                - h ** 2 / 6.0 * (2 * convexity_rows[i - 1] + convexity_rows[i])
            )

        self.value_rows = value_rows
        self.slope_rows = slope_rows
        self.convexity_rows = convexity_rows

    def eval_rows(self, z_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coefficient rows for v(z), dv/dz(z), d2v/dz2(z) at each z in z_query.

        Returns three (len(z_query), n) arrays V, Vp, Vpp such that, for
        parameter vector x, v(z_query) = V @ x, etc.
        """
        z_query = np.atleast_1d(np.asarray(z_query, dtype=float))
        n = self.n
        V = np.zeros((len(z_query), n))
        Vp = np.zeros((len(z_query), n))
        Vpp = np.zeros((len(z_query), n))
        idx = np.clip(np.searchsorted(self.knots, z_query, side="right") - 1, 0, n - 2)
        for row, (z, i) in enumerate(zip(z_query, idx)):
            if z < self.knots[0]:
                V[row] = self.value_rows[0] + (z - self.knots[0]) * self.slope_rows[0]
                Vp[row] = self.slope_rows[0]
                continue
            if z > self.knots[-1]:
                V[row] = self.value_rows[-1] + (z - self.knots[-1]) * self.slope_rows[-1]
                Vp[row] = self.slope_rows[-1]
                continue
            h = self.knots[i + 1] - self.knots[i]
            a = self.knots[i + 1] - z
            b = z - self.knots[i]
            Mi, Mi1 = self.convexity_rows[i], self.convexity_rows[i + 1]
            yi, yi1 = self.value_rows[i], self.value_rows[i + 1]
            V[row] = (Mi * a ** 3 + Mi1 * b ** 3) / (6 * h) + (yi / h - Mi * h / 6) * a + (yi1 / h - Mi1 * h / 6) * b
            Vp[row] = (-Mi * a ** 2 + Mi1 * b ** 2) / (2 * h) - yi / h + Mi * h / 6 + yi1 / h - Mi1 * h / 6
            Vpp[row] = (Mi * a + Mi1 * b) / h
        return V, Vp, Vpp
