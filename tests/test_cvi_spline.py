"""Unit tests for the CVI cubic-spline parameterization (src/cvi_spline.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.cvi_spline import SplineBasis, build_knots


def test_build_knots_monotone_and_atm_at_zero() -> None:
    knots, atm_index = build_knots(12)
    assert len(knots) == 12
    assert np.all(np.diff(knots) > 0)
    assert knots[atm_index] == 0.0


def test_build_knots_rejects_too_few_knots() -> None:
    with pytest.raises(ValueError):
        build_knots(2)


def test_spline_basis_reproduces_parameters_at_atm_knot() -> None:
    """v(0) and dv/dz(0) must equal the first two free parameters exactly."""
    knots, atm_index = build_knots(10)
    basis = SplineBasis(knots, atm_index)
    x = np.array([0.04, -0.01] + [0.1 * (i % 3) for i in range(8)])
    V, Vp, _ = basis.eval_rows(np.array([0.0]))
    assert V[0] @ x == pytest.approx(x[0])
    assert Vp[0] @ x == pytest.approx(x[1])


def test_spline_second_derivative_matches_free_convexities() -> None:
    """The second derivative at interior knot i must equal parameter c_i."""
    knots, atm_index = build_knots(10)
    basis = SplineBasis(knots, atm_index)
    x = np.zeros(10)
    x[3] = 1.0  # convexity parameter at knot index 2 (see convexity_rows mapping)
    _, _, Vpp = basis.eval_rows(np.array([knots[2]]))
    assert Vpp[0] @ x == pytest.approx(1.0, abs=1e-8)


def test_linear_extrapolation_beyond_edge_knots() -> None:
    """Edge convexities are zero, so v(z) is linear for z outside [z_0, z_{n-1}]."""
    knots, atm_index = build_knots(10)
    basis = SplineBasis(knots, atm_index)
    x = np.array([0.04, 0.02] + list(np.random.default_rng(1).normal(0, 0.05, 8)))
    z_out = np.array([knots[0] - 1.0, knots[0] - 2.0, knots[-1] + 1.0, knots[-1] + 2.0])
    V, Vp, _ = basis.eval_rows(z_out)
    # Slope must be constant (equal to the edge slope) in each extrapolated tail.
    assert Vp[0] @ x == pytest.approx(Vp[1] @ x, abs=1e-10)
    assert Vp[2] @ x == pytest.approx(Vp[3] @ x, abs=1e-10)


def test_spline_continuity_across_a_knot() -> None:
    """Value and slope evaluated just left and right of an interior knot agree."""
    knots, atm_index = build_knots(10)
    basis = SplineBasis(knots, atm_index)
    x = np.array([0.04, 0.01] + list(np.random.default_rng(2).normal(0, 0.03, 8)))
    z_k = knots[4]
    eps = 1e-6
    V, Vp, _ = basis.eval_rows(np.array([z_k - eps, z_k + eps]))
    assert V[0] @ x == pytest.approx(V[1] @ x, abs=1e-4)
    assert Vp[0] @ x == pytest.approx(Vp[1] @ x, abs=1e-3)
