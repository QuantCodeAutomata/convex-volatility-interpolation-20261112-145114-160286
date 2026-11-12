"""Parameters for the CVI (Convex Volatility Interpolation) replication.

Every module-level constant below is used by src/ and exp/ exactly as
written here; nothing is overridden at the call site. Each constant is
commented with its meaning, unit, and the paper section/equation it comes
from, or "not specified in the paper" when the paper leaves it open.
"""

# ---------------------------------------------------------------------------
# CVI spline / objective (paper "The CVI Parameterization", "Objective Function")
# ---------------------------------------------------------------------------

# Strike-regularization weight lambda in the total-variation penalty on the
# normalized convexities (paper, "Strike Regularization Penalty"): the paper
# states "lambda = 0.05 is a sensible choice across underlyings".
STRIKE_REG_LAMBDA = 0.05

# Number of discretized strikes used for the constraints that are inherently
# discrete (positivity, no-calendar-spread); the paper's Numerical Results
# section states "discretized constraints are applied across 20 strikes".
N_DISCRETIZATION_STRIKES = 20

# Margin used to enforce the strict inequalities of Lee's tail slope bound
# s < 2/sqrt(v_star*T) (resp. s > -2/sqrt(v_star*T)) and the quadratic edge
# no-butterfly bounds (3)-(4) numerically. The paper states the bounds are
# "strict" and "enforced numerically with a small margin" without giving a
# number; 1e-3 (in units of the dimensionless skew s) is assumed here.
LEE_MARGIN = 1e-3

# Number of calibration iterations: iteration 0 solves without the linearized
# no-butterfly-arbitrage constraints; iterations 1..N_BUTTERFLY_ITERATIONS-1
# add those constraints linearized around the previous solution. The paper
# states "one or a few iterations ... are sufficient in practice" without an
# exact stopping rule; 2 total iterations (one linearization step) is used
# as the default two-step calibration described in "Putting It All Together".
N_BUTTERFLY_ITERATIONS = 2

# Normalized log-moneyness range [Z_MIN, Z_MAX] spanned by the knots z_0..z_
# {n-1}. The paper requires this range to be "sufficiently wide to include
# all the listed strikes" and far enough into the tails that d1*d2 > 1 at the
# edge knots (paper, discussion after Eq. (4)); +/-6 is wide enough for the
# volatility levels and maturities used here (checked numerically in
# src/cvi_calibration.py). Not a numeric value given in the paper.
Z_MIN = -6.0
Z_MAX = 6.0

# Knot-density control: knots are placed at z_i = Z_MAX * sinh(KNOT_DENSITY *
# u_i) / sinh(KNOT_DENSITY) for u_i evenly spaced in [-1, 1], which increases
# knot density near the money as the paper recommends ("increasing knot
# density near-the-money better captures variations in variance convexity")
# without specifying an exact placement rule.
KNOT_DENSITY = 1.6

# Default number of knots per expiry used for exp_1's single-surface fit.
DEFAULT_N_KNOTS = 20

# QP solver used through CVXPY (paper, "Implementation" / Goulart and Chen 2024).
QP_SOLVER = "CLARABEL"

# ---------------------------------------------------------------------------
# exp_1: CVI volatility surface calibration (single underlying)
# ---------------------------------------------------------------------------

EXP1_UNDERLYING = "I:SPX"  # Massive index-options root ticker used for the fit
EXP1_N_KNOTS = 20          # paper's "SPX with 20 knots" reference configuration

# ---------------------------------------------------------------------------
# exp_2: calibration-time benchmarking
# ---------------------------------------------------------------------------

EXP2_KNOT_COUNTS_PAPER = [10, 20, 40]  # paper's knot counts for the timing table
EXP2_UNDERLYINGS = ["I:SPX", "I:RUT"]  # SX5E (Euro Stoxx 50) has no Massive
# coverage; the Russell 2000 index (I:RUT) is substituted as the second,
# less liquid index with a comparable number of expiries, see RESULTS.md.

# Maximum number of expiries fetched per underlying (paper benchmark) and the
# reduced value actually downloaded to keep the full option-chain pull within
# the time available for this replication (see RESULTS.md).
EXP2_MAX_EXPIRIES_PAPER = 46
EXP2_MAX_EXPIRIES_RUN = 12
EXP2_STRIKES_PER_EXPIRY_RUN = 24

# ---------------------------------------------------------------------------
# exp_3: chi-squared fit quality vs number of knots
# ---------------------------------------------------------------------------

EXP3_KNOT_RANGE_PAPER = list(range(5, 36))  # paper: knots per expiry from 5 to 35
EXP3_KNOT_RANGE_RUN = list(range(5, 36, 2))  # every other knot count, see RESULTS.md

# Underlyings in the paper's chi-squared study: 4 indices + 3 single names.
# SX5E and KOSPI200 have no coverage on Massive REST (US-centric options
# data); they are dropped and replaced by no additional index (see
# RESULTS.md, "Deviations from the paper").
EXP3_UNDERLYINGS_PAPER = ["I:SPX", "I:NDX", "I:SX5E", "I:KOSPI2", "AMZN", "MSFT", "NVDA"]
EXP3_UNDERLYINGS_RUN = ["I:SPX", "I:NDX", "I:RUT", "AMZN", "MSFT", "NVDA"]
EXP3_MAX_EXPIRIES_RUN = 8
EXP3_STRIKES_PER_EXPIRY_RUN = 20

# ---------------------------------------------------------------------------
# Market-data retrieval (data/data_loader.py)
# ---------------------------------------------------------------------------

MASSIVE_API_ROOT = "https://api.massive.com"

# Flat-rate discount proxy used to turn an option's mid/bid/ask price into an
# implied volatility via Black-76 on the forward. The paper does not discuss
# discounting explicitly; a small constant short-term rate is assumed here
# (not specified in the paper).
DISCOUNT_RATE = 0.04
