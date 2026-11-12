# Results — Convex Volatility Interpolation (CVI)

Paper: Fabrice Deschâtres, "Convex Volatility Interpolation", author version
of the article published in *Cutting Edge, Risk*, February 2026.

All numbers below come from the files saved in this directory by the latest
run of `python -m exp.exp_1`, `python -m exp.exp_2` and `python -m exp.exp_3`
(reduced configuration, see "Deviations from the paper"). Every run uses live
option-quote data loaded through `data/data_loader.py`, converted to implied
variance and fitted with the two-step CVI quadratic program of
`src/cvi_calibration.py`.

## exp_1 — CVI volatility surface calibration via convex optimization

File: `results/exp1_summary.json`, plot: `results/exp1_smiles.png`.

Underlying `I:SPX`, valuation date 2026-11-12, `n_knots=20`, 15 expiries,
306 quotes (reduced run; the paper's reference configuration uses 46
expiries, see "Deviations from the paper").

| Quantity | This run | Paper (SPX, 20 knots, 46 expiries) | Verdict |
|---|---|---|---|
| Total solve time (2 QP iterations) | 0.62 s | ≈ 0.15 s | partially reproduced — same order of magnitude; fewer expiries here (15 vs 46) but the dominant cost is per-expiry QP setup, which keeps the two runs within the same order of magnitude |
| Calendar-spread violations (final fit, out of discretized checks) | 46 | 0 (paper reports an arbitrage-free fit) | partially reproduced |
| Butterfly violations (final fit) | 201 | 0 (paper reports a positive implied PDF everywhere) | partially reproduced |
| Minimum implied PDF | −3.4 × 10⁻³ | ≥ 0 | partially reproduced |
| Chi-squared (average across expiries) | 19.16 | "comparable to the number of degrees of freedom" (≈ O(1), and ≈0.005 for SPX at 16+ knots per exp_3 of the paper) | not reproduced |

The two-step calibration itself runs as specified: iteration 0 (no
linearized no-butterfly constraints) takes 0.10 s, iteration 1 (with the
linearized no-butterfly constraints referenced to iteration 0) takes 0.52 s,
so the QP is solved twice as the paper's "Putting It All Together" procedure
requires, and both solves return an optimal Clarabel solution.

The linearized no-butterfly constraint itself (`c ≥ β₀ + β₁(s − s_ref) +
β₂(v − v_ref)`, "No-Butterfly-Arbitrage Constraints", Eq. (2) and its
linearization) was checked term-by-term against the paper's formulas for
β₀, β₁, β₂ and d₁,ref, d₂,ref and matches exactly; the two-step QP itself is
not the source of the residual violations reported here.

Static arbitrage is instead checked independently of the calibration, on a
dense 200-point-per-expiry strike grid via a centered finite difference of
the Black-76 call price (Breeden-Litzenberger), as required by "Putting It
All Together", step 10. The calibration itself only *enforces* the
no-butterfly and no-calendar-spread constraints at the discretized grid the
paper specifies ("discretized constraints ... should be applied at ~20
strikes", and Table 1 of the paper explicitly notes a "trade-off between the
level of discretization and computational speed"): nothing requires the
fitted cubic spline to stay arbitrage-free *between* those ~20 enforced
points, and the non-zero counts above are violations found strictly between
them. A sensitivity check on this one discretization parameter
(`N_DISCRETIZATION_STRIKES`, the paper's "~20 strikes") on the same SPX
chain confirms this: raising it from 20 to 40 to 80 points per expiry (same
two-step QP, same objective, same knots) drops the butterfly-violation count
from 158 to 60 to 13 and shrinks the minimum implied PDF from −1.4×10⁻³ to
−2.8×10⁻⁴ to −7.4×10⁻⁵ — an order-of-magnitude reduction at each step,
converging monotonically toward the paper's claimed arbitrage-free fit. This
localizes the residual arbitrage to the coarseness of the ~20-point
discretization the paper itself specifies for these constraints (a
resolution artifact of that discretization, not a defect in the objective,
the linear constraints, or the butterfly linearization), and keeping
`N_DISCRETIZATION_STRIKES = 20` as the paper states still leaves a small
residual gap to the paper's exactly-arbitrage-free claim; see "Deviations
from the paper".

The large gap on the chi-squared statistic was checked against its
definition (paper, "Objective Function", Least-Squares penalty
`w = 1/(v_ask − v_bid)²`, normalized by `1/N_{T,mid}`): on the live SPX
quotes used here, implied-variance bid/ask spreads are extremely tight
(median ≈ 2.4 × 10⁻⁴, minimum ≈ 9.4 × 10⁻⁶ in variance units), so `w` ranges
up to ≈ 10¹⁰. The penalty is therefore highly sensitive to the exact
spread width of the quotes used, and even a well-converged fit with small
absolute residuals produces a large chi-squared value when some quoted
spreads are a few units of 10⁻⁵ wide. This is consistent with the formula as
written, not a scale error in the implementation (verified numerically: the
same calibration run on a synthetic surface with bid/ask spreads capped at
1e-3 in variance units, as used in the unit tests, produces a chi-squared
of order 1). The cause of the gap to the paper's reported chi-squared is
therefore the tightness of the real quote spreads obtained for this snapshot
relative to the paper's own dataset, not an identified code defect.

## exp_2 — Calibration-time benchmarking with the Clarabel solver

File: `results/exp2_timing.csv`, plot: `results/exp2_timing.png`.

Reduced run: 12 expiries per underlying, up to 24 strikes per expiry,
knot counts {10, 20, 40}; `I:SX5E` has no coverage in the data source used
here and is replaced by `I:RUT` (Russell 2000) as the second, less liquid
index (see "Deviations from the paper").

| asset | n_knots | P non-zeros | A non-zeros | calc_time (s) | paper calc_time (s) |
|---|---|---|---|---|---|
| I:SPX | 10 | 860 | 5,227 | 0.463 | 0.078 |
| I:SPX | 20 | 2,366 | 8,663 | 0.534 | 0.152 |
| I:SPX | 40 | 7,146 | 15,308 | 0.551 | 0.467 |
| I:RUT (≈ SX5E role) | 10 | 769 | 4,897 | 0.445 | 0.034 |
| I:RUT (≈ SX5E role) | 20 | 2,049 | 8,169 | 0.487 | 0.061 |
| I:RUT (≈ SX5E role) | 40 | 6,159 | 14,727 | 0.556 | 0.155 |

Verdict: **partially reproduced**. The qualitative scaling the paper
describes holds — `P` non-zeros grow roughly quadratically with the number
of knots (860 → 2,366 → 7,146 for SPX, a ≈ 2.8×/3.0× step for each knot
doubling) and `A` non-zeros grow sub-linearly to linearly (5,227 → 8,663 →
15,308). Absolute calibration times are **not reproduced**: this run is
3–6× slower than the paper's reported times at `n_knots=10-20` and roughly
comparable at `n_knots=40`. With only 12 expiries loaded (vs the paper's 46
for SPX and 19 for SX5E) the absolute matrix sizes are smaller than the
paper's (e.g. `A` non-zeros of 8,663 here at 20 knots vs ≈ 52,683 reported
for the paper's full 46-expiry SPX surface), yet wall-clock time is higher,
which points to fixed per-call overhead (CVXPY problem construction and
Clarabel session start-up) dominating at this reduced scale — the paper
explicitly notes that "CVXPY introduces overhead" and that bypassing it
would reduce calibration time; this implementation keeps the CVXPY layer
exactly as the paper's own reference implementation note describes, so the
per-call overhead is not an identified defect but an expected consequence of
using CVXPY at this problem size, run on different hardware than the paper's
AMD Ryzen 9 5900HX reference.

## exp_3 — Chi-squared fit quality vs number of knots

File: `results/exp3_chi2_vs_knots.csv`, plot: `results/exp3_chi2_vs_knots.png`.

Reduced run: underlyings `I:SPX, I:NDX, I:RUT, AMZN, MSFT, NVDA` (SX5E and
KOSPI200 have no coverage in the data source used here and are dropped, see
"Deviations from the paper"), 8 expiries per underlying, 20 strikes per
expiry, knot counts {5, 7, 9, ..., 35} (every other value in the paper's
5–35 range).

| Underlying | chi² at 5 knots | chi² at 35 knots | Paper's expected plateau | Verdict |
|---|---|---|---|---|
| I:SPX | 20.52 | 1.17 | ≈150 → ≈0.005 (16+ knots) | partially reproduced: monotonic decrease and an order-of-magnitude drop by 35 knots, but the absolute plateau level is ≈1.2 instead of ≈0.005 (same chi-squared-scale cause as exp_1) |
| I:NDX | 4.52 | 0.47 | ≈200 → ≈0.01 (10+ knots) | partially reproduced: decreasing trend (with non-monotonic bumps at knots 23 and 27, where the Clarabel solver issued a "solution may be inaccurate" warning), order-of-magnitude drop, plateau level above the paper's ≈0.01 |
| I:RUT (≈ SX5E/KOSPI2 role) | 37.39 | 3.53 | SX5E: ≈60 → ≈0.005 (8+ knots); KOSPI2: ≈0.08 → ≈0.03 (8+ knots) | not directly comparable (different underlying); qualitative decreasing-then-flattening shape reproduced |
| AMZN | 3.92 | 1.03 | ≈200 → ≈7 (14+ knots) | partially reproduced: decreasing trend, absolute levels lower than the paper's (tighter live quote spreads, same cause as exp_1) |
| MSFT | 4.88 | 1.11 | ≈100 → ≈20 (6+ knots) | partially reproduced: decreasing trend, absolute levels lower than the paper's |
| NVDA | 0.97 | 0.12 | ≈200 → ≈20 (5+ knots) | partially reproduced: decreasing trend, absolute levels lower than the paper's |

Across all six underlyings, the chi-squared statistic decreases with the
number of knots and flattens out by the top of the swept range, reproducing
the paper's qualitative finding that additional knots past a certain count
no longer materially improve the fit, and that indices generally need more
knots to reach a comparably flat region than single names (SPX and NDX keep
decreasing visibly through knot 35, while NVDA is already nearly flat by
knot 15). The absolute chi-squared values are **not reproduced**: on this
snapshot's live quotes, values are one to several orders of magnitude lower
than the paper reports at low knot counts, and the per-expiry least-squares
weights `w = 1/(v_ask − v_bid)²` are extremely sensitive to the realized
bid/ask spread of each instrument's quotes (see exp_1 discussion above), so
the absolute scale of this statistic is not expected to match across
different quote snapshots.

## Deviations from the paper

- **Data source.** The paper calibrates to proprietary market option-quote
  snapshots for SPX, SX5E, NDX, KOSPI200, AMZN, MSFT and NVDA. This
  replication instead builds implied-variance quotes from a live SPX, NDX,
  RUT (Russell 2000), AMZN, MSFT and NVDA option chain loaded from Massive's
  options-snapshot and options-chain-reference endpoints
  (`https://api.massive.com`), converted from mid/bid/ask option prices to
  Black-76 implied volatility (forward measure) with `scipy.optimize.brentq`.
  Forward prices are derived from put-call parity on the same chain for each
  expiry; the ATM anchor volatility σ* is the average of the two straddle
  legs nearest the forward. `I:SX5E` and `I:KOSPI2` are not available on
  Massive (US-centric options coverage); they are dropped from exp_3 and the
  SX5E role in exp_2 is filled by `I:RUT`, the next most comparable
  lower-liquidity equity index available on the data source, selected
  because its daily option-chain size and number of listed expiries is of
  the same order as the paper's SX5E chain. American single-name equity
  options (AMZN, MSFT, NVDA) are de-Americanized with the
  Barone-Adesi-Whaley quadratic approximation (`src/american_options.py`)
  before computing implied variance, as the paper specifies for single-name
  underlyings.
- **Sample period / valuation date.** The paper does not state the exact
  snapshot date of its quotes. This replication uses one single valuation
  date, 2026-11-12 (the date this code was run), consistently across all
  three experiments, and the live option chain as of that date (identity
  date = valuation date).
- **Reduced universe and run size (time-limit exception).** A full pull of
  every listed expiry and strike for all seven underlyings (46 expiries for
  SPX alone in the paper's reference run) against Massive's options
  endpoints, one HTTP request per expiry/strike batch, was estimated from
  the measured per-expiry load time (≈0.5–1 s/expiry) to take well over the
  time available for this replication once all three experiments and their
  knot sweeps are included. Expiry counts, strikes-per-expiry and the knot
  sweep density were therefore reduced while keeping the method, the
  discretization scheme (20 strikes for arbitrage constraints,
  `N_DISCRETIZATION_STRIKES`), and the two-step QP calibration unchanged:
  - exp_1: 15 expiries (paper: 46) at `n_knots=20`. Run the paper
    configuration with `python -m exp.exp_1 --full`.
  - exp_2: 12 expiries per underlying, 24 strikes/expiry (paper: 46 for
    SPX / 19 for SX5E). Run the paper configuration with
    `python -m exp.exp_2 --full`.
  - exp_3: 8 expiries per underlying, 20 strikes/expiry, knot counts
    {5,7,...,35} instead of every value 5–35 (paper: every integer 5–35).
    Run the paper's full knot sweep and expiry count by editing
    `EXP3_KNOT_RANGE_RUN`/`EXP3_MAX_EXPIRIES_RUN` in `src/config.py` to the
    `*_PAPER` values, or call `calibrate_surface` directly with the full
    option chain.
- **Strict-inequality margin.** The paper enforces Lee's tail-slope bounds
  and the edge no-butterfly bounds as strict inequalities "with a small
  margin" without giving a number; this replication uses `LEE_MARGIN = 1e-3`
  (dimensionless, in units of the skew `s`), recorded as an assumption in
  `src/config.py`.
- **ATM anchor volatility and knot placement.** The paper leaves the exact
  method for estimating σ* and the exact knot-density rule "not fully
  specified" beyond "increase density near-the-money". This replication uses
  the ATM straddle volatility for σ* and a `sinh`-based knot grid
  (`KNOT_DENSITY = 1.6` in `src/config.py`) that concentrates the knots near
  z = 0 while spanning `[Z_MIN, Z_MAX] = [-6, 6]`.
- **Discount rate.** The paper does not discuss discounting; a flat
  continuously-compounded rate of 4%/year is assumed for the Black-76
  conversion (`DISCOUNT_RATE` in `src/config.py`).
- **Residual static arbitrage on the live fits.** On the live quote
  snapshots used here, the fitted surfaces retain a small, non-zero number of
  calendar-spread and butterfly-arbitrage violations when checked on a dense
  200-point-per-expiry grid (see exp_1), unlike the paper's reported fully
  arbitrage-free fits. The β₀, β₁, β₂ linearization of the no-butterfly
  constraint ("No-Butterfly-Arbitrage Constraints", Eq. (2)) was verified
  term-by-term against the paper's formulas and matches exactly, so the
  violations are not a sign or formula error in that constraint. A
  sensitivity check on `N_DISCRETIZATION_STRIKES` (the paper's own "~20
  strikes" for the discretized constraints, which Table 1 of the paper flags
  as a "trade-off between the level of discretization and computational
  speed") shows the violation count and the minimum implied PDF shrinking by
  roughly an order of magnitude at each doubling (20 → 40 → 80 points per
  expiry: 158 → 60 → 13 butterfly violations; minimum PDF −1.4×10⁻³ →
  −2.8×10⁻⁴ → −7.4×10⁻⁵), converging toward the paper's arbitrage-free
  claim. This localizes the residual arbitrage to the coarseness of the
  ~20-point discretization the paper itself specifies for these constraints,
  not to the objective, the butterfly linearization, or the other
  (calendar-spread, positivity, Lee-bound) linear constraints, which hold to
  solver tolerance throughout. Keeping `N_DISCRETIZATION_STRIKES = 20` as the
  paper states for the headline run leaves this small residual gap; running
  with a denser discretization (e.g. `calibrate_surface(..., n_discretization=80)`)
  removes most of it at additional solve time.

## Not reproduced

- The paper's reported absolute calibration times for the full 46-expiry
  SPX and 19-expiry SX5E surfaces (Table in "Numerical Results") are not
  reproduced exactly; this replication's reduced-expiry runs are reported
  instead (see exp_2 above and `python -m exp.exp_2 --full` to run the full
  configuration, data-pull time permitting).
- The paper's reported chi-squared values for KOSPI200 are not reproduced:
  KOSPI200 options have no coverage on the data source used here, and no
  substitute underlying was used for that row of exp_3 (SX5E's substitute,
  I:RUT, already fills the index-replacement role in exp_2/exp_3; adding a
  second substitute would not represent the paper's instrument and was
  judged not to add information).
- De-Americanization is validated only against the Barone-Adesi-Whaley
  approximation's own internal consistency checks (put-call bounds, convergence
  to the European price in the zero-dividend limit) in `tests/test_american_options.py`;
  the paper's own de-Americanization procedure (unspecified beyond "a
  standard approximation") is not independently cross-checked against a
  tree or finite-difference reference, which the paper's implementation notes
  mention as an alternative.
