# Markov-order windows on LTMLE

This study validates `LTMLE` fitted on `LongitudinalData.with_markov_order(k)`, where every
outcome regression and every treatment and censoring mechanism at node $t$ reads the baseline
covariates $W$ and only the last $k + 1$ nodes of the history. The
[Markov order section](../longitudinal-tmle.md#markov-order) states the option. The order is a
restriction on the observed-data model and does not change the estimand. Under it the windowed
nuisances can be correct, and the estimator is the ordinary LTMLE on correct models, so its
interval is the ordinary one. The row claims validity, not efficiency: the restricted model has
a smaller tangent space, and the windowed fit's influence curve is not shown to be its
efficient one.

R `ltmle` has no window option, so no canonical implementation is compared. Every cell is read
against its exact truth.

## What was fitted

| law | treatment $A_t$ | outcome $Y$ | role |
| --- | --- | --- | --- |
| LF1 | $\operatorname{expit}(-1.0 + 0.4 W - 0.5 L_t + 2.0 A_{t-1} + 1.5 L_{t-1})$ | $0.15 + 0.15 W + 0.15 L_4 + 0.35 L_3 - 0.05 A_4 - 0.05 A_3$ | order 1 is the smallest correct window; the calibration cells and the necessity control |
| LF2 | $\operatorname{expit}(-1.0 + 0.4 W - 0.5 L_t + 2.0 A_{t-1})$ | $0.05 + 0.04 W + 0.9 L_3$ | the mechanisms are correct at order 0 and the outcome window is not; the double-robustness pair |

Both laws draw $W \sim U(0, 1)$ and four nodes of binary $L_t$, $A_t$ and retention $C_t$, with
$L_t \sim \operatorname{Bernoulli}(0.15 + 0.3 W + 0.3 L_{t-1} - 0.1 A_{t-1})$ and retention
$\operatorname{expit}(2.8 + 0.3 W - 0.4 L_t)$. The plans are treatment at all four nodes
(always) and at none (never). Every fit is in sample, with `LinearRegression` outcome and
pseudo-outcome regressions, which contain the linear outcome law, and logistic mechanisms by
Newton, at n = 2,000. The laws, learners and exact limits are in
`tests/studies/_option_markov_law.py`. The exact limits are weighted fits on a quadrature form
of each law, so their bias is in units of the limit fit's plug-in curve SD, and
`tests/unit/test_option_markov_order_design.py` recomputes them.
`tests/unit/test_option_markov_order_witness.py` returns the truth at $10^{-8}$ from those
fits, with mutation controls that shorten the window by one node or start the arm block one
node late.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window); in-sample LTMLE on the order-1 window with linear outcome regressions and logistic mechanisms, n = 2,000 | `ate_regimen[always vs never]` | difference in mean outcome between treatment at all four nodes and at none | `cleverly` | -0.0026 to 0.0018 | 0.9414 | 0.9707 | pass |
| law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window); in-sample LTMLE on the order-1 window with linear outcome regressions and logistic mechanisms, n = 2,000 | `ey_regimen[always]` | mean outcome under treatment at all four nodes | `cleverly` | -0.0021 to 0.000662 | 0.9457 | 0.9742 | pass |
| law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window); in-sample LTMLE on the order-1 window with linear outcome regressions and logistic mechanisms, n = 2,000 | `ey_regimen[never]` | mean outcome under treatment at none of the nodes | `cleverly` | -0.0020 to 0.0013 | 0.9346 | 0.9801 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `order1_ate__correctly_specified` | positive | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, order 1, n = 2,000: every outcome regression and mechanism reads the window and is correctly specified on it | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9326 to 0.9518, SE ratio 0.9514 to 1.0088 | pass |
| `interval_calibration` | `order1_ate__noise_control` | control | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, order 1, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8212 to 0.8515, SE ratio 0.6891 to 0.7289 | pass |
| `interval_calibration` | `order1_ate__shrunken_se_control` | control | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, order 1, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8031 to 0.8347, SE ratio 0.6657 to 0.7058 | pass |
| `interval_calibration` | `order1_ey__correctly_specified` | positive | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), the mean under always, order 1, n = 2,000: every outcome regression and mechanism reads the window and is correctly specified on it | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9364 to 0.9550, SE ratio 0.9713 to 1.0294 | pass |
| `interval_calibration` | `order1_ey__noise_control` | control | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), the mean under always, order 1, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8232 to 0.8534, SE ratio 0.7004 to 0.7396 | pass |
| `interval_calibration` | `order1_ey__shrunken_se_control` | control | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), the mean under always, order 1, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8098 to 0.8409, SE ratio 0.6797 to 0.7209 | pass |
| `interval_calibration` | `order2_ate__correctly_specified` | positive | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, order 2, n = 2,000: every outcome regression and mechanism reads the window and is correctly specified on it | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9299 to 0.9495, SE ratio 0.9451 to 1.0025 | pass |
| `interval_calibration` | `order2_ate__noise_control` | control | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, order 2, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8051 to 0.8366, SE ratio 0.6656 to 0.7079 | pass |
| `interval_calibration` | `order2_ate__shrunken_se_control` | control | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, order 2, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8010 to 0.8327, SE ratio 0.6616 to 0.7010 | pass |
| `markov_reported` | `full_history_ate` | diagnostic | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, n = 2,000: the full-history fit on the calibration draws | none. The row is a diagnostic and states no verdict | bias -0.0021 to 0.0016, coverage 0.9400, SE ratio 0.9689, exact bias 0 SD | reported |
| `markov_reported` | `full_history_ey` | diagnostic | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), the mean under always, n = 2,000: the full-history fit on the calibration draws | none. The row is a diagnostic and states no verdict | bias -0.0015 to 0.000720, coverage 0.9463, SE ratio 0.9949, exact bias 0 SD | reported |
| `markov_reported` | `lf2_order0_ate` | diagnostic | law LF2 (four binary treatment nodes with censoring; treatment reads only A_{t-1}, the outcome reads L_3), always against never, n = 2,000: the order-0 fit with correct mechanisms | none. The row is a diagnostic and states no verdict | bias -0.0025 to 0.0027, coverage 0.9563, SE ratio 1.0379, exact bias 0 SD | reported |
| `markov_reported` | `lf2_order0_ey` | diagnostic | law LF2 (four binary treatment nodes with censoring; treatment reads only A_{t-1}, the outcome reads L_3), the mean under always, n = 2,000: the order-0 fit with correct mechanisms | none. The row is a diagnostic and states no verdict | bias -0.0021 to 0.0024, coverage 0.9437, SE ratio 0.9912, exact bias 0 SD | reported |
| `markov_reported` | `order0_ey` | diagnostic | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), the mean under always, n = 2,000: the order-0 fit of the necessity cell | none. The row is a diagnostic and states no verdict | bias 0.0111 to 0.0149, coverage 0.9319, SE ratio 1.0057, exact bias 0.4566 SD | reported |
| `order_double_robustness` | `ey_never` | positive | law LF2 (four binary treatment nodes with censoring; treatment reads only A_{t-1}, the outcome reads L_3), the mean under never, n = 2,000: order 0, the outcome window wrong and every mechanism correct | bias interval inside the equivalence margin | bias -0.0013 to 0.0015, margin 0.0055 | pass |
| `order_double_robustness` | `ey_never_blind_mechanisms` | control | law LF2 (four binary treatment nodes with censoring; treatment reads only A_{t-1}, the outcome reads L_3), the mean under never, n = 2,000: order 0 with intercept-only treatment and censoring mechanisms, so no fit removes the outcome error | the 99% interval of the standardized bias must lie wholly beyond the margin | bias 0.0264 to 0.0293, margin 0.0058 | pass |
| `order_necessity` | `ate` | control | law LF1 (four binary treatment nodes with censoring; treatment reads L_{t-1}, the outcome reads L_3, so order 1 is the smallest correct window), always against never, n = 2,000: order 0, which drops L_{t-1} from the treatment mechanism and L_3 from the outcome window | the 99% interval of the standardized bias must lie wholly beyond the margin | bias 0.0541 to 0.0597, margin 0.0108 | pass |
<!-- /generated -->

The three calibration cells read the same 4,000 draws of LF1 at order 1 and order 2. The
necessity control fits order 0 on LF1, which drops $L_{t-1}$ from the treatment mechanism and
$L_3$ from the outcome window: both sides are wrong and the ATE misses by 1.32 SD (exact 1.37).
The double-robustness pair reads the mean under never on LF2 at order 0. With correct
mechanisms the outcome window's error is removed, and with intercept-only mechanisms it stays,
at 1.20 empirical SDs against an exact 0.54 plug-in SDs. The intercept-only fit's plug-in
standard error is about 2.2 times its spread, which is why the two units differ.

## Reported readings

These rows carry no verdict. The full-history fit on the calibration draws reads coverage
0.946 for the mean under always and 0.940 for the ATE, beside the windowed fits' 0.946 and
0.943. The order-0 fit's mean under always on LF1 reads 0.446 SD against the exact 0.457, and the
LF2 order-0 rows with correct mechanisms are unbiased.

## Sizing

Every size was fixed before the run from the pilot in
`tests/studies/option_markov_order_pilot.py`, on a stream no declared cell reads. The
calibration count, 4,000, is the arm-pooled predictive rule; the product floor is 2,800. The
necessity and double-robustness cells run 1,600 and the primary phase 2,800. The design test
recomputes each count. An earlier LF1 with weaker treatment persistence was discarded before the
declaration: its heavy-tailed cumulative ratio left both the windowed and the full-history fit
covering 0.919.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/option_markov_order/regenerate.py`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2800 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 3 | truth tests passing |
| `independent_tests_total` | 3 | truth tests reported |
| `property_cells_passed` | 12 | gated property cells passing |
| `property_cells_total` | 12 | gated property cells reported |
| `max_standardized_bias` | 0.0253 | largest primary standardized bias |
| `min_coverage` | 0.9346 | lowest primary coverage |
| `properties[interval_calibration/order1_ey__correctly_specified]:coverage` | 0.9463 | coverage, order 1, mean under always |
| `properties[interval_calibration/order1_ate__correctly_specified]:coverage` | 0.9427 | coverage, order 1, ATE |
| `properties[interval_calibration/order2_ate__correctly_specified]:coverage` | 0.9403 | coverage, order 2, ATE |
| `properties[interval_calibration/order2_ate__correctly_specified]:se_ratio` | 0.9725 | SE ratio, order 2, ATE |
| `properties[order_necessity/ate]:standardized_bias` | 1.3172 | standardized bias of the order-0 control on LF1 |
| `properties[order_necessity/ate]:exact_bias_over_sd` | 1.3664 | its exact-law limit |
| `properties[order_double_robustness/ey_never_blind_mechanisms]:standardized_bias` | 1.2000 | the order-0 fit with intercept-only mechanisms |
| `properties[order_double_robustness/ey_never_blind_mechanisms]:se_ratio` | 2.2262 | its plug-in SE over its spread |
| `properties[markov_reported/full_history_ey]:coverage` | 0.9463 | the full-history fit, mean under always |
| `properties[markov_reported/full_history_ate]:coverage` | 0.9400 | the full-history fit, ATE |
| `properties[markov_reported/order0_ey]:standardized_bias` | 0.4461 | the order-0 fit on LF1, mean under always |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 28 | the same bound, in primary replications |

## Limitations

- **The restriction is the whole claim.** The interval holds when every regression and
  mechanism depends on the history only through $W$ and the window. The data can test the
  restriction, and the fit does not.
- **Validity, not efficiency.** No cell gates an efficiency band, and the row makes no
  efficiency claim for the windowed fit.
- **Orders 1 and 2 on four binary nodes.** The cells measure one law family with censoring, in
  sample, with linear outcome regressions and logistic mechanisms. Other node counts, flexible
  learners, cross-fitting, SDR, g-computation, IPW, IPSI and survival outcomes with a window
  read experimental.
- **Static plans only.** The cells read always and never; dynamic and stochastic plans with a
  window read experimental.
- **No comparator.** R `ltmle` has no window option.

## Reproduction

`tests/canonical/option_markov_order/regenerate.py` regenerates the study and states the run
form. It runs no reference process, because the study has no comparator. The manifest records
the seeds, the margins, the estimator configuration and the source and result hashes, and the
run log records the commit, the clean tree and the wall time.
