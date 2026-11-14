# Censored continuous-treatment modified policies

This study validates censoring at the nodes of a longitudinal modified treatment policy on a
continuous treatment. `LTMLE` and `SDR` fit the policy on the classifier ratio route with a
censoring model learned beside the ratio. The study law is the pricing law of
[`mtp-pricing`](mtp-pricing.md) at three nodes with dropout after each price, so "price" names the
law and not a limit of the method. The
[modified treatment policies at a node](../longitudinal-tmle.md#modified-treatment-policies-at-a-node)
section states the estimator and its conditions.

The evidence label is **Python property evidence, with comparator runs pending**. The study
declares R `lmtp` 1.5.4 as its comparator. Its runs have not registered, so this page publishes no
paired verdict. [Comparator runs pending](../../development/method-benchmarking.md#comparator-runs-pending)
states the rule. The publication policy is `reporting`, and every gated cell passes.

## What was compared

The estimand is the mean outcome when the treatment follows the plan at every node and nobody is
lost to follow-up. Díaz, Williams, Hoffman and Schenck (2023), arXiv v4, Section 3 (the paragraph
after Theorem 1) identify it by Theorem 1 with the censoring component of each intervention set to
one. The joint ratio of node $t$ is then the policy's treatment ratio times
$1\{C_t = 1\} / P(C_t = 1 \mid A_t, H_t)$, evaluated at the **observed** treatment. `cleverly`
estimates the two factors separately: the classifier gives the policy ratio, and the censoring
model gives the retention probability in the node's denominator.

Law P-C is law P of `mtp-pricing` at three nodes. After each price $A_t$ a retention indicator is
drawn:

$$C_t \sim \text{Bernoulli}\bigl(\text{expit}(2.8 + 0.5 W_1 + 0.8 L_t - 0.6 (A_t - c_t))\bigr), \quad c_t = 6 + 2 L_t.$$

Dropout is monotone. Retention rises with engagement and falls with the price gap, so the
customers who stay are the likelier buyers, and the dropout is informative given the history. The
draw changes no structural equation of law P, so every truth is law P's
(`tests/studies/mtp_pricing_common.py`). Per-node retention is at least 0.58 on the support (its minimum is 0.587), and
about 14% of units are censored by the third node. `tests/unit/test_mtp_pricing_censored_design.py`
checks the truth, the retention bound and that the censoring draw leaves law P's columns unchanged.

| scenario | fit | what a red row would mean |
| --- | --- | --- |
| `censored_ltmle_tau3` | `LTMLE`, plans `natural_prices` and `price_up`, one fold, the exact price ratio on the classifier route, the primary outcome learners of `mtp-pricing`, and a main-terms logistic censoring model | a defect in the censored recursion, with the ratio held exact |
| `censored_sdr_tau3` | the same draws, `SDR` | the same, for `SDR` |

Each primary scenario draws 1,000 samples of 2,400 rows. The property cells use the same n. The
calibration cells fit each engine over five folds with the screened quadratic logistic classifier
and the declared outcome learners of `mtp-pricing`. The necessity pairs use a wrong outcome model at
every node.

n = 2,400 comes from the tail-size rule: a Wald reading is gated only at an n of at least twice the
tail size of its quantity. Censoring multiplies the weights by the inverse retention and raises
their kurtosis. On law P-C the largest tail size is 1,098, for the contrast
(`tests/diagnostics/x77_censored_tail/law_pc_moments.csv`), and 2,400 is the first multiple of 400
at or above twice it. The three-node siblings in `mtp-pricing` run at n = 2,000, so these cells
differ from them in the dropout and in n, not in the dropout alone.

### Which cell carries which claim

| claim | cells |
| --- | --- |
| the censored recursion is unbiased and covers, with the policy ratio exact and the censoring model learned | the primary rows of both scenarios |
| the censoring factor reaches the weights | the `censoring_necessity` pairs and the exact-law witnesses of `tests/unit/test_longitudinal_mtp_censored.py` |
| the censoring factor is evaluated at the observed treatment, not the policy's | the exact-law witness with its mutation control M-a (`tests/unit/test_longitudinal_mtp_censored.py`) |
| the interval is calibrated with a learned classifier and a learned censoring model, cross-fitted | the `interval_calibration` cells and their controls |

The primary rows cannot see a censoring factor that never reaches the weights. On this law the
primary outcome learners nearly correct for the dropout on their own, so a fit whose censoring
factor is a constant per node reads a bias close to the learned fit's.
With a wrong outcome model only the weights can correct for informative dropout, so the necessity
pairs carry that claim. The exact-law witnesses fit a two-node censored law exactly and add four
mutation controls. Each control breaks one step: retention read at the shifted dose, a retention
model without the current dose, `SDR` weights without the censoring factor, and a classifier
trained on the units that stayed. Each moves the exact witness.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3-node pricing law with engagement feedback and informative dropout after each price, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under a continuous-treatment modified policy with censoring | -0.0014 to 0.0013 | 0.9410 | 1.0078 | pass |
| 3-node pricing law with engagement feedback and informative dropout after each price, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under a continuous-treatment modified policy with censoring | -0.000564 to 0.0010 | 0.9520 | 1.0132 | pass |
| 3-node pricing law with engagement feedback and informative dropout after each price, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under a continuous-treatment modified policy with censoring | -0.0015 to 0.0019 | 0.9490 | 0.9935 | pass |
| 3-node pricing law with engagement feedback and informative dropout after each price, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under a continuous-treatment modified policy with censoring | -0.0014 to 0.0013 | 0.9460 | 1.0081 | pass |
| 3-node pricing law with engagement feedback and informative dropout after each price, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under a continuous-treatment modified policy with censoring | -0.000564 to 0.0010 | 0.9520 | 1.0132 | pass |
| 3-node pricing law with engagement feedback and informative dropout after each price, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under a continuous-treatment modified policy with censoring | -0.0015 to 0.0019 | 0.9520 | 0.9937 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

**Python property evidence, with comparator runs pending.** This section holds no paired table,
because the comparator study has not run. `equivalence.csv` records no row.

The declared `lmtp` runs read the per-node joint ratio that `cleverly` computed, written beside
each sample as `ratio__<plan>__<node>`, together with the censoring columns. They are exactness
witnesses of the recursion given that ratio. Because `lmtp` would read `cleverly`'s own joint
ratio, the paired rows cannot catch a wrong retention model. The necessity pairs and the exact-law
witnesses carry that claim. A missing comparator blocks no method
([Theory and comparators](../../architecture-invariants.md#theory-and-comparators)).

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `censoring_necessity` | `up_ltmle_tau3__learned_censoring` | positive | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, LTMLE cross-fitted over five folds: a learned censoring model enters the weights, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.0026 to 0.000944, margin 0.0054 | pass |
| `censoring_necessity` | `up_ltmle_tau3__prior_censoring_control` | control | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, LTMLE cross-fitted over five folds: the same fit replaces the censoring model with the marginal retention rate, with the outcome misspecified | bias interval must fall entirely outside the margin | bias 0.0069 to 0.0102, margin 0.0050 | pass |
| `censoring_necessity` | `up_sdr_tau3__learned_censoring` | positive | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, SDR cross-fitted over five folds: a learned censoring model enters the weights, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.000478 to 0.0031, margin 0.0055 | pass |
| `censoring_necessity` | `up_sdr_tau3__prior_censoring_control` | control | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, SDR cross-fitted over five folds: the same fit replaces the censoring model with the marginal retention rate, with the outcome misspecified | bias interval must fall entirely outside the margin | bias 0.0120 to 0.0153, margin 0.0050 | pass |
| `interval_calibration` | `censored_crossfit_ltmle_tau3__correctly_specified` | positive | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, LTMLE cross-fitted over five folds with a learned classifier and a learned censoring model: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9474 to 0.9580, SE ratio 0.9982 to 1.0339 | pass |
| `interval_calibration` | `censored_crossfit_ltmle_tau3__noise_control` | control | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, LTMLE cross-fitted over five folds with a learned classifier and a learned censoring model: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8277 to 0.8461, SE ratio 0.7067 to 0.7321 | pass |
| `interval_calibration` | `censored_crossfit_ltmle_tau3__shrunken_se_control` | control | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, LTMLE cross-fitted over five folds with a learned classifier and a learned censoring model: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8248 to 0.8434, SE ratio 0.6991 to 0.7240 | pass |
| `interval_calibration` | `censored_crossfit_sdr_tau3__correctly_specified` | positive | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, SDR cross-fitted over five folds with a learned classifier and a learned censoring model: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9444 to 0.9553, SE ratio 0.9928 to 1.0280 | pass |
| `interval_calibration` | `censored_crossfit_sdr_tau3__noise_control` | control | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, SDR cross-fitted over five folds with a learned classifier and a learned censoring model: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8278 to 0.8462, SE ratio 0.7013 to 0.7271 | pass |
| `interval_calibration` | `censored_crossfit_sdr_tau3__shrunken_se_control` | control | the mean purchase under the 12% price rise at 3 nodes with dropout after each price, SDR cross-fitted over five folds with a learned classifier and a learned censoring model: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8262 to 0.8446, SE ratio 0.6949 to 0.7194 | pass |
<!-- /generated -->

| family | cells | replications and size |
| --- | --- | --- |
| `interval_calibration` | `LTMLE` and `SDR`, five folds, learned classifier and learned censoring model. Each positive has a shrunken-SE control and a noise control | 10,800 per arm at n = 2,400 |
| `censoring_necessity` | per engine, five folds, a wrong outcome GLM at every node: the learned censoring model against the marginal retention rate (`DummyClassifier(strategy="prior")`) | 1,000 pairs at n = 2,400 |

The control of each necessity pair stands for a fit that treats the dropout as uninformative. The
calibration arms were sized by the per-arm rule on 2,800-draw pilots, which read SE ratios of
1.019 and 1.020. `tests/studies/mtp_pricing_censored_properties.py` declares every cell and its
size.

## Red-cell route

The route was declared before any run (`tests/studies/canonical_mtp_pricing_censored.py`). A red
primary row or calibration positive is a defect suspect unless its 99% SE-ratio interval contains
1, its coverage interval contains 0.95, and a paired diagnosis against the complete-data sibling
finds no defect. A red necessity positive, or a control that is not displaced, is a defect suspect
and never takes an owner. A calibration control that does not fail is a red of the study. No cell
is red.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 2400 | observations per primary replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `property_cells_passed` | 10 | property cells passing |
| `property_cells_total` | 10 | property cells reported |
| `max_standardized_bias` | 0.0238 | largest primary standardized bias |
| `min_coverage` | 0.9410 | lowest primary coverage |
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
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it until the comparator runs register |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| a correctly specified main-terms logistic censoring model | the study does not measure a misspecified or flexible censoring learner. A wrong retention model biases the weights |
| censoring at random given the history and the current treatment | the data cannot check it. Dropout that depends on unrecorded causes biases every estimate |
| three nodes, cross-fitted, classifier ratio route, end-of-study binary outcome | in-sample fits, the density route, two or five nodes, a continuous outcome, survival outcomes and pooled censoring with a continuous policy stay experimental |
| one plan, a 12% multiplicative increase in the treatment that binds at the upper end of the support | other policy shapes rest on the complete-data studies and the exact-law witnesses |
| per-node retention of at least 0.58, about 14% censored by the third node | heavier dropout raises the tail size of the weights, and the study does not measure it |
| the `lmtp` comparison is pending, and would read `cleverly`'s own joint ratio | the paired rows would check the recursion given the ratio, not the retention model |

## Reproduction

`tests/canonical/mtp_pricing_censored/regenerate.py` regenerates the study. It runs no reference
process while the comparator is pending.
