# Treatment-mechanism covariates on point TMLE

This study validates point `TMLE` fitted with `treatment_covariates=`, the low-level form of
`PointTreatment(treatment_adjustment=...)`. The treatment mechanism $g$ is fitted on a declared
subset $W_g$ of the adjustment set $W$, and the outcome regression on all of $W$. The estimand
does not change. The declaration is a restriction on the observed-data model,
$P(A = 1 \mid W) = P(A = 1 \mid W_g)$, and the
[treatment-mechanism covariates section](../point-treatment-tmle.md#treatment-mechanism-covariates)
states the estimator, the restriction and why the interval holds under it. Under the
restriction the remainder term $P_0[(1 - g_0/g)(\bar Q^* - \bar Q_0)]$ vanishes when $g$ is
consistent for $g_0(W_g)$. When treatment also depends on a left-out covariate, that term
stays, and it is first order unless the outcome regression converges faster than root-n.

No maintained package exposes this option with an interval, so
no canonical implementation is compared. The R `ctmle` `Wg` fixture (`ctmle_wg.json`) is implementation parity for the
collaborative path and is not acceptance evidence here. The study rests on exact truths and on
repeated-sampling properties.

## What was fitted

| law | covariates, mechanism and outcome | fits |
| --- | --- | --- |
| LDA | $W_1, W_2, Z$ independent standard normal; $g_0 = \operatorname{expit}(0.5 W_1 - 0.5 W_2)$; $\operatorname{logit} \bar Q_0 = -0.3 + 0.6 A + 0.7 W_1 + 0.4 W_2 + 0.8 Z$. The restriction $W_g = (W_1, W_2)$ holds | unpenalized logistic GLMs by Newton: the outcome on $(A, W_1, W_2, Z)$, the treatment on $(W_1, W_2)$; in sample and stacked over the default folds; the arm means and the ATE at n = 2,000 |
| LDB | the same covariates; $g_0 = \operatorname{expit}(0.5 W_1 - 0.5 W_2 + Z)$; $\operatorname{logit} \bar Q_0 = -0.3 + 0.6 A + 0.7 W_1 + 0.4 W_2$. The restriction fails | the declared fit with the outcome GLM plus the constructed logit error $n^{-1/4} Z$, which converges at $n^{-1/4}$; the same outcome fit with $g$ on all three covariates and no declaration; and the declared fit with the correct outcome GLM; at n = 1,000, 4,000 and 16,000 |

The laws, learners and exact limits are in `tests/studies/_option_adjustment_laws.py`. The
truths come from a one-dimensional Gauss-Hermite rule, because the outcome logit reads $W$
through one normal index. The exact limits of the LDB fits are the targeted limits on a
quadrature grid, with the SD of the stacked estimating equation of the outcome and treatment
GLMs. `tests/unit/test_option_treatment_adjustment_design.py` recomputes every one, and
`tests/unit/test_option_adjustment_witness.py` checks the remainder identity on an exact finite
law at $10^{-10}$, with mutation controls that ignore the subset or drop the remainder's
centring.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three standard-normal covariates W1, W2, Z; treatment depends on W1 and W2 only and the outcome on all three (law LDA); logistic GLMs for the outcome on (A, W1, W2, Z) and for the treatment on the declared subset (W1, W2), in sample, n = 2,000 | `ate` | average treatment effect | `cleverly` | -0.0015 to 0.000730 | 0.9483 | 1.0051 | pass |
| three standard-normal covariates W1, W2, Z; treatment depends on W1 and W2 only and the outcome on all three (law LDA); logistic GLMs for the outcome on (A, W1, W2, Z) and for the treatment on the declared subset (W1, W2), in sample, n = 2,000 | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.000417 to 0.0012 | 0.9579 | 1.0245 | pass |
| three standard-normal covariates W1, W2, Z; treatment depends on W1 and W2 only and the outcome on all three (law LDA); logistic GLMs for the outcome on (A, W1, W2, Z) and for the treatment on the declared subset (W1, W2), in sample, n = 2,000 | `ey1` | counterfactual mean under treatment | `cleverly` | -0.000811 to 0.000841 | 0.9513 | 1.0077 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0239 to 0.0268, margin 0.0055, SE ratio 1.0089 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0017, margin 0.0057, SE ratio 1.0544 | pass |
| `instrument_parametric` | `n2000` | diagnostic | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE, n = 2,000: the correct outcome GLM and the treatment mechanism on the declared subset | none. The row is a diagnostic and states no verdict | bias -0.000142 to 0.0024, coverage 0.9254, SE ratio 0.9100, exact bias 0 SD, exact SE ratio 0.9161 | reported |
| `instrument_reference` | `n1000` | diagnostic | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE, n = 1,000: the same slow outcome regression with the treatment mechanism on (W1, W2, Z), without the declaration, on the slow-outcome cell's draws | none. The row is a diagnostic and states no verdict | bias -0.0034 to 0.0014, coverage 0.9413, SE ratio 0.9708, exact bias -0.000100 SD, exact SE ratio 1.0038 | reported |
| `instrument_reference` | `n16000` | diagnostic | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE, n = 16,000: the same slow outcome regression with the treatment mechanism on (W1, W2, Z), without the declaration, on the slow-outcome cell's draws | none. The row is a diagnostic and states no verdict | bias -0.000389 to 0.000824, coverage 0.9406, SE ratio 0.9767, exact bias -0.000300 SD, exact SE ratio 1.0008 | reported |
| `instrument_reference` | `n4000` | diagnostic | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE, n = 4,000: the same slow outcome regression with the treatment mechanism on (W1, W2, Z), without the declaration, on the slow-outcome cell's draws | none. The row is a diagnostic and states no verdict | bias -0.000916 to 0.0014, coverage 0.9469, SE ratio 1.0126, exact bias -0.000200 SD, exact SE ratio 1.0018 | reported |
| `instrument_slow_outcome` | `n16000` | control | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE, n = 16,000: the outcome GLM plus the constructed logit error n^(-1/4) Z and the treatment mechanism on the declared subset | the 99% interval of the standardized bias must lie wholly beyond the margin | bias -0.0166 to -0.0155, margin 0.0021 | pass |
| `instrument_slow_outcome_ladder` | `n1000` | diagnostic | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE, n = 1,000: the outcome GLM plus the constructed logit error n^(-1/4) Z and the treatment mechanism on the declared subset | none. The row is a diagnostic and states no verdict | bias -0.0345 to -0.0301, coverage 0.8025, SE ratio 0.9218, exact bias -0.9517 SD, exact SE ratio 0.9148 | reported |
| `instrument_slow_outcome_ladder` | `n4000` | diagnostic | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE, n = 4,000: the outcome GLM plus the constructed logit error n^(-1/4) Z and the treatment mechanism on the declared subset | none. The row is a diagnostic and states no verdict | bias -0.0235 to -0.0214, coverage 0.6763, SE ratio 0.9322, exact bias -1.3488 SD, exact SE ratio 0.9143 | reported |
| `instrument_slow_outcome_rate` | `bias_slope` | diagnostic | law LDB (treatment also depends on the left-out covariate Z, so the declared restriction fails), the ATE: the slope of the log absolute bias on log n over n = 1,000, 4,000 and 16,000 for the slow-outcome fit | none. The row is a diagnostic and states no verdict | slope -0.2785 to -0.2247 | reported |
| `interval_calibration` | `in_sample__correctly_specified` | positive | law LDA, the ATE, in sample, n = 2,000: the treatment mechanism fitted on the declared subset (W1, W2), which holds: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9466 to 0.9599, SE ratio 0.9874 to 1.0311 | pass |
| `interval_calibration` | `in_sample__noise_control` | control | law LDA, the ATE, in sample, n = 2,000: the treatment mechanism fitted on the declared subset (W1, W2), which holds: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8305 to 0.8534, SE ratio 0.6994 to 0.7306 | pass |
| `interval_calibration` | `in_sample__shrunken_se_control` | control | law LDA, the ATE, in sample, n = 2,000: the treatment mechanism fitted on the declared subset (W1, W2), which holds: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8234 to 0.8467, SE ratio 0.6912 to 0.7217 | pass |
| `interval_calibration` | `stacked__correctly_specified` | positive | law LDA, the ATE, stacked over the default folds, n = 2,000: the treatment mechanism fitted on the declared subset (W1, W2), which holds: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9480 to 0.9611, SE ratio 0.9915 to 1.0354 | pass |
| `interval_calibration` | `stacked__noise_control` | control | law LDA, the ATE, stacked over the default folds, n = 2,000: the treatment mechanism fitted on the declared subset (W1, W2), which holds: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8307 to 0.8536, SE ratio 0.6999 to 0.7315 | pass |
| `interval_calibration` | `stacked__shrunken_se_control` | control | law LDA, the ATE, stacked over the default folds, n = 2,000: the treatment mechanism fitted on the declared subset (W1, W2), which holds: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8255 to 0.8487, SE ratio 0.6946 to 0.7252 | pass |
<!-- /generated -->

The two calibration cells read the same 6,800 draws of LDA, in sample and stacked, with the two
shared derived controls. The double-robustness pair fits an intercept-only outcome regression:
with the declared mechanism on $(W_1, W_2)$ the bias is inside the margin, and with an
intercept-only mechanism too it lies far outside, so the pair can fail.

`instrument_slow_outcome/n16000` is the gated control of the failure mode. The declared fit on
LDB keeps the remainder the restriction hides, and with the slow outcome regression its
standardized bias reads -1.875 at n = 16,000 against the exact limit -1.910. The ladder rows
at n = 1,000 and 4,000 read -0.964 and -1.360 (exact -0.952 and -1.349). The slope of the log
absolute bias on log n is -0.252, against -1/4 for the constructed error, so the bias shrinks
more slowly than the standard error and the standardized bias grows like $n^{1/4}$. The
full-mechanism references fit the same outcome regression with $g$ on all of $W$ on the same
draws, and their bias stays near zero at every size. The cells therefore isolate the
restriction, not the slow outcome regression.

`instrument_parametric/n2000` fits the correct outcome GLM with the declared mechanism on LDB.
The point estimate is unbiased, because the outcome regression is correct. The interval is too
narrow: the SE ratio is 0.910 against the exact 0.916, and coverage is 0.925. The influence
curve of a fit on the wrong mechanism is not the curve of the estimator, so a correct outcome
regression does not rescue the interval.

## Reported readings

These rows carry no verdict. The ladder, rate, reference and parametric rows are the
`instrument_*` rows of the properties table above, each with its exact-law limit in the
`exact_bias_over_sd` and `exact_se_ratio` columns of `properties.csv`.

## Sizing

Every size was fixed before the run from the pilot in
`tests/studies/option_treatment_adjustment_pilot.py`, on a stream no declared cell reads. The
calibration count, 6,800, is the arm-pooled predictive rule at an 8,000-draw calibration pilot.
The bias cells and every ladder cell run 1,600, `instrument_parametric` 2,400, and the primary
phase 2,400. The design test recomputes each count.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/option_treatment_adjustment/regenerate.py`. Its seed is shared with
[ratio trimming on LTMLE and SDR](option-ratio-trim.md): the two declarations took the same
record seed, so their primary replicates read the same integer seeds on unrelated laws, and
every property stream hashes a label of its own.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 3 | truth tests passing |
| `independent_tests_total` | 3 | truth tests reported |
| `property_cells_passed` | 9 | gated property cells passing |
| `property_cells_total` | 9 | gated property cells reported |
| `max_standardized_bias` | 0.0256 | largest primary standardized bias |
| `min_coverage` | 0.9483 | lowest primary coverage |
| `properties[interval_calibration/in_sample__correctly_specified]:coverage` | 0.9535 | coverage of the in-sample calibration cell |
| `properties[interval_calibration/in_sample__correctly_specified]:se_ratio` | 1.0087 | SE ratio of the in-sample calibration cell |
| `properties[interval_calibration/stacked__correctly_specified]:coverage` | 0.9549 | coverage of the stacked calibration cell |
| `properties[interval_calibration/stacked__correctly_specified]:se_ratio` | 1.0135 | SE ratio of the stacked calibration cell |
| `properties[double_robustness/both_wrong]:standardized_bias` | 1.1555 | standardized bias of the both-wrong control |
| `properties[instrument_slow_outcome/n16000]:standardized_bias` | -1.8750 | standardized bias of the slow-outcome control at n = 16,000 |
| `properties[instrument_slow_outcome/n16000]:exact_bias_over_sd` | -1.9096 | its exact-law limit |
| `properties[instrument_slow_outcome_ladder/n1000]:standardized_bias` | -0.9644 | the slow-outcome fit at n = 1,000 |
| `properties[instrument_slow_outcome_ladder/n4000]:standardized_bias` | -1.3596 | the slow-outcome fit at n = 4,000 |
| `properties[instrument_slow_outcome_rate/bias_slope]:slope` | -0.2521 | slope of the log absolute bias on log n |
| `properties[instrument_parametric/n2000]:se_ratio` | 0.9100 | SE ratio of the declared fit with a correct outcome GLM on LDB |
| `properties[instrument_parametric/n2000]:coverage` | 0.9254 | its coverage |
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
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 24 | the same bound, in primary replications |

## Limitations

- **The restriction is the whole claim.** The interval holds when treatment depends on $W$ only
  through $W_g$. A regression of the treatment on the left-out covariates tests it only for the
  dependence that regression can represent, and not rejecting it does not establish it.
  Choosing the subset from the same data is a selection the interval ignores.
- **When the restriction fails the interval fails.** With a slowly converging outcome
  regression the standardized bias grows with n; with a correct parametric one the interval
  under-covers.
- **One binary outcome and parametric learners.** The cells measure logistic GLMs on two
  normal laws. A continuous outcome, flexible learners, DR-TMLE, C-TMLE and adaptive IPTW with
  the option read experimental.
- **Arm means, the ATE, and their ratios.** The `T54` rows cover families P1 and P2. The ATT,
  ATC and population attributable quantities read experimental.
- **No comparator.** No maintained implementation reports this interval.
- **No weights, clusters, strata or missing outcomes.** Each is a separate composition and
  reads experimental.

## Reproduction

`tests/canonical/option_treatment_adjustment/regenerate.py` regenerates the study and states
the run form. It runs no reference process, because the study has no comparator. The manifest
records the seeds, the margins, the estimator configuration and the source and result hashes,
and the run log records the commit, the clean tree and the wall time.
