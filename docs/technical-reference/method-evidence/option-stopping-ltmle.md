# Stopping rules on LTMLE

This study validates the interval of `LTMLE` under `stopping="scaled_var"` and
`stopping="sample_size"` on two walks: the per-node Newton fluctuation of an end-of-study
outcome, and the one-step hazard walk of a survival outcome. The
[stopping rules section](../point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate)
states the rules, which a longitudinal fit applies per node or per hazard walk. Under either
rule the walk can stop before its score equation is solved, and the reported interval adds the
remaining score to its half-width, $\psi_n \pm (z\,\widehat{se} + |r_n|)$. That interval covers
whenever the plain Wald pivot does, so it is conservative by construction.

No package widens a Wald interval by its remaining score, so no canonical implementation is compared.
The study rests on exact truths and on repeated-sampling properties.

## What was fitted

| law | construction | fits |
| --- | --- | --- |
| L3 | the `canonical-ltmle` censored two-node law, with that study's quasi-binomial outcome regressions and known treatment and censoring mechanisms | the per-node Newton walk under each rule and under the relative rule, in sample, at n = 2,000; always against never |
| L4 | the `point-survival-curve` five-visit survival law, with that study's learners | the one-step hazard walk under each rule and under the relative rule; the risk difference and the arm-1 risk at visit 5. The necessity cells use the `mechanism_correct` learners (intercept-only hazards, correct mechanisms) with every threshold times 8 and times 1 |

The laws and the threshold scale are in `tests/studies/_option_stopping_laws.py`, and
`tests/unit/test_option_stopping_ltmle_design.py` recomputes every declared number.
`tests/unit/test_stopping_rule_longitudinal.py` and `tests/unit/test_hazard_targeting_rules.py`
check each node's threshold and the walk's stop by hand.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000; the per-node Newton walk under stopping='scaled_var', the widened interval | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` | -0.0023 to 0.000668 | 0.9525 | 1.0179 | pass |
| law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000 and the arm-1 risk; stopping='scaled_var', the widened interval | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` | -0.0016 to 0.000804 | 0.9532 | 0.9887 | pass |
| law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000 and the arm-1 risk; stopping='scaled_var', the widened interval | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` | -0.000814 to 0.000878 | 0.9564 | 1.0121 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `l3_ss_newton__noise_control` | control | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='sample_size' on the Newton walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8115 to 0.8354, SE ratio 0.6761 to 0.7072 | pass |
| `interval_calibration` | `l3_ss_newton__shrunken_se_control` | control | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='sample_size' on the Newton walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8190 to 0.8426, SE ratio 0.6727 to 0.7041 | pass |
| `interval_calibration` | `l3_ss_newton__widened` | positive | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='sample_size' on the Newton walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9345 to 0.9493, SE ratio 0.9617 to 1.0066 | pass |
| `interval_calibration` | `l3_sv_newton__noise_control` | control | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='scaled_var' on the Newton walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8162 to 0.8399, SE ratio 0.6852 to 0.7159 | pass |
| `interval_calibration` | `l3_sv_newton__shrunken_se_control` | control | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='scaled_var' on the Newton walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8190 to 0.8426, SE ratio 0.6732 to 0.7043 | pass |
| `interval_calibration` | `l3_sv_newton__widened` | positive | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='scaled_var' on the Newton walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9345 to 0.9493, SE ratio 0.9619 to 1.0058 | pass |
| `interval_calibration` | `l4_ss_one_step__noise_control` | control | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='sample_size' on the one-step walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8207 to 0.8441, SE ratio 0.6927 to 0.7231 | pass |
| `interval_calibration` | `l4_ss_one_step__shrunken_se_control` | control | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='sample_size' on the one-step walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8211 to 0.8446, SE ratio 0.6936 to 0.7238 | pass |
| `interval_calibration` | `l4_ss_one_step__widened` | positive | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='sample_size' on the one-step walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9483 to 0.9614, SE ratio 0.9904 to 1.0345 | pass |
| `interval_calibration` | `l4_sv_one_step__noise_control` | control | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='scaled_var' on the one-step walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8223 to 0.8457, SE ratio 0.6957 to 0.7273 | pass |
| `interval_calibration` | `l4_sv_one_step__shrunken_se_control` | control | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='scaled_var' on the one-step walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8231 to 0.8464, SE ratio 0.6951 to 0.7270 | pass |
| `interval_calibration` | `l4_sv_one_step__widened` | positive | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='scaled_var' on the one-step walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9517 to 0.9644, SE ratio 0.9935 to 1.0384 | pass |
| `necessity_reference` | `l4_mc_contrast__plain` | diagnostic | law L4 with intercept-only hazard learners and correct mechanisms, n = 2,000, the risk difference at visit 5: stopping='scaled_var', every threshold times 8: the plain interval | none. The row is a diagnostic and states no verdict | coverage 0.7472 to 0.7739, SE ratio 1.0720, mean absolute residual 1.2864 SE | reported |
| `necessity_reference` | `l4_mc_contrast__widened` | diagnostic | law L4 with intercept-only hazard learners and correct mechanisms, n = 2,000, the risk difference at visit 5: stopping='scaled_var', every threshold times 8: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9747 to 0.9837, SE ratio 1.0720, mean absolute residual 1.2864 SE | reported |
| `necessity_reference` | `l4_mc_k1_risk__plain` | diagnostic | law L4 with intercept-only hazard learners and correct mechanisms, n = 2,000, the arm-1 risk at visit 5: stopping='scaled_var' at the declared thresholds: the plain interval | none. The row is a diagnostic and states no verdict | coverage 0.9532 to 0.9656, SE ratio 1.0337, mean absolute residual 0.0736 SE | reported |
| `necessity_reference` | `l4_mc_k1_risk__widened` | diagnostic | law L4 with intercept-only hazard learners and correct mechanisms, n = 2,000, the arm-1 risk at visit 5: stopping='scaled_var' at the declared thresholds: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9581 to 0.9698, SE ratio 1.0337, mean absolute residual 0.0736 SE | reported |
| `plain_interval` | `l3_ss_newton__plain` | diagnostic | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='sample_size' on the Newton walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9345 to 0.9493, SE ratio 0.9830, mean absolute residual 2.490e-11 SE | reported |
| `plain_interval` | `l3_sv_newton__plain` | diagnostic | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='scaled_var' on the Newton walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9345 to 0.9493, SE ratio 0.9830, mean absolute residual 2.490e-11 SE | reported |
| `plain_interval` | `l4_ss_one_step__plain` | diagnostic | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='sample_size' on the one-step walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9472 to 0.9604, SE ratio 1.0121, mean absolute residual 0.0072 SE | reported |
| `plain_interval` | `l4_sv_one_step__plain` | diagnostic | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='scaled_var' on the one-step walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9477 to 0.9608, SE ratio 1.0145, mean absolute residual 0.0338 SE | reported |
| `relative_reference` | `l3_rel_newton__widened` | diagnostic | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='relative' on the Newton walk: the default rule, whose residual is below tol | none. The row is a diagnostic and states no verdict | coverage 0.9345 to 0.9493, SE ratio 0.9830, mean absolute residual 6.479e-12 SE | reported |
| `relative_reference` | `l4_rel_one_step__widened` | diagnostic | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='relative' on the one-step walk: the default rule, whose residual is below tol | none. The row is a diagnostic and states no verdict | coverage 0.9467 to 0.9600, SE ratio 1.0116, mean absolute residual 5.147e-09 SE | reported |
| `reported_calibration` | `l4_ss_one_step_risk__widened` | diagnostic | law L4, the arm-1 risk at visit 5, n = 2,000: stopping='sample_size' on the one-step hazard walk: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9510 to 0.9637, SE ratio 1.0203 | reported |
| `reported_calibration` | `l4_sv_one_step_risk__widened` | diagnostic | law L4, the arm-1 risk at visit 5, n = 2,000: stopping='scaled_var' on the one-step hazard walk: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9547 to 0.9670, SE ratio 1.0260 | reported |
| `residual_necessity` | `l4_mc_risk__plain` | control | law L4 with intercept-only hazard learners and correct mechanisms, n = 2,000, the arm-1 risk at visit 5: stopping='scaled_var', every threshold times 8: the plain interval, without the residual | the upper limit of the coverage's exact 99% interval below 0.92 | coverage 0.8628 to 0.8837, SE ratio 1.0563, mean absolute residual 0.8513 SE | pass |
| `residual_necessity` | `l4_mc_risk__widened` | positive | law L4 with intercept-only hazard learners and correct mechanisms, n = 2,000, the arm-1 risk at visit 5: stopping='scaled_var', every threshold times 8: the widened interval of the same fits | the lower limit of the coverage's exact 99% interval at least 0.90 | coverage 0.9739 to 0.9831, SE ratio 1.0563, mean absolute residual 0.8513 SE | pass |
| `residual_witness` | `l3_ss_newton__witness` | diagnostic | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='sample_size' on the Newton walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0 to 0.000779 | reported |
| `residual_witness` | `l3_sv_newton__witness` | diagnostic | law L3 (the canonical-ltmle censored two-node law with its learners), always against never, n = 2,000: stopping='scaled_var' on the Newton walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0 to 0.000779 | reported |
| `residual_witness` | `l4_ss_one_step__witness` | diagnostic | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='sample_size' on the one-step walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0.2347 to 0.2618 | reported |
| `residual_witness` | `l4_sv_one_step__witness` | positive | law L4 (the point-survival-curve five-visit survival law, one-step hazard walk, with its learners), the risk difference at visit 5, n = 2,000: stopping='scaled_var' on the one-step walk: the share of draws whose absolute residual is at least 0.01 standard errors | the lower limit of the share's exact 99% interval at least 0.5 | share 0.8041 to 0.8284 | pass |
<!-- /generated -->

On L3 the per-node Newton walk lands at the default fit: the largest residual is
$1.3 \times 10^{-7}$ SE, no draw reaches 0.01 SE, and coverage (0.942) and SE ratio (0.983)
equal the relative fit's on the same draws. Those two cells measure the default estimator under
a rule and license only the Newton walk, not a one-step end-of-study walk. On L4 the rule binds:
`scaled_var` leaves a residual of at least 0.01 SE on 82% of draws and `sample_size` on 25%,
and the widened interval covers 0.958 and 0.955.

The necessity pair reads the arm-1 risk at visit 5 on L4 with intercept-only hazards and every
threshold times 8. The plain interval covers 0.874 and the widened interval 0.979 on the same
fits.

## Reported readings

These rows carry no verdict. The plain interval of the L4 calibration fits covers 0.954 and
0.955, within 0.004 of the widened one. On the necessity fits the risk difference covers 0.761
plain and 0.980 widened, and at the declared thresholds the arm-1 risk covers 0.960 and 0.964.
The arm-1 risk of the calibration fits covers 0.958 (`sample_size`) and 0.961 (`scaled_var`).

## Sizing

Every size was fixed before the run from the pilot in `tests/studies/option_stopping_pilot.py`,
on a stream no declared cell reads. Every property cell runs 6,800 replications, the arm-pooled
predictive rule at the pilot, and the primary phase 2,800. The design test recomputes each count.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/option_stopping_ltmle/regenerate.py`, and
`tests/unit/test_option_stopping_ltmle_design.py` applies it to the committed tables through
`gating_findings`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2800 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 3 | truth tests passing |
| `independent_tests_total` | 3 | truth tests reported |
| `property_cells_passed` | 15 | gated property cells passing |
| `property_cells_total` | 15 | gated property cells reported |
| `max_standardized_bias` | 0.0269 | largest primary standardized bias |
| `min_coverage` | 0.9525 | lowest primary coverage |
| `properties[interval_calibration/l3_sv_newton__widened]:coverage` | 0.9422 | widened coverage on L3, scaled_var |
| `properties[interval_calibration/l3_sv_newton__widened]:se_ratio` | 0.9830 | its SE ratio |
| `properties[relative_reference/l3_rel_newton__widened]:coverage` | 0.9422 | the relative fit on the same draws |
| `properties[interval_calibration/l4_sv_one_step__widened]:coverage` | 0.9584 | widened coverage on L4, scaled_var |
| `properties[interval_calibration/l4_ss_one_step__widened]:coverage` | 0.9551 | widened coverage on L4, sample_size |
| `properties[residual_witness/l4_sv_one_step__witness]:coverage` | 0.8165 | share of L4 draws with a residual of at least 0.01 SE, scaled_var |
| `properties[residual_witness/l4_ss_one_step__witness]:coverage` | 0.2481 | the same share, sample_size |
| `properties[residual_necessity/l4_mc_risk__plain]:coverage` | 0.8735 | plain coverage of the necessity fits |
| `properties[residual_necessity/l4_mc_risk__widened]:coverage` | 0.9788 | widened coverage of the necessity fits |
| `properties[necessity_reference/l4_mc_contrast__plain]:coverage` | 0.7607 | the risk difference of the necessity fits, plain |
| `properties[necessity_reference/l4_mc_contrast__widened]:coverage` | 0.9796 | the same, widened |
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

- **The widened interval is conservative.** Its coverage rests on the plain pivot alone, so
  where the rule binds hard it over-covers (0.979 in the necessity cell).
- **The L3 cells measure the default estimator.** The per-node Newton walk lands at the relative
  fit, so the end-of-study rows license that walk only; a one-step end-of-study walk under a
  rule matches no row.
- **Two laws, in sample.** One censored two-node law and one five-visit survival law at
  n = 2,000. Competing risks, repeated outcomes, cross-fitting, SDR and dynamic or stochastic
  plans under a rule read experimental.
- **No band.** The simultaneous band under a rule is not measured, and the bootstrap is refused.
- **No comparator.** No package widens a Wald interval by its remaining score.

## Reproduction

`tests/canonical/option_stopping_ltmle/regenerate.py` regenerates the study and states the run
form. It runs no reference process, because the study has no comparator. The manifest records
the seeds, the margins, the estimator configuration and the source and result hashes, and the
run log records the commit, the clean tree and the wall time.
