# Default simultaneous bands

This study measures the default simultaneous band of 25 shipped fit shapes. `TMLE` and `LTMLE`
default to `simultaneous=True`. A fit that reports two or more inferential estimates then
publishes a max-t multiplier band over all of them. The
[simultaneous bands](../inference.md#simultaneous-bands) section of the inference reference
defines the band and lists the other registered studies that measure it.

Each shape is the subject fit of one registered source study. This study refits it with
`simultaneous=True` and changes nothing else. The law, the sampler, the truth oracle, the sample
size and the fit function are the source study's own. The samples are this study's own: it draws
each replication through the source study's sampler at a seed this study owns.

**No canonical implementation is compared.** The pinned comparators publish no multiplier band
over these families. The zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| band | the shipped default: 1000 Rademacher draws on the point path and 2000 on the sequential path, seeded by the subject fit's `random_state`, over every estimate the fit reports. A ratio band is exponentiated from the log scale |
| band cell | `<shape>__simultaneous_band`. Each replication is covered when every truth lies inside the band. The 99% exact interval of the joint coverage must lie inside $[0.92, 0.98]$ |
| control cell | `<shape>__pointwise_joint_control`. The same fits, read with their pointwise 95% intervals jointly. The 99% upper endpoint of the joint coverage must lie below 0.95, which shows that the band's critical value does work |
| budget | 2,400 replications for each shape, and 4,000 for `longitudinal_msm`. `tests/unit/test_simultaneous_cell_design.py` computes the control's power from the family's correlation before any run, and requires at least 0.99 |
| primary scenario | `default_fit`: `TMLE()` with two main-terms logistic learners and `random_state=0`, and every other argument at its default, on the binary law of the [ordinary point-treatment study](canonical-point-treatment-tmle.md). Ten-fold pooled cross-fitting, the default binary estimands, $n = 1000$, 1,000 replications |
| Monte Carlo inference | exact 99% intervals around every coverage rate, and 99% intervals around every primary endpoint |

The table gives each shape, its source study and its size.

| shape | source study | what the shape adds | n |
| --- | --- | --- | ---: |
| `default_fit` | this study's primary scenario | the shipped `TMLE()` defaults | 1,000 |
| `ordinary_binary` | [ordinary point-treatment TMLE](canonical-point-treatment-tmle.md) | ten estimands, with log-scale ratio bands and the PAF | 1,000 |
| `fold_evaluated` | [fold-evaluated CV-TMLE](fold-evaluated-point-treatment-cv-tmle.md) | a cross-validated variance beside centered multiplier draws | 1,000 |
| `weighted` | [weighted point-treatment TMLE](weighted-point-treatment-tmle.md) | fixed observation weights and ratios | 2,000 |
| `learned_weighted` | [learned weighted point-treatment TMLE](learned-weighted-point-treatment-tmle.md) | weighted nuisances and a linear fluctuation | 2,000 |
| `missing_outcome` | [missing-outcome TMLE](ordinary-missing-outcome-tmle.md) | a missing outcome | 2,000 |
| `missing_outcome_drtmle` | [missing-outcome DR-TMLE](randomized-missing-outcome-dr-tmle.md) | DR-TMLE with a missing outcome | 2,000 |
| `drtmle` | [DR-TMLE for binary complete data](canonical-dr-tmle.md) | cross-fitted DR-TMLE corrected curves | 3,000 |
| `multi_arm_drtmle` | [multi-arm DR-TMLE](multi-arm-dr-tmle.md) | nine multi-arm parameters with ratios | 2,000 |
| `cde_z0`, `cde_z1` | [controlled direct-effect TMLE](controlled-direct-effect-tmle.md) | one band for each intermediate level | 2,000 |
| `point_msm` | [point-treatment MSM](point-treatment-msm-projection.md) | three MSM projection terms | 2,000 |
| `clustered` | [clustered CV-TMLE](clustered-point-treatment-cv-tmle.md) | grouped folds and multipliers drawn by cluster | 2,000 |
| `shift_grid` | [continuous modified treatment policies](continuous-modified-treatment-policies.md) | three shifts and two contrasts | 2,000 |
| `incremental_grid` | [incremental interventions](incremental-propensity-interventions.md) | three odds multipliers and two contrasts | 2,000 |
| `stochastic_regimes` | [known stochastic regimes](stochastic-point-treatment-regimes.md) | a static and a stochastic regime | 2,000 |
| `deterministic_regimes` | [deterministic regimes](deterministic-point-treatment-regimes.md) | a static regime and a dynamic rule | 2,000 |
| `ltmle_crossfit` | [cross-fitted end-of-study LTMLE](cross-fitted-end-of-study-longitudinal-tmle.md) | five cross-fitted regimen parameters | 2,000 |
| `survival_crossfit` | [cross-fitted survival-curve LTMLE](cross-fitted-survival-curve-longitudinal-tmle.md) | ten parameters with two exact duplicate pairs | 2,000 |
| `competing_crossfit` | [cross-fitted competing-risk LTMLE](cross-fitted-competing-risk-longitudinal-tmle.md) | twenty parameters with four exact duplicates | 4,000 |
| `weighted_ltmle` | [weighted end-of-study LTMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | observation weights | 2,000 |
| `weighted_ltmle_crossfit` | [cross-fitted weighted end-of-study LTMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | observation weights, cross-fitted | 2,000 |
| `categorical_ltmle` | [categorical LTMLE](ordinary-categorical-longitudinal-tmle.md) | five categorical regimens | 2,000 |
| `categorical_ltmle_crossfit` | [cross-fitted categorical LTMLE](cross-fitted-categorical-longitudinal-tmle.md) | the same, cross-fitted | 2,000 |
| `longitudinal_msm` | [longitudinal MSM](ordinary-longitudinal-msm-projection.md) | two MSM projection terms | 2,500 |

A duplicate is a parameter that equals another one exactly, because the two plans agree at node
one. The band covers it twice. The truth of a duplicate is the truth of its twin.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, fitted with the shipped TMLE() defaults | `atc` | average effect on the untreated | `cleverly` shipped TMLE() default fit | -0.0032 to 0.0020 | 0.9610 | 0.9813 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults | `ate` | average treatment effect | `cleverly` shipped TMLE() default fit | -0.0031 to 0.0019 | 0.9600 | 0.9812 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults | `att` | average effect on the treated | `cleverly` shipped TMLE() default fit | -0.0032 to 0.0020 | 0.9590 | 0.9809 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults | `ey0` | counterfactual mean under no treatment | `cleverly` shipped TMLE() default fit | -0.0020 to 0.0018 | 0.9400 | 0.9564 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults | `ey1` | counterfactual mean under treatment | `cleverly` shipped TMLE() default fit | -0.0026 to 0.0011 | 0.9380 | 0.9916 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults | `or` | marginal odds ratio, reported on the log scale | `cleverly` shipped TMLE() default fit | -0.0111 to 0.0098 | 0.9580 | 0.9807 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults | `rr` | marginal risk ratio, reported on the log scale | `cleverly` shipped TMLE() default fit | -0.0056 to 0.0059 | 0.9520 | 0.9761 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `simultaneous_coverage` | `categorical_ltmle__pointwise_joint_control` | control | five categorical regimen means and four contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6579 to 0.7072 | pass |
| `simultaneous_coverage` | `categorical_ltmle__simultaneous_band` | positive | five categorical regimen means and four contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9101 to 0.9382 | **fail** |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__pointwise_joint_control` | control | five cross-fitted categorical regimen means and four contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6864 to 0.7344 | pass |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__simultaneous_band` | positive | five cross-fitted categorical regimen means and four contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9173 to 0.9443 | **fail** |
| `simultaneous_coverage` | `cde_z0__pointwise_joint_control` | control | the controlled direct-effect fit at intermediate level zero: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8260 to 0.8643 | pass |
| `simultaneous_coverage` | `cde_z0__simultaneous_band` | positive | the controlled direct-effect fit at intermediate level zero: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9015 to 0.9309 | **fail** |
| `simultaneous_coverage` | `cde_z1__pointwise_joint_control` | control | the controlled direct-effect fit at intermediate level one: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8452 to 0.8817 | pass |
| `simultaneous_coverage` | `cde_z1__simultaneous_band` | positive | the controlled direct-effect fit at intermediate level one: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9228 to 0.9488 | pass |
| `simultaneous_coverage` | `clustered__pointwise_joint_control` | control | grouped cross-fitted TMLE with cluster multipliers: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8611 to 0.8957 | pass |
| `simultaneous_coverage` | `clustered__simultaneous_band` | positive | grouped cross-fitted TMLE with cluster multipliers: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9241 to 0.9499 | pass |
| `simultaneous_coverage` | `competing_crossfit__pointwise_joint_control` | control | cross-fitted cumulative incidence of two causes, three plans, two horizons: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6063 to 0.6574 | pass |
| `simultaneous_coverage` | `competing_crossfit__simultaneous_band` | positive | cross-fitted cumulative incidence of two causes, three plans, two horizons: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9232 to 0.9492 | pass |
| `simultaneous_coverage` | `default_fit__pointwise_joint_control` | control | the shipped TMLE() defaults on the binary law: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8408 to 0.8777 | pass |
| `simultaneous_coverage` | `default_fit__simultaneous_band` | positive | the shipped TMLE() defaults on the binary law: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9365 to 0.9600 | pass |
| `simultaneous_coverage` | `deterministic_regimes__pointwise_joint_control` | control | a static regime and a dynamic rule and their contrast: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8523 to 0.8879 | pass |
| `simultaneous_coverage` | `deterministic_regimes__simultaneous_band` | positive | a static regime and a dynamic rule and their contrast: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9301 to 0.9548 | pass |
| `simultaneous_coverage` | `drtmle__pointwise_joint_control` | control | cross-fitted DR-TMLE over the arm means and the ATE: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8743 to 0.9074 | pass |
| `simultaneous_coverage` | `drtmle__simultaneous_band` | positive | cross-fitted DR-TMLE over the arm means and the ATE: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9467 to 0.9681 | pass |
| `simultaneous_coverage` | `fold_evaluated__pointwise_joint_control` | control | fold-evaluated CV-TMLE over five estimands: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8461 to 0.8824 | pass |
| `simultaneous_coverage` | `fold_evaluated__simultaneous_band` | positive | fold-evaluated CV-TMLE over five estimands: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9411 to 0.9637 | pass |
| `simultaneous_coverage` | `incremental_grid__pointwise_joint_control` | control | three incremental odds multipliers and their two contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8518 to 0.8875 | pass |
| `simultaneous_coverage` | `incremental_grid__simultaneous_band` | positive | three incremental odds multipliers and their two contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9351 to 0.9589 | pass |
| `simultaneous_coverage` | `learned_weighted__pointwise_joint_control` | control | weighted point-treatment TMLE with learned nuisances: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8730 to 0.9063 | pass |
| `simultaneous_coverage` | `learned_weighted__simultaneous_band` | positive | weighted point-treatment TMLE with learned nuisances: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9443 to 0.9663 | pass |
| `simultaneous_coverage` | `longitudinal_msm__pointwise_joint_control` | control | the two terms of the longitudinal MSM projection: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.9038 to 0.9267 | pass |
| `simultaneous_coverage` | `longitudinal_msm__simultaneous_band` | positive | the two terms of the longitudinal MSM projection: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9337 to 0.9527 | pass |
| `simultaneous_coverage` | `ltmle_crossfit__pointwise_joint_control` | control | cross-fitted end-of-study regimen means and contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8007 to 0.8413 | pass |
| `simultaneous_coverage` | `ltmle_crossfit__simultaneous_band` | positive | cross-fitted end-of-study regimen means and contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9260 to 0.9514 | pass |
| `simultaneous_coverage` | `missing_outcome__pointwise_joint_control` | control | missing-outcome TMLE over the arm means and the ATE: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8553 to 0.8907 | pass |
| `simultaneous_coverage` | `missing_outcome__simultaneous_band` | positive | missing-outcome TMLE over the arm means and the ATE: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9232 to 0.9492 | pass |
| `simultaneous_coverage` | `missing_outcome_drtmle__pointwise_joint_control` | control | missing-outcome DR-TMLE over the arm means and the ATE: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8672 to 0.9012 | pass |
| `simultaneous_coverage` | `missing_outcome_drtmle__simultaneous_band` | positive | missing-outcome DR-TMLE over the arm means and the ATE: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9328 to 0.9570 | pass |
| `simultaneous_coverage` | `multi_arm_drtmle__pointwise_joint_control` | control | multi-arm DR-TMLE over nine parameters: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7860 to 0.8278 | pass |
| `simultaneous_coverage` | `multi_arm_drtmle__simultaneous_band` | positive | multi-arm DR-TMLE over nine parameters: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9273 to 0.9526 | pass |
| `simultaneous_coverage` | `ordinary_binary__pointwise_joint_control` | control | ordinary in-sample TMLE over ten binary-outcome estimands: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8181 to 0.8572 | pass |
| `simultaneous_coverage` | `ordinary_binary__simultaneous_band` | positive | ordinary in-sample TMLE over ten binary-outcome estimands: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9383 to 0.9615 | pass |
| `simultaneous_coverage` | `point_msm__pointwise_joint_control` | control | the three terms of the point-treatment MSM projection: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8646 to 0.8989 | pass |
| `simultaneous_coverage` | `point_msm__simultaneous_band` | positive | the three terms of the point-treatment MSM projection: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9278 to 0.9529 | pass |
| `simultaneous_coverage` | `shift_grid__pointwise_joint_control` | control | three shift policies and their two contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8316 to 0.8695 | pass |
| `simultaneous_coverage` | `shift_grid__simultaneous_band` | positive | three shift policies and their two contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9346 to 0.9585 | pass |
| `simultaneous_coverage` | `stochastic_regimes__pointwise_joint_control` | control | a static and a known stochastic regime and their contrast: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8642 to 0.8985 | pass |
| `simultaneous_coverage` | `stochastic_regimes__simultaneous_band` | positive | a static and a known stochastic regime and their contrast: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9388 to 0.9619 | pass |
| `simultaneous_coverage` | `survival_crossfit__pointwise_joint_control` | control | the cross-fitted survival curve of three plans at two horizons: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7407 to 0.7857 | pass |
| `simultaneous_coverage` | `survival_crossfit__simultaneous_band` | positive | the cross-fitted survival curve of three plans at two horizons: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9223 to 0.9484 | pass |
| `simultaneous_coverage` | `weighted__pointwise_joint_control` | control | weighted point-treatment TMLE over five estimands: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8545 to 0.8899 | pass |
| `simultaneous_coverage` | `weighted__simultaneous_band` | positive | weighted point-treatment TMLE over five estimands: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9328 to 0.9570 | pass |
| `simultaneous_coverage` | `weighted_ltmle__pointwise_joint_control` | control | weighted end-of-study regimen means and contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8037 to 0.8441 | pass |
| `simultaneous_coverage` | `weighted_ltmle__simultaneous_band` | positive | weighted end-of-study regimen means and contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9278 to 0.9529 | pass |
| `simultaneous_coverage` | `weighted_ltmle_crossfit__pointwise_joint_control` | control | cross-fitted weighted end-of-study regimen means and contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8150 to 0.8544 | pass |
| `simultaneous_coverage` | `weighted_ltmle_crossfit__simultaneous_band` | positive | cross-fitted weighted end-of-study regimen means and contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9269 to 0.9522 | pass |
<!-- /generated -->

## Result

Every primary test passed, and every pointwise control fell below 0.95, so each band's critical
value does work. The bands of 22 of the 25 shapes kept their joint-coverage interval inside
$[0.92, 0.98]$. The study publishes under the `reporting` policy, and three bands are red.
[`default-band-shortfall`](../../roadmap.md#red-cell-owners) owns them in the
[red-cell ledger](red-cells.md).

| shape | band coverage | oracle band coverage | source pointwise calibration |
| --- | --- | --- | --- |
| `categorical_ltmle` | 0.9250 | 0.9267 at 2.689 | static contrast 0.939 |
| `categorical_ltmle_crossfit` | 0.9317 | 0.9325 at 2.689 | static contrast 0.946 |
| `cde_z0` | 0.9171 | 0.9200 at 2.331 | 0.942 at Z = 0 |

The declared diagnostic read each red cell's rows again at the design critical value, which
`tests/unit/test_simultaneous_cell_design.py` computes. The oracle band also covers below the band
edge or at it, so the multiplier is not the cause. Each source study's pointwise calibration of
the same parameters sits near the lower edge of its own band. The reading is finite-sample, so
the default band stays on.

The first declared run stopped before any joint verdict. One sample of the `multi_arm_drtmle`
shape activated the propensity bound, and the source study's fit raises on that, because that
study publishes `bound_active` as false for every replication. The band does not depend on that
claim, so this study turns the check off and keeps the replication as drawn. A second run ran out
of memory at 16 workers, and the published run used 8.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 7 | primary tests against truth |
| `independent_tests_passed` | 7 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 50 | repeated-sampling property cells |
| `property_cells_passed` | 47 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0320 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9380 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9081 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0569 | highest bootstrap primary SE-ratio endpoint |
| `properties[simultaneous_coverage/default_fit__simultaneous_band]:coverage` | 0.9492 | joint coverage of the band of the shipped TMLE() default |
| `properties[simultaneous_coverage/default_fit__pointwise_joint_control]:coverage` | 0.8600 | joint coverage of its pointwise intervals |
| `properties[simultaneous_coverage/categorical_ltmle__simultaneous_band]:coverage` | 0.9250 | joint coverage of the categorical LTMLE band |
| `properties[simultaneous_coverage/categorical_ltmle__simultaneous_band]:coverage_ci_lower` | 0.9101 | its 99% lower endpoint |
| `properties[simultaneous_coverage/categorical_ltmle_crossfit__simultaneous_band]:coverage` | 0.9317 | joint coverage of the cross-fitted categorical LTMLE band |
| `properties[simultaneous_coverage/categorical_ltmle_crossfit__simultaneous_band]:coverage_ci_lower` | 0.9173 | its 99% lower endpoint |
| `properties[simultaneous_coverage/cde_z0__simultaneous_band]:coverage` | 0.9171 | joint coverage of the controlled direct-effect band at Z = 0 |
| `properties[simultaneous_coverage/cde_z0__simultaneous_band]:coverage_ci_lower` | 0.9015 | its 99% lower endpoint |
| `properties[simultaneous_coverage/drtmle__simultaneous_band]:coverage` | 0.9583 | the highest band coverage, cross-fitted DR-TMLE |
| `properties[simultaneous_coverage/longitudinal_msm__pointwise_joint_control]:coverage` | 0.9157 | the highest control coverage, two longitudinal MSM projection terms |
| `properties[simultaneous_coverage/longitudinal_msm__pointwise_joint_control]:coverage_ci_upper` | 0.9267 | its 99% upper endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration and joint-coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration and joint-coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Each shape is measured at one law and one size | The band of another law, size or learner is not measured. Each source study page states the law |
| The band uses one multiplier seed for every replication | The shipped band seeds its draws with the fit's `random_state`, and each source fit fixes it. The cells measure the band as it ships |
| No comparator | The pinned comparators publish no multiplier band over these families |
| The design correlations are estimates | The control's power comes from the curve covariance averaged over ten fits at each shape's size, or three for the four slowest shapes. The design test holds the power at the joint coverage plus 0.005 |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/default_bands/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/default_bands/manifest.json)
records the seeds, margins, shapes, source hashes, and result hashes.
