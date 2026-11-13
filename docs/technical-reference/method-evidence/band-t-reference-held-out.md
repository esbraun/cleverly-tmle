# Held-out t reference band

This study holds out one cell of the [per-estimate t reference band](band-t-reference.md) study.
It reads the band built with `band_reference="t"` on the strata law of that study at
$n = 1000$, a law and size that were not used to choose the rule or its df floor of 5. Planning
draws on a disjoint stream fixed only its size. The cell was declared at risk before its run:
at $n = 1000$ about 1% to 2% of fits hold a treatment arm with a single observed outcome in a
stratum, which no band rule can repair, and the df floor binds on most fits.

The cell licenses no support row and was outside the sizing family of the main study. It
publishes under the `reporting` policy, so a red verdict would have been published with an
owner and would have blocked nothing in the main study.

**No canonical implementation is compared.** No pinned comparator publishes a multiplier band
with a per-estimate t reference. The zero-row equivalence artifact records the absence of a
comparator.

## What was tested

| setting | declaration |
| --- | --- |
| `t` band cell | `strata_n1000__t_band`: the in-sample TMLE with three baseline strata over its 24 parameters, read with the `t` band. The 99% exact interval of the joint coverage must lie inside $[0.92, 0.98]$ |
| controls | `strata_n1000__pointwise_joint_control` (99% upper endpoint below 0.95) and `strata_n1000__normal_reference_control`, the default band of the same fits, whose row reports the paired gain of the `t` band and gates nothing |
| diagnostic arms | the four df arms of the main study, and the `t` band with the df floor at 2 and at 10 in place of 5. No arm states a verdict |
| constant arm cell | a replication whose band family holds a treatment arm with a single observed outcome in a stratum counts as not covered |
| budget | 5,600 replications, the declared count of the main study. The predicted pass probability at that count, from this study's 800-replication pilot, was 0.990 |
| declared reading | green: the rule holds out of sample. Red with the coverage of fits that hold no constant arm cell inside $[0.92, 0.98]$: the shortfall is the refusal rate. Red with that coverage also outside: a finding against the rule |
| primary scenario | `t_reference_fit`: the primary fit of the main study on this study's own seeds. $n = 1000$, 1,000 replications |
| Monte Carlo inference | exact 99% intervals around every coverage rate and paired gain, and 99% intervals around every primary endpoint |

The module docstring of `tests/studies/band_t_reference_held_out.py` states the cell and the
declared reading. `tests/unit/test_band_t_reference_design.py` checks its seeds and its pass
probability against the committed pilot.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `atc` | average effect on the untreated | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0011 to 0.0039 | 0.9530 | 1.0031 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `ate` | average treatment effect | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.000959 to 0.0040 | 0.9550 | 0.9978 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `att` | average effect on the treated | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.000929 to 0.0042 | 0.9560 | 0.9970 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `ey0` | counterfactual mean under no treatment | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0020 to 0.0015 | 0.9570 | 1.0292 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `ey1` | counterfactual mean under treatment | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.000618 to 0.0031 | 0.9400 | 0.9818 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `or` | marginal odds ratio, reported on the log scale | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0025 to 0.0181 | 0.9570 | 0.9991 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `rr` | marginal risk ratio, reported on the log scale | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0018 to 0.0093 | 0.9570 | 1.0095 | pass |
<!-- /generated -->

## Theory properties

A `reference_control` row prints the joint coverage of the default band and the paired gain of
the `t` band over it.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `simultaneous_coverage` | `strata_n1000__common_df_copula_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9829 to 0.9908 | reported |
| `simultaneous_coverage` | `strata_n1000__common_df_mvt_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9823 to 0.9904 | reported |
| `simultaneous_coverage` | `strata_n1000__floor_10_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the t band with the df floor 10 in place of 5 | none. The row is a diagnostic and states no verdict | joint coverage 0.9466 to 0.9612 | reported |
| `simultaneous_coverage` | `strata_n1000__floor_2_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the t band with the df floor 2 in place of 5 | none. The row is a diagnostic and states no verdict | joint coverage 0.9466 to 0.9612 | reported |
| `simultaneous_coverage` | `strata_n1000__model_df_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9369 to 0.9527 | reported |
| `simultaneous_coverage` | `strata_n1000__moment_df_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9380 to 0.9537 | reported |
| `simultaneous_coverage` | `strata_n1000__normal_reference_control` | reference_control | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9293 to 0.9460, paired gain of the t band 0.0121 to 0.0209 | pass |
| `simultaneous_coverage` | `strata_n1000__pointwise_joint_control` | control | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5560 to 0.5903 | pass |
| `simultaneous_coverage` | `strata_n1000__t_band` | positive | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 1,000: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9466 to 0.9612 | pass |
<!-- /generated -->

## Result

All three counted property cells pass, and the six diagnostic arms state no verdict. The cell is
green: the `t` band covers 0.9543, with its 99% interval inside $[0.92, 0.98]$, so the rule holds
out of sample and the study needs no owner. The pointwise control covers 0.5732.
The default band of the same fits covers 0.9380, and the paired gain of the `t` band over it is
0.0121 to 0.0209. That gain is reported and gates nothing.

54 of the 5,600 replications (0.96%) held a constant arm cell and count as not covered. The df
floor of 5 bound the smallest df on 56.9% of replications, and the median smallest df is 5.
The floor-2 and floor-10 arms cover 0.9543, the same as the declared floor, so this cell does
not separate the three floors. `tests/unit/test_band_t_reference_design.py` rebuilds these
counts from `t-band-replicates.csv`.

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
| `property_cells_total` | 3 | repeated-sampling property cells |
| `property_cells_passed` | 3 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0617 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9400 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9280 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0906 | highest bootstrap primary SE-ratio endpoint |
| `properties[simultaneous_coverage/strata_n1000__t_band]:coverage` | 0.9543 | joint coverage of the held-out `t` band |
| `properties[simultaneous_coverage/strata_n1000__t_band]:coverage_ci_lower` | 0.9466 | its 99% lower endpoint |
| `properties[simultaneous_coverage/strata_n1000__pointwise_joint_control]:coverage` | 0.5732 | joint coverage of the pointwise intervals |
| `properties[simultaneous_coverage/strata_n1000__normal_reference_control]:coverage` | 0.9380 | joint coverage of the default band of the same fits |
| `properties[simultaneous_coverage/strata_n1000__normal_reference_control]:coverage_gain_ci_lower` | 0.0121 | 99% lower endpoint of the paired gain, reported only |
| `properties[simultaneous_coverage/strata_n1000__normal_reference_control]:coverage_gain_ci_upper` | 0.0209 | its 99% upper endpoint |
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
| `margin:subject_failure_bound` | 0.0100 | share of `cleverly` fits a cell may lose |
| `margin:subject_failures_allowed` | 10 | `cleverly` fits a cell may lose |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes the verdict and would not fail on a red one |
| One held-out law and size | The cell checks the rule out of sample once. It licenses no support row |
| The floors are not separated | Floors 2, 5 and 10 give the same coverage on this cell, so it gives no evidence for the choice of 5 |
| No comparator | No pinned implementation publishes a band with a per-estimate t reference |

## Reproduction

The [fixture](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/band_t_reference_held_out/regenerate.py)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/band_t_reference_held_out/manifest.json)
records the seeds, margins, cell, source hashes and result hashes. The run used commit
`71076110`, which differs from the main study's commit `a11bcf62` only in a configuration key
of this study's module and in a design test.
