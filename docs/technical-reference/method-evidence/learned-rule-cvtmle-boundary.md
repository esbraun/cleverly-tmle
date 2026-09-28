# CV-TMLE of the fold-local learned-rule value at exceptional and weak-blip laws

This study reads the interval of `LearnedRuleValue` at two laws that the
[gated study](learned-rule-cvtmle.md) does not claim. The estimator, the learners, the size, the
truth rule and the oracle SE are those of the gated study. Only the laws differ.

The study publishes under the `reporting` policy, and it declares no property cell. A study with
no property cells is not a validation row, as
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row) states. The
[validation grid](validation-grid.md) therefore does not list it. Each red verdict enters the
[red-cell ledger](red-cells.md) with the owner
[F27](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions).

**No canonical implementation is compared.** The gated study gives the comparator survey. A
zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law, fit, learners, truth and oracle SE | as in the [gated study](learned-rule-cvtmle.md#what-was-tested) |
| primary budget | 6,000 replications for each law, at n = 2,000 |
| verdicts | the gated study's primary verdicts on the error, under `publication_policy="reporting"` |
| reading | the declared reading table below, read from the top |
| seeds | `seed=20263003` and `resampling_seed=20263004` |
| policy | `reporting` |

| law | $b(W)$ | $d_0$ treats | limit blip of the learner | $\Psi_{d_0}(P_0)$ | condition C3 |
| --- | --- | --- | --- | ---: | --- |
| `exceptional` | $0$ | no unit | $0$, so no fixed limit rule exists | 0.512179 | fails. Every rule has the value 0.512179 |
| `weak_blip` | $0.15 W_1$ | $W_1 > 0$ | $0.15 W_1$ | 0.521047 | holds. Equation (5) and Theorem 6 of van der Laan and Luedtke (2015) apply |

At `exceptional` the truth of every replication is 0.512179, because the effect is zero for every
unit. At `weak_blip` the truth varies with the fitted rules.

The [RM30 record](../../roadmap.md#rm30-learned-policy-value-evaluation) names the commit that
holds the declaration. That declaration fixed the laws, the budget, the seeds and the reading
table before any run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0, no treatment effect for any unit | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated CV-TMLE of the learned-rule value | -0.000748 to 0.000535 | 0.8938 | 0.8211 | **fail** |
| learned-rule law with the weak blip 0.15 W1 | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated CV-TMLE of the learned-rule value | -0.000514 to 0.000732 | 0.8948 | 0.8314 | **fail** |
<!-- /generated -->

Both laws pass the bias verdict and the SE-ratio screen. Both laws fail the coverage floor, so
each truth row is red.

## Reading

The reading reads each law's 99% Clopper-Pearson coverage interval. The first condition that holds
names the reading.

| reading, for each law | condition |
| --- | --- |
| `under-covers at the <law> law` | the interval's upper end is below 0.95 |
| `no under-coverage resolved at the declared budget` | the interval's lower end is at or above 0.90 |
| `unresolved` | otherwise |

The reading `unresolved` cannot occur at 6,000 replications. It needs an interval that reaches
below 0.90 and above 0.95, which is 0.05 wide. At 6,000 replications the 99% Clopper-Pearson
interval is at most 0.0334 wide.

| law | coverage | 99% interval | reading |
| --- | ---: | --- | --- |
| `exceptional` | 0.8938 | 0.8832 to 0.9038 | `under-covers at the exceptional law` |
| `weak_blip` | 0.8948 | 0.8842 to 0.9048 | `under-covers at the weak_blip law` |

`reading.csv` holds each reading. The measured-values table below gates every number in this
table.

## What the readings show

| law | what the reading shows | source of the interpretation |
| --- | --- | --- |
| `exceptional` | the interval under-covers where condition C3 fails. The effect is zero for every unit, so the fold rules have no fixed limit. This law is non-regular | Luedtke and van der Laan (2016), *Annals of Statistics*, Section 4.1, expect the rule estimate "to fluctuate randomly" where the blip is zero. Section 4.2 names the tools that avoid a fixed limit, and no reviewed source applies them to this target ([F27](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions)) |
| `weak_blip` | the interval under-covers at n = 2,000 on a law within C3. The red cell is a finite-sample limit at this size | van der Laan and Luedtke (2015), Equation (5) and Theorem 6, cover this law. F27 states that no published result is missing for it |

At both laws the mean reported SE is smaller than the SD of the error. The SE ratio is 0.8211 at
`exceptional` and 0.8314 at `weak_blip`. The table gives the three spreads. The oracle SE is the
standard error that the influence curves of the fold rules give under $P_0$, with each rule held
fixed. It is a descriptive
column of `harness.csv.gz`, and no rule reads it.

| law | mean reported SE | mean oracle SE | SD of the error |
| --- | ---: | ---: | ---: |
| `exceptional` | 0.0158 | 0.0158 | 0.0193 |
| `weak_blip` | 0.0156 | 0.0155 | 0.0187 |

The readings do not show the items in this list.

- The cause of the smaller SE. The study does not separate the non-regularity from other
  finite-sample effects.
- The size at which `weak_blip` reaches nominal coverage. The study runs one size.
- The coverage at other laws, learners or fold counts.
- A defect in `cleverly`. The gated study validates the interval at two laws that meet C3.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 6000 | primary replications per law |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | law tests against the replication truth |
| `independent_tests_passed` | 0 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 0 | repeated-sampling property cells, none declared |
| `property_cells_passed` | 0 | of those, passing |
| `max_standardized_bias` | 0.0058 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.8938 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.8832 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.8017 | lowest bootstrap primary SE-ratio endpoint, against 0.80 |
| `max_se_ratio_ci_upper` | 0.8519 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]]:coverage` | 0.8938 | coverage at `exceptional` |
| `performance[cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]]:coverage_ci_lower` | 0.8832 | its 99% lower endpoint |
| `performance[cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]]:coverage_ci_upper` | 0.9038 | its 99% upper endpoint, against 0.95 |
| `performance[cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]]:se_ratio` | 0.8211 | SE ratio at `exceptional` |
| `performance[cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]]:coverage` | 0.8948 | coverage at `weak_blip` |
| `performance[cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]]:coverage_ci_lower` | 0.8842 | its 99% lower endpoint |
| `performance[cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]]:coverage_ci_upper` | 0.9048 | its 99% upper endpoint, against 0.95 |
| `performance[cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]]:se_ratio` | 0.8314 | SE ratio at `weak_blip` |
| `summary[cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]]:mean_std_error` | 0.0158 | mean reported SE at `exceptional` |
| `summary[cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]]:empirical_se` | 0.0193 | SD of the error at `exceptional` |
| `summary[cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]]:mean_std_error` | 0.0156 | mean reported SE at `weak_blip` |
| `summary[cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]]:empirical_se` | 0.0187 | SD of the error at `weak_blip` |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. The study declares no calibration cell |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy | Both truth rows are red, and the policy publishes them. The fast tier recomputes each verdict and does not fail on a red one |
| No property cell | The study is not a validation row, and the validation grid does not list it |
| There is no cross-implementation evidence | No maintained package fits this construction |
| Two laws, one size, one learner class | The readings hold for the declared laws at n = 2,000 with the gated study's logistic learners |
| The gated study's other limits apply | A binary treatment and outcome, equal folds, and no weights, clusters, missing outcomes, repeats, full-refit bootstrap or strata |
| A red cell is not a defect | A red `exceptional` cell marks the boundary of condition C3. A red `weak_blip` cell is a finite-sample limit at n = 2,000 inside C3 |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle_boundary/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle_boundary/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle_boundary/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle_boundary/replicates.csv.gz),
the [harness columns](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle_boundary/harness.csv.gz)
and the [readings](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle_boundary/reading.csv)
carry every published row.
