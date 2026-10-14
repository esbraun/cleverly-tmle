# Repeated-outcome longitudinal TMLE

This study validates `LTMLE` with `outcome=Repeated(["Y1", "Y2", "Y3"])`. A repeated outcome is
measured at every node and is not absorbing. The parameter is the mean $E[Y_h^d]$ for each plan $d$
and horizon $h$. The section [repeated outcomes](../longitudinal-tmle.md#repeated-outcomes) states
the estimator and its exact-law evidence. The canonical comparison uses R
[`ltmle`](https://www.jstatsoft.org/article/view/v081i01) 1.3-0 with `survivalOutcome = FALSE`.

The study reports each unique parameter once. The dynamic plan equals the always-treated plan at the
first node, so its mean and contrast at $t = 1$ duplicate those of `always`. A fast test checks that
identity. The duplicates are not cells of this study.

## What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| datasets | 1,600 censored panels of 2,000 rows from `make_longitudinal_repeated`, with three nodes | the identical rows |
| fit | one fit with `outcome=Repeated(["Y1", "Y2", "Y3"])` | one fit of the prefix through $Y_h$ for each horizon $h$. The earlier $Y_s$ are covariates |
| plans | `never`, `always`, and a rule that treats at node 1 and then treats exactly when the previous outcome was positive | the same three plans |
| reported parameters | the mean of each plan at $t = 1, 2, 3$, and the contrasts of the other two plans against `never` | the same, 13 unique cells |
| mechanism | the generating treatment and censoring probabilities, read through a declared design width, with `g_bounds` from the study constant | a numeric `gform` with the same probabilities and bounds |
| sequential regressions | follower-stratified quasibinomial regressions on the history at each node | the same family, with a main-terms `Qform` on the same history at each block |
| fitting | in sample, `n_folds = 1` | in sample |
| intervals | pointwise 95% Wald from the influence curve | `variance.method = "ic"` |

Each truth is an exact enumeration of the binary law (`cleverly.datasets.repeated_truth`).
`tests/studies/ltmle_repeated_law.py` repeats the enumeration on the observed-data law.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 1 | `cleverly` | -0.000372 to 0.0024 | 0.9494 | 0.9971 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 1 | R `ltmle` | -0.000372 to 0.0024 | 0.9494 | 0.9971 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 2 | `cleverly` | -0.0020 to 0.0026 | 0.9444 | 0.9657 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 2 | R `ltmle` | -0.0020 to 0.0026 | 0.9444 | 0.9657 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 3 | `cleverly` | -0.0032 to 0.0039 | 0.9369 | 0.9639 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 3 | R `ltmle` | -0.0032 to 0.0039 | 0.9369 | 0.9639 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 2 | `cleverly` | -0.0023 to 0.0026 | 0.9394 | 0.9695 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 2 | R `ltmle` | -0.0023 to 0.0026 | 0.9394 | 0.9695 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 3 | `cleverly` | -0.0028 to 0.0056 | 0.9263 | 0.9506 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 3 | R `ltmle` | -0.0028 to 0.0056 | 0.9263 | 0.9506 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=1]` | mean of the repeated outcome under the plan treat at every time at horizon t = 1 | `cleverly` | 0.000054 to 0.0021 | 0.9506 | 0.9924 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=1]` | mean of the repeated outcome under the plan treat at every time at horizon t = 1 | R `ltmle` | 0.000054 to 0.0021 | 0.9506 | 0.9924 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=2]` | mean of the repeated outcome under the plan treat at every time at horizon t = 2 | `cleverly` | -0.0017 to 0.0015 | 0.9356 | 0.9764 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=2]` | mean of the repeated outcome under the plan treat at every time at horizon t = 2 | R `ltmle` | -0.0017 to 0.0015 | 0.9356 | 0.9764 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=3]` | mean of the repeated outcome under the plan treat at every time at horizon t = 3 | `cleverly` | -0.0024 to 0.0026 | 0.9387 | 0.9868 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=3]` | mean of the repeated outcome under the plan treat at every time at horizon t = 3 | R `ltmle` | -0.0024 to 0.0026 | 0.9387 | 0.9868 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=1]` | mean of the repeated outcome under the plan treat at no time at horizon t = 1 | `cleverly` | -0.000803 to 0.000971 | 0.9494 | 1.0310 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=1]` | mean of the repeated outcome under the plan treat at no time at horizon t = 1 | R `ltmle` | -0.000803 to 0.000971 | 0.9494 | 1.0310 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=2]` | mean of the repeated outcome under the plan treat at no time at horizon t = 2 | `cleverly` | -0.0020 to 0.0012 | 0.9337 | 0.9678 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=2]` | mean of the repeated outcome under the plan treat at no time at horizon t = 2 | R `ltmle` | -0.0020 to 0.0012 | 0.9337 | 0.9678 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=3]` | mean of the repeated outcome under the plan treat at no time at horizon t = 3 | `cleverly` | -0.0027 to 0.0023 | 0.9300 | 0.9494 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=3]` | mean of the repeated outcome under the plan treat at no time at horizon t = 3 | R `ltmle` | -0.0027 to 0.0023 | 0.9300 | 0.9494 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=2]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 2 | `cleverly` | -0.0021 to 0.0016 | 0.9469 | 0.9927 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=2]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 2 | R `ltmle` | -0.0021 to 0.0016 | 0.9469 | 0.9927 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=3]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 3 | `cleverly` | -0.0022 to 0.0046 | 0.9250 | 0.9409 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=3]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 3 | R `ltmle` | -0.0022 to 0.0046 | 0.9250 | 0.9409 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 1 | 9.025e-11 | 2.830e-08 | 1.0000 | 0 | 8.452e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 2 | 5.167e-11 | 9.656e-09 | 1.0000 | 0 | 7.025e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 3 | 9.053e-11 | 1.091e-08 | 1.0000 | 0 | 3.308e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 2 | -3.659e-11 | 6.368e-09 | 1.0000 | 0 | 9.177e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 3 | -9.554e-08 | 0.000010 | 1.0000 | 0 | 7.733e-07 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=1]` | mean of the repeated outcome under the plan treat at every time at horizon t = 1 | -5.650e-12 | 2.362e-09 | 1.0000 | 0 | 2.257e-11 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=2]` | mean of the repeated outcome under the plan treat at every time at horizon t = 2 | -1.453e-10 | 3.833e-08 | 1.0000 | 0 | 5.071e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=3]` | mean of the repeated outcome under the plan treat at every time at horizon t = 3 | -7.719e-11 | 1.319e-08 | 1.0000 | 0 | 2.594e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=1]` | mean of the repeated outcome under the plan treat at no time at horizon t = 1 | -9.590e-11 | 4.648e-08 | 1.0000 | 0 | 1.323e-08 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=2]` | mean of the repeated outcome under the plan treat at no time at horizon t = 2 | -1.970e-10 | 5.272e-08 | 1.0000 | 0 | 8.595e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=3]` | mean of the repeated outcome under the plan treat at no time at horizon t = 3 | -1.677e-10 | 2.880e-08 | 1.0000 | 0 | 3.694e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=2]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 2 | -2.336e-10 | 5.463e-08 | 1.0000 | 0 | 7.672e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=3]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 3 | -9.571e-08 | 0.000012 | 1.0000 | 0 | 0.000002 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

The property families draw from their own streams of `make_longitudinal_repeated`. Their draws
differ from the primary draws. The table above lists every cell. This section states what each
family tests.

| family | what it tests | size |
| --- | --- | --- |
| `interval_calibration` | pointwise coverage and SE ratio inside the calibration bands. The cells are in-sample with estimated mechanisms, cross-fitted GLM, cross-fitted boosted trees, and a continuous outcome. The in-sample cells cover every horizon, $t = 1, 2, 3$. A shrunken-SE control and a noise control must fail the same bands | $n$ = 8,000, 3,200 replications |
| `small_sample_calibration` | the in-sample and continuous arms refitted at $n$ = 2,000. The role is `diagnostic`. The table reports these cells and no rule gates them | $n$ = 2,000, 2,000 replications |
| `double_robustness` | bias inside the margin when the outcome regression omits every $Y_{t-1}$ and $L_t$ term and the mechanisms are correct. A static plan also has a both-wrong control, which must show bias | $n$ = 2,000, 1,000 replications |
| `simultaneous_coverage` | the default band over all 15 reported parameters, against its pointwise control | $n$ = 8,000, 3,200 replications |
| `rule_necessity` | the declared rule against its reverse, which treats after a zero outcome. The reversed rule must miss the declared rule's truth | $n$ = 2,000, 1,000 replications |
| `repeated_outcome_necessity` (N1) | an absorbing reading of the outcome, which fits $\max_{s \le h} Y_s$. The control must miss the truth and reach the cumulative-maximum limit. Its outcome learner is a cell-means regression with a GLM fallback on an unseen cell | $n$ = 8,000, 1,000 replications |
| `outcome_history_necessity` (N2) | a fit that leaves every $Y_s$ out of the history. The control must miss the truth and reach the coarse g-formula limit | $n$ = 2,000, 1,000 replications |

The in-sample cells use $n$ = 8,000. At $n$ = 2,000 the standard error runs a few percent below
the sampling spread for the rule. The ratio moves toward 1 at the $1/n$ rate of in-sample fitting. The module `tests/studies/ltmle_repeated_properties.py` records the pilot behind that
choice. The `small_sample_calibration` cells report the shortfall at $n$ = 2,000.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 26 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 26 | of those, passing |
| `paired_tests_total` | 13 | paired comparisons with R `ltmle` |
| `paired_tests_passed` | 13 | of those, passing |
| `property_cells_total` | 62 | repeated-sampling property cells, reported cells excluded |
| `property_cells_passed` | 50 | cells whose own and family verdicts pass |
| `property_cells_reported` | 18 | cells reported and not gated, as the `small_sample_calibration` diagnostics |
| `max_standardized_bias` | 0.0679 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9250 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9003 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0818 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit. No cell of this study reads it |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit. No cell of this study reads it |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control multiplies the standard error by |
| `margin:necessity_displacement` | 0.2500 | smallest paired displacement of a necessity control, in empirical standard deviations |

## Limitations

| limit | what it means for use |
| --- | --- |
| one law with three nodes and a binary outcome | the continuous outcome has one calibration arm, and the rate in $n$ is not measured |
| the paired comparison fixes the mechanisms | it isolates the sequential regressions, targeting and inference. It does not claim parity for learned treatment or censoring models |
| the in-sample interval is mildly anticonservative at $n$ = 2,000 | the `small_sample_calibration` cells report the shortfall and no rule gates them. The gated cells draw $n$ = 8,000 |
| the cross-fitted learners are one GLM and one shallow booster | the study does not cover a larger learner library or a different fold count |
| the absorbing and history-free controls read exact limits of this law | they show that the repeated reading and the outcome history matter here. They do not bound the difference on another law |
| positivity is comfortable | no cell validates active truncation or near-positivity behavior |
| no recurrent-event or truncation-by-death law | the parameter is the mean in a population where censoring is prevented |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/manifest.json)
records the pinned R package, source commit, image digest, formulas, seeds and artifact hashes.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/replicates.csv.gz),
[paired verdicts](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/properties.csv)
carry every published row.
