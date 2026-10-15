# C-TMLE candidate sequences

This study validates the lasso and the declared-learner candidate sequences of `cleverly`'s
C-TMLE. The canonical comparison uses R [`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at
pinned commit
[`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c).
The theory is van der Laan and Gruber (2010).

The evidence on this page is Python property evidence, with comparator runs pending. The Python
phase and the property study ran. The R `ctmle` runs wait for Study queue item 9, so this page
has no paired row and no replay result yet. The `reporting` publication policy was declared
before the run.

## What was compared

| setting | `cleverly` | R `ctmle` |
| --- | --- | --- |
| scenarios | `binary_lasso` and `binary_learners` | `ctmleGlmnet` for `binary_lasso`, and `build_gn_seq` with `ctmleGeneral` for `binary_learners` |
| laws | `x16_lasso_law` for `binary_lasso` and `x16_learners_law` for `binary_learners`. Each law has four covariates, `W1` to `W4`. `W1` is an instrument, and `W3` is a confounder and an effect modifier | the identical rows |
| candidate sequence | `binary_lasso`: the 20 penalties of its grid in `lambdas.csv` and the penalty 0. `binary_learners`: four lasso learners at entries 2, 4, 16 and 19 of its grid, and a main-terms logistic terminal | the same penalties, or the same four learners and `SL.glm` |
| datasets | binary-outcome samples generated in Python with their exact ATE | the identical rows and all four covariates |
| selection folds | `binary_lasso`: an unstratified five-fold split that the fit asserts. `binary_learners`: R's folds from `r_folds.csv.gz` | `binary_lasso`: the package's own five folds. `binary_learners`: the same folds, drawn again from the replicate's seed |
| outcome regression | a logistic GLM on `A` and `W1` to `W4`. It omits the `A W3` term of both laws | the same GLM |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| construction loss | the loss only (`penalty=False`) | the loss plus $\operatorname{var}(D^*)$, whatever `PEN` is. Inside each fold, R multiplies that term by $n_{\text{train}}/n$ |
| selection loss | the cross-validated loss, unpenalized | the cross-validated loss, unpenalized (`PEN = FALSE`) |
| cross-fitting | disabled for the comparison | not applicable |
| variance | `plugin_std_error`, the plug-in variance of $D^*$ | `var.psi` of the type-1 final step, or `varDstar_origscale[best_k] / n` |

The outcome GLM is misspecified on both laws. Each candidate's TMLE limit therefore depends on
how much of `W3` its mechanism reads. `tests/unit/test_ctmle_sequences_design.py` asserts the
spread of these limits before the run.

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_sequences/README.md#declared-differences)
lists each declared difference between the two fits. R refits the fluctuation on every row
inside each fold (quirk 2). This leak drives R's cross-validation toward the last candidate, so
the two selected estimates are not one estimator. The paired rows of the agreement section are
therefore descriptive. They carry no `passed` column. The red-cell ledger reads none of them.

The file `r-selection.csv` holds R's selected candidate of each replicate. The package's
cross-validated index, final index and accepted candidates stand beside it. The file
`r-selection-summary.csv` counts the type-1 adapter replicates. Its paired interval reads only
the replicates where `ctmleGlmnet` completed without the adapter. The interval is therefore
conditional on that event.

The replay gates carry the comparison with R. R writes its outcome and propensity coefficients.
The module `tests/studies/ctmle_sequence_replay.py` rebuilds R's arrays from them. It then runs
the package's construction and final step on R's arrays and R's initial outcome fit.

| replay gate | rule, per scenario |
| --- | --- |
| accepted candidates | identical on every unflagged replicate |
| candidate estimates and losses | within $10^{-6}$, absolute for an estimate and relative for a loss |
| final step, `binary_lasso` | for each start that R writes, the final index is identical, and the estimate and every option loss are within $10^{-6}$ |
| counts | at least 400 unflagged replicates, and for `binary_lasso` at least 100 final-step comparisons |

The replay leaves out a flagged replicate. A replicate is flagged when a deciding score gap lies
below $10^{-6}$ of the round's best score. The `clip` flag reads the bound
$[1 - \alpha, \alpha]$, which the package applies to each targeted outcome value. R applies it
only to the initial fit and to each accepted update.

| scenario | what R writes | `clip` flags a replicate when |
| --- | --- | --- |
| `binary_lasso` | unbounded estimates and losses | a targeted value of the package lies at or outside a bound |
| `binary_learners` | estimates and losses from R's targeted values, bounded as the package bounds them | a targeted value of the package on an observed row lies at or outside a bound. On those rows the two fluctuation fits solve different equations |

The `clip` flag reads every try that the construction scores. This includes a try that the path
discards.

The file `replay-summary.csv` records the counts, the flags and the largest gap of each kind. It
also records the largest gaps on the flagged replicates. No gate reads those gaps.

The property study uses a public nested cross-fit configuration: five outer folds, three
selection folds, and two inner folds. Every two-arm property cell samples a proportion and
declares an outcome support of 0 to 1. The two multi-arm cells sample the binary three-arm law.
`property-selection.csv` records the cross-validated index and the final index of every lasso
and learners property fit.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, lasso candidate sequence | `ate` | average treatment effect | `cleverly` C-TMLE lasso and declared-learner sequences | -0.0032 to 0.0034 | 0.9237 | 0.9051 | **fail** |
| binary-outcome law, four declared lasso learners | `ate` | average treatment effect | `cleverly` C-TMLE lasso and declared-learner sequences | -0.0032 to 0.0027 | 0.9500 | 0.9945 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

Python property evidence, with comparator runs pending. The R `ctmle` runs are not registered
yet, so the committed `equivalence.csv` has no rows. The replay gates above read R's output, and
they run when a later change registers the R runs.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_wrong_lasso` | control | the lasso sequence with a constant outcome regression, so both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0432 to 0.0446, margin 0.0024, SE ratio 1.2533 | pass |
| `double_robustness` | `both_wrong_learners` | control | the declared learners with a constant outcome regression, so both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0433 to 0.0448, margin 0.0025, SE ratio 1.2155 | pass |
| `double_robustness` | `lasso_outcome_correct` | positive | only the outcome regression is correct, and no lasso candidate is a correct mechanism | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000664 to 0.000605, margin 0.0021, SE ratio 0.9559 | pass |
| `double_robustness` | `lasso_treatment_correct` | positive | only the treatment mechanism is correct, and the lasso path contains it | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000666 to 0.000248, margin 0.0015, SE ratio 1.2702 | pass |
| `double_robustness` | `learners_outcome_correct` | positive | only the outcome regression is correct, and no declared learner is a correct mechanism | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0012 to 0.000072, margin 0.0021, SE ratio 0.9377 | pass |
| `double_robustness` | `learners_treatment_correct` | positive | only the treatment mechanism is correct, and the terminal learner specifies it | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000663 to 0.000279, margin 0.0016, SE ratio 1.2149 | pass |
| `multi_arm` | `empty_control` | control | the selector is forced to stop at an empty path on the three-arm law | bias interval must fall entirely outside the margin | bias 0.0692 to 0.0729, margin 0.0062 | pass |
| `multi_arm` | `lasso_treatment_correct` | positive | the lasso sequence on the three-arm law, with a constant outcome regression | bias interval inside the equivalence margin | bias -0.0034 to 0.000487, margin 0.0065 | pass |
| `selector_necessity` | `empty_control` | control | the selector is forced to stop at an empty path | bias interval must fall entirely outside the margin | bias 0.0494 to 0.0511, margin 0.0023 | pass |
| `selector_necessity` | `lasso_collaborative` | positive | the selector chooses its own penalty along the lasso sequence | bias interval inside the equivalence margin, and RMSE at most the declared ratio of the paired control's | bias -0.0025 to -0.000412, margin 0.0028, RMSE ratio 0.2205, 99% upper 0.2333 | pass |
| `selector_necessity` | `learners_collaborative` | positive | the selector chooses its own learner along the declared nested sequence | bias interval inside the equivalence margin, a declared expected red, and RMSE at most the declared ratio of the paired control's | bias 0.0029 to 0.0051, margin 0.0031, RMSE ratio 0.2542, 99% upper 0.2721 | **fail** |
<!-- /generated -->

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications per scenario |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 2 | package truth tests |
| `independent_tests_passed` | 1 | package truth tests passing |
| `property_cells_total` | 11 | repeated-sampling property cells |
| `property_cells_passed` | 8 | property cells passing |
| `properties[selector_necessity/learners_collaborative]:standardized_bias` | 0.3246 | the declared expected red bias, in empirical standard deviations |
| `properties[selector_necessity/learners_collaborative]:rmse_ratio_upper` | 0.2721 | the bootstrap 99% upper bound of its RMSE ratio, the discriminating gate |
| `properties[selector_necessity/learners_collaborative]:coverage` | 0.9487 | its coverage of the plug-in diagnostic interval |
| `properties[selector_necessity/lasso_collaborative]:rmse_ratio_upper` | 0.2333 | the same bound for the lasso sequence |
| `max_standardized_bias` | 0.0076 | largest absolute standardized bias of a package truth row |
| `min_coverage` | 0.9237 | lowest coverage of a package truth row |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples for bootstrap intervals |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical SDs |
| `margin:coverage_floor` | 0.9000 | lower coverage gate |
| `margin:over_coverage_ceiling` | 0.9900 | reported conservative-coverage ceiling |
| `margin:se_ratio_sanity_lower` | 0.8000 | lower SE-ratio screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | upper SE-ratio screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | lower calibration band |
| `margin:calibration_se_ratio_upper` | 1.0700 | upper calibration band |
| `margin:calibration_coverage_lower` | 0.9200 | lower calibration-coverage band |
| `margin:calibration_coverage_upper` | 0.9800 | upper calibration-coverage band |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate |
| `margin:paired_difference` | 0.1500 | paired similarity margin in empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio non-inferiority margin |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference non-inferiority margin |
| `margin:calibration_noninferiority` | 0.0500 | SE-calibration non-inferiority margin |
| `margin:minimum_power` | 0.8000 | lower power gate |
| `margin:root_n_slope` | -0.5000 | root-n contraction target |
| `margin:root_n_slope_lower` | -0.6250 | lower contraction band |
| `margin:root_n_slope_upper` | -0.3750 | upper contraction band |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:selector_rmse_ratio` | 0.5000 | required bound on each collaborative-to-control RMSE ratio |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## Declared design

The values in this table are declared before the run.

| setting | value | source |
| --- | --- | --- |
| paired role | descriptive, when the R runs are registered | the record's comment in `tests/studies/canonical_ctmle_sequences.py` |
| replay tolerance | $10^{-6}$ | `GLM_TOLERANCE` in `tests/studies/ctmle_sequence_replay.py` |
| replay counts | 400 unflagged replicates per scenario, 100 final-step comparisons for `binary_lasso` | `MIN_UNFLAGGED` and `MIN_FINAL` in the same module |
| selector RMSE ratio | 0.50, applied to the paired bootstrap 99% upper bound of each ratio | `SELECTOR_RMSE_RATIO` and `RMSE_BOUND_LEVEL` in `tests/studies/ctmle_sequence_properties.py` |
| property grid | 11 penalties and the penalty 0 | `PROPERTY_GRID_SIZE` in the same module |
| red-cell owners | F18 for every coverage or standard-error-ratio red cell, and for every `selector_necessity` red cell. `X16-sequence-reading` for a red package bias row, a red replay gate, or a positive double-robustness cell that is red on its bias. `X16-plugin-variance` for the `binary_lasso` coverage row, which moved from F18 after the first run | `RED_CELL_OWNERS` in `tests/studies/canonical_ctmle_sequences.py` |
| expected red rows | the `binary_lasso` coverage row. The bias clause of `selector_necessity/learners_collaborative`, where the binary selector precedent is 0.217 SD (design review S3). The other two necessity cells by the family's joint clause. The RMSE bound is the necessity family's discriminating gate | `EXPECTED_RED` in `tests/studies/canonical_ctmle_sequences.py` |
| expected adapter rate | close to every `binary_lasso` replicate. On the rev 2.1 law, R selected the last penalty on 10 of 10 probe replicates. The package's CV-selected index is a different quantity | the fixture README, "Expected adapter rate" |
| learners candidates of the property cells | a main-terms logistic regression on `W2`, then on `W1` and `W2`. The sequence is nested, so its treatment loss decreases | `LEARNER_COLUMNS` in `tests/studies/ctmle_sequence_properties.py`. On the instrument law, index 0 omits the confounder. `tests/unit/test_ctmle_sequences_design.py` reads each index along the built path. Index 0 lies outside the margin, and index 1 and the terminal lie inside it |

## Limitations

The [red-cell ledger](red-cells.md) lists each red cell of a study with the roadmap ask that owns
it. Every red row of this study is a declared expected red.

| limitation | what it means for use |
| --- | --- |
| The R comparison is pending | The page holds Python property evidence, with comparator runs pending. No paired row and no replay gate is evidence yet |
| The `binary_lasso` coverage row is red | Coverage is 0.92375 and the SE ratio 0.905. Plain TMLE with the same learners and bounds on the same 800 samples covers 0.9225 with an SE ratio of 0.913. The final step reports the penalty 0 on 771 of 800 fits. The cause is the plug-in variance under near-positivity, not the selection, and the row does not test $\Delta(P_1)$ of F18 ([`tests/diagnostics/x16_lasso_coverage/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x16_lasso_coverage)). The package refuses `ci`, `pvalue` and `std_error` on a lasso fit |
| The first run's necessity cell was a design defect | The first run declared `W2`, then `W1`. That order breaks the decreasing treatment loss, and `learners_collaborative` read a bias of 0.575 SD with an RMSE bound of 0.511 against 0.50. The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_sequences/README.md#history) records that run. The committed cells use the nested order |
| R refits on every row inside each fold | R refits the fluctuation on every row, and its evaluation refits the lasso path on every row. `cleverly` refits inside each selection fold. The selected candidate can differ for this reason alone. The paired rows are descriptive for this reason |
| R penalizes its construction | R adds $\operatorname{var}(D^*)$ to each construction score, whatever `PEN` is. `cleverly` with `penalty=False` does not. The replay gates use `penalty=True` on both sides for this reason |
| A green coverage or SE-ratio row says nothing about F18 | Where the outcome model is correct, Ju et al. (2018) Lemma 2 gives $\Delta(P_1) = 0$, so coverage holds conditionally on the outcome fit and its limit, which F18's reopening condition excludes. Where the outcome model is wrong, one law's coverage bounds no remainder term |
| The multi-arm lasso cell tests the unpenalized end of the path | The cell reported that end on every smoke seed. It does not test interior selection at three arms |
| The lasso final step returns to the unpenalized end on most fits | On these laws the lasso estimate and plain TMLE are close in distribution. The replay gates, not the truth gates, show that the sequence is built as R builds it. The `binary_learners` path accepted one candidate on 200 of 200 replicates of the design review 2 emulation, and the terminal on 151. Its replay checks the first round of fluctuations, and one round after one update on the other 49. The `binary_lasso` replay checks the longer update chain: 78 of 160 emulated replicates accepted two or more candidates |
| The parity claim is narrow | It covers binary, two-arm, complete-outcome, GLM-outcome, non-cross-fitted ATE estimation only |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_sequences/README.md)
records the declaration inputs and the regeneration commands.
