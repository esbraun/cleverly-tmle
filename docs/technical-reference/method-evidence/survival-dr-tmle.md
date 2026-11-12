# Drift-corrected survival TMLE

This study validates `SurvivalDRTMLE` in `cleverly.longitudinal`, the hazard TMLE of a risk
curve with the drift correction of Díaz (2019), Section 5, under the corrected eq. (9). The
parameter is the risk $F_a(5)$ of each arm of a binary baseline treatment at $t = 5$, and their
difference, under right censoring on five time points. The
[time-to-event section](../dr-tmle/theorem.md#time-to-event-outcomes) states the estimator, the
four corrections to the source and the case table of what each fit holds and subtracts.

No maintained package implements this construction, so no canonical implementation is compared.
The study rests on exact truths and on repeated-sampling properties. Hazard `LTMLE` is fitted on
every primary replicate as a reference, and `paired-difference.csv` records the two RMSEs.

## What was fitted

| scenario | law, nuisances and estimands | folds |
| --- | --- | --- |
| `both_1k`, `both_4k`, `both_16k` | law L-S with the observational treatment, $P(A = 1 \mid W) = \operatorname{expit}(0.6 W_1 + 0.3 W_2)$. Per-node logistic models for the event hazard ($W_1, W_2, A$), the censoring hazard at nodes 2 to 5 ($W_1, W_2, A$) and the treatment ($W_1, W_2$), all correct; the risk of each arm at $t = 5$ and the risk difference, at n = 1,000, 4,000 and 16,000 | ten |
| `known_1k`, `known_4k`, `known_16k` | law L-S with $A \sim \operatorname{Bern}(0.5)$. The treatment and retention probabilities are declared and held, and the hazard model omits $W_2$, so it is wrong; the same estimands and sizes | ten |

Law L-S draws $W_1, W_2$ independent standard normal. The event fires at $t = 1, \dots, 5$ with
hazard $\operatorname{expit}(-2 + 0.5 A - W_1 + 2 W_2 + 0.1 t)$, and an event-free unit is then
censored with hazard $\operatorname{expit}(-2.5 + 0.4 W_1 + 0.6 W_2)$ for $t < 5$. The law is in
`tests/studies/survival_dr_tmle_law.py`. The truths, $F_1(5) = 0.628096$ and
$F_0(5) = 0.552638$, come from a Gauss-Hermite rule on the one index $-W_1 + 2 W_2$ that the hazard
reads. The efficient spreads come from a product rule over the exact efficient influence curve.
`tests/unit/test_survival_dr_tmle_design.py` checks both against Monte Carlo on the law.

The `known_*` scenarios witness the held-factor rule through the arm rows. With both mechanisms
declared the remainder is exactly zero for any hazard model while the `g_bounds` floor does not
bind, and the wrong hazard model puts an untargeted plug-in about 3.6 efficient standard errors
off each arm's risk at n = 200,000. The risk-difference row of these scenarios is gated but is
no witness: that error cancels between the randomized arms (+0.03 standard errors). With every
mechanism declared the drift block is zero, so these fits equal the hazard `LTMLE` fit; the paired
differences in `paired-difference.csv` are below $2 \times 10^{-6}$ pooled standard deviations.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 16,000 | `ate_regimen[a1 vs a0 @ t=5]` | difference in the risk of the event by t = 5 between treatment and no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000339 to 0.000089 | 0.9457 | 0.9832 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 16,000 | `risk_regimen[a0 @ t=5]` | risk of the event by t = 5 under no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000114 to 0.000251 | 0.9497 | 0.9989 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 16,000 | `risk_regimen[a1 @ t=5]` | risk of the event by t = 5 under treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000227 to 0.000113 | 0.9480 | 0.9999 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 1,000 | `ate_regimen[a1 vs a0 @ t=5]` | difference in the risk of the event by t = 5 between treatment and no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000684 to 0.0010 | 0.9415 | 0.9811 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 1,000 | `risk_regimen[a0 @ t=5]` | risk of the event by t = 5 under no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000865 to 0.000607 | 0.9422 | 0.9768 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 1,000 | `risk_regimen[a1 @ t=5]` | risk of the event by t = 5 under treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000650 to 0.000727 | 0.9430 | 0.9926 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 4,000 | `ate_regimen[a1 vs a0 @ t=5]` | difference in the risk of the event by t = 5 between treatment and no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000226 to 0.000622 | 0.9482 | 0.9902 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 4,000 | `risk_regimen[a0 @ t=5]` | risk of the event by t = 5 under no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000624 to 0.000099 | 0.9513 | 1.0053 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), an observational treatment; every nuisance a correct logistic model, ten outer folds, n = 4,000 | `risk_regimen[a1 @ t=5]` | risk of the event by t = 5 under treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000404 to 0.000275 | 0.9525 | 1.0027 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 16,000 | `ate_regimen[a1 vs a0 @ t=5]` | difference in the risk of the event by t = 5 between treatment and no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000292 to 0.000244 | 0.9560 | 1.0091 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 16,000 | `risk_regimen[a0 @ t=5]` | risk of the event by t = 5 under no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000198 to 0.000197 | 0.9528 | 1.0075 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 16,000 | `risk_regimen[a1 @ t=5]` | risk of the event by t = 5 under treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000217 to 0.000167 | 0.9533 | 1.0022 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 1,000 | `ate_regimen[a1 vs a0 @ t=5]` | difference in the risk of the event by t = 5 between treatment and no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000649 to 0.0015 | 0.9553 | 1.0105 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 1,000 | `risk_regimen[a0 @ t=5]` | risk of the event by t = 5 under no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000971 to 0.000625 | 0.9457 | 0.9964 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 1,000 | `risk_regimen[a1 @ t=5]` | risk of the event by t = 5 under treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000507 to 0.0010 | 0.9550 | 1.0176 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 4,000 | `ate_regimen[a1 vs a0 @ t=5]` | difference in the risk of the event by t = 5 between treatment and no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000645 to 0.000452 | 0.9485 | 0.9870 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 4,000 | `risk_regimen[a0 @ t=5]` | risk of the event by t = 5 under no treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000361 to 0.000433 | 0.9472 | 1.0011 | pass |
| two normal covariates, a binary baseline treatment, an event hazard and a censoring hazard at five time points (law L-S), a randomized treatment; the treatment and censoring probabilities declared and held, the hazard model wrong (it omits W2), ten outer folds, n = 4,000 | `risk_regimen[a1 @ t=5]` | risk of the event by t = 5 under treatment | `cleverly` SurvivalDRTMLE: hazard TMLE with the drift loop that also targets the residuals of the treatment and censoring fits, Díaz (2019) Section 5 with the corrected eq. (9) | -0.000446 to 0.000325 | 0.9488 | 0.9967 | pass |
<!-- /generated -->

Every primary row passes. Coverage is 0.9415 to 0.9560 and the SE ratio 0.977 to 1.018 over
6,000 replicates per cell, and no fit failed.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `a1_t5__both_wrong` | control | law L-S (two normal covariates, five time points, censoring), the risk of the event by t = 5 under treatment, ten outer folds, n = 16,000: the hazard, treatment and censoring models all wrong: no sufficient condition holds | bias interval must fall entirely outside the margin | bias 0.0254 to 0.0258, margin 0.0011, SE ratio 1.0224 | pass |
| `double_robustness` | `a1_t5__mechanism_correct` | positive | law L-S (two normal covariates, five time points, censoring), the risk of the event by t = 5 under treatment, ten outer folds, n = 16,000: the hazard model wrong (it omits W2), with correct treatment and censoring models | bias interval inside the equivalence margin | bias -0.000106 to 0.000287, margin 0.0011, SE ratio 1.0871 | pass |
| `double_robustness` | `a1_t5__outcome_correct` | positive | law L-S (two normal covariates, five time points, censoring), the risk of the event by t = 5 under treatment, ten outer folds, n = 16,000: the hazard model correct, with the treatment model wrong (it omits W2) and a constant censoring model | bias interval inside the equivalence margin | bias -0.000118 to 0.000242, margin 0.0010, SE ratio 0.9981 | pass |
| `interval_calibration` | `both_rd__correctly_specified` | positive | law L-S, the difference in the risk of the event by t = 5 between treatment and no treatment, ten outer folds, n = 16,000: every nuisance (hazard, treatment and censoring) a correct logistic model, read against the exact efficient influence-curve spread with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands, with both efficiency-ratio intervals inside their bands | coverage 0.9351 to 0.9548, SE ratio 0.9669 to 1.0248, empirical efficiency ratio 0.9727 to 1.0314, reported efficiency ratio 0.9961 to 0.9981 | pass |
| `interval_calibration` | `both_rd__noise_control` | control | law L-S, the difference in the risk of the event by t = 5 between treatment and no treatment, ten outer folds, n = 16,000: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8096 to 0.8423, SE ratio 0.6816 to 0.7239, empirical efficiency ratio 1.3770 to 1.4623, reported efficiency ratio 0.9961 to 0.9981 | pass |
| `interval_calibration` | `both_rd__shrunken_se_control` | control | law L-S, the difference in the risk of the event by t = 5 between treatment and no treatment, ten outer folds, n = 16,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8093 to 0.8421, SE ratio 0.6771 to 0.7186, empirical efficiency ratio 0.9713 to 1.0307, reported efficiency ratio 0.6973 to 0.6986 | pass |
<!-- /generated -->

The robustness cells read the risk of arm 1 at $t = 5$ on the observational law at n = 16,000.
`mechanism_correct` fits the hazard model without $W_2$ beside correct treatment and censoring
models, and `outcome_correct` fits the correct hazard model beside a treatment model without
$W_2$ and a constant censoring model. Each wrong model keeps the level sets of its index on this
law only approximately, so these cells measure the drift identity at a finite sample, not its
exact form. With all three models wrong the bias is 4.7 standard deviations, far outside the
margin, so the cells can fail.

The calibration cell reads the risk difference against the exact efficient spread of
`tests/studies/survival_dr_tmle_law.py` and the efficiency band 0.90 to 1.10. The two shared
controls are derived from its rows. `tests/studies/survival_dr_tmle_properties.py` declares
every cell.

## Reported readings

These readings carry no verdict.

**The drift loop.** `drift-readings.csv` gives, per scenario, the mean iterations per arm, the
share of fits whose loop met the stopping rule, the fits with a floored denominator and the run
times. The loop met its rule in 97.2% of the `both_16k` fits, 94.2% at 4,000 and 89.3% at 1,000;
the rest stopped at `max_iter` with a warning and are kept in every row. A floored denominator,
which makes the drift covariate an approximation, occurred in 5,257 of the 6,000 `both_16k` fits
and 3,546 at 1,000. Every `known_*` fit converged.

**Against hazard `LTMLE`.** In the `both_*` scenarios the two RMSEs agree within 1% at 4,000 and
16,000. At 1,000 the drift fit's RMSE is lower for the risk of arm 0 (0.0221 against 0.0236) and
for the risk difference (0.0256 against 0.0267).

**Descriptive cells.** `reported-cells.csv` holds seven cells of 400 replicates. They are
descriptive, not controls, and no predicted drift term was computed for them.
`slow_censoring_<n>` draws the trial law, declares the treatment, and fits the hazard model
wrong beside the correct censoring model plus the logit offset $4 n^{-1/3} W_2$.
`slow_hazard_<n>` fits the hazard model correct plus the same offset. Both read standardized
biases within 0.55 at every size. At $c = 4$ the offset moves the arm risk by well under two standard deviations, so a small reading
shows neither validity nor failure under a slowly converging nuisance.

`insample_4k` fits gradient
boosting for the treatment and censoring in sample with the hazard model wrong, Theorem 2 without
its condition C2. It is a failure mode: its standardized biases reach -4.3 and its coverage
0.055 to 0.35.

## Sizing

Every size was fixed before the run from the pilot in `tests/studies/survival_dr_tmle_pilot.py`,
on seed streams no declared cell reads. The primary count, 6,000, follows the shared rule of
`tests/studies/repeated_sizing.py` with one implementation. The calibration count, 3,600, is the
smallest multiple of 400 whose family-wise pass probability reaches the shared share, read from
8,000 pilot draws. The robustness count, 5,200, is the smallest multiple of 400 at which each
positive cell passes with probability above 0.98 at its pilot's 80% upper bias; `outcome_correct`
sets it. The design test recomputes all three.

## Red-cell route

The study is gated, and no cell is red. The declared readings are in
`tests/canonical/survival_dr_tmle/regenerate.py`.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 6000 | primary replications |
| `n` | 4000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 18 | truth tests passing |
| `independent_tests_total` | 18 | truth tests reported |
| `property_cells_passed` | 6 | property cells passing |
| `property_cells_total` | 6 | property cells reported |
| `max_standardized_bias` | 0.0242 | largest primary standardized bias |
| `min_coverage` | 0.9415 | lowest primary coverage |
| `summary[cleverly-survival-dr-tmle/both_16k/ate_regimen[a1 vs a0 @ t=5]]:coverage` | 0.9457 | coverage of the risk difference, `both_16k` |
| `summary[cleverly-survival-dr-tmle/known_16k/risk_regimen[a1 @ t=5]]:coverage` | 0.9533 | coverage of the risk of arm 1 with the hazard model wrong, `known_16k` |
| `properties[double_robustness/a1_t5__both_wrong]:standardized_bias` | 4.7337 | standardized bias of the both-wrong control |
| `properties[interval_calibration/both_rd__correctly_specified]:se_ratio` | 0.9953 | SE ratio of the calibration cell |
| `properties[interval_calibration/both_rd__correctly_specified]:coverage` | 0.9456 | coverage of the calibration cell |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2000 | standardized-bias margin |
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
| `margin:subject_failures_allowed` | 60 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |

## Limits

| limit | what it means for use |
| --- | --- |
| GLM learners on one two-covariate law | the evidence covers per-node logistic models on law L-S only; Díaz's Section 6 simulation is not reproduced |
| the drift identity under a wrong model | the first-order drift term under a misspecified index rests on the exact-law check 5(b) of `tests/unit/test_survival_drtmle.py` alone; the robustness cells measure two wrong models on one law |
| slowly converging nuisances | the slow cells are descriptive at $c = 4$; the page claims neither validity nor failure |
| in-sample flexible mechanisms | `insample_4k` shows a large bias; cross-fit a flexible mechanism |
| the drift loop does not always meet its rule | 10.7% of the `both_1k` fits and 2.8% at 16,000 stop at `max_iter` and are kept |
| floored denominators are common | most `both_*` fits floor at least one product; the drift covariate is then an approximation |
| one event type, one binary baseline treatment, baseline covariates | competing risks and time-varying treatment are refused |
| no comparator | no maintained implementation was located |
| no weights, clusters or missing outcomes | each is a separate composition and reads experimental |

## Reproduction

`tests/canonical/survival_dr_tmle/regenerate.py` regenerates the study and states the run form.
It runs no reference process, because the study has no comparator. The manifest records the
seeds, the margins, the estimator configuration and the source and result hashes, and the run log
records the commit, the clean tree and the wall time.
