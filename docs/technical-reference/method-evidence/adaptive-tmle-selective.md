# Adaptive TMLE under selective enrollment

This study measures `AdaptiveTMLEMethod` on a randomized trial pooled with external rows
(`TrialWithExternalData`, `TrialATE`) when enrollment into the trial depends on covariates and the
external bias is large and wiggly. [Adaptive TMLE](../adaptive-tmle.md) gives the estimand, the
estimator and its conditions. The study is the sibling of the
[trial study](adaptive-tmle-trial.md), whose nuisance learners are splines. Here the nuisances are
a Super Learner, whose cross-fitted cost is two orders higher, so the study has its own sample
size and budget.

The study declares no property cell and no reference, so it is not a validation row, as
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row) states, and
the [validation grid](validation-grid.md) does not list it. It keeps the `gated` policy of its
declared run. The run read `s6_pooled` within its margins. It read the bias leg of `s6_cv` at 0.20
error SDs, from a finite-sample term with no defect, and the repo owner accepted the study with
that reading by a named exception, as [The accepted reading](#the-accepted-reading) states.

**No canonical implementation is compared.** The R `atmle` comparator is declared and pending, as
a separate pairing record. It never blocks this study.

## What was tested

| setting | declaration |
| --- | --- |
| law | the paper's law (b) with selective enrollment, $S \sim \mathrm{Bernoulli}(\mathrm{expit}(-0.9 + 0.8 W_1 + 0.5 W_2))$, and an external bias of four residual standard deviations that the dictionary does not span. 1.4% of the covariate mass has an enrollment probability outside [0.05, 0.95] |
| fit | `study.identify(TrialATE(population)).estimate(method=AdaptiveTMLEMethod(...))` with a Super Learner of a spline GLM, shallow regularized gradient boosting and a lasso for each nuisance. Probabilities use `nnloglik` with clip (0.01, 0.99) and the outcome uses NNLS |
| forms | `s6_cv`, the cross-validated form, and `s6_pooled`, the pooled form with no cross-fitting of the working model |
| size | n = 3,200 pooled rows |
| primary budget | 2,400 replications of each scenario, the sizing rule's floor at a 1,200-replicate pilot |
| truth | quadrature of $\Psi$ from the law, and of the working-model projection on each fit's stored fold transformers, so the truth varies by replicate |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| control | K6, the same fit with main-terms GLM nuisances on the same draws. It is a reporting row because a pilot found no power at this size |
| seeds | `seed=20290321` and `resampling_seed=20290322` |
| policy | `gated`, with one named exception for the bias leg of `s6_cv` |

Both scenarios gate $\Psi$. The laws, scenarios, budgets and seeds were fixed before any run, in
`tests/studies/adaptive_tmle_selective.py`, and
`tests/unit/test_adaptive_tmle_trial_design.py` recomputes the truths and the sizing without a run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selective enrollment (the enrollment probability depends on W1 and W2) with a large wiggly external bias, cross-validated form, Super Learner nuisances, n = 3,200 | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data, with Super Learner nuisances (spline GLM, shallow regularized gradient boosting, lasso): a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | 0.0215 to 0.0364 | 0.9367 | 0.9530 | **fail** |
| selective enrollment with a large wiggly external bias, form='pooled' (no cross-fitting of the working model), Super Learner nuisances, n = 3,200 | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data, with Super Learner nuisances (spline GLM, shallow regularized gradient boosting, lasso): a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | 0.0063 to 0.0195 | 0.9371 | 0.9549 | pass |
<!-- /generated -->

## The accepted reading

The run (2,400 replications of each scenario, declared at `c6f9e51c`, artifacts committed as
written) gives `s6_pooled` a bias of 0.0129 (0.10 error SDs, 99% interval 0.0063 to 0.0195), a
coverage of 0.9371 and an SE ratio of 0.955. Every leg is inside its margin.

`s6_cv` gives a coverage of 0.9367 and an SE ratio of 0.953, both inside their gates, and a bias of
0.0290 (0.20 error SDs). Its 99% interval, 0.0215 to 0.0364, ends above the margin of 0.0355, so
the bias leg did not clear its declared margin. The diagnosis found no defect. The cross-validated
form's targeting step fits about 37 selected bias columns on 640 validation rows, and that ratio
leaves a finite-sample term of order $K/n_v$. The cross-validated form carries 0.0161 more bias than
the pooled form (0.0290 against 0.0129, on different draws). The working-model projection sits
0.0035 from $\Psi$ (0.025 error SDs), and the interval covers the projection in 0.939 of
replications, so the term is no gap between the two targets.

The one change that removed the term, a targeting direction fitted on the validation fold,
destabilized the trial study's separating law `e1` (its SE ratio fell from 0.97 to 0.86). It is
not adopted.

The repo owner decided on 2026-11-10 not to repeat the study and to accept it with this reading by
a named exception. The exception is `ACCEPTED_BY_EXCEPTION` in
`tests/studies/evidence/red_cells.py`. It names this study, the `s6_cv/psi` truth row and the
replication file of this run, and it admits no other row. It stops applying when the cell reads
green or the study is rerun. No margin, replication count, law, seed, learner or label moved.

## Controls

K6 fits the same forms with main-terms GLM nuisances on the same draws, as a reporting row. It
reads a coverage of 0.9358 for the cross-validated form and 0.8650 for the pooled form.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications per scenario |
| `n` | 3200 | observations per primary replication |
| `independent_tests_total` | 2 | scenario tests against the replication truth |
| `independent_tests_passed` | 1 | of those, passing |
| `subject_failures` | 0 | failed primary fits |
| `performance[cleverly-adaptive-tmle-selective/s6_cv/psi]:bias` | 0.0290 | `s6_cv`, bias of $\Psi$ |
| `performance[cleverly-adaptive-tmle-selective/s6_cv/psi]:bias_ci_lower` | 0.0215 | its 99% lower endpoint |
| `performance[cleverly-adaptive-tmle-selective/s6_cv/psi]:bias_ci_upper` | 0.0364 | its 99% upper endpoint, against the margin |
| `performance[cleverly-adaptive-tmle-selective/s6_cv/psi]:bias_margin` | 0.0355 | the bias margin on this scale |
| `performance[cleverly-adaptive-tmle-selective/s6_cv/psi]:standardized_bias` | 0.2038 | `s6_cv`, bias in error SDs |
| `performance[cleverly-adaptive-tmle-selective/s6_cv/psi]:coverage` | 0.9367 | `s6_cv`, coverage of $\Psi$ |
| `performance[cleverly-adaptive-tmle-selective/s6_cv/psi]:se_ratio` | 0.9530 | `s6_cv`, SE ratio |
| `performance[cleverly-adaptive-tmle-selective/s6_pooled/psi]:bias` | 0.0129 | `s6_pooled`, bias of $\Psi$ |
| `performance[cleverly-adaptive-tmle-selective/s6_pooled/psi]:bias_ci_lower` | 0.0063 | its 99% lower endpoint |
| `performance[cleverly-adaptive-tmle-selective/s6_pooled/psi]:bias_ci_upper` | 0.0195 | its 99% upper endpoint |
| `performance[cleverly-adaptive-tmle-selective/s6_pooled/psi]:standardized_bias` | 0.1025 | `s6_pooled`, bias in error SDs |
| `performance[cleverly-adaptive-tmle-selective/s6_pooled/psi]:coverage` | 0.9371 | `s6_pooled`, coverage of $\Psi$ |
| `performance[cleverly-adaptive-tmle-selective/s6_pooled/psi]:se_ratio` | 0.9549 | `s6_pooled`, SE ratio |
| `reported[s6_cv]:projection_coverage` | 0.9387 | `s6_cv`, coverage of the projection |
| `reported[s6_cv]:projection_bias_sd` | -0.0247 | `s6_cv`, projection minus $\Psi$ in error SDs |
| `controls[k6_main_terms_cv/s6_cv]:value` | 0.9358 | K6, coverage of $\Psi$ of the cross-validated form |
| `controls[k6_main_terms_pooled/s6_pooled]:value` | 0.8650 | K6, coverage of $\Psi$ of the pooled form |
| `margin:standardized_bias` | 0.2500 | bias margin in error SDs |
| `margin:coverage_floor` | 0.9000 | coverage floor |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 0 | repeated-sampling property cells, none declared |
| `property_cells_passed` | 0 | of those, passing |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 24 | failed primary fits that bound allows |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
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
| No grid row | The study declares no property cell and no reference, so the validation grid does not list it and the method has no validation row |
| There is no cross-implementation evidence | The R `atmle` pairing is declared, pending, and not run |
| One size and one learner class | Every scenario draws n = 3,200 with one Super Learner library. No other size or library is measured |
| The interval covers the projection | The coverage of $\Psi$ is read only where the dictionary approximates the external bias, and no data check tells the cases apart |
| Binary treatment, difference scale | Other treatments and scales refuse before any learner |
| Compositions outside the study | Weights, clusters, a second outcome and repeated fits are not measured |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_selective/README.md)
gives the run command, the sizing and the declaration history. The committed artifacts carry the
replications, the controls and the reported readings of the run.
