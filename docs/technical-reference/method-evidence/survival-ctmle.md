# Collaborative TMLE of a risk curve

This study reads the point estimates of `SurvivalCTMLE` at the simulation laws of Stitelman and
van der Laan (2010), Sections 12 to 14. For each arm of a binary baseline treatment, the
estimator selects a main-term logistic treatment model and a pooled main-term logistic censoring
model greedily, by the penalized loss of the source's equation (24), and targets the risk at
each horizon through the hazard fluctuation.

The study publishes under the `reporting` policy, and it declares no property cell. A study with
no property cells is not a validation row, as
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row) states. The
[validation grid](validation-grid.md) therefore does not list it. The claim is points only
([F18](../../roadmap.md#f18-selector-path-c-tmle-inference)): the study
gates bias and an RMSE comparison, and it reports the coverage and the SE ratio of the plug-in
diagnostic `plugin_std_error` without a verdict.

**No canonical implementation is compared.** No maintained package fits collaborative TMLE of a
risk curve with right censoring. Two hazard `LTMLE` arms of this library, one with the full
treatment model and one without $W_4$, are fitted on every replicate as references. A zero-row
equivalence artifact records the absence of an external comparator.

## What was tested

| setting | declaration |
| --- | --- |
| covariates | $W_1, \dots, W_5$ multivariate normal, mean 0, variance 1, covariance 0.2, each clipped to $[-2, 2]$ |
| treatment | $P(A = 1 \mid W) = \operatorname{expit}(0.4 + 0.4 W_1 + 0.4 W_2 - 0.5 W_3 + \log(\mathrm{OR}) W_4)$, with $\mathrm{OR} \in \{1, 10, 15\}$. Scenario 2 sets the $W_1$ coefficient to $\log 15$ |
| event and censoring | event hazard $\operatorname{expit}(0.25 - 0.6 (W_1 + W_2 + W_3) - A)$; censoring hazard $\operatorname{expit}(-3 - 0.1 (W_1 + W_2 + W_3 + A))$, both constant on the grid $\{1, 2\}$ |
| truth | $F_1(2) = 0.538510$ and $F_0(2) = 0.741434$, by quadrature over the clipped law, accurate to about $10^{-8}$ |
| outcome model | correct in every gated cell: `GLM` on $W$, $A$ and the grid time, `hazard_pooling="pooled"` |
| gated cells | OR 1, 10 and 15 and scenario 2 at OR 10 and 15, at n = 1,000; OR 15 at n = 2,000; OR 10 with two outer folds |
| gated estimands | the risk of arm 1 at $t = 2$ and the risk difference at $t = 2$ |
| primary budget | 1,600 replicates per cell, from the committed pilot of 1,000 per cell by the shared sizing rule |
| reported cells | 400 replicates each: the misspecified outcome model ($A$ and $W_5$ only) at every primary and scenario-2 cell, n = 500 at each OR, OR 15 with tertile-binarized covariates, and OR 15 with the lower `g_bounds` at $10^{-8}$ |
| policy | `reporting` |

The source's Table 1 prints 0.462 for the survival of arm 1, which the unclipped law gives. The
clipped law gives 0.461490, so the risk is $F_1(2) = 0.538510$.

| gate | rule | artifact |
| --- | --- | --- |
| (a) bias | the standardized bias inside the 0.25 margin at the 99% bound, for both gated estimands in every gated cell | `performance-tests.csv` |
| (b) RMSE | the one-sided 99% upper bound of the RMSE ratio of `SurvivalCTMLE` over the full-$g$ hazard `LTMLE`, for $F_1(2)$ at OR 15, at most 1 | `rmse-comparison.csv` |
| (c) failures | the subject's fit-failure share under 0.01 | `run.log` |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 10, n = 1,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of the event by t = 2 between treatment and no treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0016 to 0.0030 | 0.8269 | 0.6977 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 10, n = 1,000 | `risk_regimen[a1 @ t=2]` | risk of the event by t = 2 under treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.000749 to 0.0022 | 0.9256 | 0.8813 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 10, two outer folds, n = 1,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of the event by t = 2 between treatment and no treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0027 to 0.0020 | 0.8350 | 0.7173 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 10, two outer folds, n = 1,000 | `risk_regimen[a1 @ t=2]` | risk of the event by t = 2 under treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.000399 to 0.0025 | 0.9219 | 0.9086 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 15, n = 1,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of the event by t = 2 between treatment and no treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0029 to 0.0017 | 0.8294 | 0.6983 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 15, n = 1,000 | `risk_regimen[a1 @ t=2]` | risk of the event by t = 2 under treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0019 to 0.0010 | 0.9275 | 0.9041 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 15, n = 2,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of the event by t = 2 between treatment and no treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0023 to 0.000968 | 0.8256 | 0.6958 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 15, n = 2,000 | `risk_regimen[a1 @ t=2]` | risk of the event by t = 2 under treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0016 to 0.000513 | 0.9169 | 0.8836 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 1, n = 1,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of the event by t = 2 between treatment and no treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.000789 to 0.0029 | 0.9156 | 0.8942 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 1, n = 1,000 | `risk_regimen[a1 @ t=2]` | risk of the event by t = 2 under treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.000890 to 0.0017 | 0.9425 | 0.9743 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 10 and the W1 coefficient of the treatment model log 15, n = 1,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of the event by t = 2 between treatment and no treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0051 to -0.000118 | 0.7788 | 0.6263 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 10 and the W1 coefficient of the treatment model log 15, n = 1,000 | `risk_regimen[a1 @ t=2]` | risk of the event by t = 2 under treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0034 to -0.000257 | 0.9094 | 0.8608 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 15 and the W1 coefficient of the treatment model log 15, n = 1,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of the event by t = 2 between treatment and no treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0046 to 0.000590 | 0.7600 | 0.6051 | pass |
| five clipped correlated normal covariates, a binary baseline treatment whose odds rise with W4, which does not affect the event, an event hazard and a censoring hazard at two time points; the outcome model correct, W4 odds ratio 15 and the W1 coefficient of the treatment model log 15, n = 1,000 | `risk_regimen[a1 @ t=2]` | risk of the event by t = 2 under treatment | `cleverly` SurvivalCTMLE: per arm, the greedy collaborative selection of main-term treatment and censoring models, with the hazard fluctuation and the printed penalty | -0.0027 to 0.000491 | 0.9044 | 0.8371 | pass |
<!-- /generated -->

Every gated row passes the bias verdict. The largest absolute standardized bias is 0.075, in the
risk of arm 1 in scenario 2 at OR 10, against the margin of 0.25. No subject fit failed.

## RMSE against the full treatment model

At OR 15 the RMSE ratio of $F_1(2)$ over the full-$g$ hazard `LTMLE` is 0.539, with the upper
bound 0.565, so gate (b) passes. The ratio is between 0.39 and 0.64 in every gated cell with
OR 10 or 15, and near 0.98 at OR 1, where $W_4$ does not predict treatment. Against the hazard
`LTMLE` without $W_4$ the ratio is between 0.98 and 1.01 in the scenario-1 cells and between
0.78 and 0.87 in scenario 2. `rmse-comparison.csv` holds every ratio.

The selection rarely admits $W_4$ into the treatment model at OR 10 or 15: at most 0.2% of arm-1
fits. At OR 1 it admits $W_4$ in 9.8% of arm-1 fits. Censoring is about 4.7% per interval, so
the selection adds $A$ to the censoring model in only 5% to 9% of fits.
`selection-readings.csv` holds the shares, the step counts and the run times.

## Reported readings

The fitted result refuses `std_error` and `ci`, so the study reads `plugin_std_error` and
`plugin_interval`, the ordinary curve at the selected candidate. Coverage of that interval reads
0.76 to 0.94 in the gated rows, and the SE ratio 0.61 to 0.97. Both are lowest for the risk
difference at OR 10 and 15, where the curve ignores the selection. These readings are the working-mechanism diagnostic the
design states, and they carry no verdict
([F18](../../roadmap.md#f18-selector-path-c-tmle-inference)).

The risk of arm 0 has no gate. Its bias is below 0.0015 in every gated cell, and its coverage
reads 0.79 to 0.92 (`arm0-risk.csv`).

With the outcome model misspecified, the standardized biases of `SurvivalCTMLE` stay within 0.29
in scenario 1. In scenario 2 its standardized bias for the risk difference reads -0.98 at OR 10
and -0.57 at OR 15, larger in size than either hazard `LTMLE` arm. `reported-cells.csv` holds every reported
cell, and `reported-replicates.csv` every replicate.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `independent_tests_total` | 14 | law tests against the truth |
| `independent_tests_passed` | 14 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `property_cells_total` | 0 | repeated-sampling property cells, none declared |
| `max_standardized_bias` | 0.0752 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.7600 | lowest measured primary coverage, reported only |
| `performance[cleverly-survival-ctmle/or15_qcorrect/risk_regimen[a1 @ t=2]]:standardized_bias` | -0.0187 | standardized bias of $F_1(2)$ at OR 15 |
| `performance[cleverly-survival-ctmle/or15_qcorrect/ate_regimen[a1 vs a0 @ t=2]]:standardized_bias` | -0.0169 | standardized bias of the risk difference at OR 15 |
| `summary[cleverly-survival-ctmle/or15_qcorrect/risk_regimen[a1 @ t=2]]:rmse` | 0.0224 | RMSE of $F_1(2)$ at OR 15 |
| `summary[cleverly-survival-ctmle/or15_qcorrect/risk_regimen[a1 @ t=2]]:coverage` | 0.9275 | plug-in coverage of $F_1(2)$ at OR 15, reported only |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Points only | The plug-in standard error ignores the selection and under-covers at OR 10 and 15. Use the point estimate; intervals stay experimental ([F18](../../roadmap.md#f18-selector-path-c-tmle-inference)) |
| No property cell | The study is not a validation row, and the validation grid does not list it |
| No cross-implementation evidence | No maintained package fits this construction. The references are this library's hazard `LTMLE` |
| The outcome model is correct in every gated cell | Under a misspecified outcome model the reported scenario-2 cells show a large bias |
| One law family, two time points | The readings hold for the declared laws, the main-term logistic candidates and the grid $\{1, 2\}$ |
| Light censoring | About 4.7% per interval, so the censoring selection is rarely exercised here. `tests/unit/test_survival_ctmle.py` holds its witness |

## Reproduction

The [regeneration script](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_ctmle/regenerate.py)
gives the run command, the run form and the red-cell readings declared before the run. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_ctmle/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_ctmle/run.log)
records the commit, the clean tree and the wall time. The
[replicates](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_ctmle/replicates.parquet)
carry every gated row, with both hazard `LTMLE` arms on each. The
[pilot](https://github.com/esbraun/cleverly-tmle/blob/main/tests/studies/survival_ctmle_pilot.csv.gz)
holds the draws that set the budget.
