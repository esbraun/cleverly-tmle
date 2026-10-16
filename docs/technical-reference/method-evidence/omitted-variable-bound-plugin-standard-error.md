# Omitted-variable bound plug-in standard error

This study validates the standard error of the plug-in omitted-variable bound. Under
`nu2_estimator="plugin"`, the estimate of $\nu^2$ is $E_n[\hat\alpha^2]$, and it moves at first
order with the fitted treatment mechanism. X26 adds the score of a declared mechanism to the
curve of $\nu^2$. The [stacked estimating equation of Saul and Hudgens
(2020)](../validation-methods.md#standard-error-of-the-omitted-variable-bound), Section 2,
Equations (1) and (2), gives that term. A declared known mechanism adds no term. The study reads
$\nu^2$ and each bound as an estimate of a known population value.

**No canonical implementation is compared.** DoubleML and `dml.sensemakr` estimate the doubly
robust $\nu^2$ only. Neither one reports limits for the plug-in estimate. The zero-row
equivalence artifact records the absence of a comparator.

**Status: declared, not yet run.** The X26 plan fixed the laws, the replicate counts, the seeds,
the margins and the red-cell route before any run. The generated tables stay empty until the
run.

## What was tested

| setting | declaration |
| --- | --- |
| binary law | `make_linear_ate(n=1000)`. Four standard normal covariates, the treatment mechanism $\operatorname{expit}(0.3 W_1 - 0.2 W_2 + 0.1 W_3)$, standard normal outcome noise, and a constant effect of 1.5 |
| three-arm law | $W \sim N(0, I_3)$. The logits are $(0,\ 0.3 W_1 - 0.2 W_2,\ -0.2 W_1 + 0.3 W_2)$. The outcome is $0.6\,\mathrm{step}[A] + W_1 - 0.5 W_2 + 0.2 W_3 + N(0, 1)$, with steps $(0, 1, 2.4)$ |
| outcome regression | `LinearRegression` in every scenario. It is correctly specified on both laws |
| estimator of $\nu^2$ | `nu2_estimator="plugin"` |
| strength | $c_Y = 0.5$, $c_D = 0.5$, $\rho = 1$ |
| primary estimands | for each parameter, $\nu^2$ and the lower and upper bound |
| standard error of $\nu^2$ | `sqrt(influence_variance(psi_nu2))` |
| standard error of a bound | `(lower - ci_lower) / z_0.95` for a lower bound, and `(ci_upper - upper) / z_0.95` for an upper bound |
| primary interval | the estimate plus or minus 1.959964 standard errors |
| independent rebuild | each replicate rebuilds the curve of $\nu^2$ from the captured coefficients and the data. The run stops if a rebuilt curve departs from the package curve by more than $10^{-8}$ of its scale. A replicate on which the package withholds the limits counts as a failed replicate |
| Monte Carlo inference | 99% intervals around every declared endpoint |

| scenario | fit | parameters |
| --- | --- | --- |
| `binary_logistic` | in-sample TMLE, `LogisticRegression(C=numpy.inf, max_iter=2000)` | ATE, ATT, ATC |
| `binary_logistic_ridge` | in-sample TMLE, `LogisticRegression(max_iter=1000)`, so $C = 1$ | ATE, ATT, ATC |
| `multinomial_logistic` | in-sample TMLE, `LogisticRegression(C=numpy.inf, max_iter=2000)`, reference arm 0 | ATE and ATT of arm 2 versus 0, ATC of arm 1 versus 0 |
| `binary_known` | in-sample TMLE with the law's own mechanism declared through `treatment_probabilities=` | ATE, ATT, ATC |

The study does not use `make_multi_arm`. On that law $E[1/g^3]$ is about $e^{14}$, and the
curve of $\nu^2$ has a standard deviation of about 709.

| truth | value |
| --- | --- |
| $\sigma^2$ | 1 on both laws |
| binary $\nu^2$ | $2 + 2e^{0.07}$ for the ATE, $4e^{0.07}$ for the ATT and the ATC |
| three-arm $\nu^2$ | Gauss-Hermite quadrature with 80 nodes on each of $W_1$ and $W_2$ |
| effects | 1.5 on the binary law. 1.44, 1.44 and 0.6 on the three-arm law |

## Theory properties

The property study holds one `interval_calibration` family. Every cell of a scenario reads the
same draws, and one fit per draw gives every row of the scenario.

| cell kind | cells | what the standard error reads | what must hold |
| --- | --- | --- | --- |
| `correctly_specified` | 36 | the package curve | the SE-ratio interval and the coverage interval lie inside their calibration bands |
| `inflated_se_control` | 18 | the package curve minus the mechanism term that the study rebuilt, for the ATT and the ATC of the three estimated-mechanism scenarios | the SE-ratio interval lies above the band |
| `shrunken_se_control` | 6 | the known-mechanism curve plus the mechanism term of a study-side `LogisticRegression(C=numpy.inf)` fit, for the ATT and the ATC of `binary_known` | the SE-ratio interval lies below the band |

The ATE has no control. At a correct model its mechanism term reflects the second-moment term,
which keeps the variance. The exact-law tests carry the ATE term.

`tests/unit/test_omitted_variable_plugin_design.py` sized the study before the run.

| design quantity | value |
| --- | --- |
| property replicates per cell | 10,000. The smallest multiple of 400 that reaches a family-wise pass probability of 0.95 under perfect calibration is 3,600 |
| that pass probability | above 0.999 at 10,000, also at true SE ratios of 0.98 and 1.02. At 3,600 it is 0.971, and 0.12 to 0.22 at those ratios |
| predicted control ratios | 2.24 to 2.62 for $\nu^2$ and 1.17 to 1.24 for a bound (inflated), 0.38 for $\nu^2$ and 0.85 for a bound (shrunken) |
| primary replicates per scenario | 10,000 |

<!-- generated: accuracy -->
<!-- /generated -->

<!-- generated: properties -->
<!-- /generated -->

## Red-cell policy

The policy was declared before the run, in `RED_CELL_POLICY` of
`tests/studies/omitted_variable_bound_plugin.py`.

| rule | when it applies | what happens |
| --- | --- | --- |
| 1 | always | the study publishes under `reporting`. No replicate count, margin, law, learner or strength changes after a verdict is seen |
| 2 | a red positive cell | the infinitesimal jackknife (unit witness W5) and the numerical sandwich are compared with the package curve on 20 replicates of the scenario. A mismatch is a defect. The defect is fixed, and the study is regenerated under a fresh declaration commit |
| 3 | rule 2 finds agreement | the cell is rerun at $n = 4000$ with 5,000 replicates. It is published red with owner `X26-plugin-finite-sample` and the reading "finite-sample, contracting" or "finite-sample, not contracting at n=4000", with a roadmap owner row |
| 4 | the reading of a red positive cell | the cell is finite-sample only when its 99% SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. Any other red positive cell is a defect suspect |
| 5 | a control that does not leave the band | the control is reported as "control underpowered", with its ratio |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes each verdict and does not fail on a red one |
| Cross-fitted fits refuse the limits | The package refuses the plug-in limits on a cross-fitted mechanism (reason `cross_fit`). The diagnostic `tests/diagnostics/x26_crossfit_bias` fitted the removed cross-fitted scenario on the binary law. At $n = 1000$ the ATE $\nu^2$ interval covered 0.8845 (99% interval 0.865 to 0.902) and the estimate sat 1.44 empirical standard deviations above the truth. The in-sample fit on the same draws covered 0.969 with a bias of 0.43 standard deviations. At $n = 4000$ the cross-fitted coverage was 0.914 and the bias 0.84 standard deviations. Use `nu2_estimator="doubly_robust"` on a cross-fitted fit |
| There is no cross-implementation evidence | No maintained implementation reports limits for the plug-in $\nu^2$ |
| Correctly specified mechanisms only | Each logistic model is correct for its law. A misspecified model covers $\nu^2$ of its own limit, which the study does not test |
| No truncation is active | The binary law moves a cell with probability below $10^{-21}$ per row. The three-arm law moves one with probability $7 \times 10^{-7}$ per row. The exact-law witness covers an active clip |
| No weights, clusters, pipelines or grouped folds | The exact-law witnesses cover these compositions. This row does not |
| One sample size and one strength | The row covers $n = 1000$ and $c_Y = c_D = 0.5$, $\rho = 1$ |
| The ridge scenario uses $C = 1$ only | The predicted first-order shrinkage of $\nu^2$ is at most 0.05 of its standard error on this law. Over 150 replicates the ATE fraction had mean 0.050 and maximum 0.073. The `penalty_bias` gate refuses above 0.1 |
| Point treatment only | The bound refuses longitudinal fits, so the row covers none |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound_plugin/README.md)
gives the regeneration command.
