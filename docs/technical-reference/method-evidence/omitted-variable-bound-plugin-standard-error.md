# Omitted-variable bound plug-in standard error

This study validates the standard error of the lower and the upper plug-in omitted-variable
bound. Under `nu2_estimator="plugin"`, the package reports the second-order corrected estimate
$\hat\nu^2 = E_n[\hat\alpha^2] - \hat B$ with the influence variance of its curve. The package
makes no second-order correction of the variance. It refuses a fit on which the correction of the
point does not hold.

The curve of $\hat\nu^2$ adds the score of a fitted treatment mechanism
to the second-moment term. The [stacked estimating equation of Saul and Hudgens
(2020)](../validation-methods.md#standard-error-of-the-omitted-variable-bound), Section 2,
Equations (1) and (2), gives that term. A declared known mechanism adds no term. The study reads
each bound as an estimate of a known population value.

**No canonical implementation is compared.** DoubleML and `dml.sensemakr` estimate the doubly
robust $\nu^2$ only. Neither one reports limits for the plug-in estimate. The zero-row
equivalence artifact records the absence of a comparator.

**Status: declared, not yet run.** The X26 plan fixed the laws, the replicate counts, the seeds,
the margins and the red-cell route before any run. The generated tables stay empty until the
run. The [pilot](#pilot) section states which pilot judges this design.

**What is claimed and what is measured.** The claim cells are the lower and the upper bound of
each parameter. The corrected $\nu^2$ is measured and not judged, because no accessor of the
package publishes its interval. At $n = 1000$ that interval over-reads the sampling spread. The
[corrected $\nu^2$ reading](#the-corrected-nu2-reading) states the size.

**Package values.** The study rebuilds the curve of $\hat\nu^2$ and the bias curve of each bound
from the captured coefficients and the data. It does not rebuild $\hat B$. That value is a
package value. The exact-law witnesses in `tests/unit/test_omitted_variable_plugin_second_order.py`
check it. The subtraction of $\hat B$ leaves the curve unchanged, so the variance is the plain
influence variance.

The reported standard error of a bound is the root of the influence variance of the combined
curve of the estimate and the bound's bias term. The study rebuilds that variance and compares it
with the reported value.

## What was tested

| setting | declaration |
| --- | --- |
| binary law | `make_linear_ate(n=1000)`. Four standard normal covariates, the treatment mechanism $\operatorname{expit}(0.3 W_1 - 0.2 W_2 + 0.1 W_3)$, standard normal outcome noise, and a constant effect of 1.5 |
| unbalanced law | the binary law with the intercept $-1.5$ added to the treatment index. The treated share is 0.189 |
| three-arm law | $W \sim N(0, I_3)$. The logits are $(0,\ 0.3 W_1 - 0.2 W_2,\ -0.2 W_1 + 0.3 W_2)$. The outcome is $0.6\,\mathrm{step}[A] + W_1 - 0.5 W_2 + 0.2 W_3 + N(0, 1)$, with steps $(0, 1, 2.4)$ |
| outcome regression | `LinearRegression` in every scenario. It is correctly specified on every law |
| estimator of $\nu^2$ | `nu2_estimator="plugin"` |
| strength | $c_Y = 0.5$, $c_D = 0.5$, $\rho = 1$ |
| claim estimands | for each parameter, the lower and the upper bound |
| standard error of a bound | `(lower - ci_lower) / z_0.95` for a lower bound, and `(ci_upper - upper) / z_0.95` for an upper bound |
| primary interval | the estimate plus or minus 1.959964 standard errors |
| independent rebuild | each replicate rebuilds the curve of $\hat\nu^2$ from the captured coefficients, the folds and the data. The run stops if the rebuilt curve departs from the package curve by more than $10^{-8}$ of its scale. The rebuilt raw $\nu^2$ must equal the package value $E_n[\hat\alpha^2]$, which is the reported $\hat\nu^2$ plus $\hat B$ |
| Monte Carlo inference | 99% intervals around every declared endpoint |

| scenario | fit | parameters |
| --- | --- | --- |
| `binary_logistic` | in-sample TMLE, `LogisticRegression(C=numpy.inf, max_iter=2000)` | ATE, ATT, ATC |
| `binary_logistic_crossfit` | the same fit with `cross_fit=True`, `n_folds=5` and `q_bounds=(-40, 45)` | ATE, ATT, ATC |
| `binary_logistic_ridge` | in-sample TMLE, `LogisticRegression(max_iter=1000)`, so $C = 1$ | ATE, ATT, ATC |
| `binary_logistic_ridge_crossfit` | the ridge fit with `cross_fit=True`, `n_folds=5` and `q_bounds=(-40, 45)` | ATE, ATT, ATC |
| `multinomial_logistic` | in-sample TMLE, `LogisticRegression(C=numpy.inf, max_iter=2000)`, reference arm 0, `g_bounds=(0.005, 0.995)` | ATE and ATT of arm 2 versus 0, ATC of arm 1 versus 0 |
| `multinomial_logistic_crossfit` | the same fit with `cross_fit=True`, `n_folds=5` and `q_bounds=(-40, 45)` | ATE and ATT of arm 2 versus 0, ATC of arm 1 versus 0 |
| `binary_known` | in-sample TMLE with the law's own mechanism declared through `treatment_probabilities=` | ATE, ATT, ATC |
| `binary_unbalanced` | in-sample TMLE, `LogisticRegression(C=numpy.inf, max_iter=2000)`, `g_bounds=(0.005, 0.995)` | ATE, ATT, ATC |

Each parameter is its own primary scenario, so the study registers 24 primary scenarios. The
three parameters of one scenario read one sample.

The cross-fitted scenarios declare `q_bounds=(-40, 45)`, the support of
`tests/diagnostics/x26_crossfit_bias`. The binary outcome has arm means 2 and 3.5 and a standard
deviation of about 1.75. Every bound lies more than 23 standard deviations from both means.
The bounds fix the scale of the logistic fluctuation and truncate no draw.

The study does not use `make_multi_arm`. On that law $E[1/g^3]$ is about $e^{14}$, and the
curve of $\nu^2$ has a standard deviation of about 709.

| truth | value |
| --- | --- |
| $\sigma^2$ | 1 on every law |
| binary $\nu^2$ | $2 + 2e^{0.07}$ for the ATE, $4e^{0.07}$ for the ATT and the ATC |
| unbalanced $\nu^2$ | Gauss-Hermite quadrature over the logit-normal treatment index. The 60-node and 80-node rules agree to $10^{-10}$ |
| three-arm $\nu^2$ | Gauss-Hermite quadrature with 80 nodes on each of $W_1$ and $W_2$ |
| effects | 1.5 on both binary laws. 1.44, 1.44 and 0.6 on the three-arm law |

## Truncation bounds of three scenarios

The package refuses the plug-in limits (reason `truncation_active`) when a fitted logit lies
within three standard errors of a truncation bound. The automatic bounds at $n = 1000$ are 0.0229
for the ATE and 0.025 for the ATT and the ATC. On the unbalanced law and the three-arm law the
extreme rows sit that close to the bounds. A 30-draw probe of the package refused 52 percent of
the unbalanced draws and 27 percent of the in-sample three-arm draws. That rate exceeds the
failure bound of 0.01 and would make every cell of those scenarios red by construction.

Three scenarios therefore declare `g_bounds=(0.005, 0.995)`: `multinomial_logistic`,
`multinomial_logistic_crossfit` and `binary_unbalanced`.

At those bounds the probe refused no
draw in 300 fits of each scenario. The study covers the untruncated regime of these laws. It does not test the check, which the exact-law
witnesses cover. The design test predicts the
refusal share from the population mechanism and the sandwich of the logistic fit. The
prediction is a statement of order of magnitude only.

| quantity | value |
| --- | --- |
| probability that a row of the unbalanced law is clipped at 0.025 | $3.7 \times 10^{-9}$ |
| probability that a row of the three-arm law is clipped at 0.025 | $7.3 \times 10^{-7}$ |
| probability that a row of the balanced law is clipped at 0.025 | below $10^{-21}$ |
| clip probabilities at the declared `(0.005, 0.995)` | below $10^{-12}$ on both laws |

## Theory properties

The property study holds one `interval_calibration` family. Each scenario has a positive cell
for the lower and for the upper bound of each parameter. One fit gives every row of one
parameter of one draw. A control and its positive cell therefore differ only in the curve their
standard error comes from.

| cell kind | cells | what the standard error reads | what must hold |
| --- | --- | --- | --- |
| `correctly_specified` | 48 | the package curve, with its influence variance | the SE-ratio interval and the coverage interval lie inside their calibration bands |
| `inflated_se_control` | 28 | the package curve minus the mechanism term that the study rebuilt, for the ATT and the ATC of the seven estimated-mechanism scenarios, with the influence variance of that curve | the SE-ratio interval lies above the band |
| `shrunken_se_control` | 4 | the known-mechanism curve plus the mechanism term of a study-side `LogisticRegression(C=numpy.inf)` fit, for the ATT and the ATC of `binary_known`, with the influence variance of that curve | the SE-ratio interval lies below the band |

The ATE has no control. At a correct model its mechanism term reflects the second-moment term,
which keeps the variance. The exact-law tests carry the ATE term.

`tests/unit/test_omitted_variable_plugin_design.py` sized the study before the run.

| design quantity | value |
| --- | --- |
| property replicates per cell | 10,000. The smallest multiple of 400 that reaches a family-wise pass probability of 0.95 under perfect calibration is 3,600 |
| family-wise pass probability | above 0.9999999 at 10,000, and 0.9624 at 3,600. At true SE ratios of 0.98 and 1.02 it is 0.9999 and 0.9997 at 10,000, and 0.14 and 0.06 at 3,600 |
| largest 99% half-width of a positive SE ratio | 0.0183 at 10,000, against a ceiling of 0.035 |
| spread of a reported standard error | at most 0.0505 of its mean, from the fourth standardized moment of each curve at $n = 1000$ |
| predicted inflated control ratios | 1.17 (binary), 1.23 to 1.24 (three-arm), 1.16 (unbalanced ATT), 1.11 (unbalanced ATC) |
| predicted shrunken control ratio | 0.85 |
| control power at the population ratio | at least 0.99 for all 32 controls. The weakest are the two ATC controls of the unbalanced law, at 0.996 |
| failure bound | 0.01 per cell, so a cell may lose 100 of 10,000 replicates |
| primary replicates per scenario | 10,000 |

The two ATC controls of the unbalanced law are the weakest. The control arm is the majority, so
the mechanism term is small, and the fourth moment of the curve is heavy. Every power derives from
the population variances of the laws. No pilot number enters it.

## Failure accounting

A replicate is one parameter of one draw.

The package may refuse the limits of a parameter
(`truncation_active`, `correction_unstable` or `corrected_nonpositive`). The study reads the
refusal from the elements of the bound and raises `DataError` for it and for no other reason.
The bound itself raises no refusal for a variance, so the study converts no `CapabilityError`. A
`CapabilityError` of the bound, and a rebuilt standard error that departs from the reported one,
are defects of the study and stop the run.

The driver records one failed replicate for the cells of a refused parameter and for no other
parameter. A cell may lose up to 1 percent of its replicates. The
failure accounting then judges the cell against the worst case of each clause. One more lost
replicate gives the verdict `bound exceeded`.

The diagnostic `tests/diagnostics/x26_crossfit_bias` measured the refusals on the binary law at
$n = 1000$.

| fit | ATE | ATT | ATC |
| --- | --- | --- | --- |
| cross-fitted, 2,000 draws | 0 | 1 (0.05%) | 1 (0.05%) |
| in sample, 2,000 draws | 0 | 1 (0.05%) | 1 (0.05%) |

Each of the four refusals has the key `truncation_active`. No draw gave `correction_unstable` or
`corrected_nonpositive`.

The study publishes the refusals of each parameter by key in `refusal-reading.csv`. The columns are
`scenario`, `parameter`, `key`, `requested`, `lost` and `rate`. The key is one of the three
package keys, `other` for a failed fit, or `all`. The file has no verdict column. The failure
accounting judges the cells.

At a rate of 0.05 percent, the chance of more than 100 losses in 10,000 replicates is below
$10^{-12}$. The 99% upper endpoint of the rate is 0.2 percent. The run reads the rate again.

## The corrected nu2 reading

The primary rows carry the corrected $\hat\nu^2$, its standard error, the raw
$E_n[\hat\alpha^2]$ and $\hat B$ of each draw. The standard error is the influence variance of
the curve, which the correction of the point leaves unchanged. The study publishes them
as `nu2-replicates.csv.gz` and `nu2-reading.csv`. The reading has no verdict column, and no
verdict reads it. The `refused` column counts the draws that the package did not return, and the
failure accounting checks that count against the failure records.

The diagnostic measured the reading on the binary law at $n = 1000$ (2,000 draws per fit). The
coverage is the share of 95% intervals that hold the truth. The bias is the mean estimate minus
the truth, in empirical standard deviations.

| fit | parameter | coverage (corrected) | SE ratio (corrected) | coverage (raw) | SE ratio (raw) | bias, corrected | bias, raw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross-fitted | ATE | 0.966 | 1.258 | 0.884 | 1.079 | -0.08 | 1.44 |
| cross-fitted | ATT | 0.952 | 1.237 | 0.976 | 1.061 | -0.06 | 0.99 |
| cross-fitted | ATC | 0.953 | 1.224 | 0.976 | 1.049 | -0.08 | 0.97 |
| in sample | ATE | 0.943 | 1.077 | 0.969 | 1.034 | -0.02 | 0.43 |
| in sample | ATT | 0.939 | 1.077 | 0.966 | 1.034 | -0.01 | 0.38 |
| in sample | ATC | 0.930 | 1.059 | 0.965 | 1.015 | -0.03 | 0.36 |

The correction removes most of the bias of $E_n[\hat\alpha^2]$. The standard error over-reads the
sampling spread of the corrected $\hat\nu^2$ at $n = 1000$, by 22 to 26 percent cross-fitted and 6
to 8 percent in sample. The lower and the upper bound rows of the same draws have SE ratios of
1.00 to 1.04. The study measures the $\nu^2$ rows and does not judge them.

At $n = 4000$ the diagnostic shows SE ratios of 1.03 to 1.05 cross-fitted and 1.00 to 1.01 in
sample for the corrected $\nu^2$. Its coverage is 0.938 to 0.957. No draw was refused in 1,000
draws of each fit. The over-reading contracts with $n$. The study covers $n = 1000$ only.

## Pilot

An earlier design of the study kept a second-order correction of the variance of $\hat\nu^2$.
Its pilot, on the seed stream `x26_pilot_sample`, led the owner to drop that correction on
2026-10-16. At $n = 1000$ the correction took about 60 percent of the cross-fitted variance, so
its expansion was not trusted. It also made the package refuse 1.7 to 4.2 percent of the
cross-fitted three-arm draws with the key `corrected_nonpositive`. The owner reports that the
influence variance gave SE ratios of 0.97 to 1.08 and coverages of 0.933 to 0.971 with at most
0.3 percent refusals. That pilot is not committed, and no claim of this page rests on it.

The design on this page is judged by a new pilot on the seed stream `x26_pilot2_sample`. The
declared run does not use that stream, and the design test checks that no declared seed does. The
pilot has not run. The study declares no red cell, no expected refusal rate and no
underpowered control before that pilot reads the design.

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
| 2 | a red positive cell | the infinitesimal jackknife (unit witness W5) and the numerical sandwich are compared with the package curve on 20 replicates of the scenario. A mismatch of the curve is a defect. The defect is fixed, and the study is regenerated under a fresh declaration commit |
| 3 | rule 2 finds agreement | the cell is rerun at $n = 4000$ with 5,000 replicates. It is published red with owner `X26-plugin-finite-sample` and the reading "finite-sample, contracting" or "finite-sample, not contracting at n=4000" |
| 4 | the reading of a red positive cell | the cell is finite-sample only when its 99% SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. Any other red positive cell is a defect suspect, except a cell that rule 7 reads |
| 5 | a control that does not leave the band | the control is reported as "control underpowered", with its ratio |
| 6 | the corrected $\nu^2$ rows | the rows are measured and not claimed. A shortfall in them opens no red cell |
| 7 | a cell that is green on the complete case and red only through `worst case fails` | the cell reads "refusal-limited" with owner `X26-refusal-limited`. It publishes the number of lost replicates, the complete-case value, the worst-case value and the refusal rate of each key in `refusal-reading.csv`. It is not a defect suspect under rule 4 |

The study declares no red cell before the run. The failure bound stays at 0.01 and the margins
stay as declared. A cell that loses more than 1 percent of its replicates ends with the verdict
`bound exceeded`, and the run reads that verdict like any other. The new pilot reads the design
first, and the declaration changes only under a fresh declaration commit.

The owner rows of the roadmap name the ids `X26-plugin-finite-sample` and `X26-refusal-limited`
before the documentation pull request.

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes each verdict and does not fail on a red one |
| The corrected $\nu^2$ is not judged | At $n = 1000$ its standard error over-reads the sampling spread. The [reading](#the-corrected-nu2-reading) states the size. The bound intervals read near 1.0 to 1.04 in the diagnostic |
| $\hat B$ is a package value | The study rebuilds the curve and the bias curve of each bound. The exact-law witnesses in `tests/unit/test_omitted_variable_plugin_second_order.py` check $\hat B$ |
| Refusals are failed replicates | A refusal fails the cells of one parameter. A cell that loses more than 1 percent of its replicates ends with the verdict `bound exceeded`. The study declares no such cell before the run |
| Three scenarios declare wide truncation bounds | `multinomial_logistic`, `multinomial_logistic_crossfit` and `binary_unbalanced` declare `g_bounds=(0.005, 0.995)`. The automatic bounds refuse 27 to 52 percent of draws on those laws. The study does not test an active clip. The exact-law witness covers it |
| There is no cross-implementation evidence | No maintained implementation reports limits for the plug-in $\nu^2$ |
| Correctly specified mechanisms only | Each logistic model is correct for its law. A misspecified model covers $\nu^2$ of its own limit, which the study does not test |
| No weights, clusters, pipelines or grouped folds | The exact-law witnesses cover these compositions. This row does not |
| One sample size and one strength | The row covers $n = 1000$ and $c_Y = c_D = 0.5$, $\rho = 1$. The registered property loop has one sample size. The diagnostic `tests/diagnostics/x26_crossfit_bias` covers $n = 4000$ for the balanced law, in sample and cross-fitted. This study has no $n = 4000$ arm |
| The ridge scenarios use $C = 1$ only | The predicted first-order shrinkage of $\nu^2$ is at most 0.05 of its standard error on the binary law |
| Point treatment only | The bound refuses longitudinal fits, so the row covers none |
| Run cost | The declared run costs about 68 CPU hours at 10,000 replicates. The fit of a parameter takes 0.1 s in sample and 1 s cross-fitted, and the bound engine dominates |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound_plugin/README.md)
gives the regeneration command.
