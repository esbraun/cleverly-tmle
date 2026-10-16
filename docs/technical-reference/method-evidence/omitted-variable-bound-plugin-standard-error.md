# Omitted-variable bound plug-in standard error

This study validates the standard error of the lower and the upper plug-in omitted-variable
bound. Under `nu2_estimator="plugin"`, the package reports the second-order corrected estimate
$\hat\nu^2 = E_n[\hat\alpha^2] - \hat B$ with a corrected variance. It refuses a fit on which the
correction does not hold.

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
run. The [pilot](#pilot) below read every cell on a disjoint seed stream.

**What is claimed and what is measured.** The claim cells are the lower and the upper bound of
each parameter. The corrected $\nu^2$ is measured and not judged, because at $n = 1000$ its
interval under-covers and no accessor of the package publishes it. The
[corrected $\nu^2$ reading](#the-corrected-nu2-reading) states the shortfall.

**Package values.** The study rebuilds the curve of $\hat\nu^2$ and the delta-method coefficient
of each bound from the captured coefficients and the data. It does not rebuild $\hat B$ or the
variance correction of $\hat\nu^2$. Those two are package values. The exact-law witnesses in
`tests/unit/test_omitted_variable_plugin_second_order.py` check them. The reported standard error
of a bound is the root of the curve variance less the package variance correction, times the
coefficient. The study checks that arithmetic and not the correction.

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
draw in 300 fits of each scenario. The [pilot](#pilot) refused 0.2 to 0.3 percent. The study
covers the untruncated regime of these laws. It does not test the check, which the exact-law
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
| `correctly_specified` | 48 | the package curve less the package variance correction | the SE-ratio interval and the coverage interval lie inside their calibration bands |
| `inflated_se_control` | 28 | the package curve minus the mechanism term that the study rebuilt, for the ATT and the ATC of the seven estimated-mechanism scenarios, with the positive's relative variance correction | the SE-ratio interval lies above the band |
| `shrunken_se_control` | 4 | the known-mechanism curve plus the mechanism term of a study-side `LogisticRegression(C=numpy.inf)` fit, for the ATT and the ATC of `binary_known`, with the positive's relative variance correction | the SE-ratio interval lies below the band |

The ATE has no control. At a correct model its mechanism term reflects the second-moment term,
which keeps the variance. The exact-law tests carry the ATE term.

**The relative variance correction.** The package subtracts a correction $c^2 v$ from the variance
of the positive curve. A control curve has a smaller variance. The same absolute subtraction can
leave a control variance below zero while the package returns limits. The pilot lost two replicates
of the unbalanced ATC this way: draw 158 (lower, $-0.00036$) and draw 339 (upper, $-0.0089$).
Each control therefore keeps the share of its variance that the positive keeps:
$v_c (v_+ - c^2 v) / v_+$.

That value is positive whenever the positive's corrected variance is positive.
`test_a_pilot_draw_that_lost_a_control_now_returns_one` runs both draws. A control never fails a
replicate that the package returned.

`tests/unit/test_omitted_variable_plugin_design.py` sized the study before the run.

| design quantity | value |
| --- | --- |
| property replicates per cell | 10,000. The smallest multiple of 400 that reaches a family-wise pass probability of 0.95 under perfect calibration is 3,600 |
| family-wise pass probability | above 0.9999999 at 10,000, and 0.9617 at 3,600. At true SE ratios of 0.98 and 1.02 it is 0.9999 and 0.9997 at 10,000, and 0.14 and 0.06 at 3,600 |
| largest 99% half-width of a positive SE ratio | 0.0183 at 10,000, against a ceiling of 0.035 |
| spread of a reported standard error | 0.05 of its mean. The diagnostic measured at most 0.0443 |
| predicted inflated control ratios | 1.17 (binary), 1.23 to 1.24 (three-arm), 1.16 (unbalanced ATT), 1.11 (unbalanced ATC) |
| predicted shrunken control ratio | 0.85 |
| control power at the nominal ratio | at least 0.99 for all 32 controls |
| control power at the unfavourable edge of the correction bracket | at least 0.9 for 30 controls. 0.65 for the two ATC controls of the unbalanced law. The design test pins 0.65 and asserts a floor of 0.6 |
| failure bound | 0.01 per cell, so a cell may lose 100 of 10,000 replicates |
| primary replicates per scenario | 10,000 |

The correction bracket allows for the share of a bound's variance that the package removes. The
diagnostic measured a mean share of 0.073 cross-fitted and 0.023 in sample. The bracket takes 0.10
and 0.04. A control subtracts or adds a curve whose own sample variance carries an unknown bias,
so the bracket is a proxy. The unbalanced ATC control is weak because the control arm is the
majority, and its mechanism term is small.

**The pilot falsified this bracket for the unbalanced ATC.** The two controls read 1.052 and 1.053
against the design ratio of 1.11. The in-sample bracket of 0.04 does not hold on that law. The
share of the bound's variance that the correction removes is about 0.16, from
$1 - (0.951 / 1.039)^2$. The per-draw share of the variance of $\hat\nu^2$ is 0.38 on average and
reaches 0.89.

At a true ratio of 1.052 the power at 10,000 replicates is below $10^{-6}$. The size at the band
edge of 1.07 is 0.005. The run is therefore expected to read "control underpowered" for these two
controls. The assertions stay as the design, and the study retunes nothing on the pilot.

## Failure accounting

A replicate is one parameter of one draw.

The package may refuse the limits of a parameter
(`truncation_active`, `correction_unstable` or `corrected_nonpositive`). The study raises
`DataError` for that refusal and for no other reason. A `CapabilityError` of the bound that does not
name `corrected_nonpositive` stops the run. So does a rebuilt variance that is not positive. Both
are defects of the study and not refusals of the package.

The driver records one failed replicate for the cells of a refused parameter and for no other
parameter. A cell may lose up to 1 percent of its replicates. The
failure accounting then judges the cell against the worst case of each clause. One more lost
replicate gives the verdict `bound exceeded`.

The diagnostic `tests/diagnostics/x26_crossfit_bias` measured the refusals on the binary law at
$n = 1000$.

| fit | ATE | ATT | ATC |
| --- | --- | --- | --- |
| cross-fitted, 2,000 draws | 6 (0.30%) | 14 (0.70%) | 11 (0.55%) |
| in sample, 2,000 draws | 0 | 1 (0.05%) | 1 (0.05%) |

The study publishes the refusals of each parameter by key in `refusal-reading.csv`. The columns are
`scenario`, `parameter`, `key`, `requested`, `lost`, `rate` and `owner`. The key is one of the three
package keys, `other` for a failed fit, or `all`. The file has no verdict column. The failure
accounting judges the cells.

The cross-fitted ATT rate leaves a thin margin. At that rate the chance of more than 100 losses
in 10,000 replicates is below 0.0005. The 99% upper endpoint of the rate is 1.4 percent, which
would exceed the bound with certainty. The [pilot](#pilot) and the run read the rate again.

## The corrected nu2 reading

The primary rows carry the corrected $\hat\nu^2$, its corrected standard error, the raw
$E_n[\hat\alpha^2]$, $\hat B$ and the variance correction of each draw. The study publishes them
as `nu2-replicates.csv.gz` and `nu2-reading.csv`. The reading has no verdict column, and no
verdict reads it. The `refused` column counts the draws that the package did not return, and the
failure accounting checks that count against the failure records.

The diagnostic measured the reading on the binary law at $n = 1000$ (2,000 draws per fit). The
coverage is the share of 95% intervals that hold the truth. The bias is the mean estimate minus
the truth, in empirical standard deviations.

| fit | parameter | coverage (corrected) | SE ratio (corrected) | coverage (raw) | SE ratio (raw) | bias, corrected | bias, raw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross-fitted | ATE | 0.812 | 0.732 | 0.884 | 1.079 | -0.07 | 1.44 |
| cross-fitted | ATT | 0.829 | 0.747 | 0.976 | 1.061 | -0.07 | 0.99 |
| cross-fitted | ATC | 0.822 | 0.738 | 0.976 | 1.049 | -0.07 | 0.97 |
| in sample | ATE | 0.901 | 0.920 | 0.969 | 1.034 | -0.02 | 0.43 |
| in sample | ATT | 0.899 | 0.912 | 0.966 | 1.034 | -0.01 | 0.38 |
| in sample | ATC | 0.888 | 0.897 | 0.965 | 1.015 | -0.03 | 0.36 |

The correction removes most of the bias of $E_n[\hat\alpha^2]$. The corrected standard error
under-reads the sampling spread at $n = 1000$: by about 27 percent cross-fitted and 8 to 10
percent in sample. The lower and the upper bound rows of the same draws have SE ratios of 0.99 to
1.01. The study measures the $\nu^2$ rows and does not judge them.

At $n = 4000$ the diagnostic
shows SE ratios of 0.95 to 0.98 for the corrected $\nu^2$. Its coverage is 0.926 to 0.934. No
draw was refused in 1,000 draws of each fit. The shortfall contracts with $n$. The study covers $n = 1000$ only.

## The variance correction reading

The package subtracts a second-order correction from the variance of $\hat\nu^2$. The correction
lowers the standard error of every bound. A cell can therefore read below the band because of the
correction and not because of the curve. The infinitesimal jackknife and the numerical sandwich
compare the curve and cannot see the correction.

The property phase publishes `correction-reading.csv`, with one row for each of the 48 positive
cells. The reading uses the draws of the cell and the intervals of `properties.csv`.

| column | what it reads |
| --- | --- |
| `se_ratio`, `coverage` and their 99% endpoints | the cell with the package variance correction |
| `se_ratio_uncorrected`, `coverage_uncorrected` and the endpoints | the same draws with the standard error before the correction |
| `uncorrected_in_band` | the 99% SE-ratio interval without the correction lies inside the calibration band |
| `mean_correction_share`, `max_correction_share` | the share of the bound's variance that the correction removes on a draw |

The reading has no verdict. Rule 9 of the [red-cell policy](#red-cell-policy) reads
`uncorrected_in_band`.

A pilot probe set the variance correction to zero on the same draws. The correction moved the SE
ratio of the unbalanced ATC from 1.039 to 0.951 (-8.5 percent). It moved the unbalanced ATE by
-6.1 percent and the three-arm cross-fitted cells by -5.4 to -7.0 percent. A cell near 0.95 has a
99% interval that excludes 1 at 10,000 replicates, because the half-width is about 0.018. The
reading separates that outcome from a defect of the curve.

## Pilot

A pilot read every cell before the declaration. It used 1,000 replicates per scenario and
parameter at $n = 1000$ on the seed stream `x26_pilot_sample`, which the declared run does not
use. No setting changed after the pilot. At 1,000 replicates the 99% half-width of an SE ratio is
about 0.058, which is wider than the calibration band. The verdict machinery therefore returns
`passed = false` for most positive cells, and that result says nothing about calibration. The
table reads the point values.

| scenario | refused ATE, ATT, ATC (%) | positive SE ratio | positive coverage | control SE ratio |
| --- | --- | --- | --- | --- |
| `binary_logistic` | 0.0, 0.1, 0.1 | 1.007 to 1.017 | 0.951 to 0.956 | 1.18 to 1.19 |
| `binary_logistic_crossfit` | 0.5, 0.6, 0.6 | 0.980 to 1.040 | 0.939 to 0.959 | 1.15 to 1.22 |
| `binary_logistic_ridge` | 0.0, 0.1, 0.1 | 0.992 to 1.035 | 0.946 to 0.958 | 1.17 to 1.22 |
| `binary_logistic_ridge_crossfit` | 0.5, 0.3, 0.4 | 1.004 to 1.040 | 0.942 to 0.964 | 1.18 to 1.22 |
| `multinomial_logistic` | 0.0, 0.0, 0.0 | 0.974 to 0.996 | 0.937 to 0.955 | 1.21 to 1.23 |
| `multinomial_logistic_crossfit` | 4.2, 2.7, 1.7 | 0.955 to 1.011 | 0.937 to 0.946 | 1.19 to 1.26 |
| `binary_known` | 0.0, 0.0, 0.0 | 0.964 to 1.002 | 0.931 to 0.956 | 0.82 to 0.85 |
| `binary_unbalanced` | 0.3, 0.3, 0.6 | 0.951 to 1.002 | 0.935 to 0.949 | 1.05 to 1.17 |

All 48 positive SE ratios lie inside the band of 0.93 to 1.07, and all 48 coverages lie above
0.93. The pilot found two results that bear on the declared run.

* **The cross-fitted three-arm scenario exceeds the failure bound.** The bound is 1 percent.
  The package refused 4.2, 2.7 and 1.7 percent of the draws for the ATE, ATT and ATC. The
  keys were `corrected_nonpositive` (35, 24 and 15 draws), `correction_unstable` (5, 1 and 0)
  and `truncation_active` (2 each). The declared run allows 100 losses in 10,000 replicates.
  These rates exceed that, so the cells of this scenario would end with the verdict
  `bound exceeded`. The study does not change the scenario after seeing this, and the
  [red-cell policy](#red-cell-policy) declares the outcome before the run.
* **The unbalanced ATC control does not leave the band.** The two controls read 1.052 and 1.053,
  with 99% intervals of 0.99 to 1.12. The design test predicted a ratio of 1.11 and a power of
  0.65 at the unfavourable edge of the correction bracket. The positive cells of the same
  parameter read 0.951 and 0.953. The run is expected to read these two controls "control
  underpowered", with owner `X26-unbalanced-control`.

The binary cross-fitted scenarios refused 0.3 to 0.6 percent of the draws. The other scenarios
refused 0.1 percent or fewer, except the unbalanced scenario at 0.3 to 0.6 percent.

Of the 6 unbalanced ATC losses, 2 came from the variance of a control and not from a refusal of the
package. The study now gives a control the positive's relative correction, so it loses no such
replicate. Three draws of each unbalanced parameter lost a replicate to `truncation_active`. The declared `(0.005, 0.995)`
bounds did not remove that key. The 300-fit probe refused no draw, and the 1,000 pilot draws
refused 0.3 percent. The cross-fitted three-arm scenario lost 0.2 percent to the same key.

The pilot reproduced the [corrected $\nu^2$ reading](#the-corrected-nu2-reading), which has no
verdict.

| scenario group | coverage, corrected $\nu^2$ | SE ratio, corrected $\nu^2$ |
| --- | --- | --- |
| balanced law, in sample (`binary_logistic`, `binary_logistic_ridge`) | 0.886 to 0.913 | 0.87 to 0.92 |
| balanced law, cross-fitted | 0.816 to 0.831 | 0.73 to 0.76 |
| three-arm law, in sample | 0.905 to 0.935 | 0.88 to 0.90 |
| three-arm law, cross-fitted | 0.838 to 0.865 | 0.75 to 0.78 |
| unbalanced law, in sample | 0.911 to 0.938 | 0.87 to 0.98 |
| `binary_known` | 0.926 to 0.958 | 0.92 to 1.03 |

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
| 2 | a red positive cell | the infinitesimal jackknife (unit witness W5) and the numerical sandwich are compared with the package curve on 20 replicates of the scenario. The diagnosis also reads `correction-reading.csv`. A mismatch of the curve is a defect. The defect is fixed, and the study is regenerated under a fresh declaration commit |
| 3 | rule 2 finds agreement | the cell is rerun at $n = 4000$ with 5,000 replicates. It is published red with owner `X26-plugin-finite-sample` and the reading "finite-sample, contracting" or "finite-sample, not contracting at n=4000" |
| 4 | the reading of a red positive cell | the cell is finite-sample only when its 99% SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. Any other red positive cell is a defect suspect, except a cell that rule 7, 8 or 9 reads |
| 5 | a control that does not leave the band | the control is reported as "control underpowered", with its ratio. The two ATC controls of `binary_unbalanced` are declared so before the run, with owner `X26-unbalanced-control` |
| 6 | the corrected $\nu^2$ rows | the rows are measured and not claimed. A shortfall in them opens no red cell |
| 7 | a cell of `multinomial_logistic_crossfit` | the 10 property cells and the 3 primary scenarios are declared to read `bound exceeded`. They are published red with owner `X26-crossfit-refusals`, the refusal rate of each key in `refusal-reading.csv`, and the complete-case SE ratio and coverage labelled "not a verdict". Rules 2 to 4 do not apply to a cell that only the bound turns red |
| 8 | a cell that is green on the complete case and red only through `worst case fails` | the cell reads "refusal-limited" with owner `X26-refusal-limited`. It publishes the number of lost replicates, the complete-case value and the worst-case value. It is not a defect suspect under rule 4 |
| 9 | a red positive cell whose 99% SE-ratio interval without the variance correction lies inside the band | if rule 2 finds no mismatch of the curve, the cell reads "variance-correction overshoot" with owner `X26-correction-overshoot`. It is rerun under rule 3 and published as "contracting" or "not contracting at n=4000". It is not a defect suspect under rule 4 |

### Reds declared before the run

The pilot and the failure accounting predict these reds. The failure bound stays at 0.01 and the margins stay as declared.

| cells | expected reading | owner | evidence |
| --- | --- | --- | --- |
| 6 positive and 4 control cells of `multinomial_logistic_crossfit`, and its 3 primary scenarios | `bound exceeded` | `X26-crossfit-refusals` | the package refused 4.2, 2.7 and 1.7 percent of the draws for the ATE, ATT and ATC |
| the two `inflated_se_control` cells of the unbalanced ATC | control underpowered | `X26-unbalanced-control` | the ratios read 1.052 and 1.053 against a design ratio of 1.11 |
| a positive cell whose uncorrected ratio is inside the band | variance-correction overshoot | `X26-correction-overshoot` | the correction moves the SE ratio of the unbalanced ATC by -8.5 percent |
| a positive cell green on the complete case and red by a worst case | refusal-limited | `X26-refusal-limited` | the worst-case clauses of the accounting act inside the failure bound |

The complete-case values of a `bound exceeded` cell are not a verdict. The accounting assigns the
verdict. Raising the bound would relax all 80 cells to cover 10, so the study declares the red
instead.

The owner rows of the roadmap name these ids before the documentation pull request. The study
keeps the package variance correction. The owner of the study confirms that decision before the run.

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes each verdict and does not fail on a red one |
| The corrected $\nu^2$ is not judged | At $n = 1000$ its interval under-covers. The [reading](#the-corrected-nu2-reading) states the size. The bound intervals do not show the shortfall |
| $\hat B$ and the variance correction are package values | The study rebuilds the curve and the delta-method coefficient. The exact-law witnesses in `tests/unit/test_omitted_variable_plugin_second_order.py` check $\hat B$ and the variance correction |
| Refusals are failed replicates | A refusal fails the cells of one parameter. A cell that loses more than 1 percent of its replicates ends with the verdict `bound exceeded`. The [pilot](#pilot) shows that the cross-fitted three-arm scenario does. The policy declares that red |
| The variance correction shifts some cells | The package correction lowers the SE ratio by 5 to 9 percent in the unbalanced and the cross-fitted cells. `correction-reading.csv` records the ratio without it. A cell can read red because of the correction and not because of the curve |
| Three scenarios declare wide truncation bounds | `multinomial_logistic`, `multinomial_logistic_crossfit` and `binary_unbalanced` declare `g_bounds=(0.005, 0.995)`. The automatic bounds refuse 27 to 52 percent of draws on those laws. The study does not test an active clip. The exact-law witness covers it |
| There is no cross-implementation evidence | No maintained implementation reports limits for the plug-in $\nu^2$ |
| Correctly specified mechanisms only | Each logistic model is correct for its law. A misspecified model covers $\nu^2$ of its own limit, which the study does not test |
| No weights, clusters, pipelines or grouped folds | The exact-law witnesses cover these compositions. This row does not |
| One sample size and one strength | The row covers $n = 1000$ and $c_Y = c_D = 0.5$, $\rho = 1$. The registered property loop has one sample size. The diagnostic `tests/diagnostics/x26_crossfit_bias` covers $n = 4000$ for the balanced law, in sample and cross-fitted. This study has no $n = 4000$ arm |
| The correction bracket of the controls is a proxy | The bracket rests on the diagnostic's mean share on the balanced law. The pilot shows that the unbalanced law differs: the share of the bound's variance there is about 0.16, and the ATC controls read 1.052 and 1.053 |
| The ridge scenarios use $C = 1$ only | The predicted first-order shrinkage of $\nu^2$ is at most 0.05 of its standard error on the binary law |
| Point treatment only | The bound refuses longitudinal fits, so the row covers none |
| Run cost | The declared run costs about 68 CPU hours at 10,000 replicates. The fit of a parameter takes 0.1 s in sample and 1 s cross-fitted, and the bound engine dominates |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound_plugin/README.md)
gives the regeneration command.
