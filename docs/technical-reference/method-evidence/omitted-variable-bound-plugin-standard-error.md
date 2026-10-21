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

**Status: run.** The declared run finished at commit `7a275bae`. The X26 plan fixed the laws, the
replicate counts, the seeds, the margins and the red-cell route before the run. The
[result](#result) section states the verdicts. Two cells read red, and the
[red cells](#red-cells) section gives their diagnosis. The [pilot](#pilot) section states what the
two pilots measured before the run and what they changed.

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
draw in 300 fits of each scenario. The pilot on `x26_pilot2_sample` refused 0.1 to 0.4 percent of
the draws on the three scenarios (see [Pilot](#pilot)), and the design test predicted none. The
study covers the untruncated regime of these laws. It does not test the check, which the
exact-law witnesses cover. The design test predicts the
refusal share from the population mechanism and the sandwich of the logistic fit. The
prediction is a statement of order of magnitude only, and the pilot rate is higher.

| quantity | value |
| --- | --- |
| probability that a row of the unbalanced law is clipped at 0.025 | $3.7 \times 10^{-9}$ |
| probability that a row of the three-arm law is clipped at 0.025 | $7.3 \times 10^{-7}$ |
| probability that a row of the balanced law is clipped at 0.025 | below $10^{-21}$ |
| clip probabilities at the declared `(0.005, 0.995)` | below $10^{-12}$ on both laws |

## Accuracy against known truth

Each row is one lower or upper bound of one parameter. The primary phase draws 10,000 samples of
1,000 rows for each parameter. The reported standard error of a bound is the error read back off
its limit.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `make_linear_ate`, in-sample TMLE with its own mechanism declared known | `plugin_atc_lower` | lower plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0024 to 0.0070 | 0.9468 | 0.9946 | pass |
| `make_linear_ate`, in-sample TMLE with its own mechanism declared known | `plugin_atc_upper` | upper plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0088 to -0.0043 | 0.9525 | 1.0141 | pass |
| `make_linear_ate`, in-sample TMLE with its own mechanism declared known | `plugin_ate_lower` | lower plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0017 to 0.0054 | 0.9453 | 0.9920 | pass |
| `make_linear_ate`, in-sample TMLE with its own mechanism declared known | `plugin_ate_upper` | upper plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0069 to -0.0031 | 0.9491 | 0.9996 | pass |
| `make_linear_ate`, in-sample TMLE with its own mechanism declared known | `plugin_att_lower` | lower plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0019 to 0.0064 | 0.9480 | 0.9985 | pass |
| `make_linear_ate`, in-sample TMLE with its own mechanism declared known | `plugin_att_upper` | upper plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0076 to -0.0030 | 0.9475 | 0.9897 | pass |
| `make_linear_ate`, in-sample TMLE with an unpenalized logistic treatment model | `plugin_atc_lower` | lower plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0011 to 0.0050 | 0.9503 | 1.0010 | pass |
| `make_linear_ate`, in-sample TMLE with an unpenalized logistic treatment model | `plugin_atc_upper` | upper plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0081 to -0.0042 | 0.9482 | 1.0131 | pass |
| `make_linear_ate`, in-sample TMLE with an unpenalized logistic treatment model | `plugin_ate_lower` | lower plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0013 to 0.0051 | 0.9491 | 0.9950 | pass |
| `make_linear_ate`, in-sample TMLE with an unpenalized logistic treatment model | `plugin_ate_upper` | upper plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0076 to -0.0039 | 0.9479 | 1.0085 | pass |
| `make_linear_ate`, in-sample TMLE with an unpenalized logistic treatment model | `plugin_att_lower` | lower plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0017 to 0.0056 | 0.9495 | 0.9982 | pass |
| `make_linear_ate`, in-sample TMLE with an unpenalized logistic treatment model | `plugin_att_upper` | upper plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0077 to -0.0038 | 0.9480 | 1.0097 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with an unpenalized logistic treatment model | `plugin_atc_lower` | lower plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0051 to -0.0011 | 0.9565 | 1.0265 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with an unpenalized logistic treatment model | `plugin_atc_upper` | upper plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0024 to 0.0064 | 0.9543 | 1.0135 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with an unpenalized logistic treatment model | `plugin_ate_lower` | lower plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0055 to -0.0018 | 0.9514 | 1.0127 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with an unpenalized logistic treatment model | `plugin_ate_upper` | upper plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0032 to 0.0070 | 0.9498 | 0.9986 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with an unpenalized logistic treatment model | `plugin_att_lower` | lower plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0048 to -0.000878 | 0.9546 | 1.0251 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with an unpenalized logistic treatment model | `plugin_att_upper` | upper plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0026 to 0.0066 | 0.9542 | 1.0143 | pass |
| `make_linear_ate`, in-sample TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_atc_lower` | lower plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0033 to 0.0072 | 0.9467 | 0.9882 | pass |
| `make_linear_ate`, in-sample TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_atc_upper` | upper plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0062 to -0.0022 | 0.9493 | 1.0011 | pass |
| `make_linear_ate`, in-sample TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_ate_lower` | lower plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0031 to 0.0069 | 0.9448 | 0.9875 | pass |
| `make_linear_ate`, in-sample TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_ate_upper` | upper plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0057 to -0.0019 | 0.9483 | 0.9974 | pass |
| `make_linear_ate`, in-sample TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_att_lower` | lower plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0032 to 0.0072 | 0.9465 | 0.9896 | pass |
| `make_linear_ate`, in-sample TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_att_upper` | upper plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0059 to -0.0019 | 0.9488 | 0.9995 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_atc_lower` | lower plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0065 to -0.0025 | 0.9531 | 1.0129 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_atc_upper` | upper plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0024 to 0.0063 | 0.9554 | 1.0219 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_ate_lower` | lower plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0066 to -0.0029 | 0.9527 | 1.0075 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_ate_upper` | upper plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0030 to 0.0067 | 0.9508 | 1.0118 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_att_lower` | lower plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0060 to -0.0021 | 0.9570 | 1.0235 | pass |
| `make_linear_ate`, five-fold cross-fitted TMLE with a logistic treatment model under a fixed L2 penalty, C = 1 | `plugin_att_upper` | upper plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0025 to 0.0064 | 0.9564 | 1.0275 | pass |
| `make_linear_ate` with the treatment intercept moved to -1.5 (about 19 percent treated), in-sample TMLE with an unpenalized logistic treatment model | `plugin_atc_lower` | lower plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0085 to 0.0147 | 0.9569 | 1.0608 | pass |
| `make_linear_ate` with the treatment intercept moved to -1.5 (about 19 percent treated), in-sample TMLE with an unpenalized logistic treatment model | `plugin_atc_upper` | upper plug-in omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0134 to -0.0071 | 0.9549 | 1.0420 | pass |
| `make_linear_ate` with the treatment intercept moved to -1.5 (about 19 percent treated), in-sample TMLE with an unpenalized logistic treatment model | `plugin_ate_lower` | lower plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0074 to 0.0132 | 0.9551 | 1.0444 | pass |
| `make_linear_ate` with the treatment intercept moved to -1.5 (about 19 percent treated), in-sample TMLE with an unpenalized logistic treatment model | `plugin_ate_upper` | upper plug-in omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0121 to -0.0062 | 0.9530 | 1.0309 | pass |
| `make_linear_ate` with the treatment intercept moved to -1.5 (about 19 percent treated), in-sample TMLE with an unpenalized logistic treatment model | `plugin_att_lower` | lower plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0048 to 0.0101 | 0.9474 | 1.0056 | pass |
| `make_linear_ate` with the treatment intercept moved to -1.5 (about 19 percent treated), in-sample TMLE with an unpenalized logistic treatment model | `plugin_att_upper` | upper plug-in omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0096 to -0.0044 | 0.9499 | 1.0069 | pass |
| three-arm law with a multinomial logistic mechanism, in-sample TMLE with an unpenalized multinomial logistic treatment model | `plugin_atc_1v0_lower` | lower plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0029 to 0.0079 | 0.9486 | 1.0014 | pass |
| three-arm law with a multinomial logistic mechanism, in-sample TMLE with an unpenalized multinomial logistic treatment model | `plugin_atc_1v0_upper` | upper plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0107 to -0.0057 | 0.9517 | 1.0116 | pass |
| three-arm law with a multinomial logistic mechanism, in-sample TMLE with an unpenalized multinomial logistic treatment model | `plugin_ate_2v0_lower` | lower plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0036 to 0.0085 | 0.9488 | 1.0028 | pass |
| three-arm law with a multinomial logistic mechanism, in-sample TMLE with an unpenalized multinomial logistic treatment model | `plugin_ate_2v0_upper` | upper plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0111 to -0.0061 | 0.9441 | 0.9955 | pass |
| three-arm law with a multinomial logistic mechanism, in-sample TMLE with an unpenalized multinomial logistic treatment model | `plugin_att_2v0_lower` | lower plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0044 to 0.0094 | 0.9494 | 1.0079 | pass |
| three-arm law with a multinomial logistic mechanism, in-sample TMLE with an unpenalized multinomial logistic treatment model | `plugin_att_2v0_upper` | upper plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0103 to -0.0053 | 0.9487 | 1.0090 | pass |
| three-arm law with a multinomial logistic mechanism, five-fold cross-fitted TMLE with an unpenalized multinomial logistic treatment model | `plugin_atc_1v0_lower` | lower plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0086 to -0.0036 | 0.9571 | 1.0405 | pass |
| three-arm law with a multinomial logistic mechanism, five-fold cross-fitted TMLE with an unpenalized multinomial logistic treatment model | `plugin_atc_1v0_upper` | upper plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.0017 to 0.0067 | 0.9581 | 1.0373 | pass |
| three-arm law with a multinomial logistic mechanism, five-fold cross-fitted TMLE with an unpenalized multinomial logistic treatment model | `plugin_ate_2v0_lower` | lower plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0073 to -0.0023 | 0.9609 | 1.0424 | pass |
| three-arm law with a multinomial logistic mechanism, five-fold cross-fitted TMLE with an unpenalized multinomial logistic treatment model | `plugin_ate_2v0_upper` | upper plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.000190 to 0.0052 | 0.9567 | 1.0402 | pass |
| three-arm law with a multinomial logistic mechanism, five-fold cross-fitted TMLE with an unpenalized multinomial logistic treatment model | `plugin_att_2v0_lower` | lower plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | -0.0071 to -0.0020 | 0.9600 | 1.0418 | pass |
| three-arm law with a multinomial logistic mechanism, five-fold cross-fitted TMLE with an unpenalized multinomial logistic treatment model | `plugin_att_2v0_upper` | upper plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, at cf_y 0.5, cf_d 0.5, rho 1 | `cleverly` omitted-variable bound with the plug-in nu^2 | 0.000665 to 0.0058 | 0.9585 | 1.0366 | pass |
<!-- /generated -->

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

The measured cells follow. Each row names a cell and its role. A positive cell must put its
SE-ratio and coverage intervals inside the calibration bands. A control must leave the band on the
declared side.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `binary_known_atc_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC, declared known mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9396 to 0.9514, SE ratio 0.9835 to 1.0199 | pass |
| `interval_calibration` | `binary_known_atc_lower__shrunken_se_control` | control | lower plug-in omitted-variable bound on the ATC, declared known mechanism: the standard errors come from the known-mechanism curve plus the mechanism term of a study-side logistic fit, which is too narrow for a declared mechanism | the SE-ratio interval must fall below the calibration band | coverage 0.8917 to 0.9073, SE ratio 0.8419 to 0.8729 | pass |
| `interval_calibration` | `binary_known_atc_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC, declared known mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9453 to 0.9565, SE ratio 0.9895 to 1.0265 | pass |
| `interval_calibration` | `binary_known_atc_upper__shrunken_se_control` | control | upper plug-in omitted-variable bound on the ATC, declared known mechanism: the standard errors come from the known-mechanism curve plus the mechanism term of a study-side logistic fit, which is too narrow for a declared mechanism | the SE-ratio interval must fall below the calibration band | coverage 0.9031 to 0.9179, SE ratio 0.8465 to 0.8779 | pass |
| `interval_calibration` | `binary_known_ate_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE, declared known mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9397 to 0.9515, SE ratio 0.9718 to 1.0086 | pass |
| `interval_calibration` | `binary_known_ate_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE, declared known mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9412 to 0.9528, SE ratio 0.9818 to 1.0188 | pass |
| `interval_calibration` | `binary_known_att_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT, declared known mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9376 to 0.9496, SE ratio 0.9757 to 1.0133 | pass |
| `interval_calibration` | `binary_known_att_lower__shrunken_se_control` | control | lower plug-in omitted-variable bound on the ATT, declared known mechanism: the standard errors come from the known-mechanism curve plus the mechanism term of a study-side logistic fit, which is too narrow for a declared mechanism | the SE-ratio interval must fall below the calibration band | coverage 0.8946 to 0.9100, SE ratio 0.8353 to 0.8671 | pass |
| `interval_calibration` | `binary_known_att_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT, declared known mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9415 to 0.9531, SE ratio 0.9844 to 1.0215 | pass |
| `interval_calibration` | `binary_known_att_upper__shrunken_se_control` | control | upper plug-in omitted-variable bound on the ATT, declared known mechanism: the standard errors come from the known-mechanism curve plus the mechanism term of a study-side logistic fit, which is too narrow for a declared mechanism | the SE-ratio interval must fall below the calibration band | coverage 0.8981 to 0.9133, SE ratio 0.8424 to 0.8740 | pass |
| `interval_calibration` | `binary_logistic_atc_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC, in-sample logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9413 to 0.9529, SE ratio 0.9757 to 1.0135 | pass |
| `interval_calibration` | `binary_logistic_atc_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATC, in-sample logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9722 to 0.9801, SE ratio 1.1417 to 1.1856 | pass |
| `interval_calibration` | `binary_logistic_atc_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC, in-sample logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9474 to 0.9584, SE ratio 0.9926 to 1.0285 | pass |
| `interval_calibration` | `binary_logistic_atc_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATC, in-sample logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9760 to 0.9833, SE ratio 1.1618 to 1.2047 | pass |
| `interval_calibration` | `binary_logistic_ate_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE, in-sample logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9543, SE ratio 0.9773 to 1.0134 | pass |
| `interval_calibration` | `binary_logistic_ate_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE, in-sample logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9556, SE ratio 0.9869 to 1.0230 | pass |
| `interval_calibration` | `binary_logistic_att_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT, in-sample logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9556, SE ratio 0.9840 to 1.0205 | pass |
| `interval_calibration` | `binary_logistic_att_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATT, in-sample logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9739 to 0.9816, SE ratio 1.1514 to 1.1943 | pass |
| `interval_calibration` | `binary_logistic_att_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT, in-sample logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9454 to 0.9566, SE ratio 0.9913 to 1.0270 | pass |
| `interval_calibration` | `binary_logistic_att_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATT, in-sample logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9769 to 0.9841, SE ratio 1.1607 to 1.2016 | pass |
| `interval_calibration` | `binary_logistic_crossfit_atc_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9484 to 0.9593, SE ratio 0.9979 to 1.0350 | pass |
| `interval_calibration` | `binary_logistic_crossfit_atc_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9766 to 0.9839, SE ratio 1.1621 to 1.2052 | pass |
| `interval_calibration` | `binary_logistic_crossfit_atc_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9502 to 0.9609, SE ratio 0.9990 to 1.0360 | pass |
| `interval_calibration` | `binary_logistic_crossfit_atc_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9768 to 0.9840, SE ratio 1.1639 to 1.2067 | pass |
| `interval_calibration` | `binary_logistic_crossfit_ate_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE, five-fold cross-fitted logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9545, SE ratio 0.9839 to 1.0205 | pass |
| `interval_calibration` | `binary_logistic_crossfit_ate_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE, five-fold cross-fitted logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9469 to 0.9580, SE ratio 0.9850 to 1.0218 | pass |
| `interval_calibration` | `binary_logistic_crossfit_att_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9464 to 0.9575, SE ratio 0.9944 to 1.0312 | pass |
| `interval_calibration` | `binary_logistic_crossfit_att_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9753 to 0.9828, SE ratio 1.1577 to 1.2012 | pass |
| `interval_calibration` | `binary_logistic_crossfit_att_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9485 to 0.9594, SE ratio 0.9985 to 1.0355 | pass |
| `interval_calibration` | `binary_logistic_crossfit_att_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9745 to 0.9820, SE ratio 1.1632 to 1.2062 | pass |
| `interval_calibration` | `binary_logistic_ridge_atc_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC, logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9464 to 0.9575, SE ratio 0.9971 to 1.0339 | pass |
| `interval_calibration` | `binary_logistic_ridge_atc_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATC, logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9764 to 0.9837, SE ratio 1.1672 to 1.2094 | pass |
| `interval_calibration` | `binary_logistic_ridge_atc_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC, logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9550, SE ratio 0.9864 to 1.0236 | pass |
| `interval_calibration` | `binary_logistic_ridge_atc_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATC, logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9724 to 0.9803, SE ratio 1.1547 to 1.1993 | pass |
| `interval_calibration` | `binary_logistic_ridge_ate_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE, logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9469 to 0.9579, SE ratio 0.9965 to 1.0332 | pass |
| `interval_calibration` | `binary_logistic_ridge_ate_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE, logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9415 to 0.9531, SE ratio 0.9829 to 1.0194 | pass |
| `interval_calibration` | `binary_logistic_ridge_att_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT, logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9561, SE ratio 0.9947 to 1.0321 | pass |
| `interval_calibration` | `binary_logistic_ridge_att_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATT, logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9741 to 0.9818, SE ratio 1.1644 to 1.2077 | pass |
| `interval_calibration` | `binary_logistic_ridge_att_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT, logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9556, SE ratio 0.9854 to 1.0225 | pass |
| `interval_calibration` | `binary_logistic_ridge_att_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATT, logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9741 to 0.9818, SE ratio 1.1540 to 1.1959 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_atc_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9475 to 0.9585, SE ratio 1.0006 to 1.0386 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_atc_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9748 to 0.9823, SE ratio 1.1665 to 1.2110 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_atc_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9490 to 0.9598, SE ratio 1.0102 to 1.0482 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_atc_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATC, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9773 to 0.9844, SE ratio 1.1781 to 1.2230 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_ate_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9478 to 0.9587, SE ratio 0.9918 to 1.0292 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_ate_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9480 to 0.9589, SE ratio 1.0025 to 1.0403 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_att_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9491 to 0.9599, SE ratio 1.0022 to 1.0392 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_att_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9754 to 0.9829, SE ratio 1.1679 to 1.2117 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_att_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9513 to 0.9619, SE ratio 1.0168 to 1.0553 | pass |
| `interval_calibration` | `binary_logistic_ridge_crossfit_att_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATT, five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9785 to 0.9854, SE ratio 1.1860 to 1.2302 | pass |
| `interval_calibration` | `binary_unbalanced_atc_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC, in-sample logistic mechanism, treated share about 0.19: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9497 to 0.9605, SE ratio 1.0159 to 1.0548 | pass |
| `interval_calibration` | `binary_unbalanced_atc_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATC, in-sample logistic mechanism, treated share about 0.19: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9681 to 0.9766, SE ratio 1.1076 to 1.1484 | pass |
| `interval_calibration` | `binary_unbalanced_atc_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC, in-sample logistic mechanism, treated share about 0.19: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9503 to 0.9610, SE ratio 1.0187 to 1.0565 | pass |
| `interval_calibration` | `binary_unbalanced_atc_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATC, in-sample logistic mechanism, treated share about 0.19: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9672 to 0.9758, SE ratio 1.1098 to 1.1504 | pass |
| `interval_calibration` | `binary_unbalanced_ate_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE, in-sample logistic mechanism, treated share about 0.19: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9476 to 0.9586, SE ratio 1.0046 to 1.0422 | pass |
| `interval_calibration` | `binary_unbalanced_ate_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE, in-sample logistic mechanism, treated share about 0.19: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9484 to 0.9593, SE ratio 1.0035 to 1.0411 | pass |
| `interval_calibration` | `binary_unbalanced_att_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT, in-sample logistic mechanism, treated share about 0.19: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9534, SE ratio 0.9789 to 1.0158 | pass |
| `interval_calibration` | `binary_unbalanced_att_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATT, in-sample logistic mechanism, treated share about 0.19: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9727 to 0.9806, SE ratio 1.1403 to 1.1833 | pass |
| `interval_calibration` | `binary_unbalanced_att_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT, in-sample logistic mechanism, treated share about 0.19: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9392 to 0.9510, SE ratio 0.9727 to 1.0098 | pass |
| `interval_calibration` | `binary_unbalanced_att_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATT, in-sample logistic mechanism, treated share about 0.19: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9696 to 0.9779, SE ratio 1.1321 to 1.1756 | pass |
| `interval_calibration` | `multinomial_logistic_atc_1v0_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9541, SE ratio 0.9867 to 1.0248 | pass |
| `interval_calibration` | `multinomial_logistic_atc_1v0_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9795 to 0.9862, SE ratio 1.2183 to 1.2631 | pass |
| `interval_calibration` | `multinomial_logistic_atc_1v0_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9477 to 0.9587, SE ratio 0.9957 to 1.0324 | pass |
| `interval_calibration` | `multinomial_logistic_atc_1v0_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9816 to 0.9879, SE ratio 1.2282 to 1.2745 | pass |
| `interval_calibration` | `multinomial_logistic_ate_2v0_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9554, SE ratio 0.9938 to 1.0315 | pass |
| `interval_calibration` | `multinomial_logistic_ate_2v0_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9478 to 0.9587, SE ratio 1.0056 to 1.0424 | pass |
| `interval_calibration` | `multinomial_logistic_att_2v0_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9452 to 0.9564, SE ratio 0.9938 to 1.0306 | pass |
| `interval_calibration` | `multinomial_logistic_att_2v0_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9812 to 0.9877, SE ratio 1.2210 to 1.2662 | pass |
| `interval_calibration` | `multinomial_logistic_att_2v0_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9489 to 0.9598, SE ratio 1.0055 to 1.0430 | pass |
| `interval_calibration` | `multinomial_logistic_att_2v0_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9837 to 0.9896, SE ratio 1.2365 to 1.2814 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_atc_1v0_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9525 to 0.9630, SE ratio 1.0246 to 1.0625 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_atc_1v0_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9854 to 0.9911, SE ratio 1.2538 to 1.2996 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_atc_1v0_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9560 to 0.9661, SE ratio 1.0291 to 1.0659 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_atc_1v0_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATC of arm 1 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9863 to 0.9918, SE ratio 1.2572 to 1.3034 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_ate_2v0_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9556 to 0.9657, SE ratio 1.0332 to 1.0723 | **fail** |
| `interval_calibration` | `multinomial_logistic_crossfit_ate_2v0_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATE of arm 2 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9534 to 0.9637, SE ratio 1.0166 to 1.0539 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_att_2v0_lower__correctly_specified` | positive | lower plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9568 to 0.9667, SE ratio 1.0288 to 1.0683 | **fail** |
| `interval_calibration` | `multinomial_logistic_crossfit_att_2v0_lower__inflated_se_control` | control | lower plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9831 to 0.9892, SE ratio 1.2502 to 1.2974 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_att_2v0_upper__correctly_specified` | positive | upper plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9515 to 0.9620, SE ratio 1.0238 to 1.0613 | pass |
| `interval_calibration` | `multinomial_logistic_crossfit_att_2v0_upper__inflated_se_control` | control | upper plug-in omitted-variable bound on the ATT of arm 2 versus arm 0, five-fold cross-fitted three-arm multinomial logistic mechanism: the standard errors come from the package curve of nu^2 minus the mechanism term the study rebuilt, which is too wide at a correct model | the SE-ratio interval must fall above the calibration band | coverage 0.9870 to 0.9923, SE ratio 1.2440 to 1.2911 | pass |
<!-- /generated -->

## Failure accounting

A replicate is one parameter of one draw.

The package may refuse the limits of a parameter. The expected keys are `truncation_active`,
`correction_unstable` and `corrected_nonpositive`, and the package has other keys. The study
reads the refusal from the elements of the bound and raises `DataError` for any refusal of the
package and for no other reason.
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
`scenario`, `parameter`, `key`, `requested`, `lost` and `rate`. The key is the text that the
package names after `refused:`. Each parameter has a row for the three expected keys, for `other`
and for `all`, and one more row for each other package key that the run records. `other` is a
failure that is no refusal of the bound, such as a failed fit. The file has no verdict column. The
failure accounting judges the cells.

At a rate of 0.05 percent, the chance of more than 100 losses in 10,000 replicates is below
$10^{-12}$. The 99% upper endpoint of the rate is 0.2 percent. The balanced law gave that rate.
The pilot on the multinomial cross-fitted scenario refused up to 0.4 percent, which is about 40
losses in 10,000 replicates. The run reads the rates again.

The declared run lost 0 to 32 of 10,000 replicates for each parameter. The largest loss, 0.32
percent, belongs to the ATE and the ATT of arm 2 against arm 0 on `multinomial_logistic_crossfit`.
Every lost replicate has the key `truncation_active`. No other key occurred. No cell reached the
allowance of 100 lost replicates, so the verdict `bound exceeded` appears in no cell.

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

The declared run measured the same reading on all eight scenarios. The SE ratio of the corrected
$\nu^2$ is 1.17 to 1.24 in the nine cross-fitted parameters. It is 1.00 to 1.10 in the fifteen
in-sample parameters. The bound rows of the same draws read 0.99 to 1.05. `nu2-reading.csv` holds
every cell. No verdict reads it.

## Result

Every primary test passed. The property study has 80 cells. The 32 controls all leave the band on
their declared side, so the instrument can fail. 46 of the 48 positive cells sit inside their
calibration bands. The two red cells are named in the [red cells](#red-cells) section.

| part | reading |
| --- | --- |
| primary tests | every bound passes against its known value |
| positive cells | all pass except the lower bounds of the ATE and the ATT of arm 2 against arm 0 on `multinomial_logistic_crossfit` |
| inflated controls | the lowest SE-ratio endpoint is 1.1076, above the band edge of 1.07 |
| shrunken controls | the highest SE-ratio endpoint is 0.8779, below the band edge of 0.93 |
| lost replicates | at most 32 of 10,000 for one parameter, each with the key `truncation_active`. The allowance is 100 |
| corrected $\nu^2$ | measured and not judged. The [reading](#the-corrected-nu2-reading) states it |

The multinomial cross-fitted scenario reads SE ratios of 1.035 to 1.052 in its six positive cells.
No other positive cell reaches 1.04.

## Red cells

Two positive cells are red under the `reporting` policy. Both are lower bounds on
`multinomial_logistic_crossfit` at $n = 1000$, each with 32 lost replicates.

| cell | SE ratio (99% interval) | coverage (99% interval) | complete case | why the cell is red |
| --- | --- | --- | --- | --- |
| `multinomial_logistic_crossfit_ate_2v0_lower__correctly_specified` | 1.0523 (1.0332 to 1.0723) | 0.9609 (0.9556 to 0.9657) | fails | the SE-ratio interval ends above the band edge of 1.07 |
| `multinomial_logistic_crossfit_att_2v0_lower__correctly_specified` | 1.0479 (1.0288 to 1.0683) | 0.9620 (0.9568 to 0.9667) | passes | the worst-case fill of the 32 lost replicates puts the SE-ratio interval above 1.07 |

The upper bounds of the same two parameters pass, with SE ratios of 1.0347 and 1.0425.

**Rule 2: the diagnosis.** The diagnostic `tests/diagnostics/x26_red_cells/agreement.py` compares
the package curve of the plug-in $\hat\nu^2$ on 20 replicates of the scenario. The replicates are
the draws that the declared run fitted for these cells, and the refit reproduces their committed
standard errors. The infinitesimal jackknife is unit witness W5, and the numerical sandwich
follows `TestTheNumericalSandwich` with one coefficient vector for each fold. Both comparators read
no package curve.

| parameter | jackknife deviation | sandwich gap | declared-fit deviation | tolerance |
| --- | --- | --- | --- | --- |
| ATE, arm 2 against arm 0 | 2.3e-7 | 2.6e-9 | 4.4e-3 | 1e-6 |
| ATT, arm 2 against arm 0 | 6.2e-7 | 2.6e-9 | 6.9e-3 | 1e-6 |

The jackknife deviation is the largest gap to the package curve over the largest jackknife value,
for a fit whose treatment model is solved to full precision. The sandwich gap is the relative gap
between the sandwich variance and the variance of the package curve. Both are below the tolerance
of the unit witnesses, so the comparison finds no mismatch.

The declared fit solves its treatment model with `lbfgs`. Its curve departs from the jackknife by
the solver tolerance. That departure moves the reported lower standard error by a mean relative
amount of $8.2 \times 10^{-5}$ (ATE) and $-1.4 \times 10^{-5}$ (ATT). It does not explain a 5
percent excess.

A deliberate mutation shows that the comparison can fail. With the mechanism term of the curve
dropped, both comparators leave the tolerance by more than a factor of $10^{3}$
(`tests/unit/test_x26_red_cells_reading.py`).

**Rule 8: the reading of the ATE cell.** Both cells clear both bands from below. The lower
endpoints of the SE-ratio intervals are 1.0332 and 1.0288, against 0.93. The lower endpoints of the
coverage intervals are 0.9556 and 0.9568, against 0.92. The SE ratios are at least 1 and the coverages are at least 0.95.
Rule 2 found agreement, so the ATE cell reads "conservative". Its owner is
`X26-plugin-conservative` in the [red-cell ledger](red-cells.md).

The plain influence variance omits the second-order variance term that the owner dropped on
2026-10-16. That omission is consistent with a ratio above 1 in cross-fitted cells. The run does
not test that cause.

**Rule 7: the reading of the ATT cell.** Its complete case is inside the band, and it is red only
through the worst case. The cell reads "refusal-limited". The table gives the facts that rule 7
asks for.

| quantity | value |
| --- | --- |
| lost replicates | 32 of 10,000, each with the key `truncation_active` |
| refusal rate by key | 0.32 percent for `truncation_active`, and 0 for every other key |
| SE-ratio interval, complete case | 1.0288 to 1.0683, inside the band |
| SE-ratio leg, complete case and worst case | 0.0017 and -0.0048, where a negative value fails |

The ledger gives the ATT cell the owner `X26-refusal-limited`. Rule 4 does not make it a defect
suspect.

**Rule 3: the rerun.** The rerun at $n = 4000$ with 5,000 replicates has run. It decides between
"conservative, contracting" and "conservative, not contracting at n=4000". The
`tests/diagnostics/x26_red_cells/rerun.py` script keeps a checkpoint for each replicate, draws from
a dedicated seed stream and writes `rerun-rows.csv.gz` and `rerun-reading.csv`. The rerun covers
both cells. It lost no replicate, so no key refused any of the 5,000 draws.

| cell | SE ratio | 99% interval | coverage | upper bound, SE ratio |
| --- | --- | --- | --- | --- |
| ATE of arm 2 against arm 0, lower | 0.998 | 0.973 to 1.025 | 0.9506 | 1.002 |
| ATT of arm 2 against arm 0, lower | 1.002 | 0.977 to 1.027 | 0.9534 | 1.007 |

The SE ratio of the ATE cell falls from 1.0523 at $n = 1000$ to 0.998 at $n = 4000$. Its interval
contains 1, so the cell reads "conservative, contracting". The omitted second-order variance term
is consistent with the excess.

The ATT cell reads the same way at $n = 4000$. Its rule 7 reading, "refusal-limited", stands on the
declared run, because the rerun has no lost replicate to fill. The rerun is a diagnostic. It does
not change a verdict of the declared run.

## Red-cell policy

The policy was declared before the run, in `RED_CELL_POLICY` of
`tests/studies/omitted_variable_bound_plugin.py`.

| rule | when it applies | what happens |
| --- | --- | --- |
| 1 | always | the study publishes under `reporting`. No replicate count, margin, law, learner or strength changes after a verdict is seen |
| 2 | a red positive cell | the infinitesimal jackknife (unit witness W5) and the numerical sandwich are compared with the package curve on 20 replicates of the scenario. A mismatch of the curve is a defect. The defect is fixed, and the study is regenerated under a fresh declaration commit |
| 3 | rule 2 finds agreement | the cell is rerun at $n = 4000$ with 5,000 replicates. It is published red with owner `X26-plugin-finite-sample` and the reading "finite-sample, contracting" or "finite-sample, not contracting at n=4000" |
| 4 | the reading of a red positive cell | the cell is finite-sample only when its 99% SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. Any other red positive cell is a defect suspect, except a red cell that rule 7 or rule 8 reads |
| 5 | a control that does not leave the band | the control is reported as "control underpowered", with its ratio |
| 6 | the corrected $\nu^2$ rows | the rows are measured and not claimed. A shortfall in them opens no red cell |
| 7 | a cell that is green on the complete case and red only through `worst case fails` | the cell reads "refusal-limited" with owner `X26-refusal-limited`. It publishes the number of lost replicates, the complete-case value, the worst-case value and the refusal rate of each key in `refusal-reading.csv`. It is not a defect suspect under rule 4 |
| 8 | a red positive cell that clears both bands from below | the cell reads "conservative" with owner `X26-plugin-conservative`. Its 99% SE-ratio interval has a lower endpoint of at least 0.93. Its 99% coverage interval has a lower endpoint of at least 0.92. Its SE ratio is at least 1 and its coverage is at least 0.95. Rule 2 runs first, and a mismatch is a defect. With agreement, the $n = 4000$ rerun of rule 3 gives the reading "conservative, contracting" or "conservative, not contracting at n=4000". The plain influence variance omits the second-order variance term that the owner dropped on 2026-10-16, so the study expects this reading for cross-fitted cells. It is not a defect suspect under rule 4 |

The study declares no red cell before the run. The failure bound stays at 0.01 and the margins
stay as declared. A cell that loses more than 1 percent of its replicates ends with the verdict
`bound exceeded`, and the run reads that verdict like any other. The new pilot reads the design
first, and the declaration changes only under a fresh declaration commit. Rule 8 is a reading
rule. It changes no law, learner, margin, replicate count or cell.

The owner rows of the roadmap name the ids `X26-plugin-finite-sample`, `X26-plugin-conservative`
and `X26-refusal-limited`.

## Pilot

Two pilots read the design before the run. Neither pilot is evidence for a claim of this page.
Pilot numbers do not count as study evidence, and the generated tables below hold the only
evidence.

**The first pilot is superseded.** An earlier design of the study kept a second-order correction
of the variance of $\hat\nu^2$. The first pilot, on the seed stream `x26_pilot_sample`, measured
that design. It led the owner to drop the correction on 2026-10-16. At $n = 1000$ the correction
took about 60 percent of the cross-fitted variance, so its expansion was not trusted. It also
made the package refuse 1.7 to 4.2 percent of the cross-fitted three-arm draws with the key
`corrected_nonpositive`.

The figures without the correction came from the same draws, read again
with the correction removed. They are not a fresh pilot, and no draw of that pilot ran the new
design. Those figures were SE ratios of 0.97 to 1.08 (cross-fitted positive cells 1.067 to
1.078), coverages of 0.933 to 0.971 and at most 0.3 percent refusals. That pilot is not
committed, and no claim of this page rests on it.

**The second pilot read the design on fresh draws.** It used the seed stream `x26_pilot2_sample`
with 1,000 draws per scenario. The declared run does not use that stream, and the design test
checks that no declared seed does. The pilot is a planning record, not evidence. Its data are in
the repository `esbraun/cleverly-beta-plans`, under `_x26/pilot/no_vc_fresh`. The design that it
read is the one at commit `f352be8e`. The planning record does not store the commit that the
pilot ran at.

| quantity | pilot reading |
| --- | --- |
| refusals | 22 in all, each with the key `truncation_active`. The most for one parameter is 0.4 percent |
| refusals on the three wide-bound scenarios | 0.1 to 0.4 percent on each of the three |
| SE ratio of the positive cells | 0.965 to 1.063 |
| coverage of the positive cells | 0.933 to 0.970 |
| SE ratio of the inflated controls | at least 1.135 |
| SE ratio of the shrunken controls | at most 0.885 |

The pilot changed only the reading rules. It added rule 8 of the [red-cell policy](#red-cell-policy)
and the reading of every refusal key. It retuned no seed, size, margin, law, learner or scenario.
The study still declares no red cell, no expected refusal rate and no underpowered control.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 10000 | primary replications |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 48 | bound-end tests against truth |
| `independent_tests_passed` | 48 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 80 | repeated-sampling property cells |
| `property_cells_passed` | 78 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0967 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9441 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9379 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9693 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0807 | highest bootstrap primary SE-ratio endpoint |
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
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. A control must rise above it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 100 | the same bound, in primary replications |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes each verdict and does not fail on a red one |
| The corrected $\nu^2$ is not judged | At $n = 1000$ its standard error over-reads the sampling spread. The [reading](#the-corrected-nu2-reading) states the size. The bound intervals read near 1.0 to 1.04 in the diagnostic |
| $\hat B$ is a package value | The study rebuilds the curve and the bias curve of each bound. The exact-law witnesses in `tests/unit/test_omitted_variable_plugin_second_order.py` check $\hat B$ |
| Refusals are failed replicates | A refusal fails the cells of one parameter. A cell that loses more than 1 percent of its replicates ends with the verdict `bound exceeded`. No cell did: the largest loss is 32 of 10,000 replicates |
| One cell reads conservative and one reads refusal-limited | The lower bounds of the ATE and the ATT of arm 2 against arm 0 on `multinomial_logistic_crossfit` have SE ratios of 1.05, and their 99% intervals end at 1.07. The ATT cell is red only through the worst-case fill of 32 lost replicates. A user of the cross-fitted three-arm limits at $n = 1000$ should expect a reported standard error up to 7 percent above the sampling spread. At $n = 4000$ the rerun reads SE ratios of 0.998 and 1.002, so the excess contracts |
| Three scenarios declare wide truncation bounds | `multinomial_logistic`, `multinomial_logistic_crossfit` and `binary_unbalanced` declare `g_bounds=(0.005, 0.995)`. The automatic bounds refuse 27 to 52 percent of draws on those laws. The study does not test an active clip. The exact-law witness covers it |
| There is no cross-implementation evidence | No maintained implementation reports limits for the plug-in $\nu^2$ |
| Correctly specified mechanisms only | Each logistic model is correct for its law. A misspecified model covers $\nu^2$ of its own limit, which the study does not test |
| No weights, clusters, pipelines or grouped folds | The exact-law witnesses cover these compositions. This row does not |
| One sample size and one strength | The row covers $n = 1000$ and $c_Y = c_D = 0.5$, $\rho = 1$. The registered property loop has one sample size. The diagnostic `tests/diagnostics/x26_crossfit_bias` covers $n = 4000$ for the balanced law, in sample and cross-fitted. This study has no $n = 4000$ arm |
| The ridge scenarios use $C = 1$ only | The predicted first-order shrinkage of $\nu^2$ is at most 0.05 of its standard error on the binary law |
| Point treatment only | The bound refuses longitudinal fits, so the row covers none |
| Run cost | The declared run costs about 15 to 17 CPU hours at 10,000 replicates in both phases. The cost comes from the second pilot, which took 530 s on 4 workers for 1,000 draws of the 8 scenarios. The multinomial cross-fitted scenario is about 40 percent of it. The rerun of rule 3 costs about 1.7 CPU seconds for each replicate, so about 2.3 CPU hours for 5,000 replicates |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound_plugin/README.md)
gives the regeneration command.
