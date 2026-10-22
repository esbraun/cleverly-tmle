# Omitted-variable bound limit coverage

This study measures the coverage of the one-sided confidence limits that X82 part G added to
`omitted_variable_bounds`. Each part G configuration resolves its limit through the evidence
registry with the instrument `ovb_limit` and one rule. A passing cell adds one registry row for the
configuration it measured. The study reads each end of a bound as an estimate of its known
population value. It reads the standard error back off the reported limit.

**No canonical implementation is compared.** No maintained implementation computes these bounds
for a response mechanism, an intermediate variable, a regime, an MSM coefficient or a shift. The
zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| cells | O1 `response`, O2 `intermediate`, O3 `regime`, O4 `msm`, O5 `shift`. One cell for each `ovb_limit` rule |
| laws | `tests/studies/sensitivity_limit_laws.py`: three `W` levels, coded as two indicators, a binary `A` and `Y`, and an `A` by `W` interaction. O1 adds a response indicator that depends on `A` and `W`. O2 adds a binary `Z` that depends on `A` and `W`. O5 reads `tests/discrete_law_shift.py` on its four dose atoms |
| fit | in-sample TMLE. Every nuisance is a logistic regression on all pairwise products of its inputs, with `C = 1e8`. On these laws that model is saturated, so each fit is correctly specified |
| parameters | O1 `ate` and `ey1`. O2 the `ate` at `Z = 0` and at `Z = 1`. O3 `ey_regime[coin]` and `ate_regime[coin vs always]`. O4 `msm[W2]` and `msm[a:W2]` of `MSM.linear(modifiers=("W2",))`. O5 `ey_policy[+1]` and `ate_policy[+1 vs natural course]` |
| strength | $c_Y = 0.3$, $c_D = 0.2$, $\rho = 1$ |
| primary estimands | the lower and the upper bound of each parameter, 20 rows |
| primary standard error | `(lower - ci_lower) / z_0.95` for a lower bound, and `(ci_upper - upper) / z_0.95` for an upper bound |
| primary interval | the bound plus or minus 1.959964 standard errors |
| truth | closed forms of $\psi$, $\sigma^2$ and $\nu^2$ from the law tables. The design test checks each one against the package bound on the weighted exact frame, to 1e-5 |
| size | 1,600 replications at $n = 2000$ per cell, and 4,000 property draws per cell |
| Monte Carlo inference | 99% intervals around every declared endpoint |

`X82-SG-plan.md` rev 2 in the plans repository fixed the cells, the budget, the seeds, the controls
and the red-cell route before any run. A pilot on seeds outside the declared streams sized the run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| part G law, outcome missing at random given A and W; saturated GLMs | `ate_lower` | lower omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0019 to 0.0017 | 0.9613 | 1.0328 | pass |
| part G law, outcome missing at random given A and W; saturated GLMs | `ate_upper` | upper omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0016 to 0.0018 | 0.9569 | 1.0241 | pass |
| part G law, outcome missing at random given A and W; saturated GLMs | `ey1_lower` | lower omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000842 to 0.0019 | 0.9544 | 1.0213 | pass |
| part G law, outcome missing at random given A and W; saturated GLMs | `ey1_upper` | upper omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000516 to 0.0020 | 0.9537 | 1.0298 | pass |
| part G law with a binary intermediate Z; saturated GLMs | `cde_z0_lower` | lower omitted-variable bound on the controlled direct effect at Z = 0, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0022 to 0.0021 | 0.9537 | 1.0096 | pass |
| part G law with a binary intermediate Z; saturated GLMs | `cde_z0_upper` | upper omitted-variable bound on the controlled direct effect at Z = 0, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0018 to 0.0021 | 0.9544 | 1.0202 | pass |
| part G law with a binary intermediate Z; saturated GLMs | `cde_z1_lower` | lower omitted-variable bound on the controlled direct effect at Z = 1, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0046 to 0.000648 | 0.9456 | 0.9631 | pass |
| part G law with a binary intermediate Z; saturated GLMs | `cde_z1_upper` | upper omitted-variable bound on the controlled direct effect at Z = 1, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000862 to 0.0041 | 0.9394 | 0.9631 | pass |
| part G law, complete data, three regimes; saturated GLMs | `coin_lower` | lower omitted-variable bound on the mean under the stochastic regime coin, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000965 to 0.000793 | 0.9506 | 0.9960 | pass |
| part G law, complete data, three regimes; saturated GLMs | `coin_upper` | upper omitted-variable bound on the mean under the stochastic regime coin, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000915 to 0.000778 | 0.9463 | 0.9929 | pass |
| part G law, complete data, three regimes; saturated GLMs | `coin_vs_always_lower` | lower omitted-variable bound on the contrast of the regime coin with always treat, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000386 to 0.000503 | 0.9513 | 0.9717 | pass |
| part G law, complete data, three regimes; saturated GLMs | `coin_vs_always_upper` | upper omitted-variable bound on the contrast of the regime coin with always treat, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000468 to 0.000482 | 0.9456 | 0.9813 | pass |
| part G law, complete data, linear MSM with modifier W2; saturated GLMs | `a_w2_lower` | lower omitted-variable bound on the treatment-by-W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0077 to 0.000854 | 0.9450 | 0.9811 | pass |
| part G law, complete data, linear MSM with modifier W2; saturated GLMs | `a_w2_upper` | upper omitted-variable bound on the treatment-by-W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0056 to 0.0032 | 0.9431 | 0.9762 | pass |
| part G law, complete data, linear MSM with modifier W2; saturated GLMs | `w2_lower` | lower omitted-variable bound on the W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0019 to 0.0027 | 0.9463 | 0.9946 | pass |
| part G law, complete data, linear MSM with modifier W2; saturated GLMs | `w2_upper` | upper omitted-variable bound on the W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0025 to 0.0019 | 0.9413 | 0.9870 | pass |
| `tests.discrete_law_shift` on its four dose atoms; saturated GLMs | `plus1_lower` | lower omitted-variable bound on the mean under the shift +1 on the dose atoms, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000945 to 0.000887 | 0.9469 | 0.9886 | pass |
| `tests.discrete_law_shift` on its four dose atoms; saturated GLMs | `plus1_upper` | upper omitted-variable bound on the mean under the shift +1 on the dose atoms, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000885 to 0.000800 | 0.9487 | 0.9921 | pass |
| `tests.discrete_law_shift` on its four dose atoms; saturated GLMs | `plus1_vs_natural_lower` | lower omitted-variable bound on the contrast of the shift +1 with the natural course, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.0011 to 0.000039 | 0.9513 | 0.9993 | pass |
| `tests.discrete_law_shift` on its four dose atoms; saturated GLMs | `plus1_vs_natural_upper` | upper omitted-variable bound on the contrast of the shift +1 with the natural course, at cf_y 0.3, cf_d 0.2, rho 1 | `cleverly` omitted-variable bound limits (X82 part G) | -0.000184 to 0.000873 | 0.9513 | 1.0136 | pass |
<!-- /generated -->

## Theory properties

Each scenario's cells read the same 4,000 draws, and one fit per draw gives every row. A positive
cell must put its SE-ratio interval inside the calibration band and its coverage interval inside
the coverage band. Each row also has an `inflated_se_control`, which multiplies the reported
standard error by 1.5. Its SE-ratio interval must rise above the band. The
`response_mask_necessity` family pairs each O1 row with a control whose $\sigma^2$ is the mean over
the observed rows only. The control's bias must leave the margin, and the pair must move by at
least 0.25 positive-arm standard deviations.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `O1_ate_lower__correctly_specified` | positive | lower omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9596, SE ratio 0.9775 to 1.0354 | pass |
| `interval_calibration` | `O1_ate_lower__inflated_se_control` | control | lower omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9936 to 0.9986, SE ratio 1.4670 to 1.5530 | pass |
| `interval_calibration` | `O1_ate_upper__correctly_specified` | positive | upper omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9396 to 0.9578, SE ratio 0.9686 to 1.0285 | pass |
| `interval_calibration` | `O1_ate_upper__inflated_se_control` | control | upper omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9940 to 0.9988, SE ratio 1.4533 to 1.5404 | pass |
| `interval_calibration` | `O1_ey1_lower__correctly_specified` | positive | lower omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9621, SE ratio 0.9781 to 1.0339 | pass |
| `interval_calibration` | `O1_ey1_lower__inflated_se_control` | control | lower omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9940 to 0.9988, SE ratio 1.4662 to 1.5526 | pass |
| `interval_calibration` | `O1_ey1_upper__correctly_specified` | positive | upper omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9575, SE ratio 0.9638 to 1.0185 | pass |
| `interval_calibration` | `O1_ey1_upper__inflated_se_control` | control | upper omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9943 to 0.9989, SE ratio 1.4448 to 1.5294 | pass |
| `interval_calibration` | `O2_cde_z0_lower__correctly_specified` | positive | lower omitted-variable bound on the controlled direct effect at Z = 0, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9603, SE ratio 0.9754 to 1.0313 | pass |
| `interval_calibration` | `O2_cde_z0_lower__inflated_se_control` | control | lower omitted-variable bound on the controlled direct effect at Z = 0, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9926 to 0.9981, SE ratio 1.4645 to 1.5491 | pass |
| `interval_calibration` | `O2_cde_z0_upper__correctly_specified` | positive | upper omitted-variable bound on the controlled direct effect at Z = 0, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9415 to 0.9594, SE ratio 0.9740 to 1.0307 | pass |
| `interval_calibration` | `O2_cde_z0_upper__inflated_se_control` | control | upper omitted-variable bound on the controlled direct effect at Z = 0, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9930 to 0.9983, SE ratio 1.4615 to 1.5463 | pass |
| `interval_calibration` | `O2_cde_z1_lower__correctly_specified` | positive | lower omitted-variable bound on the controlled direct effect at Z = 1, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9372 to 0.9557, SE ratio 0.9661 to 1.0238 | pass |
| `interval_calibration` | `O2_cde_z1_lower__inflated_se_control` | control | lower omitted-variable bound on the controlled direct effect at Z = 1, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9940 to 0.9988, SE ratio 1.4480 to 1.5366 | pass |
| `interval_calibration` | `O2_cde_z1_upper__correctly_specified` | positive | upper omitted-variable bound on the controlled direct effect at Z = 1, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9348 to 0.9536, SE ratio 0.9554 to 1.0120 | pass |
| `interval_calibration` | `O2_cde_z1_upper__inflated_se_control` | control | upper omitted-variable bound on the controlled direct effect at Z = 1, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9920 to 0.9978, SE ratio 1.4320 to 1.5177 | pass |
| `interval_calibration` | `O3_coin_lower__correctly_specified` | positive | lower omitted-variable bound on the mean under the stochastic regime coin, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9596, SE ratio 0.9687 to 1.0261 | pass |
| `interval_calibration` | `O3_coin_lower__inflated_se_control` | control | lower omitted-variable bound on the mean under the stochastic regime coin, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9917 to 0.9976, SE ratio 1.4520 to 1.5396 | pass |
| `interval_calibration` | `O3_coin_upper__correctly_specified` | positive | upper omitted-variable bound on the mean under the stochastic regime coin, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9396 to 0.9578, SE ratio 0.9748 to 1.0325 | pass |
| `interval_calibration` | `O3_coin_upper__inflated_se_control` | control | upper omitted-variable bound on the mean under the stochastic regime coin, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9933 to 0.9984, SE ratio 1.4625 to 1.5504 | pass |
| `interval_calibration` | `O3_coin_vs_always_lower__correctly_specified` | positive | lower omitted-variable bound on the contrast of the regime coin with always treat, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9404 to 0.9585, SE ratio 0.9741 to 1.0300 | pass |
| `interval_calibration` | `O3_coin_vs_always_lower__inflated_se_control` | control | lower omitted-variable bound on the contrast of the regime coin with always treat, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9933 to 0.9984, SE ratio 1.4602 to 1.5445 | pass |
| `interval_calibration` | `O3_coin_vs_always_upper__correctly_specified` | positive | upper omitted-variable bound on the contrast of the regime coin with always treat, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9432 to 0.9607, SE ratio 0.9740 to 1.0303 | pass |
| `interval_calibration` | `O3_coin_vs_always_upper__inflated_se_control` | control | upper omitted-variable bound on the contrast of the regime coin with always treat, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9930 to 0.9983, SE ratio 1.4595 to 1.5445 | pass |
| `interval_calibration` | `O4_a_w2_lower__correctly_specified` | positive | lower omitted-variable bound on the treatment-by-W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9377 to 0.9562, SE ratio 0.9630 to 1.0216 | pass |
| `interval_calibration` | `O4_a_w2_lower__inflated_se_control` | control | lower omitted-variable bound on the treatment-by-W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9926 to 0.9981, SE ratio 1.4441 to 1.5319 | pass |
| `interval_calibration` | `O4_a_w2_upper__correctly_specified` | positive | upper omitted-variable bound on the treatment-by-W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9388 to 0.9571, SE ratio 0.9666 to 1.0248 | pass |
| `interval_calibration` | `O4_a_w2_upper__inflated_se_control` | control | upper omitted-variable bound on the treatment-by-W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9926 to 0.9981, SE ratio 1.4504 to 1.5364 | pass |
| `interval_calibration` | `O4_w2_lower__correctly_specified` | positive | lower omitted-variable bound on the W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9440 to 0.9614, SE ratio 0.9729 to 1.0292 | pass |
| `interval_calibration` | `O4_w2_lower__inflated_se_control` | control | lower omitted-variable bound on the W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9940 to 0.9988, SE ratio 1.4592 to 1.5433 | pass |
| `interval_calibration` | `O4_w2_upper__correctly_specified` | positive | upper omitted-variable bound on the W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9332 to 0.9523, SE ratio 0.9664 to 1.0232 | pass |
| `interval_calibration` | `O4_w2_upper__inflated_se_control` | control | upper omitted-variable bound on the W2 coefficient of the linear MSM with modifier W2, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9947 to 0.9991, SE ratio 1.4494 to 1.5325 | pass |
| `interval_calibration` | `O5_plus1_lower__correctly_specified` | positive | lower omitted-variable bound on the mean under the shift +1 on the dose atoms, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9596, SE ratio 0.9812 to 1.0411 | pass |
| `interval_calibration` | `O5_plus1_lower__inflated_se_control` | control | lower omitted-variable bound on the mean under the shift +1 on the dose atoms, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9936 to 0.9986, SE ratio 1.4729 to 1.5616 | pass |
| `interval_calibration` | `O5_plus1_upper__correctly_specified` | positive | upper omitted-variable bound on the mean under the shift +1 on the dose atoms, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9612, SE ratio 0.9800 to 1.0372 | pass |
| `interval_calibration` | `O5_plus1_upper__inflated_se_control` | control | upper omitted-variable bound on the mean under the shift +1 on the dose atoms, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9947 to 0.9991, SE ratio 1.4674 to 1.5565 | pass |
| `interval_calibration` | `O5_plus1_vs_natural_lower__correctly_specified` | positive | lower omitted-variable bound on the contrast of the shift +1 with the natural course, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9348 to 0.9536, SE ratio 0.9667 to 1.0237 | pass |
| `interval_calibration` | `O5_plus1_vs_natural_lower__inflated_se_control` | control | lower omitted-variable bound on the contrast of the shift +1 with the natural course, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9933 to 0.9984, SE ratio 1.4480 to 1.5352 | pass |
| `interval_calibration` | `O5_plus1_vs_natural_upper__correctly_specified` | positive | upper omitted-variable bound on the contrast of the shift +1 with the natural course, at cf_y 0.3, cf_d 0.2, rho 1: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9596, SE ratio 0.9779 to 1.0344 | pass |
| `interval_calibration` | `O5_plus1_vs_natural_upper__inflated_se_control` | control | upper omitted-variable bound on the contrast of the shift +1 with the natural course, at cf_y 0.3, cf_d 0.2, rho 1: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9947 to 0.9991, SE ratio 1.4659 to 1.5529 | pass |
| `response_mask_necessity` | `O1_ate_lower__masked` | positive | lower omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the package bound, whose sigma^2 is the masked mean over all rows | bias interval inside the equivalence margin | bias -0.000913 to 0.0014, margin 0.0072 | pass |
| `response_mask_necessity` | `O1_ate_lower__unmasked_sigma2` | control | lower omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: sigma^2 read as the mean over the observed rows only | bias interval must fall entirely outside the margin | bias -0.0461 to -0.0437, margin 0.0073 | pass |
| `response_mask_necessity` | `O1_ate_upper__masked` | positive | upper omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the package bound, whose sigma^2 is the masked mean over all rows | bias interval inside the equivalence margin | bias -0.000480 to 0.0017, margin 0.0068 | pass |
| `response_mask_necessity` | `O1_ate_upper__unmasked_sigma2` | control | upper omitted-variable bound on the ATE, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: sigma^2 read as the mean over the observed rows only | bias interval must fall entirely outside the margin | bias 0.0446 to 0.0469, margin 0.0068 | pass |
| `response_mask_necessity` | `O1_ey1_lower__masked` | positive | lower omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the package bound, whose sigma^2 is the masked mean over all rows | bias interval inside the equivalence margin | bias -0.0012 to 0.000570, margin 0.0053 | pass |
| `response_mask_necessity` | `O1_ey1_lower__unmasked_sigma2` | control | lower omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: sigma^2 read as the mean over the observed rows only | bias interval must fall entirely outside the margin | bias -0.0340 to -0.0322, margin 0.0054 | pass |
| `response_mask_necessity` | `O1_ey1_upper__masked` | positive | upper omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: the package bound, whose sigma^2 is the masked mean over all rows | bias interval inside the equivalence margin | bias -0.000780 to 0.000884, margin 0.0051 | pass |
| `response_mask_necessity` | `O1_ey1_upper__unmasked_sigma2` | control | upper omitted-variable bound on the arm-1 mean, outcome missing at random, at cf_y 0.3, cf_d 0.2, rho 1: sigma^2 read as the mean over the observed rows only | bias interval must fall entirely outside the margin | bias 0.0320 to 0.0337, margin 0.0052 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The registry gains one `ovb_limit` row for
each rule: `response`, `intermediate`, `regime`, `msm` and `shift`. Each row covers the measured
configuration only, as the table below states.

| rule | covered configuration |
| --- | --- |
| `response` | `ate` or `ey1`, binary outcome, two arms |
| `intermediate` | the `ate` at a level of `Z`, binary outcome, two arms |
| `regime` | `ey_regime[coin]` or `ate_regime[coin vs always]`, binary outcome |
| `msm` | `msm[W2]` or `msm[a:W2]`, binary outcome |
| `shift` | `ey_policy[+1]` or `ate_policy[+1 vs natural course]`, a dose on declared atoms |

Every row also requires in-sample TMLE, the saturated logistic learner above for every nuisance,
influence-curve inference, no declared mechanism, no repeats, $n \ge 2000$, at most 0.001
adjustment columns per row, and a smallest event count of 250 (O1, O2) or 350 (O3, O4).

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 20 | bound-end tests against truth |
| `independent_tests_passed` | 20 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 48 | repeated-sampling property cells |
| `property_cells_passed` | 48 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0600 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9394 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9223 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9208 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0791 | highest bootstrap primary SE-ratio endpoint |
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
| `margin:response_mask_displacement` | 0.2500 | minimum displacement of the observed-rows sigma^2 control |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 16 | the same bound, in primary replications |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained implementation computes these bounds |
| One law, size, strength and learner per rule | A fit outside the covered configuration prints the descriptive sentence and names the covered one |
| Single-feature rules only | A combined rule such as `response+regime` stays descriptive |
| The scope names the measured parameters | A regime with another name, another MSM term, or another shift is outside the row |
| The MSM Gram term has no study control | Removing it moves the limit SE by at most 4% up to $n = 40000$, so no SE-ratio control can see it. The Gateaux unit test in `tests/unit/test_omitted_variable_axes.py` is its witness |
| `stratum`, `missing_treatment` and `repeats` have no cell | Those limits stay descriptive |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ovb_limit_coverage/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ovb_limit_coverage/manifest.json)
records the seeds, margins, exact estimator configuration, source hashes, and result hashes.
`cells.csv` records the fit configuration of each row.
