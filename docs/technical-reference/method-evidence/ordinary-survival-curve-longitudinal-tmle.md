# Ordinary survival-curve longitudinal TMLE

This study validates ordinary, non-cross-fitted cumulative risk estimation under absorbing
failure. It covers two horizons, monotone censoring, static plans, and a dynamic plan. The
canonical comparison uses R [`ltmle`](https://www.jstatsoft.org/article/view/v081i01) 1.3-0 with
`survivalOutcome=TRUE`.

The study reports each unique parameter once. At the first horizon, the dynamic plan equals the
always-treated plan by construction. A structural test checks that identity instead of counting
the duplicate as repeated-sampling evidence.

## What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| datasets | 1,600 censored survival panels generated in Python | the identical rows |
| horizons | cumulative risk by times one and two | one `survivalOutcome=TRUE` fit for each data prefix |
| plans | never treat, always treat, and continue after initial treatment when L2 is positive | the same unique plan and horizon combinations |
| contrasts | always-minus-never at both horizons and dynamic-minus-never at time two | differences of rowwise influence curves, preserving covariance |
| mechanisms | the generating treatment and censoring probabilities | the same fixed probabilities |
| sequential regressions | follower-stratified quasibinomial | the same formulas and link |
| cumulative-g bounds | nonbinding | nonbinding |
| intervals | pointwise 95% identity-scale Wald, influence-curve variance | the same, with `variance.method="ic"` |

R fits each requested prefix because `ltmle` returns the terminal cumulative risk for that input.
The Python implementation performs one backward pass per horizon. Comparing the prefixes tests the
same horizon-specific parameters without treating a terminal outcome as a survival curve.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` | -0.0022 to 0.000243 | 0.9494 | 1.0064 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `ltmle` | -0.0022 to 0.000243 | 0.9494 | 1.0064 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` | -0.0036 to 0.000104 | 0.9406 | 0.9938 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `ltmle` | -0.0036 to 0.000104 | 0.9406 | 0.9938 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | `cleverly` | -0.0038 to 0.000020 | 0.9387 | 0.9923 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | R `ltmle` | -0.0038 to 0.000020 | 0.9387 | 0.9923 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | `cleverly` | -0.0012 to 0.000324 | 0.9444 | 1.0076 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | R `ltmle` | -0.0012 to 0.000324 | 0.9444 | 1.0076 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | `cleverly` | -0.0018 to 0.000412 | 0.9544 | 1.0105 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | R `ltmle` | -0.0018 to 0.000412 | 0.9544 | 1.0105 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | `cleverly` | -0.000434 to 0.0015 | 0.9519 | 0.9976 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | R `ltmle` | -0.000434 to 0.0015 | 0.9519 | 0.9976 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | `cleverly` | -0.000431 to 0.0025 | 0.9425 | 0.9999 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | R `ltmle` | -0.000431 to 0.0025 | 0.9425 | 0.9999 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | `cleverly` | -0.0020 to 0.000348 | 0.9537 | 1.0104 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | R `ltmle` | -0.0020 to 0.000348 | 0.9537 | 1.0104 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 3.324e-11 | 1.160e-08 | 1.0000 | 0 | 1.353e-09 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -3.360e-10 | 7.845e-08 | 1.0000 | 0 | 4.322e-08 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | -3.903e-10 | 8.898e-08 | 1.0000 | 0 | 4.072e-08 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | -8.728e-11 | 4.910e-08 | 1.0000 | 0 | 2.193e-09 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | -2.473e-10 | 9.571e-08 | 1.0000 | 0 | 1.359e-09 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | -1.205e-10 | 5.271e-08 | 1.0000 | 0 | 1.342e-09 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | 8.867e-11 | 2.594e-08 | 1.0000 | 0 | 5.747e-08 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | -3.016e-10 | 1.104e-07 | 1.0000 | 0 | 1.204e-09 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `dynamic_t2__both_correct` | positive | dynamic plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0023 to 0.0042, margin 0.0109 | pass |
| `double_robustness` | `dynamic_t2__both_wrong` | control | dynamic plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0447 to -0.0382, margin 0.0109 | pass |
| `double_robustness` | `dynamic_t2__mechanism_correct` | positive | dynamic plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0029 to 0.0037, margin 0.0111 | pass |
| `double_robustness` | `dynamic_t2__outcome_correct` | positive | dynamic plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0010 to 0.0055, margin 0.0109 | pass |
| `double_robustness` | `static_t1__both_correct` | positive | static plan at horizon one: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0022 to 0.0018, margin 0.0067 | pass |
| `double_robustness` | `static_t1__both_wrong` | control | static plan at horizon one: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0449 to -0.0408, margin 0.0068 | pass |
| `double_robustness` | `static_t1__mechanism_correct` | positive | static plan at horizon one: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0024 to 0.0016, margin 0.0068 | pass |
| `double_robustness` | `static_t1__outcome_correct` | positive | static plan at horizon one: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.000895 to 0.0033, margin 0.0071 | pass |
| `double_robustness` | `static_t2__both_correct` | positive | static plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0025 to 0.0043, margin 0.0114 | pass |
| `double_robustness` | `static_t2__both_wrong` | control | static plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0368 to -0.0301, margin 0.0113 | pass |
| `double_robustness` | `static_t2__mechanism_correct` | positive | static plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0026 to 0.0042, margin 0.0113 | pass |
| `double_robustness` | `static_t2__outcome_correct` | positive | static plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.000743 to 0.0059, margin 0.0112 | pass |
| `interval_calibration` | `dynamic_t2__correctly_specified` | positive | dynamic plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9300 to 0.9429, SE ratio 0.9483 to 0.9849, empirical efficiency ratio 1.0094 to 1.0482, reported efficiency ratio 0.9922 to 0.9961 | pass |
| `interval_calibration` | `dynamic_t2__noise_control` | control | dynamic plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8113 to 0.8315, SE ratio 0.6760 to 0.7028, empirical efficiency ratio 1.4154 to 1.4707, reported efficiency ratio 0.9922 to 0.9961 | pass |
| `interval_calibration` | `dynamic_t2__shrunken_se_control` | control | dynamic plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8042 to 0.8247, SE ratio 0.6636 to 0.6894, empirical efficiency ratio 1.0096 to 1.0484, reported efficiency ratio 0.6945 to 0.6972 | pass |
| `interval_calibration` | `static_t1__correctly_specified` | positive | static plan at horizon one: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9393 to 0.9513, SE ratio 0.9700 to 1.0068, empirical efficiency ratio 0.9930 to 1.0306, reported efficiency ratio 0.9993 to 1.0003 | pass |
| `interval_calibration` | `static_t1__noise_control` | control | static plan at horizon one: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8175 to 0.8374, SE ratio 0.6827 to 0.7091, empirical efficiency ratio 1.4098 to 1.4644, reported efficiency ratio 0.9993 to 1.0003 | pass |
| `interval_calibration` | `static_t1__shrunken_se_control` | control | static plan at horizon one: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8177 to 0.8376, SE ratio 0.6788 to 0.7047, empirical efficiency ratio 0.9932 to 1.0310, reported efficiency ratio 0.6995 to 0.7002 | pass |
| `interval_calibration` | `static_t2__correctly_specified` | positive | static plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9276 to 0.9407, SE ratio 0.9411 to 0.9774, empirical efficiency ratio 1.0122 to 1.0515, reported efficiency ratio 0.9874 to 0.9914 | pass |
| `interval_calibration` | `static_t2__noise_control` | control | static plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8060 to 0.8264, SE ratio 0.6757 to 0.7008, empirical efficiency ratio 1.4119 to 1.4641, reported efficiency ratio 0.9874 to 0.9915 | pass |
| `interval_calibration` | `static_t2__shrunken_se_control` | control | static plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8001 to 0.8208, SE ratio 0.6585 to 0.6842, empirical efficiency ratio 1.0125 to 1.0519, reported efficiency ratio 0.6911 to 0.6941 | pass |
| `power` | `dynamic_t2__alternative` | positive | dynamic plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t1__alternative` | positive | static plan at horizon one: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t2__alternative` | positive | static plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_1000` | control | dynamic plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.0020, coverage 0.8766 to 0.9309, SE ratio 0.8690 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_2000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0016, coverage 0.9136 to 0.9585, SE ratio 0.9810 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_8000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000974, coverage 0.9150 to 0.9596, SE ratio 0.9872 | pass |
| `root_n_and_efficiency` | `static_t1__n_1000` | control | static plan at horizon one: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000396, coverage 0.9107 to 0.9565, SE ratio 0.9596 | pass |
| `root_n_and_efficiency` | `static_t1__n_2000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012, coverage 0.9341 to 0.9727, SE ratio 1.0450 | pass |
| `root_n_and_efficiency` | `static_t1__n_8000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000241, coverage 0.9179 to 0.9616, SE ratio 0.9838 | pass |
| `root_n_and_efficiency` | `static_t2__n_1000` | control | static plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.000599, coverage 0.8696 to 0.9255, SE ratio 0.8669 | pass |
| `root_n_and_efficiency` | `static_t2__n_2000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0024, coverage 0.9208 to 0.9637, SE ratio 0.9893 | pass |
| `root_n_and_efficiency` | `static_t2__n_8000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000698, coverage 0.9121 to 0.9575, SE ratio 0.9733 | pass |
| `root_n_rate` | `dynamic_t2__empirical_sd` | positive | dynamic plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5866 to -0.4928 | pass |
| `root_n_rate` | `dynamic_t2__reported_se` | positive | dynamic plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4906 to -0.4814 | pass |
| `root_n_rate` | `static_t1__empirical_sd` | positive | static plan at horizon one: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5463 to -0.4630 | pass |
| `root_n_rate` | `static_t1__reported_se` | positive | static plan at horizon one: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5008 to -0.4986 | pass |
| `root_n_rate` | `static_t2__empirical_sd` | positive | static plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5701 to -0.4752 | pass |
| `root_n_rate` | `static_t2__reported_se` | positive | static plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4811 to -0.4709 | pass |
| `survival_recursion_necessity` | `always_t2__survival` | positive | always-treat risk at horizon two: the survival estimator keeps failures in their event node and removes them afterward | bias interval inside the equivalence margin | bias -0.0038 to 0.0022, margin 0.0100 | pass |
| `survival_recursion_necessity` | `always_t2__survivor_only` | control | always-treat risk at horizon two: the same horizon-two outcome analyzed only among first-node survivors | bias interval must fall entirely outside the margin | bias -0.1523 to -0.1430, margin 0.0155 | pass |
| `targeting_necessity` | `dynamic_t2__targeted` | positive | dynamic plan at horizon two: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0035 to 0.0029, margin 0.0108 | pass |
| `targeting_necessity` | `dynamic_t2__untargeted` | control | dynamic plan at horizon two: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0444 to -0.0379, margin 0.0108 | pass |
| `targeting_necessity` | `static_t1__targeted` | positive | static plan at horizon one: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0019 to 0.0021, margin 0.0068 | pass |
| `targeting_necessity` | `static_t1__untargeted` | control | static plan at horizon one: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0448 to -0.0408, margin 0.0067 | pass |
| `targeting_necessity` | `static_t2__targeted` | positive | static plan at horizon two: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0033 to 0.0034, margin 0.0113 | pass |
| `targeting_necessity` | `static_t2__untargeted` | control | static plan at horizon two: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0366 to -0.0299, margin 0.0114 | pass |
| `type_i_error` | `dynamic_t2__sharp_null` | positive | dynamic plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0537, 0.0353 to 0.0777 | pass |
| `type_i_error` | `static_t1__sharp_null` | positive | static plan at horizon one: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0500, 0.0323 to 0.0733 | pass |
| `type_i_error` | `static_t2__sharp_null` | positive | static plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0575, 0.0384 to 0.0821 | pass |
<!-- /generated -->

The property study samples an exact binary survival law. It checks static risks at both horizons
and the dynamic risk at time two. Its independent instruments include double robustness, exact-EIF
efficiency, root-n rates, calibration, null size, power, and targeting controls.

The survival-recursion control is separate from targeting. It replaces cumulative risk with a
survivor-only terminal hazard while leaving the rest of the procedure unchanged. The control must
miss the exact time-two risk, while the correct recursion must recover it.

The two horizons do not behave alike. The study measures that difference on one set of draws, so
the comparison is internal to the calibration cell rather than across studies.

The first horizon is calibrated. Its SE ratio and its coverage both surround their nominal values.
At the second horizon the whole SE-ratio interval sits below one and the whole coverage interval
sits below nominal. The empirical spread runs above the exact efficiency bound there, while the
reported standard error sits on it. So the reported standard error understates the sampling spread
at the second horizon by a few percent. The measured-values table below carries all four endpoints.

Both intervals stay inside the declared calibration bands, so the cell passes. The second backward
regression is the source, and it is why this cell carries four times the shared calibration budget.
A reader who needs calibrated horizon-two inference at this sample size should treat the shortfall
as measured rather than as absent.

The sharp null keeps the treatment, censoring, and `L2` mechanisms of the law and replaces the two
hazards. Every replacement value is a multiple of one quarter, so an `N`-row sample realises the
null law exactly, as it does the law it derives from. All three contrasts are exactly zero under
it.

At the second horizon a baseline-only standardisation returns 0.0349 rather than zero, so the null
is one an estimator has to be longitudinal to find. At the first horizon it returns exactly zero,
because no time-varying node precedes the first event node. The first-horizon type-I cells
therefore test baseline and censoring adjustment and not longitudinal adjustment. A crude
comparison of arms is biased by -0.05 under the null at that horizon, so those cells are not
vacuous either.

The power alternative keeps the law's own first hazard and replaces the second. It gives the two
horizon-two contrasts different values, so the two horizon-two power cells report two parameters
rather than one number twice.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_total` | 16 | implementation-estimand truth tests |
| `independent_tests_passed` | 16 | truth tests passing |
| `paired_tests_total` | 8 | paired estimand comparisons |
| `paired_tests_passed` | 8 | paired comparisons passing |
| `property_cells_total` | 50 | independent property cells |
| `property_cells_passed` | 50 | property cells passing |
| `max_standardized_bias` | 0.0638 | largest primary standardized bias |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9216 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9491 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0615 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 1.104e-07 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0000 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 8.055e-08 | largest paired calibration-excess bound |
| `max_targeting_displacement` | 0.3560 | largest final-fluctuation move, in standard errors |
| `median_targeting_displacement` | 0.0343 | median final-fluctuation move, in standard errors |
| `properties[targeting_necessity/static_t1__targeted]:targeting_displacement` | 0.7387 | least-displaced contrast, in targeted standard deviations |
| `properties[survival_recursion_necessity/always_t2__survival]:recursion_displacement` | 3.6840 | survivor-only control's distance, in recursion standard deviations |
| `properties[interval_calibration/static_t1__correctly_specified]:se_ratio` | 0.9880 | horizon-one reported SE over empirical spread |
| `properties[interval_calibration/static_t1__correctly_specified]:coverage` | 0.9455 | horizon-one calibration coverage |
| `properties[interval_calibration/static_t2__correctly_specified]:se_ratio_ci_upper` | 0.9774 | highest horizon-two SE ratio the draws support |
| `properties[interval_calibration/static_t2__correctly_specified]:coverage_ci_upper` | 0.9407 | highest horizon-two coverage the draws support |
| `properties[interval_calibration/static_t2__correctly_specified]:replicates` | 9600 | replications the horizon-two calibration cell required |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper bound |
| `margin:excluded_slope` | -0.2500 | rate the interval must exclude |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |
| `margin:recursion_displacement` | 0.2500 | least the survivor-only control must move the estimate |

## Limitations

| limitation | what it means for use |
| --- | --- |
| This is the ordinary row | Nuisances are fitted on the analysis sample. The separate cross-fitted row below validates held-out nuisance fitting and fold-specific recursion |
| Inference is pointwise | The study reports each horizon separately. It does not validate a simultaneous confidence band for the whole curve. Each horizon is also targeted in its own backward pass, so the reported curve is not constrained to increase |
| Horizon-two inference is mildly anticonservative at n = 2,000 | The reported standard error sits a few percent below the sampling spread, and coverage sits below nominal. Both endpoints stay inside the declared calibration bands. The first horizon does not show this on the same draws |
| The event process has two horizons | Both backward prefixes are exercised. Longer follow-up may compound the horizon-two shortfall above, and this study cannot say by how much |
| The first-horizon null is not a longitudinal null | No time-varying node precedes the first event node, so a baseline-only standardisation recovers that null exactly. The first-horizon type-I cells test baseline and censoring adjustment only |
| `initial_estimate` measures the final fluctuation only | The earlier node's regression is of the *already targeted* later node. R's `fit$Q[[1]]` regresses the updated `Q.kplus1` and `cleverly`'s first step does the same. So `max_targeting_displacement` and `median_targeting_displacement` measure the last fluctuation rather than the whole targeting step. They are not the quantity `margin:targeting_displacement` bounds, which is the targeted-against-unfluctuated distance in the property study |
| The paired study fixes the mechanisms | It isolates survival recursion, sequential regression, targeting, and inference. It does not claim parity for learned treatment or censoring models |
| Positivity is comfortable | No primary cell validates active truncation or near-positivity behavior |
| The size ladder starts at n = 1,000 | Every other registered study starts at 500. An absorbing event thins the risk set, so at 500 some samples of the property law leave the horizon-two parameter unestimable and the estimator refuses. The rate cells therefore say nothing below 1,000 |
| Failure is absorbing and has one cause | Competing-risk cumulative incidence remains a different parameter with no R parity claim |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. The event process must also be coded so a
failure remains in the cumulative risk at later horizons.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/manifest.json)
records the pinned R package, source commit, image digest, formulas, seeds, and artifact hashes.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/replicates.csv.gz),
[paired verdicts](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/properties.csv)
carry every published row.
