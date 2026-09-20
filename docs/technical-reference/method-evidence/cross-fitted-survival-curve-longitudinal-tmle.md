# Cross-fitted survival-curve longitudinal TMLE

This study validates five-fold cumulative-risk estimation under absorbing failure. It covers two
horizons, monotone censoring, static plans, and a dynamic plan. Each fold fits and targets its
complete backward recursion on training rows before it evaluates held-out rows.

The canonical comparison uses R [`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit
`f04a2b4`, on the same panels and the same stored fold assignment, with one fitted prefix per
reported horizon. R `ltmle` has no cross-fitting, so it cannot witness this construction.

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_survival.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_survival.py).

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets and folds | 1,600 censored survival panels, each with one exact five-fold assignment | the identical rows and the identical assignment |
| horizons | cumulative risk at times one and two | one fitted prefix per horizon |
| plans | never treat, always treat, and continue after initial treatment when L2 is positive | the same three, as shifted treatment columns |
| contrasts | always-minus-never at both horizons and dynamic-minus-never at time two | the same, from rowwise influence-curve differences |
| treatment and censoring mechanisms | the generating probabilities from the law | the same probabilities, supplied as exact per-node density ratios |
| sequential regressions | quasibinomial GLMs within each outer training set | `SL.glm`, within the same training sets |
| intervals | pointwise 95% identity-scale Wald intervals | the same, after the horizon-two transformation below |

Both implementations receive the mechanism from the law rather than estimating it. `lmtp` has no
`gform` argument, so the adapter substitutes exact per-node density ratios into it, the same way
it substitutes the fold assignment. The substitution is checked on every run against `lmtp`'s own
estimate: the zero pattern must agree cell for cell, and the cumulative ratio must track it.

That choice is what makes the comparison a comparison. An earlier version let each side estimate
the mechanism its own way, and the result measured two unrelated pipelines. `lmtp` fits its ratio
with `SL.glm`, whose linear logit cannot represent the exact classifier log-odds `-log g` for a
deterministic regime, so the ratio came out shrunken. A shrunken clever covariate under-targets
and understates the influence curve: coverage ran from 0.75 to 0.91 and the SE ratio from 0.60 to
0.86, against 0.949 to 0.957 and 1.00 to 1.02 for `cleverly` on the same panels. The sequential
regressions stay misspecified on both sides, so targeting still has work to do.

The retained R runner uses the one-node binary mean at horizon one because `lmtp` requires two
event nodes for its survival path. At horizon two, it converts event-free survival to cumulative
risk and reverses the influence-curve sign. The registered run invokes that runner, so the
transformation is exercised on every regeneration rather than described.

The exact-law structural test repeats every support point in each fold. It checks every unique risk
and contrast against the finite functional and Gateaux EIF. Separate mutations check held-out
outcome predictions and the first-horizon risk-set boundary.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` cross-fitted survival LTMLE | -0.000325 to 0.0022 | 0.9506 | 0.9854 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.000333 to 0.0022 | 0.9519 | 0.9847 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.0013 to 0.0024 | 0.9463 | 0.9966 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0013 to 0.0025 | 0.9444 | 0.9953 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.0012 to 0.0026 | 0.9475 | 0.9916 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0011 to 0.0027 | 0.9437 | 0.9892 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | `cleverly` cross-fitted survival LTMLE | -0.000529 to 0.0010 | 0.9531 | 0.9998 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.000505 to 0.0010 | 0.9544 | 0.9981 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.000915 to 0.0014 | 0.9513 | 0.9908 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.000918 to 0.0014 | 0.9506 | 0.9908 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | `cleverly` cross-fitted survival LTMLE | -0.0017 to 0.000290 | 0.9463 | 1.0001 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.0016 to 0.000321 | 0.9475 | 1.0004 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.0018 to 0.0012 | 0.9456 | 1.0131 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.0018 to 0.0011 | 0.9525 | 1.0122 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.000790 to 0.0016 | 0.9500 | 0.9876 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | R `lmtp` | -0.000782 to 0.0016 | 0.9463 | 0.9866 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 0.000007 | 0.0024 | 1.0010 | -0.0012 | 0.0025 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -0.000086 | 0.0199 | 1.0057 | 0.0019 | 0.0063 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | -0.000097 | 0.0217 | 1.0048 | 0.0037 | 0.0086 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | -0.000024 | 0.0136 | 1.0005 | -0.0012 | 0.0051 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | 0.000003 | 0.000973 | 1.0034 | 0.000625 | 0.0032 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | -0.000032 | 0.0139 | 1.0028 | -0.0012 | 0.0022 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | 0.000088 | 0.0259 | 1.0086 | -0.0069 | 0.0051 vs 0.0500 | equivalent |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | -0.000008 | 0.0029 | 1.0042 | 0.0037 | 0.0055 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_survival_ltmle` | positive | five-fold horizon-two survival LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.0810 to 1.1254 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3963 to 0.4128 | pass |
| `double_robustness` | `dynamic_t2__both_correct` | positive | dynamic plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0063 to 0.000275, margin 0.0110, SE ratio 1.0003 | pass |
| `double_robustness` | `dynamic_t2__both_wrong` | control | dynamic plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0454 to -0.0386, margin 0.0114, SE ratio 0.7429 | pass |
| `double_robustness` | `dynamic_t2__mechanism_correct` | positive | dynamic plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0031 to 0.0035, margin 0.0112, SE ratio 1.0092 | pass |
| `double_robustness` | `dynamic_t2__outcome_correct` | positive | dynamic plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0037 to 0.0027, margin 0.0107, SE ratio 0.7587 | pass |
| `double_robustness` | `static_t1__both_correct` | positive | static plan at horizon one: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0017, margin 0.0069, SE ratio 1.0067 | pass |
| `double_robustness` | `static_t1__both_wrong` | control | static plan at horizon one: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0451 to -0.0411, margin 0.0068, SE ratio 0.9486 | pass |
| `double_robustness` | `static_t1__mechanism_correct` | positive | static plan at horizon one: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0025 to 0.0018, margin 0.0072, SE ratio 0.9958 | pass |
| `double_robustness` | `static_t1__outcome_correct` | positive | static plan at horizon one: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0018, margin 0.0071, SE ratio 0.8812 | pass |
| `double_robustness` | `static_t2__both_correct` | positive | static plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0055 to 0.0011, margin 0.0112, SE ratio 1.0198 | pass |
| `double_robustness` | `static_t2__both_wrong` | control | static plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0378 to -0.0308, margin 0.0117, SE ratio 0.6717 | pass |
| `double_robustness` | `static_t2__mechanism_correct` | positive | static plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0023 to 0.0045, margin 0.0114, SE ratio 1.0109 | pass |
| `double_robustness` | `static_t2__outcome_correct` | positive | static plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0044 to 0.0022, margin 0.0111, SE ratio 0.6893 | pass |
| `interval_calibration` | `dynamic_t2__correctly_specified` | positive | dynamic plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9412 to 0.9531, SE ratio 0.9910 to 1.0299, empirical efficiency ratio 0.9961 to 1.0343, reported efficiency ratio 1.0236 to 1.0273 | pass |
| `interval_calibration` | `dynamic_t2__noise_control` | control | dynamic plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8281 to 0.8476, SE ratio 0.7024 to 0.7292, empirical efficiency ratio 1.4062 to 1.4596, reported efficiency ratio 1.0236 to 1.0273 | pass |
| `interval_calibration` | `dynamic_t2__shrunken_se_control` | control | dynamic plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8248 to 0.8444, SE ratio 0.6935 to 0.7211, empirical efficiency ratio 0.9955 to 1.0348, reported efficiency ratio 0.7165 to 0.7191 | pass |
| `interval_calibration` | `static_t1__correctly_specified` | positive | static plan at horizon one: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9440 to 0.9556, SE ratio 0.9827 to 1.0196, empirical efficiency ratio 0.9828 to 1.0195, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `static_t1__noise_control` | control | static plan at horizon one: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8220 to 0.8418, SE ratio 0.6930 to 0.7196, empirical efficiency ratio 1.3921 to 1.4456, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `static_t1__shrunken_se_control` | control | static plan at horizon one: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8183 to 0.8382, SE ratio 0.6882 to 0.7137, empirical efficiency ratio 0.9826 to 1.0190, reported efficiency ratio 0.7010 to 0.7016 | pass |
| `interval_calibration` | `static_t2__correctly_specified` | positive | static plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9422 to 0.9540, SE ratio 0.9968 to 1.0356, empirical efficiency ratio 1.0004 to 1.0387, reported efficiency ratio 1.0339 to 1.0378 | pass |
| `interval_calibration` | `static_t2__noise_control` | control | static plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8361 to 0.8552, SE ratio 0.7121 to 0.7395, empirical efficiency ratio 1.4012 to 1.4546, reported efficiency ratio 1.0338 to 1.0379 | pass |
| `interval_calibration` | `static_t2__shrunken_se_control` | control | static plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8268 to 0.8464, SE ratio 0.6975 to 0.7252, empirical efficiency ratio 1.0001 to 1.0390, reported efficiency ratio 0.7237 to 0.7265 | pass |
| `power` | `dynamic_t2__alternative` | positive | dynamic plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t1__alternative` | positive | static plan at horizon one: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t2__alternative` | positive | static plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_1000` | control | dynamic plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0011, coverage 0.9179 to 0.9616, SE ratio 1.0046 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_2000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000077, coverage 0.9326 to 0.9717, SE ratio 1.0242 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_8000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000239, coverage 0.9267 to 0.9677, SE ratio 1.0034 | pass |
| `root_n_and_efficiency` | `static_t1__n_1000` | control | static plan at horizon one: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000902, coverage 0.9386 to 0.9757, SE ratio 1.0192 | pass |
| `root_n_and_efficiency` | `static_t1__n_2000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000160, coverage 0.9431 to 0.9786, SE ratio 1.0255 | pass |
| `root_n_and_efficiency` | `static_t1__n_8000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000037, coverage 0.9267 to 0.9677, SE ratio 0.9787 | pass |
| `root_n_and_efficiency` | `static_t2__n_1000` | control | static plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0017, coverage 0.9223 to 0.9647, SE ratio 1.0319 | pass |
| `root_n_and_efficiency` | `static_t2__n_2000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012, coverage 0.9371 to 0.9747, SE ratio 1.0474 | pass |
| `root_n_and_efficiency` | `static_t2__n_8000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000211, coverage 0.9311 to 0.9708, SE ratio 1.0166 | pass |
| `root_n_rate` | `dynamic_t2__empirical_sd` | positive | dynamic plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5636 to -0.4775 | pass |
| `root_n_rate` | `dynamic_t2__reported_se` | positive | dynamic plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5280 to -0.5192 | pass |
| `root_n_rate` | `static_t1__empirical_sd` | positive | static plan at horizon one: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5244 to -0.4346 | pass |
| `root_n_rate` | `static_t1__reported_se` | positive | static plan at horizon one: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5023 to -0.5003 | pass |
| `root_n_rate` | `static_t2__empirical_sd` | positive | static plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5625 to -0.4783 | pass |
| `root_n_rate` | `static_t2__reported_se` | positive | static plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5344 to -0.5251 | pass |
| `survival_recursion_necessity` | `always_t2__survival` | positive | always-treat risk at horizon two: the survival estimator keeps failures in their event node and removes them afterward | bias interval inside the equivalence margin | bias -0.0026 to 0.0035, margin 0.0103 | pass |
| `survival_recursion_necessity` | `always_t2__survivor_only` | control | always-treat risk at horizon two: the same horizon-two outcome analyzed only among first-node survivors | bias interval must fall entirely outside the margin | bias -0.1510 to -0.1413, margin 0.0163 | pass |
| `targeting_necessity` | `dynamic_t2__targeted` | positive | dynamic plan at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0045 to 0.0021, margin 0.0111 | pass |
| `targeting_necessity` | `dynamic_t2__untargeted` | control | dynamic plan at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0456 to -0.0391, margin 0.0109 | pass |
| `targeting_necessity` | `static_t1__targeted` | positive | static plan at horizon one: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0035 to 0.0010, margin 0.0075 | pass |
| `targeting_necessity` | `static_t1__untargeted` | control | static plan at horizon one: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0461 to -0.0417, margin 0.0073 | pass |
| `targeting_necessity` | `static_t2__targeted` | positive | static plan at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0038 to 0.0030, margin 0.0116 | pass |
| `targeting_necessity` | `static_t2__untargeted` | control | static plan at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0372 to -0.0303, margin 0.0115 | pass |
| `type_i_error` | `dynamic_t2__sharp_null` | positive | dynamic plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0587, 0.0394 to 0.0835 | pass |
| `type_i_error` | `static_t1__sharp_null` | positive | static plan at horizon one: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0575, 0.0384 to 0.0821 | pass |
| `type_i_error` | `static_t2__sharp_null` | positive | static plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0575, 0.0384 to 0.0821 | pass |
<!-- /generated -->

The property study preserves every ordinary survival instrument. These include horizon-specific
double robustness, targeting, root-n, efficiency, calibration, null, power, and survivor-only
recursion controls.

The overfitting pair uses a fully grown outcome tree at horizon two. Both arms use identical
nonlinear survival panels and known treatment and censoring mechanisms. The joint verdict requires
held-out fitting to restore SE scale and improve coverage over the in-sample control.

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
| `property_cells_total` | 52 | independent property cells |
| `property_cells_passed` | 52 | property cells passing |
| `max_standardized_bias` | 0.0478 | largest primary standardized bias |
| `min_coverage` | 0.9437 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9272 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9395 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0613 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_survival_ltmle]:coverage` | 0.9673 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5693 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_survival_ltmle]:coverage_gain_ci_lower` | 0.3842 | lower bound for the paired coverage gain |
| `properties[crossfit_overfitting/cross_fitted_survival_ltmle]:replicates` | 8000 | paired overfitting replications |
| `properties[survival_recursion_necessity/always_t2__survival]:recursion_displacement` | 3.5610 | survivor-only control displacement |
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
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |
| `margin:recursion_displacement` | 0.2500 | least the survivor-only control must move the estimate |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Agreement with `lmtp` is distributional, not numerical | The paired claim is similarity of means and non-inferiority. Per-replication estimates differ at statistical scale, because only the mechanism is shared and the sequential regressions are still fitted differently |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| Horizon one uses the binary-mean path in R | `lmtp` requires two event nodes for its survival path. The one-node binary mean is the same first-horizon cumulative-risk parameter |
| Inference is pointwise | The row does not validate a simultaneous curve band or enforce monotone reported estimates |
| The event process has two horizons and one cause | Longer follow-up and competing-risk cumulative incidence need separate evidence |
| Flexible learning is an independent property instrument | The paired comparison uses one GLM learner on each side. The tree pair does not establish parity for learner-library selection |
| The primary mechanism bounds are nonbinding | No primary cell validates active bounds, weights, clustering, or severe practical-positivity violations |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, conditionally independent censoring, and correct absorbing-event coding.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_survival/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_survival/manifest.json)
records the seeds, the configuration, the pinned `lmtp` version and source commit, the digest of
every study module and reference source, and the artifact hashes.
