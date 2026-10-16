# Repeated-outcome longitudinal IPSI

This study validates `IPSI` with `outcome=Repeated(["Y1", "Y2", "Y3"])`. A repeated outcome is
measured at every node and is not absorbing. The parameter is the mean $E[Y_h^{q}]$ at each
horizon $h$ under each odds tilt $\delta$ of the treatment.

Each horizon is the end-of-study fit
of the panel truncated after node $h$. The functional is Theorem 3.1 of Kim, Kennedy and Naimi
(2021) at $t = h$. The influence function is their Theorem 4.1 on the prefix. The section
[repeated outcomes](../longitudinal-tmle.md#repeated-outcomes) states the estimator and its
exact-law evidence.

The evidence is Python property evidence. No canonical implementation is compared, because no
maintained implementation fits this parameter. The `npcausal` package reads an end-of-study
outcome, has no dropout and pools its treatment fit over `(id, time)`. A dropout-free pairing at
each horizon would repeat the pairing of the
[incremental interventions study](longitudinal-incremental-interventions.md). The dropout claim
rests on the exact witnesses `E5`, `E6` and the mutation `N9` in
`tests/unit/test_influence_gateaux_repeated.py`.

## What was fitted

| setting | value |
| --- | --- |
| datasets | 2,400 censored panels of 8,000 rows per scenario from `make_longitudinal_repeated`, with three nodes |
| fit | one `IPSI` fit with `outcome=Repeated(["Y1", "Y2", "Y3"])` |
| scenarios | `repeated_folds_5` (Algorithm 1 of Kim et al.: five folds) and `repeated_folds_1` (in sample) |
| plans | the odds tilts $\delta$ = 0.5, 1 (the natural course, the reference) and 2 |
| reported parameters | the mean of each plan at $t = 1, 2, 3$, and the contrasts of the two other plans against the natural course. That is nine means and six contrasts |
| outcome regressions | cell means with a backoff (`CellMeansBackoff`) |
| mechanisms | main-terms quasibinomial GLMs for treatment and censoring, correctly specified on this law. The study declares no mechanism |
| intervals | pointwise 95% Wald from the influence curve |
| initial estimate | not published. `IPSI` fits no fluctuation, and its one-step estimate is the only estimate it reports |

Each truth is the forward composition of the law's conditionals under the tilt
(`RepeatedOutcomeLaw.tilt_counterfactual` in `tests/studies/ltmle_repeated_law.py`). A fast test
shows that the observed-data functional of Theorem 3.1 gives the same numbers to 1e-14.

With correct mechanisms, Kim's fourth condition holds for any consistent outcome regression. The
sample size follows from that. The mechanisms are parametric, so the remainder
$(\hat\pi - \pi)^2$ is of order $p / n$ with $p$ near 6. The primary pilot set $n$ = 8,000 and
2,400 replications without a doubling of $n$.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ate_regimen[odds x0.5 vs natural course @ t=1]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` | -0.000089 to 0.000075 | 0.9404 | 0.9866 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ate_regimen[odds x0.5 vs natural course @ t=2]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` | -0.000100 to 0.000138 | 0.9421 | 0.9764 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ate_regimen[odds x0.5 vs natural course @ t=3]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` | -0.000216 to 0.000100 | 0.9367 | 0.9805 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ate_regimen[odds x2 vs natural course @ t=1]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` | -0.000066 to 0.000077 | 0.9413 | 0.9869 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ate_regimen[odds x2 vs natural course @ t=2]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` | -0.000099 to 0.000122 | 0.9358 | 0.9797 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ate_regimen[odds x2 vs natural course @ t=3]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` | -0.000110 to 0.000185 | 0.9504 | 0.9952 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[natural course @ t=1]` | mean of the repeated outcome under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | `cleverly` | -0.000289 to 0.000165 | 0.9492 | 0.9871 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[natural course @ t=2]` | mean of the repeated outcome under the plan leave the observed treatment mechanism unchanged at horizon t = 2 | `cleverly` | -0.000274 to 0.000212 | 0.9537 | 1.0212 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[natural course @ t=3]` | mean of the repeated outcome under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | `cleverly` | -0.000347 to 0.000221 | 0.9492 | 0.9836 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[odds x0.5 @ t=1]` | mean of the repeated outcome under the plan multiply the treatment odds by 0.5 at horizon t = 1 | `cleverly` | -0.000327 to 0.000189 | 0.9479 | 0.9836 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[odds x0.5 @ t=2]` | mean of the repeated outcome under the plan multiply the treatment odds by 0.5 at horizon t = 2 | `cleverly` | -0.000300 to 0.000277 | 0.9546 | 1.0144 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[odds x0.5 @ t=3]` | mean of the repeated outcome under the plan multiply the treatment odds by 0.5 at horizon t = 3 | `cleverly` | -0.000462 to 0.000221 | 0.9471 | 0.9834 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[odds x2 @ t=1]` | mean of the repeated outcome under the plan multiply the treatment odds by two at horizon t = 1 | `cleverly` | -0.000278 to 0.000165 | 0.9542 | 0.9917 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[odds x2 @ t=2]` | mean of the repeated outcome under the plan multiply the treatment odds by two at horizon t = 2 | `cleverly` | -0.000263 to 0.000225 | 0.9533 | 1.0105 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, in sample | `ey_regimen[odds x2 @ t=3]` | mean of the repeated outcome under the plan multiply the treatment odds by two at horizon t = 3 | `cleverly` | -0.000323 to 0.000273 | 0.9558 | 0.9823 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ate_regimen[odds x0.5 vs natural course @ t=1]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` | -0.000045 to 0.000118 | 0.9529 | 0.9968 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ate_regimen[odds x0.5 vs natural course @ t=2]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` | -0.000136 to 0.000102 | 0.9437 | 0.9781 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ate_regimen[odds x0.5 vs natural course @ t=3]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` | -0.000153 to 0.000152 | 0.9558 | 1.0361 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ate_regimen[odds x2 vs natural course @ t=1]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` | -0.000103 to 0.000039 | 0.9517 | 0.9971 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ate_regimen[odds x2 vs natural course @ t=2]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` | -0.000089 to 0.000134 | 0.9479 | 0.9736 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ate_regimen[odds x2 vs natural course @ t=3]` | difference in the mean of the repeated outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` | -0.000113 to 0.000178 | 0.9600 | 1.0220 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[natural course @ t=1]` | mean of the repeated outcome under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | `cleverly` | 0.000039 to 0.000493 | 0.9442 | 0.9882 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[natural course @ t=2]` | mean of the repeated outcome under the plan leave the observed treatment mechanism unchanged at horizon t = 2 | `cleverly` | -0.000375 to 0.000111 | 0.9533 | 1.0213 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[natural course @ t=3]` | mean of the repeated outcome under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | `cleverly` | -0.000077 to 0.000466 | 0.9517 | 1.0344 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[odds x0.5 @ t=1]` | mean of the repeated outcome under the plan multiply the treatment odds by 0.5 at horizon t = 1 | `cleverly` | 0.000048 to 0.000557 | 0.9529 | 0.9974 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[odds x0.5 @ t=2]` | mean of the repeated outcome under the plan multiply the treatment odds by 0.5 at horizon t = 2 | `cleverly` | -0.000436 to 0.000138 | 0.9533 | 1.0185 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[odds x0.5 @ t=3]` | mean of the repeated outcome under the plan multiply the treatment odds by 0.5 at horizon t = 3 | `cleverly` | -0.000133 to 0.000521 | 0.9575 | 1.0369 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[odds x2 @ t=1]` | mean of the repeated outcome under the plan multiply the treatment odds by two at horizon t = 1 | `cleverly` | 0.000009 to 0.000458 | 0.9450 | 0.9799 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[odds x2 @ t=2]` | mean of the repeated outcome under the plan multiply the treatment odds by two at horizon t = 2 | `cleverly` | -0.000354 to 0.000135 | 0.9492 | 1.0090 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring, five folds | `ey_regimen[odds x2 @ t=3]` | mean of the repeated outcome under the plan multiply the treatment odds by two at horizon t = 3 | `cleverly` | -0.000059 to 0.000513 | 0.9529 | 1.0299 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

No canonical implementation is compared. `equivalence.csv` records no row. The accuracy table
above and the theory properties below carry the evidence of this study.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `booster_calibration` | `ipsi_0.5_t3__crossfit_boosted` | diagnostic | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: five folds, a shallow regularized booster for both outcome learners, main-terms quasibinomial mechanisms, n = 8,000 | reported, no verdict | SE ratio 0.9767 to 1.0606 | reported |
| `booster_calibration` | `ipsi_0.5_vs_1.0_t3__crossfit_boosted` | diagnostic | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: five folds, a shallow regularized booster for both outcome learners, main-terms quasibinomial mechanisms, n = 8,000 | reported, no verdict | SE ratio 0.9958 to 1.0822 | reported |
| `booster_calibration` | `ipsi_1.0_t3__crossfit_boosted` | diagnostic | mean under the natural course at horizon t = 3, repeated binary outcome: five folds, a shallow regularized booster for both outcome learners, main-terms quasibinomial mechanisms, n = 8,000 | reported, no verdict | SE ratio 0.9781 to 1.0596 | reported |
| `booster_calibration` | `ipsi_2.0_t3__crossfit_boosted` | diagnostic | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: five folds, a shallow regularized booster for both outcome learners, main-terms quasibinomial mechanisms, n = 8,000 | reported, no verdict | SE ratio 0.9870 to 1.0678 | reported |
| `booster_calibration` | `ipsi_2.0_vs_1.0_t3__crossfit_boosted` | diagnostic | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: five folds, a shallow regularized booster for both outcome learners, main-terms quasibinomial mechanisms, n = 8,000 | reported, no verdict | SE ratio 0.9889 to 1.0719 | reported |
| `interval_calibration` | `ipsi_0.5_t1__in_sample` | positive | mean under the odds tilt 0.5 at horizon t = 1, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9439 to 0.9575, SE ratio 0.9815 to 1.0255 | pass |
| `interval_calibration` | `ipsi_0.5_t1__noise_control` | control | mean under the odds tilt 0.5 at horizon t = 1, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8201 to 0.8436, SE ratio 0.6896 to 0.7207 | pass |
| `interval_calibration` | `ipsi_0.5_t1__shrunken_se_control` | control | mean under the odds tilt 0.5 at horizon t = 1, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8165 to 0.8401, SE ratio 0.6870 to 0.7176 | pass |
| `interval_calibration` | `ipsi_0.5_t2__in_sample` | positive | mean under the odds tilt 0.5 at horizon t = 2, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9420 to 0.9559, SE ratio 0.9727 to 1.0171 | pass |
| `interval_calibration` | `ipsi_0.5_t2__noise_control` | control | mean under the odds tilt 0.5 at horizon t = 2, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8177 to 0.8413, SE ratio 0.6835 to 0.7147 | pass |
| `interval_calibration` | `ipsi_0.5_t2__shrunken_se_control` | control | mean under the odds tilt 0.5 at horizon t = 2, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8202 to 0.8437, SE ratio 0.6805 to 0.7119 | pass |
| `interval_calibration` | `ipsi_0.5_t3__continuous` | positive | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the three-value outcome law, in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial mechanisms, outcome bounds (0, 1), n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9408 to 0.9548, SE ratio 0.9708 to 1.0150 | pass |
| `interval_calibration` | `ipsi_0.5_t3__crossfit` | positive | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: five folds, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9423 to 0.9562, SE ratio 0.9784 to 1.0232 | pass |
| `interval_calibration` | `ipsi_0.5_t3__in_sample` | positive | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9430 to 0.9567, SE ratio 0.9742 to 1.0179 | pass |
| `interval_calibration` | `ipsi_0.5_t3__noise_control` | control | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8183 to 0.8419, SE ratio 0.6894 to 0.7209 | pass |
| `interval_calibration` | `ipsi_0.5_t3__shrunken_se_control` | control | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8157 to 0.8394, SE ratio 0.6822 to 0.7118 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t1__in_sample` | positive | contrast of the odds tilt 0.5 against the natural course at horizon t = 1, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9480 to 0.9611, SE ratio 0.9966 to 1.0424 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t1__noise_control` | control | contrast of the odds tilt 0.5 against the natural course at horizon t = 1, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8270 to 0.8501, SE ratio 0.7011 to 0.7323 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t1__shrunken_se_control` | control | contrast of the odds tilt 0.5 against the natural course at horizon t = 1, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8269 to 0.8500, SE ratio 0.6976 to 0.7302 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t2__in_sample` | positive | contrast of the odds tilt 0.5 against the natural course at horizon t = 2, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9380 to 0.9523, SE ratio 0.9625 to 1.0050 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t2__noise_control` | control | contrast of the odds tilt 0.5 against the natural course at horizon t = 2, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8198 to 0.8433, SE ratio 0.6950 to 0.7265 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t2__shrunken_se_control` | control | contrast of the odds tilt 0.5 against the natural course at horizon t = 2, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8103 to 0.8343, SE ratio 0.6741 to 0.7038 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t3__continuous` | positive | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: the three-value outcome law, in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial mechanisms, outcome bounds (0, 1), n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9353 to 0.9499, SE ratio 0.9533 to 0.9958 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t3__crossfit` | positive | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: five folds, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9425 to 0.9563, SE ratio 0.9804 to 1.0237 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t3__in_sample` | positive | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9430 to 0.9567, SE ratio 0.9720 to 1.0155 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t3__noise_control` | control | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8219 to 0.8453, SE ratio 0.6883 to 0.7192 | pass |
| `interval_calibration` | `ipsi_0.5_vs_1.0_t3__shrunken_se_control` | control | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8171 to 0.8407, SE ratio 0.6801 to 0.7113 | pass |
| `interval_calibration` | `ipsi_1.0_t1__in_sample` | positive | mean under the natural course at horizon t = 1, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9442 to 0.9578, SE ratio 0.9801 to 1.0240 | pass |
| `interval_calibration` | `ipsi_1.0_t1__noise_control` | control | mean under the natural course at horizon t = 1, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8219 to 0.8453, SE ratio 0.6926 to 0.7240 | pass |
| `interval_calibration` | `ipsi_1.0_t1__shrunken_se_control` | control | mean under the natural course at horizon t = 1, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8180 to 0.8416, SE ratio 0.6866 to 0.7163 | pass |
| `interval_calibration` | `ipsi_1.0_t2__in_sample` | positive | mean under the natural course at horizon t = 2, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9411 to 0.9551, SE ratio 0.9741 to 1.0177 | pass |
| `interval_calibration` | `ipsi_1.0_t2__noise_control` | control | mean under the natural course at horizon t = 2, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8246 to 0.8479, SE ratio 0.6973 to 0.7285 | pass |
| `interval_calibration` | `ipsi_1.0_t2__shrunken_se_control` | control | mean under the natural course at horizon t = 2, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8178 to 0.8414, SE ratio 0.6816 to 0.7122 | pass |
| `interval_calibration` | `ipsi_1.0_t3__continuous` | positive | mean under the natural course at horizon t = 3, repeated binary outcome: the three-value outcome law, in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial mechanisms, outcome bounds (0, 1), n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9408 to 0.9548, SE ratio 0.9815 to 1.0263 | pass |
| `interval_calibration` | `ipsi_1.0_t3__crossfit` | positive | mean under the natural course at horizon t = 3, repeated binary outcome: five folds, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9440 to 0.9577, SE ratio 0.9834 to 1.0271 | pass |
| `interval_calibration` | `ipsi_1.0_t3__in_sample` | positive | mean under the natural course at horizon t = 3, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9423 to 0.9562, SE ratio 0.9765 to 1.0197 | pass |
| `interval_calibration` | `ipsi_1.0_t3__noise_control` | control | mean under the natural course at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8310 to 0.8539, SE ratio 0.6994 to 0.7311 | pass |
| `interval_calibration` | `ipsi_1.0_t3__shrunken_se_control` | control | mean under the natural course at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8137 to 0.8376, SE ratio 0.6837 to 0.7144 | pass |
| `interval_calibration` | `ipsi_2.0_t1__in_sample` | positive | mean under the odds tilt 2 at horizon t = 1, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9445 to 0.9581, SE ratio 0.9817 to 1.0254 | pass |
| `interval_calibration` | `ipsi_2.0_t1__noise_control` | control | mean under the odds tilt 2 at horizon t = 1, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8222 to 0.8456, SE ratio 0.6927 to 0.7235 | pass |
| `interval_calibration` | `ipsi_2.0_t1__shrunken_se_control` | control | mean under the odds tilt 2 at horizon t = 1, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8187 to 0.8423, SE ratio 0.6873 to 0.7178 | pass |
| `interval_calibration` | `ipsi_2.0_t2__in_sample` | positive | mean under the odds tilt 2 at horizon t = 2, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9392 to 0.9534, SE ratio 0.9700 to 1.0119 | pass |
| `interval_calibration` | `ipsi_2.0_t2__noise_control` | control | mean under the odds tilt 2 at horizon t = 2, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8181 to 0.8417, SE ratio 0.6854 to 0.7159 | pass |
| `interval_calibration` | `ipsi_2.0_t2__shrunken_se_control` | control | mean under the odds tilt 2 at horizon t = 2, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8112 to 0.8351, SE ratio 0.6788 to 0.7093 | pass |
| `interval_calibration` | `ipsi_2.0_t3__continuous` | positive | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the three-value outcome law, in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial mechanisms, outcome bounds (0, 1), n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9439 to 0.9575, SE ratio 0.9760 to 1.0192 | pass |
| `interval_calibration` | `ipsi_2.0_t3__crossfit` | positive | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: five folds, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9453 to 0.9588, SE ratio 0.9889 to 1.0324 | pass |
| `interval_calibration` | `ipsi_2.0_t3__in_sample` | positive | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9423 to 0.9562, SE ratio 0.9716 to 1.0134 | pass |
| `interval_calibration` | `ipsi_2.0_t3__noise_control` | control | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8193 to 0.8429, SE ratio 0.6821 to 0.7129 | pass |
| `interval_calibration` | `ipsi_2.0_t3__shrunken_se_control` | control | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8125 to 0.8364, SE ratio 0.6795 to 0.7097 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t1__in_sample` | positive | contrast of the odds tilt 2.0 against the natural course at horizon t = 1, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9473 to 0.9605, SE ratio 0.9984 to 1.0440 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t1__noise_control` | control | contrast of the odds tilt 2.0 against the natural course at horizon t = 1, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8334 to 0.8561, SE ratio 0.7013 to 0.7326 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t1__shrunken_se_control` | control | contrast of the odds tilt 2.0 against the natural course at horizon t = 1, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8310 to 0.8539, SE ratio 0.6985 to 0.7309 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t2__in_sample` | positive | contrast of the odds tilt 2.0 against the natural course at horizon t = 2, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9456 to 0.9590, SE ratio 0.9788 to 1.0223 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t2__noise_control` | control | contrast of the odds tilt 2.0 against the natural course at horizon t = 2, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8216 to 0.8450, SE ratio 0.6844 to 0.7148 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t2__shrunken_se_control` | control | contrast of the odds tilt 2.0 against the natural course at horizon t = 2, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8160 to 0.8397, SE ratio 0.6849 to 0.7159 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t3__continuous` | positive | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: the three-value outcome law, in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial mechanisms, outcome bounds (0, 1), n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9412 to 0.9552, SE ratio 0.9664 to 1.0099 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t3__crossfit` | positive | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: five folds, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9420 to 0.9559, SE ratio 0.9778 to 1.0228 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t3__in_sample` | positive | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial treatment and censoring mechanisms, n = 8,000 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9466 to 0.9599, SE ratio 0.9759 to 1.0201 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t3__noise_control` | control | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8211 to 0.8446, SE ratio 0.6837 to 0.7148 | pass |
| `interval_calibration` | `ipsi_2.0_vs_1.0_t3__shrunken_se_control` | control | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8149 to 0.8387, SE ratio 0.6832 to 0.7139 | pass |
| `outcome_history_necessity` | `ipsi_0.5_t3__history_free` | control | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: an end-of-study fit with no earlier outcome in any history, cell-mean outcome regressions and mechanisms on the coarse key | bias interval outside the margin, and inside it around the coarse tilt functional | bias -0.0122 to -0.0110, margin 0.0016, bias around its limit -0.000702 to 0.000501, margin 0.0016 | pass |
| `outcome_history_necessity` | `ipsi_0.5_t3__repeated` | positive | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000655 to 0.000552, margin 0.0017 | pass |
| `outcome_history_necessity` | `ipsi_2.0_t3__history_free` | control | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: an end-of-study fit with no earlier outcome in any history, cell-mean outcome regressions and mechanisms on the coarse key | bias interval outside the margin, and inside it around the coarse tilt functional | bias 0.0045 to 0.0056, margin 0.0014, bias around its limit -0.000420 to 0.000627, margin 0.0014 | pass |
| `outcome_history_necessity` | `ipsi_2.0_t3__repeated` | positive | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000464 to 0.000557, margin 0.0014 | pass |
| `repeated_outcome_necessity` | `ipsi_0.5_t2__absorbing_reading` | control | mean under the odds tilt 0.5 at horizon t = 2, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, with the earlier outcomes in the history | bias interval outside the margin, and inside it around the absorbing tilt functional | bias 0.1259 to 0.1269, margin 0.0015, bias around its limit -0.000680 to 0.000330, margin 0.0015 | pass |
| `repeated_outcome_necessity` | `ipsi_0.5_t2__repeated` | positive | mean under the odds tilt 0.5 at horizon t = 2, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000668 to 0.000250, margin 0.0014 | pass |
| `repeated_outcome_necessity` | `ipsi_0.5_t3__absorbing_reading` | control | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, with the earlier outcomes in the history | bias interval outside the margin, and inside it around the absorbing tilt functional | bias 0.1932 to 0.1943, margin 0.0017, bias around its limit -0.000644 to 0.000457, margin 0.0017 | pass |
| `repeated_outcome_necessity` | `ipsi_0.5_t3__repeated` | positive | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000693 to 0.000369, margin 0.0016 | pass |
| `repeated_outcome_necessity` | `ipsi_1.0_t2__absorbing_reading` | control | mean under the natural course at horizon t = 2, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, with the earlier outcomes in the history | bias interval outside the margin, and inside it around the absorbing tilt functional | bias 0.1213 to 0.1222, margin 0.0014, bias around its limit -0.000612 to 0.000278, margin 0.0014 | pass |
| `repeated_outcome_necessity` | `ipsi_1.0_t2__repeated` | positive | mean under the natural course at horizon t = 2, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000552 to 0.000211, margin 0.0012 | pass |
| `repeated_outcome_necessity` | `ipsi_1.0_t3__absorbing_reading` | control | mean under the natural course at horizon t = 3, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, with the earlier outcomes in the history | bias interval outside the margin, and inside it around the absorbing tilt functional | bias 0.1916 to 0.1925, margin 0.0015, bias around its limit -0.000540 to 0.000417, margin 0.0015 | pass |
| `repeated_outcome_necessity` | `ipsi_1.0_t3__repeated` | positive | mean under the natural course at horizon t = 3, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000423 to 0.000444, margin 0.0013 | pass |
| `repeated_outcome_necessity` | `ipsi_2.0_t2__absorbing_reading` | control | mean under the odds tilt 2 at horizon t = 2, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, with the earlier outcomes in the history | bias interval outside the margin, and inside it around the absorbing tilt functional | bias 0.1178 to 0.1187, margin 0.0014, bias around its limit -0.000620 to 0.000284, margin 0.0014 | pass |
| `repeated_outcome_necessity` | `ipsi_2.0_t2__repeated` | positive | mean under the odds tilt 2 at horizon t = 2, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000525 to 0.000221, margin 0.0011 | pass |
| `repeated_outcome_necessity` | `ipsi_2.0_t3__absorbing_reading` | control | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, with the earlier outcomes in the history | bias interval outside the margin, and inside it around the absorbing tilt functional | bias 0.1862 to 0.1872, margin 0.0016, bias around its limit -0.000517 to 0.000505, margin 0.0016 | pass |
| `repeated_outcome_necessity` | `ipsi_2.0_t3__repeated` | positive | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions | bias interval inside the equivalence margin | bias -0.000233 to 0.000659, margin 0.0014 | pass |
| `simultaneous_coverage` | `all_reported__pointwise_joint_control` | control | every parameter the calibration fit reports: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6288 to 0.6589 | pass |
| `simultaneous_coverage` | `all_reported__simultaneous_band` | positive | every parameter the calibration fit reports: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9415 to 0.9555 | pass |
| `single_robustness` | `ipsi_0.5_t3__one_step_necessity` | control | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the tilt plug-in with the same wrong regression and the same fitted treatment mechanism, written without `cleverly` | bias interval outside the margin, and inside it around the plug-in's exact limit | bias -0.0291 to -0.0270, margin 0.0029, coverage 0.4350 | pass |
| `single_robustness` | `ipsi_0.5_t3__w_only_m` | positive | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the outcome regression reads the baseline covariate and the current arm alone, which is wrong for the history, with the correctly specified mechanisms, n = 2,000 | bias interval inside the equivalence margin | bias -0.0012 to 0.0012, margin 0.0033, coverage 0.9600 | pass |
| `single_robustness` | `ipsi_0.5_vs_1.0_t3__one_step_necessity` | control | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: the tilt plug-in with the same wrong regression and the same fitted treatment mechanism, written without `cleverly` | bias interval outside the margin, and inside it around the plug-in's exact limit | bias -0.0263 to -0.0257, margin 0.000847, coverage 0.0000 | pass |
| `single_robustness` | `ipsi_0.5_vs_1.0_t3__w_only_m` | positive | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: the outcome regression reads the baseline covariate and the current arm alone, which is wrong for the history, with the correctly specified mechanisms, n = 2,000 | bias interval inside the equivalence margin | bias -0.000858 to 0.000236, margin 0.0015, coverage 0.9625 | pass |
| `single_robustness` | `ipsi_2.0_t3__one_step_necessity` | control | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the tilt plug-in with the same wrong regression and the same fitted treatment mechanism, written without `cleverly` | bias interval outside the margin, and inside it around the plug-in's exact limit | bias 0.0208 to 0.0228, margin 0.0028, coverage 0.5125 | pass |
| `single_robustness` | `ipsi_2.0_t3__w_only_m` | positive | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the outcome regression reads the baseline covariate and the current arm alone, which is wrong for the history, with the correctly specified mechanisms, n = 2,000 | bias interval inside the equivalence margin | bias -0.000696 to 0.0013, margin 0.0028, coverage 0.9537 | pass |
| `single_robustness` | `ipsi_2.0_vs_1.0_t3__one_step_necessity` | control | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: the tilt plug-in with the same wrong regression and the same fitted treatment mechanism, written without `cleverly` | bias interval outside the margin, and inside it around the plug-in's exact limit | bias 0.0235 to 0.0242, margin 0.000948, coverage 0.0025 | pass |
| `single_robustness` | `ipsi_2.0_vs_1.0_t3__w_only_m` | positive | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: the outcome regression reads the baseline covariate and the current arm alone, which is wrong for the history, with the correctly specified mechanisms, n = 2,000 | bias interval inside the equivalence margin | bias -0.000502 to 0.000530, margin 0.0014, coverage 0.9600 | pass |
| `small_sample_calibration` | `ipsi_0.5_t3__continuous` | diagnostic | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9169 to 0.9920 | reported |
| `small_sample_calibration` | `ipsi_0.5_t3__in_sample` | diagnostic | mean under the odds tilt 0.5 at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9462 to 1.0256 | reported |
| `small_sample_calibration` | `ipsi_0.5_vs_1.0_t3__continuous` | diagnostic | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: the continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.8884 to 0.9612 | reported |
| `small_sample_calibration` | `ipsi_0.5_vs_1.0_t3__in_sample` | diagnostic | contrast of the odds tilt 0.5 against the natural course at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9163 to 0.9898 | reported |
| `small_sample_calibration` | `ipsi_1.0_t3__continuous` | diagnostic | mean under the natural course at horizon t = 3, repeated binary outcome: the continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9454 to 1.0250 | reported |
| `small_sample_calibration` | `ipsi_1.0_t3__in_sample` | diagnostic | mean under the natural course at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9738 to 1.0568 | reported |
| `small_sample_calibration` | `ipsi_2.0_t3__continuous` | diagnostic | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9388 to 1.0187 | reported |
| `small_sample_calibration` | `ipsi_2.0_t3__in_sample` | diagnostic | mean under the odds tilt 2 at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9589 to 1.0406 | reported |
| `small_sample_calibration` | `ipsi_2.0_vs_1.0_t3__continuous` | diagnostic | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: the continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9086 to 0.9863 | reported |
| `small_sample_calibration` | `ipsi_2.0_vs_1.0_t3__in_sample` | diagnostic | contrast of the odds tilt 2.0 against the natural course at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9179 to 0.9956 | reported |
<!-- /generated -->

The property families draw from their own streams of `make_longitudinal_repeated`. The
continuous arm draws the three-value law `CONTINUOUS`. The table above lists every cell. This
section states what each family tests.

| family | what it tests | size |
| --- | --- | --- |
| `interval_calibration` | pointwise coverage and SE ratio inside the calibration bands. The cells are in sample for all 15 parameters, cross-fitted at five folds and continuous for the five parameters at $t = 3$. A shrunken-SE control and a noise control read each in-sample cell | $n$ = 8,000, 6,800 replications |
| `simultaneous_coverage` | the default band over all 15 reported parameters, against its pointwise control | $n$ = 8,000, 6,800 replications |
| `booster_calibration` | the cross-fitted fit with a shallow regularized booster for both outcome learners. The role is `diagnostic`. The table reports these cells and no rule gates them | $n$ = 8,000, 2,000 replications |
| `small_sample_calibration` | the in-sample and continuous arms refitted at $n$ = 2,000. The role is `diagnostic` | $n$ = 2,000, 2,000 replications |
| `single_robustness` | bias inside the margin when the outcome regression reads $W$ and the current arm alone and the mechanisms are correct. The control is the tilt plug-in with the same regression and propensity. It must miss the truth and reach its exact limit | $n$ = 2,000, 800 replications |
| `repeated_outcome_necessity` | an absorbing reading of the outcome, which fits $\max_{s \le h} Y_s$ with the earlier outcomes in the history. The control must miss the truth and reach the absorbing tilt functional | $n$ = 8,000, 1,000 replications |
| `outcome_history_necessity` | a fit that leaves every $Y_s$ out of every history. Every nuisance is the coarse projection. The control must miss the truth and reach the coarse tilt functional | $n$ = 8,000, 800 replications |

There is no double-robustness family. `IPSI` is singly robust. It needs the treatment and
censoring mechanisms. A wrong treatment mechanism with a correct outcome regression leaves a
bias (witness `E7`).

In sample, the cell-means regression reproduces every seen cell. The outcome-residual correction
of the one-step estimator is then zero on seen cells. The in-sample cells test the plug-in and the
mechanism terms. Only the cross-fitted cells and `single_robustness` exercise the residual term
with a nonzero residual. The tests `E5`, `E6`, `N9` and `W1` cover that term exactly.

## Sizing

Every size is a rule of `tests/studies/repeated_sizing.py` applied to a pilot table in
`tests/studies/ipsi_repeated_properties.py`. The pilots draw from streams that no published cell
reads. The pilot tables record the commands in `tests/canonical/ipsi_repeated/README.md`.

| pilot | draws | sets |
| --- | --- | --- |
| P4 primary, two scenarios | 2,000 each | the primary $n$ and the primary replications |
| P5 calibration arms | 8,000 each | the calibration replications |
| P6 every property family | 2,000 each | the size of each family and the $n$ of the two necessity families |

The study fixed every size, replication count, margin, law, learner, fold count and seed before
the run.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications per scenario |
| `n` | 8000 | observations per primary replication |
| `independent_tests_total` | 30 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 30 | of those, passing |
| `property_cells_total` | 81 | repeated-sampling property cells, reported cells excluded |
| `property_cells_passed` | 81 | cells whose own and family verdicts pass |
| `property_cells_reported` | 15 | cells reported and not gated, as the `small_sample_calibration` diagnostics |
| `max_standardized_bias` | 0.0625 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9358 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9393 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0769 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control multiplies the standard error by |
| `margin:necessity_displacement` | 0.2500 | smallest paired displacement of a necessity control, in empirical standard deviations |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 24 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| one law with three nodes and a binary outcome | the continuous outcome has one calibration arm, and the rate in $n$ is not measured |
| the study has no comparator | the claim rests on exact truths, exact-law witnesses and the controls. No other implementation fits this parameter |
| the in-sample interval runs below nominal at $n$ = 2,000 for the continuous arm | the `small_sample_calibration` cells report the shortfall and no rule gates them. The gated calibration cells draw $n$ = 8,000 |
| cell-mean outcome regressions and one booster | the study does not cover a larger learner library or another fold count than five |
| the mechanisms are correctly specified | the study covers no wrong treatment or censoring mechanism. `IPSI` is singly robust, and witness `E7` shows the bias |
| the band covers the declared tilts and horizons | the study does not test a band over values of $\delta$ between the declared ones |
| the absorbing and history-free controls read exact limits of this law | they show that the repeated reading and the earlier outcomes matter here. They do not bound the difference on another law |
| the second-order dependence acts at node 3 only | the law has three nodes, so the depth-two terms act at the last node. No cell reads a longer history |
| no weights, clusters, working model or survival outcome in this study | the [dropout study](longitudinal-incremental-interventions-dropout.md) measures them for the end-of-study and survival outcomes |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ipsi_repeated/README.md)
gives the regeneration and pilot commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ipsi_repeated/manifest.json)
records the seeds, the pinned learners and the artifact hashes.
