# Clustered TMLE intervals on a t reference at few clusters

This study measures the Student $t$ reference that a clustered fit takes below 40 clusters with
positive weight mass. An estimate that reads $J$ such clusters uses $J-2$ degrees of freedom, as
Nugent et al. (2024), Section 2.2, last paragraph, recommend. A fold-evaluated estimate over $V$
folds uses $\min(J-2, J-V)$, because its variance centers the cluster totals inside each fold.
Below 10 clusters a fit reports no interval, and an `LTMLE` fit below 20, because no registered
study measures one there.
[Clusters](../inference.md#clusters) states the rules.

The publication policy is `reporting`. Every cell may read red, and
[F28](../../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns any red cell. The
policy, the laws, the learners and every budget were declared before the run.

## What was compared

The primary rows pair in-sample `LTMLE` on one treatment node with R `ltmle` 1.3-0 at 20
clusters. Each draw reads the informative law of `tests/studies/clustered_unequal_laws.py`, with
sizes uniform on 2 to 18.

| setting | `cleverly` | R `ltmle` 1.3-0 |
| --- | --- | --- |
| construction | in-sample LTMLE, plans `always` and `never` | `ltmle` with `abar = 1` and `abar = 0` |
| treatment mechanism | exact propensity | the identical probabilities as a numeric `gform` |
| outcome regression | quasibinomial GLM in W1 and W2 among each plan's followers | `Q.kplus1 ~ W1 + W2`, `stratify = TRUE` |
| independent unit | cluster sums | `id=`, the household curve |
| reference distribution | $t$ with $J-2$ degrees of freedom | $t$ with $J-1$ below 100 clusters |

The point estimates and the household standard errors are the same quantity. The intervals differ
by a declared convention, so this package's interval is never narrower than `ltmle`'s. The R runner
checks that its household standard error equals `ltmle`'s own to $10^{-10}$.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` | 0.0031 to 0.0242 | 0.9410 | 0.9388 | pass |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `ltmle` | 0.0031 to 0.0242 | 0.9400 | 0.9388 | pass |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` | -0.000256 to 0.0137 | 0.9340 | 0.9537 | pass |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `ltmle` | -0.000256 to 0.0137 | 0.9330 | 0.9537 | pass |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` | -0.0130 to -0.000978 | 0.9320 | 0.9288 | pass |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `ltmle` | -0.0130 to -0.000978 | 0.9310 | 0.9288 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -6.790e-11 | 3.501e-09 | 1.0000 | 0.0010 | 1.276e-10 vs 0.0500 | equivalent |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[always]` | mean outcome under the plan treat at both times | -4.716e-11 | 3.687e-09 | 1.0000 | 0.0010 | 1.021e-10 vs 0.0500 | equivalent |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 2.074e-11 | 1.885e-09 | 1.0000 | 0.0010 | 2.356e-10 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j10__iid_t_control` | control | cross-fitted DR-TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0013 to 0.0105, coverage 0.8874 to 0.9121, SE ratio 0.8499 to 1.0008 | **fail** |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j10__normal_reference` | diagnostic | cross-fitted DR-TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0013 to 0.0105, coverage 0.9104 to 0.9325, SE ratio 1.1094 to 1.2916 | reported |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j10__t_reference` | positive | cross-fitted DR-TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0013 to 0.0105, coverage 0.9434 to 0.9610, SE ratio 1.1112 to 1.2929 | pass |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j20__iid_t_control` | control | cross-fitted DR-TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.000551 to 0.0077, coverage 0.8646 to 0.8915, SE ratio 0.8119 to 0.9042 | **fail** |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j20__normal_reference` | diagnostic | cross-fitted DR-TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.000551 to 0.0077, coverage 0.9332 to 0.9523, SE ratio 1.0873 to 1.1980 | reported |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j20__t_reference` | positive | cross-fitted DR-TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.000551 to 0.0077, coverage 0.9442 to 0.9617, SE ratio 1.0878 to 1.1972 | pass |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j30__iid_t_control` | control | cross-fitted DR-TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0011 to 0.0057, coverage 0.8531 to 0.8810, SE ratio 0.7967 to 0.8764 | **fail** |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j30__normal_reference` | diagnostic | cross-fitted DR-TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0011 to 0.0057, coverage 0.9372 to 0.9557, SE ratio 1.0741 to 1.1688 | reported |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j30__t_reference` | positive | cross-fitted DR-TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0011 to 0.0057, coverage 0.9451 to 0.9623, SE ratio 1.0746 to 1.1695 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j10__iid_t_control` | control | cross-fitted DR-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0242 to 0.0394, coverage 0.8049 to 0.8363, SE ratio 0.6849 to 0.8247 | **fail** |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j10__normal_reference` | diagnostic | cross-fitted DR-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0242 to 0.0394, coverage 0.8874 to 0.9121, SE ratio 0.9764 to 1.1084 | reported |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j10__t_reference` | positive | cross-fitted DR-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0242 to 0.0394, coverage 0.9216 to 0.9423, SE ratio 0.9733 to 1.1117 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j20__iid_t_control` | control | cross-fitted DR-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0123 to 0.0228, coverage 0.7699 to 0.8034, SE ratio 0.6584 to 0.7729 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j20__normal_reference` | diagnostic | cross-fitted DR-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0123 to 0.0228, coverage 0.9206 to 0.9414, SE ratio 1.0137 to 1.1255 | reported |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j20__t_reference` | positive | cross-fitted DR-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0123 to 0.0228, coverage 0.9356 to 0.9543, SE ratio 1.0136 to 1.1279 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j30__iid_t_control` | control | cross-fitted DR-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0044 to 0.0127, coverage 0.7604 to 0.7945, SE ratio 0.6336 to 0.7138 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j30__normal_reference` | diagnostic | cross-fitted DR-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0044 to 0.0127, coverage 0.9278 to 0.9477, SE ratio 1.0138 to 1.1161 | reported |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j30__t_reference` | positive | cross-fitted DR-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0044 to 0.0127, coverage 0.9402 to 0.9582, SE ratio 1.0129 to 1.1196 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j20__iid_t_control` | control | in-sample LTMLE on one node, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0043 to 0.0040, coverage 0.8383 to 0.8674, SE ratio 0.6884 to 0.7292 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j20__normal_reference` | diagnostic | in-sample LTMLE on one node, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0043 to 0.0040, coverage 0.9187 to 0.9398, SE ratio 0.9628 to 1.0222 | reported |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j20__t_reference` | positive | in-sample LTMLE on one node, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0043 to 0.0040, coverage 0.9307 to 0.9502, SE ratio 0.9637 to 1.0216 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j30__iid_t_control` | control | in-sample LTMLE on one node, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0031 to 0.0038, coverage 0.8349 to 0.8642, SE ratio 0.6894 to 0.7304 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j30__normal_reference` | diagnostic | in-sample LTMLE on one node, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0031 to 0.0038, coverage 0.9297 to 0.9493, SE ratio 0.9623 to 1.0235 | reported |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j30__t_reference` | positive | in-sample LTMLE on one node, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0031 to 0.0038, coverage 0.9396 to 0.9578, SE ratio 0.9630 to 1.0226 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j20__iid_t_control` | control | in-sample LTMLE on one node, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0079 to 0.0181, coverage 0.7488 to 0.7835, SE ratio 0.5500 to 0.5835 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j20__normal_reference` | diagnostic | in-sample LTMLE on one node, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0079 to 0.0181, coverage 0.9163 to 0.9377, SE ratio 0.9307 to 0.9911 | reported |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j20__t_reference` | positive | in-sample LTMLE on one node, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0079 to 0.0181, coverage 0.9329 to 0.9520, SE ratio 0.9327 to 0.9925 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j30__iid_t_control` | control | in-sample LTMLE on one node, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0021 to 0.0103, coverage 0.7522 to 0.7867, SE ratio 0.5623 to 0.5961 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j30__normal_reference` | diagnostic | in-sample LTMLE on one node, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0021 to 0.0103, coverage 0.9254 to 0.9456, SE ratio 0.9545 to 1.0141 | reported |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j30__t_reference` | positive | in-sample LTMLE on one node, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0021 to 0.0103, coverage 0.9348 to 0.9536, SE ratio 0.9548 to 1.0131 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j10__iid_t_control` | control | stacked CV-TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0080 to 0.0041, coverage 0.8733 to 0.8993, SE ratio 0.7013 to 0.7431 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j10__normal_reference` | diagnostic | stacked CV-TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0080 to 0.0041, coverage 0.8874 to 0.9121, SE ratio 0.9529 to 1.0141 | reported |
| `few_cluster_reference` | `tmle_crossfit__equal10__j10__t_reference` | positive | stacked CV-TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0080 to 0.0041, coverage 0.9243 to 0.9446, SE ratio 0.9522 to 1.0142 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j20__iid_t_control` | control | stacked CV-TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0042 to 0.0042, coverage 0.8516 to 0.8796, SE ratio 0.7055 to 0.7479 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j20__normal_reference` | diagnostic | stacked CV-TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0042 to 0.0042, coverage 0.9203 to 0.9412, SE ratio 0.9754 to 1.0353 | reported |
| `few_cluster_reference` | `tmle_crossfit__equal10__j20__t_reference` | positive | stacked CV-TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0042 to 0.0042, coverage 0.9324 to 0.9516, SE ratio 0.9738 to 1.0363 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j30__iid_t_control` | control | stacked CV-TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0029 to 0.0039, coverage 0.8414 to 0.8702, SE ratio 0.7001 to 0.7419 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j30__normal_reference` | diagnostic | stacked CV-TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0029 to 0.0039, coverage 0.9329 to 0.9520, SE ratio 0.9725 to 1.0315 | reported |
| `few_cluster_reference` | `tmle_crossfit__equal10__j30__t_reference` | positive | stacked CV-TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0029 to 0.0039, coverage 0.9423 to 0.9601, SE ratio 0.9708 to 1.0312 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j10__iid_t_control` | control | stacked CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0182 to 0.0332, coverage 0.7922 to 0.8245, SE ratio 0.5482 to 0.5841 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j10__normal_reference` | diagnostic | stacked CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0182 to 0.0332, coverage 0.8822 to 0.9074, SE ratio 0.8948 to 0.9579 | reported |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j10__t_reference` | positive | stacked CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0182 to 0.0332, coverage 0.9200 to 0.9409, SE ratio 0.8942 to 0.9574 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j20__iid_t_control` | control | stacked CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0078 to 0.0180, coverage 0.7573 to 0.7916, SE ratio 0.5612 to 0.5966 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j20__normal_reference` | diagnostic | stacked CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0078 to 0.0180, coverage 0.9192 to 0.9402, SE ratio 0.9453 to 1.0059 | reported |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j20__t_reference` | positive | stacked CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0078 to 0.0180, coverage 0.9353 to 0.9541, SE ratio 0.9443 to 1.0053 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j30__iid_t_control` | control | stacked CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0022 to 0.0104, coverage 0.7599 to 0.7940, SE ratio 0.5701 to 0.6046 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j30__normal_reference` | diagnostic | stacked CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0022 to 0.0104, coverage 0.9264 to 0.9465, SE ratio 0.9642 to 1.0229 | reported |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j30__t_reference` | positive | stacked CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0022 to 0.0104, coverage 0.9372 to 0.9557, SE ratio 0.9635 to 1.0228 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__iid_t_control` | control | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0080 to 0.0041, coverage 0.9139 to 0.9356, SE ratio 0.7155 to 0.7578 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0080 to 0.0041, coverage 0.8675 to 0.8941, SE ratio 0.9389 to 1.0017 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias -0.0080 to 0.0041, coverage 0.9059 to 0.9286, SE ratio 0.9395 to 1.0023 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__t_reference` | positive | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0080 to 0.0041, coverage 0.9302 to 0.9497, SE ratio 0.9402 to 1.0033 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__iid_t_control` | control | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0042 to 0.0042, coverage 0.8620 to 0.8891, SE ratio 0.7132 to 0.7556 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0042 to 0.0042, coverage 0.9158 to 0.9372, SE ratio 0.9695 to 1.0294 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias -0.0042 to 0.0042, coverage 0.9318 to 0.9511, SE ratio 0.9698 to 1.0311 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__t_reference` | positive | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0042 to 0.0042, coverage 0.9350 to 0.9539, SE ratio 0.9709 to 1.0319 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__iid_t_control` | control | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0029 to 0.0039, coverage 0.8471 to 0.8755, SE ratio 0.7067 to 0.7476 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0029 to 0.0039, coverage 0.9283 to 0.9481, SE ratio 0.9689 to 1.0288 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias -0.0029 to 0.0039, coverage 0.9410 to 0.9589, SE ratio 0.9689 to 1.0298 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__t_reference` | positive | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0029 to 0.0039, coverage 0.9415 to 0.9594, SE ratio 0.9703 to 1.0290 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j10__iid_t_control` | control | fold-evaluated CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.1146 to 0.1310, coverage 0.7871 to 0.8197, SE ratio 0.5599 to 0.5934 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j10__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.1146 to 0.1310, coverage 0.8046 to 0.8361, SE ratio 0.8285 to 0.8842 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j10__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias 0.1146 to 0.1310, coverage 0.8610 to 0.8881, SE ratio 0.8293 to 0.8840 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j10__t_reference` | positive | fold-evaluated CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.1146 to 0.1310, coverage 0.8940 to 0.9180, SE ratio 0.8297 to 0.8849 | **fail** |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j20__iid_t_control` | control | fold-evaluated CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0528 to 0.0638, coverage 0.7424 to 0.7774, SE ratio 0.5557 to 0.5890 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j20__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0528 to 0.0638, coverage 0.8888 to 0.9133, SE ratio 0.9003 to 0.9565 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j20__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias 0.0528 to 0.0638, coverage 0.9051 to 0.9279, SE ratio 0.8992 to 0.9568 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j20__t_reference` | positive | fold-evaluated CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0528 to 0.0638, coverage 0.9078 to 0.9302, SE ratio 0.8994 to 0.9575 | **fail** |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j30__iid_t_control` | control | fold-evaluated CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0300 to 0.0386, coverage 0.7414 to 0.7765, SE ratio 0.5651 to 0.5991 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j30__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0300 to 0.0386, coverage 0.9118 to 0.9337, SE ratio 0.9387 to 0.9959 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j30__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias 0.0300 to 0.0386, coverage 0.9214 to 0.9421, SE ratio 0.9381 to 0.9970 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__unequal_informative__j30__t_reference` | positive | fold-evaluated CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0300 to 0.0386, coverage 0.9224 to 0.9430, SE ratio 0.9391 to 0.9971 | **fail** |
| `few_cluster_reference` | `tmle_in_sample__equal10__j10__iid_t_control` | control | in-sample TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0078 to 0.0041, coverage 0.8576 to 0.8850, SE ratio 0.6636 to 0.7037 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j10__normal_reference` | diagnostic | in-sample TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0078 to 0.0041, coverage 0.8809 to 0.9062, SE ratio 0.9240 to 0.9831 | reported |
| `few_cluster_reference` | `tmle_in_sample__equal10__j10__t_reference` | positive | in-sample TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0078 to 0.0041, coverage 0.9192 to 0.9402, SE ratio 0.9247 to 0.9850 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j20__iid_t_control` | control | in-sample TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0041 to 0.0042, coverage 0.8386 to 0.8676, SE ratio 0.6882 to 0.7292 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j20__normal_reference` | diagnostic | in-sample TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0041 to 0.0042, coverage 0.9176 to 0.9388, SE ratio 0.9622 to 1.0222 | reported |
| `few_cluster_reference` | `tmle_in_sample__equal10__j20__t_reference` | positive | in-sample TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0041 to 0.0042, coverage 0.9297 to 0.9493, SE ratio 0.9623 to 1.0211 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j30__iid_t_control` | control | in-sample TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0030 to 0.0038, coverage 0.8349 to 0.8642, SE ratio 0.6885 to 0.7299 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j30__normal_reference` | diagnostic | in-sample TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0030 to 0.0038, coverage 0.9307 to 0.9502, SE ratio 0.9634 to 1.0216 | reported |
| `few_cluster_reference` | `tmle_in_sample__equal10__j30__t_reference` | positive | in-sample TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0030 to 0.0038, coverage 0.9404 to 0.9585, SE ratio 0.9632 to 1.0219 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j10__iid_t_control` | control | in-sample TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0184 to 0.0334, coverage 0.7737 to 0.8071, SE ratio 0.5251 to 0.5584 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j10__normal_reference` | diagnostic | in-sample TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0184 to 0.0334, coverage 0.8717 to 0.8979, SE ratio 0.8700 to 0.9295 | reported |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j10__t_reference` | positive | in-sample TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0184 to 0.0334, coverage 0.9126 to 0.9344, SE ratio 0.8686 to 0.9295 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j20__iid_t_control` | control | in-sample TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0079 to 0.0181, coverage 0.7470 to 0.7818, SE ratio 0.5494 to 0.5832 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j20__normal_reference` | diagnostic | in-sample TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0079 to 0.0181, coverage 0.9168 to 0.9381, SE ratio 0.9315 to 0.9919 | reported |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j20__t_reference` | positive | in-sample TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0079 to 0.0181, coverage 0.9332 to 0.9523, SE ratio 0.9312 to 0.9916 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j30__iid_t_control` | control | in-sample TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.0021 to 0.0103, coverage 0.7514 to 0.7860, SE ratio 0.5624 to 0.5965 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j30__normal_reference` | diagnostic | in-sample TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0021 to 0.0103, coverage 0.9251 to 0.9453, SE ratio 0.9540 to 1.0132 | reported |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j30__t_reference` | positive | in-sample TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0021 to 0.0103, coverage 0.9348 to 0.9536, SE ratio 0.9540 to 1.0131 | pass |
<!-- /generated -->

The `few_cluster_reference` family shares each draw among five fits, four at 10 clusters: stacked CV-TMLE,
fold-evaluated CV-TMLE, in-sample TMLE, cross-fitted DR-TMLE and in-sample LTMLE. It crosses 10, 20
and 30 clusters with two size laws: clusters of 10 rows from the law of
`clustered_dgp(10, "binomial")`, and the informative law. Each fit publishes three arms. The
`t_reference` arm is the reported interval. The `iid_t_control` arm keeps the point and the
quantile and treats the rows as independent. The `normal_reference` arm keeps the cluster-robust
standard error with the normal quantile and is reported only.

The fold-evaluated fit adds a
reported `t_j_minus_2_reference` arm, which keeps $J-2$ on the same fits. Every learner is
parametric. Wang, Park, Small and Li (2024), Remark 3, caution against complex learners at about
20 clusters.

## Readings of the red cells

The run publishes seven red cells under the `reporting` policy, and
[F28](../../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns each one.

| cells | reading |
| --- | --- |
| `tmle_cv_evaluation__unequal_informative__j10`, `j20` and `j30`, `t_reference` | the fold-evaluated point is biased by 0.61, 0.43 and 0.32 empirical standard deviations. It averages one ratio estimate per validation fold, and each fold holds 2, 4 or 6 clusters of informative size. A ratio of means over few clusters carries a bias that shrinks with the cluster count of the fold. On the same draws the stacked, in-sample and DR-TMLE fits read 0.14, 0.14 and 0.17. At 10 clusters the coverage lower endpoint is 0.894 |
| `drtmle_crossfit__equal10__j10`, `j20` and `j30`, and `drtmle_crossfit__unequal_informative__j10`, `iid_t_control` | the IID control of the cross-fitted DR-TMLE reads an SE ratio of 0.67 to 0.92, so its upper endpoint stays above the 0.80 ceiling at four of six cells. The control does not separate the two variances on these fits. The cluster-robust `t_reference` arms of the same fits cover 0.93 to 0.95 |

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 200 | nominal observations per primary replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 49 | property cells passing |
| `property_cells_total` | 56 | property cells reported |
| `max_standardized_bias` | 0.1057 | largest primary standardized bias |
| `min_coverage` | 0.9310 | lowest primary coverage |
| `max_margin_utilization` | 3.687e-09 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | coverage floor of a `t_reference` cell |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |

## Limits

| limit | what it means for use |
| --- | --- |
| 10 to 30 clusters | 4 to 9 clusters are unmeasured, and a fit there reports no interval. At 4 clusters a pre-run probe found 0.25% to 9% of draws on which a fit cannot run. In-sample `LTMLE` starts at 20 clusters: the first full run lost 1 `LTMLE` fit in 4,000 at 10 clusters in each size law, so those cells left the grid before any verdict was read, and the package floor for `LTMLE` is 20 |
| parametric learners | the study does not establish flexible learners at few clusters |
| binary outcome and treatment, exact propensity | the study isolates targeting and inference |
| one primary cluster count | the paired comparison runs at 20 clusters only |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_few_cluster/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_few_cluster/manifest.json)
records the container, runner, harness, configuration and every result-determining module.
