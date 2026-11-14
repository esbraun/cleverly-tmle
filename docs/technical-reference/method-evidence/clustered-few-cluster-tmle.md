# Clustered TMLE intervals on a t reference at few clusters

This study measures the Student $t$ reference that a clustered fit takes below 40 clusters with
positive weight mass. An estimate that reads $J$ such clusters uses $J-2$ degrees of freedom, as
Nugent et al. (2024), Section 2.2, last paragraph, recommend. A fold-evaluated estimate over $V$
folds uses $J-V$ at every cluster count, because its variance centers the cluster totals inside
each fold. Below 10 clusters a fit carries the experimental label, and an `LTMLE` fit below 20,
because no registered study measures an interval there.
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
Comparator-backed: a paired comparison with `ltmle` on the same samples. Each row reports estimate equivalence within the declared margin, the RMSE ratio and the coverage difference, and names its comparison kind.

| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | comparison kind | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -6.790e-11 | 3.501e-09 | 1.0000 | 0.0010 | matching implementation | 1.276e-10 vs 0.0500 | equivalent |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[always]` | mean outcome under the plan treat at both times | -4.716e-11 | 3.687e-09 | 1.0000 | 0.0010 | matching implementation | 1.021e-10 vs 0.0500 | equivalent |
| 20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 2.074e-11 | 1.885e-09 | 1.0000 | 0.0010 | matching implementation | 2.356e-10 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j10__iid_t_control` | control | cross-fitted DR-TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7586 to 0.7735, paired positive SE-ratio upper endpoint 1.1064 | pass |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j10__normal_reference` | diagnostic | cross-fitted DR-TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0062 to 0.0059, coverage 0.9073 to 0.9297, SE ratio 1.0352 to 1.1053 | reported |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j10__t_reference` | positive | cross-fitted DR-TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0062 to 0.0059, coverage 0.9380 to 0.9564, SE ratio 1.0342 to 1.1064 | pass |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j20__iid_t_control` | control | cross-fitted DR-TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7298 to 0.7379, paired positive SE-ratio upper endpoint 1.0737 | pass |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j20__normal_reference` | diagnostic | cross-fitted DR-TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0021 to 0.0063, coverage 0.9286 to 0.9483, SE ratio 1.0116 to 1.0747 | reported |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j20__t_reference` | positive | cross-fitted DR-TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0021 to 0.0063, coverage 0.9432 to 0.9607, SE ratio 1.0120 to 1.0737 | pass |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j30__iid_t_control` | control | cross-fitted DR-TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7248 to 0.7308, paired positive SE-ratio upper endpoint 1.0524 | pass |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j30__normal_reference` | diagnostic | cross-fitted DR-TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0017 to 0.0052, coverage 0.9364 to 0.9550, SE ratio 0.9916 to 1.0523 | reported |
| `few_cluster_reference` | `drtmle_crossfit__equal10__j30__t_reference` | positive | cross-fitted DR-TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0017 to 0.0052, coverage 0.9442 to 0.9617, SE ratio 0.9911 to 1.0524 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j10__iid_t_control` | control | cross-fitted DR-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.6750 to 0.6929, paired positive SE-ratio upper endpoint 0.9918 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j10__normal_reference` | diagnostic | cross-fitted DR-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0215 to 0.0368, coverage 0.8817 to 0.9069, SE ratio 0.9229 to 0.9920 | reported |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j10__t_reference` | positive | cross-fitted DR-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0215 to 0.0368, coverage 0.9222 to 0.9428, SE ratio 0.9240 to 0.9918 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j20__iid_t_control` | control | cross-fitted DR-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.6189 to 0.6270, paired positive SE-ratio upper endpoint 1.0110 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j20__normal_reference` | diagnostic | cross-fitted DR-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0091 to 0.0195, coverage 0.9216 to 0.9423, SE ratio 0.9504 to 1.0100 | reported |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j20__t_reference` | positive | cross-fitted DR-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0091 to 0.0195, coverage 0.9372 to 0.9557, SE ratio 0.9504 to 1.0110 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j30__iid_t_control` | control | cross-fitted DR-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.6058 to 0.6118, paired positive SE-ratio upper endpoint 1.0204 | pass |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j30__normal_reference` | diagnostic | cross-fitted DR-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0029 to 0.0112, coverage 0.9267 to 0.9467, SE ratio 0.9605 to 1.0211 | reported |
| `few_cluster_reference` | `drtmle_crossfit__unequal_informative__j30__t_reference` | positive | cross-fitted DR-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0029 to 0.0112, coverage 0.9361 to 0.9548, SE ratio 0.9615 to 1.0204 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j20__iid_t_control` | control | in-sample LTMLE on one node, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7103 to 0.7178, paired positive SE-ratio upper endpoint 1.0216 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j20__normal_reference` | diagnostic | in-sample LTMLE on one node, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0043 to 0.0040, coverage 0.9187 to 0.9398, SE ratio 0.9628 to 1.0222 | reported |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j20__t_reference` | positive | in-sample LTMLE on one node, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0043 to 0.0040, coverage 0.9307 to 0.9502, SE ratio 0.9637 to 1.0216 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j30__iid_t_control` | control | in-sample LTMLE on one node, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7119 to 0.7180, paired positive SE-ratio upper endpoint 1.0226 | pass |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j30__normal_reference` | diagnostic | in-sample LTMLE on one node, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0031 to 0.0038, coverage 0.9297 to 0.9493, SE ratio 0.9623 to 1.0235 | reported |
| `few_cluster_reference` | `ltmle_in_sample__equal10__j30__t_reference` | positive | in-sample LTMLE on one node, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0031 to 0.0038, coverage 0.9396 to 0.9578, SE ratio 0.9630 to 1.0226 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j20__iid_t_control` | control | in-sample LTMLE on one node, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.5862 to 0.5926, paired positive SE-ratio upper endpoint 0.9925 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j20__normal_reference` | diagnostic | in-sample LTMLE on one node, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0079 to 0.0181, coverage 0.9163 to 0.9377, SE ratio 0.9307 to 0.9911 | reported |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j20__t_reference` | positive | in-sample LTMLE on one node, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0079 to 0.0181, coverage 0.9329 to 0.9520, SE ratio 0.9327 to 0.9925 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j30__iid_t_control` | control | in-sample LTMLE on one node, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.5862 to 0.5911, paired positive SE-ratio upper endpoint 1.0131 | pass |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j30__normal_reference` | diagnostic | in-sample LTMLE on one node, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0021 to 0.0103, coverage 0.9254 to 0.9456, SE ratio 0.9545 to 1.0141 | reported |
| `few_cluster_reference` | `ltmle_in_sample__unequal_informative__j30__t_reference` | positive | in-sample LTMLE on one node, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0021 to 0.0103, coverage 0.9348 to 0.9536, SE ratio 0.9548 to 1.0131 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j10__iid_t_control` | control | stacked CV-TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7282 to 0.7402, paired positive SE-ratio upper endpoint 1.0142 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j10__normal_reference` | diagnostic | stacked CV-TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0080 to 0.0041, coverage 0.8874 to 0.9121, SE ratio 0.9529 to 1.0141 | reported |
| `few_cluster_reference` | `tmle_crossfit__equal10__j10__t_reference` | positive | stacked CV-TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0080 to 0.0041, coverage 0.9243 to 0.9446, SE ratio 0.9522 to 1.0142 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j20__iid_t_control` | control | stacked CV-TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7188 to 0.7265, paired positive SE-ratio upper endpoint 1.0363 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j20__normal_reference` | diagnostic | stacked CV-TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0042 to 0.0042, coverage 0.9203 to 0.9412, SE ratio 0.9754 to 1.0353 | reported |
| `few_cluster_reference` | `tmle_crossfit__equal10__j20__t_reference` | positive | stacked CV-TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0042 to 0.0042, coverage 0.9324 to 0.9516, SE ratio 0.9738 to 1.0363 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j30__iid_t_control` | control | stacked CV-TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7174 to 0.7235, paired positive SE-ratio upper endpoint 1.0312 | pass |
| `few_cluster_reference` | `tmle_crossfit__equal10__j30__normal_reference` | diagnostic | stacked CV-TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0029 to 0.0039, coverage 0.9329 to 0.9520, SE ratio 0.9725 to 1.0315 | reported |
| `few_cluster_reference` | `tmle_crossfit__equal10__j30__t_reference` | positive | stacked CV-TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0029 to 0.0039, coverage 0.9423 to 0.9601, SE ratio 0.9708 to 1.0312 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j10__iid_t_control` | control | stacked CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.6062 to 0.6163, paired positive SE-ratio upper endpoint 0.9574 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j10__normal_reference` | diagnostic | stacked CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0182 to 0.0332, coverage 0.8822 to 0.9074, SE ratio 0.8948 to 0.9579 | reported |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j10__t_reference` | positive | stacked CV-TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0182 to 0.0332, coverage 0.9200 to 0.9409, SE ratio 0.8942 to 0.9574 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j20__iid_t_control` | control | stacked CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.5902 to 0.5965, paired positive SE-ratio upper endpoint 1.0053 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j20__normal_reference` | diagnostic | stacked CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0078 to 0.0180, coverage 0.9192 to 0.9402, SE ratio 0.9453 to 1.0059 | reported |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j20__t_reference` | positive | stacked CV-TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0078 to 0.0180, coverage 0.9353 to 0.9541, SE ratio 0.9443 to 1.0053 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j30__iid_t_control` | control | stacked CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.5885 to 0.5935, paired positive SE-ratio upper endpoint 1.0228 | pass |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j30__normal_reference` | diagnostic | stacked CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0022 to 0.0104, coverage 0.9264 to 0.9465, SE ratio 0.9642 to 1.0229 | reported |
| `few_cluster_reference` | `tmle_crossfit__unequal_informative__j30__t_reference` | positive | stacked CV-TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0022 to 0.0104, coverage 0.9372 to 0.9557, SE ratio 0.9635 to 1.0228 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__iid_t_control` | control | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7492 to 0.7674, paired positive SE-ratio upper endpoint 1.0033 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0080 to 0.0041, coverage 0.8675 to 0.8941, SE ratio 0.9389 to 1.0017 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias -0.0080 to 0.0041, coverage 0.9059 to 0.9286, SE ratio 0.9395 to 1.0023 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j10__t_reference` | positive | fold-evaluated CV-TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0080 to 0.0041, coverage 0.9302 to 0.9497, SE ratio 0.9402 to 1.0033 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__iid_t_control` | control | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7293 to 0.7386, paired positive SE-ratio upper endpoint 1.0319 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0042 to 0.0042, coverage 0.9158 to 0.9372, SE ratio 0.9695 to 1.0294 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias -0.0042 to 0.0042, coverage 0.9318 to 0.9511, SE ratio 0.9698 to 1.0311 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j20__t_reference` | positive | fold-evaluated CV-TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0042 to 0.0042, coverage 0.9350 to 0.9539, SE ratio 0.9709 to 1.0319 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__iid_t_control` | control | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7238 to 0.7308, paired positive SE-ratio upper endpoint 1.0290 | pass |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__normal_reference` | diagnostic | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0029 to 0.0039, coverage 0.9283 to 0.9481, SE ratio 0.9689 to 1.0288 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__t_j_minus_2_reference` | diagnostic | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the fold-evaluated standard error with t at J - 2 degrees of freedom | reported only | bias -0.0029 to 0.0039, coverage 0.9410 to 0.9589, SE ratio 0.9689 to 1.0298 | reported |
| `few_cluster_reference` | `tmle_cv_evaluation__equal10__j30__t_reference` | positive | fold-evaluated CV-TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0029 to 0.0039, coverage 0.9415 to 0.9594, SE ratio 0.9703 to 1.0290 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j10__iid_t_control` | control | in-sample TMLE, 10 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7110 to 0.7227, paired positive SE-ratio upper endpoint 0.9850 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j10__normal_reference` | diagnostic | in-sample TMLE, 10 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0078 to 0.0041, coverage 0.8809 to 0.9062, SE ratio 0.9240 to 0.9831 | reported |
| `few_cluster_reference` | `tmle_in_sample__equal10__j10__t_reference` | positive | in-sample TMLE, 10 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0078 to 0.0041, coverage 0.9192 to 0.9402, SE ratio 0.9247 to 0.9850 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j20__iid_t_control` | control | in-sample TMLE, 20 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7106 to 0.7182, paired positive SE-ratio upper endpoint 1.0211 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j20__normal_reference` | diagnostic | in-sample TMLE, 20 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0041 to 0.0042, coverage 0.9176 to 0.9388, SE ratio 0.9622 to 1.0222 | reported |
| `few_cluster_reference` | `tmle_in_sample__equal10__j20__t_reference` | positive | in-sample TMLE, 20 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0041 to 0.0042, coverage 0.9297 to 0.9493, SE ratio 0.9623 to 1.0211 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j30__iid_t_control` | control | in-sample TMLE, 30 clusters of 10 rows: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.7122 to 0.7183, paired positive SE-ratio upper endpoint 1.0219 | pass |
| `few_cluster_reference` | `tmle_in_sample__equal10__j30__normal_reference` | diagnostic | in-sample TMLE, 30 clusters of 10 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0030 to 0.0038, coverage 0.9307 to 0.9502, SE ratio 0.9634 to 1.0216 | reported |
| `few_cluster_reference` | `tmle_in_sample__equal10__j30__t_reference` | positive | in-sample TMLE, 30 clusters of 10 rows: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0030 to 0.0038, coverage 0.9404 to 0.9585, SE ratio 0.9632 to 1.0219 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j10__iid_t_control` | control | in-sample TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.5968 to 0.6069, paired positive SE-ratio upper endpoint 0.9295 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j10__normal_reference` | diagnostic | in-sample TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0184 to 0.0334, coverage 0.8717 to 0.8979, SE ratio 0.8700 to 0.9295 | reported |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j10__t_reference` | positive | in-sample TMLE, 10 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0184 to 0.0334, coverage 0.9126 to 0.9344, SE ratio 0.8686 to 0.9295 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j20__iid_t_control` | control | in-sample TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.5862 to 0.5925, paired positive SE-ratio upper endpoint 0.9916 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j20__normal_reference` | diagnostic | in-sample TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0079 to 0.0181, coverage 0.9168 to 0.9381, SE ratio 0.9315 to 0.9919 | reported |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j20__t_reference` | positive | in-sample TMLE, 20 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0079 to 0.0181, coverage 0.9332 to 0.9523, SE ratio 0.9312 to 0.9916 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j30__iid_t_control` | control | in-sample TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the same point and t quantile with the IID row standard error | the paired ratio of mean IID SE to mean cluster SE on the same fits has its upper endpoint below 0.80, and the paired t_reference arm's SE-ratio upper endpoint is at most 1.20 | paired IID/cluster SE ratio 0.5861 to 0.5910, paired positive SE-ratio upper endpoint 1.0131 | pass |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j30__normal_reference` | diagnostic | in-sample TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the cluster-robust standard error with the normal quantile | reported only | bias 0.0021 to 0.0103, coverage 0.9251 to 0.9453, SE ratio 0.9540 to 1.0132 | reported |
| `few_cluster_reference` | `tmle_in_sample__unequal_informative__j30__t_reference` | positive | in-sample TMLE, 30 clusters of size uniform on 2 to 18, the size in the outcome: the reported interval, t with J - 2 degrees of freedom, or J - V for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.0021 to 0.0103, coverage 0.9348 to 0.9536, SE ratio 0.9540 to 1.0131 | pass |
<!-- /generated -->

The `few_cluster_reference` family shares each draw among five fits, four at 10 clusters: stacked CV-TMLE,
fold-evaluated CV-TMLE, in-sample TMLE, cross-fitted DR-TMLE and in-sample LTMLE. It crosses 10, 20
and 30 clusters with two size laws: clusters of 10 rows from the law of
`clustered_dgp(10, "binomial")`, and the informative law. The fold-evaluated fit runs at equal
sizes only, because `cv_evaluation=True` refuses clusters of unequal size.

Each fit publishes three arms. The `t_reference` arm is the reported interval. The `iid_t_control` arm keeps the
point and the quantile and treats the rows as independent. The `normal_reference` arm keeps the
cluster-robust standard error with the normal quantile and is reported only.

The `iid_t_control` arm shows that the law's clustering changes the variance of the fit by more
than any overstatement the SE screen admits. Its verdict needs two clauses. The 99% upper
endpoint of the paired ratio, the mean IID SE over the mean cluster-robust SE of the same fits,
must fall below 0.80. The paired `t_reference` arm's SE-ratio upper endpoint must be at most
1.20, the study's SE screen. The paired ratio alone would pass a cluster SE inflated by a factor
$c$ at about $1/c$ on a law with no cluster effect, and the second clause bounds $c$.

Under the second clause, a
paired ratio below 0.80 implies an IID SE more than 4% below the empirical SD at the upper
endpoints. The control row still publishes the IID SE over the empirical SD, and no verdict reads
it. `tests/unit/test_few_cluster_control.py` holds the witness, the inflated-SE mutation and the
lost-replicate worst case.

The fold-evaluated fit adds a
reported `t_j_minus_2_reference` arm, which keeps $J-2$ on the same fits. Every learner is
parametric. Wang, Park, Small and Li (2024), Remark 3, caution against complex learners at about
20 clusters.

## Changes since the first run

The first run, at cleverly 0.1.2, published seven red cells. The study was regenerated with the
same seeds after three changes, and every retained cell passes.

The fold-evaluated fit at informative sizes left the grid: 12 cells, three counts by four arms,
three of them red. Its point was biased by 0.61, 0.43 and 0.32 empirical standard deviations at
10, 20 and 30 clusters, against 0.14, 0.10 and 0.06 for the stacked fit on the same draws. It
takes the equal $1/V$ average of one ratio estimate per validation fold, each over 2, 4 or 6
clusters of informative size, so it carries about $V$ times the ratio bias of the stacked fit.
`tests/unit/test_fold_evaluated_ratio_bias.py` rebuilds that bias from the exact law with no fit.
`cv_evaluation=True` now refuses such clusters and redirects to the stacked report, which
estimates the same ATE ([clusters of unequal mass](../cv-tmle.md#clusters-of-unequal-mass)).

The fold-evaluated fit reports $t_{J-V}$ at every cluster count. At 10 to 30 clusters in 5 folds
that equals the earlier $\min(J-2, J-V)$, so no retained cell moves with it.

The `iid_t_control` verdict reads the two clauses above. The first run compared the IID SE with
the empirical SD, and its four cross-fitted DR-TMLE controls read red: the IID SE ratio was 0.92,
0.85 and 0.83 at 10, 20 and 30 equal clusters and 0.75 at 10 informative clusters, with upper
endpoints at 0.80 or more. On those fits the cluster-robust SE overstated the empirical SD by 19%,
14% and 12%, and the IID SE, read from the same curve, rose with it while staying 23% to 28%
below the cluster-robust SE. A ratio to the empirical SD moves with any noise in the SE, not with
the clustering alone; the paired ratio reads the clustering, and the second clause keeps a wide
cluster SE from helping the control.

The committed DR-TMLE rows do not describe the shipped estimator. Refitting the first run's
replicates with its own seeds reproduces the stacked `TMLE` rows bit for bit and does not
reproduce the `DRTMLE` rows, ordinary replicates as well as the tail draws. The `foldwise`
reduced regressions changed between 0.1.2 and 0.1.5. The `pooled` path of `canonical-drtmle`,
also run at 0.1.2, still reproduces, and no other `foldwise` artifact predates 0.1.5, so this
study was the only stale one.

The regenerated DR-TMLE cells give the shipped reading. The cluster-robust SE ratio of the
`t_reference` arm is 1.07, 1.04 and 1.02 at 10, 20 and 30 equal clusters, and 0.96, 0.98 and
0.99 at informative sizes. Its 99% upper endpoint is at most 1.11, inside the 1.20 screen, and
its coverage is 0.933 to 0.954. The paired IID/cluster ratio has upper endpoints of 0.61 to 0.77,
so all four controls that read red now pass both clauses.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 200 | nominal observations per primary replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 50 | property cells passing |
| `property_cells_total` | 50 | property cells reported |
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
| `margin:subject_failure_bound` | 0.0100 | share of `cleverly` fits a cell may lose |
| `margin:subject_failures_allowed` | 10 | `cleverly` fits a cell may lose |
| `margin:reference_failure_bound` | 0.0100 | share of R `ltmle` fits a cell may lose |
| `margin:reference_failures_allowed` | 10 | R `ltmle` fits a cell may lose |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |

## Limits

| limit | what it means for use |
| --- | --- |
| 10 to 30 clusters | 4 to 9 clusters are unmeasured, and a fit there carries the experimental label. At 4 clusters a pre-run probe found 0.25% to 9% of draws on which a fit cannot run. In-sample `LTMLE` starts at 20 clusters: the first full run lost 1 `LTMLE` fit in 4,000 at 10 clusters in each size law, so those cells left the grid before any verdict was read, and the package floor for `LTMLE` is 20 |
| parametric learners | the study does not establish flexible learners at few clusters |
| binary outcome and treatment, exact propensity | the study isolates targeting and inference |
| one primary cluster count | the paired comparison runs at 20 clusters only |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_few_cluster/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_few_cluster/manifest.json)
records the container, runner, harness, configuration and every result-determining module.
