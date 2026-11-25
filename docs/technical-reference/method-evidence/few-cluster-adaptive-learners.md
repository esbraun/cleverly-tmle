# Few-cluster intervals with the default learner library

This study measures the clustered interval on $t_{J-2}$ ([few clusters](../inference.md#few-clusters))
when every nuisance is fitted by the default learner library: histogram gradient boosting, a
random forest and an L1 logistic or lasso regression, stacked by a Super Learner with grouped
inner folds. Wang, Park, Small and Li (2024), Remark 3, caution against complex learners at about
20 clusters. The registered few-cluster studies measured parametric learners only, so this study
reads stacked `TMLE` at 10, 20 and 40 clusters and cross-fitted `LTMLE` at 20.

**Status of this page.** The study is registered, and its run has completed. No other
implementation fits a few-cluster TMLE with a Super Learner and a $t$ reference, so the study has
no comparator. The publication policy is `reporting`.

Every gated row passes, so stacked `TMLE`
at 10 to 40 clusters and cross-fitted `LTMLE` at 20 clusters, with the default library, keep
their row's label (`T11.stacked`, `T10.unweighted` at 40 clusters, and `G07` from 20 clusters).
Another adaptive learner, `LTMLE` below 20 clusters and the fold-evaluated, `DRTMLE` and `SDR`
fits are not measured, and read experimental. The in-sample fits are reported only: an in-sample
fit with a data-adaptive learner stays experimental by theory. No row is red, so the study holds no row
in the [red-cell ledger](red-cells.md).

| part | what it is | status |
| --- | --- | --- |
| primary | stacked `TMLE` with the default library at 20 clusters of `equal10`, 1,200 replications: the ATE | the accuracy table below. It passes |
| property family `adaptive_learners` | stacked and in-sample `TMLE` on two laws at 10, 20 and 40 clusters, and cross-fitted `LTMLE` at 20, 4,000 draws per cell, each draw also fitted with GLM nuisances and two overfit controls | the tables below give the verdicts |
| comparator runs | none | no other implementation computes this interval with these learners |

## What was fitted

Law L1, `equal10`, is the equal-size law of `clustered-few-cluster-tmle`: clusters of 10 rows, a
cluster latent in the outcome and the arm-by-latent coefficient 8. Law L2, `nonlinear10`, has the
same layout with an outcome logit in $\sin(2W_1)$, $W_1W_2$ and the latent, and a propensity
logit in $W_1^2$, so a main-terms GLM is wrong for both nuisances. Its truth is exact quadrature.
Law L3 is the end-of-study law of `few-cluster-cross-fitted-ltmle` at 20 clusters of 40 units.

Each draw of L1 and L2 is fitted six times: in-sample and stacked `TMLE` (five grouped folds) with
the default library, the same two with main-terms logistic nuisances, an in-sample fit with an
unpruned tree outcome, and a stacked control whose out-of-fold outcome predictions come from an
unpruned random forest fitted on the full sample. Each draw of L3 is fitted by cross-fitted
`LTMLE` with the default library for every outcome node and by the committed GLM subject. The laws,
fits, arms and seeds are in `tests/studies/few_cluster_adaptive_learners.py`, and
`tests/unit/test_few_cluster_adaptive_design.py` recomputes them.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 clusters of 10 rows with a cluster latent in the outcome, a stacked TMLE with five grouped folds and the default learner library for both nuisances, the shipped report | `ate` | average treatment effect | `cleverly` | -0.0075 to 0.0090 | 0.9358 | 0.9756 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

None claimed. No canonical implementation is compared, because no other implementation computes
this interval with these learners. [0/0 external comparisons; the row rests on the other two columns]

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `adaptive_learners` | `glm_in_sample__equal10__j10__glm_reference` | diagnostic | in-sample TMLE, main-terms logistic outcome and propensity, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0055 to 0.0064, coverage 0.9216 to 0.9423, SE ratio 0.9506 | reported |
| `adaptive_learners` | `glm_in_sample__equal10__j20__glm_reference` | diagnostic | in-sample TMLE, main-terms logistic outcome and propensity, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0045 to 0.0040, coverage 0.9294 to 0.9490, SE ratio 0.9750 | reported |
| `adaptive_learners` | `glm_in_sample__equal10__j40__glm_reference` | diagnostic | in-sample TMLE, main-terms logistic outcome and propensity, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0032 to 0.0027, coverage 0.9206 to 0.9414, SE ratio 0.9899 | reported |
| `adaptive_learners` | `glm_in_sample__nonlinear10__j10__glm_reference` | diagnostic | in-sample TMLE, main-terms logistic outcome and propensity, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0031 to 0.0091, coverage 0.9208 to 0.9416, SE ratio 0.9368 | reported |
| `adaptive_learners` | `glm_in_sample__nonlinear10__j20__glm_reference` | diagnostic | in-sample TMLE, main-terms logistic outcome and propensity, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.000223 to 0.0083, coverage 0.9337 to 0.9527, SE ratio 0.9747 | reported |
| `adaptive_learners` | `glm_in_sample__nonlinear10__j40__glm_reference` | diagnostic | in-sample TMLE, main-terms logistic outcome and propensity, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0011 to 0.0049, coverage 0.9264 to 0.9465, SE ratio 0.9903 | reported |
| `adaptive_learners` | `glm_stacked__equal10__j10__glm_reference` | diagnostic | stacked TMLE with five grouped folds, main-terms logistic nuisances, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0058 to 0.0067, coverage 0.9356 to 0.9543, SE ratio 1.0255 | reported |
| `adaptive_learners` | `glm_stacked__equal10__j20__glm_reference` | diagnostic | stacked TMLE with five grouped folds, main-terms logistic nuisances, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0046 to 0.0041, coverage 0.9356 to 0.9543, SE ratio 1.0122 | reported |
| `adaptive_learners` | `glm_stacked__equal10__j40__glm_reference` | diagnostic | stacked TMLE with five grouped folds, main-terms logistic nuisances, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0033 to 0.0028, coverage 0.9251 to 0.9453, SE ratio 1.0083 | reported |
| `adaptive_learners` | `glm_stacked__nonlinear10__j10__glm_reference` | diagnostic | stacked TMLE with five grouped folds, main-terms logistic nuisances, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0033 to 0.0094, coverage 0.9313 to 0.9507, SE ratio 1.0106 | reported |
| `adaptive_learners` | `glm_stacked__nonlinear10__j20__glm_reference` | diagnostic | stacked TMLE with five grouped folds, main-terms logistic nuisances, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.000476 to 0.0083, coverage 0.9391 to 0.9573, SE ratio 1.0106 | reported |
| `adaptive_learners` | `glm_stacked__nonlinear10__j40__glm_reference` | diagnostic | stacked TMLE with five grouped folds, main-terms logistic nuisances, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0014 to 0.0047, coverage 0.9281 to 0.9479, SE ratio 1.0078 | reported |
| `adaptive_learners` | `interpolating_control__equal10__j10__interpolating_control` | diagnostic | in-sample TMLE, an unpruned tree outcome and a logistic propensity, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the interpolating outcome fit | none. The row is a diagnostic and states no verdict | bias -0.0182 to -0.0062, coverage 0.4548 to 0.4957, SE ratio 0.3231 | reported |
| `adaptive_learners` | `interpolating_control__equal10__j20__interpolating_control` | diagnostic | in-sample TMLE, an unpruned tree outcome and a logistic propensity, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the interpolating outcome fit | none. The row is a diagnostic and states no verdict | bias -0.0129 to -0.0040, coverage 0.4790 to 0.5200, SE ratio 0.3582 | reported |
| `adaptive_learners` | `interpolating_control__equal10__j40__interpolating_control` | diagnostic | in-sample TMLE, an unpruned tree outcome and a logistic propensity, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the interpolating outcome fit | none. The row is a diagnostic and states no verdict | bias -0.0078 to -0.0014, coverage 0.5138 to 0.5546, SE ratio 0.4019 | reported |
| `adaptive_learners` | `interpolating_control__nonlinear10__j10__interpolating_control` | diagnostic | in-sample TMLE, an unpruned tree outcome and a logistic propensity, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the interpolating outcome fit | none. The row is a diagnostic and states no verdict | bias -0.0168 to -0.0048, coverage 0.4768 to 0.5177, SE ratio 0.3265 | reported |
| `adaptive_learners` | `interpolating_control__nonlinear10__j20__interpolating_control` | diagnostic | in-sample TMLE, an unpruned tree outcome and a logistic propensity, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the interpolating outcome fit | none. The row is a diagnostic and states no verdict | bias -0.0110 to -0.0023, coverage 0.5100 to 0.5509, SE ratio 0.3784 | reported |
| `adaptive_learners` | `interpolating_control__nonlinear10__j40__interpolating_control` | diagnostic | in-sample TMLE, an unpruned tree outcome and a logistic propensity, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the interpolating outcome fit | none. The row is a diagnostic and states no verdict | bias -0.0072 to -0.000755, coverage 0.5557 to 0.5962, SE ratio 0.4145 | reported |
| `adaptive_learners` | `ltmle_crossfit__equal40__j20__iid_t_control` | control | cross-fitted LTMLE with five folds, the default library for every node, 20 clusters of 40 units on the end-of-study longitudinal law (L3): the same point and t quantile with the IID row standard error | on the shown draws, the 99% upper end of the paired ratio of mean IID SE to mean cluster SE below 0.80, and the positive arm's shown-only SE-ratio upper end at most 1.20 | bias -0.0041 to 0.0023, coverage 0.8051 to 0.8366, SE ratio 0.6404 | pass |
| `adaptive_learners` | `ltmle_crossfit__equal40__j20__t_reference` | positive | cross-fitted LTMLE with five folds, the default library for every node, 20 clusters of 40 units on the end-of-study longitudinal law (L3): the shipped interval, t with J - 2 degrees of freedom | the exact 99% lower end of all-draw coverage at least 0.90, bias inside 0.25 empirical SD, the 99% bootstrap lower end of the variance ratio at least 0.8649, and the exact 99% lower end of shown coverage at least 0.92 | bias -0.0041 to 0.0023, coverage 0.9307 to 0.9502, SE ratio 0.9719 | pass |
| `adaptive_learners` | `ltmle_crossfit__equal40__j20__wang_inflated` | diagnostic | cross-fitted LTMLE with five folds, the default library for every node, 20 clusters of 40 units on the end-of-study longitudinal law (L3): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0041 to 0.0023, coverage 0.9426 to 0.9603, SE ratio 1.0245 | reported |
| `adaptive_learners` | `ltmle_glm__equal40__j20__glm_reference` | diagnostic | cross-fitted LTMLE with five folds, a quasibinomial GLM for every node, 20 clusters of 40 units on the end-of-study longitudinal law (L3): the shipped interval of the GLM fit on the same draws | none. The row is a diagnostic and states no verdict | bias -0.0040 to 0.0024, coverage 0.9315 to 0.9509, SE ratio 0.9716 | reported |
| `adaptive_learners` | `stacked_control__equal10__j10__stacked_control` | control | stacked TMLE whose out-of-fold outcome predictions come from an unpruned random forest fitted on the full sample, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the overfit out-of-fold outcome | the exact 99% upper end of all-draw coverage below 0.90 | bias -0.0124 to -0.0014, coverage 0.8497 to 0.8779, SE ratio 0.7003 | pass |
| `adaptive_learners` | `stacked_control__equal10__j20__stacked_control` | control | stacked TMLE whose out-of-fold outcome predictions come from an unpruned random forest fitted on the full sample, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the overfit out-of-fold outcome | the exact 99% upper end of all-draw coverage below 0.90 | bias -0.0091 to -0.0010, coverage 0.8212 to 0.8515, SE ratio 0.6939 | pass |
| `adaptive_learners` | `stacked_control__equal10__j40__stacked_control` | control | stacked TMLE whose out-of-fold outcome predictions come from an unpruned random forest fitted on the full sample, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval of the overfit out-of-fold outcome | the exact 99% upper end of all-draw coverage below 0.90 | bias -0.0063 to -0.000542, coverage 0.8041 to 0.8356, SE ratio 0.6960 | pass |
| `adaptive_learners` | `stacked_control__nonlinear10__j10__stacked_control` | control | stacked TMLE whose out-of-fold outcome predictions come from an unpruned random forest fitted on the full sample, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the overfit out-of-fold outcome | the exact 99% upper end of all-draw coverage below 0.90 | bias -0.0104 to 0.000636, coverage 0.8414 to 0.8702, SE ratio 0.7012 | pass |
| `adaptive_learners` | `stacked_control__nonlinear10__j20__stacked_control` | control | stacked TMLE whose out-of-fold outcome predictions come from an unpruned random forest fitted on the full sample, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the overfit out-of-fold outcome | the exact 99% upper end of all-draw coverage below 0.90 | bias -0.0066 to 0.0013, coverage 0.8274 to 0.8573, SE ratio 0.7061 | pass |
| `adaptive_learners` | `stacked_control__nonlinear10__j40__stacked_control` | control | stacked TMLE whose out-of-fold outcome predictions come from an unpruned random forest fitted on the full sample, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval of the overfit out-of-fold outcome | the exact 99% upper end of all-draw coverage below 0.90 | bias -0.0054 to 0.000339, coverage 0.8088 to 0.8400, SE ratio 0.7053 | pass |
| `adaptive_learners` | `tmle_in_sample__equal10__j10__iid_t_control` | diagnostic | in-sample TMLE, the default learner library, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the same point and t quantile with the IID row standard error | none. The row is a diagnostic and states no verdict | bias -0.000490 to 0.0114, coverage 0.8297 to 0.8594, SE ratio 0.6242 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j10__t_reference` | diagnostic | in-sample TMLE, the default learner library, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval, t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.000490 to 0.0114, coverage 0.9038 to 0.9267, SE ratio 0.8609 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j10__wang_inflated` | diagnostic | in-sample TMLE, the default learner library, 10 clusters of 10 rows, a cluster latent in the outcome (L1): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.000490 to 0.0114, coverage 0.9270 to 0.9470, SE ratio 0.9625 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j20__iid_t_control` | diagnostic | in-sample TMLE, the default learner library, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the same point and t quantile with the IID row standard error | none. The row is a diagnostic and states no verdict | bias -0.0027 to 0.0057, coverage 0.7995 to 0.8313, SE ratio 0.6486 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j20__t_reference` | diagnostic | in-sample TMLE, the default learner library, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval, t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0027 to 0.0057, coverage 0.9075 to 0.9300, SE ratio 0.9034 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j20__wang_inflated` | diagnostic | in-sample TMLE, the default learner library, 20 clusters of 10 rows, a cluster latent in the outcome (L1): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0027 to 0.0057, coverage 0.9208 to 0.9416, SE ratio 0.9522 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j40__iid_t_control` | diagnostic | in-sample TMLE, the default learner library, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the same point and t quantile with the IID row standard error | none. The row is a diagnostic and states no verdict | bias -0.0030 to 0.0029, coverage 0.7915 to 0.8238, SE ratio 0.6699 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j40__t_reference` | diagnostic | in-sample TMLE, the default learner library, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval, t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0030 to 0.0029, coverage 0.9075 to 0.9300, SE ratio 0.9364 | reported |
| `adaptive_learners` | `tmle_in_sample__equal10__j40__wang_inflated` | diagnostic | in-sample TMLE, the default learner library, 40 clusters of 10 rows, a cluster latent in the outcome (L1): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0030 to 0.0029, coverage 0.9222 to 0.9428, SE ratio 0.9607 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j10__iid_t_control` | diagnostic | in-sample TMLE, the default learner library, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the same point and t quantile with the IID row standard error | none. The row is a diagnostic and states no verdict | bias -0.000105 to 0.0121, coverage 0.7938 to 0.8260, SE ratio 0.5944 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j10__t_reference` | diagnostic | in-sample TMLE, the default learner library, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval, t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.000105 to 0.0121, coverage 0.8938 to 0.9178, SE ratio 0.8320 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j10__wang_inflated` | diagnostic | in-sample TMLE, the default learner library, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.000105 to 0.0121, coverage 0.9171 to 0.9384, SE ratio 0.9302 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j20__iid_t_control` | diagnostic | in-sample TMLE, the default learner library, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the same point and t quantile with the IID row standard error | none. The row is a diagnostic and states no verdict | bias 0.0020 to 0.0104, coverage 0.7807 to 0.8136, SE ratio 0.6173 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j20__t_reference` | diagnostic | in-sample TMLE, the default learner library, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval, t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias 0.0020 to 0.0104, coverage 0.8977 to 0.9213, SE ratio 0.8719 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j20__wang_inflated` | diagnostic | in-sample TMLE, the default learner library, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias 0.0020 to 0.0104, coverage 0.9160 to 0.9374, SE ratio 0.9190 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j40__iid_t_control` | diagnostic | in-sample TMLE, the default learner library, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the same point and t quantile with the IID row standard error | none. The row is a diagnostic and states no verdict | bias -0.000061 to 0.0059, coverage 0.7683 to 0.8020, SE ratio 0.6291 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j40__t_reference` | diagnostic | in-sample TMLE, the default learner library, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval, t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.000061 to 0.0059, coverage 0.8961 to 0.9199, SE ratio 0.8976 | reported |
| `adaptive_learners` | `tmle_in_sample__nonlinear10__j40__wang_inflated` | diagnostic | in-sample TMLE, the default learner library, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.000061 to 0.0059, coverage 0.9134 to 0.9351, SE ratio 0.9209 | reported |
| `adaptive_learners` | `tmle_stacked__equal10__j10__iid_t_control` | control | stacked TMLE with five grouped folds, the default learner library, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the same point and t quantile with the IID row standard error | on the shown draws, the 99% upper end of the paired ratio of mean IID SE to mean cluster SE below 0.80, and the positive arm's shown-only SE-ratio upper end at most 1.20 | bias -0.0035 to 0.0087, coverage 0.8922 to 0.9163, SE ratio 0.7631 | pass |
| `adaptive_learners` | `tmle_stacked__equal10__j10__t_reference` | positive | stacked TMLE with five grouped folds, the default learner library, 10 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval, t with J - 2 degrees of freedom | the exact 99% lower end of all-draw coverage at least 0.90, bias inside 0.25 empirical SD, the 99% bootstrap lower end of the variance ratio at least 0.8649, and the exact 99% lower end of shown coverage at least 0.92, and the exact 99% lower end of the paired all-draw coverage difference to the GLM arm above -0.02 | bias -0.0035 to 0.0087, coverage 0.9324 to 0.9516, SE ratio 1.0344 | pass |
| `adaptive_learners` | `tmle_stacked__equal10__j10__wang_inflated` | diagnostic | stacked TMLE with five grouped folds, the default learner library, 10 clusters of 10 rows, a cluster latent in the outcome (L1): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0035 to 0.0087, coverage 0.9483 to 0.9651, SE ratio 1.1565 | reported |
| `adaptive_learners` | `tmle_stacked__equal10__j20__iid_t_control` | control | stacked TMLE with five grouped folds, the default learner library, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the same point and t quantile with the IID row standard error | on the shown draws, the 99% upper end of the paired ratio of mean IID SE to mean cluster SE below 0.80, and the positive arm's shown-only SE-ratio upper end at most 1.20 | bias -0.0041 to 0.0046, coverage 0.8539 to 0.8817, SE ratio 0.7385 | pass |
| `adaptive_learners` | `tmle_stacked__equal10__j20__t_reference` | positive | stacked TMLE with five grouped folds, the default learner library, 20 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval, t with J - 2 degrees of freedom | the exact 99% lower end of all-draw coverage at least 0.90, bias inside 0.25 empirical SD, the 99% bootstrap lower end of the variance ratio at least 0.8649, and the exact 99% lower end of shown coverage at least 0.92, and the exact 99% lower end of the paired all-draw coverage difference to the GLM arm above -0.02 | bias -0.0041 to 0.0046, coverage 0.9383 to 0.9566, SE ratio 1.0143 | pass |
| `adaptive_learners` | `tmle_stacked__equal10__j20__wang_inflated` | diagnostic | stacked TMLE with five grouped folds, the default learner library, 20 clusters of 10 rows, a cluster latent in the outcome (L1): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0041 to 0.0046, coverage 0.9494 to 0.9660, SE ratio 1.0691 | reported |
| `adaptive_learners` | `tmle_stacked__equal10__j40__iid_t_control` | control | stacked TMLE with five grouped folds, the default learner library, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the same point and t quantile with the IID row standard error | on the shown draws, the 99% upper end of the paired ratio of mean IID SE to mean cluster SE below 0.80, and the positive arm's shown-only SE-ratio upper end at most 1.20 | bias -0.0035 to 0.0026, coverage 0.8274 to 0.8573, SE ratio 0.7285 | pass |
| `adaptive_learners` | `tmle_stacked__equal10__j40__t_reference` | positive | stacked TMLE with five grouped folds, the default learner library, 40 clusters of 10 rows, a cluster latent in the outcome (L1): the shipped interval, t with J - 2 degrees of freedom | the exact 99% lower end of all-draw coverage at least 0.90, bias inside 0.25 empirical SD, the 99% bootstrap lower end of the variance ratio at least 0.8649, and the exact 99% lower end of shown coverage at least 0.92, and the exact 99% lower end of the paired all-draw coverage difference to the GLM arm above -0.02 | bias -0.0035 to 0.0026, coverage 0.9267 to 0.9467, SE ratio 1.0093 | pass |
| `adaptive_learners` | `tmle_stacked__equal10__j40__wang_inflated` | diagnostic | stacked TMLE with five grouped folds, the default learner library, 40 clusters of 10 rows, a cluster latent in the outcome (L1): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0035 to 0.0026, coverage 0.9402 to 0.9582, SE ratio 1.0355 | reported |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j10__iid_t_control` | control | stacked TMLE with five grouped folds, the default learner library, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the same point and t quantile with the IID row standard error | on the shown draws, the 99% upper end of the paired ratio of mean IID SE to mean cluster SE below 0.80, and the positive arm's shown-only SE-ratio upper end at most 1.20 | bias -0.0050 to 0.0076, coverage 0.8909 to 0.9152, SE ratio 0.7441 | pass |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j10__t_reference` | positive | stacked TMLE with five grouped folds, the default learner library, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval, t with J - 2 degrees of freedom | the exact 99% lower end of all-draw coverage at least 0.90, bias inside 0.25 empirical SD, the 99% bootstrap lower end of the variance ratio at least 0.8649, and the exact 99% lower end of shown coverage at least 0.92 | bias -0.0050 to 0.0076, coverage 0.9337 to 0.9527, SE ratio 1.0115 | pass |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j10__wang_inflated` | diagnostic | stacked TMLE with five grouped folds, the default learner library, 10 clusters of 10 rows, both nuisances nonlinear in W (L2): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0050 to 0.0076, coverage 0.9491 to 0.9657, SE ratio 1.1309 | reported |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j20__iid_t_control` | control | stacked TMLE with five grouped folds, the default learner library, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the same point and t quantile with the IID row standard error | on the shown draws, the 99% upper end of the paired ratio of mean IID SE to mean cluster SE below 0.80, and the positive arm's shown-only SE-ratio upper end at most 1.20 | bias -0.0018 to 0.0068, coverage 0.8599 to 0.8872, SE ratio 0.7409 | pass |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j20__t_reference` | positive | stacked TMLE with five grouped folds, the default learner library, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval, t with J - 2 degrees of freedom | the exact 99% lower end of all-draw coverage at least 0.90, bias inside 0.25 empirical SD, the 99% bootstrap lower end of the variance ratio at least 0.8649, and the exact 99% lower end of shown coverage at least 0.92 | bias -0.0018 to 0.0068, coverage 0.9432 to 0.9607, SE ratio 1.0227 | pass |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j20__wang_inflated` | diagnostic | stacked TMLE with five grouped folds, the default learner library, 20 clusters of 10 rows, both nuisances nonlinear in W (L2): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0018 to 0.0068, coverage 0.9554 to 0.9709, SE ratio 1.0780 | reported |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j40__iid_t_control` | control | stacked TMLE with five grouped folds, the default learner library, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the same point and t quantile with the IID row standard error | on the shown draws, the 99% upper end of the paired ratio of mean IID SE to mean cluster SE below 0.80, and the positive arm's shown-only SE-ratio upper end at most 1.20 | bias -0.0021 to 0.0039, coverage 0.8245 to 0.8546, SE ratio 0.7231 | pass |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j40__t_reference` | positive | stacked TMLE with five grouped folds, the default learner library, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): the shipped interval, t with J - 2 degrees of freedom | the exact 99% lower end of all-draw coverage at least 0.90, bias inside 0.25 empirical SD, the 99% bootstrap lower end of the variance ratio at least 0.8649, and the exact 99% lower end of shown coverage at least 0.92 | bias -0.0021 to 0.0039, coverage 0.9329 to 0.9520, SE ratio 1.0133 | pass |
| `adaptive_learners` | `tmle_stacked__nonlinear10__j40__wang_inflated` | diagnostic | stacked TMLE with five grouped folds, the default learner library, 40 clusters of 10 rows, both nuisances nonlinear in W (L2): Wang et al.'s inflation J / (J - 2) of the cluster variance on t with J - 2 degrees of freedom | none. The row is a diagnostic and states no verdict | bias -0.0021 to 0.0039, coverage 0.9426 to 0.9603, SE ratio 1.0396 | reported |
<!-- /generated -->

| clause | rows | what it reads |
| --- | --- | --- |
| coverage floor | `t_reference` of the gated fits | the exact 99% lower end of coverage over every requested draw, a withheld or lost draw a miss, at least 0.90 |
| bias equivalent | `t_reference` of the gated fits | the 99% interval of the bias inside 0.25 empirical SD |
| variance ratio lower end | `t_reference` of the gated fits | the 99% bootstrap lower end of the mean squared SE over the variance of the estimates, on shown fits, at least 0.8649 |
| shown coverage lower end | `t_reference` of the gated fits | the exact 99% lower end of shown-only coverage at least 0.92 |
| paired non-inferiority | stacked `t_reference` on L1 | the exact 99% lower end of the paired all-draw coverage difference to the GLM arm on the same draws above -0.02 |
| paired iid/cluster se ratio under its ceiling | `iid_t_control` of the gated fits | the exact 99% upper end of the paired IID-to-cluster SE ratio below 0.80 |
| paired positive se ratio at most the sanity ceiling | `iid_t_control` of the gated fits | the upper end of the positive's shown-only SE ratio at most 1.20 |
| must fail | `stacked_control` | the exact 99% upper end of all-draw coverage below 0.90 |

## The declared reading

The gated fits are stacked `TMLE` and cross-fitted `LTMLE`. A configuration keeps its row's label
at a law and count only where its positive row passes all of the clauses above, its IID control
passes and its must-fail control fails. The cross-fitted `LTMLE` cell reads the stacked control of
L1 at 20 clusters, because no control plants an overfit `LTMLE` outcome fit. Every stacked count of
both laws and the `LTMLE` cell pass, so no configuration takes the `adaptive_nuisance` shortfall
that the declaration reserved for a failing count.

The pilot of 400 draws per cell predicted three rows red: stacked L1 and L2 at 10 clusters and
stacked L2 at 40, each with a predictive pass probability below 0.50. At 10 clusters the pilot read
coverage 0.92 for the default library and for the stacked GLM arm on the same draws, a deficit
the two arms shared. None went red in the run. At 4,000 draws they cover 0.9425, 0.9437 and 0.9430,
and the stacked GLM arm on the same draws covers 0.9455, 0.9415 and 0.9385. The committed
exact-propensity GLM of `clustered-few-cluster-tmle` reads 0.935 at L1 and 10 clusters.

## Reported readings

These readings carry no verdict.

- In sample, the default library under-covers at every count: coverage 0.906 to 0.919 and a
  variance ratio of 0.72 to 0.88, 1.2 to 3.4 points under the in-sample GLM arm on the same draws.
  Each learner predicts from its fit to the full sample, so the residuals the interval reads
  shrink. This is why an in-sample fit with a data-adaptive learner keeps the experimental label
  (`cleverly._support.ADAPTIVE_IN_SAMPLE_REASON`).
- The interpolating in-sample control covers 0.48 to 0.58, with a variance ratio of 0.14 to 0.19.
- Wang et al.'s inflation $J/(J-2)$ on the gated fits covers 0.950 to 0.964 and is reported only.
- On L2 the GLM arm is a misspecified main-terms GLM. The cluster latent dominates the variance of
  the estimate there, so the paired comparison still isolates the library's effect on the interval.

## Red-cell route

The study is registered under `reporting`. The red-cell readings and the routes were declared
before the run in `tests/canonical/few_cluster_adaptive_learners/regenerate.py`. No row is red.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 200 | rows per primary replication |
| `independent_tests_passed` | 1 | truth tests passing |
| `independent_tests_total` | 1 | truth tests reported |
| `property_cells_passed` | 20 | gated property cells passing |
| `property_cells_total` | 20 | gated property cells reported |
| `max_standardized_bias` | 0.0070 | largest primary standardized bias |
| `min_coverage` | 0.9358 | lowest primary coverage |
| `properties[adaptive_learners/tmle_stacked__equal10__j10__t_reference]:coverage` | 0.9425 | a predicted red that passed |
| `properties[adaptive_learners/tmle_stacked__nonlinear10__j10__t_reference]:coverage` | 0.9437 | a predicted red that passed |
| `properties[adaptive_learners/tmle_stacked__nonlinear10__j40__t_reference]:coverage` | 0.9430 | a predicted red that passed |
| `properties[adaptive_learners/glm_stacked__equal10__j10__glm_reference]:coverage` | 0.9455 | the stacked GLM arm on the same draws at 10 clusters of L1 |
| `properties[adaptive_learners/tmle_stacked__equal10__j10__t_reference]:noninferiority_lower` | -0.0077 | paired coverage difference to the GLM arm, lower end, at 10 clusters of L1 |
| `properties[adaptive_learners/ltmle_crossfit__equal40__j20__t_reference]:coverage` | 0.9410 | cross-fitted `LTMLE` at 20 clusters |
| `properties[adaptive_learners/tmle_in_sample__nonlinear10__j10__t_reference]:coverage` | 0.9062 | the lowest in-sample coverage |
| `properties[adaptive_learners/tmle_in_sample__nonlinear10__j10__t_reference]:variance_ratio` | 0.7242 | the lowest in-sample variance ratio |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen, and the ceiling of each positive cell's paired SE ratio |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications of each cell's interval for a variance ratio or a mean difference |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound. No cell reads this margin; the variance-ratio floor, its square, is declared in the study module |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound. No cell reads this margin; the shown-coverage floor is declared in the study module with the same value |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell reads this margin; the paired non-inferiority margin against the GLM reference, -0.02, is declared in the study module |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 12 | the same bound, in primary replications |

## Limitations

- **Equal clusters and binary outcomes.** Clusters of 10 rows on two laws, and 40 units on one
  longitudinal law; a binary treatment and outcome, and no weights, strata or missing outcomes.
- **Three counts.** 10, 20 and 40 clusters for stacked `TMLE`, and 20 for cross-fitted `LTMLE`.
  The fold-evaluated fit, `DRTMLE` and `SDR` are not measured with these learners, nor `LTMLE`
  below 20 clusters, and each reads experimental.
- **One library.** The default library only. Another adaptive library is not measured, and Remark
  3 remains the source's caution for it.
- **No overfit `LTMLE` control.** The `LTMLE` cell reads the stacked point control of L1 at 20
  clusters.
- **No comparator.** No other implementation computes this interval with these learners.
