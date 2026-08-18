# Inference, diagnostics, and sensitivity

## Influence-curve inference

For an asymptotically linear estimate with estimated influence values $D_i$, `cleverly` uses

$$
\widehat{\operatorname{Var}}(\hat\psi)=\frac{1}{n}\cdot\frac{1}{n-1}\sum_{i=1}^n(D_i-\bar D)^2
$$

for independent observations, with the corresponding covariance matrix for multiple parameters.
The curve is centered rather than assumed to be centered: targeting drives $\bar D$ to
approximately zero, and reading the mean off the sample instead of substituting zero is what makes
the reported variance a statement about the curve that was actually computed.

With $m$ independent clusters, the influence values are summed within each cluster and

$$
\widehat{\operatorname{Var}}(\hat\psi)=\frac{m}{n^2}\cdot\frac{1}{m-1}\sum_{c=1}^m(S_c-\bar S)^2,
\qquad S_c=\sum_{i\in c}D_i,
$$

so the independent unit is the cluster and not the row. Smooth ratios, odds ratios, and user
contrasts propagate the joint influence curve by the delta method.

Implementation: [`inference/influence.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/influence.py),
[`inference/delta.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/delta.py),
and [`inference/cluster.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/cluster.py).
Evidence: [`tests/unit/test_inference.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_inference.py)
pins exact covariance identities, weighted effective sample size, cluster aggregation, and the
delta-method transformations; repeated-cross-fit covariance is in
[`tests/unit/test_repeated_crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_repeated_crossfit.py).

## Simultaneous intervals and bootstrap

Multiplier bootstrap draws from the joint influence matrix to estimate a familywise critical
value for simultaneous confidence bands. Ordinary resampling and cluster resampling preserve the
declared independent unit. Bootstrap configuration is refused for engines that cannot implement
it rather than accepted and discarded.

Implementation: [`inference/multiplier.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/multiplier.py)
and [`inference/bootstrap.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/bootstrap.py).
The multiple-testing reference is [Benjamini & Hochberg (1995)](../references.md#multiple-testing)
for FDR-adjusted reporting. Evidence: the multiplier critical-value, simultaneous-band and
resampling cases in [`tests/unit/test_inference.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_inference.py).

## Cross-fitting and repeated splits

Outer folds isolate nuisance training from prediction. Learner folds tune a nuisance model within
its outer training data. Repeats vary the primary split, run a complete estimator per draw, average
the estimates, and aggregate influence curves elementwise. Clusters stay intact in every split.

Implementation: [`learners/crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/crossfit.py)
and [`learners/_fitting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/_fitting.py).
Evidence: leakage sentinels and fold-integrity refusals in
[`tests/unit/test_crossfit_leakage.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_crossfit_leakage.py), repeated-split
identities in [`tests/unit/test_repeated_crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_repeated_crossfit.py),
and serial/parallel equivalence in
[`tests/unit/test_parallel_invariance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_parallel_invariance.py).

## Diagnostics and validation

The post-fit assessment facade declares required artifacts, cost, execution mode, replayability,
and method-specific interpretation for every operation. Cache-only operations include support,
nuisance performance, targeting scores, and validation summaries. Refutation and benchmark
operations that fit new models are explicit.

Status is part of the contract: `not_applicable` means no such analysis exists for the question;
`unavailable` means it is meaningful but a derivation or fitted artifact is missing. Persistence
preserves completed reports and replayability metadata.

Implementation: [`assessment.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/assessment.py),
[`validation/api.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/api.py),
and [`validation/score.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/score.py).
Evidence: the status and artifact contract in
[`tests/unit/test_assessment_contract.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_assessment_contract.py), and
persistence, cached-report survival and replayability in
[`tests/unit/test_serialization.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_serialization.py).

## Sensitivity analysis

The point-treatment sensitivity suite includes:

- positivity summaries and truncation curves;
- omitted-variable bounds based on Chernozhukov et al. (2022);
- E-values based on VanderWeele & Ding (2017);
- missingness tilts based on Scharfstein, Rotnitzky & Robins (1999);
- explicit refit-based benchmark operations.

See [sensitivity references](../references.md#sensitivity-analysis). The implementation lives in
[`sensitivity/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/sensitivity).
Evidence: units, scales, target applicability and refit boundaries in
[`tests/unit/test_sensitivity_units.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_sensitivity_units.py), and arm
selection in [`tests/unit/test_sensitivity_multi_arm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_sensitivity_multi_arm.py).
A point-treatment formula is not reused for longitudinal data without a published derivation; the
facade reports it unavailable.

## Variable importance

`variable_importance` summarizes target-relevant changes under supported refits and returns a
typed `VariableImportanceResult`. It is an assessment of the fitted causal workflow, not a generic
predictive feature-importance score. Implementation is in
[`variable_importance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/variable_importance.py)
with estimates, intervals and configuration propagation covered by
[`tests/unit/test_variable_importance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_variable_importance.py).
