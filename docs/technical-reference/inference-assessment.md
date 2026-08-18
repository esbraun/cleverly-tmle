# Inference, diagnostics, and sensitivity

## Influence-curve inference

For an asymptotically linear estimate with estimated influence values $D_i$, `cleverly` uses

$$
\widehat{\operatorname{Var}}(\hat\psi)=\frac{1}{n^2}\sum_{i=1}^n D_i^2
$$

for independent observations, with the corresponding centered covariance matrix for multiple
parameters. Cluster-robust inference first sums weighted influence values within each independent
cluster and uses cluster-level finite-sample scaling. Smooth ratios, odds ratios, and user
contrasts propagate the joint influence curve by the delta method.

Implementation: [`inference/influence.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/influence.py),
[`inference/delta.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/delta.py),
and [`inference/cluster.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/cluster.py).
Unit tests pin exact covariance identities, weighted effective sample size, cluster aggregation,
repeated-cross-fit covariance, and transformations.

## Simultaneous intervals and bootstrap

Multiplier bootstrap draws from the joint influence matrix to estimate a familywise critical
value for simultaneous confidence bands. Ordinary resampling and cluster resampling preserve the
declared independent unit. Bootstrap configuration is refused for engines that cannot implement
it rather than accepted and discarded.

Implementation: [`inference/multiplier.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/multiplier.py)
and [`inference/bootstrap.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/bootstrap.py).
The multiple-testing reference is [Benjamini & Hochberg (1995)](../references.md#multiple-testing)
for FDR-adjusted reporting; simultaneous bands have their own multiplier critical-value tests.

## Cross-fitting and repeated splits

Outer folds isolate nuisance training from prediction. Learner folds tune a nuisance model within
its outer training data. Repeats vary the primary split, run a complete estimator per draw, average
the estimates, and aggregate influence curves elementwise. Clusters stay intact in every split.

Implementation: [`learners/crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/crossfit.py)
and [`learners/_fitting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/_fitting.py).
Evidence includes leakage sentinels, fold-integrity refusals, deterministic random-state behavior,
serial/parallel equivalence, and repeated-cross-fit identities.

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
Assessment-contract, score, nuisance, support, cache, serialization, and point/longitudinal facade
tests enforce these semantics.

## Sensitivity analysis

The point-treatment sensitivity suite includes:

- positivity summaries and truncation curves;
- omitted-variable bounds based on Chernozhukov et al. (2022);
- E-values based on VanderWeele & Ding (2017);
- missingness tilts based on Scharfstein, Rotnitzky & Robins (1999);
- explicit refit-based benchmark operations.

See [sensitivity references](../references.md#sensitivity-analysis). The implementation lives in
[`sensitivity/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/sensitivity).
Unit tests pin units and scales, target applicability, multi-arm selection, refit boundaries, and
serialization. A point-treatment formula is not reused for longitudinal data without a published
derivation; the facade reports it unavailable.

## Variable importance

`variable_importance` summarizes target-relevant changes under supported refits and returns a
typed `VariableImportanceResult`. It is an assessment of the fitted causal workflow, not a generic
predictive feature-importance score. Implementation is in
[`variable_importance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/variable_importance.py)
with dedicated unit tests for estimates, intervals, and configuration propagation.
