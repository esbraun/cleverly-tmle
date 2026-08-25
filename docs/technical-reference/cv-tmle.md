# CV-TMLE and cross-fitting

## What this solves

You want to fit the nuisances with a flexible learner. A gradient-boosted outcome regression and a
random-forest propensity fit the data well, and an ordinary TMLE built on them can report an
interval that is too narrow. The reason is not overfitting in the usual sense. It is that the
theory behind the interval assumes the nuisance estimators come from a class that is not too rich,
and a modern learner does not satisfy that assumption.

Cross-fitting removes that assumption. Every nuisance prediction used for an observation is
produced by a model that never saw the observation.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| flexible learners for either nuisance | the empirical-process term is controlled without a Donsker condition on the nuisance estimators | one nuisance fit per fold, multiplied by the learner library |
| you want the package default | cross-fitting is on by default, at ten outer folds and five learner folds | the two fold layers multiply. Ten by five is fifty model fits per library candidate |
| the fold draw itself worries you | `repeats=` runs a complete estimator per draw and aggregates | linear cost in the repeat count |
| clustered data | clusters stay intact in every split | fewer effective folds than rows suggest |

**Cross-fitting does not buy the rest of efficiency.** Four conditions stand behind a valid
interval, and folds address one of them.

| condition | what supplies it |
| --- | --- |
| the empirical-process term is negligible | cross-fitting |
| positivity bounds the clever covariate | the `g_bounds` truncation, and your design |
| the estimated influence curve converges in $L_2$ | your learners |
| the second-order remainder is $o_P(n^{-1/2})$ by a **product rate** on both nuisances | your learners, and nothing the fluctuation can do |

The last condition is the one a *variant* of the estimator can weaken. That variant is
[DR-TMLE](dr-tmle/index.md).

A worked applied analysis is in the [CV-TMLE tutorial](../examples/cross-fitting.md). It
measures what an in-sample interval costs under a flexible learner.

## The algorithm as implemented

Outer folds isolate nuisance training from prediction. Learner folds tune a model inside its outer
training data. The two layers are separate, and neither borrows the other's count.

`cleverly` ships two constructions over those folds, and they are different estimators.

**Stacked CV-TMLE is the default.** It stacks all out-of-fold predictions and fits one targeting
regression over the validation rows. It then evaluates the plug-in parameter on the whole sample.
Zheng and van der Laan (2011) supplies the original cross-validated TMLE framework. Levy (2018)
identifies this stacked construction. The pinned `tmle3` `cvtmle=TRUE` source and its `sl3`
fold-and-full prediction dependency corroborate the engineering semantics.

Conditional on the training-fold fits, each fold's outcome regression is fixed, and the fluctuated
family is indexed only by a fixed finite-dimensional coefficient over a compact set. The validation
outcomes do fit that coefficient. Sample splitting makes the *initial nuisances* out of fold, not
the targeted predictions.

**Fold evaluation is the original construction.** It averages the fold plug-ins with weight $1/V$.
Its validation risks and observation weights are normalised inside the fold to match that
aggregation. The variance is

$$
V^{-2} \sum_v n_v^{-2} \sum_{i \in v} D_{v,i}^2 ,
$$

and not a fold-averaged second moment divided by the total $n$. The two coincide only for exactly
equal folds.

Implementation:
[`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py),
[`learners/crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/crossfit.py),
and
[`learners/_fitting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/_fitting.py).

## Variations

| option | what it changes | is it a different estimator? |
| --- | --- | --- |
| `cross_fit=False` | one fold, no splitting. This is ordinary TMLE | **yes**. See [point-treatment TMLE](point-treatment-tmle.md) |
| `n_folds=` | the outer split count. Default 10 | no |
| `learner_folds=` | model-selection folds inside an outer training set. Default 5 | no |
| `repeats=` | repeats the outer split, runs a complete estimator per draw, averages the estimates, and aggregates the influence curves elementwise | no. It is the same estimator over several draws |
| `stratify_folds=` | `"treatment"`, or `"treatment+outcome"` for a rare binary outcome | no. Refused on a continuous outcome or dose |
| `targeting_scheme="pooled"` | one targeting regression over the stacked validation rows. The default | this is stacked CV-TMLE |
| `targeting_scheme="fold"` | one coefficient fitted inside each validation fold | **yes**. It is a package extension. It removes cross-fold coupling through the coefficient, and a row still contributes to the coefficient used on its own fold. Only the common update is attributed to the cited literature |
| `cv_evaluation=True` | fold plug-in evaluation with cross-validated variance | **yes**. This is fold-evaluated CV-TMLE |

**Two refusals under `cv_evaluation=True`.** A nonlinear fold aggregate has a fold-varying
gradient. Until a common targeting score for it is implemented, `rr`, `or`, and MSM coefficients
are refused rather than given an interval whose reported curve has a nonzero score.

**Two refusals for aggregation over `repeats=`.** Median-of-estimates aggregation is refused,
because the median of the estimates is not the estimator whose curve is the median of the curves. A
cross-validated variance of the across-draw average curve is refused, because at equal fold sizes
it collapses to the pooled uncentred second moment for every partition.

## Validation issues special to this method

**Leakage is checked without a tolerance.** `tests/unit/test_crossfit_leakage.py` rigs a law in
which one covariate is constant inside a cluster and the outcome *is* that covariate with no noise.
A nearest-neighbour learner then reproduces a held-out row bit for bit if and only if a same-cluster
row was in its training set. The assertions are array equality and array inequality. There is no
tolerance and no seed sensitivity, so leakage is not a matter of degree.

**Fold integrity is refused rather than repaired.** An externally supplied fold assignment that
splits a declared cluster is rejected. Buying more folds that way shrinks the standard error in
exactly the direction the cluster role was declared to prevent.

**Serial and parallel runs must agree exactly.**
`tests/unit/test_parallel_invariance.py` pins that, because a fold-parallel implementation that
reseeds per worker would give a different answer at a different `n_jobs`.

**Three registered studies, and none of them inherits another's result.** Ordinary TMLE, stacked
CV-TMLE, and fold-evaluated CV-TMLE may share a limit while differing in finite samples. Each has
its own row.

| where to read the evidence | what is there |
| --- | --- |
| [stacked point-treatment CV-TMLE](method-evidence/stacked-point-treatment-cv-tmle.md) | paired against R `tmle3` CV-TMLE on **identical realized folds**, plus flexible-learner cross-fit versus in-sample controls |
| [fold-evaluated point-treatment CV-TMLE](method-evidence/fold-evaluated-point-treatment-cv-tmle.md) | no canonical comparator exists, so the study records a zero-row equivalence artifact and rests on accuracy against known truth and on the theory properties |
| [the implementation validation grid](method-evidence/validation-grid.md) | both rows, with their declared limits |

The fold-evaluated row is worth reading for what it is *not*. It is not parity evidence for stacked
R CV-TMLE. No maintained package ships this construction, and a study that had no comparator says
so in its own cell rather than borrowing a surrogate.
