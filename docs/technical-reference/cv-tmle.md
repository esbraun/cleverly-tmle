# CV-TMLE and cross-fitting

## What this solves

You want to fit the nuisances with a flexible learner. A gradient-boosted outcome regression and a
random-forest propensity fit the data well, and an ordinary TMLE built on them can report an
interval that is too narrow. The reason is not overfitting in the usual sense. Classical theory
controls an empirical-process term with complexity conditions such as a Donsker condition. Rich,
adaptively tuned learners need not satisfy those conditions.

Cross-fitting avoids that empirical-process reliance under its remaining conditions. Every initial
nuisance prediction used for an observation comes from a model that never saw the observation.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| flexible learners for either nuisance | cross-fitting can avoid a Donsker restriction under its remaining conditions | one nuisance fit per outer fold; a Super Learner also fits its candidates on inner folds |
| you want the package default | cross-fitting is on by default, at ten outer folds | a Super Learner uses five additional inner folds unless configured otherwise. Ten by five is fifty model fits per library candidate |
| the fold draw itself worries you | `repeats=` runs a complete estimator per draw and aggregates | linear cost in the repeat count |
| clustered data | clusters stay intact in every split | a cluster-robust interval, because clusters, not rows, are the independent units. With fewer clusters than folds, the fold count drops to the cluster count, and a warning names both. Unequal cluster sizes and fewer than 40 clusters withhold the interval, as [clusters](inference.md#clusters) states |

**Cross-fitting does not buy the rest of efficiency.** Four conditions stand behind a valid
interval, and folds address one of them.

| condition | what supplies it |
| --- | --- |
| the empirical-process term is negligible | cross-fitting |
| the inverse mechanism stays controlled | support in the study design. The `g_bounds` truncation only regularises the fitted denominator |
| the estimated influence curve converges in $L_2$ | your learners |
| the second-order remainder is $o_P(n^{-1/2})$ by a **product rate** on both nuisances | your learners, and nothing the fluctuation can do |

The last condition is the one a *variant* of the estimator can weaken. That variant is
[DR-TMLE](dr-tmle/index.md).

A worked applied analysis is in the [CV-TMLE tutorial](../examples/cross-fitting.ipynb). It
measures what an in-sample interval costs under a flexible learner.

## The algorithm as implemented

Outer folds isolate nuisance training from prediction. Learner folds tune a model inside its outer
training data. The two layers are separate, and neither borrows the other's count.

`cleverly` ships targeting and evaluation choices over those folds. Their combinations are
different estimators.

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
| `split_plan=` | replaces generated outer assignments with a realized plan. See [reusable outer split plans](#reusable-outer-split-plans) for the counts it must match and the rows it is bound to | no |
| `learner_folds=` | model-selection folds inside an outer training set. Default 5. It reaches the Super Learner `cleverly` builds when you pass no learner. An explicitly supplied `SuperLearner` keeps its own `n_folds` | no |
| `repeats=` | repeats the outer split, runs a complete estimator per draw, and reports the median over draws with split-adjusted variance. The value must be at least 1, and a value above 1 requires `cross_fit=True`. The engine raises `ValueError` and `CrossFitting` raises `MethodConfigurationError`, both at construction | no. It is the same estimator over several draws |
| `stratify_folds="none"` | draws an unstratified, near-balanced outer partition from the seed alone. This is the default, and it is the only policy a fit that draws a split accepts | no |
| `stratify_folds="treatment"` or `"treatment+outcome"` | nothing. Every fit that draws a split refuses them, including selector-based C-TMLE at every setting. See [fold and outcome-scale rules](#fold-and-outcome-scale-rules) | no |
| `targeting_scheme="pooled"` | one targeting regression over the stacked validation rows. The default | this is stacked CV-TMLE |
| `targeting_scheme="fold"` | one fluctuation fit inside each validation fold | **yes**. It removes cross-fold coupling through the fluctuation fit. Python `zEpid` 0.9.1 corroborates this construction at two folds |
| `cv_evaluation=True` | fold plug-in evaluation with cross-validated variance | **yes**. This is fold-evaluated CV-TMLE |

**Two refusals under `cv_evaluation=True`.** A nonlinear fold aggregate has a fold-varying
gradient. Until a common targeting score for it is implemented, `rr`, `or`, and MSM coefficients
are refused rather than given an interval whose reported curve has a nonzero score.

**The repeat rule is median-only.** For draw-specific points $\hat\psi_r$ and variances
$\hat\sigma_r^2$, `cleverly` reports

$$
\widetilde\psi = \operatorname{median}_r(\hat\psi_r), \qquad
\widetilde\sigma^2 = \operatorname{median}_r\left\{
\hat\sigma_r^2 + (\hat\psi_r - \widetilde\psi)^2
\right\}.
$$

Risk ratios and odds ratios apply both operations on the log scale. This is the median rule in
Chernozhukov et al. (2018), equation (3.14), and the same calculation used by zEpid's repeated
cross-fit TMLE aggregator. zEpid corroborates this reporting layer, but it is not a full-estimator
comparator. It trains each nuisance on one partition and targets separately inside validation
partitions. `cleverly` retains its validated complement-trained, stacked pooled update. There is no
mean option.

Coordinatewise medians do not preserve identities among several estimands. Repeated fits therefore
refuse joint covariance and post-fit contrasts. They also refuse simultaneous bands because a
multiplier construction would use the retained central-draw curve. That curve supports marginal
diagnostics; the split-adjusted variance above supplies the pointwise interval.

### Missing-outcome natural-course mean

The binary missing-outcome `NaturalCourseMean` supports one stacked CV-TMLE configuration. It
uses one generated V-fold partition, pooled targeting, and whole-sample evaluation. Set
`stratify_folds="none"`, `repeats=1`, `targeting_scheme="pooled"`, `cv_evaluation=False`, and
`n_folds` of 2 or more. One fold is the in-sample estimator, so the fit refuses it. The refusal
stops that estimate from being reported under the stacked contract and its second-moment covariance
rule.

[Missing-outcome natural-course contracts](scope-and-refusals.md#missing-outcome-natural-course-contracts)
lists every refused composition. The paragraphs below give the reasons for the fold refusals.

The audited sources draw the partition externally. They do not support outcome-adaptive or
treatment-stratified partitions for this target, and no fit draws one. A supplied plan also
remains refused until its balance and weighting rules receive a separate audit.

Each nuisance fit uses a training complement and predicts its held-out rows. The outcome learner
fits respondents only. The response learner fits the response indicator. The targeting step then
uses one coefficient across all stacked held-out rows. The point and influence curve use those same
rows. No treatment mechanism is fitted or read.

Package-generated folds can differ in size by one row. Levy's exact finite-sample equality with the
original fold-evaluated estimator applies to equal-size folds. For fixed $V$ and bounded fold
estimates, the weighting difference is $O(1/n)$. It is first-order negligible, but it is not zero.

The original pooled, fold-evaluated construction has published support. It remains a separate
estimator because its fold plug-in and variance law differ. Bounded-continuous stacked outcomes also
remain refused until their scale transform has an exact contract. The
[roadmap](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants) tracks both follow-ups.

Fold-specific targeting and repeated-split reporting remain separate hard stops. The source audit
found no direct interval result for either composition. See
[F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants).

The registered
[stacked missing-outcome natural-course CV-TMLE study](method-evidence/stacked-missing-outcome-natural-course-cvtmle.md)
records the repeated-sampling evidence. It also compares the pooled target against the R `tmle`
2.1.1 population-mean path with the same stitched out-of-fold nuisance predictions.

### Missing-outcome arm-indexed means and contrasts

Arm-indexed means and contrasts with missing outcomes support one stacked CV-TMLE configuration.
It uses one generated unstratified V-fold partition, one pooled logistic fluctuation with one
coefficient for each arm, and whole-sample evaluation. The treatment can have two or more arms.
A continuous outcome needs a fixed `q_bounds` equal to its known support.

| setting | required value |
| --- | --- |
| `stratify_folds` | `"none"`, which is the default |
| `n_folds` | 2 or more |
| `repeats` | 1 |
| `targeting_scheme` | `"pooled"` |
| `cv_evaluation` | `False` |
| `split_plan` | `None` |
| `estimands` | from `ey`, `ey0`, `ey1`, `ate`, `rr`, and `or`. The default list of a two-arm fit includes `att` and `atc`, so name the estimands |

The fit refuses every other value before fold generation. The reasons follow the natural-course
section above. The audited sources draw the partition externally. The
[fold and outcome-scale rules](#fold-and-outcome-scale-rules) found no result for
treatment-stratified folds, and none for an outcome scale from held-out rows. Those two rules now
govern every cross-fitted fit, so this contract restates them rather than owning them. The
fold-evaluated construction, supplied plans, fold targeting, and repeated splits each need their
own result.

Unlike the natural-course mean, each arm-indexed estimate declares the centered covariance rule.
A simultaneous band over the reported estimands is therefore available.
[Stacked CV-TMLE for arm-indexed targets](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets)
gives the construction, the preflight, and the sources.
[Missing-outcome arm-indexed contract](scope-and-refusals.md#missing-outcome-arm-indexed-contract)
lists every refusal in its order. The registered
[stacked arm-indexed missing-outcome CV-TMLE study](method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md)
records the evidence.

## Fold and outcome-scale rules

A cross-fitted fit makes two choices before it fits a nuisance. It chooses how to split the rows.
For a continuous outcome it also chooses the scale the outcome regression works on.

A source audit on 2026-09-18 read the shipped estimators against the published results. The audit
found no reviewed result for a split that reads the treatment or the outcome. It found no reviewed
result for a scale taken from the observed sample. This section holds the audit record, and it
states the rules the package ships because of it.

The audit establishes no bias and no invalid coverage. It records what the sources do not cover.

### What the sources cover

Each row names a source, the passage this project read, and what that passage settles. The line
locators in the *verdict* column refer to the audit commit `4811661`.

| source | locator | verdict |
| --- | --- | --- |
| Zheng and van der Laan (2011), Working Paper 273 | Theorem 1, page 7, and Theorem 2, pages 14-15: "Bn is uniformly distributed over a finite support" | no stratification wording, and no split that depends on treatment or outcome values |
| Levy (2018) | Section 1, page 2: a "random split of 1, .., n" | no stratification wording |
| Chernozhukov et al. (2018), arXiv:1608.00060v7 | Definitions 3.1 and 3.2, pages 23-24: "Take a K-fold random partition" with folds of size $N/K$ | no stratification wording. A result for DML scores, and not for TMLE |
| Rafi (2023), arXiv:2305.08340v1 | Assumption 4.2, page 20: the folds of each treatment-by-stratum cell "depend only on" an independent uniform draw and the cell size | does not cover the package. It treats a cross-fitted AIPW estimator under covariate-adaptive randomization, with target proportions set by design, and with covariate strata rather than outcome strata |
| Gruber and van der Laan (2010), Working Paper 265 | Section 3, PDF page 9; Section 4, PDF pages 13-14 | data-derived bounds as a practice, with no derivation and no cross-fitting split |
| Gruber and van der Laan (2012) | Section 3.2, page 16 | warns about observed-range bounds under missing outcomes. No cross-fitting split |
| Smith et al. (2025) | Sections 2.1 and 6 | describes sample-range scaling. It studies binary outcomes only, and it calls for continuous-outcome research |
| Polley (2010) | Section 1.1.2, PDF page 16 (printed page 4), and Section 2.2, Theorem 1, PDF pages 24-25 (printed pages 12-13) | the Super Learner oracle result uses folds independent of its learning sample |
| Ju et al. (2019), read in the 2017 author preprint | Algorithm 1, PDF page 6; Section 5.5, PDF pages 10-11 | selects by cross-validated loss. It defines no treatment or outcome strata, and it does not describe the shipped inner selection-training folds |
| Benkeser, Cai and van der Laan (2020) | Section 3.1 and Appendix D | uses random near-balanced folds for variance estimation, and sketches CV-C-TMLE. Neither passage covers the shipped nested selection split, which the package draws inside each selection fold's training rows |
| Díaz, Williams, Hoffman and Schenck (2023) | Section 5.2, journal pages 852 and 853, and Theorem 3, page 853 | defines a random near-balanced row partition for a longitudinal TMLE, and one pooled all-row fluctuation per node after untargeted fold regressions. The shipped cross-fitted longitudinal fit follows both. The authors' `lmtp` 1.5.4 fits a training-fold fluctuation instead |

Every source draws its partition from outside the data. No source in the table stratifies a
partition on the treatment or the outcome. [References](../references.md) gives each entry in full.

### What the probes measured

Three probes ran at commit `1cf6628`. Each fits a complete-outcome ATE with ten folds and a
logistic treatment learner. They measure sensitivity. They do not measure coverage.

| probe | setup | result |
| --- | --- | --- |
| held-out scale | `make_nonlinear_ate(n=400, seed=11)`, `LinearRegression`, treatment strata; one outcome raised by five times the range | the scale moves from (-3.37, 10.39) to (-9.11, 73.49). In one validation fold, a `LinearRegression` trained with that row removed by hand moves its mean scaled validation prediction from 0.450 to 0.145 |
| scale only | the unmodified draw and the same folds; the scale fixed at the moved-row range against `q_bounds=None` | the ATE moves by -1.7e-4 (0.0009 SE) with `LinearRegression`, and by +3.2e-4 (0.0021 SE) with a random forest of 50 trees and `min_samples_leaf=5` |
| fold strata | one `make_linear_ate(n, seed=5)` draw at each size, 25 fold seeds, treatment strata against unstratified folds through a probe subclass | the standard deviation of the paired difference is 0.114 SE at n = 300 and 0.077 SE at n = 1200. The mean differences are -0.0035 and 0.0003 in outcome units, against Monte Carlo standard errors of 0.0027 and 0.0009, so both lie within two Monte Carlo standard errors of zero |

The first probe shows that a held-out outcome reaches a training fit through the scale. The second
shows that the effect on one point estimate is small at this law. The third finds no fold-policy
difference this design can resolve.

The registered stacked study now carries the same comparison at a declared budget. Its
[fold-policy diagnostics](method-evidence/stacked-point-treatment-cv-tmle.md) run three outer-split
policies on one binary law and one set of draws. Coverage is 0.9450 unstratified, 0.9500 under
treatment strata and 0.9475 under treatment-by-outcome strata. Each paired difference from the
unstratified arm straddles zero. The three cells report and gate nothing, because a difference
inside its own interval is equally consistent with both policies being right and with both being
wrong in the same direction.

### The rules the package ships

| surface | rule | where the fit applies it |
| --- | --- | --- |
| point-treatment and DR-TMLE outer folds | `random_partition` draws them from the row count, the cluster labels and a seed. The draw reads no treatment, outcome or covariate | `random_partition`, reached through `make_folds` from `TMLE._folds` |
| declared fold strata | `"none"` is the default, and it is the only policy a fit draws a split under. `"treatment"` and `"treatment+outcome"` are refused whenever the fit draws a split | `fold_strata_refusal`, called by `CrossFitting`, `TMLEMethod`, and the engine, and again at fit time by `TMLE._cross_fit_policy_reason` |
| selector-based C-TMLE selection and nested folds | the same unstratified draw, from the repeat's own seed. The strata refusal applies at every `cross_fit` setting, because these strategies draw the folds without cross-fitting. OAT draws none of them | `CTMLE._selection_partition`, `CTMLE._nested_partition`, `CollaborativeTMLEMethod` |
| longitudinal folds | `random_partition` draws them. There are no first-node treatment strata | `LTMLE._folds` |
| Super Learner inner folds | retained, and stratified on the learner's classification target inside the outer training rows | `SuperLearner` |
| outcome scale under cross-fitting | a declared `q_bounds`. A cross-fitted continuous outcome with `q_bounds=None` is refused before nuisance fitting | `TMLE._refuse_unbounded_cross_fitted_scale`, `LTMLE._refuse_cross_fitted_design` |
| `q_bounds` itself | the caller's known support, declared in advance. The package checks only that the interval holds the observed outcomes. No package or study code derives it from a realized sample | `cleverly.utils.bounds` |
| supplied outer plans | accepted only with a package generator record. The fit draws each repeat again and compares the labels | `SplitPlan.verify` |
| one fold with cross-fitting | refused at construction, and again at fit time | `_cross_fit_policy_refusal` |
| clusters | a grouped whole-cluster draw for point-treatment TMLE and DR-TMLE. C-TMLE refuses `id=`, and cross-fitted longitudinal TMLE refuses `id=` | `random_partition(cluster=...)`, `CTMLE._resolve_estimands_for_data`, `LTMLE._refuse_cross_fitted_design` |
| arm and class support | checked on the realized draw, before the first learner | `TMLE._preflight_training_support`, `CTMLE._preflight_selection_folds`, `preflight_mechanism_support` |

Stratification used to buy arm support, and `resolve_n_folds` capped the fold count to keep it. A
split that must not read the treatment cannot buy it. The package therefore checks the property on
the realized draw instead. No refusal raised after a draw names a redraw, a new seed or a different
fold count. A fold count that happens to fit was chosen by reading the values the split must not
read.

### The refusals a caller can meet

Each row gives the shipped remedy in the message's own words.

| what you declare | exception | the remedy the message names |
| --- | --- | --- |
| `stratify_by="treatment"` or `"treatment+outcome"` with cross-fitting | `MethodConfigurationError` from `CrossFitting` and `TMLEMethod`, `ValueError` from the engine | "Set stratify_folds='none' (CrossFitting(stratify_by='none')), which is the default. Otherwise fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False)), which draws no split for a policy to apply to." |
| the same policy on selector-based C-TMLE, at any `cross_fit` setting | `MethodConfigurationError`, or `ValueError` from the engine | the same first sentence, then "A collaborative fit draws those folds whether or not cross_fit is set, so cross_fit=False does not make this policy available." |
| a restored result or a copied estimator carrying a refused policy | `ValueError` at fit time | the reason above, then "This fit was configured under a fold policy this version refuses, which a restored result or a copied estimator can still carry" |
| cross-fitting with fewer than two folds | `MethodConfigurationError`, or `ValueError` from the engine | "Set n_folds to at least 2, or fit in sample with CrossFitting(enabled=False)" |
| a cross-fitted continuous outcome with `q_bounds=None` | `CapabilityError` | "Declare the known outcome support (Targeting(q_bounds=(lower, upper))). Without a known finite support, fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False))" |
| a cross-fitted shift, incremental, regime, MSM, or controlled-direct-effect fit with `delta=` | `CapabilityError` before the first learner | "To estimate them, fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False))". The message names the target family and F21, which holds the missing result. See [F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants) |
| a cross-fitted longitudinal continuous outcome with `q_bounds=None` | `LongitudinalError` | "Declare the known outcome support (Targeting(q_bounds=(lower, upper))), or fit in sample (CrossFitting(enabled=False), or n_folds=1 on the engine)." |
| `id=` on a collaborative fit | `CapabilityError` | "Drop id= from fit (PointTreatment(cluster=None)), or use the ordinary TMLE (TMLE, or TMLEMethod), which has a clustered result." |
| `id=` on a cross-fitted longitudinal fit | `LongitudinalError` | "Fit in sample (CrossFitting(enabled=False), or n_folds=1 on the engine), which is clustered and evidenced, or drop id= from fit." |
| a `split_plan=` with no generator record | `MethodConfigurationError` at construction, `DataError` at fit time | "Pass result.split_plan from a fit with unstratified folds (stratify_by='none'), or build the plan with SplitPlan.from_folds over random_partition draws" |
| a `split_plan=` whose labels the record does not draw | `DataError` before the first learner | "A fit accepts only the labels the recorded fold count and seed draw, because other labels could have been chosen by reading the outcome". The message names the repeat and the number of differing rows |
| a treatment arm held by fewer than two independent units | `DataError` before the first learner | "A split moves whole rows, so every partition leaves some training complement without that arm and no fold count or seed can fit the treatment mechanism; fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False))". Under `id=` the message says "clusters" in place of "rows" |
| an outcome class held by fewer than two independent units | `DataError` before the first learner | "A split moves whole rows, so every partition leaves some training complement without that class and no fold count or seed can fit the outcome regression; collect more observations with outcome 1". Under `id=` the message says "clusters" in place of "rows" |
| a drawn split whose training complement lacks an arm, a row with an observed outcome, an observed outcome class, or two rows of a Super Learner class | `DataError` before the first learner | "The split is drawn from the seed alone and reads no treatment or outcome, so trying fold counts or seeds until one fits would choose the partition by the values it must not read. Either fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False)), or collect more observations at the rare level." |
| a drawn longitudinal split that leaves a node level or a regimen unsupported | `LongitudinalError` before the first learner | "The split reads none of the data, so trying fold counts or seeds until one fits would choose the partition by the values it must not read. Fit in sample (CrossFitting(enabled=False), or n_folds=1 on the engine), or collect more observations at the rare level." The tail changes with the shortfall |
| a generated-outcome refutation on a fit that declares `q_bounds` | `CapabilityError` before any refit | "Leave q_bounds=None on the fit you refute (Targeting(q_bounds=None)). A cross-fitted continuous fit has to declare them, so refute a continuous outcome on a fit with cross_fit=False (CrossFitting(enabled=False))" |

`CrossFitting(n_folds=1)` is not the in-sample spelling. That declaration keeps `enabled=True`, so
it is refused at construction. Write `CrossFitting(enabled=False)`, which the engine receives as
`n_folds=1`.

Four refusals name this page in the message, as
`docs/technical-reference/cv-tmle.md, fold and outcome-scale rules`. They are the fold-policy
refusal, the general point-treatment outcome-scale refusal, the arm-indexed contract's own
outcome-scale refusal, and the longitudinal outcome-scale refusal. A reader who meets one in a
traceback can find the reason from the message alone.

### Grouped folds

A declared `id=` makes the cluster the independent unit. `random_partition` then permutes the
distinct cluster labels and cuts the permutation into near-equal parts, so every row of a cluster
lands in one fold. The number of clusters in two folds differs by at most one. The draw reads the
cluster labels and the seed, and it reads nothing else. This is the split
`GroupKFold(shuffle=True)` makes in scikit-learn 1.6 and later. The package owns the generator, so
the assignment does not change with the installed scikit-learn.

A second audit read the cluster-level sources against that composition. The table gives each
source, the passage read, and what it settles. Each row names the version whose section numbers
the audit used.

| source | locator | verdict |
| --- | --- | --- |
| Wang, Park, Small and Li (2024) | Section 4.2 and Theorem 4(b), read in the NIHMS author manuscript of the published article | a random, roughly equal partition of the clusters, with a cross-fitted result under it. The estimator is AIPW-type with a cluster-level treatment, and not a TMLE with a row-level treatment. Remark 5 defers treatment-balanced folds to Rafi (2023) |
| Chiang, Kato, Ma and Sasaki (2022) | Sections 3.1.2 and 3.2, Algorithm 1, Assumptions 1 and 3(i), and Theorem 1, read in the Taylor and Francis online-first PDF | a random, equal, data-independent partition of the cluster indices, for two-way clustering and linear Neyman-orthogonal DML scores. No stratification, no TMLE, and no one-way theorem. The online-first pages do not match the issue pages, so this project cites the sections and not the pages |
| Park and Kang, arXiv:2110.07740 | Section 3.3 and Supplement A.1, Theorem A.1 with condition (M1)', read in v3 | an AIPW or DML estimator under a random two-fold cluster split, with independent clusters of bounded size. It weights cluster averages, which equals row weighting only at equal cluster sizes. Not a TMLE, and unrefereed |
| Karim (2026), arXiv:2606.30918 | Section 3.3, condition (C2), and Theorems 1 and 2, read in the v2 main text | the closest estimator: a row-level TMLE with a pooled fluctuation. Its design has sampling strata rather than treatment strata, and (C2) excludes treatment strata. Unrefereed, and this project did not read the web appendix that holds the proofs |
| Benitez et al. (2023) | Section 3.2.1, read in the NIHMS author manuscript | supports the cluster-sum aggregation for a row-weighted estimand, and states that sample splitting "must respect the cluster as the independent unit". It gives no fold law and no theorem. Its Section 3.1.2 aggregates to the cluster level first, for a cluster-level estimand |
| Balzer, Zheng, van der Laan and Petersen (2019) | Section 3.1, Equation (9), and Section 4.2, Equations (20) and (21), read in the NIHMS author manuscript | a cluster-level exposure and a cluster-level estimand. No cross-fitting, and no split law |
| Balzer, van der Laan and Petersen (2016) | Sections 3.1, 4.1, 5.1 and 6, read in the NIHMS author manuscript | selection by cross-validation over independent units, where a row is a cluster, in a randomized trial with a fixed GLM library. No post-selection theorem, and no row-level C-TMLE |
| Balzer et al. (2023) | Sections 3.2 and 3.3 | a two-stage cluster-level estimator. No cross-fitting |
| Nugent et al. (2024) | Sections 2.1.3, 2.2 and 3 | no cross-fitting. Section 2.2 gives a $t$ reference below 40 clusters |
| Schnitzer, van der Laan, Moodie and Platt (2014) | Section 3.4.1, read in the arXiv reprint | a clustered longitudinal TMLE with a sandwich variance and no splitting. No theorem, and no cross-fitting |

No source covers treatment-stratified grouped folds for an observational estimator. No source
covers cross-fitted longitudinal TMLE with whole-cluster folds. The package refuses both.

**What supports the grouped split, and what does not.** One reviewed source covers the *partition*
and nothing more. Wang, Park, Small and Li (2024) partition the clusters at random into parts of
roughly equal size, and they prove a cross-fitted result under that partition. Their estimator is
AIPW-type with a cluster-level treatment. The source supports the split law. It does not prove this
package's estimator.

The rest is this package's own estimating-equation argument. It treats clusters as the independent
units, and it needs four conditions.

| condition | what it asks |
| --- | --- |
| independent clusters | one cluster's rows carry no information about another cluster's rows |
| equal cluster sizes | every cluster holds the same number of rows, so the row-weighted target equals the cluster-weighted one |
| no interference | one cluster's treatment does not change another cluster's outcome |
| remainder rates | the product rate on the two nuisances holds at the cluster level, as it does at the row level for iid data |

At equal cluster sizes the argument reduces to Zheng and van der Laan (2011), Theorem 2, with
clusters in place of rows. The registered
[clustered point-treatment CV-TMLE study](method-evidence/clustered-point-treatment-cv-tmle.md) is
the empirical witness, and it is the only one. Its design satisfies all four conditions by
construction. No source read here proves that the estimator is valid under clustering.

Row weighting and cluster weighting agree only at equal, or non-informative, cluster sizes. The
documented scope is equal cluster sizes. A weighted fit targets the weight-weighted mean, so on
that fit the size of a cluster is its row count and its weight mass. A cross-fitted fit whose
clusters differ in either takes the `"unequal_cluster_plugin"` status. `ci`, `pvalue`, and
`std_error` then raise `CapabilityError`, and `plugin_std_error` and `plugin_interval` keep the
diagnostic.

No source read here supports a normal reference interval with few clusters. The table gives what
each source recommends. $J$ is the cluster count, which Nugent et al. write as $N$.

| source | locator | recommendation |
| --- | --- | --- |
| Nugent et al. (2024) | Section 2.2, last paragraph, citing Hayes and Moulton (2009) | a $t$ reference with $J - 2$ degrees of freedom below 40 clusters |
| Benitez et al. (2023) | Section 3.1.2, paragraph on inference, and Section 3.2.1, last paragraph | a $t$ reference with $J - 2$ degrees of freedom at every cluster count, as a finite-sample approximation |

The package keeps its normal reference. A fit with fewer than 40 clusters takes the
`"few_cluster_plugin"` status, in sample or cross-fitted. So does a fit with fewer than 40 clusters
in one baseline stratum that it reports. [Clusters](inference.md#clusters) gives
both statuses, and [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle)
holds the routes that reopen them.

### The Super Learner inner split

The Super Learner keeps its inner split, and that split stratifies on the learner's classification
target. The inner folds see only the rows of one outer training complement. With an outer split
that reads no outcome, the inner assignment reads no outer-held-out outcome. The C-TMLE selection
folds cross the outer split, which [F18](../roadmap.md#f18-selector-path-c-tmle-inference) records.
Polley (2010) proves the Super Learner oracle result for folds drawn independently of the learning
sample, and not for these strata. The package therefore keeps the inner split as part of the
training algorithm, and it makes no oracle claim for it.

### Witnesses

`tests/unit/test_fold_policy_rules.py` holds the rules' witnesses. Each one has a mutation control
that patches a plausible wrong implementation into production and requires the witness to fail.

| claim | test class |
| --- | --- |
| generated outer folds do not move when the arms and the outcomes move | `TestTheOuterSplitReadsNeitherArmNorOutcome` |
| a refused policy reaches no learner, at construction and after a restore | `TestAPolicyThisVersionRefusesNeverReachesALearner` |
| a split that strands an arm reaches no learner, and the refusal names no redraw | `TestAStrandedArmIsRefusedBeforeAnyLearner` |
| an arm held by one cluster is refused, and counting rows instead misses it | `TestAnArmInsideOneClusterIsRefused` |
| a declared `q_bounds` keeps a held-out outcome out of its own fold's predictions | `TestADeclaredSupportKeepsTheScaleOutOfTheFolds` |
| flipping a held-out outcome moves no inner Super Learner split | `TestTheInnerSplitDoesNotReadHeldOutRows` |
| the C-TMLE selection split reads no arm, and records its scheme without cross-fitting | `TestTheSelectionSplitReadsNoArm` |
| the collaborative strata and cluster refusals fire, and the method catalog reports them | `TestTheCollaborativeRefusals` |
| the longitudinal split does not move when the first-node arms move | `TestTheLongitudinalSplitReadsNoTreatment` |
| the longitudinal support and cluster refusals fire, and the same data fit in sample | `TestTheLongitudinalRefusals` |
| a bootstrap replicate the preflight refuses is dropped, and fits no learner | `TestTheBootstrapDropsWhatThePreflightRefuses` |

Three further witnesses live with the surfaces they check.

| claim | test |
| --- | --- |
| `random_partition` reproduces `KFold(shuffle=True)` and `GroupKFold(shuffle=True)`, and the grouped draw reads only the labels and the seed | `tests/unit/test_learners.py::TestTheOuterFoldGenerator` |
| only a recorded draw is accepted as a supplied plan, and a forged label is refused before any learner | `tests/unit/test_split_plan.py::TestOnlyARecordedDrawIsAccepted` |
| every registered study seed stays inside the generator's range | `tests/unit/test_fold_policy_rules.py::test_every_registered_fold_seed_stays_inside_the_generators_range` |

Eleven point-treatment and five longitudinal registered studies were regenerated under these rules.
The [validation grid](method-evidence/validation-grid.md) carries their verdicts.

## Reusable outer split plans

Every point-treatment result exposes its realized outer assignments as `result.split_plan`. Pass
them to `CrossFitting(split_plan=...)` to reuse them instead of generating new ones. A fit accepts
a plan only when the plan records the draw that made it. This section states the reuse contract. The [user guide](../user-guide/methods-learners.md#reuse-an-outer-split)
and the [tutorial](../examples/cross-fitting.ipynb#step-9-reuse-the-same-outer-split) link here rather than
restating it.

A `SplitPlan` stores one tuple of row-level fold labels per repeat. Its `n`, `n_folds`, and
`n_repeats` properties describe those assignments. Its `provenance` holds one `FoldOrigin` per
repeat, which records the generator, the scheme, the requested fold count, and the seed of that
repeat's draw.

**A fit accepts only labels the recorded generator draws.** `SplitPlan.verify` draws each repeat
again from its record and compares the labels one by one. It runs before the first learner. Labels
that no recorded draw produces could have been chosen by reading the outcome, the treatment, or a
covariate, and a fit refuses them. The refusal names the repeat.

The check has one limit, and the limit is the caller's declaration. The record holds the fold count
and the seed that the caller gave, so `verify` accepts any labels that pair draws. It rules out a
hand-built assignment, a stratified one, and one edited after the draw. It does not rule out a seed
the caller chose after reading the data, because the package has nothing local to audit a
declaration against. Pass `result.split_plan` from an earlier fit, which is the source this
contract is written for.

| the plan | what it records | a fit |
| --- | --- | --- |
| `SplitPlan.from_folds(...)` over `random_partition` draws | one `FoldOrigin` per draw | accepts it |
| `result.split_plan` of a fit whose folds `random_partition` drew | one `FoldOrigin` per repeat | accepts it |
| `result.split_plan` of an in-sample fit, or of a restored fit with stratified folds | nothing | refuses it |
| `SplitPlan(labels)` built from labels alone | nothing | refuses it |

The record also fixes the scheme. A draw is `"grouped"` when the data declare clusters, and
`"vfold"` when they do not. A fit refuses the other scheme, because a row-level draw cuts across
clusters and a grouped draw needs the cluster labels it split. The draw reads the row count, the
cluster labels, and the seed, and it reads no other column.

**A fold label is a row position.** It is not a pandas index label, and it is not Polars row
metadata. A plan is therefore meaningful only for the rows it was realized on. Reorder those rows,
and every label points at a different unit while the row count still agrees.

A plan read off a result carries the data fingerprint of the fit that produced it, under
`source_fingerprint`. Validation compares that value with the fingerprint of the data in hand. It
refuses a plan whose fingerprint differs. A plan you build by hand carries `None` and binds to no
data. Call `plan.unbound()` to reuse the labels on other rows deliberately. The method drops the
binding and keeps the generator record, so `verify` still draws the labels again.

A clustered design applies one further rule. Every row from one cluster must receive the same fold
label within each repeat.

**A cap makes the declared fold count and the realized fold count differ.** The resolver caps the
count at the rarest stratum, and again at the cluster count. The declaration is therefore an upper
bound on the plan, and it is the only rule that reads a fold count.

| when | what it checks | what it refuses |
| --- | --- | --- |
| you construct `CrossFitting` | that the plan records a generator, and the plan's fold count against the declared `n_folds` | a plan with no record, and a plan holding more folds than the declaration. No cap produces that direction |
| before nuisance fitting, first | each repeat against the split its record draws again | a plan whose labels, scheme, or row count the record does not produce |
| before nuisance fitting, then | the plan's labels against the data in hand | a plan whose folds cannot serve these rows |

So `n_folds=10` on data carrying four clusters realizes a four-fold plan, and `n_folds=10` accepts
that plan back. The fit still records the declared 10, under `config.crossfit.n_folds`. The fit
resolves no fold count for a supplied plan, because it generates no split. A rare stratum must
reach every training complement, and it need not appear once in every fold. The `repeats` count
must equal the plan's repeat count exactly.

The plan controls only the outer nuisance split. Inner Super Learner folds still follow
`learner_folds`. Collaborative TMLE selection folds still follow their repeat-specific seeds.

Verification and validation both happen before nuisance fitting. Verification checks the record.
Validation checks row count, repeat count, label contiguity, required training arms, requested
stratification, and whole-cluster assignment. The implementation
rejects a supplied plan rather than repairing an invalid one. The `SplitPlan.validate` method names
the balancing vector `stratify=`, as `make_folds` does, because `strata` is the survey design role.

A fit records `scheme="supplied"` for the fact that nothing was generated. It records
`config.crossfit.stratify_by` empty, because this version holds the folds to no balancing vector.

[Scope and refusals](scope-and-refusals.md) indexes what a supplied plan refuses. The refutation
battery is the entry worth reading here. A refutation refits, and a supplied plan can only label
the rows it was realized on.

| refutation | what its refit does to the rows | under a supplied plan |
| --- | --- | --- |
| `subset` | drops rows | refused |
| `bootstrap_measurement_error` | draws rows with replacement | refused |
| `placebo` | replaces the treatment column | runs |
| `random_common_cause` | adds a covariate column | runs |

`refute()` raises `CapabilityError` for the two refused operations before it refits anything.
The refusal reads the requested operation rather than the row count. A bootstrap draw holds the
declared number of rows, so a count check cannot see it. A battery run under
`assess_result(..., include_refits=True)` reports the refusal as an `unavailable` row and runs the
rest of the battery.

A result from `cross_fit=False` still records a one-fold plan. Passing that plan back through
`split_plan=` is refused. `random_partition` draws two folds at least, so a one-fold plan records
no draw.

Point-treatment bootstrap samples do not preserve the original positional unit sequence. Targeted
bootstrap inference therefore refuses a supplied plan. Longitudinal estimation also refuses it
because its sequential fold contract is separate.

Result serialization preserves the plan. The `SplitPlan.fingerprint` property and the result
provenance read one shared fold digest, so a reused plan and its fit compare directly.
Generated-versus-reused acceptance tests require exact fold, nuisance-prediction, point-estimate,
and influence-curve identity.

Those checks establish deterministic reuse. They do not supply a new estimator or a statistical
guarantee. The registered studies below remain the applicable estimator evidence. Reusing a plan
adds no validation-grid row because it changes neither the estimator nor its statistical
assumptions.

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

**Eight registered studies, and no study's result stands in for another's.** Ordinary TMLE, stacked
CV-TMLE, clustered CV-TMLE, pooled-targeted fold-evaluated CV-TMLE, fold-targeted CV-TMLE,
repeated stacked CV-TMLE, stacked missing-outcome natural-course CV-TMLE, and stacked arm-indexed
missing-outcome CV-TMLE can share a limit while differing in finite samples. Each has its own row.

| where to read the evidence | what is there |
| --- | --- |
| [stacked point-treatment CV-TMLE](method-evidence/stacked-point-treatment-cv-tmle.md) | paired against R `tmle3` CV-TMLE on **identical realized folds**, plus flexible-learner cross-fit versus in-sample controls |
| [clustered point-treatment CV-TMLE](method-evidence/clustered-point-treatment-cv-tmle.md) | paired against pinned R `lmtp` and `ife` on identical grouped folds, with cluster-calibration evidence and an IID variance control |
| [fold-evaluated point-treatment CV-TMLE](method-evidence/fold-evaluated-point-treatment-cv-tmle.md) | no comparator pairs a pooled update with fold evaluation, so the study records a zero-row equivalence artifact and rests on accuracy against known truth and on the theory properties |
| [fold-targeted point-treatment CV-TMLE](method-evidence/fold-targeted-point-treatment-cv-tmle.md) | paired against Python `zEpid` on identical equal two-fold assignments, where `cleverly`'s equal $1/V$ aggregation equals zEpid's size weighting; all property cells pass, and the report discloses zEpid's `ddof=1` fold variance |
| [repeated point-treatment CV-TMLE](method-evidence/repeated-cross-fitting.md) | exact-truth and repeated-sampling evidence for the median report |
| [stacked missing-outcome natural-course CV-TMLE](method-evidence/stacked-missing-outcome-natural-course-cvtmle.md) | paired against the R `tmle` 2.1.1 population-mean path with the same stitched out-of-fold nuisance predictions, plus flexible-learner cross-fit versus in-sample controls |
| [stacked arm-indexed missing-outcome CV-TMLE](method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md) | two-arm and three-arm means and contrasts, binary and bounded continuous, paired against R `tmle` 2.1.1 with the same stitched out-of-fold nuisance predictions, plus calibration, band, union-model, and overfitting cells |
| [the implementation validation grid](method-evidence/validation-grid.md) | all rows, with their declared limits |

The fold-evaluated row is worth reading for what it is *not*. It is not parity evidence for stacked
R CV-TMLE. No maintained package pairs a pooled update with fold evaluation. Python `zEpid`
targets and evaluates inside each fold. The separate fold-targeted row compares that estimator at
two folds, where zEpid nuisance training is the complete validation-fold complement. zEpid's mean
over stacked targeted rows size-weights folds; it equals `cleverly`'s $1/V$ fold average here only
because those folds are equal.
