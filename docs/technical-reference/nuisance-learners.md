# Nuisance learners

Every estimator in `cleverly` starts from fitted nuisance functions: an outcome regression, a
treatment mechanism and, where the design has them, missingness, censoring, intermediate and
hazard models. This page states how the learners in `src/cleverly/learners/` build those fits,
which checks exist for them, and what none of the checks establishes. How the rows of each fit are
chosen is the subject of [CV-TMLE and cross-fitting](cv-tmle.md). The user guide's
[methods and learners](../user-guide/methods-learners.md) page shows how to pass a learner.

## What a learner must provide

A nuisance slot takes a scikit-learn estimator object with `fit` and either `predict` or
`predict_proba`. A string such as `"glm"` is refused at configuration time, so the configuration
always names the model that is fitted (`_validate_learner` in
`src/cleverly/learners/library.py`; `tests/unit/test_learner_object_contract.py` checks every slot
of `ModelSpec`, `TMLE`, `LTMLE` and `DRTMLE`, and a string library). A learner whose `fit` names
`sample_weight` or takes `**kwargs` receives the fit's observation weights. A learner that takes
neither is fitted unweighted: with a `UserWarning` when the estimator fits it, and silently inside
a Super Learner's cross-validation. The cluster codes reach only a `fit` that names `groups`
(`fit_learner`, `supports_sample_weight` and `accepts_groups` in
`src/cleverly/learners/_fitting.py`).

When an estimator carries a seed, every `random_state` of the learner that is `None`, at any depth
of `get_params(deep=True)`, takes that seed on a clone, and an explicit seed is kept
(`seed_unset` in `src/cleverly/learners/_fitting.py`). A seeded fit is then a function of its data
and its seed. `tests/unit/test_learner_seeding.py` checks it under two global seeds, after showing
that the global seed does move an unseeded SAGA fit.

## The default learner

`resolve_learner` (`src/cleverly/learners/super_learner.py`) turns an empty slot with no fallback
into `SuperLearner(task=task, n_folds=learner_folds, clip=(0.0, 1.0), random_state=seed, n_jobs=1)`
over the default library. `learner_folds` is an estimator argument, 5 by default, and the
adaptive TMLE passes its own count (`inner_folds` of `fit_adaptive_tmle`). A `SuperLearner` that
the caller constructs keeps its own `n_folds`, 10 unless set. Several slots never reach that
default.

| empty slot | what it fits | code |
| --- | --- | --- |
| missingness and intermediate (`TMLE`); censoring (`LTMLE`, `SDR`, `IPSI`) | the treatment learner, and the default ensemble only when the treatment slot is empty too | `TMLE._plan_learners` in `src/cleverly/estimators/tmle.py`; `_SequentialEstimator` in `src/cleverly/longitudinal/estimator.py` |
| an eliminated competing event | the censoring learner | `_SequentialEstimator` |
| the pseudo-outcome regression of the earlier nodes | the outcome learner | `_SequentialEstimator._pseudo_learner` |
| the `DRTMLE` reduced regressions | the outcome learner for the reduced outcome and the mechanism residual, the treatment learner for the reduced treatment | `src/cleverly/estimators/drtmle.py` |
| the `SDR` and `IPSI` pseudo-outcome regression, with the outcome slot empty too | the default ensemble with no clip, because a transformed pseudo-outcome leaves $[0, 1]$ | `SDR._pseudo_learner`, `IPSI._pseudo_learner` |
| a regression of `StructuralNestedMean` | the outcome learner, and when that slot is empty too, the default ensemble with no clip; its classifications keep the clip | `src/cleverly/longitudinal/structural.py` |
| every learner of `SequentialGComputation`, `SequentialIPW` and `ParametricGFormula` | a main-terms `GLM`, as R `ltmle`'s `SL.library="glm"` | `_Baseline` in `src/cleverly/longitudinal/estimator.py`; `src/cleverly/longitudinal/gformula.py` |

The outer split is separate. `TMLE` cross-fits by default at ten outer folds, so the default fit
of one nuisance trains each library candidate sixty times: fifty on the inner folds and ten on the
outer training sets ([CV-TMLE and cross-fitting](cv-tmle.md)).

## Super Learner

`SuperLearner` (`src/cleverly/learners/super_learner.py`) is a cross-validated convex ensemble.
On each call to `fit` it:

1. draws the inner folds with `make_folds`. A classification task is stratified on its target
   (`StratifiedKFold`, or `StratifiedGroupKFold` with clusters), and a regression task takes the
   package's `random_partition`, grouped when clusters are passed. The fold count is capped at the
   rarer class count and at the cluster count (`resolve_n_folds`). With clusters, no candidate is
   scored on a row of a cluster it trained on. A candidate's own internal cross-validation, such as
   the penalty choice of the default library's lasso, receives no cluster codes;
2. fits each candidate on each inner training set and predicts the held-out rows, after
   clipping to `clip` when one is set;
3. drops, with a `UserWarning`, a candidate that raised on any fold, and raises `DataError` when
   every candidate raised;
4. scores each remaining candidate by its weighted cross-validated risk: squared error for a
   regression task, and the negative log-likelihood for a classification task, with predictions
   clipped to $[10^{-9}, 1 - 10^{-9}]$ before the log;
5. combines the out-of-fold predictions with the meta-learner, and refits the kept candidates on
   every row.

| `meta_learner` | what it solves | R `SuperLearner` analogue | when its solver fails |
| --- | --- | --- | --- |
| `"cc_ls"` | weighted least squares over the simplex, by SLSQP and an exact re-solve on the active set | `method.CC_LS` | the discrete selector, also used when the solution's risk exceeds the best candidate's |
| `"nnloglik"` | weighted negative log-likelihood over the simplex | `method.NNloglik` | `"cc_ls"`, and through it the discrete selector |
| `"nnls"` | non-negative least squares, then divided by the sum of the weights | `method.NNLS` | the discrete selector |
| `"discrete"` | all weight on the candidate of smallest cross-validated risk | `discreteSL = TRUE` | none needed |
| `"auto"`, the default | `"nnloglik"` for a classification task, `"cc_ls"` otherwise | none | as above |

A target with more than two levels is fitted one-vs-rest: one binary ensemble per class, each
with its own weights and risks, normalized at prediction time so each row sums to one. A row whose
class ensembles all predict zero reads the uniform distribution.

### What the oracle result covers

Polley (2010, Theorem 1) states the Super Learner oracle result under bounded data, squared-error
loss, folds drawn independently of the data, and a finite grid of meta-learner values, as
[references](../references.md) records them. Under those conditions the selected combination does
asymptotically as well as the best combination on the grid. `"cc_ls"` and `"discrete"` on a
regression task minimize squared error. The package solves `"cc_ls"` on the continuous simplex,
which the theorem reaches only through a grid approximation that the package does not state.

For `"nnloglik"`, and for `"discrete"` on a classification task, the package applies the
bounded-loss oracle inequality that its `SuperLearner` docstring attributes to van der Laan, Polley
and Hubbard (2007). This project has not read that source, so the claim is unverified here. The
log-likelihood loss is bounded only because predictions are clipped to $[10^{-9}, 1 - 10^{-9}]$,
which bounds it by about $\log 10^{9} \approx 20.7$. An oracle inequality's constants grow with
that bound and with a variance bound of the loss, so even granted, the statement is asymptotic and
says nothing usable at a practical sample size. This affects the default combination of every
binary nuisance.

When `"nnloglik"` falls back to `"cc_ls"` on a binary target, that fit minimizes squared error,
so its oracle statement is the squared-error one. Its reported `cv_risk` stays the log-likelihood
risk, because the reported loss follows the task. No oracle result covers `"nnls"`: the division
does not give the simplex minimizer, and
`tests/unit/test_super_learner_refit_and_simplex.py::TestTheSimplexLeastSquaresMetaLearner` holds a
law on which it keeps the weight on a much worse candidate. None covers the one-vs-rest
normalization or the stratified and grouped inner split either. The package keeps the inner split
as part of the training algorithm and makes no oracle claim for it
([the Super Learner inner split](cv-tmle.md#the-super-learner-inner-split)).

## The default library

`default_library(task)` (`src/cleverly/learners/library.py`) returns three scikit-learn
estimators. The settings are fixed in the code. Only the two penalized models choose a tuning
parameter from the data, by their own internal cross-validation.

| candidate | classification task | regression task |
| --- | --- | --- |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` | `HistGradientBoostingRegressor` |
| `random_forest` | `RandomForestClassifier` | `RandomForestRegressor` |
| `lasso` | standardization, then `LogisticRegressionCV` with an L1 penalty (`saga`) over eight penalties `np.logspace(-3, 2, 8)`, chosen by 5-fold log loss | standardization, then `LassoCV` with 5 folds |

Both boosting models use 200 iterations, learning rate 0.05, 15 leaves, at least 20 rows per leaf
and no early stopping. Both forests use 300 trees with at least 10 rows per leaf. Every candidate
that takes a `random_state` receives the ensemble's seed.

## Parametric learners with a stacked score

A Super Learner has no estimating equation the package can differentiate. `GLM`,
`ParametricOutcome` and `ParametricMechanism` are parametric fits whose score an estimator can
stack with its own.

| learner | model | used by |
| --- | --- | --- |
| `GLM` (`src/cleverly/learners/glm.py`) | unpenalized maximum likelihood with a canonical link, in the families of R `glm`: binomial (a fractional target is quasi-binomial), baseline-category multinomial, Gaussian. Main terms, with products up to `interactions`. A column aliased on the weighted fit rows takes a zero coefficient and leaves the score, as in R | the stacked intervals of [sequential g-computation and IPW](sequential-baselines.md) and of the [parametric g-formula](parametric-gformula.md) |
| `ParametricOutcome`, `ParametricMechanism` (`src/cleverly/learners/parametric.py`) | unpenalized logistic regression on a fixed vocabulary of feature maps, solved by Newton steps on equilibrated columns. Non-convergence or an ill-conditioned information matrix raises `DataError` | the stacked interval of the per-arm outcome-adaptive C-TMLE ([parametric per-arm intervals](collaborative-tmle.md#parametric-per-arm-intervals)) |

## Learners built from classifiers

A continuous treatment has a conditional density, which no scikit-learn learner predicts. The
binned density (`src/cleverly/learners/density.py`) factorizes it into discrete hazards on a grid
of bins, so one binary classifier fitted on the long `(unit, bin)` expansion estimates all of
them (Díaz Muñoz and van der Laan 2011, Section 2). The package refines the outer bins and reads
the density as a corrected log frequency polygon, its own construction.
[Modified treatment policies](point-treatment-tmle.md#modified-treatment-policies) states the grid,
the reading and their conditions, and [the density route](longitudinal-tmle.md#the-density-route)
gives its memory cost over time.

The classifier density ratio (`src/cleverly/learners/density_ratio.py`) stacks the observed rows
and the rows the policy assigns. A classifier of the copy label then gives the ratio as the odds
$\hat u / (1 - \hat u)$ (Díaz, Williams, Hoffman and Schenck 2023, Section 5.4). Two of its
rules are stated on no other page. First, $\hat u$ is read below $1 - 10^{-12}$. That guard
prevents a division by zero and caps the ratio near $10^{12}$, far above any ratio a fit that
passes the positivity checks reports. Second, the outer folds split
units, never stacked rows, and the Super Learner's inner folds are grouped by unit, because a
unit's two copies share its covariates.

## Other learners

| learner | what it fits | where it is stated |
| --- | --- | --- |
| the covariate screen (`src/cleverly/learners/screeners.py`) | a lasso of the outcome on the treatment-model covariates with the treatment held fixed, which keeps the covariates the treatment mechanism then reads. R `tmle`'s `prescreenW.g` screens marginally instead | [covariate screening](point-treatment-tmle.md#covariate-screening) |
| the L1 logistic path (`src/cleverly/learners/_l1_logistic.py`) | the propensity candidates of `strategy="lasso"` on a penalty path: the R `glmnet` binomial objective, and the ungrouped multinomial one, with standardized columns and an unpenalized intercept. A proximal Newton solver accepts a solution at a Karush-Kuhn-Tucker residual of at most $10^{-10}$ and otherwise raises `L1ConvergenceError`. No estimator stacks its score | [candidate sequences](collaborative-tmle.md#candidate-sequences) |
| hazard learners (`cleverly.survival`) | Cox models, selected by cross-validated partial likelihood, and stratified Nelson-Aalen estimators as a one-member library | [hazard learners](continuous-time-survival.md#hazard-learners) |
| rule learners | each fold's treatment rule, fitted to a doubly robust pseudo-outcome by any learner the caller passes | [learned rules](point-treatment-tmle.md#learned-rules), "Rule learners" |
| the EP learner | the efficient plug-in curve of a conditional effect | [the EP learner](conditional-effects.md#the-ep-learner), and [learned rules](point-treatment-tmle.md#learned-rules), "The EP rule learner" |

## The checks that exist

| check | what it establishes | test or study |
| --- | --- | --- |
| the object contract | every nuisance slot and a library refuse a string, and a learner the fit cannot clone is refused by name | `tests/unit/test_learner_object_contract.py` |
| seeding | a seeded ensemble and a seeded learner are functions of their data and seed; explicit seeds are kept and the caller's object is not mutated | `tests/unit/test_learner_seeding.py` |
| the default library | the three candidate names and their regression classes, and the library's naming and refusal rules. No test asserts the settings above, whose source is `default_library` | `tests/unit/test_learners.py::TestLibrary` |
| the ensemble | simplex weights, the discrete selector following the cross-validated risk, `"nnloglik"` as the binary default, a failing candidate dropped and all failing refused, clusters kept whole by the inner folds, weights moving the fit | `tests/unit/test_learners.py::TestSuperLearner` |
| the meta-learners | `"cc_ls"` returns the simplex minimizer (against a brute-force search) independently of the outcome's scale, `"nnls"` does not, and both fallbacks fire | `tests/unit/test_super_learner_refit_and_simplex.py::TestTheSimplexLeastSquaresMetaLearner` |
| refits | a refit replaces every fitted attribute, whichever branch either fit took | `tests/unit/test_super_learner_refit_and_simplex.py::TestARefitReplacesTheEarlierFit` |
| the inner split | flipping a held-out row's target moves no inner split of the fold that holds it out, and does move the others | `tests/unit/test_fold_policy_rules.py::TestTheInnerSplitDoesNotReadHeldOutRows` |
| grouped folds | a learner that can memorize a cluster reproduces held-out outcomes exactly under an ungrouped split and cannot under the grouped one | `tests/unit/test_crossfit_leakage.py` |
| the L1 path | agreement with a frozen copy of the coordinate-descent solver it replaced, with mutation controls on its branches | `tests/unit/test_l1_solver.py` |
| the L1 path against `glmnet` | the solver meets its residual tolerance, reads R `glmnet`'s coefficients as near-optimal on the same objective, and matches its predictions within ten times `glmnet`'s own gap, weighted and unweighted | `tests/unit/test_ctmle_sequence_witness.py::TestTheSolver` |
| `GLM` | equality with the unpenalized maximum-likelihood fits, the Jacobian and information against finite differences, aliasing and the refusals | `tests/unit/test_glm.py` |
| the parametric C-TMLE learners | equality with unpenalized scikit-learn fits, invariance to column scale, the refusals, and parity with the study pipelines, which a mutated basis breaks | `tests/unit/test_parametric_learners.py` |
| the binned density | exact arithmetic invariants; the grid and the reading against closed forms with deliberate-mutation controls; a tail error that falls with $n$ under the default bin count | `tests/unit/test_density.py`, `tests/unit/test_density_reconstruction.py`, `tests/unit/test_density_bins_consistency.py` |
| the classifier ratio | the folds split units on a vector treatment block | `tests/unit/test_density_ratio_vector.py` |
| the covariate screen | each component of the screen (partialling on the treatment, the interactions, the spline basis, the floor), each with a mutation control that fails | `tests/unit/test_outcome_screen.py` |
| an adaptive learner at few clusters | interval calibration of stacked `TMLE` at 10 to 40 clusters and cross-fitted `LTMLE` at 20, with the default library on every nuisance | the reporting [few-cluster default library](method-evidence/few-cluster-adaptive-learners.md) study |
| the density and classifier routes | interval calibration of a point modified treatment policy through the classifier ratio, and through the binned density with an oracle hazard learner, which isolates the grid and the reading from the learner | the gated [density tails](method-evidence/mtp-density-tails.md) study |
| weights reaching the learners | with linear and logistic learners, the plug-in from weighted nuisance fits is on target for the weighted population and the unweighted plug-in is not; R `tmle` agrees on the same samples | the gated [learned weighted point-treatment TMLE](method-evidence/learned-weighted-point-treatment-tmle.md) study |
| rule learners | interval calibration of the learned-rule value with the blip and weighted-classification learners | the gated [rule learners](method-evidence/learned-rule-learners.md) study |

[Nuisance model quality](validation-methods.md#nuisance-model-quality) reports the loss,
calibration and Super Learner weights of one fit's nuisances. It describes that fit and makes no
repeated-sampling claim.

## What the checks do not establish

The unit tests above establish that each learner computes what its construction states. The
studies measure estimators on their own laws with their own learners. Neither kind certifies a
learner on a user's law.

No check shows that a learner meets the rate an influence-curve interval needs: that the product
of the $L_2(P_0)$ errors of the outcome regression and the treatment mechanism is $o_P(n^{-1/2})$.
Over time, and with missingness or censoring, the condition covers the other mechanisms too.
The interval assumes it ([CV-TMLE and cross-fitting](cv-tmle.md) lists it among the conditions a
valid interval needs). A fit that is not cross-fitted also needs the nuisance estimators to lie in
a Donsker class, which a data-adaptive learner need not satisfy. Cross-fitting removes that
condition, though it still needs $L_2$ consistency. It is the default for `TMLE` and `LTMLE`, but
not for every estimator: the C-TMLE selection uses its own folds, and the sequential baselines fit
in sample.

The package decides "data-adaptive" by exact class (`learner_family` in `src/cleverly/_support.py`).
A learner is parametric when its class is exactly `LogisticRegression`, `LinearRegression` or
`GLM`, or declares `cleverly_learner_family = "parametric"`. Every other learner is adaptive,
`Ridge`, `LogisticRegressionCV`, any subclass and the default library included. On the $t$ tier
below, `nuisance_family` also counts as parametric a `SuperLearner` whose candidates are all
parametric, and a `Pipeline` of fixed-basis transforms before a parametric step.

The package applies these conditions to the support label of every estimator only on a clustered
fit on the $t$ tier, from its interval floor to 39 positive-mass clusters. There, an in-sample fit
with a data-adaptive learner is experimental by the Donsker argument, with its interval shown. A
cross-fitted fit is experimental unless every adaptive learner is the default library and the fit
lies inside the ground that `few-cluster-adaptive-learners` measured (`adaptive_nuisance_shortfall`
and `_adaptive_qualifiers` in `src/cleverly/_support.py`; [few clusters](inference.md#few-clusters)
states the rule).

Some fits read the learner at any cluster count. `AdaptiveIPTW` ([adaptive IPTW](adaptive-iptw.md#contract))
and the learned rule fitted on all rows ([learned rules](point-treatment-tmle.md#learned-rules),
"One rule fitted on all rows") read `learner_family` in their support keys: an adaptive learner
there matches no support row, so the fit reads experimental. Some support rows also pin the
learner class: the parametric g-formula rows require `GLM`, and the ratio-trim rows require
`LinearRegression`, so a fit with another class matches none of them (`SupportRow.matches`;
`src/cleverly/_support_rows.py` declares the rows). Apart from the density-route policy fits
below, every other fit off the $t$ tier keeps its row's label whatever learner it uses, so the
rate and Donsker conditions are the analyst's to judge.

An in-sample point policy fit of a continuous dose on the density route reads
`density_learner_family`. It is `known` when the treatment learner's class declares
`cleverly_density = "known"`, meaning the learner returns the law's exact density, as the oracle
learners of the shift studies do; otherwise it is the learner's family. The shift rows give the
method the exact density and declare `known`, so a fit that learns its density matches none of them
and reads experimental, with the shift row as the nearest. A user who passes the true density may
declare it the same way.

A cross-fitted density-route fit does not read the family: the
[learned-density study](method-evidence/point-shift-learned-density.md) measured a cross-fitted
additive shift whose density is a regularized `HistGradientBoostingClassifier` on 40 bins. Out
of fold the key reads the density learner's class and every setting but its seed
(`density_learner_signature`) and the bin count, so only a fit with exactly that learner and bin
count matches the study's row; the row's limits name what it does not read, among them the outcome
learner and the policy kind. The same study found the in-sample learned interval narrow, covering
0.8967 for a `+1.0` shift.

The rate at which a density ratio read from the binned density converges is the package's own
argument, measured by the witnesses and the density-tails study, and not a published theorem
(module docstring of `src/cleverly/learners/density.py`). The oracle gaps of the Super Learner are
stated [above](#what-the-oracle-result-covers).

TMLE is doubly robust in this exact sense: under positivity, the point estimate is consistent when
either the outcome regression or the treatment mechanism is estimated consistently. The interval
needs both, at the product rate above.
