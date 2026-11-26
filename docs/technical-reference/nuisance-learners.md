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
of `ModelSpec`, `TMLE`, `LTMLE` and `DRTMLE`, and a string library). A learner whose `fit` accepts
`sample_weight` receives the fit's observation weights, and one whose `fit` accepts `groups`
receives the cluster codes (`fit_learner` in `src/cleverly/learners/_fitting.py`).

When an estimator carries a seed, every `random_state` of the learner that is `None`, at any depth
of `get_params(deep=True)`, takes that seed on a clone, and an explicit seed is kept
(`seed_unset` in `src/cleverly/learners/_fitting.py`). A seeded fit is then a function of its data
and its seed. `tests/unit/test_learner_seeding.py` checks it under two global seeds, after showing
that the global seed does move an unseeded SAGA fit.

## The default learner

A slot left empty resolves to
`SuperLearner(task=task, n_folds=learner_folds, clip=(0.0, 1.0), random_state=seed, n_jobs=1)`
over the default library (`resolve_learner` in `src/cleverly/learners/super_learner.py`).
`learner_folds` is an estimator argument, 5 by default. A `SuperLearner` that the caller
constructs keeps its own `n_folds`, 10 unless set. The outer split is separate: cross-fitting is
on by default at ten outer folds, so the default fit of one nuisance trains each library candidate
fifty times on the inner folds ([CV-TMLE and cross-fitting](cv-tmle.md)).

## Super Learner

`SuperLearner` (`src/cleverly/learners/super_learner.py`) is a cross-validated convex ensemble.
On each call to `fit` it:

1. draws the inner folds with `make_folds`: stratified on the target for a classification task,
   and grouped by cluster codes when the fit passes them (`StratifiedGroupKFold`), so no candidate
   is scored on a row of a cluster it trained on;
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
| `"nnloglik"` | weighted negative log-likelihood over the simplex | `method.NNloglik` | `"cc_ls"` |
| `"nnls"` | non-negative least squares, then divided by the sum of the weights | `method.NNLS` | the discrete selector |
| `"discrete"` | all weight on the candidate of smallest cross-validated risk | `discreteSL = TRUE` | none needed |
| `"auto"`, the default | `"nnloglik"` for a classification task, `"cc_ls"` otherwise | none | as above |

A target with more than two levels is fitted one-vs-rest: one binary ensemble per class, each
with its own weights and risks, normalized at prediction time so each row sums to one. A row whose
class ensembles all predict zero reads the uniform distribution.

### What the oracle result covers

The oracle result concerns the weights that minimize the cross-validated risk over the simplex.
With bounded data and a bounded loss, the ensemble then does asymptotically as well as the best
weighted combination of the library (van der Laan, Polley and Hubbard 2007). Polley (2010,
Theorem 1) states it for squared-error loss and folds drawn independently of the data, which
covers `"cc_ls"` and `"discrete"` on a regression task. The log-likelihood risk of `"nnloglik"`,
and of `"discrete"` on a classification task, is bounded here only because of the clip at
$10^{-9}$, and the bounded-loss form applies.

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

A Super Learner has no estimating equation the package can differentiate. Three learners are
parametric fits whose score an estimator can stack with its own.

| learner | model | used by |
| --- | --- | --- |
| `GLM` (`src/cleverly/learners/glm.py`) | unpenalized maximum likelihood with a canonical link, in the families of R `glm`: binomial (a fractional target is quasi-binomial), baseline-category multinomial, Gaussian. Main terms, with products up to `interactions`. A column aliased on the weighted fit rows takes a zero coefficient and leaves the score, as in R | the stacked intervals of [sequential g-computation and IPW](sequential-baselines.md) and of the [parametric g-formula](parametric-gformula.md) |
| `ParametricOutcome`, `ParametricMechanism` (`src/cleverly/learners/parametric.py`) | unpenalized logistic regression on a fixed vocabulary of feature maps, solved by Newton steps on equilibrated columns. Non-convergence or an ill-conditioned information matrix raises `DataError` | the stacked interval of the per-arm outcome-adaptive C-TMLE ([parametric per-arm intervals](collaborative-tmle.md#parametric-per-arm-intervals)) |
| the L1 logistic path (`src/cleverly/learners/_l1_logistic.py`) | the R `glmnet` binomial objective, and the ungrouped multinomial one, with standardized columns and an unpenalized intercept. A proximal Newton solver accepts a solution at a Karush-Kuhn-Tucker residual of at most $10^{-10}$ and otherwise raises `L1ConvergenceError` | the candidates of `strategy="lasso"` ([candidate sequences](collaborative-tmle.md#candidate-sequences)) |

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
rules are stated only here. First, $\hat u$ is read below $1 - 10^{-12}$; that guard prevents a
division by zero and truncates no finite ratio the package reports. Second, the outer folds split
units, never stacked rows, and the Super Learner's inner folds are grouped by unit, because a
unit's two copies share its covariates.

## Other learners

| learner | what it fits | where it is stated |
| --- | --- | --- |
| the covariate screen (`src/cleverly/learners/screeners.py`) | a lasso of the outcome on the treatment-model covariates with the treatment held fixed, which keeps the covariates the treatment mechanism then reads. R `tmle`'s `prescreenW.g` screens marginally instead | [covariate screening](point-treatment-tmle.md#covariate-screening) |
| hazard learners (`cleverly.survival`) | Cox models and stratified Nelson-Aalen estimators, selected by cross-validated partial likelihood | [hazard learners](continuous-time-survival.md#hazard-learners) |
| rule learners | each fold's treatment rule, fitted to a doubly robust pseudo-outcome by any learner the caller passes | [rule learners](point-treatment-tmle.md#rule-learners) |
| the EP learner | the efficient plug-in curve of a conditional effect | [the EP learner](conditional-effects.md#the-ep-learner), and [the EP rule learner](point-treatment-tmle.md#the-ep-rule-learner) |

## The checks that exist

| check | what it establishes | test or study |
| --- | --- | --- |
| the object contract | every nuisance slot and a library refuse a string, and a learner the fit cannot clone is refused by name | `tests/unit/test_learner_object_contract.py` |
| seeding | a seeded ensemble and a seeded learner are functions of their data and seed; explicit seeds are kept and the caller's object is not mutated | `tests/unit/test_learner_seeding.py` |
| the default library | the three named candidates with the declared settings, and the library's naming and refusal rules | `tests/unit/test_learners.py::TestLibrary` |
| the ensemble | simplex weights, the discrete selector following the cross-validated risk, `"nnloglik"` as the binary default, a failing candidate dropped and all failing refused, clusters kept whole by the inner folds, weights moving the fit | `tests/unit/test_learners.py::TestSuperLearner` |
| the meta-learners | `"cc_ls"` returns the simplex minimizer (against a brute-force search) independently of the outcome's scale, `"nnls"` does not, and both fallbacks fire | `tests/unit/test_super_learner_refit_and_simplex.py::TestTheSimplexLeastSquaresMetaLearner` |
| refits | a refit replaces every fitted attribute, whichever branch either fit took | `tests/unit/test_super_learner_refit_and_simplex.py::TestARefitReplacesTheEarlierFit` |
| the inner split | flipping a held-out outcome moves no inner split | `tests/unit/test_fold_policy_rules.py::TestTheInnerSplitDoesNotReadHeldOutRows` |
| grouped folds | a learner that can memorize a cluster reproduces held-out outcomes exactly under an ungrouped split and cannot under the grouped one | `tests/unit/test_crossfit_leakage.py` |
| the L1 path | agreement with a frozen copy of the coordinate-descent solver it replaced, with mutation controls on its branches | `tests/unit/test_l1_solver.py` |
| `GLM` | equality with the unpenalized maximum-likelihood fits, the Jacobian and information against finite differences, aliasing and the refusals | `tests/unit/test_glm.py` |
| the parametric C-TMLE learners | equality with unpenalized scikit-learn fits, invariance to column scale, the refusals, and parity with the study pipelines, which a mutated basis breaks | `tests/unit/test_parametric_learners.py` |
| the binned density | exact arithmetic invariants; the grid and the reading against closed forms with deliberate-mutation controls; a tail error that falls with $n$ under the default bin count | `tests/unit/test_density.py`, `tests/unit/test_density_reconstruction.py`, `tests/unit/test_density_bins_consistency.py` |
| the classifier ratio | the folds split units on a vector treatment block | `tests/unit/test_density_ratio_vector.py` |
| the covariate screen | each part of the screen, each with a mutation control that fails | `tests/unit/test_outcome_screen.py` |
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

No check shows that a learner meets the rate an influence-curve interval needs: that the
product of the outcome and treatment errors is $o_P(n^{-1/2})$. The interval assumes it
([CV-TMLE and cross-fitting](cv-tmle.md) lists it among the conditions a valid interval needs).
A fit that is not cross-fitted also needs the nuisance estimators to lie in a Donsker class, which
a data-adaptive learner need not satisfy. Cross-fitting removes that condition and is the
default.

The package applies these conditions to the support label of every estimator only on a clustered
fit on the $t$ tier, from its interval floor to 39 positive-mass clusters. There, an in-sample fit
with a data-adaptive learner is experimental by the Donsker argument, with its interval shown. A
cross-fitted fit is experimental unless every adaptive learner is the default library and the fit
lies inside the ground that `few-cluster-adaptive-learners` measured (`adaptive_nuisance_shortfall`
and `_adaptive_qualifiers` in `src/cleverly/_support.py`; [few clusters](inference.md#few-clusters)
states the rule). Two fit families also read the learner family in their own support key at any
cluster count: `AdaptiveIPTW` ([adaptive IPTW](adaptive-iptw.md#contract)) and the learned rule
fitted on all rows ([one rule fitted on all rows](point-treatment-tmle.md#one-rule-fitted-on-all-rows)).
Any other unclustered fit keeps its support row's label whatever learner it uses, so the rate and
Donsker conditions are the analyst's to judge.

The rate at which a density ratio read from the binned density converges is the package's own
argument, measured by the witnesses and the density-tails study, and not a published theorem
(module docstring of `src/cleverly/learners/density.py`). The oracle gaps of the Super Learner are
stated [above](#what-the-oracle-result-covers).

TMLE is doubly robust in this exact sense: under positivity, the point estimate is consistent when
either the outcome regression or the treatment mechanism is estimated consistently. The interval
needs both, at the product rate above.
