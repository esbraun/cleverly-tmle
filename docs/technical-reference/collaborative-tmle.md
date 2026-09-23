# Collaborative TMLE

## What this solves

You have many measured covariates and you do not know which ones matter. Some are confounders. Some
are instruments: they predict the treatment strongly and do not affect the outcome. Putting an
instrument into the propensity model is not a neutral act. It makes the propensity extreme, which
makes the clever covariate large, which inflates the variance without removing any bias.

A propensity model chosen by predictive loss will happily include an instrument, because an
instrument is exactly what predicts treatment best. Collaborative TMLE chooses the treatment
mechanism against the *target parameter* instead.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| a large adjustment set with unknown structure | a treatment mechanism selected by cross-validated loss on the targeted estimate, so an instrument is left out | one nuisance fit per candidate along the selection path, on top of cross-fitting |
| near-positivity failure driven by strong treatment predictors | a less adaptive mechanism when the data says a less adaptive one estimates better | selection is data-dependent, and post-selection coverage has not been established |
| the outcome regression is already good | the *empty* propensity model is a legitimate choice, and the selector will make it | that is not evidence the search discriminates. See the validation section |

Reach for a different entry when your worry is the *inference* rather than the *selection*
([DR-TMLE](dr-tmle/index.md)), or when the adjustment set is small and you would include all of it
([point-treatment TMLE](point-treatment-tmle.md)).

Collaborative TMLE is available for point-treatment, arm-axis fits whose target is `ate`, `ey`,
`ey1`, `ey0`, `rr`, or `or`. It has no longitudinal derivation, and `available_methods()` says so
before any model is fitted.

A worked applied analysis is in the
[collaborative TMLE tutorial](../examples/collaborative-tmle.ipynb). It shows both the
comparison that discriminates and the one that does not.

The selector does not identify a causal adjustment set. Establish the eligible baseline set from
the study design before the selector chooses an assignment nuisance model.

## The algorithm as implemented

The implementation follows the published pooled construction step by step.

| paper operation | implementation invariant | how it is validated |
| --- | --- | --- |
| build increasingly adaptive candidates | the greedy, preordered, and discrete paths retain nested candidate state | exact path and treatment-risk tests |
| target the outcome regression with each candidate | every candidate carries its complete targeted regression, its fluctuation, and its influence curve | longhand loss, penalty, and score equations |
| cross-validate the stopping index | every selection fold refits the outcome regression, the auxiliary mechanisms, and every candidate. Training-row predictions are inner-fold out of fold, and validation-row predictions come from the full selection-training fit | row-identity spy learners, and the two-fold degeneracy case |
| select the candidate estimator | the final fit persists both the selected mechanism and the selected targeted regression | a mutation test in which discarding the targeted regression fails |
| report and retarget | pooled targeting continues from the selected state, and the continuation score is numerically zero | score, sensitivity, and serialization round trips |

**Selection folds are separate from nuisance cross-fitting.** Inside each selection-training set, a
dedicated inner split produces out-of-fold predictions for its training rows, and one additional
model trained on the whole set predicts the selection-validation rows. Matching fold counts or
seeds is never relied on for independence. A selection-validation observation contributes to no
nuisance fit used to score it, and no training row contributes to the model that predicts that row.
The outcome support transform is fixed from the outer fit, so every fold's loss and influence curve
stay in the same unit.

**Selector-based searches draw folds at every setting, so the fold rules bind in sample.** Both the
selection split and the nested split come from `random_partition`, which reads the row count and
the seed and no column. `cross_fit=False` removes the outer split, and it leaves these two. A
selector-based fit therefore refuses `stratify_by="treatment"` and `stratify_by="treatment+outcome"`
at every `cross_fit` setting. The message says so: "A collaborative fit draws those folds whether
or not cross_fit is set, so cross_fit=False does not make this policy available." This is the one
fit whose fold policy is refused in sample. Outcome-adaptive C-TMLE selects nothing and draws no
selection folds. Its in-sample fit accepts an unused policy. The
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give the audit behind it.

**The selection split gets its own preflight.** `CTMLE._preflight_selection_folds` asks the
realized selection partition the same support questions the outer split answers, before the first
learner. A selection fold whose training rows lack an arm makes the candidate's propensity
unfittable inside the search, and the fit refuses it up front instead.

**Selection is joint across arms.** With `K` arms, every candidate is one `n x K` categorical
mechanism, and the mean fluctuation solves all `K` arm equations. For curve matrix `D` the penalty
is

$$
\operatorname{trace}(\operatorname{cov}(D)) + n \lVert \operatorname{mean}(D) \rVert^2 ,
$$

so no contrast is privileged by its position. `ey` contributes all `K` arm curves. `ate`, `rr`, and
`or` contribute all `K - 1` curves against the reference. `CTMLESelection.target_names` records the
exact vector that was optimized.

Theory: van der Laan and Gruber (2010), Gruber and van der Laan (2010), and Ju et al. (2019); see
[collaborative TMLE](../references.md#collaborative-tmle). Implementation:
[`estimators/ctmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/ctmle.py).

## Variations

| option | what it does |
| --- | --- |
| `strategy="greedy"` | the package's scalable candidate path. The default |
| `strategy="ordered"` | both scalable preorders from Ju et al. (2019). `preorder="logistic"` uses one-variable targeting loss and is the default. `preorder="partial_correlation"` conditions the residual-covariate correlation on one-hot treatment indicators rather than on numeric arm codes. Marginal correlation with the outcome is not a published preorder and is not used |
| `strategy="discrete"` | selection among explicitly supplied candidate covariate sets |
| `strategy="oat"` | the outcome-adaptive categorical mechanism of the archived `ctmle3`, fitted on the matrix of arm-specific outcome predictions. It has no candidate path and no parameter-specific selector |
| `candidates=`, `ordering=` | supply the candidate sets or the preorder explicitly |
| `selection_folds=` | folds for the stopping-index cross-validation. Default 5. The draw is unstratified, and it reads the row count and the seed |
| `selection_inner_folds=` | the explicit cost control. Default 2, so a selection path uses three fits per nuisance or candidate rather than silently borrowing the outer nuisance fold count. This draw is unstratified as well |
| `loss=`, `penalty=` | the selection loss, and whether the variance-plus-squared-mean penalty is applied |
| `selection_estimand=` | which parameter vector the selection optimizes |

**Only the pooled collaborative estimator is exposed.** `targeting_scheme="fold"` and
`cv_evaluation=True` are refused. Composing collaborative model selection with fold-targeted or
canonical CV-TMLE changes the estimator and needs a separate derivation.

**Two other refusals reach every collaborative fit.** The first is `id=`: C-TMLE has no clustered
result. No reviewed result covers clustered inference for the outcome-adaptive mechanism.
Selector-based fits also lack a result for grouped selection and nested folds or candidate-selection
variance. The message names two ways out: "Drop id= from fit (PointTreatment(cluster=None)), or use
the ordinary TMLE (TMLE, or TMLEMethod), which has a clustered result." `available_methods` reports
`collaborative_tmle` as unavailable with the same reason when the design declares `cluster=`, so a
study reader sees it before anyone fits. The second is inherited from ordinary TMLE: a cross-fitted
continuous outcome needs a declared `q_bounds`.

**Retargeting holds the selection fixed.** A sensitivity analysis begins each perturbed targeting
step from the selected candidate's own targeted regression, not from the initial one. Both the
iterative and the one-step sweeps are checked to solve the perturbed score. Rerun the fit to redo
selection.

**Omitted-variable outputs refuse a collaborative fit.** The robustness value, the bounds, and
`elements()` raise `CapabilityError` on every strategy. The working mechanism conditions on a
function $V$ of $W$: the selected set $W_S$ on the selector paths, and the fitted outcome
regression under `oat`. Where the working mechanism is $P(A \mid V)$ in the limit, the
representer is $E[\alpha_W \mid A, V]$. That representer averages the full representer $\alpha_W$
within each arm and each value of $V$. Its $\nu^2$ is therefore never larger, and a collaborative robustness value would overstate robustness for the declared
adjustment set. The constant `_CTMLE_BOUND_REFUSAL` in `cleverly.sensitivity.omitted_variable`
gives the full reason, and
[RM11](../roadmap.md#rm11-sensitivity-bounds-outside-their-derivation) records the refusal.

**A simulated common-cause surface reruns selection.** Binary complete-outcome fits accept fixed
probability weights for this operation. The operation refuses estimated weights. Its clustered
refusal is now unreachable through C-TMLE, because the fit itself refuses `id=`.

Each cell keeps the weight with its observed row and refits the complete collaborative estimator.
Selector strategies use the normalized weights in their nuisance fits, losses, penalties,
cross-validated risks, targeting, and plug-in. The outcome-adaptive strategy uses them in its
outcome and categorical mechanism fits, targeting, and plug-in. See the
[simulated common-cause contract](validation-methods.md#simulated-common-cause-stress-surface).

The canonical R `ctmle` and archived `ctmle3` implementations provide no weighted comparator.
This fixed-weight surface claims no numerical parity with either implementation.

**`CTMLE` on an `incremental=` fit is wrong by construction**, not a gap. Each candidate mechanism
defines a different estimand, so the search would select between estimands rather than between
estimators.

## Validation issues special to this method

**A favourable comparison against plain TMLE can be won by a selector that selects nothing.** When
the outcome regression is correctly specified, the *empty* propensity model is a legitimate
mean-squared-error-minimising choice, and the ordered selector makes it on all five fixed
`n = 700` unit-test seeds. That is right rather than a defect. It also means such a comparison is
not evidence that the search discriminates between covariates.

The claim that it does discriminate is therefore tested where selecting nothing is *wrong*. With
the outcome model reduced to a constant, the search includes the confounder in every seed and still
leaves the instrument out. A do-nothing selector has mean absolute error 0.696 there, against
0.017.

**No R package implements the same complete selector.** `tmle3` is a design reference for the
shared out-of-fold-nuisance and pooled-fluctuation architecture, and it localises the convention
that training rows use fold fits while new rows use a full-training refit. It does not implement
C-TMLE selection. The candidate search is therefore accepted on the paper equations, the exact
identities, the row-membership audits, the mutation controls, the score checks, and the registered
studies. It is not accepted on numerical R parity.

**Scoring only one contrast is a load-bearing mutation.** The multi-arm selector's joint penalty is
checked by a mutation that scores only the first contrast. It changes the penalty by more than 100.

**The selector paths publish no inference.** The greedy, ordered, and discrete paths refuse
`ci`, `pvalue`, and `std_error`. Each accessor raises `CapabilityError`. The refusal says that the
reported curve is the ordinary efficient influence curve at the candidate the search stopped at,
and that no result shows it is this estimator's influence curve when that working mechanism is not
consistent for the treatment law.

The point estimate, the selection path, and the curve remain. Two accessors report the retained
diagnostic. `plugin_std_error` gives the plug-in standard error of that curve. `plugin_interval`
gives its Wald interval. Both are a diagnostic. Neither is a confidence statement.

The reports follow the accessors. `summary()` prints the diagnostic in a `working-mechanism se`
column, and prints no interval column. `to_frame()` emits `inference`, `plugin_std_err`,
`plugin_interval_lower`, and `plugin_interval_upper` in place of `std_err`, `ci_lower`, `ci_upper`,
and `p_value`.

Five derived operations refuse for the same reason. A contrast of two refused estimates is itself
refused. A simultaneous band is a joint confidence statement, so a selector fit builds none. An
E-value reads the estimate and its interval, so every E-value branch reports `unavailable` on these
paths. `variable_importance()` adjusts one p-value per candidate, so it refuses at its entry point
rather than after it has fitted one model per candidate. `tipping_gamma(use_ci=True)` follows a
confidence limit to the null, and it returns one float that cannot name the limit a diagnostic.

Two sweeps keep running, and each renames three columns. `truncation_curve()` and
`missingness_tilt()` report one point estimate per grid point, which needs no influence curve. Each
one emits `plugin_std_err`, `plugin_interval_lower`, and `plugin_interval_upper` in place of
`std_err`, `ci_lower`, and `ci_upper`. `tipping_gamma()` keeps its default `use_ci=False` search of
the point estimate, and it answers for these paths.

[F18](../roadmap.md#f18-selector-path-c-tmle-inference) is the condition that reopens this. It
reopens when it supplies the estimator's influence curve.

**One configuration is refused more than it needs to be.** A `discrete` fit whose only candidate is
the complete adjustment set selects nothing. It is bit-identical to a plain TMLE fit, and
`tests/unit/test_ctmle.py` pins that identity. The package still refuses its interval, because the
refusal keys on the strategy, as [RM12](../roadmap.md#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism)
correction 2 and F18 both require. A key that read the fitted path instead would give inference
back to a caller whose candidate list happened to collapse, which that caller cannot predict before
fitting. The over-refusal is deliberate and it is the conservative direction. Use `TMLE` for that
configuration.

Van der Laan and Gruber (2010), Theorem 2, establishes the population mean-zero identity that
supports collaborative consistency when the outcome regression is correct. It does not establish
the variance of an estimator using an estimated outcome regression and an inconsistent working
mechanism. Theorem 4 assumes the needed mean-zero rate and adaptive-mechanism expansion rather
than proving them for the selector. The pinned R `ctmle` computes a binary parametric term for each
candidate, then uses the selected candidate's variance. Neither source derives the influence
function of the package's nested stopping-index procedure. The cross-validation oracle inequality
does not close that gap either, because it bounds risk and states no limit law.

[F18](../roadmap.md#f18-selector-path-c-tmle-inference) records the gap. Adaptive debiased machine
learning and selective inference for cross-validation are candidate frameworks, but their fixed
oracle-model, Gaussian quadratic-selection, and joint-Gaussian-selection conditions have not been
established for this selector.

Leeb and Pötscher (2006) supply a nonuniformity warning in a finite-dimensional regression
subset-selection model; their theorem has not been transferred here. A working-mechanism plug-in
standard error makes no conditional-coverage claim and is not inferential output. The package now
enforces that sentence rather than only asserting it: the number is reachable under
`plugin_std_error` and `plugin_interval`, and under no inferential name. Near-ties are an
especially important unresolved regime.

**Outcome-adaptive intervals report an ordinary adaptive-propensity curve.** This strategy selects
no candidate. It fits one categorical mechanism on estimated arm-specific outcome predictions.
`strategy="oat"` is **unaffected** by the selector refusal above. It keeps `ci`, `pvalue`, and
`std_error`, and its frame keeps every ordinary column. F19 owns its open questions, and F18 owns
the three selector paths.
Benkeser, Cai and van der Laan (2020) prove that curve without an extra first-order design term for
one binary treatment-specific mean under six regularity conditions. The cross-fitted implementation
follows their fold-local nuisance nesting; the package does not diagnose those asymptotic
conditions.

Three readings narrow that gap, and none closes it. Benkeser, Cai and van der Laan (2020) prove a
binary treatment-specific-mean result and explicitly construct a two-arm-design ATE using one
signed fluctuation coefficient. The package uses a shared categorical mechanism and arm-specific
fluctuation columns, and exposes a joint means, ATE, RR, and OR vector. Cramér--Wold can combine
already-established scalar expansions; it cannot establish the missing component expansions,
remainders, or covariance for this exact joint fit. Its result is also limited to iid,
complete-outcome, unweighted data.

DOPE allows finitely many treatment levels and fixed contrasts, but its proved ordinary-curve
result conditions on a representation learned on an independent sample; it also shows how
root-rate representation learning can add a first-order term for a fixed target. Outcome-adapted
AutoDML likewise proves a sample-split rather than cross-fitted result. The shipped estimator's
multi-arm extension fits one shared multinomial mechanism and targets all arm means jointly. No
reviewed theorem supplies that vector influence function or covariance.

The remaining questions are the joint all-arm covariance, simultaneous inference, and uniformity
for a superefficient estimator. At `n = 1,000`,
the registered point-treatment pair resolves a finite-sample standard-error-ratio deficit without
showing invalid coverage; the multi-arm pair does not resolve a deficit. Neither measurement
identifies a first-order term.
[F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) records those items.

The selector paths report no interval. The outcome-adaptive path reports one, and that interval
is the ordinary cross-fitted EIF plug-in covariance. Ordinary TMLE conditions alone do not
establish validity after selection or representation learning, and `cleverly` claims neither
conditional selector coverage nor collaborative-double-robust coverage.

A refit bootstrap reruns each adaptive construction, but no reviewed theorem validates it for
either shipped path. On a selector path `summary()` therefore prints the bootstrap standard error
and a `percentile range`, under the same diagnostic framing, rather than a percentile confidence
interval. `to_frame()` emits the two limits as `bootstrap_range_lower` and `bootstrap_range_upper`
in place of `bootstrap_ci_lower` and `bootstrap_ci_upper`. The `bootstrap_std_err` column keeps its
name. The targeted-HAL bootstrap result cited in the audit fixes its data-adaptive complexity
bound rather than reselecting it.

| where to read the evidence | what is there |
| --- | --- |
| [selector-based point-treatment C-TMLE](method-evidence/selector-based-point-treatment-c-tmle.md) | greedy, ordered, and discrete selectors against R `ctmle` 0.1.2, with a forced-selection versus empty-path control. Parity is unpenalized, non-cross-fitted, and binary-ATE only |
| [outcome-adaptive point-treatment C-TMLE](method-evidence/outcome-adaptive-point-treatment-c-tmle.md) | against the archived `ctmle3`, including a pinned-versus-estimated design pair that measures the finite-sample cost of estimating the design |
| [estimator variants over registered targets](evidence.md#estimator-variants-over-registered-targets) | the candidate-path identities, the selection mutations, and the outcome-adaptive design witnesses |
