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
| `strategy="oat"` | the outcome-adaptive mechanism, fitted on the arm-specific outcome predictions. It has no candidate path and no parameter-specific selector |
| `oat_design=` | the design of `strategy="oat"`. `"per_arm"` (the default) fits one binary mechanism per arm on that arm's own prediction. `"shared"` fits the one categorical mechanism of the archived `ctmle3`. See [the per-arm outcome-adaptive design](#the-per-arm-outcome-adaptive-design) |
| `candidates=`, `ordering=` | supply the candidate sets or the preorder explicitly. A refit that adds a covariate, such as the `random_common_cause` refutation, places it after the declared ordering. Independent noise has no prior relevance, so it ranks last |
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
[RM11](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm11-sensitivity-bounds-outside-their-derivation) records the refusal.

**A simulated common-cause surface reruns selection.** Binary complete-outcome fits accept fixed
probability weights for this operation. The operation refuses estimated weights. Its clustered
refusal is now unreachable through C-TMLE, because the fit itself refuses `id=`.

Each cell keeps the weight with its observed row and refits the complete collaborative estimator.
Selector strategies use the normalized weights in their nuisance fits, losses, penalties,
cross-validated risks, targeting, and plug-in. The outcome-adaptive strategy uses them in its
outcome and mechanism fits, targeting, and plug-in. See the
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
the outcome model reduced to a constant, the e2e test fits three seeds at `n = 1,500`. The search
includes the confounder on each seed, and the test allows the instrument on at most one. A
do-nothing selector has mean absolute error 0.695 there, against 0.036
(`tests/e2e/test_ctmle.py::TestSelectionIsForcedWhenTheOutcomeModelCannotHelp`).

**No R package implements the same complete selector.** `tmle3` is a design reference for the
shared out-of-fold-nuisance and pooled-fluctuation architecture, and it localises the convention
that training rows use fold fits while new rows use a full-training refit. It does not implement
C-TMLE selection. The candidate search is therefore accepted on the paper equations, the exact
identities, the row-membership audits, the mutation controls, the score checks, and the registered
studies. It is not accepted on numerical R parity.

**Scoring only one contrast is a load-bearing mutation.** The multi-arm selector's joint penalty is
checked by a mutation that scores only the first contrast. It changes the penalty by more than 100.

**The selector paths publish no inference.** The greedy, ordered, and discrete paths refuse
`ci`, `pvalue`, and `std_error`. Each accessor raises `CapabilityError`. One `discrete` fit is the
exception, and the paragraph on a declared full candidate below states it. The refusal says that the
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
explicit `simultaneous=True` also builds no band and emits no warning. The default is `True`, and
`summary()` names the omission when the fit reports more than one estimate.

An E-value reads the estimate and its interval, so every E-value branch reports `unavailable` on these
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

**A declared full candidate is the ordinary TMLE.** A `discrete` fit whose declared list holds one
candidate, equal to the full adjustment set, takes the ordinary TMLE status. It reports `ci`,
`pvalue`, `std_error`, a simultaneous band, and every derived operation that the TMLE fit reports.
The [natural-extension verdicts](natural-extension-verdicts.md) record this as part (j).

| item | content |
| --- | --- |
| base result | the TMLE interval result that [point-treatment TMLE](point-treatment-tmle.md) cites |
| argument | the search has one candidate, so it selects nothing. The candidate is the full set, so the mechanism is the TMLE mechanism, and the estimator is the TMLE estimator |
| key | `declares_full_adjustment_only` in `src/cleverly/estimators/ctmle.py`. It reads the strategy, the declared list, and the prepared covariate names. It never reads the fitted path |
| match rule | one candidate that holds each prepared covariate name once, in any order. A repeated candidate counts as two candidates. A repeated name is not the full set |
| admitted configurations | every configuration that `CTMLE` accepts: in sample and cross-fitted, two or more arms, fixed weights, `delta=` in sample, `ey`, `ate`, `rr`, `or`, a simultaneous band, `repeats`, and `n_bootstrap` |
| refit rule | a refit appends each covariate that `CausalData.with_extra_covariate` recorded to every candidate. So the `random_common_cause` refit of an admitted fit is the TMLE refit, and it keeps the status |
| still refused | the omitted-variable bound. It keys on the fitted method, and its sentence tells the caller to fit `TMLE` |
| evidence | `tests/unit/test_ctmle.py::TestEquivalenceWithPlainTmle` pins point, curve, standard error, interval, p-value, band, and bootstrap draws against `TMLE` to `1e-12` on ten configurations. The registered `TMLE` studies cover the identical estimator |

The key reads the declaration, so a caller knows the status before the fit. A key on the fitted path
would admit a fit whose candidate list happened to stop at the full set. The caller cannot predict
that stop. `TestOnlyTheDeclaredFullCandidateIsAdmitted` holds the controls. A single partial
candidate, a two-candidate path that stops at the full set, and `[full, full]` all refuse. A
mutation that keys on the fitted path admits the stopping control, and so fails it.

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

The pinned R `ctmle` computes a binary parametric term for each candidate and uses the selected
candidate's variance, as the paragraph above states. The table separates the two intervals.

| interval | what it is | R `ctmle` 0.1.2 at `18de559` |
| --- | --- | --- |
| `plugin_interval` | the Wald interval of the plug-in variance of $D^*$ at the bounded $g$ | the first value `calc_varIC` returns. `ctmleDiscrete` does not report it |
| `logistic_plugin(result)["ate"].plugin_logistic_interval` | the Wald interval of $D^* + \text{term1}\,I^{-1}(A - g)\tilde W$, with $\tilde W$ the selected covariates and an intercept | the second value of `calc_varIC(..., ICg = TRUE)`, which `ctmleDiscrete` reports as `var.psi` and `CI` (`R/functions_discrete.R`, lines 200 and 201, and `R/ctmle_discrete.R`, lines 181 to 196) |

`logistic_plugin` covers `ate`, `ey1` and `ey0` of a greedy, ordered or discrete fit of a binary
treatment, fitted in sample without weights, missing outcomes or `intermediate=`. The term is a
parametric-estimation correction only for an unpenalized main-terms logistic treatment learner.
For any other learner it is R's formula at that learner's prediction. Neither interval carries a
coverage claim, and the `working_mechanism_plugin` status does not change. At a correctly specified
outcome regression the term is close to zero.
`tests/unit/test_ctmle_logistic_plugin.py` compares the diagnostic with R's own `calc_varIC` on the
inputs of one fit, and with a misspecified-outcome witness where the term is not zero.

Leeb and Pötscher (2006) supply a nonuniformity warning in a finite-dimensional regression
subset-selection model; their theorem has not been transferred here. A working-mechanism plug-in
standard error makes no conditional-coverage claim and is not inferential output. The package now
enforces that sentence rather than only asserting it: the number is reachable under
`plugin_std_error` and `plugin_interval`, and under no inferential name. Near-ties are an
especially important unresolved regime.

### The per-arm outcome-adaptive design

`strategy="oat"` has two treatment designs. The `oat_design=` setting selects one.

| design | treatment mechanism | status |
| --- | --- | --- |
| `"per_arm"`, the default | for each arm $a$, a binary regression of $1\{A = a\}$ on the one column $\bar Q_n(a, W)$. Column $a$ holds $P(A = a \mid \bar Q_n(a, W))$, so the rows are not a distribution over the arms | the ordinary TMLE status on complete data without baseline strata. Otherwise `"generated_design_plugin"` |
| `"shared"` | one categorical regression of $A$ on the vector $[\bar Q_n(a, W) : a]$, as the archived `ctmle3` `LF_oat` fits it | `"generated_design_plugin"` on every fit |

Both designs use the ordinary $K$-column mean fluctuation. Its columns $1\{A = a\}/g_a$ have
disjoint supports, so under the per-arm design each arm keeps its own coefficient. At
convergence, each coefficient equals the scalar fluctuation on that arm's column alone.
`tests/unit/test_outcome_adaptive_per_arm.py` checks this to `1e-9`.

The source is Benkeser, Cai and van der Laan (2020), *Statistical Science* 35(3), 484-495,
doi:10.1214/19-STS735. The published article and its supplement were not readable here, so the
locators below are those of the preprint, arXiv:1901.05056v1. The preprint numbers its
appendices inconsistently: the text cites "Appendix G" for the conditions, which are in
Appendix F, and the Remark cites "Appendix F" for the direct ATE estimator, which is in
Appendix D.

| item | content |
| --- | --- |
| base result | Theorem 1 (preprint p. 9) and its construction, Section 3, steps 1-7 (pp. 8-9). Conditions (i)-(vi) and the proof are in Appendix F, "Details for Theorem 1" (pp. 28-33) |
| the reported curve | $D(O \mid \bar Q_0, Q_{0,W}, G_0(\cdot \mid \bar Q_0))$, the influence function of Theorem 1, with $G_0(a \mid \bar Q_0) = P(A = a \mid \bar Q_0(a, W))$. It is not the efficient influence function. Its variance is at most the efficient variance, because the estimator is superefficient |
| why no generated-design term | the proof writes the remainder as $R_{21} + R_{22} + R_{23} + R_{24}$ (pp. 30-31). $R_{24}$ is the effect of the generated regressor. Lemma 1 and condition (v) bound it by $\lVert \bar Q_n - \bar Q_0 \rVert$, and condition (ii) makes the product second order. $R_{22}$ is condition (vi) and $R_{23}$ is condition (iii) |
| the measurability step | the proof replaces $A$ by $G_0(\cdot \mid \bar Q_n, \bar Q_0)$ inside $P_0[(G_n - A) f]$. That step needs $f$ to be a function of $(\bar Q_n(W), \bar Q_0(W))$. In `cleverly`, $Q^*(a, W) = \operatorname{expit}(\operatorname{logit} \bar Q_n(a, W) + \epsilon_a / g_a(\bar Q_n(a, W)))$, and the fluctuation reads column $a$ only. So $Q^*(a, \cdot)$ is a function of $\bar Q_n(a, W)$, in sample and in each fold |
| (a) one arm mean | Theorem 1 directly. The design is the arm's own initial prediction |
| (b) every arm | an indicator reduction to $1\{A = a\}$ for each arm, as the Remark on p. 10 states: "repeating the entire procedure but switching the labeling of the treatment". Then a fixed-dimension stack of the $K$ curves on the same rows. The joint covariance comes from the stacked curves |
| (c) contrasts | linearity for `ate` and `par`. The delta method for `rr`, `or` and `paf`, with `rr` and `or` on the log scale. `ey_obs` has the curve $Y - E[Y]$. The default band is the multiplier band of the stack |
| (d) cross-fitted | Appendix D, "Cross-validated CTMLE" (p. 26): fold-trained $\bar Q_{n,v}$ and $G_{n,v}(\cdot \mid \bar Q_{n,v})$, and one coefficient pooled over the validation rows. Sample splitting replaces the Donsker part of condition (iv). The fit pools one coefficient per arm and has no fold-local update. The plug-in is the stacked whole-sample mean, not the mean of fold means. Near-equal folds make the difference $O(V/n)$ |
| (e) fixed weights | Theorem 1 under the weight-tilted law. The weighted projection gives the same identity, and the weight is a known bounded factor. Weights declared estimated take the `"estimated_weight_plugin"` status. The argument that an interval conditions on the weights concerns the efficient curve, and this curve is Theorem 1's. No cell of the registered study measures estimated weights |
| (f) repeated splits | each draw is asymptotically linear with the same split-free curve, so the median estimate is the estimator of Chernozhukov et al. (2018), equation (3.14). A default band beside `repeats > 1` is refused, as for `TMLE` |
| variance | in sample, the empirical variance of the reported curve. Cross-fitted, the variance of the stacked out-of-fold curve |
| conditions | (i) the curve equation is solved to $o_P(n^{-1/2})$. (ii) $\bar Q_n$ and $Q^*_n$ converge at $o_P(n^{-1/4})$ in $L_2(P_0)$. (iii) the treatment learner is consistent for $P(A = a \mid \bar Q_n(a, W))$ at $o_P(n^{-1/4})$. A misspecified learner makes $R_{23}$ first order whenever $Q^*_n - \bar Q_0 = O_P(n^{-1/2})$, which includes every correctly specified parametric outcome model. (iv) the curve converges in $L_2$, and in sample its class is Donsker. (v) $G_0(\cdot \mid \bar Q_n, \bar Q_0)$ is differentiable in $\bar Q_0$ with a bounded derivative. (vi) the term from fitting $G$ on the initial $\bar Q_n$ is negligible. Positivity: the projection $P(A = a \mid \bar Q_0(a, W))$ lies inside `g_bounds`. Also iid rows, or fixed known weights, and a consistent outcome regression. A correct $g$ alone does not give consistency |
| the source's warnings | Section 4.1 (p. 12): "the estimated standard errors of CTMLE had poor performance, often underestimating the true variability of the estimator". The Discussion (pp. 17-18) calls the CTMLE "an irregular estimator, it may perform poorly for certain data generating distributions", and notes "the poor behavior of the confidence intervals in both simulations". The limit is pointwise at a fixed law. It is not locally uniform |
| admitted | `strategy="oat"`, `oat_design="per_arm"`, complete outcomes and treatment, no `strata=`, no weights or fixed weights, and `repeats >= 1`. In sample and cross-fitted, at any arm count. Estimands `ey`, `ey1`, `ey0`, `ate`, `rr`, `or`, `ey_obs`, `par` and `paf`. Outputs: the pointwise interval, the p-value, the default band at one split, the derived contrasts, the E-value with its interval, and `variable_importance` |
| withheld | every `oat_design="shared"` fit, which [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) holds, and a per-arm fit with `delta=` or `strata=`, which [X29](../roadmap.md#x29-per-arm-outcome-adaptive-c-tmle-with-baseline-strata-and-missing-data) holds |
| bootstrap | `n_bootstrap=` is a diagnostic on every `oat` fit. Theorem 1 does not cover the bootstrap of a superefficient estimator |
| key | `per_arm_design_admits(estimator, data)` in `cleverly.estimators.ctmle`. It reads the strategy, the design, the missing outcomes and treatment, the strata and the revert flag, and never a fitted array |

**The revert flag.** `OAT_PER_ARM_INFERENTIAL` in `cleverly.estimators.ctmle` is `True` while the
registered study `ctmle-oat-per-arm` supports the admitted fits. Its declaration lists the
positive coverage and calibration cells that the flag reads. If one of them is red and no defect
is found, the flag becomes `False`. Every per-arm fit then takes `"generated_design_plugin"`, and
the reason names the red cells.

**The shared design publishes no inference.** It fits one categorical mechanism on the outcome
predictions of every arm. Theorem 1 covers one scalar design, and no reviewed result supplies the
vector influence function or the covariance of the shared construction. Cramér--Wold combines
scalar expansions that are already established. It cannot establish the missing ones. DOPE and
outcome-adapted AutoDML prove sample-split results with a representation learned on an
independent sample.

Every `oat_design="shared"` fit takes the `"generated_design_plugin"` status.
`ci`, `pvalue` and `std_error` raise `CapabilityError` with the reason of the status, and the
`summary()` column is `generated-design se`. At `n = 1,000`, the registered point-treatment pair
resolves a finite-sample deficit in the standard-error ratio without invalid coverage. The
multi-arm pair resolves no deficit. Neither measurement identifies a first-order term.
[F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) records the shared
design and uniformity for both designs.
[RM20](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm20-intervals-outside-every-claimed-contract) records the first decision.

The selector paths report no interval. Each one reports the spread of the ordinary plug-in curve
as a diagnostic. Ordinary TMLE conditions alone do not establish validity after selection, and
`cleverly` claims neither conditional selector coverage nor collaborative-double-robust coverage.

The full-refit bootstrap reruns each adaptive construction. No reviewed theorem validates it for a
selector path or for either outcome-adaptive design. On those paths, `summary()` therefore prints
the standard deviation of the replicate estimates as `bootstrap sd`, and a `percentile range`,
under the diagnostic framing. It prints no percentile confidence interval. `to_frame()` emits
`bootstrap_sd`, `bootstrap_range_lower` and `bootstrap_range_upper` in place of
`bootstrap_std_err`, `bootstrap_ci_lower` and `bootstrap_ci_upper`. The targeted-HAL bootstrap
result cited in the audit fixes its data-adaptive complexity bound rather than reselecting it.

| where to read the evidence | what is there |
| --- | --- |
| [selector-based point-treatment C-TMLE](method-evidence/selector-based-point-treatment-c-tmle.md) | greedy, ordered, and discrete selectors against R `ctmle` 0.1.2, with a forced-selection versus empty-path control. Parity is unpenalized, non-cross-fitted, and binary-ATE only |
| [outcome-adaptive point-treatment C-TMLE](method-evidence/outcome-adaptive-point-treatment-c-tmle.md) | the shared design against the archived `ctmle3`, including a pinned-versus-estimated design pair that measures the finite-sample cost of estimating the design |
| the registered study `ctmle-oat-per-arm` | the per-arm design against R `drtmle` 1.1.2 with `adapt_g = TRUE`, by a binary recode per arm, on two laws with an arm-specific outcome index and an instrument. Declared, and not yet run |
| `tests/unit/test_outcome_adaptive_per_arm.py` | an exact law on which the two designs differ, the curve of Theorem 1 row by row, in-sample and cross-fitted longhands to `1e-8`, the joint covariance, and fixed weights against duplicated rows. Its class `TestEachMutationFailsItsCheck` applies 13 mutations by `monkeypatch`, and each one fails its check |
| `tests/unit/test_shared_oat_replay.py` | the shared design refits replication 0 of each registered `ctmle3` study to its committed rows |
| [estimator variants over registered targets](evidence.md#estimator-variants-over-registered-targets) | the candidate-path identities, the selection mutations, and the outcome-adaptive design witnesses |
