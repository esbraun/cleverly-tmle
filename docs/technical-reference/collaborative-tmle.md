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
| build increasingly adaptive candidates | the greedy, preordered, and discrete paths retain nested candidate state. The lasso and learners paths index a declared sequence of mechanisms | exact path and treatment-risk tests |
| construct the indexed candidates (lasso, learners) | from the current initial fit, fluctuate with every remaining candidate and take the first smallest criterion. Stop when it does not improve on the best so far. Otherwise record the skipped candidates as fluctuations of the current fit, and continue from the accepted candidate's targeted fit | a scripted criterion table, and R `ctmleGeneral` and `ctmleGlmnet` on R's own propensity arrays |
| take the C-TMLE1 final step (lasso) | fluctuate the selected candidate's targeted fit with each smaller penalty, and report the one with the smallest unpenalized loss | the R `ctmleGlmnet` final step on R's explicit-penalty columns |
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
| `strategy="lasso"` | the L1-penalized logistic mechanism at each penalty of a decreasing grid, then the C-TMLE1 final step. The analogue of R `ctmleGlmnet(ctmletype = 1)`. See [candidate sequences](#candidate-sequences) |
| `lambdas=` | the penalty grid of `strategy="lasso"`, with two or more penalties. The default is the `glmnet` rule: 100 log-spaced values from `lambda_max` down to `lambda_max` times `1e-4` (or `1e-2` when `n < p`), then 0. Under `strategy="derivative"` it is the pair `(mechanism_penalty, partner_penalty)` |
| `strategy="learners"` | each declared treatment learner in the declared order, then `treatment_learner` on every covariate. The analogue of R `build_gn_seq` with `ctmleGeneral(ctmletype = 1)` |
| `treatment_candidates=` | the declared entries of `strategy="learners"`, least adaptive first. An entry is an unfitted learner or a `PropensityArrays(full=..., cv=...)` pair. Under `strategy="derivative"` it is the pair `(mechanism, partner)`, each an unfitted learner or a full-sample array |
| `strategy="derivative"` | C-TMLE0: one fluctuation that adds the derivative of each arm's clever covariate. It selects nothing and takes the ordinary TMLE status. The analogue of R `ctmleGlmnet(ctmletype = 3)` and `ctmleGeneral(ctmletype = 3)`. See [C-TMLE0](#c-tmle0-the-derivative-covariate) |
| `selection_split=` | fold labels `0` to `V - 1`, one for each row, in place of the random selection split. Valid on every selector strategy. `selection_folds=` must equal `V` |
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

**Two other refusals reach collaborative fits.** The first is `id=`, on every strategy except
`"derivative"`: C-TMLE has no clustered result. No reviewed result covers clustered inference for
the outcome-adaptive mechanism. Selector-based fits also lack a result for grouped selection and
nested folds or candidate-selection variance. The message names two ways out: "Drop id= from fit
(PointTreatment(cluster=None)), or use the ordinary TMLE (TMLE, or TMLEMethod), which has a
clustered result." `strategy="derivative"` selects nothing, and it takes the clustered result of
the ordinary TMLE. So `available_methods` keeps `collaborative_tmle` on a design that declares
`cluster=`, and every other strategy refuses `id=` before its first learner. The second refusal
reaches every strategy, and it is inherited from ordinary TMLE: a cross-fitted continuous outcome
needs a declared `q_bounds`.

**Retargeting holds the selection fixed.** A sensitivity analysis begins each perturbed targeting
step from the selected candidate's own targeted regression, not from the initial one. Both the
iterative and the one-step sweeps are checked to solve the perturbed score. Rerun the fit to redo
selection. A C-TMLE0 fit has no selection. Its retarget rebuilds the derivative columns at the
requested bounds, as the [C-TMLE0](#c-tmle0-the-derivative-covariate) section states.

**Omitted-variable outputs refuse a collaborative fit.** The robustness value, the bounds, and
`elements()` raise `CapabilityError` on every strategy except `"derivative"`. The working mechanism conditions on a
function $V$ of $W$: the selected set $W_S$ on the selector paths, and the fitted outcome
regression under `oat`. Where the working mechanism is $P(A \mid V)$ in the limit, the
representer is $E[\alpha_W \mid A, V]$. That representer averages the full representer $\alpha_W$
within each arm and each value of $V$. Its $\nu^2$ is therefore never larger, and a collaborative robustness value would overstate robustness for the declared
adjustment set. The constant `_CTMLE_BOUND_REFUSAL` in `cleverly.sensitivity.omitted_variable`
gives the full reason, and
[RM11](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm11-sensitivity-bounds-outside-their-derivation) records the refusal.

**A simulated common-cause surface reruns selection.** Binary complete-outcome fits accept fixed
probability weights for this operation. The operation refuses estimated weights. Its clustered
refusal reaches C-TMLE only through `strategy="derivative"`, the one strategy that admits `id=`.

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

## Candidate sequences

The `lasso` and `learners` strategies index a declared sequence of treatment mechanisms. They do
not search over covariate sets. The table gives the base result and the locator of each part.

| item | content | locator |
| --- | --- | --- |
| general template | candidates over an index set, ordered by empirical fit. The most adaptive end is consistent for $g_0$. Each candidate gives a TMLE, and cross-validation selects the index | van der Laan and Gruber (2010), Section 2.4 |
| ordering requirement | the empirical losses of the candidates decrease in the index, and the last candidate is consistent | Ju, Gruber et al. (2019), Section 4.1, Algorithm 1, step 2 |
| indexed construction rule | from the current initial fit, fluctuate with every remaining candidate and take the smallest criterion. Record every skipped candidate as a fluctuation of the current fit. The accepted candidate's targeted fit becomes the next initial fit | Ju, Wyss et al. (2019), Appendix A, Algorithm 1, steps 5 to 12. R `ctmle` `construct_candidates_general` |
| construction criterion and stop | the selection `risk`: the loss, plus the penalty when `penalty=True`. Construction stops when the best try does not improve on the best score so far. The remaining candidates then fluctuate the last initial fit. With `penalty=False` the stop never binds | R `ctmle` `R/functions_general.R`, lines 60 to 129 |
| C-TMLE1 final step (`lasso` only) | after the cross-validation selects a penalty, fluctuate the selected candidate's targeted fit with each smaller penalty. Report the one with the smallest unpenalized loss. With no smaller penalty, report the selected candidate | Ju, Wyss et al. (2019), Section 4.2, and Appendix A, Algorithm 1, steps 13 to 16. R `ctmle` `R/ctmle_glmnet.R`, lines 265 to 325 |
| lasso candidate | the `glmnet` objective: weighted mean binomial deviance over two, with the weights normalized to mean 1, columns standardized by the weighted population standard deviation, an unpenalized intercept, and an L1 penalty. At `K > 2` arms it is the symmetric `ungrouped` multinomial | Ju, Wyss et al. (2019), Section 4.2 |
| declared candidates | each entry in the declared order, least adaptive first, then `treatment_learner` on every covariate | R `ctmle` `R/build_gn_seq.R` and `ctmleGeneral` |

The final step of Ju, Wyss et al. (2019), Algorithm 1, fluctuates the selected candidate's
*initial* fit. R `ctmleGlmnet` and this package fluctuate its *targeted* fit
(`R/ctmle_glmnet.R`, line 291).

**The solver has a residual contract.** `cleverly.learners._l1_logistic` solves the lasso by
coordinate descent with a Newton polish on the active set, warm-started along the grid. Its
Karush-Kuhn-Tucker residual is at most `1e-10` at every penalty. When the residual is larger, the
fit raises `L1ConvergenceError`, and the message names the layer and the penalty. The usual cause
is separation at penalty 0. A grid that ends above 0 avoids it.

**The ends of the default grid hold on the analysis sample.** There, the first
default candidate is the intercept-only mechanism exactly, and the last is the unpenalized
main-terms logistic fit. The grid is fixed on the analysis sample. The training rows of a
selection fold or an outer fold have their own `lambda_max`, so there the first candidate can
have a nonzero coefficient.

**A `PropensityArrays` pair is the `ctmleGeneral` input.** `full` holds the predictions of a fit on
every row. `cv` holds out-of-fold predictions under the split that `selection_split=` declares. The
analysis-sample path reads `full`. Inside each selection fold, the training and the validation
rows read `cv`, as R does. The pair needs `selection_split=`, `cross_fit=False`, `repeats=1`, and
`n_bootstrap=0`, and a refit refuses it.

**The pair carries a declared exception.** Each training-row prediction in `cv` came from a fit
that used the validation fold's treatment and covariates, though never its outcome. R `ctmle` has
the same property. This is an exception to the rule that a selection-validation observation
contributes to no nuisance fit that scores it. The package cannot check that `cv` respects
`selection_split`, so the caller must make sure it does. A learner entry has no exception: the
package fits it on the same rows a covariate-set candidate uses.

**`CTMLESelection` records the sequence.** `labels` names each candidate, for example
`"lambda=0.0123"`, `"lambda=0"`, `"LogisticRegression[2]"`, or `"arrays[1]"`. `final` is the
final-step candidate, or `None` when no final step ran. `selected` stays the cross-validated index.
The reported mechanism is `final` when it is set, and `selected` otherwise. Under `lasso`,
`path[k]` holds the covariates with a nonzero coefficient in some fit of candidate `k`. `summary()`
also reports whether the treatment risk decreases along the declared order, which the template
requires. A fit does not refuse when it does not.

**R `ctmle` differs in these ways.** The package does not reproduce them.

| R behavior | locator | package behavior |
| --- | --- | --- |
| `cv_general` discards the supplied `folds` and draws new ones | `R/functions_general.R`, line 287 | `selection_split=` is used as given |
| inside each fold, the evaluation refits the fluctuation on all rows before it scores the validation rows | `R/functions_glmnet.R`, lines 32 and 86 to 97; `R/functions_general.R`, lines 184 and 213 to 218 | the fluctuation uses the training rows only |
| inside each fold, `glmnet` is refitted on every row | `R/functions_glmnet.R`, line 24 | each fold fits its own training rows |
| `predict(s = )` interpolates the default path and clamps below its smallest penalty, and the default path can stop early | `glmnet` | each penalty is solved exactly |
| the next accepted column is found by a squared distance below `1e-5` | `R/functions_general.R`, line 209 | candidates are matched by index |
| `calc_varIC(ICg = TRUE)` reads only the first covariate column | `R/functions_general.R`, line 198 | not used |
| type 1 has no branch for a selected last penalty | `R/ctmle_glmnet.R`, line 270 | the selected candidate is reported |
| the construction criterion adds `var(D*)` whatever `PEN` is | `R/functions_glmnet.R`, lines 172 to 177; `R/functions_general.R`, lines 65 to 72 | the construction adds the penalty only when `penalty=True` |
| the construction penalty carries the factor `n_train / n` inside each fold | `R/functions_glmnet.R`, line 172 | no factor |

Under `lasso`, `treatment_learner` remains the fallback for a missingness mechanism, and the lasso
candidates do not use it. A scikit-learn `FrozenEstimator` passes as a learner entry and is not
refitted, because `clone` returns it unchanged. This holds for every learner argument.

**Evidence.** `tests/unit/test_ctmle_candidate_sequences.py` holds the scripted construction rule,
the solver residual, the path ends, the row-identity audits, the refusals, and forced selection.
`tests/unit/test_ctmle_sequence_witness.py` compares the solver with `glmnet`, and the construction
and the final step with R `ctmleGeneral` and `ctmleGlmnet`, on the fixture in
`tests/canonical/ctmle_sequence_witness/`. The registered study
[C-TMLE candidate sequences](method-evidence/c-tmle-candidate-sequences.md) compares repeated
samples with R `ctmle`.

## C-TMLE0: the derivative covariate

`strategy="derivative"` is C-TMLE0. It is an ordinary TMLE whose logistic fluctuation adds one
column for each arm: the derivative of that arm's clever covariate along a path of treatment
mechanisms. A finite difference between the fitted mechanism $g$ and a partner mechanism $g^+$
stands in for the derivative. Nothing is selected collaboratively. The fit has no candidate sequence
and no selection cross-validation.

| item | content | locator |
| --- | --- | --- |
| definition | C-TMLE0 "is exactly the same as the TMLE algorithm, except it updates the estimate by equation 6": $\operatorname{logit} \bar Q^* = \operatorname{logit} \bar Q + \epsilon_1 H_g + \epsilon_2 \tilde H_g$, with $\tilde H$ the derivative of $H$ in the penalty | Ju, Wyss et al. (2019), Section 4.2, equation (6) |
| algorithm | the initial $\bar Q_n$, the mechanism at the cross-validated penalty, then one targeting step with $H$ and $\tilde H$ | Ju, Wyss et al. (2019), Appendix B, Algorithm 2 |
| goal equation | $P_n \frac{d}{dh} D^*(\bar Q^*, G_h) = 0$ together with $P_n D^* = 0$ | Ju, Chambaz and van der Laan (2018), Section 4.2 |
| lasso instance | the mechanism at `lambda.min`, and the partner at the nearest penalty of the grid | Ju, Chambaz and van der Laan (2018), Section 4.2, steps 1 to 3 |
| R comparator | type 3 bounds `g1W` and `g1WPrev`, builds four columns, and fits one `glm` | R `ctmle` 0.1.2, `R/ctmle_glmnet.R`, lines 133 to 174 |

The package builds the update as the table states.

| part | package behavior |
| --- | --- |
| derivative column | $dh_a = h_a r_a$ with $r_a = (g_a - g^+_a)/g_a$. Both mechanisms are bounded at `g_bounds` first. With complete data it is $1\{A = a\}(g_a - g^+_a)/g_a^2$, which is R's `ddH1W` and `ddH0W`, sign included |
| counterfactual block | arm $a$'s block gets $r_a / (g_a \pi_a)$. The factor $1/\pi_a$ is the chain rule for an in-sample missing outcome |
| normalization | each $dh_a$ is divided by its root mean square over the rows with $A = a$ and an observed outcome. The fit does not change. `epsilon` is reported on R's scale |
| solve | one solve of the $2K$ columns on every row, pooled over the outer folds, with the iterative logistic or linear solver |
| reported curve | $D^*(\bar Q^*, g_n)$ with the bounded $g$ only. The derivative columns move $\bar Q^*$ and never enter the curve |
| continuation | the ordinary $K$-column fluctuation from $\bar Q^*$. Its gradient is already below `tol`, so its `epsilon` is 0 |

Three sources supply the two mechanisms.

| call | mechanism $g$ | partner $g^+$ | R comparator |
| --- | --- | --- | --- |
| `lambdas=None` | the L1 logistic at the penalty that `learner_folds`-fold weighted deviance cross-validation chooses over the default grid, inside each training set | the nearest penalty of the grid in absolute distance. A tie goes to the larger penalty | C-LASSO Algorithm 2. Ju, Chambaz and van der Laan (2018), Section 4.2 |
| `lambdas=(lam, lam_plus)` | the L1 logistic at `lam` | the L1 logistic at `lam_plus`, larger or smaller | `ctmleGlmnet(ctmletype = 3)` with `glmnet` at the two penalties |
| `treatment_candidates=(g, plus)` | an unfitted learner, or a full-sample array | the same kinds | `ctmleGeneral(ctmletype = 3, g1W, g1WPrev)` |

The partner is fitted on exactly the rows of the mechanism: every row in sample, and each outer
fold's training rows when cross-fitted. The grid is fixed once on the analysis sample, and it reads
the treatment and the covariates only. The fit stores the partner in
`NuisanceEstimates.treatment_partner`. `CTMLEDerivativeFit` under `result.extra["ctmle"]` records
the source, the penalty pair of each training set, the cross-validated risk, the $2K$ coefficients,
and the derivative scores before and after the update.

**The sources disagree in print, and the published derivation governs.** The goal equation
$P_n \frac{d}{dh} D^* = 0$ resolves each discrepancy.

| # | discrepancy | resolution |
| --- | --- | --- |
| D1 | Algorithm 2 prints $A/g\,(g_{\lambda+\delta} - g_\lambda)$, without the square. Equation (6) and R use $A/g^2$ | the square, because $\tilde H = dH/d\lambda$ |
| D2 | JCvdL equation (33) prints the direction $C(G)\,G'$. The goal equation needs $\frac{d}{dh} C(G_h) = -G'/(lG)^2$. Per arm, (33) spans $A\,G'/G$, which is the D1 form | the derivative of the clever covariate, as R fits it |
| D3 | equation (6) uses one combined $H$ and one $\tilde H$. R splits each by arm | the per-arm split, $2K$ columns. Its span contains every combined pair |
| D4 | the partner is $\lambda + \delta$ (C-LASSO), the nearest grid value (JCvdL), $1.05\,\lambda$ interpolated (the `ctmle` README), or any `g1WPrev` (R) | the column enters only through its span, so its sign and scale do not matter. The internal default follows JCvdL step 2 |
| D5 | Algorithm 2 prints $H$ as a sum of the two arm terms, not the ATE covariate | the per-arm split contains both combined forms |

**The interval rests on the ordinary TMLE result.** C-TMLE0 selects nothing collaboratively, so
the F18 boundary does not apply. The status is the TMLE status: `influence_curve`, or
`few_cluster_plugin` as for `TMLE`.

| item | content |
| --- | --- |
| base result in sample | van der Laan and Rubin (2006), Sections 1, 3 and 4.2: a fluctuation whose scores span the influence curve, with the expansion, the remainder, the Donsker condition and the $L_2$ conditions |
| base result cross-fitted | Zheng and van der Laan (2011), Section 2, Theorems 1 and 2: a pooled TMLE with a finite-dimensional fluctuation over out-of-fold nuisances |
| the efficient influence curve equation | the $2K$ columns contain the $K$ columns $h_a$. A logistic fit with an offset solves every column's score, so $P_n D^*(\bar Q^*, g_n) = 0$ |
| the expansion | $\psi_n - \psi_0 = (P_n - P_0) D^*(\bar Q^*, g_n) + R_2$, with $R_2 = \sum_a P_0 (g_{0a}\pi_{0a}/(g_{na}\pi_{na}) - 1)(\bar Q_{0a} - \bar Q^*_a)$. It does not depend on how $\bar Q^*$ was built |
| why the penalty is not an F18 selection | the cross-validated penalty minimizes a treatment deviance. It never reads $Y$ or a targeted fit, as a Super Learner treatment learner does not |
| conditions | (i) the $h_a$ scores are solved. (ii) $\bar Q_n$ and $g_n$ are consistent, with a product of rates $o_p(n^{-1/2})$. (iii) the Donsker condition in sample, or cross-fitting. (iv) $g_0$ lies inside `g_bounds`. (v) each derivative column is neither zero nor a constant multiple of $h_a$, and the $2K$ solve converges. (vi) iid rows or independent clusters, and fixed weights |
| what is not claimed | the collaborative claim that the interval holds when $\bar Q_n$ is inconsistent. That claim stays with [F18](../roadmap.md#f18-selector-path-c-tmle-inference) |
| the sources' own warnings | C-LASSO Section 6.4, Table 3: C-TMLE0 covers 0.920 and 0.872 with $\bar Q$ fitted on 10 or 20 of 40 confounders. Section 6.5.1 names "its under-estimated standard error". JCvdL Section 5.4 names "the apparent failure in estimating well their variance". Each outcome regression there is misspecified, outside condition (ii) |

Under condition (ii) the remainder $R_2$ is a product of the two errors, so the interval does not
rest on a remainder linear in one nuisance error.

**R `ctmle` type 3 differs in these ways.** The package does not reproduce them.

| R behavior | locator | package behavior |
| --- | --- | --- |
| a zero finite difference becomes `1e-10` | `R/ctmle_glmnet.R`, lines 143 and 145 | the row keeps 0. Such a row has both mechanisms clipped at one bound |
| `fluctuation=` and `family=` are ignored, and the logistic fit always runs | lines 111, 128 and 150 to 153 | `fluctuation="linear"` is honoured |
| no bound after the update | lines 155 to 162 | predictions shrink at `alpha` |
| a missing outcome enters as 0 | line 105 | `delta=` declares the missing outcome |
| `var.psi` divides by $n - 1$ | lines 173 and 174 | `std_error` divides by $n$. Multiply its square by $n/(n-1)$ to compare |
| `glm` drops an aliased column when `g1W` equals `g1WPrev`, and `est` becomes `NA` | `glm` | the fit refuses |

**Retargeting rebuilds the update.** `retarget(g_bounds=b)` and each `truncation_curve` point
recompute the derivative columns at `b` from the initial fit, the mechanism, and the stored partner,
and then run the continuation. So each point is C-TMLE0 at its bound. The omitted-variable bound
admits a C-TMLE0 fit, because the fit assumes a consistent mechanism as the TMLE does.

**Refusals.** The [refusal reference](scope-and-refusals.md) lists each message. The fit refuses
`targeting="one_step"`, `target_weights=True`, `screen_treatment=True`, baseline strata, a zero or collinear derivative
column, and a $2K$ solve that does not converge. An array entry refuses cross-fitting, the
bootstrap, and every refit, because it describes one fit on the original rows and columns.

**Evidence.** The registered study [C-TMLE0 derivative covariate](method-evidence/ctmle-derivative.md)
compares repeated samples with R `ctmle` type 3. Both scenarios pass their truth tests and are
equivalent to R. The paired mean difference is 4.4e-07 at the declared pair and -2.1e-06 for the
two pipelines. Sixteen of 17 property cells pass. The red cell is the both-wrong control, which
the law's mild mechanism misspecification leaves inside the margin. Three test modules hold the
C-TMLE0 checks.

| test module | checks |
| --- | --- |
| `tests/unit/test_ctmle_derivative.py` | an independent transcription of R type 3; the solved derivative scores (W1); the move away from the TMLE estimate (W2); span invariance (W3); the counterfactual block (W4); three arms (W5) |
| `tests/unit/test_ctmle_derivative_strategy.py` | the partner rule and the penalty cross-validation (W6); the row identity (W7); the refusals (W8); the status (W9); the retarget and the save round trip (W10); the omitted-variable and refit paths (W11) |
| `tests/unit/test_ctmle_derivative_witness.py` | the update on R's own arrays from `tests/canonical/ctmle_derivative_witness/`, four blocks. The coefficients match R to a relative 1e-6. The targeted fits, the curve, the estimate and `var.psi` match to 1e-6. Block B fails with an unbounded partner, and block C fails with an unbounded mechanism |

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
studies. It is not accepted on numerical R parity. The indexed construction rule and the final step
are the exception. R `ctmle` implements both, and a fixture compares them on R's own arrays (see
[candidate sequences](#candidate-sequences)).

**Scoring only one contrast is a load-bearing mutation.** The multi-arm selector's joint penalty is
checked by a mutation that scores only the first contrast. It changes the penalty by more than 100.

**The selector paths publish no inference.** The greedy, ordered, discrete, lasso, and learners
paths refuse `ci`, `pvalue`, and `std_error`. Each accessor raises `CapabilityError`. Two declared
fits are the exception, and the paragraph on a declared full candidate below states them. The
refusal says that the reported curve is the ordinary efficient influence curve at the candidate the
search stopped at, or at the final-step candidate. It also says that no result shows it is this
estimator's influence curve when that working mechanism is not consistent for the treatment law.

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
reopens when it supplies the estimator's influence curve. A result that reopens a path must bound
the adaptive-mechanism term without conditioning on the outcome regression, or must show that the
term is a product. A term that is linear in $g_n - g_0$ when $\bar Q$ is wrong does not qualify.

**A declared full candidate is the ordinary TMLE.** Two declarations select nothing. A `discrete`
fit whose declared list holds one candidate, equal to the full adjustment set, is the first. A
`learners` fit with an empty `treatment_candidates=` list is the second: its one candidate is
`treatment_learner` on every covariate. Each takes the ordinary TMLE status. It reports `ci`,
`pvalue`, `std_error`, a simultaneous band, and every derived operation that the TMLE fit reports.
The [natural-extension verdicts](natural-extension-verdicts.md) record this as part (j).

A single declared `learners` entry gives two candidates. That fit selects and keeps the plug-in
status.

| item | content |
| --- | --- |
| base result | the TMLE interval result that [point-treatment TMLE](point-treatment-tmle.md) cites |
| argument | the search has one candidate, so it selects nothing. The candidate is the full set, so the mechanism is the TMLE mechanism, and the estimator is the TMLE estimator |
| key | `declares_no_selection` in `src/cleverly/estimators/ctmle.py`. It reads the strategy, the declared lists, and the prepared covariate names. It never reads the fitted path |
| match rule | one candidate that holds each prepared covariate name once, in any order. A repeated candidate counts as two candidates. A repeated name is not the full set |
| admitted configurations | every configuration that `CTMLE` accepts: in sample and cross-fitted, two or more arms, fixed weights, `delta=` in sample, `ey`, `ate`, `rr`, `or`, a simultaneous band, `repeats`, and `n_bootstrap` |
| refit rule | a refit appends each covariate that `CausalData.with_extra_covariate` recorded to every candidate. So the `random_common_cause` refit of an admitted fit is the TMLE refit, and it keeps the status |
| still refused | the omitted-variable bound. It keys on the fitted method, and its sentence tells the caller to fit `TMLE` |
| evidence | `tests/unit/test_ctmle.py::TestEquivalenceWithPlainTmle` pins point, curve, standard error, interval, p-value, band, and bootstrap draws against `TMLE` to `1e-12` on ten configurations. `tests/unit/test_ctmle_candidate_sequences.py::test_an_empty_learners_list_is_the_ordinary_tmle` does the same for the empty `learners` list. The registered `TMLE` studies cover the identical estimator |

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
| `"per_arm"`, the default | for each arm $a$, a binary regression of $1\{A = a\}$ on the one column $\bar Q_n(a, W)$. Column $a$ holds $P(A = a \mid \bar Q_n(a, W))$, so the rows are not a distribution over the arms | `"generated_design_plugin"` on every fit. See [the applied revert](#the-applied-revert) |
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

The table states the construction of the per-arm curve. `plugin_std_error` and
`plugin_interval` report it as a diagnostic. No fit reports it as an interval, because condition
(v) fails for an estimated outcome regression.

| item | content |
| --- | --- |
| base result | Theorem 1 (preprint p. 9) and its construction, Section 3, steps 1-7 (pp. 8-9). Conditions (i)-(vi) and the proof are in Appendix F, "Details for Theorem 1" (pp. 28-33) |
| the diagnostic curve | $D(O \mid \bar Q_0, Q_{0,W}, G_0(\cdot \mid \bar Q_0))$, the influence function of Theorem 1, with $G_0(a \mid \bar Q_0) = P(A = a \mid \bar Q_0(a, W))$. It is not the efficient influence function. Its variance is at most the efficient variance |
| the generated-design term | the proof writes the remainder as $R_{21} + R_{22} + R_{23} + R_{24}$ (pp. 30-31). $R_{24}$ is the effect of the generated regressor. Lemma 1 and condition (v) bound it by $\lVert \bar Q_n - \bar Q_0 \rVert$, and condition (ii) makes the product second order. $R_{22}$ is condition (vi) and $R_{23}$ is condition (iii). Condition (v) fails for an estimated outcome regression, so $R_{24}$ is first order. See [the applied revert](#the-applied-revert) |
| the measurability step | the proof replaces $A$ by $G_0(\cdot \mid \bar Q_n, \bar Q_0)$ inside $P_0[(G_n - A) f]$. That step needs $f$ to be a function of $(\bar Q_n(W), \bar Q_0(W))$. In `cleverly`, $Q^*(a, W) = \operatorname{expit}(\operatorname{logit} \bar Q_n(a, W) + \epsilon_a / g_a(\bar Q_n(a, W)))$, and the fluctuation reads column $a$ only. So $Q^*(a, \cdot)$ is a function of $\bar Q_n(a, W)$, in sample and in each fold |
| (a) one arm mean | Theorem 1 directly. The design is the arm's own initial prediction |
| (b) every arm | an indicator reduction to $1\{A = a\}$ for each arm, as the Remark on p. 10 states: "repeating the entire procedure but switching the labeling of the treatment". Then a fixed-dimension stack of the $K$ curves on the same rows. The joint covariance comes from the stacked curves |
| (c) contrasts | linearity for `ate` and `par`. The delta method for `rr`, `or` and `paf`, with `rr` and `or` on the log scale. `ey_obs` has the curve $Y - E[Y]$. The default band is the multiplier band of the stack |
| (d) cross-fitted | Appendix D, "Cross-validated CTMLE" (p. 26): fold-trained $\bar Q_{n,v}$ and $G_{n,v}(\cdot \mid \bar Q_{n,v})$, and one coefficient pooled over the validation rows. Sample splitting replaces the Donsker part of condition (iv). The fit pools one coefficient per arm and has no fold-local update. The plug-in is the stacked whole-sample mean, not the mean of fold means. Near-equal folds make the difference $O(V/n)$ |
| (e) fixed weights | Theorem 1 under the weight-tilted law. The weighted projection gives the same identity, and the weight is a known bounded factor. Weights declared estimated take the `"estimated_weight_plugin"` status. The argument that an interval conditions on the weights concerns the efficient curve, and this curve is Theorem 1's. No cell of the registered study measures estimated weights |
| (f) repeated splits | each draw is asymptotically linear with the same split-free curve, so the median estimate is the estimator of Chernozhukov et al. (2018), equation (3.14). A default band beside `repeats > 1` is refused, as for `TMLE` |
| variance | in sample, the empirical variance of the reported curve. Cross-fitted, the variance of the stacked out-of-fold curve |
| conditions | (i) the curve equation is solved to $o_P(n^{-1/2})$. (ii) $\bar Q_n$ and $Q^*_n$ converge at $o_P(n^{-1/4})$ in $L_2(P_0)$. (iii) the treatment learner is consistent for $P(A = a \mid \bar Q_n(a, W))$ at $o_P(n^{-1/4})$. A misspecified learner makes $R_{23}$ first order whenever $Q^*_n - \bar Q_0 = O_P(n^{-1/2})$, which includes every correctly specified parametric outcome model. (iv) the curve converges in $L_2$, and in sample its class is Donsker. (v) $G_0(\cdot \mid \bar Q_n, \bar Q_0)$ is differentiable in $\bar Q_0$ with a bounded derivative. This condition fails for a root-$n$ outcome regression on a continuous $W$, as the next section states. (vi) the term from fitting $G$ on the initial $\bar Q_n$ is negligible. Positivity: the projection $P(A = a \mid \bar Q_0(a, W))$ lies inside `g_bounds`. Also iid rows, or fixed known weights, and a consistent outcome regression. A correct $g$ alone does not give consistency |
| the source's warnings | Section 4.1 (p. 12): "the estimated standard errors of CTMLE had poor performance, often underestimating the true variability of the estimator". The Discussion (pp. 17-18) calls the CTMLE "an irregular estimator, it may perform poorly for certain data generating distributions", and notes "the poor behavior of the confidence intervals in both simulations". The limit is pointwise at a fixed law. It is not locally uniform |
| the key's cases | `strategy="oat"`, `oat_design="per_arm"`, complete outcomes and treatment, no `strata=`, no weights or fixed weights, and `repeats >= 1`. In sample and cross-fitted, at any arm count. Estimands `ey`, `ey1`, `ey0`, `ate`, `rr`, `or`, `ey_obs`, `par` and `paf`. These fits withhold their interval while the revert flag is `False` |
| withheld | every `strategy="oat"` fit, of either design. [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) holds the generated-design term of both designs. A per-arm fit with `strata=`, `delta=` or `treatment_delta=` reports a point estimate under [the contract below](#baseline-strata-and-missing-data-in-the-per-arm-design), and the key excludes it even with the flag set |
| bootstrap | `n_bootstrap=` is a diagnostic on every `oat` fit. Theorem 1 does not cover the bootstrap of a superefficient estimator |
| key | `per_arm_design_admits(estimator, data)` in `cleverly.estimators.ctmle`. It reads the strategy, the design, the missing outcomes and treatment, the strata and the revert flag, and never a fitted array |

**The shared design publishes no inference.** It fits one categorical mechanism on the outcome
predictions of every arm. Theorem 1 covers one scalar design, and no reviewed result supplies the
vector influence function or the covariance of the shared construction. Cramér--Wold combines
scalar expansions that are already established. It cannot establish the missing ones. DOPE and
outcome-adapted AutoDML prove sample-split results with a representation learned on an
independent sample.

Every `oat_design="shared"` fit takes the `"generated_design_plugin"` status.
`ci`, `pvalue` and `std_error` raise `CapabilityError` with the reason of the status, and the
`summary()` column is `generated-design se`.

At `n = 1,000`, the registered point-treatment pair
resolves a finite-sample deficit in the standard-error ratio without invalid coverage. The
multi-arm pair resolves no deficit. Neither measurement identifies a first-order term. The
per-arm measurement does, because its study pins the design to the truth in a paired cell.
[F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) records the shared
design, the generated-design term and uniformity for both designs.
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
| [C-TMLE0 derivative covariate](method-evidence/ctmle-derivative.md) | `strategy="derivative"` against R `ctmle` 0.1.2 `ctmleGlmnet(ctmletype = 3)`, at a declared penalty pair and at each side's own penalty cross-validation. Both paired rows are equivalent. The both-wrong control is the one red cell, and the diagnosis finds no defect |
| [outcome-adaptive point-treatment C-TMLE](method-evidence/outcome-adaptive-point-treatment-c-tmle.md) | the shared design against the archived `ctmle3`, including a pinned-versus-estimated design pair that measures the finite-sample cost of estimating the design |
| [outcome-adaptive per-arm C-TMLE](method-evidence/outcome-adaptive-per-arm-c-tmle.md) | the per-arm design against R `drtmle` 1.1.2 with `adapt_g = TRUE`, by a binary recode per arm, on two laws with an arm-specific outcome index and an instrument. Its declared revert cells are red, so every per-arm fit withholds its interval. [The applied revert](#the-applied-revert) gives the cells |
| `tests/diagnostics/x17_generated_design/` | the generated-design term $c^\top \mathrm{IF}_\beta$ restores the standard-error ratio on the binary law |
| `tests/unit/test_outcome_adaptive_per_arm.py` | an exact law on which the two designs differ, the curve of Theorem 1 row by row, in-sample and cross-fitted longhands to `1e-8`, the joint covariance, and fixed weights against duplicated rows. Its class `TestEachMutationFailsItsCheck` applies 13 mutations by `monkeypatch`, and each one fails its check |
| `tests/unit/test_shared_oat_replay.py` | the shared design refits replication 0 of each registered `ctmle3` study to its committed rows |
| [estimator variants over registered targets](evidence.md#estimator-variants-over-registered-targets) | the candidate-path identities, the selection mutations, and the outcome-adaptive design witnesses |

### The applied revert

`OAT_PER_ARM_INFERENTIAL` in `cleverly.estimators.ctmle` is `False`. Every per-arm fit takes
`"generated_design_plugin"`, and the reason names condition (v) and the registered study.

The study `ctmle-oat-per-arm` declared its positive coverage and calibration cells as the cells
that the flag reads. Before the run, it declared that a red cell with no defect sets the flag to
`False`. The table gives the run at the declared budgets.

| cells | standard-error ratio | coverage |
| --- | --- | --- |
| primary, binary law, seven estimands | 0.78 to 0.85 | 0.888 to 0.916 |
| primary, three-arm law, nine estimands | 0.76 to 0.92 | 0.859 to 0.936 |
| calibration, binary, three-arm, weighted and repeated fits | 0.84 to 0.89 | 0.900 to 0.911 |
| generated design, estimated $\bar Q_n$ | 0.84 and 0.89 | 0.901 and 0.919 |
| generated design, $\bar Q_n$ pinned to the truth | 0.997 and 1.009 | 0.946 and 0.950 |
| the $n$ ladder, $n$ = 500, 2,000 and 8,000 | 0.83, 0.86 and 0.86 | 0.90 to 0.92 |
| type I error at the sharp null | 0.88 | 0.904 |
| the default band over the three-arm family | not applicable | 0.871 joint coverage |

The shipped fit builds no band. The study's band cell sets the flag for its own fit only, so it
measures the band that the per-arm design would build from its stacked curves.

The deficit does not shrink with $n$. R `drtmle` 1.1.2, with `adapt_g = TRUE`, agrees with
`cleverly` to `1e-8` in each estimate and standard error. The bias is equivalent to zero in each
cell. So the code computes the curve that the paper states, and that curve omits a term.

The cause is condition (v). For a root-$n$ outcome regression on a continuous $W$, the
derivative in that condition grows like $n^{1/2}$, so $R_{24}$ is first order. The remainder is
linear in $\bar Q_n - \bar Q_0$, with the factor $1 - g_0 / G$. For a parametric outcome
model, the omitted term is $c^\top \mathrm{IF}_\beta$, with
$c = E[(1 - g_0 / G)\, \partial \bar Q / \partial \beta]$. The paper reports the same
symptom: its Section 4.1 (p. 12) and its Discussion (pp. 17-18) state that the standard errors
underestimate the variability.

`tests/diagnostics/x17_generated_design/run.py` adds $c^\top \mathrm{IF}_\beta$ to the
reported curve, with the true $g_0$, on the binary law. Its `reading.txt` gives the result over
400 draws.

| fit | $n$ = 1,500: ratio, coverage | $n$ = 6,000: ratio, coverage |
| --- | --- | --- |
| per-arm, estimated $\bar Q_n$ | 0.876, 0.900 | 0.831, 0.905 |
| the same curve with $c^\top \mathrm{IF}_\beta$ | 1.034, 0.960 | 0.983, 0.950 |
| per-arm, $\bar Q_n$ pinned to the truth | 0.997, 0.938 | 0.988, 0.950 |

The term needs the full mechanism $g_0$ and the influence function of the outcome learner. The
per-arm design exists to avoid the first, and no source read here gives it for a flexible
learner. [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) holds it.
The point estimate stands, so the per-arm design stays the default.

### Baseline strata and missing data in the per-arm design

The per-arm design accepts `strata=`, `delta=` and, in sample, `treatment_delta=`. These fits
report point estimates and the `plugin_std_error` diagnostic. They report no interval, p-value or
standard error.

For arm $a$, with $q_a = \bar Q_n(a, W)$, the fit regresses each factor on the one column $q_a$.

| factor | regression | rows it is fitted on |
| --- | --- | --- |
| treatment observation, with `treatment_delta=` only | $\Delta_A$ on $q_a$ | every row |
| treatment | $1\{A = a\}$ on $q_a$ | the rows with a recorded treatment |
| outcome observation, with `delta=` | $\Delta$ on $q_a$ | the recorded rows of arm $a$ |

With `strata=`, the fit regresses each factor inside each stratum. By the chain rule, the
product is $P(C_a = 1 \mid q_a)$, or $P(C_a = 1 \mid q_a, S)$, for
$C_a = \Delta_A \Delta 1\{A = a\}$. This is the composite indicator of
[the DR-TMLE missing-data construction](dr-tmle/theorem.md), regressed on the design and not on
$W$. A factor whose target has one value on its rows predicts that value and fits no learner. The
fit uses the treatment learner for every factor.

The positivity report names the design as the condition of the product row, for example
`P(A=a,Delta=1|Qbar(a,W))`. With strata, the condition is `Qbar(a,W),S`.

| item | content |
| --- | --- |
| base construction | Benkeser, Cai and van der Laan (2020), Section 3, steps 1-7, for each arm, as the per-arm table above states it |
| what the point claims | consistency for each arm mean when $\bar Q_n(a, \cdot)$ is consistent for $\bar Q_0(a, \cdot)$ in $L_2(P_0)$. With $D(Q, G) = C_a / G\,(Y - Q) + Q - \Psi(Q)$ and $P_n D(Q^*_n, G_n) = 0$, the expansion is $\psi_n - \psi_0 = (P_n - P_0) D(Q^*_n, G_n) + R_2$, with $R_2 = P_0[(P(C_a = 1 \mid W) - G_n) / G_n\,(\bar Q_0 - Q^*_n)]$. The bounds `g_bounds` and `nuisance_bound` bound each factor below, so $R_2 \to 0$ by Cauchy-Schwarz. The empirical-process term vanishes for a Glivenko-Cantelli class in sample, and for bounded functions when cross-fitted. The one-step and linear variants are consistent through their plug-in form. `rr`, `or`, `par` and `paf` follow by the continuous mapping, strata by a finite partition, and `repeats > 1` as a median of consistent points |
| what the point does not claim | asymptotic linearity, a rate or an interval. Condition (v) fails for an estimated outcome regression, as [the applied revert](#the-applied-revert) states, and these fits inherit that failure. With $\bar Q_n$ inconsistent, the estimator has neither leg |
| baseline strata | a fixed finite partition with $P(S = s) > 0$. A stratum parameter is the parameter of the law given $S = s$. The update is a function of $(q_a, S)$. The marginal point is the $P_n(S = s)$ mixture. The marginal design conditions on $(q_a, S)$, so the marginal point differs from an unstratified fit. Stratified DR-TMLE uses the same convention |
| missing outcome | an indicator reduction to $O'_a = (W, C_a, C_a Y)$, with $\bar Q_0(W) = E[Y \mid C_a = 1, W]$. $E[C_a (Y - \bar Q_0(W)) \mid W] = 0$, so $P_0 D(\bar Q_0, G) = 0$ for every design $G$. Missingness at random enters through $\bar Q$ only |
| missing treatment | the same reduction with three factors. In sample only. The identification conditions are those of the DR-TMLE composite construction |
| natural course with a missing outcome | the construction with $\Delta$ as the treatment and the design $(A, \bar Q_n(A, W))$. The fit reads the outcome observation factor at the realised arm. `par` follows by linearity, and `paf` by the delta method on the stack with the arm means. With strata, the stratum step applies on top |
| identification | $Y(a) \perp A \mid W$; $Y(a) \perp \Delta_A \mid A, W$; $Y \perp \Delta \mid A, \Delta_A = 1, W$; $W$ recorded on every row; positivity $P(C_a = 1 \mid W) > 0$ almost surely, per stratum with strata. The design bounds do not imply positivity |
| the missingness tilt | `missingness_tilt` and `tipping_gamma` read the full-covariate model $P(\Delta = 1 \mid A, W)$, which the fit stores as `nuisance.tilt_missingness`. No targeting step reads it |
| refused | a cross-fitted fit with a missing treatment, as for every estimator ([F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants)); `ey_obs`, `par` and `paf` with a missing treatment, which are not identified; a stratum or a training complement in which a factor has no row to fit; with `delta=`, a training complement with no nonrespondent among its recorded rows, because the full-covariate response model of the tilt needs one; and, when the treatment learner is a package classification `SuperLearner`, a factor target with fewer than two rows of a class in a training complement. Each refusal reads the data and every drawn split before any learner. The selector strategies and the shared design keep their refusals |
| evidence | `tests/unit/test_outcome_adaptive_per_arm_extensions.py`: in-sample and cross-fitted longhands of the factors and the targeting to `1e-8`, a longhand of `target_weights=True`, each stratum point against a fit on the stratum's rows, the marginal mixture and its mass term, the solved scores, the natural course, the route identity, an exact law on which the factor product is the cell share of $C_a = 1$ and the oracle $\bar Q$ gives the truth, fold-local nesting, fixed weights against duplicated rows, the tilt, the status matrix, and 14 mutation controls. `tests/unit/test_capability_row_sweep.py` sweeps the capability rows of a per-arm fit with missing outcomes. The registered study `ctmle-oat-per-arm-strata-missing` is designed and waits in the study queue. It lands with its artifacts |
