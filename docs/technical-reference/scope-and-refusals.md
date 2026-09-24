# Scope and refusals

`cleverly` refuses a composition it has not derived, rather than returning a convenient
approximation to a different estimand. This page says how to read such a refusal, and it tabulates
the one axis where the answer is least obvious: which surfaces take more than two treatment arms.

## How to read a refusal

A refusal is always *by name*. The keyword is accepted and rejected with a stated reason. It does
not arrive as an `unexpected keyword argument` that names none.

Refusals are not all the same kind of thing. What you should do about one depends on where the
problem is, and there are three places it can be.

| section | where the problem is | what to do about it |
| --- | --- | --- |
| [Not written yet](#not-written-yet) | in this package | the parameter is well defined and nobody has written it here. Ask for it, compute it elsewhere, or contribute it. Proposed work is on the [roadmap](../roadmap.md) |
| [A different question](#a-different-question) | in the question | what was asked for is a different estimand, usually with its own identification assumptions. Decide which one you meant. No flag here produces the other, and one that quietly did would answer something nobody asked |
| [Wrong by construction](#wrong-by-construction) | in the method | the naive version *runs* and returns a plausible number that is wrong, usually with a known direction of error. Read these as warnings about the analysis, not about this package's coverage |

A fourth group needs no taxonomy. A fit whose *data* cannot support what you declared is refused
where the problem arises. Examples are a horizon at which no event was observed among a regimen's
followers, a cause with no events, a regimen nobody followed, and two absorbing causes firing at
one node. Those are statements about the sample.

Cross-fitting narrows what the sample supports. Each outer fold fits its regressions and its
mechanism on its training rows alone, so every fold must carry enough events for each declared
cause. A rare cause therefore needs more data under cross-fitting than under one fold. The split
reads no treatment and no outcome, so it cannot protect a rare level. The package checks the
realized draw instead, before the first learner, and the refusal names no redraw. The
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give every such message.

`make_longitudinal_competing(n=220, seed=41)` shows that boundary. It fits at `n_folds=1` and is
refused at `n_folds=2`, because one fold's training rows carry no event of one cause among the
regimen's followers. The message names the in-sample fit, or an estimand this fold count
supports.

### Not written yet

Nothing is wrong with wanting any of these. They are gaps in coverage, and the message says so
rather than implying the request was ill-posed.

| refused | where |
| --- | --- |
| missing-outcome `NaturalCourseMean` outside its two scalar TMLE contracts | [missing-outcome natural-course contracts](#missing-outcome-natural-course-contracts) lists every refusal. [Observed-data extensions](point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) defines both estimators |
| cross-fitted arm-indexed means and contrasts with missing outcomes outside the stacked CV-TMLE contract | [missing-outcome arm-indexed contract](#missing-outcome-arm-indexed-contract) lists every refusal. [Stacked CV-TMLE for arm-indexed targets](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets) defines the estimator |
| a cross-fitted shift, incremental, regime, MSM, or controlled-direct-effect fit with missing outcomes (`delta=`) | [the refusals a caller can meet](cv-tmle.md#the-refusals-a-caller-can-meet). No audit read a source for these fits. The fit raises `CapabilityError` before any learner is fitted. The in-sample fit with `delta=` remains available. [F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants) reopens it |
| `DRTMLE` with observational missing outcomes, cross-fitted missing outcomes at every `guard` including `guard=()`, missing treatment, `intermediate=`, fold-wise targeting, `treatment_probabilities=` under `n_bootstrap=`, composition with `CTMLE`, or `reduction="bivariate"` composed with `delta=` | [method presets](../user-guide/methods-learners.md#method-presets), and the [DR-TMLE refusals](dr-tmle/supported-estimands.md#refused-by-name) |
| the MNAR tilt on a `shifts=` fit | [modified treatment policies](../user-guide/estimands.md#modified-treatment-policies) |
| `intermediate=` and a multi-valued treatment with `incremental=` | [incremental interventions](../user-guide/estimands.md#incremental-propensity-score-interventions) |
| the targeted bootstrap and sample sensitivity-bound estimation for `LTMLE` | [longitudinal diagnostics](../user-guide/longitudinal.md#diagnostics). See [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) for the contracts Tan (2025) leaves open |
| longitudinal `msm=` with `n_folds > 1` | [MSM projections](msm-projections.md#the-longitudinal-projection). It needs an unsaturated projection property and a repeated-sampling study for coefficient inference |
| blocked-temporal and rolling-origin splits | [two fold layers](../user-guide/methods-learners.md#two-fold-layers) |
| a confidence interval, a p-value, or a standard error on a `CTMLE` fit whose `strategy` is `"greedy"`, `"ordered"`, or `"discrete"` | [collaborative TMLE](collaborative-tmle.md). No result shows the reported curve is the estimator's influence curve when the selected working mechanism is not consistent for the treatment law. `ci`, `pvalue`, and `std_error` raise `CapabilityError`. The point estimate, the selection path, and the curve remain, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. A `discrete` fit with one full-adjustment candidate is refused too, because the rule keys on the strategy. [F18](../roadmap.md#f18-selector-path-c-tmle-inference) reopens it |
| a confidence interval, a p-value, or a standard error on every `CTMLE` fit with `strategy="oat"`, including a fit with `delta=` and a fit that requests one arm mean | [collaborative TMLE](collaborative-tmle.md). The fit takes the `"generated_design_plugin"` status. Its mechanism is fitted on the outcome predictions of every arm, and every arm mean is targeted jointly. Theorem 1 of Benkeser, Cai and van der Laan (2020) covers one binary treatment-specific mean with one scalar design. The point estimate remains, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) reopens it |
| a confidence interval, a p-value, or a standard error on a `DRTMLE` fit with a non-empty `guard` and varying weights declared estimated (`weights_estimated=True`) | [DR-TMLE refusals](dr-tmle/supported-estimands.md#refused-by-name). The fit takes the `"estimated_weight_plugin"` status. The argument that an interval conditions on the weights concerns $D^*$, and not the reduced regressions. The point estimate remains, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. `guard=()` fits the ordinary TMLE and keeps its interval. Constant weights fit the unweighted estimator and keep it too. [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) reopens it |
| a confidence interval, a p-value, or a standard error on a saved cross-fitted `LTMLE` result with `id=` | [RM29](../roadmap.md#rm29-saved-cross-fitted-clustered-longitudinal-results). The result takes `"cross_fitted_longitudinal_plugin"` when it loads, at every cluster count and size. New fits refuse the design. The point estimate and plug-in diagnostic remain; saved bands and assessment answers are dropped. [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle) reopens it |
| a confidence interval, a p-value, or a standard error on a cross-fitted `TMLE` or `DRTMLE` fit with `id=` whose clusters differ in row count or weight mass, overall or within a reported baseline stratum | [clusters](inference.md#clusters). The fit takes the `"unequal_cluster_plugin"` status. The [grouped folds](cv-tmle.md#grouped-folds) argument and registered study cover equal sizes and masses. The point estimator stays row weighted at unequal sizes, but its cross-fitted interval lacks validation there. `plugin_std_error` and `plugin_interval` retain the diagnostic. The same clusters fitted in sample keep the interval with 40 or more contributing clusters. [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle) reopens it |
| a confidence interval, a p-value, or a standard error on a `TMLE`, `DRTMLE`, or in-sample `LTMLE` fit with `id=` and fewer than 40 positive-mass clusters in the fit or one reported baseline stratum | [clusters](inference.md#clusters). The fit takes the `"few_cluster_plugin"` status, in sample or cross-fitted. One such stratum withholds every interval. The package uses a normal reference. Nugent et al. (2024), Section 2.2, recommend a $t$ reference below 40 clusters, and no registered study covers few clusters. The point estimate remains, and `plugin_std_error` and `plugin_interval` retain the diagnostic. [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle) reopens it |
| a contrast, a simultaneous band, any E-value branch, `variable_importance()`, or `tipping_gamma(use_ci=True)` on the fits of the six rows above | [inference status](inference.md#inference-status). Each one is a confidence statement, or reads an interval that these fits do not supply. `variable_importance()` refuses at its entry point, because it adjusts one p-value per candidate. The E-value capability reports `unavailable` rather than raising inside the computation. `truncation_curve()`, `missingness_tilt()`, and the default `tipping_gamma()` still run. On a point-treatment fit the two frames rename their three spread columns. The longitudinal `truncation_curve()` publishes no spread. `LongitudinalResult.curve()` and `incidence_total()` rename their spread columns |
| `stratify_by="treatment"` and `stratify_by="treatment+outcome"` on any fit that draws a split, including selector-based C-TMLE at every setting | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). No shipped result covers a partition read off the data the fit then conditions on |
| a cross-fitted continuous outcome with `q_bounds=None`, for `TMLE`, `DRTMLE`, `CTMLE`, and `LTMLE` above one fold | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). The sample outcome range would take the scale from held-out rows |
| `CTMLE` with `id=`, at every `cross_fit` setting | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). No clustered result covers the outcome-adaptive mechanism. Selector-based fits also lack a result for grouped selection and nested folds or candidate-selection variance |
| `LTMLE` with `id=` above one fold | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). The cluster-robust variance of the targeted sequential recursion under a grouped draw is not established |
| the `subset` and `bootstrap_measurement_error` refutations on a fit that declared `split_plan=` | [reusable outer split plans](cv-tmle.md#reusable-outer-split-plans). Each one refits on rows the fit never ran, and no rule here says which fold labels those rows inherit. `refute()` raises `CapabilityError` before it refits anything |
| replicate weights (BRR, jackknife) | [observation weights](../user-guide/data-design.md#observation-weights-are-not-estimand-weights). These are a set of designs rather than one weight vector, so the shape they want is a refit per replicate outside the estimator |
| omitted-variable sensitivity after `repeats=` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The median bound needs an influence function; a coordinatewise median of per-draw influence terms is not one. The five omitted-variable capability rows declare this refusal as `unavailable` before any call, and `test_the_five_declared_rows_carry_that_same_reason` checks them |
| omitted-variable sensitivity on a `DRTMLE` or `CTMLE` fit | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). Neither estimator assumes a consistent treatment mechanism, and no derivation gives $\nu^2$ or the bound's standard error without that assumption |
| omitted-variable sensitivity on a fit with a response mechanism | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The bound is well posed, and the implementation is missing. The implemented representer omits the response indicator, and $\sigma^2$ averages the respondents only. Use the missingness tilt for response |
| omitted-variable sensitivity on a fit with an intermediate variable | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The bound is well posed, and the implementation is missing. The representer carries the intermediate weight, so the treatment strength $c_D$ would also measure the intermediate mechanism |
| omitted-variable sensitivity on a `regime`, `shift`, or `msm` parameter axis | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). Each parameter has a Riesz representer, so the bound is well posed. Only the implementation is missing |
| omitted-variable sensitivity on an `ipsi` parameter axis | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). An incremental intervention tilts the mechanism, so the mechanism is part of the estimand rather than a nuisance |
| omitted-variable sensitivity when the doubly robust estimate of $\nu^2$ is not positive | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). This row reads the data, not the fit class. The other omitted-variable rows refuse a whole class of fit before any computation, and their capability rows report `unavailable`. This one leaves the capability row `available`, and the call raises `CapabilityError` only when the fitted representer gives a nonpositive second moment. An ordinary TMLE script that received a bound before can therefore raise now, because the package substitutes no plug-in value for the refused one |
| any E-value request on a fit with an intermediate variable and a discrete treatment | [the E-value paths](validation-methods.md#e-value). No read source derives an E-value for a controlled direct effect. The E-value bounds the confounding of one exposure-outcome relation. A controlled direct effect also assumes no unmeasured confounding of the intermediate variable and the outcome. The capability row reports `unavailable` before any computation. A continuous-treatment fit with an intermediate variable reports `not_applicable`, because that check runs first. [F25](../roadmap.md#f25-e-value-for-a-controlled-direct-effect) reopens it |
| the standardized Gaussian E-value on a fit with a response mechanism | [the E-value paths](validation-methods.md#e-value). The conversion needs the population outcome standard deviation, and the respondents' one estimates a different quantity. A ratio E-value on the same fit stays available |
| correction diagnostics after ordinary TMLE or collaborative TMLE | those methods do not use the DR-TMLE correction system. A DR-TMLE fit with an empty guard also subtracts no correction term |
| a binomial ATT or ATC E-value conversion | the analysis needs a conditional baseline risk and a supported conditional risk-ratio target. The library implements neither |
| a fixed-baseline E-value on a reference-arm mean that its own standard error does not separate from zero | [the E-value paths](validation-methods.md#e-value). The conversion divides by that mean, so the ratio has no stable denominator |
| a fixed-baseline E-value whose risk difference is at or below the negative of the reference-arm mean | [the E-value paths](validation-methods.md#e-value). The difference implies a nonpositive risk in the contrast arm, and no risk ratio describes one |
| a cached-nuisance risk ratio after CV evaluation | the estimator refuses nonlinear ratio targets on this evaluation path. Post-fit retargeting keeps that boundary |
| a controlled direct risk ratio from cached nuisances | the fitted result can report a controlled direct risk ratio. The post-fit retarget path does not pass the fitted intermediate intervention level, so it cannot derive one from cached nuisances |

Which multi-arm surfaces are covered, and which five are not, is tabulated in one place:
[where a multi-valued treatment is supported](#where-a-multi-valued-treatment-is-supported).

Several former gaps have landed. `cleverly` now supports multi-valued longitudinal treatment
nodes. It supports multi-valued selector-based C-TMLE, outcome-adaptive C-TMLE, DR-TMLE, `ATT`,
and `ATC`. `LTMLE` supports observation weights and a working model over regimens. Shift fits
support `intermediate=` and weights. They support `delta=` in sample. `cleverly` also supports multi-arm
omitted-variable and MNAR sensitivity analyses. TMLE supports the scalar missing-outcome
natural-course mean with ordinary fitting or one generated split of stacked CV-TMLE.
[Missing-outcome natural-course contracts](#missing-outcome-natural-course-contracts) gives the
boundaries of each path.

TMLE also cross-fits arm-indexed means and contrasts with missing outcomes under one stacked CV-TMLE
contract.
[Missing-outcome arm-indexed contract](#missing-outcome-arm-indexed-contract) gives its
boundaries.

The remaining shift gap is narrower than it was. The tilt itself is written. The missing derivation
must establish whether the tilted parameter is still the shift parameter.

### Missing-outcome natural-course contracts

`NaturalCourseMean()` with `missingness=` and at least one missing outcome has two TMLE contracts.
`CrossFitting(enabled=...)` selects the contract. The table gives each requirement and the
compositions that each contract refuses.

| requirement | ordinary TMLE, `enabled=False` | stacked CV-TMLE, `enabled=True` |
| --- | --- | --- |
| target | `ey_obs` alone. A joint request is refused | `ey_obs` alone. A joint request is refused |
| estimator | `TMLE`. `CTMLE` and `DRTMLE` are refused | `TMLE`. `CTMLE` and `DRTMLE` are refused |
| treatment | two arms. A multi-valued treatment is refused | two arms. A multi-valued treatment is refused |
| outcome | binary, or continuous with declared `q_bounds`. A continuous outcome without `q_bounds` is refused | binary. A continuous outcome is refused |
| outer folds | one fold | package-generated folds with `stratify_by="none"` and `n_folds` of at least 2. One fold, stratified folds, and `split_plan=` are refused |
| repeats | `repeats=1`. A larger value fails when `CrossFitting` is constructed, because no split exists to repeat | `repeats=1`. A larger value is refused when the fit starts |
| targeting | `Targeting(fluctuation="logistic", algorithm="iterative", target_weights=False)`. A linear fluctuation, one-step targeting, and weighted targeting are refused | the same, with `targeting_scheme="pooled"` and `fold_evaluation=False`. Fold targeting and fold evaluation are refused |
| inference | influence-curve Wald interval. `n_bootstrap > 0` is refused | influence-curve Wald interval. `n_bootstrap > 0` is refused |
| rows | observation weights, clusters, baseline strata, and `intermediate=` are refused | observation weights, clusters, baseline strata, and `intermediate=` are refused |
| response support | no added check | at least two respondents and two nonrespondents in the sample, and one of each in every training complement |

Three checks enforce the two contracts. Each check runs at a different time and raises a different
error.

| check | error | when it runs |
| --- | --- | --- |
| a cross-fitting declaration that is invalid by itself | `MethodConfigurationError`. The engine raises `ValueError` with the same reason | when `CrossFitting` or `TMLEMethod` is constructed |
| a composition this target does not support | `CapabilityError` | when the fit starts, before any learner is fitted |
| response support | `DataError` | after fold generation, before either learner receives data |

The declaration check is not specific to this target. One function runs its steps in a fixed order,
and it returns the first reason it finds. The declaration and the engine therefore give one reason
for one input. A reason that names the cross-fitting switch names `enabled` from the declaration
and `cross_fit` from the engine.
`tests/unit/test_split_plan.py::TestOneInputEarnsOneReasonAtBothLayers` checks that both layers
give the same reason.

| order | the declaration check refuses |
| ---: | --- |
| 1 | `repeats` below 1 |
| 2 | a `split_plan` that is not a `SplitPlan` |
| 3 | a plan that records no generator. See [reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) for the two plans that carry the record |
| 4 | a plan that the declared fold policy cannot use. A plan requires cross-fitting with at least two folds, and its fold and repeat counts must fit the declaration |
| 5 | a plan together with `n_bootstrap` above 0. `TMLEMethod` runs this step, because it holds both settings |
| 6 | `repeats` above 1 with cross-fitting disabled |
| 7 | cross-fitting declared with fewer than two folds |
| 8 | a fold-stratification policy this package draws no split under. `stratify_by` other than `"none"` is refused whenever the fit draws a split, including selector-based C-TMLE at every setting |

A `split_plan` can pass the declaration check and still fail the natural-course contract. Under
`enabled=True`, a valid plan constructs, and the stacked contract refuses it when the fit starts.
Under `enabled=False`, step 4 refuses any plan at construction.

| response-support failure | what the message tells you to do |
| --- | --- |
| the sample holds fewer than two respondents or fewer than two nonrespondents | use the in-sample estimator. No fold count or `random_state` can succeed |
| one training complement holds no respondent or no nonrespondent | the message names the repeat and the fold. Use the in-sample estimator, or collect more observations at the rare level |

Both messages name the in-sample estimator as `CrossFitting(enabled=False)`, or `cross_fit=False`
on the engine. One fold balances nothing, so no fold policy is part of the remedy. Neither message
names a redraw, because the split reads neither the response indicator nor the outcome.

`stratify_by="none"` is the default, and it is the only policy any fit that draws a split accepts.
It is not special to these two contracts. `"treatment"` and `"treatment+outcome"` are refused when
the fit draws a split, and at every setting for selector-based C-TMLE, whose search draws selection
folds without cross-fitting. An in-sample outcome-adaptive fit draws no split. It accepts the
declaration and applies none of it. The
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give the audit behind the
refusal and the message each layer raises.

### Missing-outcome arm-indexed contract

The contract covers the arm-indexed mean group `ey`, `ey0`, `ey1`, `ate`, `rr`, `or`, `att`, and
`atc` under cross-fitting with at least one missing outcome. It applies to an ordinary TMLE or
C-TMLE fit under three conditions. The parameters are indexed by the arms of a discrete treatment.
The design declares no intermediate. The request contains no `NaturalCourseMean`, PAR, or PAF.
[Stacked CV-TMLE for arm-indexed targets](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets)
defines the estimator, its preflight, and its evidence.

| requirement | admitted | refused |
| --- | --- | --- |
| estimator | `TMLE` or `TMLEMethod` | `CTMLE` or `CollaborativeTMLEMethod`. `DRTMLE` raises its own refusal at every `guard`, including `guard=()` |
| targets | `ey`, `ey0`, `ey1`, and `ate`. `rr` and `or` for a binary outcome | `att` and `atc`, also when the default list or `"all"` includes them. The message lists the admitted estimands. A continuous `rr` or `or` stays refused, as for every fit |
| treatment | two or more arms | none |
| outcome | binary, or continuous with a fixed `q_bounds` equal to the known support | continuous with `q_bounds=None` |
| outer folds | package-generated folds with `stratify_by="none"` and `n_folds` of at least 2 | `split_plan=`. A stratified policy and one fold are refused earlier, by the declaration check |
| repeats and targeting | `repeats=1` and `targeting_scheme="pooled"` | `repeats` above 1 and `targeting_scheme="fold"` |
| fluctuation | `Targeting(fluctuation="logistic", algorithm="iterative", target_weights=False)` and `fold_evaluation=False` | a linear fluctuation, one-step targeting, weighted targeting, and fold evaluation |
| rows | unweighted iid rows | observation weights, clusters, and baseline strata |
| inference | pointwise influence-curve Wald intervals and the simultaneous band | `n_bootstrap > 0` |
| content | at least the minimum in the [preflight table](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets) | a sample or a training complement below that minimum. A sample below its minimum is refused with a remedy that does not repartition, because no partition can succeed |

Three checks enforce the contract. Each check runs at a different time and raises a different
error.

| check | error | when it runs |
| --- | --- | --- |
| a composition outside the contract | `CapabilityError` | when the fit starts, before fold generation and before any learner is fitted |
| `DRTMLE` with `delta=` and `cross_fit=True` | `NotImplementedError` | when the fit starts, before any learner is fitted |
| minimum content | `DataError` | after fold generation, before any learner is fitted |

A `DRTMLE` fit with missing outcomes and `cross_fit=True` raises the `NotImplementedError` in the
table at every `guard`, including `guard=()`, and under every fold policy.
`tests/unit/test_drtmle_missing.py` checks the `guard=()` case with zero learner calls.

The composition check runs its steps in a fixed order. A fit that breaks several rules receives
the first one. Each message names the missing result and the setting that the contract admits.
A setting message gives both the engine keyword and the public spelling. `tests/unit/test_arm_indexed_stacked_mar.py` checks each step through the engine and
through `TMLEMethod`, with zero learner calls.

| order | the composition check refuses |
| ---: | --- |
| 1 | C-TMLE |
| 2 | `fluctuation="linear"` |
| 3 | `algorithm="one_step"` |
| 4 | `target_weights=True` |
| 5 | `fold_evaluation=True` |
| 6 | `att` or `atc` |
| 7 | a continuous outcome with `q_bounds=None` |
| 8 | `split_plan=` |
| 9 | `repeats` above 1 |
| 10 | `targeting_scheme="fold"` |
| 11 | observation weights |
| 12 | clusters |
| 13 | baseline strata |
| 14 | `n_bootstrap > 0` |

The list once held a stratified fold policy and a fold count below two. The declaration check
refuses both, so neither step can run. A restored result or a copied estimator can still carry the
old policy, and the fit then raises `ValueError` from the declaration check's own reason.

A refusal that has an in-sample alternative names it as `CrossFitting(enabled=False)`. The engine
form is `cross_fit=False`. An in-sample fit that draws no split keeps whatever fold policy the
declaration carries and applies none of it. Selector-based C-TMLE draws selection folds at every
setting, so it refuses a stratified policy in sample as well.

### Replay-only unavailability

The longitudinal truncation grid can be scientifically supported while one saved result cannot
replay it. `result.replayability.unreconstructible` carries the codes below as data. The capability
row states the same codes inside its prose `reason`, so read the attribute when you want them as
data. `cleverly` exports no name for the code literals, so compare a code against the spelling in
the table. The fitted-bound equality check is different: it can only run when the curve is invoked.

Replay accepts an outcome or pseudo-outcome learner only when every `random_state` parameter it
exposes is an explicit integer. The check reads that learner, each nested learner, and each member
of a Super Learner library. It reads the declared parameter and not the fitted behavior, so a
declared `random_state=None` is refused even where the fit is deterministic.

| the learner you pass | what replay does |
| --- | --- |
| `LinearRegression()` | accepts it, because the class declares no `random_state` |
| `LogisticRegression(random_state=0)` | accepts it, because the declared state is an explicit integer |
| `LogisticRegression()` | refuses it as `longitudinal_replay_random_state_unseeded` |
| `LogisticRegression(random_state=rng)`, for a generator `rng` | refuses it as `longitudinal_replay_random_state_non_integer` |

`tests/unit/test_longitudinal_truncation_refit.py` checks an unseeded learner, a non-integer state,
an unseeded member of a nested Super Learner library, and a seeded stochastic learner that replays
exactly.

| omission code | replay boundary |
| --- | --- |
| `longitudinal_replay_recipe_missing` | the result predates the stored recursive replay recipe |
| `longitudinal_replay_learner_unclonable` | an outcome or pseudo-outcome learner cannot be cloned |
| `longitudinal_replay_random_state_unseeded` | a replayed learner or nested learner declares an unspecified random state |
| `longitudinal_replay_random_state_non_integer` | a replayed learner or nested learner declares a non-integer random state |
| `longitudinal_replay_fitted_bound_mismatch` | invocation-time preflight found that a full replay at the fitted bound differs from the stored fitted artifacts |

The static recipe omissions make the capability row `unavailable`. A fitted-bound mismatch instead
raises a stable `CapabilityError` from that invocation; it is not known when the capability row is
built. Missing `bounds` or the refit opt-in makes a combined report `deferred`, because the caller
can supply either request.

### A different question

These are well-posed parameters, but the selected estimator does not target them. No setting turns
one into the other.

| refused | the question it would answer instead |
| --- | --- |
| `eliminate=` on a competing-risks fit | the incidence of a cause if the competing events were *removed*. That intervenes on them rather than conditioning on the history. It needs a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions for the competing event. What is reported instead is the incidence with the competing causes left alone |
| `intermediate=` on `LTMLE` | a controlled direct effect fixes a mediator at one time point. Over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| `ey1` and `ey_regime` from one fit; `msm=` with `interventions=` or `shifts=` | each keyword declares what "counterfactual" means for the fit, or how the counterfactuals are summarised. One fluctuation solves one set of score equations, so a fit reporting parameters from two axes would put two of them under one heading |
| the per-arm propensity table on a continuous fit | a per-arm table has no rows when there are no arms. `diagnostics.support()` is not itself refused. On a fit that declared `shifts=` it dispatches to the question that does apply, which is whether the density *ratio* stays bounded |
| `res.sensitivity`, `res.diagnostics` and `res.validate()` on an `LTMLE` result | each is part of the shared result contract. Stagewise support, scores, nuisance loss, and the descriptive full-recursion truncation grid are supported. Sensitivity operations without a longitudinal derivation report `unavailable`. `res.save()` is supported, and [persistence and replayability](../user-guide/results-assessment.md#persistence-and-replayability) states its contract |
| a non-empty `assess(arguments=...)` block for `score_equations` | the validation battery owns that name and runs it argument-free. The battery presents one row per name, and it presents the validation row. A caller's tolerance would be computed and then hidden, so a check that failed at that tolerance would never reach `attention`. Call `res.diagnostics.run_all(arguments=...)`, or the operation itself. The battery also owns `support` and `nuisance_models`, which accept no argument, so an argument for either is a `TypeError` from the signature |

### Wrong by construction

These are worth reading even if you never reach for the keyword. Most are mistakes that are easy to
make by hand, in any framework, and none announces itself. They can produce a number for the wrong
target or an interval whose stated conditions are not met.

| refused | what goes wrong if it is done anyway |
| --- | --- |
| a population-mechanism odds tilt supplied as a `Stochastic` density | the population-law target depends on `P` through `g*`. Its curve needs a derivative that the regime curve lacks. On one exact law, the reported standard error is 0.62 of the exact one at `delta = 2` (`tests/unit/test_stochastic_regime_densities.py`). A realized learned density instead defines a data-adaptive target; its inference needs conditions this API does not check. `Stochastic` requires `density_kind="known"` and refuses `None` or `"estimated"` at construction and fit. A false `"known"` declaration still fits because code cannot inspect a closure. [RM25](../roadmap.md#rm25-declared-stochastic-regime-densities) records the witness and both targets |
| a population-law incremental intervention built by hand as a `Stochastic` regime | the missing term has variance `Var(delta * (Qbar(1,W) - Qbar(0,W)) / D^2 * (A - g))`. For that target, omitting the term understates variance wherever this variance is positive. `Stochastic` refuses the tilt when `density_kind` is `None` or `"estimated"`. `TMLE(incremental=)` fits the population odds tilt and includes the term |
| a user-written `Intervention` class whose density depends on the population mechanism | it can omit the same derivative while reporting an interval for the population-law target. On the RM25 witness law, a class that computes the sample-mechanism odds tilt reports 0.62 of that target's exact standard error (`tests/unit/test_rule_and_intervention_declarations.py`). A realized learned density has a different inference contract. The class must carry `density_kind = "known"`. `TMLE` refuses `None`, a missing attribute, and `"estimated"` before any learner. `RegimeSet.evaluate` refuses them before `density` runs. A false `"known"` declaration still fits because code cannot inspect what `density` computes. [RM28](../roadmap.md#rm28-declared-densities-of-user-written-interventions) records the witness |
| a `Rule` or `DynamicRegimen` learned from the analysis sample | the fit treats its rule as fixed. A learned policy can instead target a realized policy value or a population-indexed policy. Those targets need distinct inference conditions; an optimal rule can also be nonregular at ties. For a threshold at the sample mean of a continuous covariate, the curve omits a derivative through that mean. On one exact law, the reported standard error is 0.73 of the exact one at one node (`tests/unit/test_rule_and_intervention_declarations.py`). It is 0.77 for a two-node regimen (`tests/unit/test_regimen_rule_declarations.py`). `Rule` and `DynamicRegimen` require `rule_kind="known"` and refuse `None` or `"estimated"` at construction and fit. A plan of labels alone needs no declaration. `LTMLE` refuses a callable written inline in `regimens=`, because it carries no declaration. A false `"known"` declaration still fits because code cannot inspect a closure. [RM28](../roadmap.md#rm28-declared-densities-of-user-written-interventions) records the witnesses |
| a shift's inference taken from the regime inducing the same density | the means and the clever covariates agree entry for entry. The curves do not. The gap is `Var(Qbar(d(A,W),W) - E[Qbar(d(A,W),W) | W])`. Too small, always |
| a shift fit run on the complete cases when outcomes are missing | it is an ordinary shift fit on a *different* joint law of `(A, W)`, so it converges to a different number, and nothing in its own output says so. Measured on the dose fixture at 0.17, four standard errors, with a mechanism whose slopes are mild. `delta=` is what corrects it |
| a missingness or intermediate mechanism read at the observed dose rather than the assigned one | the fluctuation updates `Qbar` as a function of the dose, so `Qbar*(d(A,W),W)` is the update evaluated where the policy sends the unit. Silent wherever the mechanism does not depend on the dose, and invisible to a Gateaux check on an exact law |
| a "stabilised" MSM weighting `h` by the estimated mechanism | the same argument once more. `h` becomes a functional of `P`, and a term goes missing from the influence curve. `MSM` refuses `weights_kind="estimated"` and a `weights=` callable with no declaration. [MSM projections](msm-projections.md#variations) gives the declaration |
| an MSM `design` that reads a sample statistic, such as a covariate centred at its sample mean | for the population-law target centred at $E_P[W]$, the projection depends on $P$ through the design. Its reported curve omits that derivative. On one exact law, the reported intercept standard error is 0.893 of the exact one (`tests/unit/test_msm_design_declaration.py`). An interval conditional on the centre learned from the same rows also needs separate validation. `MSM` refuses `design_kind="estimated"` and an undeclared callable. A design built by `MSM.linear` needs no declaration. Code cannot detect a false `"known"` declaration. [MSM projections](msm-projections.md#variations) gives the declaration, and [RM27](../roadmap.md#rm27-declared-msm-design-functions) records the witness |
| `g_bounds=` or `truncation_curve()` on an `incremental=` fit | `g` is *inside* the estimand, so truncating it moves the parameter rather than regularising a denominator. The result is a number for a parameter nobody declared |
| a `cap=` fitted from the data on a shift | the estimand becomes data-dependent. The interval conditions on an estimated boundary, and every bootstrap replicate targets a slightly different policy |
| `CTMLE` on an `incremental=` fit | each candidate `ghat` defines a different estimand, so the cross-validated search selects between *estimands* rather than between estimators |
| splitting a cluster across folds to buy more of them | the out-of-fold predictions stop being independent of the rows they are used on, and the standard error shrinks in exactly the direction `id=` was passed to prevent |
| a supplied `SplitPlan` whose labels no recorded draw produces | the labels could have been chosen by reading the outcome, the treatment, or a covariate, which is the leak cross-fitting exists to prevent. The fit draws each repeat again from the plan's record and compares it label for label. [Reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) states what a plan records |
| reusing a supplied `SplitPlan` on rows it was not realized on | a fold label is a row position. A reordering, a replaced column, or an added covariate leaves every label pointing at a different unit, while the row count still agrees. The out-of-fold guarantee is gone, and nothing in the output says so. [Reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) states how a plan binds to its rows |
| joint covariance, post-fit contrasts, or simultaneous bands after `repeats=` | coordinatewise medians do not preserve linear identities among estimates; a central-draw curve also does not represent the split-adjusted median estimator needed for a multiplier band |
| a cross-validated variance for the curve a repeated fit retains | at equal fold sizes it collapses to the pooled uncentred second moment for *every* partition, so the partition carries no information. That rule was rejected, and the split-adjusted median variance replaced it |
| a one-shot non-identity-link MSM | the derivative of the inverse link depends on the coefficient, so a single pass reports a standard error for an equation it did not solve. The link is supported. What is refused is skipping the alternation it needs |
| frequency (count) weights | they assert a sample size the variance does not use. Expand the rows instead, which says the same thing where every part of the fit can see it |
| an `LTMLE` outcome missing for a reason other than censoring | its probability of being observed is silently taken to be one. Encode it as a final censoring column so that it is estimated and enters the cumulative product |
| a binary-only target on a multi-arm fit | it would report a contrast of arms `0` and `1` out of five, under the name of a parameter about all of them. Targets declare `requires_binary_treatment` for this |
| `MSM.linear` on non-numeric arm labels | a model linear in the arm reads it as a dose to interpolate between, and the fallback coding is the sort order. That is a dose scale nobody chose |

The first three rows share one mechanism, and it generalises past this package. **If an
intervention's density is a functional of `P`, the influence function carries its pathwise
derivative.** Omitting that term changes the variance. The direction depends on its covariance
with the retained curve. The MSM rows on a "stabilised" weighting and on a sample-statistic
`design` apply the same argument to a working-model function. The incremental and shift rows above give their own variance
gaps. The rule row shares the
mechanism when a population quantity indexes the rule, as in the threshold witness.

Two more share another. **A nuisance that sits inside the estimand is not a knob.** Truncating `g`
on an incremental fit, or fitting a shift's `cap` from the data, moves the target rather than
regularising the estimator.

Three method entries carry their own detailed refusal tables, each with a `kind` column:
[marginal structural models](msm-projections.md#variations),
[longitudinal TMLE](longitudinal-tmle.md#variations), and
[the simulated common-cause surface](validation-methods.md#simulated-common-cause-stress-surface).
The column names which of the three sections above the row belongs to. The simulated common-cause
table also uses `waiting on published theory` from the
[roadmap's eligibility rules](../roadmap.md#eligibility).

The simulated common-cause surface refuses six compositions for the whole fit. They are a
longitudinal fit, multi-arm treatment, a missing outcome, an intermediate variable, estimated
observation weights, and a clustered fit. Each one waits on published theory, so none of the three
sections above covers it. The
[future investigations grid](../roadmap.md#future-investigations) tracks each stop against the
result a paper must supply.

The surface accepts fixed probability weights under ordinary TMLE, under binary complete-outcome
collaborative TMLE, and under binary complete-outcome DR-TMLE. Read
[the surface's own refusal table](validation-methods.md#simulated-common-cause-stress-surface) for
every composition it refuses and the `kind` of each one.

## Where a multi-valued treatment is supported

A multinomial treatment mechanism is the default construction here rather than a variant of a
binary one. `CausalData` carries `treatment_levels`, `Propensity` holds an `(n, K)` simplex, and
`learners._fitting.predict_probabilities` produces it. Two arms are the `K = 2` case of that
construction and not a separate path, which is what the
[bit-for-bit invariant](../architecture-invariants.md#dataframes-and-labels) is about.

So "does this estimator take more than two arms" usually has the answer "yes, through the same code
as two". The informative entries are the five that do not. The `status` column uses the vocabulary
of [How to read a refusal](#how-to-read-a-refusal) above, plus `waiting on published theory` from
the [roadmap's eligibility rules](../roadmap.md#eligibility).

| surface | status | why |
| --- | --- | --- |
| `TMLE`: `ey` per arm, `ate` / `att` / `atc` per non-reference arm, regimes, MSMs over arms | supported | one counterfactual mean per arm and one contrast per non-reference arm. The [oracle-law gate](validation-methods.md#the-oracle-law-gate) states that a target meant for more than two arms needs a branch on the three-armed law |
| `DRTMLE`: univariate and bivariate reductions | supported | [armwise one-vs-rest](dr-tmle/index.md#variations), with each reduction and correction indexed by a free level. The cited theorem is binary, so this is an implementation-backed armwise extension rather than a claim about that theorem's literal scope |
| `CTMLE`: selectors and `strategy="oat"` | supported | one shared `n x K` categorical mechanism, selected against one nonredundant vector. See the [standing decision](../architecture-invariants.md#targets-interventions-and-variants) |
| `LTMLE`: categorical nodes, static and dynamic regimens | supported | [treatment over time](longitudinal-tmle.md#the-algorithm-as-implemented). Each node owns its level set, and the clever covariate selects the assigned label's probability |
| positivity, omitted-variable, E-value and MNAR sensitivity | supported | each is one parameter per contrast, and each reads its arms from the parameter's structured index rather than assuming two |
| simulated common-cause sensitivity | supported for binary and continuous treatment; named theory stops cover the remaining fit-wide gaps | The binary surface accepts marginal and baseline-stratum arm means, ATE, and ratios under ordinary TMLE and C-TMLE. Ordinary TMLE also accepts ATT and ATC with perturbed group membership. Ordinary TMLE alone accepts marginal and baseline-stratum PAR, PAF, fixed-regime parameters, and identity-link MSM coefficients. It also accepts marginal incremental targets and nonlinear MSM coefficients. Complete-outcome DR-TMLE supports marginal arm means, ATE, and ratios only. The continuous surface accepts marginal and baseline-stratum modified-policy means and contrasts under ordinary TMLE. Continuous MSMs require marginal fits. Every listed row accepts fixed probability weights. Longitudinal, multi-arm, missing-outcome, intermediate, estimated-weight, and clustered fits report unavailable before work begins. Each of those six waits on published theory, and so does logical categorical calibration. `NaturalCourseMean`, the zero-delta policy mean, and the multiplier-one incremental mean remain refused. See the [population contract and refusal table](validation-methods.md#simulated-common-cause-stress-surface) |
| `ey1` / `ey0` and the incremental estimands, on a multi-arm fit | wrong by construction | they *name* one of exactly two arms, so on five arms they would report a contrast of arms `0` and `1` under the name of a parameter about all of them. Declared by `requires_binary_treatment`. The multi-arm path reports per-arm `ey` instead |
| `incremental=` itself, above two arms | a different question | an odds multiplier names two arms. One odds per contrast is well posed, and it is a *different intervention* with a different influence function rather than a generalisation of this one |
| stochastic categorical policies and continuous doses at a longitudinal node | a different question | both change the intervention *density* rather than which label is assigned, so neither is the parameter the sequential regression identifies |
| `DRTMLE` with `delta=` at more than two arms | waiting on published theory | Diaz and van der Laan's missing-outcome theorem is stated for a binary randomized treatment, and the per-arm assembly of its observation, treatment and outcome correction blocks is not in it. See the [future investigation](../roadmap.md#f4-multi-arm-missing-outcome-dr-tmle) |

A source could close two entries as they stand. The first is the multi-arm part of the simulated
common-cause row. The second is the `DRTMLE` with `delta=` row. Neither `a different question` row
would be closed by a source. Each would be answered by a different estimand, with its own
derivation, oracle law, and evidence.
