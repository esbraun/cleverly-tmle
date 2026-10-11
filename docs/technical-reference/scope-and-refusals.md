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
where the problem arises. Examples are a regimen nobody followed and two absorbing causes firing
at one node. A known policy that can draw a level no fitted row of its node received is another, and
[known stochastic policies](longitudinal-tmle.md#known-stochastic-policies) states why. Those are
statements about the sample.

A horizon at which no follower of a regimen had the event is not refused. Its regression is
zero, the maximum-likelihood hazard, and
[a node with no event](longitudinal-tmle.md#a-node-with-no-event) gives the rule.

Cross-fitting narrows what the sample supports. Each outer fold fits its regressions and its
mechanism on its training rows alone, so every fold must carry the followers and treatment levels
that each node needs. The split reads no treatment and no outcome, so it cannot protect a rare
level. The package checks the realized draw instead, before the first learner, and the refusal
names no redraw. The [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give
every such message.

### Not written yet

Nothing is wrong with wanting any of these. They are gaps in coverage, and the message says so
rather than implying the request was ill-posed.

| refused | where |
| --- | --- |
| missing-outcome `NaturalCourseMean`, PAR or PAF outside their two TMLE contracts | [missing-outcome natural-course contracts](#missing-outcome-natural-course-contracts) lists every refusal. [Observed-data extensions](point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) defines both estimators, and [PAR and PAF with missing outcomes](point-treatment-tmle.md#par-and-paf-with-missing-outcomes) defines the stack |
| cross-fitted arm-indexed means and contrasts with missing outcomes outside the stacked CV-TMLE contract | [missing-outcome arm-indexed contract](#missing-outcome-arm-indexed-contract) lists every refusal. [Stacked CV-TMLE for arm-indexed targets](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets) defines the estimator |
| a cross-fitted shift, incremental, regime, MSM, or controlled-direct-effect fit with missing outcomes (`delta=`) | [the refusals a caller can meet](cv-tmle.md#the-refusals-a-caller-can-meet). No audit read a source for these fits. The fit raises `CapabilityError` before any learner is fitted. The in-sample fit with `delta=` remains available, except for a continuous-dose MSM, which the next row refuses. [F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants) reopens it |
| a continuous-dose MSM with missing outcomes (`delta=`, or `PointTreatment(missingness=...)`) or with `intermediate=`, in sample or cross-fitted | [MSM projections](msm-projections.md#variations). The clever covariate that the package builds divides by the treatment density at each grid dose. In that construction, this composition also needs the response or intermediate mechanism at each grid dose, and the package does not write it. `TMLE` raises `CapabilityError` before any learner. `CausalStudy.identify` raises it for `MSMProjection` with a missing outcome. `tests/unit/test_continuous_msm_mechanism_refusals.py` checks both. [X10](../roadmap.md#x10-continuous-dose-msm-with-a-second-mechanism) reopens it |
| `DRTMLE` with cross-fitted missing outcomes at every `guard` including `guard=()`, `intermediate=`, fold-wise targeting, composition with `CTMLE`, `reduction="bivariate"` composed with `randomized=True`, `evaluation=` or `reduced_crossfit="nested"` on the composite indicator, or `targeting="one_step"` with `reduced_crossfit="nested"` at a non-empty `guard` | [method presets](../user-guide/methods-learners.md#method-presets), and the [DR-TMLE refusals](dr-tmle/supported-estimands.md#refused-by-name) |
| a declared missing treatment (`treatment_delta=`) with `att`, `atc`, `ey_obs`, `par`, `paf`, `incremental=`, `policies=`, `learned_rule=`, `intermediate=`, `cross_fit=True`, `randomized=True`, a declared known mechanism, or `CTMLE` | [the composite indicator](dr-tmle/theorem.md#observational-missing-data-the-composite-indicator). The recording may depend on the treatment, so `P(A = a \| W)` is not identified, and the first seven read it. No composite derivation is written for a learned rule ([F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions)) or a controlled direct effect. One gate in the shared preflight refuses each one by name before any learner. `tests/unit/test_refusals_before_the_nuisance_fit.py` pins each refusal and a mutation that removes the gate. Cross-fitting waits on [F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants), and C-TMLE on [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| an undeclared missing treatment value | the package does not infer a missing treatment from a missing value. `CausalData` raises `DataError` before any learner and names the declaration |
| a declared known treatment mechanism beside a continuous treatment or `policies=` with `ratio="classifier"`, a declared `LTMLE` design that leaves a regimen unidentified, and `n_bootstrap=` on a guarded `DRTMLE` with `delta=` on declared data | [known treatment mechanism](point-treatment-tmle.md#known-treatment-mechanism). A known density is not supported, and the classifier learns the ratio that the declaration fixes. A declared zero on the arm a regimen needs leaves no unit able to follow it. The [DR-TMLE contract](dr-tmle/theorem.md#a-known-treatment-mechanism) gives the bootstrap reason: the observation mechanism is still estimated, so the remainder of that route is linear in its error. Each refuses before any learner. `tests/unit/test_known_treatment_mechanism.py` and `tests/unit/test_known_node_mechanisms.py` pin the texts |
| baseline strata (`strata=`) with `targeting_scheme="fold"` or `cv_evaluation=True`, an incremental target with a stratum that lacks a treatment arm, an MSM whose design is singular inside a stratum, or a `DRTMLE` stratum with no trainable rows in some training complement | [weights, strata, and clusters](point-treatment-tmle.md#weights-strata-and-clusters). Fold targeting needs a fold-local update, and no published result covers one. Fold evaluation needs stratum shares and a stratum-indexed fold average in each validation fold ([X28](../roadmap.md#x28-fold-evaluated-cv-tmle-with-baseline-strata)). The other three are data conditions with no finite root or no fittable regression. Each refuses before any learner. `tests/unit/test_refusals_before_the_nuisance_fit.py`, `tests/unit/test_stratified_incremental_exact.py` and `tests/unit/test_stratified_msm_exact.py` check them |
| the MNAR tilt, `missingness_tilt()` and `tipping_gamma()`, on a shift, incremental, regime, MSM, ratio-only, or PAR-and-PAF-only fit with missing outcomes | [missingness tilt and tipping gamma](validation-methods.md#missingness-tilt-and-tipping-gamma). The tilt re-mixes the arm-indexed means and their linear contrasts. No derivation here covers the tilt of these parameters. Both entry points raise the sentence of `fit_wide_tilt_refusal`. Both capability rows read `unavailable` with that sentence before any call. `TestTheTiltRowsReadTheCallsPredicate` in `tests/unit/test_capability_row_predicates.py` checks each kind |
| a learned-rule value from an in-sample or one-fold fit, the stacked report, fold-specific targeting, a categorical treatment, or a contrast with a known regime or an arm, `reference=` included | [learned rules](point-treatment-tmle.md#learned-rules). Each request has a published source and no implementation here. `TMLE` raises `CapabilityError` before any learner. `CausalStudy.identify` raises it for a categorical treatment, and `IdentifiedEffect.estimate` for a setting. `CrossFitting(n_folds=1)` raises `MethodConfigurationError` at construction, before the estimand is known. [The refusals a caller can meet](cv-tmle.md#the-refusals-a-caller-can-meet) quotes each remedy. [X11](../roadmap.md#x11-learned-policy-follow-ups) reopens them |
| a learned-rule value with a continuous treatment, `intermediate=`, `weights=`, `id=`, `strata=`, `repeats` above 1, the full-refit bootstrap, `CTMLE` or `DRTMLE`; and every refutation and sensitivity analysis on a learned-rule fit | [learned rules](point-treatment-tmle.md#learned-rules). No reviewed source gives the result. The fit refuses before any learner. `refute` reads `unavailable` because a refit relearns the rules, and each sensitivity row reads `unavailable` because no derivation for this target was reviewed. A learned-rule fit with missing outcomes refuses too, and [F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants) holds it. [F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) holds the rest |
| `intermediate=` and a multi-valued treatment with `incremental=` | [incremental interventions](../user-guide/estimands.md#incremental-propensity-score-interventions) |
| `incremental=` on `LTMLE` | [longitudinal TMLE](longitudinal-tmle.md#variations). Kennedy (2019) treats time-varying treatments. `LTMLE` raises `TypeError` by name at construction or at `fit`. [X19](../roadmap.md#x19-incremental-interventions-over-time) reopens it |
| a vector treatment node with a continuous component, and a vector point treatment | [modified treatment policies at a node](longitudinal-tmle.md#modified-treatment-policies-at-a-node). The ratio of a joint policy needs a joint density or the classifier route over the joint policy. `LongitudinalData.from_frame` raises `CapabilityError` before any learner. [X31](../roadmap.md#x31-continuous-vector-components-at-a-node) owns it |
| sample sensitivity-bound estimation for `LTMLE` | [longitudinal diagnostics](../user-guide/longitudinal.md#diagnostics). See [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) for the contracts Tan (2025) leaves open |
| blocked-temporal and rolling-origin splits | [two fold layers](../user-guide/methods-learners.md#two-fold-layers) |
| a confidence interval, a p-value, or a standard error on a `CTMLE` fit whose `strategy` is `"greedy"`, `"ordered"`, or `"discrete"` | [collaborative TMLE](collaborative-tmle.md). No result shows the reported curve is the estimator's influence curve when the selected working mechanism is not consistent for the treatment law. `ci`, `pvalue`, and `std_error` raise `CapabilityError`. The point estimate, the selection path, and the curve remain, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. A `discrete` fit with one declared candidate, equal to the full adjustment set, selects nothing and keeps the TMLE interval. [F18](../roadmap.md#f18-selector-path-c-tmle-inference) reopens it |
| a confidence interval, a p-value, or a standard error on every `CTMLE` fit with `strategy="oat"`, of either `oat_design=`, including a fit with `delta=` and a fit that requests one arm mean | [collaborative TMLE](collaborative-tmle.md#the-applied-revert). The fit takes the `"generated_design_plugin"` status. No result covers the shared design. The per-arm design applies Theorem 1 of Benkeser, Cai and van der Laan (2020) to each arm. That theorem's condition (v) fails for an estimated outcome regression, and the registered study `ctmle-oat-per-arm` measured standard errors 8 to 24 percent too small. The point estimate remains, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) reopens it |
| a confidence interval, a p-value, or a standard error on a `DRTMLE` fit with a non-empty `guard` and varying weights declared estimated (`weights_estimated=True`) | [DR-TMLE refusals](dr-tmle/supported-estimands.md#refused-by-name). The fit takes the `"estimated_weight_plugin"` status. The argument that an interval conditions on the weights concerns $D^*$, and not the reduced regressions. The point estimate remains, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. `guard=()` fits the ordinary TMLE and keeps its interval. Constant weights fit the unweighted estimator and keep it too. [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) reopens it |
| a confidence interval, a p-value, or a standard error on a `TMLE` or `DRTMLE` fit with `id=` and fewer than 10 positive-mass clusters in the fit or one reported baseline stratum, or an `LTMLE` fit with fewer than 20, in sample or cross-fitted | [clusters](inference.md#clusters). The fit takes the `"few_cluster_plugin"` status, in sample or cross-fitted. One such stratum withholds every interval. From that floor to 39 such clusters the fit reports intervals on a Student $t$ reference with $J - 2$ degrees of freedom, as Nugent et al. (2024), Section 2.2, recommend. Each floor is the smallest count that the registered few-cluster study measures for that fit. The point estimate remains, and `plugin_std_error` and `plugin_interval` retain the diagnostic on the normal reference. [F28](../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns the counts below each floor |
| `simultaneous_bands()` over an estimate with a Student $t$ reference, below 40 positive-mass clusters | [clusters](inference.md#few-clusters). No source read here gives a $t$-calibrated joint band. `simultaneous_bands()` raises `CapabilityError`, and a fit skips its default band and names the reason in `summary()`. The pointwise intervals remain. `tests/unit/test_few_cluster_reference.py` checks both |
| a contrast, a simultaneous band, any E-value branch, `variable_importance()`, or `tipping_gamma(use_ci=True)` on the fits of the four status rows above | [inference status](inference.md#inference-status). Each one is a confidence statement, or reads an interval that these fits do not supply. `variable_importance()` refuses at its entry point, because it adjusts one p-value per candidate. The E-value capability reports `unavailable` rather than raising inside the computation. `truncation_curve()`, `missingness_tilt()`, and the default `tipping_gamma()` still run. On a point-treatment fit the two frames rename their three spread columns. The longitudinal `truncation_curve()` publishes no spread. `LongitudinalResult.curve()` and `incidence_total()` rename their spread columns |
| `stratify_by="treatment"` and `stratify_by="treatment+outcome"` on any fit that draws a split, including selector-based C-TMLE at every setting | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). No shipped result covers a partition read off the data the fit then conditions on |
| a cross-fitted continuous outcome with `q_bounds=None`, for `TMLE`, `DRTMLE`, `CTMLE`, and `LTMLE` above one fold | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). The sample outcome range would take the scale from held-out rows |
| `CTMLE` with `id=`, at every `cross_fit` setting | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). No clustered result covers the outcome-adaptive mechanism. Selector-based fits also lack a result for grouped selection and nested folds or candidate-selection variance |
| `cv_evaluation=True` or `targeting_scheme="fold"` with `id=`, when a validation fold holds 1 cluster | [the algorithm as implemented](cv-tmle.md#the-algorithm-as-implemented). Each fold's variance takes the variance of the cluster totals inside the fold, and 1 total has none. `TMLE` raises `CapabilityError` before any learner, for a generated or a supplied split. A generated split passes at no more folds than half the cluster count. `tests/unit/test_fold_evaluated_cluster_variance.py` checks it |
| the `subset` and `bootstrap_measurement_error` refutations on a fit that declared `split_plan=` | [reusable outer split plans](cv-tmle.md#reusable-outer-split-plans). Each one refits on rows the fit never ran, and no rule here says which fold labels those rows inherit. `refute()` raises `CapabilityError` before it refits anything. The `refute` capability row reads `deferred` on `tests`, with the same sentence. A request whose `tests=` names neither operation runs |
| replicate weights (BRR, jackknife) | [observation weights](../user-guide/data-design.md#observation-weights-are-not-estimand-weights). These are a set of designs rather than one weight vector, so the shape they want is a refit per replicate outside the estimator |
| omitted-variable sensitivity after `repeats=` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The median bound needs an influence function; a coordinatewise median of per-draw influence terms is not one. The five omitted-variable capability rows declare this refusal as `unavailable` before any call, and `test_the_five_declared_rows_carry_that_same_reason` checks them |
| omitted-variable sensitivity on a `DRTMLE` or `CTMLE` fit | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). Neither estimator assumes a consistent treatment mechanism, and no derivation gives $\nu^2$ or the bound's standard error without that assumption |
| omitted-variable sensitivity on a fit with a response mechanism | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The bound is well posed, and the implementation is missing. The implemented representer omits the response indicator, and $\sigma^2$ averages the respondents only. Use the missingness tilt for response on the arm means. The tilt does not cover `ey_obs`, PAR or PAF. An explicit tilt request that names one of them raises a sentence that names them. On a fit that reports one of them, the `tipping_gamma` row reads `deferred` and requires `estimand`, and the bare call raises the row's reason. The default tilt sweep skips them |
| omitted-variable sensitivity on a fit with a declared missing treatment | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The Riesz representer of the bound is derived for a recorded treatment. This fit divides by the composite mechanism, whose representer and confounding strengths are not derived |
| omitted-variable sensitivity on a fit with an intermediate variable | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The bound is well posed, and the implementation is missing. The representer carries the intermediate weight, so the treatment strength $c_D$ would also measure the intermediate mechanism |
| omitted-variable sensitivity on a `regime`, `shift`, or `msm` parameter axis | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). Each parameter has a Riesz representer, so the bound is well posed. Only the implementation is missing |
| omitted-variable sensitivity on an `ipsi` parameter axis | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). An incremental intervention tilts the mechanism, so the mechanism is part of the estimand rather than a nuisance |
| omitted-variable sensitivity on an arm-indexed fit that reports no counterfactual mean and no linear contrast. A ratio-only fit, a PAR or PAF fit, and a complete-outcome `ey_obs` fit are such fits | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). One bound is the second moment of one contrast's Riesz representer, and the fit reports no such contrast. The five capability rows read `unavailable` with the sentence that each call raises. On a fit that reports `rr` or `or`, the sentence names `sensitivity.evalue()`. `TestTheBoundRowsReadTheCallsTable` in `tests/unit/test_capability_row_predicates.py` checks four such fits |
| `benchmark()` with every covariate of the fit named | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The short model would adjust for nothing, and no estimator here fits without a covariate. `benchmark()` raises `CapabilityError` before the refit. On a fit with one covariate, the `benchmark` capability row reads `unavailable`, because no value of `covariates` runs. On a wider fit, a request that names every covariate reads `unavailable`. An unknown name raises `DataError` before this refusal |
| omitted-variable sensitivity when the doubly robust estimate of $\nu^2$ is not positive | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). This row reads the data, not the fit class. The other omitted-variable rows refuse a whole class of fit before any computation, and their capability rows report `unavailable`. This one leaves the capability row `available`, and the call raises `CapabilityError` only when the fitted representer gives a nonpositive second moment. An ordinary TMLE script that received a bound before can therefore raise now, because the package substitutes no plug-in value for the refused one |
| the one-sided limits and the confidence-limit robustness value under `nu2_estimator="plugin"` | [validation and sensitivity methods](validation-methods.md#standard-error-of-the-omitted-variable-bound). No derivation in a source this package cites gives the standard error of a bound built on the plug-in $\nu^2$, which moves at first order with the fitted treatment mechanism. The bounds, `rv`, `max_bias`, `benchmark()` and `contour()` remain. [F26](../roadmap.md#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) reopens it |
| omitted-variable limits when $\sigma^2=0$ | [the bound's standard error](validation-methods.md#standard-error-of-the-omitted-variable-bound) needs a positive maximal-bias scale. The point bounds remain, but the limit accessors refuse. An unreachable point robustness threshold is `None` |
| any E-value request on a fit with an intermediate variable and a discrete treatment | [the E-value paths](validation-methods.md#e-value). This package has no implemented bound for the fitted controlled direct effect and its confounding model. The ordinary conversion does not establish sensitivity for that target. The capability row reports `unavailable` before any computation. A continuous-treatment fit with an intermediate variable reports `not_applicable`, because that check runs first. [F25](../roadmap.md#f25-e-value-for-a-controlled-direct-effect) tracks the support gap |
| the standardized Gaussian E-value on a fit with a response mechanism | [the E-value paths](validation-methods.md#e-value). The conversion needs the population outcome standard deviation, and the respondents' one estimates a different quantity. A ratio E-value on the same fit stays available |
| correction diagnostics after ordinary TMLE or collaborative TMLE | those methods do not use the DR-TMLE correction system. A DR-TMLE fit with an empty guard also subtracts no correction term |
| a binomial ATT or ATC E-value conversion | the analysis needs a conditional baseline risk and a supported conditional risk-ratio target. The library implements neither |
| a fixed-baseline E-value on a reference-arm mean that its own standard error does not separate from zero | [the E-value paths](validation-methods.md#e-value). The conversion divides by that mean, so the ratio has no stable denominator |
| a fixed-baseline E-value whose risk difference is at or below the negative of the reference-arm mean | [the E-value paths](validation-methods.md#e-value). The difference implies a nonpositive risk in the contrast arm, and no risk ratio describes one |
| a cached-nuisance risk ratio after CV evaluation | the estimator refuses nonlinear ratio targets on this evaluation path. Post-fit retargeting keeps that boundary |
| a controlled direct risk ratio from cached nuisances | the fitted result can report a controlled direct risk ratio. The post-fit retarget path does not pass the fitted intermediate intervention level, so it cannot derive one from cached nuisances |

Which multi-arm surfaces are covered, and which are not, is tabulated in one place:
[where a multi-valued treatment is supported](#where-a-multi-valued-treatment-is-supported).

Several former gaps have landed. `cleverly` now supports multi-valued longitudinal treatment
nodes. It supports multi-valued selector-based C-TMLE, outcome-adaptive C-TMLE, DR-TMLE, `ATT`,
and `ATC`. `LTMLE` supports observation weights and a working model over regimens. Shift fits
support `intermediate=` and weights. They support `delta=` in sample. `cleverly` also supports multi-arm
omitted-variable and MNAR sensitivity analyses. TMLE supports the scalar missing-outcome
natural-course mean, and PAR and PAF, with ordinary fitting or one generated split of stacked
CV-TMLE. [Missing-outcome natural-course contracts](#missing-outcome-natural-course-contracts) gives
the boundaries of each path.

TMLE also cross-fits arm-indexed means and contrasts with missing outcomes under one stacked CV-TMLE
contract.
[Missing-outcome arm-indexed contract](#missing-outcome-arm-indexed-contract) gives its
boundaries.

The remaining shift gap is narrower than it was. The tilt itself is written. The missing derivation
must establish whether the tilted parameter is still the shift parameter.

### Missing-outcome natural-course contracts

`NaturalCourseMean()`, `PopulationAttributableRisk()` and `PopulationAttributableFraction()` with
`missingness=` and at least one missing outcome have two TMLE contracts. A request that reads the
natural course beside arm targets meets the same contracts.
`CrossFitting(enabled=...)` selects the contract. The table gives each requirement and the
compositions that each contract refuses.

| requirement | ordinary TMLE, `enabled=False` | stacked CV-TMLE, `enabled=True` |
| --- | --- | --- |
| target | `ey_obs`, `par`, `paf`, or `ey_obs` beside arm means, contrasts, ATT and ATC. `estimands="all"` keeps its arm-only list | the same, without ATT and ATC, which the arm-indexed contract refuses |
| estimator | `TMLE`. `CTMLE` and `DRTMLE` are refused. `DRTMLE(guard=())` is the ordinary TMLE, and the message names `TMLE` | the same |
| treatment | two or more arms | two or more arms. With three or more, the scalar `ey_obs` also needs two respondents in each arm |
| outcome | binary, or continuous with declared `q_bounds`. A continuous outcome without `q_bounds` is refused | binary. A continuous outcome is refused |
| outer folds | one fold | package-generated folds with `stratify_by="none"` and `n_folds` of at least 2. One fold, stratified folds, and `split_plan=` are refused |
| repeats | `repeats=1`. A larger value fails when `CrossFitting` is constructed, because no split exists to repeat | `repeats=1`. A larger value is refused when the fit starts |
| targeting | `Targeting(fluctuation="logistic", algorithm="iterative", target_weights=False)`. A linear fluctuation, one-step targeting, and weighted targeting are refused | the same, with `targeting_scheme="pooled"` and `fold_evaluation=False`. Fold targeting and fold evaluation are refused |
| inference | influence-curve Wald interval. `n_bootstrap > 0` is refused ([F2](../roadmap.md#f2-targeted-bootstrap-inference)) | the same |
| rows | fixed `weights=`, `id=` and baseline strata are admitted. `intermediate=` is refused | unweighted iid rows. Weights, clusters, and `intermediate=` are refused. The scalar `ey_obs` admits baseline strata. PAR, PAF, and a joint fit meet the arm-indexed contract, which refuses them ([F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants)) |
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
The design declares no intermediate. The request is not the scalar `NaturalCourseMean`.

A request that reports `ey_obs`, PAR or PAF beside arm targets meets this contract and the
natural-course contract, and its admitted list includes `ey_obs`, `par` and `paf`.
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

Three checks enforce the contract. The table gives the error of each check and the time at
which it runs.

| check | error | when it runs |
| --- | --- | --- |
| a composition outside the contract | `CapabilityError` | when the fit starts, before fold generation and before any learner is fitted |
| `DRTMLE` with `delta=` and `cross_fit=True` | `CapabilityError` | when the fit starts, before any learner is fitted |
| minimum content | `DataError` | after fold generation, before any learner is fitted |

A `DRTMLE` fit with missing outcomes and `cross_fit=True` raises the second `CapabilityError` in
the table at every `guard`, including `guard=()`, and under every fold policy.
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
refuses both, so neither step can run. A copied or modified estimator can still carry the refused
policy. The fit then raises `CapabilityError`, a subclass of `ValueError`, with the declaration
check's own reason. A result that holds such an estimator reads its `refit_nuisances` replay slot
false, with the code `point_replay_refit_configuration`.

A refusal that has an in-sample alternative names it as `CrossFitting(enabled=False)`. The engine
form is `cross_fit=False`. An in-sample fit that draws no split keeps whatever fold policy the
declaration carries and applies none of it. Selector-based C-TMLE draws selection folds at every
setting, so it refuses a stratified policy in sample as well.

### Replay-only unavailability

The longitudinal truncation grid can be scientifically supported while one result cannot replay
it. `result.replayability.unreconstructible` carries the codes below as data. The capability
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
| `longitudinal_replay_learner_unclonable` | an outcome or pseudo-outcome learner cannot be cloned |
| `longitudinal_replay_random_state_unseeded` | a replayed learner or nested learner declares an unspecified random state |
| `longitudinal_replay_random_state_non_integer` | a replayed learner or nested learner declares a non-integer random state |
| `longitudinal_replay_fitted_bound_mismatch` | invocation-time preflight found that a full replay at the fitted bound differs from the stored fitted artifacts |

The static recipe omissions make the capability row `unavailable`. A fitted-bound mismatch instead
raises a stable `CapabilityError` from that invocation; it is not known when the capability row is
built. Missing `bounds` or the refit opt-in makes a combined report `deferred`, because the caller
can supply either request.

A point-treatment result has two codes of its own. `replayability()` reads each one without a
fit, and every row that needs the false slot reads `unavailable`.

| omission code | replay boundary | slots |
| --- | --- | --- |
| `point_replay_function_declaration` | a stored regime or MSM function lacks an accepted known-function declaration | both slots read false |
| `point_replay_refit_configuration` | the package refuses the stored estimator configuration before a refit. The slot runs every check that `refit()` runs before a learner, including the design checks of an estimator subclass. A stratified fold policy, a cross-fitted continuous outcome with `q_bounds=None`, and a C-TMLE fit on clustered data are such configurations | `refit_nuisances` reads false. `retarget_cached_nuisances` stays true, because the cached nuisances still retarget |

The second code reads the checks that `refit()` runs before its first learner. These are the fold
policy and four data contracts. `TestTheRefitSlotReadsTheRefitPreflight` in
`tests/unit/test_capability_row_predicates.py` checks each slot against the call it describes.

### A different question

These are well-posed parameters, but the selected estimator does not target them. No setting turns
one into the other.

| refused | the question it would answer instead |
| --- | --- |
| `eliminate=` on a competing-risks fit | the incidence of a cause if the competing events were *removed*. That intervenes on them rather than conditioning on the history. It needs a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions for the competing event. What is reported instead is the incidence with the competing causes left alone |
| `CTMLE` on data that declares its treatment mechanism | C-TMLE selects a treatment mechanism, and the declaration leaves nothing to select. Fit `TMLE` or `DRTMLE` on the declared data, which is the question a known mechanism answers |
| a declared known treatment mechanism beside `treatment_delta=` | a missing treatment needs $P(A=a\mid\Delta_A=1,W)$, the mechanism among the recorded rows. It equals the design mechanism only when recording is independent of treatment given $W$, which the fit cannot check |
| `intermediate=` on `LTMLE` | a controlled direct effect fixes a mediator at one time point. Over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| `ey1` and `ey_regime` from one fit; `msm=` with `interventions=` or `policies=`; an item of another intervention kind in a typed estimand, such as an `Incremental` in `RegimeContrast` | each keyword declares what "counterfactual" means for the fit, or how the counterfactuals are summarised. One fluctuation solves one set of score equations, so a fit reporting parameters from two axes would put two of them under one heading. `CausalStudy.identify` refuses the typed request with `CapabilityError`, and it names the typed estimand for the item |
| the per-arm propensity table on a continuous fit | a per-arm table has no rows when there are no arms. `diagnostics.support()` is not itself refused. On a fit that declared `policies=` it dispatches to the question that does apply, which is whether the density *ratio* stays bounded |
| `res.sensitivity`, `res.diagnostics` and `res.validate()` on an `LTMLE` result | each is part of the shared result contract. Stagewise support, scores, nuisance loss, and the descriptive full-recursion truncation grid are supported. Sensitivity operations without a longitudinal derivation report `unavailable`. `res.save()` is supported, and [persistence and replayability](../user-guide/results-assessment.md#persistence-and-replayability) states its contract |
| `ratio(..., view="survival")` on an end-of-study fit, or on a fit with two or more causes; `rmst` and `rmtl` on an end-of-study fit or a working-model fit; `ratio` on a working-model fit | an end-of-study fit reports means and no survival curve. One minus one cause's incidence is not all-cause survival. A coefficient is not a regimen mean. [Functionals of a fitted result](longitudinal-tmle.md#functionals-of-a-fitted-result) gives each message |
| a plan that changes the treatment after baseline on a held design (`treatment="A"` or `TimeToEvent`) | a time-varying treatment plan. Declare one treatment column per node on the wide layout. The held design raises `ValueError` before any learner, and a plan that repeats one label is read as that label |
| `logistic_plugin` on a fit that is not a selector `CTMLE` fit, or on an `oat`, multi-arm, cross-fitted, weighted, missing-outcome or `intermediate=` fit | R `ctmle` defines the logistic-estimation term for a binary treatment whose propensity is fitted once on all rows. [Collaborative TMLE](collaborative-tmle.md) gives the term |
| a non-empty `assess(arguments=...)` block for `score_equations` | the validation battery owns that name and runs it argument-free. The battery presents one row per name, and it presents the validation row. A caller's tolerance would be computed and then hidden, so a check that failed at that tolerance would never reach `attention`. Call `res.diagnostics.run_all(arguments=...)`, or the operation itself. The battery also owns `support` and `nuisance_models`, which accept no argument, so an argument for either is a `TypeError` from the signature |

### Wrong by construction

These are worth reading even if you never reach for the keyword. Most are mistakes that are easy to
make by hand, in any framework, and none announces itself. They can produce a number for the wrong
target or an interval whose stated conditions are not met.

| refused | what goes wrong if it is done anyway |
| --- | --- |
| `ratio(..., kind="or")` when the outcome family is not binary | the odds of a mean outside $(0, 1)$ is not defined. The refusal applies the rule of the registered `or` target, `TARGETS["or"].requires_family`, and names `kind="rr"` |
| a population-mechanism odds tilt supplied as a `Stochastic` density | the population-law target depends on `P` through `g*`. Its curve needs a derivative that the regime curve lacks. On one exact law, the reported standard error is 0.62 of the exact one at `delta = 2` (`tests/unit/test_stochastic_regime_densities.py`). A realized learned density instead defines a data-adaptive target; its inference needs conditions this API does not check. `Stochastic` requires `density_kind="known"` and refuses `None` or `"estimated"` at construction and fit. A false `"known"` declaration still fits because code cannot inspect a closure. [RM25](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm25-declared-stochastic-regime-densities) records the witness and both targets |
| a population-law incremental intervention built by hand as a `Stochastic` regime | the missing term has variance `Var(delta * (Qbar(1,W) - Qbar(0,W)) / D^2 * (A - g))`. For that target, omitting the term understates variance wherever this variance is positive. `Stochastic` refuses the tilt when `density_kind` is `None` or `"estimated"`. `TMLE(incremental=)` fits the population odds tilt and includes the term |
| a user-written `Intervention` class whose density depends on the population mechanism | it can omit the same derivative while reporting an interval for the population-law target. On the RM25 witness law, a class that computes the sample-mechanism odds tilt reports 0.62 of that target's exact standard error (`tests/unit/test_rule_and_intervention_declarations.py`). A realized learned density has a different inference contract. The class must carry `density_kind = "known"`. `TMLE` refuses `None`, a missing attribute, and `"estimated"` before any learner. `RegimeSet.evaluate` refuses them before `density` runs. A false `"known"` declaration still fits because code cannot inspect what `density` computes. [RM28](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm28-declared-densities-of-user-written-interventions) records the witness |
| a `Rule` or `DynamicRegimen` learned from the analysis sample | the fit treats its rule as fixed. A learned policy can instead target a realized policy value or a population-indexed policy. Those targets need distinct inference conditions; an optimal rule can also be nonregular at ties. For a threshold at the sample mean of a continuous covariate, the curve omits a derivative through that mean. On one exact law, the reported standard error is 0.73 of the exact one at one node (`tests/unit/test_rule_and_intervention_declarations.py`). It is 0.77 for a two-node regimen (`tests/unit/test_regimen_rule_declarations.py`). `Rule` and `DynamicRegimen` require `rule_kind="known"` and refuse `None` or `"estimated"` at construction and fit. A plan of labels alone needs no declaration. `LTMLE` refuses a callable written inline in `regimens=`, because it carries no declaration. A false `"known"` declaration still fits because code cannot inspect a closure. For a point treatment, `LearnedRuleValue` estimates the average value of rules learned inside each training fold ([learned rules](point-treatment-tmle.md#learned-rules)). A learned longitudinal regimen has no supported path ([X11](../roadmap.md#x11-learned-policy-follow-ups)). [RM28](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm28-declared-densities-of-user-written-interventions) records the witnesses |
| a shift's inference taken from the regime inducing the same density | the means and the clever covariates agree entry for entry. The curves do not. The gap is `Var(Qbar(d(A,W),W) - E[Qbar(d(A,W),W) | W])`. Too small, always |
| a shift fit run on the complete cases when outcomes are missing | it is an ordinary shift fit on a *different* joint law of `(A, W)`, so it converges to a different number, and nothing in its own output says so. Measured on the dose fixture at 0.17, four standard errors, with a mechanism whose slopes are mild. `delta=` is what corrects it |
| a missingness or intermediate mechanism read at the observed dose rather than the assigned one | the fluctuation updates `Qbar` as a function of the dose, so `Qbar*(d(A,W),W)` is the update evaluated where the policy sends the unit. Silent wherever the mechanism does not depend on the dose, and invisible to a Gateaux check on an exact law |
| a "stabilised" MSM weighting `h` by the estimated mechanism | the same argument once more. `h` becomes a functional of `P`, and a term goes missing from the influence curve. `MSM` refuses `weights_kind="estimated"` and a `weights=` callable with no declaration. [MSM projections](msm-projections.md#variations) gives the declaration |
| an MSM `design` that reads a sample statistic, such as a covariate centred at its sample mean | for the population-law target centred at $E_P[W]$, the projection depends on $P$ through the design. Its reported curve omits that derivative. On one exact law, the reported intercept standard error is 0.893 of the exact one (`tests/unit/test_msm_design_declaration.py`). An interval conditional on the centre learned from the same rows also needs separate validation. `MSM` refuses `design_kind="estimated"` and an undeclared callable. A design built by `MSM.linear` needs no declaration. Code cannot detect a false `"known"` declaration. [MSM projections](msm-projections.md#variations) gives the declaration, and [RM27](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm27-declared-msm-design-functions) records the witness |
| `g_bounds=` or `truncation_curve()` on an `incremental=` fit | `g` is *inside* the estimand, so truncating it moves the parameter rather than regularising a denominator. The result is a number for a parameter nobody declared. `truncation_curve()` refuses the treatment axis with `CapabilityError`, and the omitted `mechanism` selects that axis. The module call and the facade call raise the same sentence. The capability row reads `unavailable`. On a fit with missing outcomes it reads `deferred` on `mechanism`, because `mechanism=True` runs |
| a `cap=` fitted from the data on a shift | the estimand becomes data-dependent. The interval conditions on an estimated boundary, and every bootstrap replicate targets a slightly different policy |
| `CTMLE` on an `incremental=` fit | each candidate `ghat` defines a different estimand, so the cross-validated search selects between *estimands* rather than between estimators |
| splitting a cluster across folds to buy more of them | the out-of-fold predictions stop being independent of the rows they are used on, and the standard error shrinks in exactly the direction `id=` was passed to prevent |
| a supplied `SplitPlan` whose labels no recorded draw produces | the labels could have been chosen by reading the outcome, the treatment, or a covariate, which is the leak cross-fitting exists to prevent. The fit draws each repeat again from the plan's record and compares it label for label. [Reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) states what a plan records |
| reusing a supplied `SplitPlan` on rows it was not realized on | a fold label is a row position. A reordering, a replaced column, or an added covariate leaves every label pointing at a different unit, while the row count still agrees. The out-of-fold guarantee is gone, and nothing in the output says so. [Reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) states how a plan binds to its rows |
| joint covariance, post-fit contrasts, or simultaneous bands after `repeats=` | coordinatewise medians do not preserve linear identities among estimates; a central-draw curve also does not represent the split-adjusted median estimator needed for a multiplier band |
| a cross-validated variance for the curve a repeated fit retains | at equal fold sizes it collapses to the pooled uncentred second moment for *every* partition, so the partition carries no information. That rule was rejected, and the split-adjusted median variance replaced it |
| a one-shot non-identity-link MSM | the derivative of the inverse link depends on the coefficient, so a single pass reports a standard error for an equation it did not solve. The link is supported. What is refused is skipping the alternation it needs |
| a censoring time between two grid times, read as censored at a grid time | the censoring cannot be ordered against the events of its interval, so either reading biases the risk. On one interval with exponential times the two readings give 0.280 and 0.236 against a truth of 0.259 (`test_both_off_grid_censoring_conventions_are_biased`). `LongitudinalData.from_time_to_event` raises `DataError` before any learner and names the visit grid as the remedy |
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

The `simulated_confounding` capability row resolves each request by the predicate that the call
raises from. A request for a refused parameter reads `unavailable` with the call's sentence. The
natural-course mean and a zero-delta policy mean are such parameters. A categorical or a constant
benchmark covariate reads `unavailable` in the same way. The bare row of a natural-course fit reads
`unavailable`, because no reported parameter runs.

## Where a multi-valued treatment is supported

A multinomial treatment mechanism is the default construction here rather than a variant of a
binary one. `CausalData` carries `treatment_levels`, `Propensity` holds an `(n, K)` simplex, and
`learners._fitting.predict_probabilities` produces it. Two arms are the `K = 2` case of that
construction and not a separate path, which is what the
[bit-for-bit invariant](../architecture-invariants.md#dataframes-and-labels) is about.

So "does this estimator take more than two arms" usually has the answer "yes, through the same code
as two". The informative entries are those that do not. The `status` column uses the vocabulary
of [How to read a refusal](#how-to-read-a-refusal) above, plus `waiting on published theory` from
the [roadmap's eligibility rules](../roadmap.md#eligibility).

| surface | status | why |
| --- | --- | --- |
| `TMLE`: `ey` per arm, `ate` / `att` / `atc` per non-reference arm, regimes, MSMs over arms | supported | one counterfactual mean per arm and one contrast per non-reference arm. The [oracle-law gate](validation-methods.md#the-oracle-law-gate) states that a target meant for more than two arms needs a branch on the three-armed law |
| `DRTMLE`: univariate and bivariate reductions | supported | [armwise one-vs-rest](dr-tmle/index.md#variations), with each reduction and correction indexed by a free level. The cited theorem is binary, so this is an implementation-backed armwise extension rather than a claim about that theorem's literal scope |
| `DRTMLE` with `delta=` in a randomized trial | supported | Díaz and van der Laan's missing-outcome construction, applied to each arm indicator and stacked. Each arm's treatment, observation and outcome tilts are its own, so no fluctuation parameter is shared across arms. See [the multi-arm contract](dr-tmle/theorem.md#more-than-two-arms) |
| `TMLE` and `DRTMLE` with an observational missing outcome or a missing treatment | supported | the composite indicator, applied to each arm and stacked. Each arm's column of the composite mechanism is tilted alone, two arms included. See [the composite contract](dr-tmle/theorem.md#observational-missing-data-the-composite-indicator) |
| `CTMLE`: selectors and `strategy="oat"` | supported | the selectors fit one shared `n x K` categorical mechanism, selected against one nonredundant vector. See the [standing decision](../architecture-invariants.md#targets-interventions-and-variants). `strategy="oat"` fits one binary mechanism per arm by default, and the `"shared"` design fits one categorical mechanism. See [the per-arm design](collaborative-tmle.md#the-per-arm-outcome-adaptive-design) |
| `LTMLE`: categorical nodes, static and dynamic regimens | supported | [treatment over time](longitudinal-tmle.md#the-algorithm-as-implemented). Each node owns its level set, and the clever covariate selects the assigned label's probability |
| positivity, omitted-variable, E-value and MNAR sensitivity | supported | each is one parameter per contrast, and each reads its arms from the parameter's structured index rather than assuming two |
| simulated common-cause sensitivity | supported for binary and continuous treatment; named theory stops cover the remaining fit-wide gaps | The binary surface accepts marginal and baseline-stratum arm means, ATE, and ratios under ordinary TMLE and C-TMLE. Ordinary TMLE also accepts ATT and ATC with perturbed group membership. Ordinary TMLE alone accepts marginal and baseline-stratum PAR, PAF, fixed-regime parameters, and identity-link MSM coefficients. It also accepts marginal incremental targets and nonlinear MSM coefficients. Complete-outcome DR-TMLE supports marginal arm means, ATE, and ratios only. The continuous surface accepts marginal and baseline-stratum modified-policy means and contrasts under ordinary TMLE. A risk-ratio tilt fit is refused, because its policy reads the treatment mechanism. Continuous MSMs require marginal fits. Every listed row accepts fixed probability weights. Longitudinal, multi-arm, missing-outcome, intermediate, estimated-weight, and clustered fits report unavailable before work begins. Each of those six waits on published theory, and so does logical categorical calibration. `NaturalCourseMean`, the zero-delta policy mean, and the multiplier-one incremental mean remain refused. See the [population contract and refusal table](validation-methods.md#simulated-common-cause-stress-surface) |
| `ey1` / `ey0` and the incremental estimands, on a multi-arm fit | wrong by construction | they *name* one of exactly two arms, so on five arms they would report a contrast of arms `0` and `1` under the name of a parameter about all of them. Declared by `requires_binary_treatment`. The multi-arm path reports per-arm `ey` instead |
| `incremental=` itself, above two arms | a different question | an odds multiplier names two arms. One odds per contrast is well posed, and it is a *different intervention* with a different influence function rather than a generalisation of this one |
| `LTMLE`: known stochastic categorical policies at a node | supported | [known stochastic policies](longitudinal-tmle.md#known-stochastic-policies). A node holds a policy density fixed before the fit. The recursion carries the policy mean of the per-level predictions, and the clever covariate is the ratio of the policy to the bounded mechanism. Exact-law tests and the [registered study](method-evidence/stochastic-categorical-longitudinal-tmle.md) cover it |
| `LTMLE`: modified treatment policies at a continuous, categorical or vector node | supported | [modified treatment policies at a node](longitudinal-tmle.md#modified-treatment-policies-at-a-node). A plan node holds a `Shift`, `Scale`, `Piecewise`, `ModifiedPolicy` or `RiskRatioTilt`. A continuous node reads a pooled-hazard density or the classifier ratio. Exact-law tests cover it, and the registered study `longitudinal-mtp` validates it. A vector node with a continuous component is refused |

A source could close one entry as it stands. It is the multi-arm part of the simulated
common-cause row. A source would not close the `a different question` row. A different estimand
would answer it, with its own derivation, oracle law, and evidence.

The `DRTMLE` row needs no new source either. It needs the implementation that its roadmap row
holds.
