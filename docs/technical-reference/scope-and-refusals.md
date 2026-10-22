# Scope and refusals

`cleverly` refuses a composition it has not derived, rather than returning a convenient
approximation to a different estimand. This page says how to read such a refusal. It also tabulates
the one axis where the answer is least obvious: which surfaces take more than two treatment arms.

## How to read a refusal

A refusal is always *by name*. The keyword is accepted and rejected with a stated reason. It does
not arrive as an `unexpected keyword argument` that names none.

Every `CleverlyError` carries the record of its refusal in `error.refusal`. The record is a frozen
`Refusal` from `cleverly.exceptions`. `str(error)` prints `error.refusal.message()`.

| field | holds |
| --- | --- |
| `reason` | the `RefusalReason` code. The next table lists the five codes |
| `condition` | the sentences that state the failed condition |
| `alternative` | the sentence that names a working call, or `NO_ALTERNATIVE` when no call in the package answers the request. Only `invalid_declaration` may leave it empty |
| `roadmap` | the id of the [roadmap](../roadmap.md) item that tracks a missing capability, or `None` |

The `reason` of the record tells you where the problem is. That decides what to do about it.
A `RefusalReason` is a string enumeration, so `error.refusal.reason == "inference_gap"` is true
for an inference gap. The error class does not fix the code. Read `error.refusal.reason` and not
the class.

| code | where the problem is | what the message holds | what to do about it |
| --- | --- | --- | --- |
| `invalid_declaration` | in the declared sample or design | the failed condition | Correct the declaration, the data, or the design. The request is not well posed as declared |
| `different_estimand` | in the question | both questions: the one you asked and the one the option answers | Decide which question you mean. No flag produces the other, and one that quietly did would answer something nobody asked |
| `implementation_gap` | in this package | the alternatives and the roadmap item that tracks the gap | The parameter is well defined and nobody has written it here. Use the alternative, ask for it, compute it elsewhere, or contribute it. Proposed work is on the [roadmap](../roadmap.md) |
| `inference_gap` | in the uncertainty | the supported inference | The estimate exists, and the requested interval, p-value, or band is not derived. Report the supported inference |
| `api_redirection` | in the call | a working call | The feature exists through another argument or object. Make the call that the message names. This code does not mark a missing capability |

A capability row carries the same record before any call. A row of
`result.diagnostics.capabilities`, an `AssessmentItem`, and a `MethodAvailability` row of
`CausalStudy.available_methods()` each hold it in `refusal`. For these rows `refusal` is `None` when no refusal
applies. The call, when you make it, raises an error with the same code and text. The row and the raise
therefore state one reason in one text.

A built-in `ValueError` or `TypeError` means a malformed argument, such as a wrong type or a value
out of range. It is an invalid declaration, and it carries no record. A misspelled keyword raises
`TypeError` for the same reason.

Some refused requests would run and return a plausible number whose target or interval is wrong.
The [Wrong by construction](#wrong-by-construction) section lists them. Read them as warnings about
the analysis, and not about this package's coverage.

A fit whose *data* cannot support what you declared is an `invalid_declaration` that the package
refuses where the problem arises. Examples are a regimen nobody followed and two absorbing causes
firing at one node. A known policy that can draw a level no fitted row of its node received is
another, and [known stochastic policies](longitudinal-tmle.md#known-stochastic-policies) states
why. Those are statements about the sample.

A horizon at which no follower of a regimen had the event is not refused. Its regression is
zero, the maximum-likelihood hazard, and
[a node with no event](longitudinal-tmle.md#a-node-with-no-event) gives the rule.

Cross-fitting narrows what the sample supports. Each outer fold fits its regressions and its
mechanism on its training rows alone, so every fold must carry the followers and treatment levels
that each node needs. The split reads no treatment and no outcome, so it cannot protect a rare
level. The package checks the realized draw instead, before the first learner, and the refusal
names no redraw. The [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give
every such message.

The sections below list refusals by topic. Each table has a `code` column. The column holds the
code that the raise site assigns, so a section can hold rows of several codes.

| section | what it lists |
| --- | --- |
| [Not written yet](#not-written-yet) | refusals of the request, mostly `implementation_gap` and `inference_gap` |
| [Targeting and fitting option refusals](#targeting-and-fitting-option-refusals) | the refusals of `stopping=`, `step_bound=`, `iterate=`, `outcome_by_arm=`, `patience=` and `treatment_adjustment=` |
| [Experimental compositions](#experimental-compositions) | requests that run and carry the experimental label, and the ones that stay refused |
| [Replay-only unavailability](#replay-only-unavailability) | the omission codes of a result that cannot replay, all `invalid_declaration` |
| [A different question](#a-different-question) | requests that ask for another estimand, mostly `different_estimand` |
| [Wrong by construction](#wrong-by-construction) | mistakes that would run and return a wrong number |

### Not written yet

Nothing is wrong with wanting any of these. Most rows are gaps in coverage, and the message says
so rather than implying the request was ill-posed. A row marked `implementation_gap` names an
alternative and the roadmap item that tracks the gap. `tests/unit/test_refusal_codes.py` checks
that each such refusal names an item. A row marked `inference_gap` names an alternative, and the
roadmap item when one tracks the gap. A row marked `invalid_declaration` or
`api_redirection` names the declaration to correct or the call to make.

| refused | code | where |
| --- | --- | --- |
| missing-outcome `NaturalCourseMean`, PAR or PAF outside their two TMLE contracts | `implementation_gap`, `inference_gap`, `invalid_declaration`. The [subsection](#missing-outcome-natural-course-contracts) gives the code of each check | [missing-outcome natural-course contracts](#missing-outcome-natural-course-contracts) lists every refusal. [Observed-data extensions](point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) defines both estimators, and [PAR and PAF with missing outcomes](point-treatment-tmle.md#par-and-paf-with-missing-outcomes) defines the stack |
| cross-fitted arm-indexed means and contrasts with missing outcomes outside the stacked CV-TMLE contract | `implementation_gap`, `inference_gap`, `invalid_declaration`. The [subsection](#missing-outcome-arm-indexed-contract) gives the code of each check | [missing-outcome arm-indexed contract](#missing-outcome-arm-indexed-contract) lists every refusal. [Stacked CV-TMLE for arm-indexed targets](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets) defines the estimator |
| a cross-fitted shift, incremental, regime, MSM, or controlled-direct-effect fit with missing outcomes (`delta=`) | `implementation_gap` | [the refusals a caller can meet](cv-tmle.md#the-refusals-a-caller-can-meet). The [hard-stop review](hard-stop-review.md) reads Díaz et al. (2023), Theorem 3, for the shift, regime, MSM and controlled-direct-effect targets. The incremental target waits on a source read. The package implements neither. The fit raises `CapabilityError` before any learner is fitted. The in-sample fit with `delta=` remains available, except for a continuous-dose MSM, which the next row refuses. [X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) part (d) reopens it |
| a continuous-dose MSM with missing outcomes (`delta=`, or `PointTreatment(missingness=...)`) or with `intermediate=`, in sample or cross-fitted | `implementation_gap` | [MSM projections](msm-projections.md#variations). The clever covariate that the package builds divides by the treatment density at each grid dose. In that construction, this composition also needs the response or intermediate mechanism at each grid dose, and the package does not write it. `TMLE` raises `CapabilityError` before any learner. `CausalStudy.identify` raises it for `MSMProjection` with a missing outcome. `tests/unit/test_continuous_msm_mechanism_refusals.py` checks both. [X10](../roadmap.md#x10-continuous-dose-msm-with-a-second-mechanism) reopens it |
| `DRTMLE` with cross-fitted missing outcomes at every `guard` including `guard=()`, `intermediate=`, fold-wise targeting, composition with `CTMLE`, `reduction="bivariate"` composed with `randomized=True`, `evaluation=` or `reduced_crossfit="nested"` on the composite indicator, or `targeting="one_step"` with `reduced_crossfit="nested"` at a non-empty `guard` | `implementation_gap`. `api_redirection` for `reduction="bivariate"` with `randomized=True`, and for `targeting="one_step"` with `reduced_crossfit="nested"` | [method presets](../user-guide/methods-learners.md#method-presets), and the [DR-TMLE refusals](dr-tmle/supported-estimands.md#refused-by-name) |
| a declared missing treatment (`treatment_delta=`) with `att`, `atc`, `ey_obs`, `par`, `paf`, `incremental=`, `policies=`, `learned_rule=`, `intermediate=`, `cross_fit=True`, `randomized=True`, a declared known mechanism, a selector-based `CTMLE`, or a shared-design `oat` fit | `invalid_declaration` for `att`, `atc`, `ey_obs`, `par`, `paf`, `incremental=`, `policies=`, `randomized=True`, and a declared known mechanism. `implementation_gap` for the rest | [the composite indicator](dr-tmle/theorem.md#observational-missing-data-the-composite-indicator). The recording may depend on the treatment, so `P(A = a \| W)` is not identified, and the first seven read it. No composite derivation is written for a learned rule ([F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions)) or a controlled direct effect. One gate in the shared preflight refuses each one by name before any learner. `CausalData` refuses `treatment_delta=` beside a continuous treatment as `invalid_declaration` for the same reason. `tests/unit/test_refusals_before_the_nuisance_fit.py` pins each refusal and a mutation that removes the gate. A cross-fitted `TMLE` waits on [X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) part (c), and a cross-fitted `DRTMLE` on [F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants). A selector-based `CTMLE` waits on [X51](../roadmap.md#x51-c-tmle-point-estimates-for-further-targets) part (d). A cross-fitted per-arm `oat` fit waits on [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference), which holds the cross-fitted outcome-adaptive construction. An in-sample per-arm `oat` fit reports a point estimate under [the per-arm contract](collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design) |
| an undeclared missing treatment value | `invalid_declaration` | the package does not infer a missing treatment from a missing value. `CausalData` raises `DataError` before any learner and names the declaration |
| a declared known treatment mechanism beside a continuous treatment, a vector treatment with a dose, or `policies=` with `ratio="classifier"`, and a declared `LTMLE` design that leaves a regimen unidentified | `implementation_gap` for a known density. `api_redirection` for `ratio="classifier"`. `invalid_declaration` for a regimen that no unit follows | [known treatment mechanism](point-treatment-tmle.md#known-treatment-mechanism). A known density is not supported, and the classifier learns the ratio that the declaration fixes. A declared zero on the arm a regimen needs leaves no unit able to follow it. Each refuses before any learner. `tests/unit/test_known_treatment_mechanism.py` and `tests/unit/test_known_node_mechanisms.py` pin the texts |
| baseline strata (`strata=`) with `targeting_scheme="fold"` or `cv_evaluation=True`, an incremental target with a stratum that lacks a treatment arm, an MSM whose design is singular inside a stratum, or a `DRTMLE` stratum with no trainable rows in some training complement | `implementation_gap` for fold targeting and fold evaluation. `invalid_declaration` for the three data conditions | [weights, strata, and clusters](point-treatment-tmle.md#weights-strata-and-clusters). Fold targeting fits the fluctuation on the validation rows. Montoya et al. (2023), Section 3.2, Step 2(c), publish that update, and the package does not implement it with strata ([X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) and [X11](../roadmap.md#x11-learned-policy-follow-ups) part (h) hold its other compositions). [X61](../roadmap.md#x61-fold-targeting-and-fold-evaluation) part (b) holds fold targeting with strata. Fold evaluation needs stratum shares and a stratum-indexed fold average in each validation fold ([X28](../roadmap.md#x28-fold-evaluated-cv-tmle-with-baseline-strata)). The other three are data conditions where the root may not exist or no regression is fittable. Each refuses before any learner. `tests/unit/test_refusals_before_the_nuisance_fit.py`, `tests/unit/test_stratified_incremental_exact.py` and `tests/unit/test_stratified_msm_exact.py` check them |
| the MNAR tilt, `missingness_tilt()` and `tipping_gamma()`, on a shift, incremental, regime, MSM, ratio-only, or PAR-and-PAF-only fit with missing outcomes | `implementation_gap` | [missingness tilt and tipping gamma](validation-methods.md#missingness-tilt-and-tipping-gamma). The tilt re-mixes the arm-indexed means and their linear contrasts. No derivation here covers the tilt of these parameters. Both entry points raise the sentence of `fit_wide_tilt_refusal`. Both capability rows read `unavailable` with that sentence before any call. `TestTheTiltRowsReadTheCallsPredicate` in `tests/unit/test_capability_row_predicates.py` checks each kind |
| a learned-rule value from an in-sample or one-fold fit, the stacked report, fold-specific targeting, a categorical treatment, or a contrast with a known regime or an arm, `reference=` included | `implementation_gap` | [learned rules](point-treatment-tmle.md#learned-rules). Each request has a published source and no implementation here. `TMLE` raises `CapabilityError` before any learner. `CausalStudy.identify` raises it for a categorical treatment, and `IdentifiedEffect.estimate` for a setting. `CrossFitting(n_folds=1)` raises `MethodConfigurationError` at construction, before the estimand is known. [The refusals a caller can meet](cv-tmle.md#the-refusals-a-caller-can-meet) quotes each remedy. [X11](../roadmap.md#x11-learned-policy-follow-ups) reopens them |
| a learned-rule value with a continuous treatment, `intermediate=`, `weights=`, `id=`, `strata=`, `repeats` above 1, the full-refit bootstrap, `CTMLE` or `DRTMLE`; and every refutation except `placebo` and every sensitivity analysis on a learned-rule fit | `implementation_gap`. `inference_gap` for the full-refit bootstrap | [learned rules](point-treatment-tmle.md#learned-rules). No reviewed source gives the result. The fit refuses before any learner. `refute` reads `unavailable` by default because a refit relearns the rules. Only `refute(tests=("placebo",))` runs, because the placebo law gives every rule the value `E[Y]`. Each sensitivity row reads `unavailable` because no derivation for this target was reviewed. A learned-rule fit with missing outcomes refuses too. [X11](../roadmap.md#x11-learned-policy-follow-ups) parts (i) to (o) hold the repeats, the intermediate variable, weights, clusters, strata, the omitted-variable bound, simulated confounding and `refute`. [F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) holds the continuous treatment, the bootstrap, `CTMLE`, a fold-evaluated `DRTMLE` rule, missing outcomes, a missing treatment, the E-value and the tilt |
| `intermediate=` with `incremental=` | `api_redirection` | [incremental interventions](../user-guide/estimands.md#incremental-propensity-score-interventions) |
| a hazard-based survival fit outside a held baseline categorical treatment with no covariate after baseline, and the isotonic projection on an end-of-study fit or a working model | `implementation_gap`, `invalid_declaration` | [hazard and isotonic survival refusals](#hazard-and-isotonic-survival-refusals) lists every refusal. On a design that H1, H2, H7 or H11 refuses, `LTMLE` with `survival_model="iterated_mean"` and `isotonic=True` reports a monotone curve |
| an odds tilt on `LTMLE` with an estimated treatment mechanism, or `incremental=` on `LTMLE` | `api_redirection` | [incremental interventions over time](longitudinal-tmle.md#ipsi-refusals). `LTMLE` raises `CapabilityError` with code `api_redirection` for `incremental=` at construction. It raises `CapabilityError` before any learner for a tilt node whose mechanism is estimated. Both quote `LTMLE_TILT_REFUSAL`: "an odds tilt reads the estimated treatment mechanism, so its influence function has a mechanism term at every node (Kennedy 2019, Theorem 2), and LTMLE fits no mechanism fluctuation over time. Fit IPSI(regimens=[Incremental(2.0), ...]), which runs Kennedy's cross-fitted estimator. With a known treatment mechanism, LTMLE fits the tilt as a known stochastic policy". `tests/unit/test_ipsi_e2e.py` pins the text |
| an odds tilt on `SDR` | `api_redirection` | [incremental interventions over time](longitudinal-tmle.md#ipsi-refusals). `SDR` raises `CapabilityError` before any learner: "the SDR estimator needs a policy that does not depend on the observed-data law (Díaz, Williams, Hoffman and Schenck 2023, Section 4). An odds tilt reads the treatment mechanism, so its estimator is not doubly robust (Kennedy 2019, Section 3.3). Fit IPSI(regimens=[...])". `tests/unit/test_ipsi_e2e.py` pins the text |
| an `IPSI` plan that sets a node by a modified treatment policy or a `RiskRatioTilt` | `implementation_gap` | [incremental interventions over time](longitudinal-tmle.md#ipsi-refusals). `IPSI` raises `CapabilityError` before any learner. For a `Shift` at node `A2` the message reads: "regimen 'tilt then shift' sets node 'A2' by a modified treatment policy (Shift). IPSI estimates no density ratio for a modified treatment policy or a risk-ratio tilt, so it has no estimator for a plan with such a node. F33 in docs/roadmap.md tracks this. Fit that plan without an odds tilt by LTMLE or SDR". `tests/unit/test_ipsi_e2e.py` pins the text. [F33](../roadmap.md#f33-other-intervention-and-design-compositions) holds the request |
| `alpha=`, `max_iter=`, `tol=`, `stopping=`, `step_bound=`, `step_search=`, `iterate=`, `targeting=` or `fluctuation=` on `IPSI`, or a non-default `submodel_alpha`, `max_iter`, `tol`, `stopping`, `step_bound`, `step_search` or `iterate` on `IPSIMethod` | `invalid_declaration` | [incremental interventions over time](longitudinal-tmle.md#ipsi-choices-that-differ-from-ltmle). The estimator fits no fluctuation. `IPSI` and `IPSIMethod` raise `MethodConfigurationError` with code `invalid_declaration`, `IPSI` at construction. `tests/unit/test_ipsi_e2e.py` pins both |
| `ratio_trim=` on `IPSI` or `IPSIMethod` | `invalid_declaration` | [ratio trimming](longitudinal-tmle.md#ratio-trimming). The trim follows the `lmtp` rule for a per-node density ratio, and an IPSI tilt node has no such ratio in its weight. `IPSI` raises `CapabilityError` at construction, and `IPSIMethod` raises `MethodConfigurationError`. Both quote `IPSI_RATIO_TRIM_REFUSAL`. `tests/unit/test_longitudinal_ratio_trim.py` checks both |
| `ratio()` of an `IPSI` level at or below 0 on the log scale, or outside $(0, 1)$ on the odds scale, and an `IPSI` working model under a logit or log link whose projection does not converge | `invalid_declaration` | [incremental interventions over time](longitudinal-tmle.md#ipsi-choices-that-differ-from-ltmle). An average of influence-function values can leave the outcome range. `CapabilityError` names the level or the link. On a fit with `repeats` above 1 the check runs on every draw, and the message names the draw. The difference scale and the identity link stay available |
| `baseline=` with no column on `LongitudinalTreatment`, `LongPanel` or `TimeToEvent` | `implementation_gap` | [what the reader refuses](../user-guide/longitudinal.md#what-the-reader-refuses). The package fits no intercept-only nuisance design. The three layouts share one container, so `LongitudinalData` raises `DataError` before any learner, and the point-treatment covariate check refuses the same way. `tests/unit/test_empty_baseline_refusal.py` pins the code on all three layouts. [X63](../roadmap.md#x63-missing-data-intermediates-and-designs) part (c) reopens it |
| `n_folds` above 1 on `SequentialGComputation` or `SequentialIPW` | `implementation_gap` | [sequential g-computation and IPW](sequential-baselines.md#refusals). Neither baseline has a root-n result with a data-adaptive learner, in sample or cross-fitted. The fit raises `CapabilityError` before any learner. [F35](../roadmap.md#f35-inference-for-sequential-baselines-with-data-adaptive-nuisances) tracks the missing result |
| `repeats` above 1 on `SequentialGComputation` or `SequentialIPW` | `implementation_gap` | [sequential g-computation and IPW](sequential-baselines.md#refusals). A baseline fits in sample, and its interval is the stacked estimating equation of one parametric fit. The median rule of Chernozhukov et al. (2018, Definition 3.3) has no split to combine. The fit raises `CapabilityError` before any learner. `tests/unit/test_sequential_baselines_repeats.py` pins the record. [F35](../roadmap.md#f35-inference-for-sequential-baselines-with-data-adaptive-nuisances) tracks the missing result |
| `msm=`, `ratio_trim=`, `hazard_pooling="pooled"`, or an odds tilt with an estimated mechanism, on `SequentialGComputation` or `SequentialIPW`; and a modified treatment policy on `SequentialIPW` | `api_redirection` | [sequential g-computation and IPW](sequential-baselines.md#refusals). The fit raises `CapabilityError` before any learner and names `LTMLE`, `SDR`, `IPSI` or `SequentialGComputation`. `tests/unit/test_sequential_baselines.py` pins each text |
| a fluctuation setting on `SequentialGComputation` or `SequentialIPW`, including `stopping=`, `step_bound=`, `step_search=` and `iterate=`, an outcome learner or a density setting on `SequentialIPW`, and a continuous outcome with a binomial `GLM` and `q_bounds=None` on `SequentialGComputation` | `invalid_declaration` | [sequential g-computation and IPW](sequential-baselines.md#refusals). A fluctuation setting raises `MethodConfigurationError`, and the others raise `CapabilityError`, before any learner. The scale refusal names `q_bounds=` and `GLM(family="gaussian")` |
| `SequentialGComputation` or `SequentialIPW` on a repeated outcome | `implementation_gap` | [repeated outcomes](longitudinal-tmle.md#repeated-outcomes). The fit raises the shared repeated-outcome refusal before any learner. [X71](../roadmap.md#x71-repeated-outcomes-for-the-sequential-baselines) owns the reduction |
| a static level other than the reference, a contrast, or `check_model` on `StructuralNestedMean` when a blip feature reads a time-varying covariate | `implementation_gap` | [structural dynamic effects](structural-dynamic-effects.md#refusals). The fit reports the blips and `ey_reference`. The static value needs the law of the covariate under the regimen. [X72](../roadmap.md#x72-static-regimen-values-from-structural-blips-with-time-varying-features) owns it |
| a rule, a modified treatment policy, a known stochastic plan, an incremental policy, censoring, a survival outcome, a repeated outcome, a held decision, a declared mechanism, a declared `markov_order`, or `msm=` on `StructuralNestedMean` | `api_redirection` | [structural dynamic effects](structural-dynamic-effects.md#refusals). The fit raises `CapabilityError` before any learner and names `LTMLE`, `SDR` or `IPSI`. `tests/unit/test_structural_nested_mean.py` (W7) pins each code |
| a static dose outside the observed range, a dose node whose range excludes the reference dose 0, a singular diagonal Jacobian block, a blip feature that reads a later covariate or an unknown column, and `check_model` on a continuous node, on `StructuralNestedMean` | `invalid_declaration` | [structural dynamic effects](structural-dynamic-effects.md#refusals). The message names the range, the block or the node |
| `ParametricGFormula` on a repeated outcome, or `n_folds` above 1 | `implementation_gap` | [parametric g-formula](parametric-gformula.md#refusals). The fit raises `CapabilityError` before any learner. [X71](../roadmap.md#x71-repeated-outcomes-for-the-sequential-baselines) owns the repeated outcome, and [F35](../roadmap.md#f35-inference-for-sequential-baselines-with-data-adaptive-nuisances) owns cross-fitting |
| `msm=`, a risk-ratio tilt, `markov_order=`, `ratio_trim=` or `hazard_pooling="pooled"` on `ParametricGFormula`, and a truncation curve of its result | `api_redirection` | [parametric g-formula](parametric-gformula.md#refusals). The fit raises `CapabilityError` and names `LTMLE`, `SDR` or `SequentialIPW` |
| an undeclared time-varying column that is not coded 0/1, a learner that is not a `GLM` for a Gaussian column, a learner that nothing reads, and a declared treatment mechanism with a missing value at a tilt or policy node, on `ParametricGFormula` | `invalid_declaration` | [parametric g-formula](parametric-gformula.md#refusals). The fit raises `CapabilityError` before any learner. The message names `covariate_types=` where that applies |
| sample sensitivity-bound estimation for `LTMLE` and `SDR` | `implementation_gap` | [longitudinal diagnostics](../user-guide/longitudinal.md#diagnostics). See [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) for the contracts Tan (2025) leaves open |
| `alpha=`, `max_iter=`, `tol=`, `stopping=`, `step_bound=`, `step_search=` or `iterate=` on `SDR`, or a non-default `submodel_alpha`, `max_iter`, `tol`, `stopping`, `step_bound`, `step_search` or `iterate` on `SDRMethod` | `invalid_declaration` | [the sequentially doubly robust estimator](longitudinal-tmle.md#the-sequentially-doubly-robust-estimator). `SDR` fits no fluctuation, so it has no logistic shrink, iteration limit or score tolerance. `SDR` and `SDRMethod` raise `MethodConfigurationError`, `SDR` at construction. The message names the `LTMLE` keyword that takes the setting. `tests/unit/test_sdr_e2e.py` pins both texts |
| `method="sdr"` on a point-treatment functional | `api_redirection` | [the sequentially doubly robust estimator](longitudinal-tmle.md#the-sequentially-doubly-robust-estimator). The estimator is the sequential regression of a longitudinal fit. `available_methods()` lists it as unavailable with the reason, and `estimate()` raises `CapabilityError` before any learner |
| `ratio()` of an `SDR` level at or below 0 on the log scale, or outside $(0, 1)$ on the odds scale | `invalid_declaration` | [the sequentially doubly robust estimator](longitudinal-tmle.md#the-sequentially-doubly-robust-estimator). An average of transforms can leave the outcome range (Díaz et al. 2023, page 854), and the scale is undefined there. `CapabilityError` names the level and its value. On a fit with `repeats` above 1 the check runs on every draw, and the message names the draw. The difference scale stays available |
| `n_bootstrap` above 0 with `repeats` above 1 on `LTMLE`, `SDR` or `IPSI` | `inference_gap` | [repeated fold draws](longitudinal-tmle.md#repeated-fold-draws). No registered study measures a bootstrap of the median over draws, and each replicate would refit every draw. The constructor raises `MethodConfigurationError` and names `n_bootstrap=0` for the split-adjusted median interval. The point-treatment engine runs that bootstrap as a diagnostic |
| an `SDR` working model under a logit or log link whose projection does not converge | `invalid_declaration` | [MSM projections](msm-projections.md#the-longitudinal-projection). A transform mean outside the link's range has no finite solution. The fit raises `CapabilityError` and names the identity link and `LTMLE(msm=...)` |
| blocked-temporal and rolling-origin splits | `implementation_gap` | [two fold layers](../user-guide/methods-learners.md#two-fold-layers) |
| a confidence interval, a p-value, or a standard error on a `CTMLE` fit whose `strategy` is `"greedy"`, `"ordered"`, `"discrete"`, `"lasso"`, or `"learners"` | `inference_gap` | [collaborative TMLE](collaborative-tmle.md). No result shows the reported curve is the estimator's influence curve when the selected working mechanism is not consistent for the treatment law. `ci`, `pvalue`, and `std_error` raise `CapabilityError`. The point estimate, the selection path, and the curve remain, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. A `discrete` fit with one declared candidate, equal to the full adjustment set, selects nothing and keeps the TMLE interval. A `learners` fit with an empty `treatment_candidates=` list does the same. [F18](../roadmap.md#f18-selector-path-c-tmle-inference) reopens it |
| a confidence interval, a p-value, or a standard error on every `CTMLE` fit with `strategy="oat"`, of either `oat_design=`, including a fit with `strata=`, `delta=` or `treatment_delta=` and a fit that requests one arm mean | `inference_gap` | [collaborative TMLE](collaborative-tmle.md#the-applied-revert). The fit takes the `"generated_design_plugin"` status. No result covers the shared design. The per-arm design applies Theorem 1 of Benkeser, Cai and van der Laan (2020) to each arm. That theorem's condition (v) fails for an estimated outcome regression, and the registered study `ctmle-oat-per-arm` measured standard errors 8 to 24 percent too small. The point estimate remains, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) reopens it |
| a confidence interval, a p-value, or a standard error on a `DRTMLE` fit with a non-empty `guard` and varying weights declared estimated (`weights_estimated=True`) | `inference_gap` | [DR-TMLE refusals](dr-tmle/supported-estimands.md#refused-by-name). The fit takes the `"estimated_weight_plugin"` status. The argument that an interval conditions on the weights concerns $D^*$, and not the reduced regressions. The point estimate remains, and `plugin_std_error` and `plugin_interval` report the retained diagnostic. `guard=()` fits the ordinary TMLE and keeps its interval. Constant weights fit the unweighted estimator and keep it too. [X50](../roadmap.md#x50-further-dr-tmle-compositions) part (f) reopens it |
| a confidence interval, a p-value, or a standard error on a `TMLE` or `DRTMLE` fit with `id=` and fewer than 10 positive-mass clusters in the fit or one reported baseline stratum, or an `LTMLE` or `SDR` fit with fewer than 20, in sample or cross-fitted | `inference_gap` | [clusters](inference.md#clusters). The fit takes the `"few_cluster_plugin"` status, in sample or cross-fitted. One such stratum withholds every interval. From that floor to 39 such clusters the fit reports intervals on a Student $t$ reference with $J - 2$ degrees of freedom, as Nugent et al. (2024), Section 2.2, recommend. Each floor is the smallest count that the registered few-cluster study measures for that fit. The point estimate remains, and `plugin_std_error` and `plugin_interval` retain the diagnostic on the normal reference. [F28](../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns the counts below each floor. [X54](../roadmap.md#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) holds the measurements from 4 clusters to each floor, and F28 keeps fits below 4 clusters |
| a confidence interval, a p-value, or a standard error on an `LTMLE` or `SDR` fit with `ratio_trim=` whose trim moves a positive per-node density ratio | `inference_gap` | [ratio trimming](longitudinal-tmle.md#ratio-trimming). The fit takes the `"trimmed_ratio_plugin"` status. A trim that binds keeps a fixed share of the ratios at every sample size, so the remainder keeps a term of the size of the outcome-regression error. No source read here gives an interval for the trimmed fit. The point estimate remains, and `plugin_std_error` and `plugin_interval` report the plug-in diagnostic, which is the standard error `lmtp` reports. [F30](../roadmap.md#f30-inference-after-quantile-ratio-trimming) reopens it |
| `simultaneous_bands()` over an estimate with a Student $t$ reference, below 40 positive-mass clusters | `inference_gap` | [clusters](inference.md#few-clusters). No source read here gives a $t$-calibrated joint band. `simultaneous_bands()` raises `CapabilityError`, and a fit skips its default band and names the reason in `summary()`. The pointwise intervals remain. `tests/unit/test_few_cluster_reference.py` checks both |
| a contrast, a simultaneous band, any E-value branch, `variable_importance()`, or `tipping_gamma(use_ci=True)` on the fits of the five status rows above | `inference_gap` | [inference status](inference.md#inference-status). Each one is a confidence statement, or reads an interval that these fits do not supply. `variable_importance()` refuses at its entry point, because it adjusts one p-value per candidate. The E-value capability reports `unavailable` rather than raising inside the computation. `truncation_curve()`, `missingness_tilt()`, and the default `tipping_gamma()` still run. On a point-treatment fit the two frames rename their three spread columns. The longitudinal `truncation_curve()` publishes no spread. `LongitudinalResult.curve()` and `incidence_total()` rename their spread columns |
| `stratify_by="treatment"` and `stratify_by="treatment+outcome"` on any fit that draws a split, including selector-based C-TMLE at every setting | `implementation_gap` | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). No shipped result covers a partition read off the data the fit then conditions on |
| a cross-fitted continuous outcome with `q_bounds=None`, for `TMLE`, `DRTMLE`, `CTMLE`, `LTMLE` and `SDR` above one fold | `invalid_declaration` | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). The sample outcome range would take the scale from held-out rows |
| `CTMLE` with `id=`, at every `cross_fit` setting, on every strategy except `"derivative"` | `implementation_gap` | [fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules). No clustered result covers the outcome-adaptive mechanism. Selector-based fits also lack a result for grouped selection and nested folds or candidate-selection variance. A `"derivative"` fit selects nothing and takes the clustered result of `TMLE` |
| `cv_evaluation=True` or `targeting_scheme="fold"` with `id=`, when a validation fold holds 1 cluster | `invalid_declaration` | [the algorithm as implemented](cv-tmle.md#the-algorithm-as-implemented). Each fold's variance takes the variance of the cluster totals inside the fold, and 1 total has none. `TMLE` raises `CapabilityError` before any learner, for a generated or a supplied split. A generated split passes at no more folds than half the cluster count. `tests/unit/test_fold_evaluated_cluster_variance.py` checks it |
| the `bootstrap_measurement_error` refutation on a fit that declared `split_plan=` | `invalid_declaration` | [reusable outer split plans](cv-tmle.md#reusable-outer-split-plans). It refits on rows the fit never ran, and no rule here says which fold labels those rows inherit. `refute()` raises `CapabilityError` before it refits anything. The `subset` refutation runs, because each kept row carries its original fold label |
| replicate weights (BRR, jackknife) | `invalid_declaration` | [observation weights](../user-guide/data-design.md#observation-weights-are-not-estimand-weights). These are a set of designs rather than one weight vector, so the shape they want is a refit per replicate outside the estimator |
| omitted-variable sensitivity after `repeats=` | `implementation_gap` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The median bound needs an influence function; a coordinatewise median of per-draw influence terms is not one. The five omitted-variable capability rows declare this refusal as `unavailable` before any call, and `test_the_five_declared_rows_carry_that_same_reason` checks them |
| omitted-variable sensitivity on a `DRTMLE` or `CTMLE` fit, except a `CTMLE` fit with `strategy="derivative"` | `implementation_gap` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). Neither estimator assumes a consistent treatment mechanism, and no derivation gives $\nu^2$ or the bound's standard error without that assumption. A `"derivative"` fit assumes a consistent mechanism as `TMLE` does, so the bound admits it |
| omitted-variable sensitivity on a fit with a response mechanism | `implementation_gap` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The bound is well posed, and the implementation is missing. The implemented representer omits the response indicator, and $\sigma^2$ averages the respondents only. Use the missingness tilt for response on the arm means. The tilt does not cover `ey_obs`, PAR or PAF. An explicit tilt request that names one of them raises a sentence that names them. On a fit that reports one of them, the `tipping_gamma` row reads `deferred` and requires `estimand`, and the bare call raises the row's reason. The default tilt sweep skips them |
| omitted-variable sensitivity on a fit with a declared missing treatment | `implementation_gap` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The Riesz representer of the bound is derived for a recorded treatment. This fit divides by the composite mechanism, whose representer and confounding strengths are not derived |
| omitted-variable sensitivity on a fit with an intermediate variable | `implementation_gap` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The bound is well posed, and the implementation is missing. The representer carries the intermediate weight, so the treatment strength $c_D$ would also measure the intermediate mechanism |
| omitted-variable sensitivity on a `regime`, `shift`, or `msm` parameter axis | `implementation_gap` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). Each parameter has a Riesz representer, so the bound is well posed. Only the implementation is missing |
| omitted-variable sensitivity on an `ipsi` parameter axis | `invalid_declaration` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). An incremental intervention tilts the mechanism, so the mechanism is part of the estimand rather than a nuisance |
| omitted-variable sensitivity on an arm-indexed fit that reports no counterfactual mean and no linear contrast. A ratio-only fit, a PAR or PAF fit, and a complete-outcome `ey_obs` fit are such fits | `invalid_declaration` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). One bound is the second moment of one contrast's Riesz representer, and the fit reports no such contrast. The five capability rows read `unavailable` with the sentence that each call raises. On a fit that reports `rr` or `or`, the sentence names `sensitivity.evalue()`. `TestTheBoundRowsReadTheCallsTable` in `tests/unit/test_capability_row_predicates.py` checks four such fits |
| `benchmark()` with every covariate of the fit named | `implementation_gap` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). The short model would adjust for nothing, and no estimator here fits without a covariate. `benchmark()` raises `CapabilityError` before the refit. On a fit with one covariate, the `benchmark` capability row reads `unavailable`, because no value of `covariates` runs. On a wider fit, a request that names every covariate reads `unavailable`. An unknown name raises `DataError` before this refusal |
| omitted-variable sensitivity when the doubly robust estimate of $\nu^2$ is not positive | `invalid_declaration` | [validation and sensitivity methods](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours). This row reads the data, not the fit class. The other omitted-variable rows refuse a whole class of fit before any computation, and their capability rows report `unavailable`. This one leaves the capability row `available`, and the call raises `CapabilityError` only when the fitted representer gives a nonpositive second moment. An ordinary TMLE script that received a bound before can therefore raise now, because the package substitutes no plug-in value for the refused one |
| the one-sided limits and the confidence-limit robustness value under `nu2_estimator="plugin"`, when the treatment learner is not a qualifying `LogisticRegression` | `inference_gap` | [the plug-in learner rule](validation-methods.md#plug-in-limits-and-the-learner-rule). A qualifying learner is a scikit-learn `LogisticRegression`, binary or multinomial, with no penalty or a fixed L2 penalty, fitted bare or after `StandardScaler` steps, in sample or cross-fitted. No source this package cites gives the standard error of a bound built on the plug-in $\nu^2$ for another learner, because it moves at first order with the fitted treatment mechanism. `SuperLearner`, `LogisticRegressionCV`, an L1 or elastic-net penalty, `solver="liblinear"`, `class_weight`, other pipeline steps and `screen_treatment=True` refuse with the roadmap item [F26](../roadmap.md#f26-confidence-limits-of-the-plug-in-omitted-variable-bound). The bounds, `rv`, `max_bias`, `benchmark()` and `contour()` remain. The limits under the default `nu2_estimator="doubly_robust"` remain available |
| the same limits under `nu2_estimator="plugin"`, when a qualifying `LogisticRegression` fails a check on its fit | `inference_gap` | [the plug-in learner rule](validation-methods.md#plug-in-limits-and-the-learner-rule). The check is a missing arm in a fold, a score that is not zero at the coefficients, a singular information matrix, or a mismatch with the stored mechanism. A treatment model that read a different covariate set refuses the same way. These refusals carry no roadmap item, because F26 does not track a declared model that failed a check |
| the same limits under `nu2_estimator="plugin"`, when the second-order correction of $\nu^2$ does not hold on a qualifying fit | `inference_gap` | [the second-order correction](validation-methods.md#plug-in-limits-and-the-learner-rule). Three keys apply. `truncation_active`: a truncation bound clips a cell that the representer reads, or a fitted logit lies within 3 standard errors of a bound. `correction_unstable`: the predicted remainder $\hat B^2/\hat\nu^2$ exceeds 0.25 of the standard error of $\nu^2$. `corrected_nonpositive`: the corrected $\nu^2$ is not positive. These refusals carry no roadmap item. The remedy is `nu2_estimator="doubly_robust"`. The property study of the corrected limits is pending |
| the same limits under `nu2_estimator="plugin"` with a declared known mechanism that the truncation bounds move | `invalid_declaration` | [the plug-in learner rule](validation-methods.md#plug-in-limits-and-the-learner-rule). The fit refuses `g_bounds` that move a known probability, and the bound repeats the check. Pass `g_bounds` that contain every known probability |
| omitted-variable limits when $\sigma^2=0$ | `inference_gap` | [the bound's standard error](validation-methods.md#standard-error-of-the-omitted-variable-bound) needs a positive maximal-bias scale. The point bounds remain, but the limit accessors refuse. An unreachable point robustness threshold is `None` |
| any E-value request on a fit with an intermediate variable and a discrete treatment | `implementation_gap` | [the E-value paths](validation-methods.md#e-value). This package has no implemented bound for the fitted controlled direct effect and its confounding model. The ordinary conversion does not establish sensitivity for that target. The capability row reports `unavailable` before any computation. A continuous-treatment fit with an intermediate variable reports `not_applicable`, because that check runs first. [X49](../roadmap.md#x49-e-value-for-a-controlled-direct-effect) tracks the support gap |
| the standardized Gaussian E-value on a fit with a response mechanism | `implementation_gap` | [the E-value paths](validation-methods.md#e-value). The conversion needs the population outcome standard deviation, and the respondents' one estimates a different quantity. A ratio E-value on the same fit stays available |
| correction diagnostics after ordinary TMLE or collaborative TMLE | `invalid_declaration` | those methods do not use the DR-TMLE correction system. A DR-TMLE fit with an empty guard also subtracts no correction term |
| a binomial ATT or ATC E-value conversion | `implementation_gap` | the analysis needs a conditional baseline risk and a supported conditional risk-ratio target. The library implements neither |
| a fixed-baseline E-value on a reference-arm mean that its own standard error does not separate from zero | `invalid_declaration` | [the E-value paths](validation-methods.md#e-value). The conversion divides by that mean, so the ratio has no stable denominator |
| a fixed-baseline E-value whose risk difference is at or below the negative of the reference-arm mean | `invalid_declaration` | [the E-value paths](validation-methods.md#e-value). The difference implies a nonpositive risk in the contrast arm, and no risk ratio describes one |
| a cached-nuisance risk ratio after CV evaluation | `implementation_gap` | the estimator refuses nonlinear ratio targets on this evaluation path. Post-fit retargeting keeps that boundary |
| a controlled direct risk ratio from cached nuisances | `implementation_gap` | the fitted result can report a controlled direct risk ratio. The post-fit retarget path does not pass the fitted intermediate intervention level, so it cannot derive one from cached nuisances |

Which multi-arm surfaces are covered, and which are not, is tabulated in one place:
[where a multi-valued treatment is supported](#where-a-multi-valued-treatment-is-supported).

Several former gaps have landed. `cleverly` now supports multi-valued longitudinal treatment
nodes. It supports multi-valued selector-based C-TMLE, outcome-adaptive C-TMLE, DR-TMLE, `ATT`,
and `ATC`. `LTMLE` supports observation weights and a working model over regimens. `IPSI` fits
incremental interventions over time. Shift fits
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

### Targeting and fitting option refusals

These options are `Targeting(stopping=, step_bound=, step_search=, iterate=, node_bounds=, ratio_trim=)`,
`ModelSpec(outcome_by_arm=)`, `CollaborativeTMLEMethod(patience=)` and
`PointTreatment(treatment_adjustment=)`. A refusal is raised before any learner fits, unless the
row says otherwise. A malformed value raises `ValueError` with no record. Every other row carries
the code shown.
`tests/unit/test_stopping_rule.py`, `test_step_bound.py`, `test_best_iterate.py`,
`test_longitudinal_node_bounds.py`, `test_outcome_by_arm.py`, `test_patience.py`,
`test_treatment_covariates.py`,
`test_treatment_covariates_collaborative.py`, `test_hazard_targeting_rules.py`,
`test_hazard_node_bounds.py` and `test_hazard_refusals.py` assert each text
with `pytest.raises(match=re.escape(...))`.

| id | refused | class and code | message |
| --- | --- | --- | --- |
| R-a1 | `iterate=` other than `"last"` or `"best"` | `ValueError` | `iterate must be 'last' or 'best'; got {v!r}` |
| R-b1 | `stopping=` other than `"relative"`, `"scaled_var"` or `"sample_size"` | `ValueError` | `stopping must be 'relative', 'scaled_var' or 'sample_size'; got {v!r}` |
| R-c1 | `step_bound=` that is not a positive finite number, a `bool` included | `ValueError` | `step_bound must be a positive finite number; got {v!r}. step_bound=None takes the full Newton step` |
| R-c1a | `step_search=` that is not a `bool` | `ValueError` | `step_search must be True or False; got {v!r}` |
| R-c2 | `step_bound=` with `algorithm="one_step"` | `MethodConfigurationError`, `api_redirection` | `algorithm='one_step' already walks the universal least-favorable submodel in steps of step_size. step_bound= bounds the steps of algorithm='iterative'.` |
| R-c3 | `step_search=False` without `step_bound=` | `MethodConfigurationError`, `invalid_declaration` | `step_search= chooses how a bounded step is taken, so it needs step_bound=` |
| R-c4 | `step_bound=` with `survival_model="hazard"` and `targeting="one_step"`. `LTMLE` raises it when it validates its settings, after the malformed-value checks | `MethodConfigurationError`, `api_redirection` | `targeting='one_step' with survival_model='hazard' already walks the universal least-favorable submodel in steps of step_size. step_bound= bounds the steps of targeting='iterative'.` |
| R-d1 | `ratio_trim=` with `survival_model="hazard"`. `LTMLE` raises it when it validates its settings, after the malformed-value checks | `CapabilityError`, `implementation_gap`, roadmap X80 | `ratio_trim= with survival_model='hazard' is not built. lmtp's trim acts on the density ratios of the iterated-mean recursion, and the hazard targeting has no trim of its clever covariates. X80 in docs/roadmap.md tracks this. Fit ratio_trim= with survival_model='iterated_mean', or fit survival_model='hazard' without ratio_trim= and bound the treatment mechanism with g_bounds=.` |
| R-abc1 | `iterate="best"`, a declared `stopping=` or `step_bound=` with `fluctuation="linear"` | `MethodConfigurationError`, `invalid_declaration` | `fluctuation='linear' solves the score equation in one closed-form step, so it has no iterate, stopping rule or step bound to choose. Remove iterate=, stopping= and step_bound=, or use fluctuation='logistic'` |
| R-abc2 | a non-default `stopping=`, `step_bound=`, `step_search=` or `iterate=` on `SDR`, `SDRMethod` | `MethodConfigurationError`, `invalid_declaration` | `the SDR estimator fits no fluctuation, so it has no logistic shrink, iteration limit or score tolerance. LTMLE takes stopping=, so pass the setting there.` The text names the keyword that was declared |
| R-abc3 | the same settings on `IPSI`, `IPSIMethod` | `MethodConfigurationError`, `invalid_declaration` | `IPSI fits no fluctuation: Kennedy's estimator averages the influence function at the fitted nuisances, so it has no logistic shrink, iteration limit, score tolerance or submodel.` |
| R-abc3a | the same settings on `GComputationMethod`, `IPWMethod`, `SequentialGComputation`, `SequentialIPW` | `MethodConfigurationError`, `invalid_declaration` | `the sequential g-computation and IPW estimators fit no fluctuation, so they have no logistic shrink, iteration limit or score tolerance. LTMLE takes stopping=, so pass the setting there.` |
| R-i1 | `node_bounds=` that names a grid time outside the fit, a cause the fit did not declare, a pair with `lower >= upper`, or a bound outside the outcome range. On a fit with `survival_model="hazard"` the range is $[0, 1]$, the hazard scale. The check runs at fit time before any learner. A value that is not a pair of numbers, or a `node_bounds=` that is not a mapping, raises `ValueError` with its own text | `ValueError` | `node_bounds names grid time {t!r}, which is not a node of this fit; its grid is {grid}`, `node_bounds names cause {c!r}; this fit declared {causes}`, `node_bounds at {t!r} needs lower < upper; got ({l!r}, {u!r})`, `node_bounds at {t!r} lies outside the outcome range [{a:g}, {b:g}]`, and on a hazard fit `node_bounds at {t!r} lies outside the hazard range [0, 1]` |
| R-i2 | a node of a fit with `survival_model="hazard"` whose upper bounds sum to more than one. The check runs at fit time before any learner | `CapabilityError`, `invalid_declaration` | `node_bounds at {t!r} has upper bounds summing to {s:g}; a hazard fit needs them to sum to at most 1, so that the bounded hazards leave room for no event. Lower the upper bounds at {t!r} so that they sum to at most 1.` |
| R-i3 | `node_bounds=` with `msm=` | `CapabilityError`, `implementation_gap`, roadmap X78 | `node_bounds= bounds the intercept fluctuation of a per-regimen mean. A working model over regimens pools one fluctuation across the regimens, and its bounded form is not built. Fit node_bounds= without msm=, and report the regimen levels.` |
| R-i4 | `node_bounds=` on `SDR`, `IPSI`, `SequentialGComputation`, `SequentialIPW`, `ParametricGFormula`, `SDRMethod`, `IPSIMethod`, `GComputationMethod` or `IPWMethod` | `MethodConfigurationError`, `invalid_declaration` | `{estimator} fits no logistic node fluctuation, so a node bound has nothing to bound. LTMLE(node_bounds=...) bounds the logistic fluctuation of each node.` |
| R-i5 | `node_bounds=` on a point-treatment functional | method configuration, `MethodConfigurationError`, `invalid_declaration` | `node_bounds= bounds the logistic fluctuation of each node of a longitudinal iterated-mean fit, and a point-treatment functional has no such nodes. Remove it, or declare a longitudinal or TimeToEvent design.` |
| R-i6 | a competing-risk fit with `survival_model="hazard"` that bounds some causes of a node and not others. The check runs at fit time before any learner | `CapabilityError`, `invalid_declaration` | `node_bounds bounds cause(s) {named} at {t!r} and not {missing}; a hazard fit bounds every cause of a node or none. A bounded node holds each cause on a fixed scale, and an unbounded cause there would share the mass the bounded causes leave. Bound every cause at {t!r}, or none of them.` |
| R-f1 | `outcome_by_arm=True` with a continuous dose, including a dose MSM and a shift policy on a dose. The check runs before any learner fits | `CapabilityError`, `invalid_declaration` | `outcome_by_arm=True fits one outcome regression per treatment arm, so it needs an arm-coded treatment. A continuous dose has no arms. Use outcome_by_arm=False, which fits one regression on the dose.` |
| R-f2 | an arm with no training row in a fold. It is raised at fit time | `DataError` | `outcome_by_arm=True found no row with {treatment} = {arm!r} and an observed outcome in {where}, so that arm's outcome regression has nothing to fit. Use fewer folds, or outcome_by_arm=False`. `{where}` reads "training fold {k}" or "the training sample" |
| R-f3 | `outcome_by_arm=` on a longitudinal design | `MethodConfigurationError`, `invalid_declaration` | `longitudinal TMLE cannot apply point-treatment option(s): ['outcome_by_arm']. Remove them or use a PointTreatment design.` |
| R-g1 | `patience=` that is not a positive integer, a `bool` included | `ValueError` | `patience must be a positive integer; got {p!r}` |
| R-g2 | `patience=` with `strategy="oat"` or `"derivative"`, which score no path | `ValueError` | `patience= stops the cross-validated scoring of a candidate path, and strategy={s!r} scores no path. It applies to 'greedy', 'ordered', 'discrete', 'lasso' and 'learners'` |
| R-h1 | `treatment_adjustment=` that names a column outside `adjustment=` | `DataError` | `treatment_adjustment names {cols} outside adjustment=. The treatment mechanism conditions on a subset of the adjustment set` |
| R-h2 | `treatment_adjustment=` with `CTMLE(strategy="oat")` | `CapabilityError`, `invalid_declaration` | `strategy='oat' fits the treatment mechanism on the outcome predictions, so treatment_adjustment= has no covariates to restrict. Use another strategy, or omit treatment_adjustment=.` |
| R-h3 | `treatment_adjustment=` with a point `incremental=` estimand | `CapabilityError`, `different_estimand` | `incremental= shifts the fitted propensity score, so a mechanism on {Wg} defines the shift on {Wg} and changes the estimand. Omit treatment_adjustment=, or treatment_covariates= in TMLE.fit.` |
| R-h4 | `treatment_adjustment=` with a known `treatment_probabilities=` | `DataError` | `treatment_adjustment= restricts a fitted treatment mechanism, and treatment_probabilities= supplies a known one. Remove one of them` |
| R-h5 | a C-TMLE `candidates=` or `ordering=` that names a covariate outside the declared subset | `ValueError` | `{arg} names {cols} outside treatment_adjustment=` |
| R-h6 | a repeated name in `treatment_adjustment=` | `DataError` | `treatment_adjustment repeats {cols}` |

[X78](../roadmap.md#x78-bounded-node-fits-under-a-working-model) tracks R-i3.
[X80](../roadmap.md#x80-ratio-trimming-on-the-hazard-path) tracks R-d1.

`TMLE.fit(treatment_covariates=)` raises R-h1, R-h4 and R-h6 with the argument named
`treatment_covariates` in place of `treatment_adjustment`.

A declared stopping rule can leave a fit unconverged. That is a result and not a refusal. The
fit warns, reports `converged=False`, and `score_check()` fails it.
[Stopping rules](point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate) gives the
text.

### AdaptiveIPTW refusals

`AdaptiveIPTW` and `AdaptiveIPTWMethod` raise every refusal in the first table before any learner
fits, except the range rule. A spy learner shows zero `fit` calls in
`tests/unit/test_adaptive_iptw_refusals.py`. The [contract page](adaptive-iptw.md) states the
estimator. The ids follow the numbering of that test module.

| id | refused | class and code | where |
| --- | --- | --- | --- |
| R1 | a treatment that is not arm-coded, including a deterministic regime on a continuous dose | `CapabilityError`, `api_redirection` | use `TMLE` with `policies=[Shift(...)]` |
| R2b | `incremental=`, `learned_rule=` or `policies=` | `CapabilityError`, `api_redirection` | use `TMLE` |
| R3 | `att`, `atc` or an MSM | `CapabilityError`, `api_redirection` | use `TMLE` |
| R4 | `intermediate=`, or a non-default `intermediate_learner=` | `CapabilityError`, `implementation_gap` | [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions). No weighting of `AdaptiveIPTW` is derived for a controlled direct effect |
| R8 | `targeting_scheme="fold"` | `MethodConfigurationError`, `implementation_gap` | [X61](../roadmap.md#x61-fold-targeting-and-fold-evaluation) part (d) |
| R9 | `cv_evaluation=True` | `MethodConfigurationError`, `implementation_gap` | [X61](../roadmap.md#x61-fold-targeting-and-fold-evaluation) part (d) |
| R10 | `n_bootstrap` above 0, or a non-default `bootstrap_resampling` | `MethodConfigurationError`, `inference_gap` | [X55](../roadmap.md#x55-bootstrap-extensions) part (d). Pass `treatment_probabilities=` for a Wald interval |
| R11 | `form` other than `"targeted"` or `"one_step"` | `ValueError` | `form must be 'targeted' or 'one_step'; got {form!r}` |
| R12 | longitudinal data, or `estimator_kwargs(longitudinal=True)` | `MethodConfigurationError`, `api_redirection` | use `SequentialIPW` (`method="ipw"`) |
| R13 | a non-default `fluctuation`, `q_bounds`, `alpha`, `target_weights`, `step_size`, `targeting="one_step"`, `outcome_by_arm`, `iterate`, `stopping`, `step_bound`, `step_search`, `ratio` or `density_bins`; `outcome_learner=` together with `reduced_outcome_learner=` | `MethodConfigurationError`, `invalid_declaration` | the estimator fits no outcome regression, so the keyword does not apply |
| R16 | `repeats` above 1 | `MethodConfigurationError`, `implementation_gap` | [X81](../roadmap.md#x81-adaptiveiptw-follow-ups) part (a) |
| R17 | a regime with missing data and `cross_fit=True` | `CapabilityError`, `implementation_gap` | [X81](../roadmap.md#x81-adaptiveiptw-follow-ups) part (b). Pass `cross_fit=False` |
| R19 | a classifier as the reduced learner of a stochastic regime | `MethodConfigurationError`, `invalid_declaration` | the regression target $g^*(a \mid W) Y$ is not binary |
| R20 | a known treatment mechanism with a declared missing treatment | the shipped refusal, `invalid_declaration` | the known $P(A = a \mid W)$ cancels from the composite mechanism, so it adds nothing |
| range rule | `rr`, `or` or `paf` requested when a Horvitz-Thompson mean leaves its domain | `DataError`, `invalid_declaration` | raised after targeting. The default report and `estimands="all"` drop the target and warn with `DataWarning` |
| status | a standard error, an interval or a p-value on a fit with an estimated mechanism | `CapabilityError`, `inference_gap` | the status `estimated_mechanism_plugin`, reopened by [X36](../roadmap.md#x36-projected-variance-for-adaptiveiptw-with-a-declared-parametric-mechanism) |

An operation on the result refuses by name as well.

| operation | code | where |
| --- | --- | --- |
| `validate` | `api_redirection` | use `diagnostics.nuisance_models()` |
| `diagnostics.truncation_curve` | `implementation_gap` | [X81](../roadmap.md#x81-adaptiveiptw-follow-ups) part (c) |
| `diagnostics.refute` | `implementation_gap` | [X67](../roadmap.md#x67-refutation-and-assessment-adapters) part (f) |
| `sensitivity.omitted_confounding`, `robustness_value`, `elements`, `benchmark`, `contour` | `api_redirection` | use `TMLE`. The bound needs an outcome regression |
| `sensitivity.simulated_confounding` | `implementation_gap` | [X66](../roadmap.md#x66-simulated-confounding-beyond-x47-and-x48) part (e) |
| `sensitivity.missingness`, `tipping_gamma` | `api_redirection` | use `TMLE`. Each tilts an outcome regression |

### Experimental compositions

A composition that no study covers, and whose code computes the requested quantity, runs. The
fit returns its estimates under the experimental label. The old refusal text becomes the reason of
the label, and the roadmap id becomes its evidence. `result.summary()` prints the reason, and
`estimate.support` holds it as data.

The preflight collects each such condition and goes on. A later refusal therefore still fires, and
no condition masks it. The fit reads no learner before the preflight ends.

`tests/unit/test_group_a_experimental_fits.py` checks each site below on four points.

1. The preflight collects the condition before any learner.
2. The fit carries the label, the reason, and the roadmap id.
3. The point lies within three standard errors of the truth, where the complete-case answer does not.
4. A deliberate mutation moves the point.

| contract | settings that run as experimental | roadmap |
| --- | --- | --- |
| arm-indexed, stacked | linear fluctuation, one-step update, weighted fluctuation | X60 |
| arm-indexed, stacked | `att` and `atc`, observation weights, clusters | X60 |
| arm-indexed, stacked | fold evaluation, `split_plan=`, `repeats` above 1, baseline strata | X53 |
| arm-indexed, stacked | `n_bootstrap > 0` | F2 |
| natural course, in sample | a linear fluctuation, one-step targeting, weighted targeting, `n_bootstrap > 0` | X60, X55 |
| natural course, stacked | the same settings, and `repeats` above 1, `split_plan=`, weights, clusters, a continuous outcome | X53, X60, F2 |
| `DRTMLE` with missing outcomes | `weights=` that tilt the target law | X62 |

The targeting and fitting options also run as experimental. The condition is collected in the
same preflight, and `tests/unit/test_option_support.py` checks each label against the fit
without the option.

| option | runs as experimental | keeps the label of the fit | roadmap |
| --- | --- | --- | --- |
| `stopping=` | `"scaled_var"` and `"sample_size"`, on a point fit and on `LTMLE` | `"relative"` | X76 |
| `iterate=`, `step_bound=` | never on their own | under `stopping="relative"` the walk stops only at the solution of the score equation, to `tol` | none |
| `outcome_by_arm=` | `True` | `False` | X76 |
| `patience=` | any value | `None` | X76 |
| `treatment_adjustment=` | a declared subset | the whole adjustment set | X76 |

These requests stay refused, because the code returns a misleading number or has no path:

| refused | why |
| --- | --- |
| a cross-fitted missing-outcome fit outside the two contracts (`tmle.off_contract_axes`) | the interval is misleading |
| `targeting_scheme="fold"` under either contract | the interval is misleading |
| a clustered `CTMLE` fit with missing outcomes | the interval is misleading |
| fold evaluation of the stacked natural-course mean | the fold-evaluated report has no natural-course mean to read |
| a continuous outcome without `q_bounds`, `intermediate=`, a selector-based `CTMLE`, `DRTMLE` on the natural-course target | an invalid declaration, or an estimator with no written result |

### Missing-outcome natural-course contracts

`NaturalCourseMean()`, `PopulationAttributableRisk()` and `PopulationAttributableFraction()` with
`missingness=` and at least one missing outcome have two TMLE contracts. A request that reads the
natural course beside arm targets meets the same contracts.
`CrossFitting(enabled=...)` selects the contract. The table gives each requirement, the
compositions that each contract refuses, and the compositions that it runs with the
[experimental label](#experimental-compositions).

| requirement | ordinary TMLE, `enabled=False` | stacked CV-TMLE, `enabled=True` |
| --- | --- | --- |
| target | `ey_obs`, `par`, `paf`, or `ey_obs` beside arm means, contrasts, ATT and ATC. `estimands="all"` keeps its arm-only list | the same. ATT and ATC run as experimental under the arm-indexed contract |
| estimator | `TMLE`. `CTMLE` and `DRTMLE` are refused. `DRTMLE(guard=())` is the ordinary TMLE, and the message names `TMLE` | the same |
| treatment | two or more arms | two or more arms. With three or more, the scalar `ey_obs` also needs two respondents in each arm |
| outcome | binary, or continuous with declared `q_bounds`. A continuous outcome without `q_bounds` is refused | binary. A continuous outcome runs as experimental |
| outer folds | one fold | package-generated folds with `stratify_by="none"` and `n_folds` of at least 2. One fold and stratified folds are refused. `split_plan=` runs as experimental |
| repeats | `repeats=1`. A larger value fails when `CrossFitting` is constructed, because no split exists to repeat | `repeats=1`. A larger value runs as experimental ([X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants)) |
| targeting | `Targeting(fluctuation="logistic", algorithm="iterative", target_weights=False)`. A linear fluctuation, one-step targeting, and weighted targeting run as experimental ([X60](../roadmap.md#x60-settings-of-the-stacked-and-natural-course-contracts)) | the same, with `targeting_scheme="pooled"` and `fold_evaluation=False`. Fold targeting and fold evaluation are refused |
| inference | influence-curve Wald interval. `n_bootstrap > 0` runs as experimental ([X55](../roadmap.md#x55-bootstrap-extensions) part (a)) | influence-curve Wald interval. `n_bootstrap > 0` runs as experimental ([F2](../roadmap.md#f2-targeted-bootstrap-inference) holds the bootstrap of the stacked fit) |
| rows | fixed `weights=`, `id=` and baseline strata are admitted. `intermediate=` is refused | unweighted iid rows. Weights and clusters run as experimental. `intermediate=` is refused. The scalar `ey_obs` admits baseline strata. PAR, PAF, and a joint fit meet the arm-indexed contract ([X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) part (e)) |
| response support | no added check | at least two respondents and two nonrespondents in the sample, and one of each in every training complement |

Three checks enforce the two contracts. Each check runs at a different time and raises a different
error. A composition that runs as experimental raises nothing. The preflight collects its
condition, and the fit carries the condition as the reason of the label.

| check | error | code | when it runs |
| --- | --- | --- | --- |
| a cross-fitting declaration that is invalid by itself | `MethodConfigurationError`. The engine raises `ValueError` with the same reason | `invalid_declaration`, and `implementation_gap` for a stratified fold policy | when `CrossFitting` or `TMLEMethod` is constructed |
| a composition this target does not support | `CapabilityError` | `implementation_gap` or `invalid_declaration`, by requirement | when the fit starts, before any learner is fitted |
| response support | `DataError` | `invalid_declaration` | after fold generation, before either learner receives data |

The composition check refuses a `CTMLE` fit and a `DRTMLE` fit as an `implementation_gap`. The
message names [X51](../roadmap.md#x51-c-tmle-point-estimates-for-further-targets) for `CTMLE` and
[X50](../roadmap.md#x50-further-dr-tmle-compositions) for `DRTMLE`. On the stacked fit it also
refuses fold targeting and fold evaluation ([X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants)),
and `intermediate=`
([X60](../roadmap.md#x60-settings-of-the-stacked-and-natural-course-contracts)). It refuses a
continuous outcome without `q_bounds` as an `invalid_declaration`. Fold targeting and the
cross-fitted off-contract regimes stay refused because the code returns a misleading interval
there. Fold evaluation stays refused because the fold-evaluated report has no natural-course mean
to read.

Every other row of the table above runs. The condition text of the old refusal becomes the reason of
the label, and the roadmap id becomes its evidence. The check collects the condition and
continues, so a condition never hides a later refusal.

The declaration check is not specific to this target. One function runs its steps in a fixed order,
and it returns the first reason it finds. The declaration and the engine therefore give one reason
for one input. A reason that names the cross-fitting switch names `enabled` from the declaration
and `cross_fit` from the engine.
`tests/unit/test_split_plan.py::TestOneInputEarnsOneReasonAtBothLayers` checks that both layers
give the same reason.

| order | the declaration check refuses | code |
| ---: | --- | --- |
| 1 | `repeats` below 1 | `invalid_declaration` |
| 2 | a `split_plan` that is not a `SplitPlan` | `invalid_declaration` |
| 3 | a plan that records no generator. See [reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) for the two plans that carry the record | `invalid_declaration` |
| 4 | a plan that the declared fold policy cannot use. A plan requires cross-fitting with at least two folds, and its fold and repeat counts must fit the declaration | `invalid_declaration` |
| 5 | a plan together with `n_bootstrap` above 0. `TMLEMethod` runs this step, because it holds both settings | `invalid_declaration` |
| 6 | `repeats` above 1 with cross-fitting disabled | `invalid_declaration` |
| 7 | cross-fitting declared with fewer than two folds | `invalid_declaration` |
| 8 | a fold-stratification policy this package draws no split under. `stratify_by` other than `"none"` is refused whenever the fit draws a split, including selector-based C-TMLE at every setting | `implementation_gap` |

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

A per-arm outcome-adaptive C-TMLE fit reports a point estimate only, so this contract does not
govern it. [The per-arm contract](collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design)
states its own preflight.

A request that reports `ey_obs`, PAR or PAF beside arm targets meets this contract and the
natural-course contract, and its admitted list includes `ey_obs`, `par` and `paf`.
[Stacked CV-TMLE for arm-indexed targets](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets)
defines the estimator, its preflight, and its evidence.

| requirement | admitted | refused |
| --- | --- | --- |
| estimator | `TMLE` or `TMLEMethod` | `CTMLE` or `CollaborativeTMLEMethod`, other than the per-arm `oat` design. `DRTMLE` raises its own refusal at every `guard`, including `guard=()` |
| targets | `ey`, `ey0`, `ey1`, and `ate`. `rr` and `or` for a binary outcome | `att` and `atc` run as experimental, also when the default list or `"all"` includes them. A continuous `rr` or `or` stays refused, as for every fit |
| treatment | two or more arms | none |
| outcome | binary, or continuous with a fixed `q_bounds` equal to the known support | continuous with `q_bounds=None` |
| outer folds | package-generated folds with `stratify_by="none"` and `n_folds` of at least 2 | `split_plan=` runs as experimental. A stratified policy and one fold are refused earlier, by the declaration check |
| repeats and targeting | `repeats=1` and `targeting_scheme="pooled"` | `repeats` above 1 runs as experimental. `targeting_scheme="fold"` is refused |
| fluctuation | `Targeting(fluctuation="logistic", algorithm="iterative", target_weights=False)` and `fold_evaluation=False` | a linear fluctuation, one-step targeting, weighted targeting, and fold evaluation run as experimental |
| rows | unweighted iid rows | observation weights, clusters, and baseline strata run as experimental |
| inference | pointwise influence-curve Wald intervals and the simultaneous band | `n_bootstrap > 0` runs as experimental |
| content | at least the minimum in the [preflight table](point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets) | a sample or a training complement below that minimum. A sample below its minimum is refused with a remedy that does not repartition, because no partition can succeed |

Three checks enforce the contract. The table gives the error of each check and the time at
which it runs.

| check | error | code | when it runs |
| --- | --- | --- | --- |
| a composition outside the contract that stays refused | `CapabilityError` | `implementation_gap` or `invalid_declaration`, by requirement | when the fit starts, before fold generation and before any learner is fitted |
| `DRTMLE` with `delta=` and `cross_fit=True` | `CapabilityError` | `implementation_gap` | when the fit starts, before any learner is fitted |
| minimum content | `DataError` | `invalid_declaration` | after fold generation, before any learner is fitted |

A `DRTMLE` fit with missing outcomes and `cross_fit=True` raises the second `CapabilityError` in
the table at every `guard`, including `guard=()`, and under every fold policy.
`tests/unit/test_drtmle_missing.py` checks the `guard=()` case with zero learner calls.

The composition check runs its steps in a fixed order. A fit that breaks several rules meets the
first refusal. Each message names the missing result and the setting that the contract admits.
A setting message gives both the engine keyword and the public spelling. A step marked
*experimental* does not stop the fit. It adds a condition to the label and the check goes on, so a
later refusal still fires.
`tests/unit/test_arm_indexed_stacked_mar.py` checks each step through the engine and through
`TMLEMethod`, with zero learner calls.

| order | the composition check meets | outcome | code | roadmap item |
| ---: | --- | --- | --- | --- |
| 1 | C-TMLE | refused | `implementation_gap` | [X51](../roadmap.md#x51-c-tmle-point-estimates-for-further-targets) |
| 2 | `fluctuation="linear"` | experimental | `implementation_gap` | [X60](../roadmap.md#x60-settings-of-the-stacked-and-natural-course-contracts) |
| 3 | `algorithm="one_step"` | experimental | `implementation_gap` | X60 |
| 4 | `target_weights=True` | experimental | `implementation_gap` | X60 |
| 5 | `fold_evaluation=True` | experimental | `implementation_gap` | [X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) |
| 6 | `att` or `atc` | experimental | `implementation_gap` | X60 |
| 7 | a continuous outcome with `q_bounds=None` | refused | `invalid_declaration` | none |
| 8 | `split_plan=` | experimental | `implementation_gap` | X53 |
| 9 | `repeats` above 1 | experimental | `implementation_gap` | X53 |
| 10 | `targeting_scheme="fold"` | refused | `implementation_gap` | X53 |
| 11 | observation weights | experimental | `implementation_gap` | X60 |
| 12 | clusters | experimental | `implementation_gap` | X60 |
| 13 | baseline strata | experimental | `implementation_gap` | X53 |
| 14 | `n_bootstrap > 0` | experimental | `inference_gap` | [F2](../roadmap.md#f2-targeted-bootstrap-inference) |

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
it. `result.replayability.unreconstructible` carries the omission codes below as data. The capability
row states the same codes inside its prose `reason`, so read the attribute when you want them as
data. A refused replay carries the reason code `invalid_declaration`. `cleverly` exports
no name for the omission code literals. Compare an omission code against the spelling in the
table. The fitted-bound equality check is different: it can only run when the curve is invoked.

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

| omission code | refusal code | replay boundary |
| --- | --- | --- |
| `longitudinal_replay_learner_unclonable` | `invalid_declaration` | an outcome or pseudo-outcome learner cannot be cloned |
| `longitudinal_replay_random_state_unseeded` | `invalid_declaration` | a replayed learner or nested learner declares an unspecified random state |
| `longitudinal_replay_random_state_non_integer` | `invalid_declaration` | a replayed learner or nested learner declares a non-integer random state |
| `longitudinal_replay_fitted_bound_mismatch` | `invalid_declaration` | invocation-time preflight found that a full replay at the fitted bound differs from the stored fitted artifacts |

The static recipe omissions make the capability row `unavailable`. A fitted-bound mismatch instead
raises a stable `CapabilityError` from that invocation; it is not known when the capability row is
built. Missing `bounds` or the refit opt-in makes a combined report `deferred`, because the caller
can supply either request.

A point-treatment result has two codes of its own. `replayability()` reads each one without a
fit, and every row that needs the false slot reads `unavailable`.

| omission code | refusal code | replay boundary | slots |
| --- | --- | --- | --- |
| `point_replay_function_declaration` | `invalid_declaration` | a stored regime or MSM function lacks an accepted known-function declaration | both slots read false |
| `point_replay_refit_configuration` | `invalid_declaration` | the package refuses the stored estimator configuration before a refit. The slot runs every check that `refit()` runs before a learner, including the design checks of an estimator subclass. A stratified fold policy, a cross-fitted continuous outcome with `q_bounds=None`, and a C-TMLE fit on clustered data are such configurations | `refit_nuisances` reads false. `retarget_cached_nuisances` stays true, because the cached nuisances still retarget |

The second code reads the checks that `refit()` runs before its first learner. These are the fold
policy and four data contracts. `TestTheRefitSlotReadsTheRefitPreflight` in
`tests/unit/test_capability_row_predicates.py` checks each slot against the call it describes.

### Hazard and isotonic survival refusals

[Hazard-based survival targeting](longitudinal-tmle.md#hazard-based-survival-targeting) and
[the isotonic projection](longitudinal-tmle.md#the-isotonic-projection) refuse each composition
below by name. Each refusal is raised before any learner. `tests/unit/test_hazard_refusals.py` pins
the type and the message of each one, with learners that raise if they are fitted. `tests/unit/test_hazard_targeting_rules.py` pins H13.

| id | trigger | when | type and code | message |
| --- | --- | --- | --- | --- |
| H1 | `survival_model="hazard"` on a design with a treatment at more than one node | `fit` | `CapabilityError`, `implementation_gap` | "survival_model='hazard' is built for one baseline treatment held over every node, and this design declares a treatment at {k} nodes. Hazard targeting for a treatment at several nodes is not built. X70 in docs/roadmap.md tracks this. LTMLE with survival_model='iterated_mean' estimates this curve, and isotonic=True makes it monotone. If one baseline treatment is held over every node, declare treatment='A'." |
| H2 | `survival_model="hazard"` with a covariate measured after baseline (`time_varying=`) | `fit` | `CapabilityError`, `implementation_gap` | "survival_model='hazard' models the event and censoring hazards given the baseline treatment and the baseline covariates. This fit declares covariates measured after baseline ({names}), and hazard targeting with such covariates is not built: it needs a model and a fluctuation of each later covariate. X70 in docs/roadmap.md tracks this. LTMLE with survival_model='iterated_mean' adjusts for them, and isotonic=True makes its curve monotone." |
| H3 | `survival_model="hazard"` on an end-of-study outcome | `fit` | `CapabilityError`, `invalid_declaration` | "survival_model='hazard' needs an event indicator at every node, and this fit declares one end-of-study outcome. A hazard is the chance of the event at a node among the units still at risk, so an outcome measured once has none." |
| H3 | `isotonic=True` on an end-of-study outcome | `fit` | `CapabilityError`, `invalid_declaration` | "isotonic=True makes a curve over the horizons monotone, and an end-of-study fit reports one horizon." |
| H4 | `targeting="one_step"` with `survival_model="iterated_mean"` | construction | `ValueError` | "targeting='one_step' requires survival_model='hazard'. The one-step walk moves one set of hazards toward the score equations of every horizon together (Cai and van der Laan 2020). A one-step update of the iterated-mean recursion is not built. Set survival_model='hazard' on a survival design with one baseline treatment, or keep targeting='iterative' and set isotonic=True for a monotone curve" |
| H5 | `msm=` with `survival_model="hazard"` or `isotonic=True` | construction | `CapabilityError`, `implementation_gap` | "msm= with {option} is not built. A working model reports coefficients over regimen and horizon cells. X70 in docs/roadmap.md tracks this. Fit msm= with survival_model='iterated_mean' and isotonic=False, or report the curve and take contrasts." |
| H6 | `pseudo_learner=` with `survival_model="hazard"` | construction | `ValueError` | "pseudo_learner= is not used with survival_model='hazard'. A hazard fit regresses the event indicator at each node and fits no pseudo-outcome. Pass the hazard learner as outcome_learner=" |
| H7 | a modified treatment policy regimen with `survival_model="hazard"` | `fit` | `CapabilityError`, `implementation_gap` | "regimen {label!r} is a modified treatment policy, and survival_model='hazard' does not support one. Its plug-in evaluates the event and censoring hazards at the shifted dose, and that construction is not built. X70 in docs/roadmap.md tracks this. Use survival_model='iterated_mean', which supports a modified treatment policy at baseline." |
| H8 | an unknown `targeting`, `survival_model` or `isotonic` value, or a `step_size` that is not positive and finite | construction | `ValueError` | "targeting must be 'iterative' or 'one_step'; got {v!r}", "survival_model must be 'iterated_mean' or 'hazard'; got {v!r}", "isotonic must be True or False; got {v!r}", and "step_size must be a positive finite number; got {v!r}". A `step_size` under `targeting="iterative"` is accepted and unused |
| H9 | `q_bounds=` with `survival_model="hazard"` | `fit` | `ValueError` | the binary-outcome refusal, "q_bounds does not apply to a binary outcome" |
| H10 | `survival_model="hazard"` or `isotonic=True` on a point-treatment functional | method configuration | `MethodConfigurationError` | "survival_model= and isotonic= apply to a survival curve over time nodes, and a point-treatment functional has no curve. Remove them, or declare a longitudinal or TimeToEvent design" |
| H11 | `survival_model="hazard"` with a continuous baseline treatment, or a vector with a dose | `fit` | `CapabilityError`, `implementation_gap` | "survival_model='hazard' needs a categorical baseline treatment, and treatment {name!r} is continuous. The hazard plug-in would need the treatment density at every dose, and that construction is not built. X70 in docs/roadmap.md tracks this. Use survival_model='iterated_mean'." A vector with a dose reads "is a vector with a dose" in place of "is continuous" |
| H12 | a non-default `survival_model`, `targeting`, `step_size` or `isotonic` on `SDRMethod`, `IPSIMethod`, `GComputationMethod` or `IPWMethod` | method configuration | `MethodConfigurationError` | "survival_model=, a hazard targeting= algorithm, step_size= and isotonic= are settings of the targeted fit, method='tmle'. The {method} estimator fits no hazard, and no registered study measures its projected curve. Fit method='tmle', or remove them". The `SDR`, `IPSI`, `SequentialGComputation` and `SequentialIPW` engines do not take these keywords, and raise `TypeError` |
| H13 | `step_bound=` with `targeting="one_step"` (R-c4) | construction | `MethodConfigurationError`, `api_redirection` | "targeting='one_step' with survival_model='hazard' already walks the universal least-favorable submodel in steps of step_size. step_bound= bounds the steps of targeting='iterative'." |

The fit checks H3 first, then H1, H2, H11 and H7. So a continuous treatment meets H11 before a
modified treatment policy meets H7.

H1 refuses an extension that qualifies and is not built. A static arm or a known rule at several
nodes, without a covariate measured after baseline, reduces to an indicator. That indicator is the
composite "uncensored and on the regimen through node $k$". The hazard construction then applies
to the composite. This package does not build that reduction.

H2 needs a model and a fluctuation
of each later covariate. Stitelman, De Gruttola and van der Laan (2012) and Rytgaard, Gerds and van
der Laan (2022) treat such covariates. Neither source was read for this contract.

[X70](../roadmap.md#x70-hazard-targeting-beyond-a-held-baseline-treatment) tracks H1, H2, H5, H7 and H11. Each message names the supported route. The roadmap carries no row for H4.

### A different question

These are well-posed parameters, but the selected estimator does not target them. No setting turns
one into the other. The `code` column holds `different_estimand` for most rows. Another code marks a
row where the raise site names a different remedy.

| refused | code | the question it would answer instead |
| --- | --- | --- |
| `eliminate=` on a competing-risks fit | `different_estimand` | the incidence of a cause if the competing events were *removed*. That intervenes on them rather than conditioning on the history. It needs a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions for the competing event. What is reported instead is the incidence with the competing causes left alone |
| `CTMLE` on data that declares its treatment mechanism | `api_redirection` | C-TMLE selects or fits a treatment mechanism, and the declaration leaves nothing to select or fit. The question does not change. Fit `TMLE` or `DRTMLE` on the declared data |
| a declared known treatment mechanism beside `treatment_delta=` | `invalid_declaration` | a missing treatment needs $P(A=a\mid\Delta_A=1,W)$, the mechanism among the recorded rows. It equals the design mechanism only when recording is independent of treatment given $W$, which the fit cannot check |
| `ratio_trim=` on a point-treatment fit | `api_redirection` | a trim of the per-node density ratios of a longitudinal fit, as `lmtp`'s `.trim` does. A point-treatment fit bounds its mechanism with `g_bounds=`. Declare the point treatment as a one-node `LongitudinalTreatment` to trim its ratio. `TMLEMethod.estimator_kwargs` raises `MethodConfigurationError` before any learner |
| `intermediate=` on `LTMLE` | `different_estimand` | a controlled direct effect fixes a mediator at one time point. Over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| `ey1` and `ey_regime` from one fit; `msm=` with `interventions=` or `policies=`; an item of another intervention kind in a typed estimand, such as an `Incremental` in `RegimeContrast` | `implementation_gap` for two axis keywords, for `msm=` beside another axis, and for a typed set that holds two kinds. The message names [X57](../roadmap.md#x57-joint-intervention-axes-and-tilt-compositions). `api_redirection` for an estimand of another axis and for a typed set that holds only the other kind | each keyword declares what "counterfactual" means for the fit, or how the counterfactuals are summarised. One fluctuation solves one set of score equations, so a fit reporting parameters from two axes would put two of them under one heading. `CausalStudy.identify` refuses the typed request with `CapabilityError`, and it names the typed estimand for the item |
| the per-arm propensity table on a continuous fit | `different_estimand` | a per-arm table has no rows when there are no arms. `diagnostics.support()` is not itself refused. On a fit that declared `policies=` it dispatches to the question that does apply, which is whether the density *ratio* stays bounded |
| `res.sensitivity`, `res.diagnostics` and `res.validate()` on an `LTMLE` result | `implementation_gap` for a sensitivity operation. `api_redirection` for the score equations of `SDR` and `IPSI` | each is part of the shared result contract. Stagewise support, scores, nuisance loss, and the descriptive full-recursion truncation grid are supported. Sensitivity operations without a longitudinal derivation report `unavailable`. `res.save()` is supported, and [persistence and replayability](../user-guide/results-assessment.md#persistence-and-replayability) states its contract |
| `ratio(..., view="survival")` on an end-of-study fit, on a repeated-outcome fit, or on a fit with two or more causes; `rmst`, `rmtl` and `incidence_total()` on an end-of-study, repeated-outcome or working-model fit; `curve(scale="risk")` or `curve(scale="survival")` on a repeated-outcome fit; `ratio` on a working-model fit | `different_estimand` | an end-of-study fit reports means and no survival curve. A repeated outcome can return to zero, so its mean at a horizon is not a cumulative risk. One minus one cause's incidence is not all-cause survival. A coefficient is not a regimen mean. [Functionals of a fitted result](longitudinal-tmle.md#functionals-of-a-fitted-result) gives each message |
| a plan that changes the treatment after baseline on a held design (`treatment="A"` or `TimeToEvent`) | `different_estimand` | a time-varying treatment plan. Declare one treatment column per node on the wide layout. The held design raises `CapabilityError` before any learner, and a plan that repeats one label is read as that label |
| `logistic_plugin` on a fit that is not a selector `CTMLE` fit, or on an `oat`, multi-arm, cross-fitted, weighted, missing-outcome or `intermediate=` fit | `different_estimand` | R `ctmle` defines the logistic-estimation term for a binary treatment whose propensity is fitted once on all rows. [Collaborative TMLE](collaborative-tmle.md) gives the term |
| `logistic_plugin` on a `lasso` or `learners` fit | `different_estimand` | the term is defined for one logistic model on a covariate set. A `lasso` fit fits an L1-penalized path. R `ctmleGlmnet` type 1 reports the plain $D^*$ variance, and the square of `plugin_std_error` equals it. `tests/unit/test_ctmle_sequence_witness.py` checks the equality at a relative `1e-6`. A `learners` fit fits declared candidates |
| `logistic_plugin` on a `derivative` fit | `different_estimand` | the fit adds derivative columns to the fluctuation. R `ctmleGlmnet(ctmletype=3)` reports the plain $D^*$ variance, which `std_error` gives up to $n/(n-1)$ |
| `strategy="derivative"` with `targeting="one_step"` or `target_weights=True` | `api_redirection` | C-TMLE0 is R's single logistic fit of the $2K$ clever covariates. The one-step path targets one parameter along another path. `target_weights=True` moves a row's covariate into one weight, and each treated row loads two columns. [C-TMLE0](collaborative-tmle.md#c-tmle0-the-derivative-covariate) |
| `strategy="derivative"` with baseline strata | `implementation_gap` | No source defines a stratified derivative update. Fit the ordinary TMLE for stratum estimates. [X62](../roadmap.md#x62-other-dr-tmle-and-c-tmle-compositions) part (e) tracks the gap |
| `strategy="derivative"` with a zero or collinear derivative column, or a $2K$ solve that does not converge | none | the partner equals the mechanism after `g_bounds` on every row of an arm, or differs from it by a constant factor, so that arm's derivative equation is empty or repeats $h_a$. The message names the arm and the remedy |
| `strategy="derivative"` with `screen_treatment=True` | `api_redirection` | the lasso pair, the declared learners and the arrays each fit the mechanism and its partner on every covariate. A screen would apply to neither. Fit `TMLE` with a screened mechanism |
| a `strategy="derivative"` pair that is not two entries, a `PropensityArrays` entry, or an array entry with `cross_fit=True`, `n_bootstrap` above 0, or a refit | `invalid_declaration` | the pair is `(mechanism, partner)`. An array holds one full-sample value for each analysis row, so an outer fold, a resample, or a changed covariate breaks it |
| a single propensity array in `treatment_candidates=` | `api_redirection` | an array fitted on every row would score a mechanism that saw the validation rows. Pass `PropensityArrays(full=..., cv=...)` with the split of `cv` as `selection_split=`. [Candidate sequences](collaborative-tmle.md#candidate-sequences) |
| `PropensityArrays` without `selection_split=`, with `cross_fit=True`, or under a refit | `invalid_declaration` | `cv` was fitted under one declared split on the analysis sample. A random split, an outer fold, or an added covariate breaks that match. The message names the way out |
| a `PropensityArrays` row that does not sum to 1 within `1e-9` | `invalid_declaration` | the row is not a distribution over the arms, so the clever covariate weights the arms wrongly. Pass `predict_proba` output in arm order, or divide each row by its sum |
| `lambdas=` with fewer than two penalties | `invalid_declaration` | one penalty gives one candidate, and the fit selects nothing. Pass two or more penalties, or `lambdas=None` for the default grid |
| `selection_split=` with `repeats` above 1, `n_bootstrap` above 0, a `selection_folds=` that differs from the split's fold count, or `strategy="oat"` | `invalid_declaration` | a declared split is fixed on the analysis rows. A repeat redraws it, and a resample changes the rows. `selection_folds=` records the fold count, so it must equal the split's count. `oat` draws no split |
| a non-empty `assess(arguments=...)` block for `score_equations` | `api_redirection` | the validation battery owns that name and runs it argument-free. The battery presents one row per name, and it presents the validation row. A caller's tolerance would be computed and then hidden, so a check that failed at that tolerance would never reach `attention`. Call `res.diagnostics.run_all(arguments=...)`, or the operation itself. The battery also owns `support` and `nuisance_models`, which accept no argument, so an argument for either is a `TypeError` from the signature |

### Wrong by construction

These are worth reading even if you never reach for the keyword. Most are mistakes that are easy to
make by hand, in any framework, and none announces itself. They can produce a number for the wrong
target or an interval whose stated conditions are not met. The `code` column names the code of the
raise site. A row marked `none` has no raise site. The package cannot detect the mistake, or its
API does not offer the construction.

| refused | code | what goes wrong if it is done anyway |
| --- | --- | --- |
| `ratio(..., kind="or")` when the outcome family is not binary | `api_redirection` | the odds of a mean outside $(0, 1)$ is not defined. The refusal applies the rule of the registered `or` target, `TARGETS["or"].requires_family`, and names `kind="rr"` |
| a population-mechanism odds tilt supplied as a `Stochastic` density | `implementation_gap` for `density_kind="estimated"`. `invalid_declaration` for `None` | the population-law target depends on `P` through `g*`. Its curve needs a derivative that the regime curve lacks. On one exact law, the reported standard error is 0.62 of the exact one at `delta = 2` (`tests/unit/test_stochastic_regime_densities.py`). A realized learned density instead defines a data-adaptive target; its inference needs conditions this API does not check. `Stochastic` requires `density_kind="known"` and refuses `None` or `"estimated"` at construction and fit. A false `"known"` declaration still fits because code cannot inspect a closure. [RM25](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm25-declared-stochastic-regime-densities) records the witness and both targets |
| a population-law incremental intervention built by hand as a `Stochastic` regime | `implementation_gap` for `density_kind="estimated"`. `invalid_declaration` for `None` | the missing term has variance `Var(delta * (Qbar(1,W) - Qbar(0,W)) / D^2 * (A - g))` on two levels and `Var(h(A) * {Qbar(A,W) - m(W)})` with `h(a) = delta_a / D` on K levels. For that target, omitting the term understates variance wherever this variance is positive. `Stochastic` refuses the tilt when `density_kind` is `None` or `"estimated"`. `TMLE(incremental=)` fits the population odds tilt and includes the term |
| a user-written `Intervention` class whose density depends on the population mechanism | `implementation_gap` for `"estimated"`. `invalid_declaration` for `None` and for a missing attribute | it can omit the same derivative while reporting an interval for the population-law target. On the RM25 witness law, a class that computes the sample-mechanism odds tilt reports 0.62 of that target's exact standard error (`tests/unit/test_rule_and_intervention_declarations.py`). A realized learned density has a different inference contract. The class must carry `density_kind = "known"`. `TMLE` refuses `None`, a missing attribute, and `"estimated"` before any learner. `RegimeSet.evaluate` refuses them before `density` runs. A false `"known"` declaration still fits because code cannot inspect what `density` computes. [RM28](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm28-declared-densities-of-user-written-interventions) records the witness |
| a `Rule` or `DynamicRegimen` learned from the analysis sample | `implementation_gap` for `"estimated"`. The message names [X11](../roadmap.md#x11-learned-policy-follow-ups). `invalid_declaration` for `None` and for an inline callable | the fit treats its rule as fixed. A learned policy can instead target a realized policy value or a population-indexed policy. Those targets need distinct inference conditions; an optimal rule can also be nonregular at ties. For a threshold at the sample mean of a continuous covariate, the curve omits a derivative through that mean. On one exact law, the reported standard error is 0.73 of the exact one at one node (`tests/unit/test_rule_and_intervention_declarations.py`). It is 0.77 for a two-node regimen (`tests/unit/test_regimen_rule_declarations.py`). `Rule` and `DynamicRegimen` require `rule_kind="known"` and refuse `None` or `"estimated"` at construction and fit. A plan of labels alone needs no declaration. `LTMLE` refuses a callable written inline in `regimens=`, because it carries no declaration. A false `"known"` declaration still fits because code cannot inspect a closure. For a point treatment, `LearnedRuleValue` estimates the average value of rules learned inside each training fold ([learned rules](point-treatment-tmle.md#learned-rules)). A learned longitudinal regimen has no supported path ([X11](../roadmap.md#x11-learned-policy-follow-ups)). [RM28](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm28-declared-densities-of-user-written-interventions) records the witnesses |
| a shift's inference taken from the regime inducing the same density | none | the means and the clever covariates agree entry for entry. The curves do not. The gap is `Var(Qbar(d(A,W),W) - E[Qbar(d(A,W),W) | W])`. Too small, always |
| a shift fit run on the complete cases when outcomes are missing | none | it is an ordinary shift fit on a *different* joint law of `(A, W)`, so it converges to a different number, and nothing in its own output says so. Measured on the dose fixture at 0.17, four standard errors, with a mechanism whose slopes are mild. `delta=` is what corrects it |
| a missingness or intermediate mechanism read at the observed dose rather than the assigned one | none | the fluctuation updates `Qbar` as a function of the dose, so `Qbar*(d(A,W),W)` is the update evaluated where the policy sends the unit. Silent wherever the mechanism does not depend on the dose, and invisible to a Gateaux check on an exact law |
| a "stabilised" MSM weighting `h` by the estimated mechanism | `implementation_gap` for `weights_kind="estimated"`. `invalid_declaration` for a callable with no declaration and for an array | the same argument once more. `h` becomes a functional of `P`, and a term goes missing from the influence curve. `MSM` refuses `weights_kind="estimated"` and a `weights=` callable with no declaration. [MSM projections](msm-projections.md#variations) gives the declaration |
| an MSM `design` that reads a sample statistic, such as a covariate centred at its sample mean | `implementation_gap` for `design_kind="estimated"`. `invalid_declaration` for a callable with no declaration | for the population-law target centred at $E_P[W]$, the projection depends on $P$ through the design. Its reported curve omits that derivative. On one exact law, the reported intercept standard error is 0.893 of the exact one (`tests/unit/test_msm_design_declaration.py`). An interval conditional on the centre learned from the same rows also needs separate validation. `MSM` refuses `design_kind="estimated"` and an undeclared callable. A design built by `MSM.linear` needs no declaration. Code cannot detect a false `"known"` declaration. [MSM projections](msm-projections.md#variations) gives the declaration, and [RM27](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm27-declared-msm-design-functions) records the witness |
| `g_bounds=` or `truncation_curve()` on an `incremental=` fit | `different_estimand` | `g` is *inside* the estimand, so truncating it moves the parameter rather than regularising a denominator. The result is a number for a parameter nobody declared. `truncation_curve()` refuses the treatment axis with `CapabilityError`, and the omitted `mechanism` selects that axis. The module call and the facade call raise the same sentence. The capability row reads `unavailable`. On a fit with missing outcomes it reads `deferred` on `mechanism`, because `mechanism=True` runs |
| a `cap=` fitted from the data on a shift | none | the estimand becomes data-dependent. The interval conditions on an estimated boundary, and every bootstrap replicate targets a slightly different policy |
| `CTMLE` on an `incremental=` or `policies=` fit | `implementation_gap` | each candidate `ghat` defines a different estimand, so the cross-validated search selects between *estimands* rather than between estimators. The policy mean itself is well posed, and a plain `TMLE` fits it. [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) keeps the collaborative route |
| splitting a cluster across folds to buy more of them | `invalid_declaration` | the out-of-fold predictions stop being independent of the rows they are used on, and the standard error shrinks in exactly the direction `id=` was passed to prevent |
| a supplied `SplitPlan` whose labels no recorded draw produces | `invalid_declaration` | the labels could have been chosen by reading the outcome, the treatment, or a covariate, which is the leak cross-fitting exists to prevent. The fit draws each repeat again from the plan's record and compares it label for label. [Reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) states what a plan records |
| reusing a supplied `SplitPlan` on rows it was not realized on | `invalid_declaration` | a fold label is a row position. A reordering, a replaced column, or an added covariate leaves every label pointing at a different unit, while the row count still agrees. The out-of-fold guarantee is gone, and nothing in the output says so. [Reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) states how a plan binds to its rows |
| joint covariance, post-fit contrasts, or simultaneous bands after `repeats=` on a point-treatment fit | `inference_gap` for the covariance and the bands. `api_redirection` for `contrast()` and `ratio()` | coordinatewise medians do not preserve linear identities among estimates. A central-draw curve also does not represent the split-adjusted median estimator that a multiplier band needs. A longitudinal fit forms each of them per draw, as [repeated fold draws](longitudinal-tmle.md#repeated-fold-draws) states |
| a cross-validated variance for the curve a repeated fit retains | none | at equal fold sizes it collapses to the pooled uncentred second moment for *every* partition, so the partition carries no information. That rule was rejected, and the split-adjusted median variance replaced it |
| a one-shot non-identity-link MSM | none | the derivative of the inverse link depends on the coefficient, so a single pass reports a standard error for an equation it did not solve. The link is supported. What is refused is skipping the alternation it needs |
| a censoring time between two grid times, read as censored at a grid time | `invalid_declaration` | the censoring cannot be ordered against the events of its interval, so either reading biases the risk. On one interval with exponential times the two readings give 0.280 and 0.236 against a truth of 0.259 (`test_both_off_grid_censoring_conventions_are_biased`). `LongitudinalData.from_time_to_event` raises `DataError` before any learner and names the visit grid as the remedy |
| an event value on a row whose censoring is 0, on a long panel or on the wide layout | `invalid_declaration` | the order at a node is $A_t, C_t, Y_t$, so the event is unobserved once the unit is censored. Reading past it drops an event that was meant as "had it and left". `LongitudinalData.from_frame` used to drop the value without a message. Both layouts raise `DataError`. Record an event as censoring 1 and event 1, and a loss after an event-free interval as censoring 0 with a missing event (`test_refuses_an_event_on_a_censored_row`, `test_the_wide_path_refuses_an_event_on_a_censored_row`) |
| a long-panel `baseline`, `cluster` or `weights` column that varies within a unit or is first recorded after the first row; an `outcome` that varies within a unit | `invalid_declaration` | a unit-level column is measured before the first decision. A later value read as `W` lets a rule at an earlier node read the future. `LongitudinalData.from_long_panel` raises `DataError` and names the unit |
| a long-panel covariate that equals the outcome, or the row's event, on every row where it is recorded; one column in two roles | `invalid_declaration` | a rule or a history frame could read the outcome under another name. `from_long_panel` raises `DataError` and names both columns. A constant covariate is not mistaken for a copy of a constant outcome |
| a long panel whose time labels are not numbers, dates or durations, with no `times=` | `invalid_declaration` | the sort order of such labels need not be their time order, and the node order is the time order. `from_long_panel` raises `DataError` and asks for `times=` in time order |
| a long panel with a duplicated `(unit, time)` row, late entry, a gap in a unit's rows, rows after a unit left, or rows that stop with no censoring 0 and no event | `invalid_declaration` | intermittent observation is not censoring, and a unit that left does not return. `from_long_panel` raises `DataError` and names the unit and the time. `times=` names every decision time and cannot coarsen a grid |
| a long-panel time-varying column with no value at a node after its first recorded node | `invalid_declaration` | leaving the column out there removes a confounder from that node's adjustment set. A leading run of missing values is not refused. The column starts at its first recorded node, and `PanelLayout.omitted` records each omission |
| a long-panel treatment that never changes within a unit | `invalid_declaration` | one baseline decision held over every node, not a decision at each time. Build the wide frame and declare `treatment="A"` |
| frequency (count) weights | `invalid_declaration` | they assert a sample size the variance does not use. Expand the rows instead, which says the same thing where every part of the fit can see it |
| an `LTMLE` outcome missing for a reason other than censoring | `api_redirection` | its probability of being observed is silently taken to be one. Encode it as a final censoring column so that it is estimated and enters the cumulative product |
| a binary-only target on a multi-arm fit | `invalid_declaration` | it would report a contrast of arms `0` and `1` out of five, under the name of a parameter about all of them. Targets declare `requires_binary_treatment` for this |
| `MSM.linear` on non-numeric arm labels | `invalid_declaration` | a model linear in the arm reads it as a dose to interpolate between, and the fallback coding is the sort order. That is a dose scale nobody chose |

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

Three method entries carry their own detailed refusal tables, each with a `code` column:
[marginal structural models](msm-projections.md#variations),
[longitudinal TMLE](longitudinal-tmle.md#variations), and
[the simulated common-cause surface](validation-methods.md#simulated-common-cause-stress-surface).
The column holds the reason code of the raise site, as the `code` column of the tables above does.

The simulated common-cause surface refuses seven compositions for the whole fit. They are a
longitudinal fit, multi-arm treatment, a missing outcome, a missing treatment, an intermediate
variable, estimated observation weights, and a clustered fit. Each one is an `implementation_gap`
(`_FIT_WIDE_RULES` in `src/cleverly/sensitivity/_simulated_confounding_request.py`).
[X47](../roadmap.md#x47-simulated-confounding-laws-for-point-treatment-fits) and
[X48](../roadmap.md#x48-longitudinal-simulated-confounding-replay-with-a-time-fixed-latent) track the
laws a caller can declare. The
[future investigations grid](../roadmap.md#future-investigations) tracks each part that waits on
published theory, against the result a paper must supply.

The surface accepts fixed probability weights under ordinary TMLE, under binary complete-outcome
collaborative TMLE, and under binary complete-outcome DR-TMLE. Read
[the surface's own refusal table](validation-methods.md#simulated-common-cause-stress-surface) for
every composition it refuses and the code of each one.

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
as two". The informative entries are those that do not. The `status` column holds `supported`,
a reason code from [How to read a refusal](#how-to-read-a-refusal) above, or `waiting on published
theory` from the [roadmap's eligibility rules](../roadmap.md#eligibility).

| surface | status | why |
| --- | --- | --- |
| `TMLE`: `ey` per arm, `ate` / `att` / `atc` per non-reference arm, regimes, MSMs over arms | supported | one counterfactual mean per arm and one contrast per non-reference arm. The [oracle-law gate](validation-methods.md#the-oracle-law-gate) states that a target meant for more than two arms needs a branch on the three-armed law |
| `DRTMLE`: univariate and bivariate reductions | supported | [armwise one-vs-rest](dr-tmle/index.md#variations), with each reduction and correction indexed by a free level. The cited theorem is binary, so this is an implementation-backed armwise extension rather than a claim about that theorem's literal scope |
| `DRTMLE` with `delta=` in a randomized trial | supported | Díaz and van der Laan's missing-outcome construction, applied to each arm indicator and stacked. Each arm's treatment, observation and outcome tilts are its own, so no fluctuation parameter is shared across arms. See [the multi-arm contract](dr-tmle/theorem.md#more-than-two-arms) |
| `TMLE` and `DRTMLE` with an observational missing outcome or a missing treatment | supported | the composite indicator, applied to each arm and stacked. Each arm's column of the composite mechanism is tilted alone, two arms included. See [the composite contract](dr-tmle/theorem.md#observational-missing-data-the-composite-indicator) |
| `CTMLE`: selectors, `strategy="oat"` and `strategy="derivative"` | supported | the selectors fit one shared `n x K` categorical mechanism, selected against one nonredundant vector. See the [standing decision](../architecture-invariants.md#targets-interventions-and-variants). `strategy="oat"` fits one binary mechanism per arm by default, and the `"shared"` design fits one categorical mechanism. See [the per-arm design](collaborative-tmle.md#the-per-arm-outcome-adaptive-design). `strategy="derivative"` fits one multinomial lasso and its partner, and one derivative column per arm |
| `LTMLE`: categorical nodes, static and dynamic regimens | supported | [treatment over time](longitudinal-tmle.md#the-algorithm-as-implemented). Each node owns its level set, and the clever covariate selects the assigned label's probability |
| positivity, omitted-variable, E-value and MNAR sensitivity | supported | each is one parameter per contrast, and each reads its arms from the parameter's structured index rather than assuming two |
| simulated common-cause sensitivity | supported for binary and continuous treatment; named theory stops cover the remaining fit-wide gaps | The binary surface accepts marginal and baseline-stratum arm means, ATE, and ratios under ordinary TMLE and C-TMLE. Ordinary TMLE also accepts ATT and ATC with perturbed group membership. Ordinary TMLE alone accepts marginal and baseline-stratum PAR, PAF, fixed-regime parameters, and identity-link MSM coefficients. It also accepts marginal incremental targets and nonlinear MSM coefficients. Complete-outcome DR-TMLE supports marginal arm means, ATE, and ratios only. The continuous surface accepts marginal and baseline-stratum modified-policy means and contrasts under ordinary TMLE. A risk-ratio tilt fit is refused, because the replay declares no reading of its rule on a perturbed draw ([X66](../roadmap.md#x66-simulated-confounding-beyond-x47-and-x48) part (d)). Continuous MSMs require marginal fits. Every listed row accepts fixed probability weights. Longitudinal, multi-arm, missing-outcome, intermediate, estimated-weight, and clustered fits report unavailable before work begins. Each of those six waits on published theory, and so does logical categorical calibration. `NaturalCourseMean`, the zero-delta policy mean, and the multiplier-one incremental mean remain refused. See the [population contract and refusal table](validation-methods.md#simulated-common-cause-stress-surface) |
| `ey1` / `ey0` on a multi-arm fit | `invalid_declaration` | they *name* one of exactly two arms, so on five arms they would report a contrast of arms `0` and `1` under the name of a parameter about all of them. Declared by `requires_binary_treatment`. The multi-arm path reports per-arm `ey` instead. The incremental estimands declare no such requirement, because a categorical tilt names a factor for every level |
| a scalar `Incremental(delta)`, as `incremental=` or in an `IPSI` plan, on a treatment or tilt node with other than two levels (R1) | `invalid_declaration` | a scalar odds multiplier names two arms. The refusal runs before any learner, names the treatment or node and its levels, and shows the mapping form: "a scalar odds multiplier names two arms; node 'A1' has 3 levels (['high', 'low', 'standard']). Declare one factor per level, for example Incremental({'high': 2.0}), which multiplies the odds of 'high' against every unnamed level." Declare `Incremental({label: factor})` instead. A level that the mapping does not name keeps the factor 1. See [categorical treatment](point-treatment-tmle.md#incremental-propensity-score-interventions) |
| an `Incremental` mapping with a key that is not a level of the treatment or of a tilt node (R2) | `invalid_declaration` | the refusal runs before any learner and names the key, the treatment or node, and its levels. A key is a label as the data reports it, and a position is not a label. One `Incremental` in a plan meets each node in turn, so a key that is a level of node 1 and not of node 2 is refused at node 2 |
| `LTMLE`: known stochastic categorical policies at a node | supported | [known stochastic policies](longitudinal-tmle.md#known-stochastic-policies). A node holds a policy density fixed before the fit. The recursion carries the policy mean of the per-level predictions, and the clever covariate is the ratio of the policy to the bounded mechanism. Exact-law tests and the [registered study](method-evidence/stochastic-categorical-longitudinal-tmle.md) cover it |
| `LTMLE`: modified treatment policies at a continuous, categorical or vector node | supported | [modified treatment policies at a node](longitudinal-tmle.md#modified-treatment-policies-at-a-node). A plan node holds a `Shift`, `Scale`, `Piecewise`, `ModifiedPolicy` or `RiskRatioTilt`. A continuous node reads a pooled-hazard density or the classifier ratio. Exact-law tests cover it, and the registered study `longitudinal-mtp` validates it |
| `LTMLE` and `SDR`: modified treatment policies at a mixed vector node | supported | [vector nodes with a continuous component](longitudinal-tmle.md#vector-nodes-with-a-continuous-component). A mixed node holds a `JointPolicy`, one policy for each component. The ratio is a chain-rule density or the classifier ratio. Exact-law tests cover it, and the registered study [`vector-mtp`](method-evidence/vector-modified-treatment-policies.md) validates it. [Vector treatments](#vector-treatments) lists the refusals |
| `TMLE`: modified treatment policies on a point vector treatment | supported | [vector treatments](point-treatment-tmle.md#vector-treatments). A vector of categorical columns is one categorical treatment over its tuples. A vector with a dose reports `ey_policy` and `ate_policy`. The registered study `vector-mtp` validates both |

A source could close one entry as it stands. It is the multi-arm part of the simulated
common-cause row.

The `DRTMLE` row needs no new source either. It needs the implementation that its roadmap row
holds.

## Vector treatments

A vector treatment holds several treatment columns. A vector of categorical columns is one
categorical treatment over the observed tuples. A vector with a continuous component has no
arms. `TMLE(policies=[JointPolicy(...)])` fits it at a point treatment, and `LTMLE` and `SDR` fit
it at a node. The first table says what ships. The second table lists what a vector refuses. Each
refusal arrives before the nuisance fit.

| surface | status | where |
| --- | --- | --- |
| point treatment, all columns categorical | supported. Every estimand of a categorical treatment runs on the tuple levels | [vector treatments](point-treatment-tmle.md#vector-treatments) |
| point treatment, with a dose | supported for `ey_policy` and `ate_policy`, on `ratio="density"` and `ratio="classifier"`. A dose component may have a data-tuned cap on either `target=`, with the curve terms of the longitudinal node | [vector treatments](point-treatment-tmle.md#vector-treatments), `tests/unit/test_point_tuned_components.py` |
| longitudinal node, all columns categorical | supported. The node is one categorical node over the tuples | [modified treatment policies at a node](longitudinal-tmle.md#modified-treatment-policies-at-a-node) |
| longitudinal node, with a dose | supported on both ratio routes, for `LTMLE` and `SDR`. A component may declare atoms, `dose_atoms={name: (...)}`, and each dose component may have a data-tuned cap, `Shift(cap=DoseQuantile(...))` or `Scale(cap=SupportBoundary(...))`, on either `target=`. The curve adds one derivative term for each tuned component in each branch of a `randomizer`, weighted by the branch probability | [vector nodes with a continuous component](longitudinal-tmle.md#vector-nodes-with-a-continuous-component), `tests/unit/test_mtp_component_atoms.py`, `tests/unit/test_mtp_tuned_components.py`, `tests/unit/test_mtp_joint_tuning.py` |

Each row of the next table gives the theoretical reason for its refusal.

| refused | raised | reason | test |
| --- | --- | --- | --- |
| a `RiskRatioTilt` on a dose or on a categorical component with other than the levels 0 and 1 | `DataError` | the tilt is the `ipsi` rule of `lmtp`. It keeps a treated unit treated with probability $\delta$ and otherwise sets it to 0. A dose or a three-level component has no treated and untreated label, so the two-point randomizer is not defined. A binary component coded 0 and 1 is supported | `tests/unit/test_vector_component_randomizer.py` |
| an odds tilt, `Incremental`, at a mixed node, and `IPSI` at a mixed node | `CapabilityError` | an odds tilt multiplies the odds of one categorical node (Kennedy 2019, Equation (1)). The joint density of a node with a dose has no odds to multiply. `IPSI` estimates no density ratio, so it has no estimator for a policy node | `tests/unit/test_longitudinal_mixed_node.py` |
| a dose map that reads the natural value of a later dose | `DataError` | the Jacobian determinant of the joint map is the product of the declared derivatives only when each map reads earlier doses. Any acyclic read has an order that makes the map triangular. The message names it, and `JointPolicy(..., order=...)` declares it | `tests/unit/test_longitudinal_mixed_node.py`, `tests/unit/test_vector_density_route.py` |
| dose maps that read each other in a cycle, and any other joint map that is not triangular | `DataError` | Equation (3) of Díaz et al. (2023) needs a map that is one-to-one on each piece. A determinant checked on a grid does not establish that a multivariate map is one-to-one, so no declaration of such a map is checkable | `tests/unit/test_vector_density_route.py` |
| a policy of one column on a point vector with a dose, and a static label, a rule, a known policy density or a policy of one column at a mixed node | `CapabilityError` | a static or dynamic value on a dose is not pathwise differentiable (Díaz et al. 2023, Section 4). A policy of one column does not say which component it moves. Declare a `JointPolicy` | `tests/unit/test_point_vector_treatment.py`, `tests/unit/test_longitudinal_mixed_node.py` |
| a known treatment mechanism, `treatment_probabilities=`, on a vector with a dose | `CapabilityError` | a known mechanism is a probability for each level. A vector with a dose has a conditional density and no levels, and a known density is outside this package | `tests/unit/test_point_vector_treatment.py`, `tests/unit/test_longitudinal_mixed_node.py` |
| a piece that sends an interval to an atom, `Piece(..., to_atom=)`, at a dose of a vector other than the last dose, on `ratio="density"` | `CapabilityError` | the mass that such a piece sends to the atom is an integral of the factors of the later components over the interval. Each later factor conditions on the collapsed dose, so the integral is not a finite sum over preimages. At the last dose, in the column order and in the policy's `order`, the mass is the integral of one binned density, which is a finite sum, and the density route fits the piece. A categorical map that reads that dose is refused. `ratio="classifier"` fits the piece at any dose. Declared atoms, `dose_atoms={name: (...)}`, make the factor of that dose mixed on both routes. A declared support, `treatment_support={name: Support}`, is supported | `tests/unit/test_mtp_component_atoms.py` |
| a declared missing treatment, `treatment_delta=`, on a vector with a dose | `DataError` | the composite indicator divides by the probability that the treatment is recorded and that the arm is $a$. A dose has no arm indicator | `tests/unit/test_point_vector_treatment.py` |
| `DRTMLE` and `CTMLE` on a vector with a dose, and `DRTMLE` with `policies=` at any treatment | `CapabilityError` | the `CTMLE` strategies need a discrete treatment, because `strategy="oat"` fits one prediction of $\bar Q$ for each arm. The reduced-dimension regressions of `DRTMLE` are derived for static treatment, and no theorem read here gives the reductions of a policy | `tests/unit/test_vector_study_spine.py` |
| `msm=` on a point vector with a dose | `CapabilityError` | an `MSM` projects the dose-response curve of one dose. The clever covariate integrates over a grid on that dose by the trapezoid rule. A vector has no one dose axis, and the package defines no grid or cubature over a vector. A longitudinal `msm=` over joint-policy plans ships, because its cells are plans | `tests/unit/test_vector_study_spine.py` |
| an arm-indexed estimand, `interventions=`, `incremental=`, a learned rule, or a fit with no `policies=`, on a point vector with a dose | `DataError` for `interventions=`, `incremental=`, a learned rule and a fit with no `policies=`. `CapabilityError` for an arm-indexed `estimands=` and for a study estimand | such a treatment has no arms, and each of these parameters is indexed by an arm. Say which values to compare with `policies=` | `tests/unit/test_point_vector_treatment.py` |
| `ratio="classifier"` where no policy ratio is estimated, such as an `ATE` or a declared mechanism | `MethodConfigurationError` | the setting would have no effect, and a silent no-op reads as a validated choice | `tests/unit/test_vector_study_spine.py` |

The refusal of a later-dose read is a scope limit and not a statement about the theory. A map
that reads a later dose is valid after a reorder, and the message names the order that resolves
the read. A cycle has no such order.
