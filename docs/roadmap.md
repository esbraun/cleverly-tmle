# Roadmap

This is the single planning contract for `cleverly`. It contains proposed work only. Implemented
capabilities belong in the [user guide](user-guide/index.md), scientific contracts in the
[technical reference](technical-reference/index.md) and [DR-TMLE contract](technical-reference/dr-tmle/index.md), validation results in
[evidence manifest](technical-reference/evidence.md), and cross-module standing decisions in the
[architecture invariants](architecture-invariants.md).

The main grid is one binding sequence. Complete lower numbers before higher numbers. Items with no
published theory do not enter this sequence.

`cleverly` is alpha, and it keeps no backward compatibility. `cleverly.load` warns with
`VersionMismatchWarning` when a different version saved the artifact, and it does not migrate the
artifact. This roadmap therefore carries no compatibility or migration items. The "Compatibility"
section of `CLAUDE.md` states the rule.

## Remediation

This queue is the triage set that must be complete before a beta release. It holds shipped
behavior that needs a correction or a recorded decision, and a published method needed to
resolve a shipped refusal. That behavior includes wrong numbers,
intervals that no claimed contract covers, capability rows that do not match their calls, and late
refusals. Other new capabilities and work that waits on published theory stay in the main roadmap
or in the future investigations grid. Deliver the rows in priority order, and complete every row before
main-roadmap priority 1.

A new capability still needs its own contract and evidence, even in this queue.

A queue row has five columns: priority, item, next action, problem and details. The "next
action" column states the remediation work, and it is not a readiness label.

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) is the current record.
It lists every red verdict that a registered study publishes. It names the ask that owns each
one, by the `id` in the "What this row asks for" table of RM18.
`tests/unit/test_red_cell_ledger.py` checks the ledger against the committed results. Each red
verdict stays red under a `reporting` policy, so no verdict is hidden and no margin moved.

The detail section of a delivered row keeps a short record of what shipped. Commit `dea3297e`,
or the commit that its detail section names, holds the full plan, probe and review record of each
delivered row that this roadmap still describes. Read a record with, for example,
`git show dea3297e:docs/roadmap.md`.

Pull request 250 delivered [RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes).
Its review found reused simulation seeds at the expanded replication budget.

| priority | item | next action | problem | details |
| ---: | --- | --- | --- | --- |
| 0.11 | RM33. Reused simulation seeds | repair the seed allocation and regenerate affected studies | repeated draws enter bootstrap intervals as separate observations | [contract](#rm33-reused-simulation-seeds) |

Each row takes a tier by the harm that its defect does to a user today. The table gives the tiers,
from the most harmful. Inside a tier, a row with a wider reach comes first. A row that another row
depends on comes before that row.

| tier | reason | rows |
| --- | --- | --- |
| a | a published number that is wrong, or that no derivation or read source covers. An anti-conservative number ranks above a conservative one | RM33 |
| b | a crash, an exception that is not a refusal, a capability row that reads available and then raises, or an assessment that returns no report | no open row |
| c | a correct refusal that arrives late or as the wrong type | no open row |
| d | a diagnostic or a warning that misleads | no open row |
| e | a display or a message that misstates a fact that the fit records. By extension, an argument check or a capability row that misstates what a call accepts or needs, when no number moves and nothing raises that is not a refusal | no open row |
| f | an investigation or a declared design that moves no verdict | no open row |
| g | a published method needed to resolve a shipped refusal | no open row |

The simulation group holds RM33. A delivery group joins rows that share a boundary. Each group
holds consecutive priorities, so the group order is the priority order. Deliver the items inside a
group in priority order. The first decimal digit of a priority names its group, and the second
digit orders the rows inside that group. Each remediation priority is below 1, so it does not
collide with a main-roadmap priority.

Priorities give the current delivery order. This project reassigns them when it re-triages the
queue. The RM IDs and their anchors never change, so a commit names a row by its ID. A delivered
row takes its priority with it, and the other rows keep theirs.

Main-roadmap priority 1 waits until every remediation row is complete, as the rule above states.
RM33 blocks priority 1 until its acceptance checks pass.

The F18 and F19 derivations do not block priority 1, because an item with no published theory does
not enter the sequence. Their cells stay red under `reporting` until F18 or F19 meets its
acceptance. That needs a published result, or a natural extension that meets the
[Eligibility](#eligibility) conditions.

The gaps below already have full line items. Keep each one there instead of creating a duplicate
contract.

| gap | item |
| --- | --- |
| post-selection inference for the shipped global selector | [F18](#f18-selector-path-c-tmle-inference) |
| the finite-dimensional joint binary extension and the shared-multinomial vector extension of the outcome-adaptive design. The fold-local binary nuisance replacement has direct support | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| the collaborative and doubly robust compositions that the package refuses, including an omitted-variable bound on a DR-TMLE or C-TMLE fit | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| competing-event intervention targets | [F3](#f3-additional-longitudinal-estimands) |
| a multi-arm stress surface | [F8](#f8-multi-arm-simulated-confounding-stress-surface) |
| longitudinal refutation replay | [F13](#f13-longitudinal-simulated-confounding-replay) |
| longitudinal sensitivity bounds | [F16](#f16-longitudinal-sensitivity-bound-estimation) |
| ordinary TMLE under `missingness=` for an attributable effect, which the RM8 audit moved | [F20](#f20-missing-outcome-attributable-effects) |
| a fold-targeted update and the repeated-split report with missing outcomes, the follow-ups of the arm-indexed stacked contract, and the cross-fitted shift, incremental, regime, MSM and controlled-direct-effect fits with missing outcomes, which refuse before any learner | [F21](#f21-other-missing-outcome-cv-tmle-variants) |
| the joint point-treatment parameter axes | [F17](#f17-joint-point-treatment-parameter-axes) |

## Main roadmap

| priority | item | readiness | dependency | details |
| ---: | --- | --- | --- | --- |
| 1 | Replicate-weight designs | source audit | weighted-law variance construction | [X2](#x2-replicate-weight-designs) |
| 2.1 | Sequential doubly robust longitudinal estimation | published support; pending source read | implemented longitudinal targets | [X4](#x4-sequential-doubly-robust-longitudinal-estimation) |
| 2.2 | Natural and interventional mediation effects | published support; pending source read | target-specific identification and evidence | [X5](#x5-natural-and-interventional-mediation-effects) |
| 2.3 | Continuous-time survival and competing risks | published support; pending source read | continuous-time intensity and targeting contracts | [X6](#x6-continuous-time-survival-and-competing-risks) |
| 2.4 | Two-phase and outcome-dependent sampling | published support; pending source read | observed-data likelihood and influence correction | [X7](#x7-two-phase-and-outcome-dependent-sampling) |
| 2.5 | Stratified incremental and MSM targeting | source audit | implemented pooled stratified fluctuation, and marginal incremental and MSM targeting | [X8](#x8-stratified-incremental-and-msm-targeting) |
| 2.6 | Omitted-variable bounds on the other linear functionals | published support; pending source read | the shipped arm-axis bound | [X9](#x9-omitted-variable-bounds-on-the-other-linear-functionals) |
| 2.7 | Continuous-dose MSM with a second mechanism | source audit | implemented continuous-dose MSM targeting, missing-outcome arm targeting, and controlled-direct-effect targeting | [X10](#x10-continuous-dose-msm-with-a-second-mechanism) |
| 2.8 | Learned-policy follow-ups | published support for parts (g) and (h); published support; pending source read for parts (a), (b), (e) and (f); source audit for parts (c) and (d) | [RM30](#rm30-learned-policy-value-evaluation) | [X11](#x11-learned-policy-follow-ups) |
| 3 | EP learner | published support; pending source read | shared study, fold, learner, and assessment contracts | [P1](#p1-ep-learner) |
| 4.1 | Nested Riesz engine and initial catalog | published support; source audit complete | typed study, identification, result, and assessment contracts | [R1](#r1-nested-riesz-engine-and-initial-catalog) |
| 4.2 | Evidence-gated Riesz catalog expansion | source audit for each target | R1 and a target-specific derivation | [R2](#r2-evidence-gated-riesz-catalog-expansion) |

## Future investigations

These items are hard stops. Move one into the main roadmap only after a published paper supplies
the missing result. Package code and a related estimator do not remove the stop.

| investigation | missing published result | current boundary | details |
| --- | --- | --- | --- |
| Multi-arm simulated-confounding stress surface | a contrast-specific, label-invariant category-valued latent perturbation law and its interpretation | binary flips and continuous linear dose perturbations only | [F8](#f8-multi-arm-simulated-confounding-stress-surface) |
| Clustered simulated-confounding stress surface | a source-backed choice of row-level, cluster-level, or mixed latent perturbation and its interpretation | row-level iid perturbations on unclustered fits only | [F9](#f9-clustered-simulated-confounding-stress-surface) |
| Logical categorical confounder calibration | a category-invariant benchmark mapped to the surface's perturbation strengths | numeric covariate calibration only | [F10](#f10-logical-categorical-confounder-calibration) |
| Estimated-weight simulated-confounding replay | stored weight-model provenance, target-population semantics, and a source-backed regeneration rule | fixed probability weights only | [F11](#f11-estimated-weight-simulated-confounding-replay) |
| Missing-outcome simulated-confounding replay | a joint observation, treatment, and outcome law with identified refit semantics | complete outcomes only | [F12](#f12-missing-outcome-simulated-confounding-replay) |
| Longitudinal simulated-confounding replay | a time-indexed latent law for treatments, censoring, histories, outcomes, and contrasts | point-treatment results only | [F13](#f13-longitudinal-simulated-confounding-replay) |
| Controlled-direct-effect simulated-confounding replay | an ordered treatment, intermediate, observation, and outcome law with a contrast contract | fits without an intermediate only | [F15](#f15-controlled-direct-effect-simulated-confounding-replay) |
| Simulated confounding on a declared outcome scale | a latent perturbation law for an outcome confined to a known support, and the reading of its strength | additive perturbation of an unbounded outcome only, so a fit that declares `q_bounds` refuses the outcome axis | [F23](#f23-simulated-confounding-on-a-declared-outcome-scale) |
| Controlled-direct-effect E-value | a bound on the bias of a controlled direct effect from unmeasured confounding, of the treatment and the outcome or of the intermediate variable and the outcome, and its inversion to an E-value on a stated outcome scale | every E-value request on a fit with an intermediate variable and a discrete treatment reports `unavailable`. A continuous-treatment fit reports `not_applicable` first | [F25](#f25-e-value-for-a-controlled-direct-effect) |
| Plug-in omitted-variable confidence limits | an influence function or an inference result for the plug-in estimate of the Riesz second moment with an estimated treatment mechanism | the plug-in bounds, `rv`, `max_bias`, `benchmark()` and `contour()`. The one-sided limits and `rva` refuse | [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) |
| Stochastic categorical policies at a longitudinal node | longitudinal identification, influence function, remainder, and interval conditions for a distribution-valued policy | deterministic categorical regimens only | [F1](#f1-stochastic-categorical-policies-at-a-longitudinal-node) |
| Targeted bootstrap inference | a construction that defines what is fixed, resampled, refitted, and retargeted, plus the sampling law of the interval | existing bootstrap inference is not this procedure | [F2](#f2-targeted-bootstrap-inference) |
| Longitudinal sensitivity-bound estimation | sample estimation of the bound functionals, a specialized algorithm, and sampling inference | no sensitivity bound on a longitudinal fit | [F16](#f16-longitudinal-sensitivity-bound-estimation) |
| Additional longitudinal estimands | target-specific identification, influence function, targeting construction, and inference conditions | existing end-of-study, survival, competing-risk, and MSM targets only | [F3](#f3-additional-longitudinal-estimands) |
| Multi-arm missing-outcome DR-TMLE | joint and contrast inference across arms, and one treatment mechanism compatible with a separate logistic tilt for each arm | binary randomized treatment only | [F4](#f4-multi-arm-missing-outcome-dr-tmle) |
| Other refused C-TMLE and DR-TMLE compositions | composition-specific score, reduced regressions, correction, remainder, and rate conditions | named pre-fit refusals. A `DRTMLE` fit with a non-empty `guard` and estimated weights reports its point estimate under the `estimated_weight_plugin` status, and no interval | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| Selector-path C-TMLE inference | an influence function and covariance after the shipped data-adaptive stopping-index selection | point estimates and path diagnostics only. The greedy, ordered, and discrete paths refuse `ci`, `pvalue`, and `std_error`, and report a named working-mechanism plug-in diagnostic | [F18](#f18-selector-path-c-tmle-inference) |
| Outcome-adaptive C-TMLE generated-design inference | exact scalar expansions for the shipped joint binary fit and a multi-arm vector extension of the paper-backed fold-local construction | point estimates only. Every `strategy="oat"` fit refuses `ci`, `pvalue`, and `std_error` under the `generated_design_plugin` status, and reports a named generated-design plug-in diagnostic. No shipped fit is the proved binary scalar construction | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| Missing-outcome attributable effects | a direct observed-data derivation for the joint natural-course and reference-intervention means, their remainder, and attributable-effect inference | complete-data PAR and PAF, one iid MAR natural-course mean, and arm-specific missing-outcome means remain separate | [F20](#f20-missing-outcome-attributable-effects) |
| Other missing-outcome CV-TMLE variants | a direct interval result for fold-specific targeting and for the fixed-repeat median and split-dispersion report after CV-TMLE targeting | the package supports the ordinary natural-course estimator, the stacked natural-course estimator for a binary outcome, and the stacked arm-indexed means and contrasts. Each stacked estimator uses one repeat and pooled targeting. F21 also records their follow-ups. Shift, incremental, regime, MSM, and controlled-direct-effect targets refuse a cross-fitted fit with missing outcomes before any learner | [F21](#f21-other-missing-outcome-cv-tmle-variants) |
| Grouped cross-fitting beyond point-treatment TMLE | a split law and a cluster-robust variance for the C-TMLE selection folds, a cluster-robust variance for the cross-fitted longitudinal recursion under a grouped draw, and for the point-treatment fits an interval result at unequal cluster sizes and a $t$-reference claim at few clusters | whole-cluster outer folds for cross-fitted point-treatment TMLE and DR-TMLE only. A cross-fitted fit at unequal cluster sizes, in rows or in weight mass, overall or within a reported stratum, takes the `unequal_cluster_plugin` status. A fit with fewer than 40 positive-mass clusters, in total or in one reported stratum, takes `few_cluster_plugin`. Neither reports an interval | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| MNAR and incremental-intermediate compositions | identification and influence-function results for the exact compositions | point-treatment sensitivity and implemented interventions remain separate | [F6](#f6-mnar-and-incremental-intermediate-compositions) |
| Joint point-treatment parameter axes | a targeting and inference result for one fit that carries an MSM projection together with a regime, shift, or incremental intervention, including its joint score and covariance | single-axis point-treatment fits only | [F17](#f17-joint-point-treatment-parameter-axes) |
| Time-respecting cross-fitting | dependence and split-specific TMLE inference for blocked-temporal or rolling-origin folds | iid and grouped cross-fitting only | [F7](#f7-time-respecting-cross-fitting) |
| Learned-policy value outside the published conditions | an interval for the fold-average learned-rule value without a limiting rule, and results for the refused compositions | `LearnedRuleValue` fits the fold-evaluated CV-TMLE of [RM30](#rm30-learned-policy-value-evaluation) under the limiting-rule condition. Its boundary study measures under-coverage at an exceptional law. The other F27 requests refuse before any learner | [F27](#f27-learned-policy-value-outside-the-published-conditions) |

## Eligibility

`cleverly` implements established statistical methods; it does not use a package feature as the
place to invent one. A scientific feature enters implementation only when a published derivation
covers the estimand and requested inference regime. A new estimator or composition requires an
identified parameter, its influence function, the targeting or estimating equations, and the
remainder and rate conditions needed for the claimed interval. If one is absent, the item is to
locate published theory, not create it here.

One exception applies. A trivial or natural extension of a published derivation can close the gap.
Such an extension needs no new fundamental proof. It must satisfy every condition in the table
below.

| condition | what it requires |
| --- | --- |
| a published base | the extension starts from one published result, and it cites that result's exact locator |
| an established argument | each step reuses the base result's own argument, or a standard result such as linearity, the delta method, a fixed-dimension stack by Cramér–Wold, or the chain rule for influence functions |
| no new fundamental proof | no step needs a new kind of limit theorem, and no source records a step as open or as a defect |
| inherited conditions | the remainder and rate conditions carry over from the base result, and the contract states each one |
| a written record | the item's contract states the base result, each step, and the argument that carries it |
| its own evidence | the extension has a registered study, and each step that can vanish at the truth has a nonzero witness |

An extension that fails one condition is new theory. It stays in the future investigations grid.
Three examples fail. A data-adaptive selection step has no established argument. A fold-local
update replaces a pooled one, and upstream `lmtp` commit `9996b04` records its training-fold update
as a bug. A growing target
dimension needs a new kind of limit theorem.

A canonical public implementation is valuable provenance for control flow, data layout, and named
conventions, but it is not acceptance evidence by itself. Where code and paper disagree, the
published derivation governs and the discrepancy becomes a nonzero regression or mutation test.

The readiness labels rate published-method support, not programming effort:

- **published support**: a paper derives the method and inference claim;
- **source audit**: a published paper appears to cover it, but the exact construction must be
  matched and discrepancies resolved before implementation;
- **waiting on published theory**: related methods exist, but not the requested composition or
  inference claim; it belongs only in the future investigations grid; and
- **pending source read**: the governing result is identified but has not been read first-hand
  into this package's contract.

## Definition of done

An item is complete only when all applicable conditions hold:

- the estimand is registered and covered in both directions by the oracle and evidence gates in
  `tests/unit/test_registry.py`, with an [evidence](technical-reference/evidence.md) row naming
  which instruments check its influence curve and which mistakes none can see;
- every well-posed composition still refused has a pre-fit test pinning the refusal and message;
- signs, masks, guards, and counterfactual blocks that can vanish at truth have a nonzero witness
  or deliberate-mutation control in addition to exact-law checks;
- cross-module changes satisfy [the architecture invariants](architecture-invariants.md);
- reader-facing behavior, methodology, references, and evidence are updated without
  presenting a proposal as a release claim; and
- every relevant check has run locally and GitHub Actions is green. CI is the final merge signal,
  not a substitute for the local validation record.

## Sensitivity and validation priority

The [implementation validation grid](technical-reference/method-evidence/validation-grid.md)
records completed studies. The ordinary and binary stacked missing-at-random natural-course means
are implemented and registered there. The stacked arm-indexed missing-outcome means and contrasts
are also registered there. The ordinary arm-indexed missing-outcome row covers only binary `ey1`,
`ey0`, and `ate`. The
[fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) hold
the source audit that changed the default cross-fitted fit.
The [red-cell ledger](technical-reference/method-evidence/red-cells.md) carries the property
cells that went red when sixteen registered studies were regenerated under it.

Replicate-weight designs are the next source-audit item in the main grid.
Implement them after RM33 closes and the source audit supports the planned variance construction.

Longitudinal sensitivity-bound estimation remains in
[F16](#f16-longitudinal-sensitivity-bound-estimation).

## Detailed implementation contracts

The sections below group contracts by subsystem. Their physical order does not override the main
grid.

### RM8. Missing-outcome attributable effects

The 2026-09-12 source audit did not find a published derivation for the exact missing-outcome
attributable-effect construction. The reviewed sources establish only separate component results.

Hubbard and van der Laan (2008) derive complete-outcome PAR and PAF. Díaz, Carone and van der Laan
(2016) derive one scalar MAR natural-course mean. Díaz and van der Laan (2017) derive arm-specific
means for randomized treatment with missing outcomes. Van der Laan and Rubin (2006) require the
observed-data influence curve before targeting it.

Combining these pieces locally would create the missing observational response transfer, joint
remainder, and inference contract. The [audit record](references.md#point-treatment-and-stochastic-interventions)
therefore moves this item to [F20](#f20-missing-outcome-attributable-effects).

Keep the public pre-fit refusal. Complete-data PAR and PAF remain supported. The iid MAR
natural-course mean and arm-specific missing-outcome means also remain separate supported results.

### RM11. Sensitivity bounds outside their derivation

Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026) bound the omitted-variable bias of a
linear functional of the outcome regression. The bound is $\sqrt{\sigma^2 \nu^2}$ times a strength
factor. Here $\nu^2$ is the second moment of the Riesz representer for the declared adjustment set.
The default estimate of $\nu^2$ is low when the fitted mechanism is wrong, which is the case that
DR-TMLE guards against. A C-TMLE representer comes from the selected working mechanism. The bound
has no response mechanism.

Pull request 223 delivered this row. The table gives what shipped. A symbol is in
`src/cleverly/sensitivity/omitted_variable.py` unless the row names another file.

| part | what shipped |
| --- | --- |
| fit-wide refusal | `_FIT_WIDE_BOUND_RULES` refuses every omitted-variable operation on seven kinds of fit, in this order: `longitudinal`, `drtmle`, `collaborative_tmle`, `response_mechanism`, `intermediate`, `parameter_axis` and `repeats`. `sensitivity_elements` raises the refusal before any computation. The five entry points and the five assessment rows share it |
| E-value on a missing-outcome fit | `_select_evalue` in `src/cleverly/sensitivity/evalue.py` refuses the standardized E-value. An E-value from a reported ratio still answers |
| nonpositive $\nu^2$ | a refusal that names the estimator. The code had fallen back to `"plugin"` with no message |
| messages | each per-axis refusal names the parameter axis. The `evalue` pointer appears only for an `rr` or `or` request. Five docstrings list the accepted `nu2_estimator` values |
| ATT and ATC $\nu^2$ | `_m_alpha` weights the contrast by the observed $1\{A = c\} / P(A = c)$, and not by the fitted propensity |

`tests/unit/test_omitted_variable_refusals.py` and `tests/e2e/test_sensitivity_and_validation.py`
hold the witnesses. [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) holds the derivation
that would reopen the bound on a DR-TMLE or C-TMLE fit.

### RM12. Collaborative intervals at an inconsistent working mechanism

The greedy, ordered, and discrete C-TMLE paths reported the ordinary EIF plug-in covariance at the
selected working mechanism. With a correct linear outcome regression and an intercept-only working
mechanism, that variance ignores the association between the treatment and the covariates.

| measurement | law and fits | result |
| --- | --- | --- |
| one draw | `make_instrument(n=2000, seed=44)`, greedy path | reported standard error 0.0441. The HC0 standard error of the same least-squares coefficient is 0.0545 |
| least-squares sampling spread | `make_instrument(n=2000)`, seeds 1000 to 1999 | empirical standard deviation 0.0538 |
| greedy C-TMLE repeated sampling | `make_instrument(n=2000)`, seeds 3000 to 3299, three outer folds, three selection folds | mean reported standard error 0.0454 against an empirical standard deviation of 0.0537, a ratio of 0.844. Nominal 95% intervals cover in 0.92 of fits |

Pull request 223 delivered this row. The table gives what shipped.

| part | what shipped |
| --- | --- |
| status | `ParameterEstimate` carries an `inference` field. `CTMLE` sets it to `working_mechanism_plugin` on the greedy, ordered, and discrete paths. `ci`, `pvalue`, and `std_error` raise `CapabilityError`. `plugin_std_error` and `plugin_interval` keep the diagnostic |
| downstream refusals | a contrast, a simultaneous band, every E-value branch, `variable_importance()`, and `tipping_gamma(use_ci=True)` refuse on these paths |
| names | `influence.spread_name` gives the name that each spread takes under each status |
| path record | the docstrings state that the unpenalized fluctuation cannot worsen its own loss, while the recorded penalized selection risk can rise |

`TestTheWorkingMechanismDiagnosticMissesTheExactVariance` and
`TestTheRecordedRiskMayRiseWhileTheLossMayNot` in `tests/unit/test_ctmle.py` are the witnesses.
The planned coverage cell was retired, because a selector fit publishes no interval. The cells that
the F18 acceptance names measure the diagnostic.
[F18](#f18-selector-path-c-tmle-inference) holds the influence curve that would reopen
selector-path inference.

### RM13. Estimated MSM projection weights

A `weights` callable receives an arm label and a covariate frame, and it can close over any
estimate. A weight that closes over an estimate is a functional of $P$, and the influence curve
omits its pathwise derivative. The package cannot inspect a closure, so the status of the weight is
a declaration.

Pull request 224 delivered this row. The table gives what shipped.

| part | what shipped |
| --- | --- |
| declaration | the field `MSM.weights_kind`, which takes `"known"`, `"estimated"`, or `None`. It is the last field, and its default is `None` |
| refusals | a callable weight with `weights_kind=None` raises `CapabilityError` and asks for `"known"`. The value `"estimated"` raises `CapabilityError` and names the missing pathwise-derivative term. An array in place of a callable raises `CapabilityError` |
| check sites | `cleverly.msm.refuse_msm_functions`, which RM27 renamed from `refuse_projection_weights`. `MSM.__post_init__`, `TMLE._resolve_estimands_for_data`, `TMLE._retarget_detailed`, `LTMLE.fit`, `MSMSet.evaluate` and `evaluate_regimen_msm` call it |

`tests/unit/test_msm_projection_weights.py` holds the witnesses. In the exact-law witness, a weight
equal to the sample arm share and declared `"known"` reports a `msm[W]` standard error of 0.7417 of
the exact one, against a bound of 0.8. The [MSM reference](technical-reference/msm-projections.md#variations)
records the declaration and both refusals.

### RM14. Intervention refusals at identification

The typed estimands did not check the kinds of their elements. A mixed request passed
`CausalStudy.identify`. Six probed requests then raised `AttributeError` at `estimate`, four of
them after one learner fit. A `Shift` in `IncrementalEffect` raised nothing, and the fit read the
shift as an odds multiplier. Each refusal named a `TMLE` keyword, and the top level does
not export `TMLE`.

Pull request 239 delivered this row. The table gives what shipped. Commit `b876ec3f` holds the
probe table and the corrections. Read it with `git show b876ec3f:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| identification | `_identify_point` in `src/cleverly/study.py` runs `refuse_mixed_interventions` on the set of each regime, shift and incremental estimand. An item of another kind raises `CapabilityError` before any learner |
| point sets | `_identify_point` refuses a mapping, an iterator, a string or a single item as a point set with `DataError`, before it reads the set. The message names the tuple form |
| messages | each refusal names the field and the item. For an item of another kind it names the typed estimands of that kind and cites [F17](#f17-joint-point-treatment-parameter-axes). For a bare value in a shift or incremental set it names the object to write. The two refusals of an estimated density also name `IncrementalMean` and `IncrementalEffect` |
| estimator guard | `as_interventions` and the `TMLE` constructor run the same check on `interventions=`, `shifts=` and `incremental=`. The constructor refuses two intervention keywords together, or one of them with `msm=`, with `CapabilityError` |
| zero-dimensional plan | commit `b64d334d` refused it with `DataError`. The message now names the sequence form and `DynamicRegimen` |

`tests/unit/test_intervention_kind_refusals.py` holds the witnesses and two mutation controls.
`tests/unit/test_regimen_rule_declarations.py` holds the zero-dimensional plan test.

### RM15. Calibration-slope warning rule

`NuisanceDiagnostics.findings` warned when a pooled one-intercept calibration slope fell outside
[0.7, 1.4]. The finding said that the model "biases the weights", which the rule did not measure.
The band warned on 27% of fits of the true propensity at a weak signal, and on every mean-only fit
of a randomized law. There, the pooled intercept made the slope $-2$ at three folds. The band
applied to in-sample slopes too. The longitudinal report put a logistic and a linear slope under
one name.

Pull request 242 delivered this row. The table gives what shipped. Commit `87b13e7d` holds the
probe tables, the design and the study declaration. Read it with `git show 87b13e7d:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| statistic | `_recalibration` in `src/cleverly/validation/nuisance.py` regresses the label on one indicator for each validation fold and on $\operatorname{logit} \hat p$. `calibration_slope_se` is the sandwich standard error with the fit's cluster codes. A prediction that is constant within every fold or that separates the labels, a failed fit, or a standard error that is not finite gives no slope, and `calibration_omission` names the cause |
| rule | `_calibration_finding` warns when the Bonferroni interval over the tested models lies above 0 and excludes 1. An in-sample fit tests no model, and `NuisanceDiagnostics.evaluation` records the basis |
| message | the direction, the slope, the interval, the AUC, and for a weight model the largest untruncated inverse weight. It asks for a review of the learner and makes no claim about bias |
| names and summary | `regression_slope` holds the linear slope in the point and the longitudinal reports. The summary table gains `cal_se` and `reg_slope`, a line for the basis and a line for each omission |
| evidence | the [calibration-slope study](technical-reference/method-evidence/calibration-slope-warning.md): the rule's rate is 0.0225 to 0.0539 on five laws where a warning is false, the old band's rate is 0.2543 to 0.9869 on four of them, and both tempered learners are detected in every fit. `tests/unit/test_calibration_slope_rule.py` holds the witnesses |

The plan corrected the premise of this row. A warning on a correct model is not always false.

| premise | what the delivery found |
| --- | --- |
| a correct model should have a slope of 1 | noise in the fitted coefficients can make the out-of-fold predictions of a correct weak-signal model more extreme than the truth. The study's correct model on four weak covariates has a mean slope near 0.58. The rule does not remove that over-dispersion. It reads the slope only when the interval lies above 0 |
| a slope finding means the model is misspecified | the population slope of any logistic limit with an intercept and main effects is 1, whether the model is correct or not. The slope cannot see misspecification of that kind |

### RM16. Summary and error-message accuracy

Summaries and messages omitted, misstated, or repeated facts that the fit records. A longitudinal
summary printed a reference beside regime means, and its identification summary listed only the
baseline covariates. A DR-TMLE summary named no guard, and the support table did not say which
mechanism each column read. Each summary printed the protocol record again, and the provenance
line of a result summary printed the fingerprint a second time. A non-inferential fit published
its bootstrap spread as a standard error, and the package called its full-refit bootstrap
"targeted". Messages sent a user to a refused fit or a field that the design does not have, and
argument checks accepted requests that measure nothing.

Pull request 243 (RM16a) and pull request 244 (RM16b) delivered this row. The table gives what
shipped. Commit `d8c654b6` holds the probe table, the corrections and the routed findings. Read it
with `git show d8c654b6:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| messages | one missing-outcome remedy, `MISSING_OUTCOME_DECLARATION` in `src/cleverly/data/validate.py`, names `missingness=` and `delta=` at three sites. The in-sample C-TMLE selection-fold refusal names the fold and two remedies that run as written. On a cross-fitted `TMLE` with missing outcomes, the continuous-treatment message names [F21](#f21-other-missing-outcome-cv-tmle-variants) and the in-sample fit |
| argument checks | `benchmark` refuses an empty request and an indicator column, and accepts the logical name of an encoded covariate. A facade reads a one-shot iterator once. The guarded DR-TMLE truncation curve refuses a refused or changed reduction construction on the row, the facade and the module call |
| types and references | `_solve_reduction` and `refuse_unsupported("link")` raise `ValueError`, and `refuse_scheme` raises `CapabilityError`. The `DRTMLE` docstrings name test classes that exist, and `test_every_named_test_exists` checks each reference |
| longitudinal summaries | `LongitudinalConfig.describe` prints the reference only for a result that holds a contrast. `config.reference` stays set, because the replay reads it. The identification summary prints the history at each treatment node from `BackdoorMeanContrast.history` |
| DR-TMLE and support | `ReducedFit.describe` gives the guard and the reduction, or the empty guard, in `TMLEResult.summary`. `SupportReport.summary` names the mechanism that each column reads |
| protocol | `summary(protocol="fingerprint")` prints only the fingerprint line of the record, through `protocol_summary_lines` in `src/cleverly/protocol.py`. The default prints the whole record. `Provenance.describe` no longer prints the fingerprint. Each of the 10 notebooks prints the record once |
| spread names | the split-spread fact of the `nuisance_models` row and the bootstrap rows read their names through `spread_name`. A non-inferential fit publishes `bootstrap_sd` and `bootstrap sd`. `BootstrapSummary` keeps its field names. The summary, the docstrings and the pages call the procedure the full-refit bootstrap, and F2 keeps its name |

`tests/unit/test_summary_and_message_accuracy.py` holds the witnesses. The pinned tests in
`tests/unit/test_drtmle_fit.py`, `tests/unit/test_msm.py` and `tests/unit/test_learners.py` check
the three exception types.

| decision | what it chose |
| --- | --- |
| D1 | the bootstrap of a non-inferential fit publishes each number under the name that the status allows |
| D2 | `protocol="full"` or `"fingerprint"` on the identification and result summaries, with `"full"` the default |
| D3 | no provenance line for the protocol fingerprint |
| D4 | the notebooks print each protocol record once |
| D5 | one missing-outcome remedy that names both spellings |
| D6 | the selection calls of `cross_fit_predictions` pass the selection remedy |
| D7 | the guarded truncation curve needs the `retarget_cached_nuisances` slot, as pull request 238 chose. It refuses a refused reduction setting, and a `guard`, `reduction` or `reduced_crossfit` other than the fitted construction |
| D8 | two pull requests, RM16a for the messages and RM16b for the displays |
| D9 | the three exception types are part of this row |
| D10 | `available_methods` stays per method name, and its docstring states the stratified `DRTMLE` case |

An explicit `simultaneous=True` on a fit that supplies no inference makes no band and raises no
warning. Its summary names the omission, and
[inference status](technical-reference/inference.md#inference-status) records the policy.

Commit `cdb55d6e` delivered the fold-policy refusal of a reconfigured continuous-dose estimator.
Commit `b64d334d` delivered the refusal of a longitudinal plan written as a mapping. Pull request
229 corrected the reasons of `_risk_ratio_refusal` and `_select_evalue`.

### RM18. Red property cells after the fold, scale and law changes

Sixteen registered studies moved to unstratified folds. Six of them also moved to a bounded
outcome law with a declared support, and one moved to a binary clustered law. The six are the
manifests that record `tests/studies/bounded_cv_laws.py`. The multi-arm selector row, the
multi-arm outcome-adaptive row and both DR-TMLE rows stay on a binary outcome and declare no
`q_bounds`. The pooled longitudinal update then regenerated five cross-fitted longitudinal
studies. Property cells went red after each change.

This row asked for the missing results. It did not ask for greener numbers. Every gate named here
is interval-shaped (`tests/studies/evidence/registry.py`). A larger budget narrows Monte Carlo
uncertainty around the truth and can change an unresolved verdict. Three post-run remedies are
therefore refused for every entry below. They are raising a budget, moving a margin, and
re-declaring a positive cell's law, learner or size after seeing its verdict.

Pull request 245 and pull request 250 delivered this row. The table gives what shipped. Commit
`985849c6` holds the declarations of the five follow-up designs and their amendments. It also
holds the readings of designs FW, OW, BD and CD, and the review record. Read it with
`git show 985849c6:docs/roadmap.md`. Commit `dea3297e` holds the earlier declarations and
readings.

Each commit in the table below was pushed before the run that it governs. Commits `4a44545a` and
`bbbb887e` are on the remote branch `agent/remediation-rm18b`. Commits `0ac9753e` and `d42306fb`
carry them onto the final stack without change. The annotated tag `rm18-slopes-declaration`
marks `bbbb887e`, and its ancestry holds `4a44545a`. The tag is the record that the Design SL
declaration came before any run.

| commit | what it declared |
| --- | --- |
| `ebd3d1b9` | designs FW, OW, BD, CD and SL, and the rules R1 to R7 |
| `7e65bea7` | the seed-collision rule |
| `03fe89f1` | the amendments of the harness review: the collision rule, the failed-fit rule, R6 and four source statements |
| `4a44545a` | set the declared Design SL budgets in code: 73,000 replications on each outer rung, and a verdict budget of 600 |
| `bbbb887e` | the first amendment of the Design SL run form: the reference phase runs, and its artifacts must reproduce byte for byte |
| `985849c6` | the second amendment of the Design SL run form: the stopped first run, the stack order, the scratch output, the run log and the guard |

| part | what shipped |
| --- | --- |
| ledger | the [red-cell ledger](technical-reference/method-evidence/red-cells.md), which `python -m tests.studies.evidence.red_cells` writes from the committed results. Every red verdict stays red under `reporting` at its registered budget and margin |
| earlier readings | `RM18-pooled`, `RM18-n500`, `RM18-binary-slope`, `RM18-attribution`, `RM18-one-sided-bias` and `RM18-learner-weight-se`. The cross-fitted longitudinal fit uses the pooled update of Section 5.2, Steps 1 to 4, of Díaz, Williams, Hoffman and Schenck (2023) |
| Design FW | [`tests/diagnostics/rm18_fixed_weights/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_fixed_weights). FW-A reads `finite-sample, contracting`, and FW-B reads `equivalent at the declared budget` |
| Design OW | [`tests/diagnostics/rm18_ordinary_weighted/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_ordinary_weighted). OW-A reads `finite-sample, contracting`, and OW-B reads `resolved within gate`. OW-C reads `control underpowered by design` and `family resolved` |
| Design BD | [`tests/diagnostics/rm18_boundary/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_boundary), which also holds FW-B. The six re-read cells read `resolved: truth satisfies the gate`, and the two paired rows read `comparator SE convention` |
| Design CD | [`tests/diagnostics/rm18_comparator_density/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_comparator_density). The paired shift row reads `cleverly density representation` |
| Design SL | the original regeneration of `canonical-multi-arm-drtmle` at 73,000 replications on each outer rung. Both original positive slopes read `contracts`. "[What design SL found](#what-design-sl-found)" records that run. [RM33](#rm33-reused-simulation-seeds) repairs its reused sample seeds |
| routes | no reading names a defect in `cleverly`, so no reading opened a new row. The two slope cells pass under the registered rule, and the ledger no longer lists them. Every other cell keeps its owner and stays red under `reporting` |

The four `tests/unit/test_rm18_*_diagnostic.py` files of designs FW, OW, BD and CD rebuild each
reading from the committed rows. `tests/unit/test_rm18_slopes_reading.py` applies the Design SL
reading table to the committed rows. The test
`test_paper_property_verdicts_are_recomputed_from_the_replication_rows` in
`tests/unit/test_method_evidence.py` recomputes each published verdict.

#### The red-cell ledger

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) holds the current state.
It lists every red verdict that a registered study publishes, with the endpoints that verdict
failed on. It names the `id` of the ask in "What this row asks for" that owns each row.
`python -m tests.studies.evidence.red_cells` writes the ledger from the committed results.
`tests/unit/test_red_cell_ledger.py` fails on four states: a red row with no owner, an owner with
no red row, an owner that the `id` column does not list, and a red row in a `gated` study.

Every red verdict stays red under `reporting` at its registered budget and margin. The rows that
F18 and F19 own stay red until F18 or F19 meets its acceptance.
[RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle) read the one measured increment
that the one-sided robustness reading opened. On fresh draws it reads `no increment at the
declared resolution`, and it moved no owner.

(what-this-row-asks-for)=
#### What this row asks for

The `id` column names each ask. A red cell names the ask that owns it by this `id`. The last
seven rows are the follow-ups this row planned. Each acceptance cell starts with the state of its
ask: `delivered`, or `open` with the reason. "[What each owner holds](#what-each-owner-holds)"
gives the cells of each owner and the reason for each assignment.

| id | work | acceptance |
| --- | --- | --- |
| `F18` | an inference result for the shipped selector path | open. An influence curve derived after the stopping-index selection, and a registered study whose `selector_necessity` and `type_i_error` cells pass their existing margins at their existing budgets. The multi-arm `interval_calibration/correctly_specified` cell must also pass its band, because a selector-path covariance is the claim F18 supplies |
| `F19` | an inference result for the generated design | open. A published result, or an Eligibility-qualifying natural extension, must establish whether the ordinary adaptive-propensity EIF suffices for the exact cross-fitted shared-multinomial joint target. Implement any representation contribution the result requires, then register covariance and coverage evidence at predeclared budgets. Keep the `generated_design` pair as a reporting diagnostic. Acceptance does not require a negative standard-error deficit |
| `F27` | an interval for the fold-average learned-rule value outside the published conditions | open. [F27](#f27-learned-policy-value-outside-the-published-conditions) gives the acceptance for each request. Its `exceptional` row needs a published result. Its `weak_blip` row names no missing result, because that law is within C3 of RM30 |
| `RM18-pooled` | a targeting result for the fold-local longitudinal recursion | delivered by the pooled route. The package now implements Section 5.2, Steps 1 to 4, which Theorem 3 certifies. `crossfit_overfitting/cross_fitted_ltmle` sits inside the shared ceiling at 8,000 draws, with a 99% interval from 0.996092 to 1.036997. "What the pooled update found", in commit `dea3297e`, gives every moved cell, including two weighted cells that went red |
| `RM18-n500` | a reading of the two `n_500` coverage endpoints | delivered. A registered fold-policy diagnostic reads both endpoints as a boundary resolution. "What the two readings found", in commit `dea3297e`, gives the numbers |
| `RM18-binary-slope` | a reading of the DR-TMLE contraction slope | delivered. A declared rung design resolves the slope, and its interval now sits below zero. "What the two readings found", in commit `dea3297e`, gives the numbers |
| `RM18-attribution` | an attribution of the verdicts that changed with the pooled update | delivered for the end-of-study and weighted studies. A declared code-by-runtime diagnostic reads the two weighted cells and the end-of-study overfitting cell as code changes. "What the runtime isolation found", in commit `985849c6`, gives the numbers. The weighted cells stay red as finite-sample reporting evidence |
| `RM18-fixed-weights` | a fixed-known-weight follow-up for the two weighted cross-fitted cells that went red with the pooled update | delivered. Design FW ran once. FW-A reads `finite-sample, contracting`, and FW-B reads `equivalent at the declared budget`. "What design FW found", in commit `985849c6`, gives the numbers. The two cells stay red under `reporting` at their registered budgets |
| `RM18-one-sided-bias` | a reading of the three one-sided-robustness bias rows | delivered. The reading that "The one-sided robustness reading, declared before it is computed", in commit `dea3297e`, declares ran once. "[What the one-sided reading found](#what-the-one-sided-reading-found)" gives its result. The three cells stay red under `reporting` |
| `RM18-slopes` | a reading of the two multi-arm DR-TMLE contraction slopes | delivered. The original Design SL run reads `contracts` for both slopes. Both original rate cells pass under the registered rule. "[What design SL found](#what-design-sl-found)" records those numbers. [RM33](#rm33-reused-simulation-seeds) corrects the reused sample seeds |
| `RM18-ordinary-weighted` | an owner for the six red rows of the ordinary weighted longitudinal study | delivered. Design OW ran once. OW-A reads `finite-sample, contracting`, OW-B reads `resolved within gate`, and OW-C reads `control underpowered by design` and `family resolved`. "What design OW found", in commit `985849c6`, gives the numbers. The six rows stay red under `reporting` |
| `RM18-learner-weight-se` | a diagnostic for the standard-error explosions in the cross-fitted `learner_weight_necessity` rows | delivered. A complete review declaration was pushed before a clean rerun that reproduced the refit table. "What the learner-weight diagnostic found", in commit `dea3297e`, gives its reading. This row owns no red cell |
| `RM18-boundary` | an owner for the red cells that sit at a finite-budget boundary | delivered. Design BD ran once. The six re-read cells read `resolved: truth satisfies the gate`, and the two paired rows read `comparator SE convention`. "What design BD found", in commit `985849c6`, gives the numbers. Each cell stays red under `reporting` at its declared budget |
| `RM18-comparator-density` | an attribution of the red paired row of the continuous modified treatment policy study | delivered. Design CD ran once and reads `cleverly density representation`. "What design CD found", in commit `985849c6`, gives the numbers. The row stays red under `reporting` |

(what-each-owner-holds)=
##### What each owner holds

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) lists every red cell by
its owner. This subsection gives the reason for each assignment.

`F18` holds the red cells of both selector studies, except the multi-arm
`root_n_and_efficiency/n_500` endpoint. On both laws of the point-treatment study, the population
one-step remainder is exactly zero at the nuisance limits. That limit does not isolate the
finite-sample nuisance remainder or the effect of the selector's stopping rule on the red cells.
On the multi-arm study, the greedy and ordered paths
miss their bias margins, and the discrete path stops at the empty candidate. The reversed
standard-error ratio of `selector_necessity/collaborative` is also a nonzero witness for the
[RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism) refusal, and a label
alone does not close it.

The multi-arm `interval_calibration/correctly_specified` cell stays with `F18`. Its
standard-error ratio interval runs 0.8987 to 0.9862 under the covariance that treats the selected
candidate as fixed. A selector-path covariance is the claim that F18 exists to supply, so the F18
acceptance names the cell.

The multi-arm selector `n_500` endpoint belongs to `RM18-boundary`. `RM18-n500` read it as a
boundary resolution. `RM18-boundary` holds the multi-arm DR-TMLE `n_500` cell for the same
reading.

`F19` holds the `generated_design` pair of `canonical-multi-arm-ctmle-oat`. Both designs are
judged by calibration to their own sampling spread. The exact shared-multinomial generated-design
expansion remains open.

`F27` holds the two truth rows of `learned-rule-cvtmle-boundary`, the boundary study of
[RM30](#rm30-learned-policy-value-evaluation). Each row fails its coverage floor only. The RM30
declaration names F27 as the owner of every red cell of that study. The F27 table states that the
`exceptional` row has no published result and that the `weak_blip` row has one.

`RM18-fixed-weights` holds `interval_calibration/static__correctly_specified` and the paired
`ey_regimen[never]` row of `weighted-ltmle-crossfit`. Under this study's iid baseline-selection
law, normalized fixed weights change the empirical law. The pooled update applies the
corresponding weighted targeting equation. The code attribution therefore does not show that
fold-local targeting was correct. The two near-boundary verdicts are not a reason to revert.

Transport to complex surveys, estimated or calibrated weights, clustering and other longitudinal
weight laws remains open. Karim (2026), arXiv:2606.30918v2, is adjacent point-treatment survey
theory. Landsiedel, Petersen and van der Laan (2026), arXiv:2607.02702, treat a different
two-stage outcome-subsampling estimator. Neither is a result for this exact construction.

`RM18-one-sided-bias` holds `double_robustness/outcome_correct` and
`double_robustness/treatment_correct` of `canonical-drtmle`, and
`double_robustness/treatment_correct` of `canonical-multi-arm-drtmle`. The reading opened
[RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle). The declaration names no other
owner, so this `id` stays the ledger owner of the three cells.

`RM18-ordinary-weighted` holds `interval_calibration/static__correctly_specified`,
`type_i_error/static__sharp_null`, and the four `targeting_necessity` cells of `weighted-ltmle`.
Three `targeting_necessity` cells pass their own rules and are red only through the family clause.
The static untargeted control fails its own discrimination rule. The calibration and sharp-null
rows are separate inferential boundary failures. They are not evidence against the pooled
cross-fitted update, because this ordinary study does not use that path.

`RM18-learner-weight-se` owns no red cell. Thirteen of 1,200 replicates in each arm report a
committed standard error above 1, up to 47,459.22. No property verdict reads those standard
errors, so the diagnostic changes no verdict and no estimator.

`RM18-boundary` holds the cells that sit at a finite-budget boundary.

| study | cells |
| --- | --- |
| `canonical-multi-arm-ctmle-selector` | `root_n_and_efficiency/n_500` |
| `canonical-drtmle` | `double_robust_contraction/treatment_correct_n1500` |
| `canonical-multi-arm-drtmle` | `double_robust_contraction/outcome_correct_n4000`, `interval_calibration/correctly_specified` and `root_n_and_efficiency/n_500` |
| `weighted-ltmle-crossfit` | `double_robustness/static__both_wrong`, and the paired `ey_regimen[always]` and `ey_regimen[treat then continue if l2 positive]` rows |

`RM18-comparator-density` holds the paired `ate_shift[+0.25 vs natural course]` row of
`shift-policies`. Its calibration leg fails with a 99% upper endpoint of 0.065856 against a margin
of 0.05, at a resolution of 0.045019. The conclusion is therefore `inconclusive`.

The `F19` acceptance does not require a negative paired standard-error deficit. Theorem 1 of
Benkeser, Cai and van der Laan, arXiv:1901.05056 v1, gives no first-order generated-design term
for one binary treatment-specific mean. The registered gate now requires each design's coverage and
standard-error intervals to fit inside their calibration bands. The point-treatment pair passes
that rule. The multi-arm pair passes its coverage band but fails because both standard-error
intervals cross the 1.07 upper bound. The paired difference is descriptive.

A future oracle comparison may use a two-sided equivalence margin or a predeclared sample-size
ladder that contracts the difference towards zero. A deliberately invalid generated design may
serve as a negative control; the learned design itself may not. The multi-arm `oracle_design`
SE-ratio interval runs 0.979701 to 1.115078, while its coverage interval runs 0.928173 to 0.968751.
The width of the SE interval describes the resolution of the instrument at the registered 800
replications; its coverage interval is already inside the 0.92 to 0.98 band.

(what-the-one-sided-reading-found)=
#### What the one-sided reading found

The reading asks whether R `drtmle` reproduces each finite-sample bias on the same rows. It ran
once, under a declaration written before the computation, and commit `7ac37dc` recorded it. Every
value below comes from `readings.csv` in
[`tests/diagnostics/rm18_one_sided_bias/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_one_sided_bias).
`tests/unit/test_rm18_one_sided_bias_diagnostic.py` rebuilds that file from the committed
artifacts. Each binary interval is a 99% Student interval over 800 primary replications at
n = 3,000.

| configuration | (i) `cleverly` bias | (ii) R `drtmle` bias | (iii) `cleverly` minus R | reading |
| --- | --- | --- | --- | --- |
| `outcome_correct` | 0.000105 to 0.003504 | 0.000204 to 0.003588 | -0.000295 to 0.000111 | `shared` |
| `treatment_correct` | 0.001284 to 0.004921 | 0.000277 to 0.003917 | 0.000068 to 0.001942 | `mixed` |
| `both_correct` | not read | not read | -0.000081 to 0.000078 | enters both readings |

The `both_correct` paired interval covers zero, so neither binary reading is `unresolved`. On
`outcome_correct`, the two implementations share the bias at this size. On `treatment_correct`,
the unadjusted paired comparison finds a between-implementation increment of 0.001005 against the
`cleverly` bias of 0.003102. The declaration adjusted for no multiplicity, so the signal is
exploratory.

At the Bonferroni level of 1 - 0.01/3 over the three paired intervals, the
`treatment_correct` interval runs from -0.000063 to 0.002073 and covers zero. It is a
supplementary row of `readings.csv`.
[RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle) gives the reading of a declared
design on fresh draws.

No R comparator fits the multi-arm `treatment_correct` configuration. The declared multi-arm
statistic (iv) is a Welch interval for the property-cell bias minus the primary bias at the same
size.

| statistic | rows | 99% interval |
| --- | --- | --- |
| (i) | the primary `ate[medium vs high]` rows at n = 2,000, both nuisances correct, 800 replications | -0.003016 to 0.001850 |
| the property-cell bias | `double_robustness/treatment_correct` at n = 2,000, 600 replications | 0.001469 to 0.007176 |
| (iv) | the property-cell bias minus (i), by Welch | 0.001160 to 0.008650 |

Statistic (iv) excludes zero, so the declared multi-arm reading is `finite-sample excess detected`.
`readings.csv` also carries rows with `scope` set to `supplementary`, which no rule reads.

| supplementary row | 99% interval, by Welch |
| --- | --- |
| the rung bias minus (i) | -0.002581 to 0.004917 |
| the property-cell bias minus the rung bias | -0.000296 to 0.007771 |
| both samples, 1,200 replications, minus (i) | -0.000123 to 0.006196 |

The independent `double_robust_contraction/treatment_correct_n2000` rung does not reproduce the
excess, and the pooled comparison covers zero. The overall multi-arm evidence is inconclusive. No
verdict, margin, budget or artifact moved. The three cells stay red under `reporting`, and the
ledger keeps `RM18-one-sided-bias` as their owner.

(what-design-sl-found)=
#### What design SL found

This subsection records the original results in [pull request 250](https://github.com/esbraun/cleverly-tmle/pull/250).
Its [immutable artifact snapshot](https://github.com/esbraun/cleverly-tmle/tree/17179f2fe7026c06d242dd1f29368725742a0e76/tests/canonical/multi_arm_drtmle) preserves these measurements.
[RM33](#rm33-reused-simulation-seeds) records the corrective seed repair. The method page describes the current evidence.

Design SL ran once, under its declaration in `ebd3d1b9` and the amendments in `7e65bea7`,
`03fe89f1`, `bbbb887e` and `985849c6`. Commit `985849c6` holds all of them.

The regeneration ran from the clean pushed commit `67deae8e` into a scratch directory. It started
at 14:36 UTC on 2026-09-28. It ran for 70,427 seconds and ended with exit code 0. The runtime was
Python 3.13.7 with SciPy 1.18.0, and `cleverly` came from the tree's `src`.
The [original run log](https://github.com/esbraun/cleverly-tmle/blob/17179f2fe7026c06d242dd1f29368725742a0e76/tests/canonical/multi_arm_drtmle/run.log)
records the commit, the versions, the thread limits and each file's SHA-256.

Each check that the declaration names held after the run.

| check | result |
| --- | --- |
| reference-phase artifacts | `replicates.csv.gz`, `summary.csv`, `equivalence.csv`, `performance-tests.csv`, `fit-diagnostics.csv` and `reference_sha256` are byte-identical to the committed files |
| committed property rows | each of the 26,600 committed rows reproduces as text. The one change is `requested_replicates`, from 600 to 73,000, on rows 0 to 599 of the six outer rungs |
| other cells | each row of `properties.csv` outside the three slope rows is byte-identical |
| rung-cost inputs | `tests/unit/test_rm18_rung_cost_diagnostic.py` passes on the committed `cost.csv`. Its inputs are read at the verdict budget of 600 |
| one-sided reading | `readings.csv` of `tests/diagnostics/rm18_one_sided_bias/` rebuilds byte for byte under Python 3.11.13 with SciPy 1.17.1 |
| manifest | it records commit `67deae8e` and `cleverly_worktree_clean: true` |

Each slope interval is the registered 99% percentile bootstrap. Each slope reads 146,600
replications: 73,000 at n = 2,000, 600 at n = 4,000 and 73,000 at n = 8,000. Each bias interval is
the 99% Student interval over the 73,000 replications of its rung, which are the slope's own
inputs. The reading rule reads the n = 2,000 interval. No rule reads the n = 8,000 interval.

| cell | slope, 99% interval | n = 2,000 bias, 99% interval | n = 8,000 bias, 99% interval | reading |
| --- | --- | --- | --- | --- |
| `rate_outcome_correct` | -0.8587, -1.0694 to -0.6652 | 0.001748, 0.001495 to 0.002001 | 0.000532, 0.000405 to 0.000658 | `contracts` |
| `rate_treatment_correct` | -0.8740, -1.0235 to -0.7403 | 0.002539, 0.002285 to 0.002793 | 0.000756, 0.000630 to 0.000882 | `contracts` |
| `rate_both_wrong`, the control | 0.0020, -0.0021 to 0.0061 | 0.049143, 0.048892 to 0.049394 | 0.049277, 0.049152 to 0.049402 | passes |

Both positive slope intervals lie below 0. Each bias interval at n = 2,000 and at n = 8,000
excludes 0. The declared reading of both arms is therefore `contracts`. The registered rule passes
both rate cells, and `CLAIMS` in `tests/studies/evidence/red_cells.py` no longer lists them. At
1,800 replications, the slope intervals ran -1.2342 to 3.7735 and -3.2412 to 3.3387.

The declared sizing rule set the outer-rung budget, and that rule reads no verdict. The
`RM18-binary-slope` rung design used the same rule. The budget is therefore not a budget raised
after a verdict was seen.

The declaration names "the n = 2,000 bias interval", and it does not name the budget of that
interval. This pull request resolved the budget after the run, when both readings were known. It
first chose the 600 replications of the verdict budget. After review, it chose the slope's own
73,000 rows. The table gives the reasons for the second choice.

| kind | reason |
| --- | --- |
| text | the declaration explains the noise-floor reading by the slope's inputs: "The absolute bias estimate then tracks its standard error, which falls as `n^-1/2`." Those inputs are the rung means over 73,000 rows |
| text | the declaration keeps 600 replications for "each rung's coverage verdict" (R1), and the extra outer rows serve the slope |
| text | rows 0 to 599 of each rung, and their n = 2,000 bias intervals, were published before the declaration. At that budget, the run could not decide the bias condition |
| data | at these budgets, a zero bias does not give a slope interval below 0. A review computation redrew each rung at zero bias with the committed spread, and every slope interval covered 0: about -3.9 to 2.7 and -4.3 to 2.0 |

Rows 0 to 599 of each rung were committed before the declaration. Over those rows, the
n = 2,000 bias intervals run -0.001651 to 0.003905 and -0.002274 to 0.003444. Both cover 0, so a
reading at that budget is `contracts at the noise floor` for both arms. These intervals are
supplementary. Both labels take the same route in the ledger.
`tests/unit/test_rm18_slopes_reading.py` asserts the declared label and computes both intervals.

The control's slope interval covers 0, so the control passes its registered rule. The raised
control budget carried a declared risk. An `O(1/n)` transient in the both-wrong bias could place
the interval below 0 at 73,000 replications. Then, as the declaration states, "no `contracts`
reading of a positive arm counts as evidence of contraction". The interval does not lie below 0,
so that consequence does not apply.

The declared prediction for the control also held. Its slope half-width is 0.004123, against a
prediction of about 0.0040 within 15%. At 600 replications per rung it was 0.043840.

What the readings do not show:

- A slope interval below 0 shows the direction of the change from n = 2,000 to 8,000. No declared
  rule reads the value of the slope, so the readings do not give the order of the bias.
- They do not re-read a rung's coverage verdict. Each rung keeps its verdict at 600 replications,
  and `outcome_correct_n4000` stays red under `RM18-boundary`.
- They do not re-read `double_robustness/treatment_correct`, which stays red under
  `RM18-one-sided-bias`. No rule reads the ladder's bias against that cell's margin.
- They cover one law, one learner configuration and three sizes. The source theorem is a
  binary-treatment result, and this study measures the armwise extension.

#### Deferred findings and review resolutions

| finding | why it was deferred | what it needs |
| --- | --- | --- |
| `simulated_confounding` cannot perturb an outcome on a fit that declares `q_bounds`. `_gaussian_outcome` subtracts the strength times a standard normal latent value from the outcome, so every nonzero outcome strength sends the perturbed outcome outside the declared support and the refit refuses it | a cross-fitted continuous fit must declare `q_bounds`, so the outcome axis of the surface is unavailable to every such fit. No perturbation on the declared scale exists to put in its place | [F23](#f23-simulated-confounding-on-a-declared-outcome-scale) |
| the static untargeted control of the `weighted-ltmle` `targeting_necessity` family. OW-C's population reading is `control underpowered by design`: at the fresh SD, the registered design of 1,200 replicates discriminates it with a probability of about 0.72 by the normal approximation, and that probability crosses 0.80 inside the 99% interval of the SD | the declared rule R1 refuses a change at a re-read. The family shares its draws with its positive arm, so its size cannot change without a re-declaration of the positive arm | a re-declaration of the control law before any regeneration of the study, as the control precedent at `69de6f8` allows |
| the `lmtp` standard-error convention of the weighted paired mean rows. BD-P reads `ey_regimen[always]` and `ey_regimen[treat then continue if l2 positive]` as `comparator SE convention` | RM18 records a comparator convention and does not re-declare it. The study page states the convention | a re-declaration of the comparator convention before a future regeneration of `weighted-ltmle-crossfit`, if that regeneration chooses one |

### RM19. One-sided robustness bias increment in DR-TMLE

RM18's one-sided robustness reading opened this row. On the binary `treatment_correct`
configuration of `canonical-drtmle`, the treatment mechanism is correct and the outcome
regression is not. On the 800 committed primary replications at n = 3,000, the unadjusted paired
99% interval of `cleverly` minus R `drtmle` ran from 0.000068 to 0.001942. The Bonferroni interval
covered zero. "[What the one-sided reading found](#what-the-one-sided-reading-found)" gives each
interval.

Pull request 246 delivered this row. The table gives what shipped.

Commit `b7ca9460` holds the declaration, "The localization design, declared before it runs", with
its amendments. They are
"The rule by round and conditioning", "The amendment after the harness review" and "The amendment
after the Part A validation failure". The subsection "Reads made before this declaration" lists
every read before each run. Read them with `git show b7ca9460:docs/roadmap.md`. Commit `b51447cb`
holds the Part A record under the first validation rule.

| part | what shipped |
| --- | --- |
| diagnostic | [`tests/diagnostics/rm19_one_sided_increment/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm19_one_sided_increment) holds a transcription of the binary loop of R `drtmle` 1.1.2, the harness, the per-draw rows, the validation records and the readings. Its README gives each run, the checks V1 to V5 and the bias of each arm. `tests/unit/test_rm19_one_sided_increment_diagnostic.py` rebuilds each reading from the committed rows |
| primary reading | Part B, 2,000 fresh draws at n = 3,000: `no increment at the declared resolution` |
| localization | Part B: `nothing to localize`. Part A, on the committed draws: `committed-draw attribution: shared by J and G`, which confirms nothing |
| Part C | the Part C rule labels `sqrt(n)` times `C - T(0, 0, 0, 0)` `increment` at n = 1,500, and gives no label at 3,000 and 6,000. No route reads a Part C label |
| route | record, and close. No estimator changed. The three one-sided cells stay red under `reporting`, and `RM18-one-sided-bias` stays their owner |
| documentation | "[The update order](technical-reference/dr-tmle/targeting.md#the-update-order)" names the prime step that R `drtmle` does not take. [Supported estimands](technical-reference/dr-tmle/supported-estimands.md) cites the J main effect and Part C. The [canonical DR-TMLE study page](technical-reference/method-evidence/canonical-dr-tmle.md) cites the reading. The `solve_with_reduction` docstring gives the settings of the registered R run. Theorem 3 of van der Laan (2014) is cited through Benkeser et al. (2017), Section 3.1 |

(what-the-localization-design-found)=
#### What the localization design found

Every value below comes from a `*-reading.csv` file of the diagnostic directory. The arm `C` is
the registered `cleverly` fit. The arm `T(j, p, s, k)` is the transcription with the factors J, P,
S and K at the named levels. The factors are the tilt geometry, the priming step, the exit and the
solver numerics. `T(0, 0, 0, 0)` is R. The arm `T0+G` is `T(0, 0, 0, 0)` without the step guards
of R.

Each interval is a Student interval over draws. Every check V1 to V5 of the harness holds, and
the diagnostic README gives each one.

Each declared part ran once. The step AV, which recomputes V2 for Part A, ran twice, and the
README gives the reason. G entered the family in both parts, so each main effect has the
Bonferroni level 1 - 0.01/5 = 0.998. The table gives each declared reading verbatim.

| part | statistic | point | interval | reading |
| --- | --- | ---: | --- | --- |
| B | `C - T(0, 0, 0, 0)` | 0.000566 | -0.000023 to 0.001155, half-width 0.000589 | `no increment at the declared resolution` |
| B | the bracket, `T(1, 1, 1, 1) - C` | 0 | 0 to 0 | `bracket holds` |
| B | guard events of `T(0, 0, 0, 0)` | 531 | | `G in the family` |
| B | main effect J | 0.001275 | 0.000833 to 0.001718 | level 0.998 |
| B | main effects P, S and K | -0.000534, 0.000056, 0.000003 | each covers 0 | level 0.998 |
| B | main effect G | 0.034474 | 0.023561 to 0.045387 | level 0.998 |
| B | localization | | | `nothing to localize` |
| A | `C - T(0, 0, 0, 0)` | 0.001005 | 0.000068 to 0.001942 | `committed-draw attribution: increment confirmed` |
| A | the bracket | 0 | 0 to 0 | `committed-draw attribution: bracket holds` |
| A | main effects J and G | 0.001224 and 0.029386 | 0.000614 to 0.001834, and 0.011343 to 0.047429 | level 0.998. P, S and K cover 0 |
| A | localization | | | `committed-draw attribution: shared by J and G` |
| A, `outcome_correct` | `T(1, 0, 0, 0) - T(0, 0, 0, 0)` | -0.000068 | -0.000272 to 0.000135. The SD is 0.002228, against a bound of 0.002223 | `P-ctrl fails` |
| A, `both_correct` | the same | 0.000015 | -0.000055 to 0.000085. The SD is 0.000763, against a bound of 0.000869 | `P-ctrl holds` |
| C, n = 1,500 | `sqrt(n)` times `C - T(0, 0, 0, 0)` | 0.0789 | 0.0389 to 0.1189, at the level 1 - 0.01/3 | `increment` |
| C, n = 3,000 | the same, from the Part B rows | 0.0310 | -0.0057 to 0.0678 | no label |
| C, n = 6,000 | the same | 0.0053 | -0.0274 to 0.0381 | no label |

Each of the six two-way interactions covers 0 in both parts. No rule reads them.

##### What each reading means

The declared rules decide each meaning. The last column states what the reading does not show.
Hn is the declared descriptive hypothesis of Part C: `sqrt(n)` times the mean paired difference is
the same at n = 1,500, 3,000 and 6,000.

| reading | meaning under the declared rules | what it does not show |
| --- | --- | --- |
| `no increment at the declared resolution` | the fresh-draw interval covers 0, and its half-width of 0.000589 is below the declared 0.000598. The declared route reads: "record. The committed signal was a Monte Carlo excursion, and RM19 closes". The excursion clause is the wording of the declared route. The fresh-draw readings neither confirm nor rule out a small increment | that the increment is 0. The interval also covers the committed point of 0.001005. It excludes only a mean increment above 0.001155 at n = 3,000. Both `C` and `T(0, 0, 0, 0)` keep a bias above 0 on the fresh draws |
| `nothing to localize`, beside a J and a G main effect that exclude 0 | the localization table is read from the top. The bracket holds, so `unexplained` does not apply. The next row, `nothing to localize`, needs only the primary reading. The factorial is reported, and no label and no route apply to J or G | that J and G have no effect. The primary difference compares the two corners of the design. In a 2^4 design, the corner difference equals the sum of the four main effects and the four three-way contrasts. So J can exclude 0 while the corner interval covers 0. The P main effect has the opposite sign |
| the J main effect | averaged over the eight settings of P, S and K, the `cleverly` level of J raises `ate` by 0.000833 to 0.001718 on fresh draws | the effect of the geometry alone. At K = 0, J also changes three settings of the mechanism step of R: the `glm` tolerance, the offset trim and the guard. The harness README lists them |
| the G main effect | G is `T0+G` minus `T(0, 0, 0, 0)`, so it reads R against R without its guards. `T(0, 0, 0, 0)` records 531 guard events on 232 of the 2,000 fresh draws. On those 232 draws, removing the guards moves `ate` by 0.30 on average. The unguarded loop often fails to converge: it reaches the cap on 303 draws, against 95 for `T(0, 0, 0, 0)` | a difference between `cleverly` and R. `cleverly` takes no guard, and at K = 1 no guard of R acts |
| `committed-draw attribution: increment confirmed` and `shared by J and G` | Part A reads the 800 draws that produced the RM18 signal. V1 and V2 hold, so it reproduces the committed interval. The declaration states that Part A confirms nothing | evidence for the increment. Part A and Part B read different draws, and each interval covers the point estimate of the other |
| `P-ctrl fails` on `outcome_correct` | P-ctrl is a prediction that the declaration at `b7ca9460` states. It is not a control of factor P. It reads the J contrast `T(1, 0, 0, 0) - T(0, 0, 0, 0)` on the two control scenarios. On `outcome_correct`, the mean interval covers 0, and the SD exceeds its bound by 0.24%. The standard error of an SD from 800 draws is about 2.5% of the SD, so the SD equals its bound within sampling error. The declaration gives neither prediction a route | a mean effect of J on `outcome_correct`, or a defect |
| Part C | the point estimates of `sqrt(n)` times the increment fall with n: 0.0789, 0.0310 and 0.0053. Only the n = 1,500 interval lies above 0. There the increment is 0.0020 on the raw scale, from 0.0010 to 0.0031. The n = 1,500 and n = 6,000 intervals do not overlap: 0.0389 against 0.0381 | a proof of a rate. Three sizes cannot identify a rate. The adjacent intervals overlap. The intervals at 3,000 and 6,000 cover 0, and they reach 0.0678 and 0.0381. No rule reads Hn, and a gap between two intervals is not a declared test. Part C fits `C` and `T(0, 0, 0, 0)` alone, so it cannot name the factor that changes with n |

So the Part C intervals agree with an increment that vanishes faster than `1/sqrt(n)`, and they
do not prove it.

Benkeser, Carone, van der Laan and Gilbert (2017), Section 3.2, Theorem 1, has three conditions.
First, one nuisance limit is correct. Second, the score equations (5) are $o_P(n^{-1/2})$. Third,
the second-order terms of Appendix B are $o_P(n^{-1/2})$. The estimator is then asymptotically
linear with the influence function $D^{*\#}(Q, g)$, which depends on the nuisance limits and not
on the route. Appendix B gives one sufficient condition: rates of $o_P(n^{-1/4})$ in the
$L^2(P_0)$ norm.

If both `C` and `T(0, 0, 0, 0)` met these conditions, their difference would be
$o_P(n^{-1/2})$. That step follows from the definition of asymptotic linearity. Two statements
that the step would need have no read source.

| statement | status |
| --- | --- |
| the registered GLM reduced regressions meet the conditions of Theorem 1 on this configuration. A derivation before the declaration found that a misspecified `Qr` leaves a term of order $n^{-1/2}$ in general | an unsourced derivation. No rule depends on it |
| the two tilt geometries differ at second order. [Supported estimands](technical-reference/dr-tmle/supported-estimands.md) stated it before this row, and it now labels the statement unsourced | an unsourced derivation. Part C reads the whole increment and not J alone, so it does not test this statement |

##### The route and the acceptance

The primary reading takes the declared route of `no increment at the declared resolution`:
record, and RM19 closes.

| acceptance | state |
| --- | --- |
| the declared design runs once and publishes its reading, whichever way the reading falls | met. Parts A, B and C and the check V4 each ran once under the declaration as amended. The step AV ran twice. Part A read `harness not validated, no reading` under the first rule, and `b51447cb` keeps that record |
| the three one-sided cells stay red under `reporting`, `RM18-one-sided-bias` stays their owner, and no margin, budget or law moves | met. The ledger and every registered artifact are unchanged |
| a reading that names a defect in `cleverly` opens a separate fix | no reading names a defect, so no fix row opens. V5 reproduces `C` exactly from the choices J, P, S and K on 7,200 committed rows, and the bracket is 0 on the fresh draws. Each choice has a documented source. J is in [supported estimands](technical-reference/dr-tmle/supported-estimands.md). P is in the `solve_with_reduction` docstring (`prime:`) and in Step 2 of Benkeser et al. (2017), Section 3.2. S is in "[The update order](technical-reference/dr-tmle/targeting.md#the-update-order)". K is `submodel_alpha` of `Targeting` and "[The bound-inactive scope](technical-reference/dr-tmle/targeting.md#the-bound-inactive-scope)". "The update order" said that the `"drtmle"` order follows the R package and did not name the prime. That was an error in the text, and this row corrects it |
| the declaration states whether its design can read the multi-arm excess | met. The design cannot read it, because no validated R comparator fits that configuration. On more than two arms, `cleverly` uses the armwise tilt of R, so factor J does not exist there. Part B does not read `attributed to J`, so the source stays open. The multi-arm cell keeps its owner |

### RM20. Intervals outside every claimed contract

Before this row shipped, each surface in the table below reported `ci`, `pvalue`, and `std_error`
under the `influence_curve` status. No claim covered the interval, or no audit had read a source for
it. Pull request 224 delivered one decision for each surface. The table gives each decision, the
control that keeps its interval, and the item that holds the reopen route.

| surface | decision that shipped | control that keeps its interval | reopen route |
| --- | --- | --- | --- |
| DR-TMLE with `weights_estimated=True` | the status `estimated_weight_plugin`, when `guard` is not empty and the weights vary | the same weights declared fixed. Also `guard=()` with the weights declared estimated, because that setting fits the ordinary TMLE, whose interval conditions on the weights. Also constant weights declared estimated, which fit the unweighted estimator (commit fbee899). Also the ordinary `TMLE` | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| outcome-adaptive C-TMLE, with complete or missing outcomes | the status `generated_design_plugin`, on every `strategy="oat"` fit | the ordinary `TMLE` on the same law | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| cross-fitted clustered `TMLE` and `DRTMLE` at unequal cluster sizes: row counts, or weight mass on a weighted fit, overall or within a reported stratum | the status `unequal_cluster_plugin`. It includes `cv_evaluation=True` and fold targeting | equal cluster sizes and masses within every reported stratum, cross-fitted, at 40 clusters. Also the same unequal clusters, fitted in sample, on `TMLE` and `DRTMLE` | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| clustered `TMLE` and `DRTMLE` with fewer than 40 positive-mass clusters in the fit or in one reported baseline stratum, in sample and cross-fitted | the status `few_cluster_plugin`. `FEW_CLUSTER_THRESHOLD` in `src/cleverly/_inference_status.py` holds the threshold of 40 | a fit with 40 contributing clusters, which pins `<` against `<=`. Also 80 clusters in two strata of 40 | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| shift, incremental, regime, MSM, and controlled-direct-effect targets, cross-fitted with `delta=` | a pre-fit `CapabilityError` that names F21. Its remedy is the in-sample fit | each composition in sample. Also the cross-fitted `ate` with `delta=`, which reaches the arm-indexed contract, and `par` with `delta=`, which keeps its F20 refusal | [F21](#f21-other-missing-outcome-cv-tmle-variants) |

A status keeps the point estimate. `ci`, `pvalue`, and `std_error` raise `CapabilityError` with
the reason of the status. `plugin_std_error` and `plugin_interval` keep the diagnostic, as RM12 set
up. `summary()` prints the label and the reason of the status, the nuisance note marks it, and the
E-value row reads `unavailable`. `variable_importance()` refuses before its first fit.

The estimated-weight decision is a status and not a refusal. The `weights_estimated` flag changes
no number (`src/cleverly/data/weighting.py`), so a caller could drop the flag to bypass a refusal.

`FEW_CLUSTER_THRESHOLD` follows Nugent, Marquez, Charlebois, Abbott and Balzer (2024), *Biostatistics*
25(3):599-616, Section 2.2. That section recommends a Student's $t$ reference with $J - 2$ degrees
of freedom below 40 clusters. The package keeps its normal reference. F22 holds the $t$-reference
claim that would reopen it.

One fit has one status. When more than one surface applies, the fit takes the first status in the
table below. `NON_INFERENTIAL` in `src/cleverly/_inference_status.py` holds the order, and
`precedent_status` applies it. The rule matches the ordered refusals of the arm-indexed
missing-outcome contract. The docstring of `TMLE._resolve_arm_indexed_missing_contract` says that
"a fit that breaks several rules receives the first one"
(`src/cleverly/estimators/tmle.py:1522-1523` at commit 75e86be).

| order | status | premise that fails |
| ---: | --- | --- |
| 1 | `working_mechanism_plugin` | no result shows that the reported curve is the influence curve of the estimator |
| 2 | `generated_design_plugin` | the same premise |
| 3 | `estimated_weight_plugin` | the same premise as order 1 |
| 4 | `unequal_cluster_plugin` | the cross-fitted interval lacks validation at unequal cluster sizes or masses |
| 5 | `few_cluster_plugin` | no read source supports the reference distribution |

RM20 shipped all five orders. `PRECEDENCE` in `tests/unit/test_inference_status_registry.py` pins
`NON_INFERENTIAL` to this order.

Among the five statuses, only three pairs can meet. On DR-TMLE, the estimated-weight status meets
each clustered status. On TMLE and DR-TMLE, the two clustered statuses meet each other. C-TMLE
refuses `id=` at every setting.

Each status reads the estimator configuration and the prepared data alone, so it is known before a
learner runs. No status reads a fitted quantity. The estimator stamps the status after nuisance
fitting.

Two neighboring surfaces keep their intervals without a registered study. This row records each
one and adds no row for it.

| surface | why it keeps its interval | evidence |
| --- | --- | --- |
| in-sample clustered point-treatment fit with 40 or more clusters, in the fit and in each reported stratum | Benitez et al. (2023), Section 3.2.1, support the cluster-sum aggregation for a row-weighted estimand | a recorded source. The registered clustered study is cross-fitted, so no registered study covers the in-sample fit |
| in-sample shift and incremental fits with `delta=` | exact-law instruments check the influence curve | `tests/unit/test_influence_gateaux_shift_cde.py` and `tests/unit/test_influence_gateaux_ipsi_mar.py`, which the [evidence manifest](technical-reference/evidence.md) lists for `ey_shift` and `ey_ipsi`. No registered study covers either fit |

The witnesses are in `tests/unit/test_estimated_weight_status.py`,
`tests/unit/test_outcome_adaptive_status.py`, `tests/unit/test_cluster_status.py`,
`tests/unit/test_cross_fitted_missing_off_contract.py` and
`tests/unit/test_inference_status_reach.py`. A committed mutation restores the `influence_curve`
status on each surface, and its test fails. [Inference status](technical-reference/inference.md#inference-status)
lists each status.

### RM21. E-value on a controlled-direct-effect fit

`_select_evalue` in `src/cleverly/sensitivity/evalue.py` returned an E-value on a fit with an
intermediate variable. The Gaussian branch divided by the observed sd(Y), which supplies no
source-backed confounding bound for the controlled direct effect. Identification also assumes no
unmeasured confounding of the intermediate variable and the outcome.

Pull request 229 delivered this row. The predicate `declares_intermediate(result)` in
`src/cleverly/estimators/direct_effect.py` decides the fit. `_select_evalue` raises
`_DIRECT_EFFECT_REFUSAL` with the `unavailable` status before any branch, so the capability row
reads `unavailable`. The reason cites [F25](#f25-e-value-for-a-controlled-direct-effect), which
holds the missing result. `tests/unit/test_evalue_direct_effect_refusals.py` holds the witnesses
and the mutations. The [E-value paths](technical-reference/validation-methods.md#e-value) hold the
table of read sources.

### RM22. Standard error of the omitted-variable bound

For the ATT and the ATC, the $\nu^2$ estimate divides by the share $p = P(A = c)$ of the
conditioning arm. Its influence curve has the term $-2 \nu^2 (1\{A = c\} - p) / p$, and the code
omitted it. The standard error of the bound omitted it as well.

Pull request 230 delivered this row. The table gives what shipped.

| part | what shipped |
| --- | --- |
| share term | `_conditioning_share_influence` in `src/cleverly/sensitivity/omitted_variable.py`. `_elements_for` adds the term under the doubly robust estimator for every parameter that conditions on an arm. The point $\nu^2$ does not move |
| plug-in limits | `SensitivityBounds` records `nu2_estimator`. Under `"plugin"` the six limit accessors raise `CapabilityError` with the reason that cites [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) |
| witnesses | `tests/unit/test_omitted_variable_standard_error.py` compares the curve of $\hat\nu^2$ with its Gateaux derivative, row by row |
| study | the registered [omitted-variable bound study](technical-reference/method-evidence/omitted-variable-bound-standard-error.md) reads the standard-error ratio of the doubly robust ATT limits |

The RM22 plan in the record at commit `dea3297e` declares that study, its margins and its budget.
The [standard error of the omitted-variable bound](technical-reference/validation-methods.md#standard-error-of-the-omitted-variable-bound)
states the method.

### RM23. Capability rows that read available and then refuse

The assessment declares each operation in a capability row before any call. Some rows read
`available`, and the call then refused or raised. `replayability()` also reported both replay
slots as true without the checks that `retarget()` and `refit()` run.

Pull request 231 delivered this row. Each row now resolves from the predicate that its call uses:
`fit_wide_tilt_refusal`, `truncation_refusal`, `refute_refusal`, `benchmark_refusal`,
`simulated_confounding_refusal`, and the `bound_parameters` rule. The two replay slots read
`_refit_configuration_refusal`, which runs the checks of `refit()` without a fit.
`tests/unit/test_capability_row_sweep.py` fits 30 kinds of fit and asserts that each row that a
request resolves as available answers it. The delivery routed its other findings to RM16, RM24
and RM32.

### RM24. Refusals after the nuisance fit

Stratified requests were refused only in the targeting loop. An incremental fit or a log- or
logit-link MSM fit with `strata=` raised `NotImplementedError` after two learner fits. A `DRTMLE`
fit with `strata=` at a non-empty `guard` raised after eight. `DRTMLE` with
`targeting="one_step"` and `reduced_crossfit="nested"` raised after 42. `CausalStudy.identify`
admitted the stratified incremental and MSM estimands. Other composition refusals raised
`ValueError` or `NotImplementedError` before any learner, and the refit replay slot read both as
refusals.

Pull request 241 delivered this row. The table gives what shipped. Commit `e4bb35c3` holds the
probe table and the corrections. Read it with `git show e4bb35c3:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| stratified targeting | `refuse_stratified_targeting` in `src/cleverly/estimators/tmle.py` runs in `_resolve_estimands_for_data`, in `DRTMLE._check_drtmle` and in `_identify_point` in `src/cleverly/study.py`. It raises `CapabilityError` before any learner and cites [X8](#x8-stratified-incremental-and-msm-targeting). The `DRTMLE` message names `guard=()`, which is the ordinary TMLE. The targeting-loop guard stays as a backstop |
| identification | `CausalStudy.identify` refuses `IncrementalMean`, `IncrementalEffect`, and an `MSMProjection` with a link other than the identity or with a continuous dose, on a design with `strata=`. `estimate` refuses the `DRTMLE` case before any learner |
| one-step nested | `DRTMLE._check_drtmle` refuses it at a non-empty `guard`, before any learner, on a copied estimator as well. The `solve_submodel` guard stays as a backstop |
| types | these refusals raise `CapabilityError` before any learner: each composition refusal of `CTMLE` and `DRTMLE`, `reduced.refuse_unsupported`, the incremental-intermediate refusal, the fold-targeting refusal of `incremental=`, and the refusals of strata or `cv_evaluation=True` in the fit. Other constructor checks of `TMLE._validate_settings` keep `ValueError`, among them `targeting="one_step"` with `fluctuation="linear"` and the fold-policy reason. A copied estimator meets the fold-policy reason as `CapabilityError`. The backstops in `build_submodel`, `_check_companion` and `_solve_missing_outcome_reduction`, which no fit reaches, keep their types. The message of commit `8d48589d` says "every other composition refusal". This row gives the exceptions. The incremental-intermediate refusal cites [F6](#f6-mnar-and-incremental-intermediate-compositions) |
| replay slot | `_refit_configuration_refusal` reads each `ValueError` of the chain as a refusal. Any other exception propagates |

`tests/unit/test_refusals_before_the_nuisance_fit.py` holds the witnesses and five mutation
controls.

### RM25. Declared stochastic regime densities

A `density_fn` receives the covariate frame, and it can close over any estimate. For the
population-law target $\psi(P;q_\delta(g_P))$, the odds-tilt density depends on $P$ through its
treatment mechanism. The regime curve has no term for that derivative. A realized learned density
$\hat q$ instead defines a data-adaptive target $\psi(P;\hat q)$. Same-sample inference for that
target needs separate conditions that this API does not check.

Pull request 225 delivered this row. The field `Stochastic.density_kind` takes `"known"`,
`"estimated"`, or `None`, and its default is `None`. `cleverly.interventions.base.refuse_regime_densities`
refuses an undeclared or estimated density with `CapabilityError`. `Stochastic.__post_init__`,
`TMLE._resolve_estimands_for_data`, `TMLE._retarget_detailed` and `RegimeSet.evaluate` call it
before any learner. The estimated text sends a population odds tilt to `TMLE(incremental=...)`.
`FunctionDeclaration` in `cleverly._declarations` holds the three-state check that RM13, RM25, RM27
and RM28 share. `tests/unit/test_stochastic_regime_densities.py` holds the witnesses.

### RM26. Longitudinal clustered intervals at few clusters

[RM20](#rm20-intervals-outside-every-claimed-contract) gives a point-treatment clustered fit with
fewer than 40 clusters the `few_cluster_plugin` status. An in-sample `LTMLE` fit with `id=`
reported a normal-reference Wald interval and no status.

Pull request 226 delivered this row. `_inference_status(data, folds)` in
`cleverly.longitudinal.estimator` calls `cluster_inference_status` on the prepared cluster labels
and weights. An `LTMLE` fit with fewer than 40 positive-mass clusters takes `few_cluster_plugin`.
`summary()`, `curve()`, `incidence_total()` and the simultaneous bands follow the status.
`tests/unit/test_longitudinal_cluster_status.py` holds the witnesses.

### RM27. Declared MSM design functions

A `design` callable receives an arm label and a covariate frame, and it can close over any
statistic of the sample. A design that centres a covariate at its sample mean is a common example.
The projection is then a functional of $P$ through the design, and the reported influence curve
omits that pathwise derivative. The published MSM derivations cover a user-supplied working design.

Pull request 227 delivered this row. The field `MSM.design_kind` takes `"known"`, `"estimated"`, or
`None`, and its default is `None`. `cleverly.msm.refuse_msm_functions` checks the design and then
the weight. An undeclared design raises `CapabilityError` and asks for `design_kind="known"`. An
estimated design raises `CapabilityError` and names the pathwise derivative through the sample
statistic. A model whose design has the exact type `_LinearDesign`, which `MSM.linear` builds,
reads as known. `tests/unit/test_msm_design_declaration.py` holds the witnesses.

### RM28. Declared densities of user-written interventions

`Intervention.density`, `Rule.rule`, and callable nodes of `DynamicRegimen` all admit functions
learned from the analysis sample. A realized learned policy has a different target from a
population-indexed policy. The current regime paths implement no learned-policy contract, and
[RM30](#rm30-learned-policy-value-evaluation) records the published follow-up. The
[source audit](references.md#point-treatment-and-stochastic-interventions) records these results.

The decision is to require a known-function declaration for all three surfaces. A function trained
independently of the analysis sample may be declared fixed for an evaluation conditional on that
training. A function learned from the analysis sample is refused by these generic paths. This is a
conservative API decision. It does not claim that every learned rule's fixed-rule influence curve
is mathematically wrong. The declaration remains user-supplied because code cannot inspect a
closure.

Pull request 228 delivered this row. The table gives what shipped.

| part | what shipped |
| --- | --- |
| declarations | `Rule.rule_kind` and `DynamicRegimen.rule_kind` take `"known"`, `"estimated"`, or `None`. `Intervention` gains a read-only `density_kind`. `_RULE_DECLARATION` and `_INTERVENTION_DECLARATION` in `src/cleverly/interventions/base.py` hold the refusal texts |
| point check | `refuse_regime_densities` checks each item before any learner |
| longitudinal check | `refuse_regimen_rules` in `cleverly.longitudinal.regimen` reads the raw `regimens=` value. A callable written inline in a `regimens=` mapping carries no declaration, and the fit refuses it |
| witnesses | `tests/unit/test_rule_and_intervention_declarations.py` and `tests/unit/test_regimen_rule_declarations.py`. `tests/unit/_policy_declaration_support.py` holds the exact-law values that the plan in the record at commit `dea3297e` declares |

### RM30. Learned-policy value evaluation

The fixed-rule paths refuse a `Rule` or a `DynamicRegimen` that is learned from the analysis
sample, and no path estimated the value of a learned rule.
[RM28](#rm28-declared-densities-of-user-written-interventions) holds that refusal. This row adds
a typed target for the value of rules that are learned inside the training folds. It does not
read a learned function as a declared fixed rule.

Pull request 247 (RM30 A), pull request 248 (RM30 B) and pull request 249 (RM30 C) delivered this
row. Pull requests 248 and 249 merge together, so no release carries the interval before its
study. The table gives what shipped. Commit `edc70d7f` holds the contract, the sources and their
locators, the refusal order, the declarations of both studies and the reads made before them.
Read it with `git show edc70d7f:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| target | `LearnedRuleValue` and `TMLE(learned_rule=LearnedRule())` estimate the average, over the outer folds, of the value of the plug-in rule that each fold's training rows learn. Van der Laan and Luedtke (2015), Section 7 and Appendix B, define the target and its CV-TMLE. The `learned_rule` parameter axis reuses the regime fluctuation |
| contract | [Learned rules](technical-reference/point-treatment-tmle.md#learned-rules) gives the algorithm, the conditions C1 to C6 and the refusal order. `refuse_learned_rule_composition` in `src/cleverly/interventions/learned.py` refuses every other composition before any learner, and each refusal cites [X11](#x11-learned-policy-follow-ups), [F27](#f27-learned-policy-value-outside-the-published-conditions), F17 or F21 |
| record | `LearnedRuleRecord` in `result.extra["learned_rule"]` holds the fold sizes, the fold estimates, the treated shares and the blip quantiles. It survives `save` and `cleverly.load` |
| RM28 text | `_ESTIMATED_RULE` names `LearnedRuleValue` for a point treatment and X11 for a longitudinal regimen. The refusal is unchanged |
| evidence framework | `StudyRecord.truth_varies_by_replicate` reads every statistic on the error. [A truth that varies by replication](development/method-benchmarking.md#a-truth-that-varies-by-replication) states the rule. A study with no property cell registers as a reporting study and takes no grid row |
| gated study | the [gated study](technical-reference/method-evidence/learned-rule-cvtmle.md) passes every primary test and every property cell. Coverage is 0.9465 at `non_exceptional` and 0.9413 at `misspecified_limit`, over 6,000 replications at n = 2,000 |
| boundary study | the [boundary study](technical-reference/method-evidence/learned-rule-cvtmle-boundary.md) reads `under-covers at the exceptional law` and `under-covers at the weak_blip law`. Coverage is 0.8938 and 0.8948, and the SE ratio is 0.8211 and 0.8314. F27 owns both red rows |

The unit witnesses are `test_learned_rule_fold_locality.py`, `test_learned_rule_targeting.py`,
`test_learned_rule_influence.py`, `test_learned_rule_variance.py`,
`test_learned_rule_refusals.py` and `test_learned_rule_persistence.py` in `tests/unit/`.
`tests/unit/test_per_replicate_truth.py` holds the witness of the evidence framework.

### RM32. Continuous-dose MSM fit with missing outcomes

An in-sample `TMLE` fit of a continuous-dose MSM with `delta=` raised `ValueError` "need at least
one array to concatenate" from `_mechanism_columns` in `src/cleverly/estimators/_nuisance.py`. It
raised after two learner fits. `CausalStudy.estimate` raised the same error, and a fit with
`intermediate=` failed at the same place. The cross-fitted fit met the F21 refusal, whose remedy
was the in-sample fit.

Pull request 240 delivered this row. The table gives what shipped. Commit `e6111a2b` holds the
probe table and the corrections. Read it with `git show e6111a2b:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| estimator | `_resolve_estimands_for_data` runs `refuse_continuous_msm_mechanisms` from `src/cleverly/msm.py` on each MSM fit. A continuous dose with a missing outcome or an intermediate variable raises `CapabilityError` before any learner, at every `cross_fit` setting. It runs before the F21 refusal |
| identification | `_identify_point` in `src/cleverly/study.py` runs the same check on `MSMProjection` |
| message | it names each mechanism that the clever covariate would need at each grid dose, and it cites [X10](#x10-continuous-dose-msm-with-a-second-mechanism). It names no remedy, because a shift and an arm-coded MSM estimate different parameters |
| rule | the check keys on a missing outcome. A declared indicator with every outcome observed keeps its fit |
| suggestion | the `DataError` of `TMLE._check_shifts` offers the MSM only with every outcome observed and no `intermediate=` |

`tests/unit/test_continuous_msm_mechanism_refusals.py` holds the witnesses and three mutation
controls. The `build_submodel` guard in `src/cleverly/estimators/targeting.py` stays as a backstop
for its internal callers.

### RM33. Reused simulation seeds

The review scope is September 29, 2026, in America/Los_Angeles. It contains merged pull request 250.
Its expanded outer rungs contain 232 reused draws in 231 duplicate-seed groups.
The six rungs contain 438,000 rows. These counts do not establish a changed slope verdict.

`CoverageStudy.run` reads successive 32-bit words from one `SeedSequence` as replication seeds.
Repeated words draw identical samples. Different rung roots also produce overlapping sample seeds.
The slope bootstrap treats each recorded row as a separate observation.

NumPy documents [spawning and independent streams](https://numpy.org/doc/stable/reference/random/parallel.html).
Distinct PRNG inputs are the repair requirement. A seed audit does not prove mathematical independence.

#### The amendment before regeneration

This amendment supersedes the unchanged-seed condition for collided inputs in Design SL.
It preserves the laws, learners, sample sizes, replication budgets, margins, and reading rules.
It does not remove a replication or select a replacement by its fitted result.

| part | declaration |
| --- | --- |
| allocation | consider replication indices in ascending order, then root seeds in sorted order |
| original inputs | keep each original seed unless an earlier allocation already owns it |
| pairing | cells that intentionally share a root share one seed vector |
| replacement | derive deterministic 32-bit retry candidates from the root, index, and retry count; reject occupied candidates |
| preflight | check uniqueness within each stream and disjointness between distinct roots before any fit |
| explicit inputs | validate supplied replication seeds before estimator construction |
| affected studies | audit every registered path that uses the changed allocator; regenerate each study with changed sample inputs |
| execution | use audited targeted reruns under the amendment below; match each original runtime and retain pinned reference evidence |
| publication | retain the old evidence in Git history; write complete combined artifacts and composite manifests; regenerate published tables |
| reading | publish every resulting verdict, including a failed positive or control; keep each original acceptance margin |

For a fixed root set, budget extensions preserve allocations below the old common replication budget.
An extension can change a longer stream's later tail when the shorter stream claims a colliding input.
Acceptance requires this prefix stability, intentional pairing, collision witnesses, and unchanged collision-free streams.
Independent review must verify the corrected allocation and every changed artifact.
The full fast suite, lint, formatting, types, documentation, and package checks must pass.

The audit follows the actual `run_cells` batches, including separate fold-policy and repeat-stability paths.
Five calibration cells replace replication 1,552. The multi-arm study replaces 269 outer-rung inputs.
Every affected cell retains its first 600 inputs. No other registered sample input changes.

| affected study | complete-rerun fallback module |
| --- | --- |
| `canonical-tmle` | `python -m tests.canonical.tmle3.regenerate` |
| `canonical-cvtmle` | `python -m tests.canonical.tmle3_cvtmle.regenerate` |
| `fold-evaluated-cvtmle` | `python -m tests.canonical.cvtmle_fold.regenerate` |
| `fold-targeted-cvtmle` | `python -m tests.canonical.zepid_cvtmle.regenerate` |
| `repeated-crossfit-tmle` | `python -m tests.canonical.repeated_crossfit.regenerate` |
| `canonical-multi-arm-drtmle` | `python -m tests.canonical.multi_arm_drtmle.regenerate` |

The complete-rerun commands use `--jobs 16` and separate empty outputs outside the repository.
The multi-arm command uses its declared-run guard. The shared driver runs the other studies.

#### Targeted execution amendment

The user requests targeted reruns for this seed repair before the corrected multi-arm study starts.
This amendment changes execution and provenance. It preserves the allocation amendment and every scientific acceptance rule.
The [targeted-rerun protocol](development/testing-strategy.md#targeted-reruns) governs reuse.

| part | declaration |
| --- | --- |
| baseline | inherit artifacts from immutable commit `1a6de6ae9141ce65936a47efc3147ff36672dd6a`; verify the original manifests and hashes |
| selection | replace the five calibration inputs and 269 multi-arm inputs identified by the allocation audit before fitting |
| preservation | copy primary and reference files as exact bytes; retain every unaffected property row exactly |
| fits | refit each selected sample under its new seed, using the original law, learners, configuration, and runtime |
| replay | refit unchanged witness samples under their original seeds to test the claimed reuse boundary |
| analysis | recompute complete property summaries, bootstrap intervals, controls, and verdicts from all combined rows |
| provenance | retain original source, hashes, runtime, and clean-state facts; record replacement and analysis execution separately |
| review | independently verify the complete selection, inherited evidence, replay, replacement fits, analysis, and publication |
| fallback | investigate an unexplained replay difference; use complete regeneration if the reuse boundary cannot be established |

The initial complete-run pipeline finishes four studies before the protocol changes.
It stops during repeated-crossfit regeneration. The 20-hour multi-arm rerun never starts.
The completed outputs remain audit evidence. They do not replace the original primary artifacts in targeted publication.

Canonical CV-TMLE's complete rerun changes inspected subject results at floating-point roundoff scale.
Its original Python 3.11 and SciPy 1.17 runtime differs from the rerun's Python 3.13 and SciPy 1.18 runtime.
Targeted fits use each original runtime. The final audit must establish the actual preservation and replay results.

The targeted runner uses `tests/canonical/seed-repair-plan.json` and separate empty outputs outside the repository.
Each command uses `--jobs 16`. The five native thread variables remain `1`.

| study | targeted command | Python / SciPy |
| --- | --- | --- |
| `canonical-tmle` | `python -m tests.canonical.repair_property_seeds --plan tests/canonical/seed-repair-plan.json --study canonical-tmle --output <output>` | 3.13.7 / 1.18.0 |
| `canonical-cvtmle` | `python -m tests.canonical.repair_property_seeds --plan tests/canonical/seed-repair-plan.json --study canonical-cvtmle --output <output>` | 3.11.13 / 1.17.1 |
| `fold-evaluated-cvtmle` | `python -m tests.canonical.repair_property_seeds --plan tests/canonical/seed-repair-plan.json --study fold-evaluated-cvtmle --output <output>` | 3.11.13 / 1.17.1 |
| `fold-targeted-cvtmle` | `python -m tests.canonical.repair_property_seeds --plan tests/canonical/seed-repair-plan.json --study fold-targeted-cvtmle --output <output>` | 3.11.13 / 1.17.1 |
| `repeated-crossfit-tmle` | `python -m tests.canonical.repair_property_seeds --plan tests/canonical/seed-repair-plan.json --study repeated-crossfit-tmle --output <output>` | 3.11.13 / 1.17.1 |
| `canonical-multi-arm-drtmle` | `python -m tests.canonical.repair_property_seeds --plan tests/canonical/seed-repair-plan.json --study canonical-multi-arm-drtmle --output <output>` | 3.13.7 / 1.18.0 |

### P1. EP learner

Van der Laan, Carone and Luedtke (2024), arXiv:2402.01972, govern this item. After first-hand
review of their EP derivation, add `ConditionalContrast` estimands, modifier schema, sieve/basis
strategy, efficient plug-in risk and targeting, bounded outcome predictions, a second-stage
contrast learner, out-of-fold risk/calibration, and a conditional prediction result. Reuse
study/identification objects, nuisance strategies, folds, data backends, provenance, persistence,
and capability-aware assessment.

The first catalog is paper-derived CATE and conditional relative risk. Other losses and contrasts
require their own derivations. Aggregating an EP curve is a separate parameter and receives scalar
inference only after its influence contribution is implemented and tested.

Acceptance requires exact score and risk checks, bounded predictions, out-of-fold calibration,
modifier and split/basis stability diagnostics, mutation controls for targeting sign, basis
contribution, and contrast construction, plus registered oracle-efficiency and stability studies.

### R1. Nested Riesz engine and initial catalog

#### Purpose and scientific boundary

Add the general nested Riesz engine as one scientific review unit. Direct Riesz learning replaces
the analytic construction of a representer from propensity or density components; it does not
generally replace the outcome regression. The single-stage form remains a plug-in term plus
`alpha * (Y - f)`. Engine expressibility is not permission to register a causal target.

The governing nested-TMLE source is Balkus, Testa & Hejazi (2026), arXiv:2604.21721v1: Theorems
1–2, Algorithm 1, and Sections 4 and 5. Chernozhukov, Newey & Singh (2022), Econometrica 90(3),
governs the direct representer moment problem and product-bias identity. RieszCML is pinned only as
secondary implementation evidence; paper/code discrepancies must be recorded and tested.

Stages are stored **innermost first**, matching the order the backward recursion *executes* in.
Two nearby orderings run the other way and neither is the storage convention. Theorem 2 indexes
outermost first, so the paper's *prefix* product is a *suffix* product here:

```text
paper order:       outer 1, ..., inner T
stored order:      inner 0, ..., outer J-1
```

And `longitudinal/sequential.py` reverses its steps before returning them, so the stored
`SequentialStep` tuple is time-ascending. `steps[0]` is the outermost stage whose targeted
prediction is averaged into the estimate. Do not mirror it. Every public description, serialized
manifest, fixture, and diagnostic names the storage direction.

#### Contracts and public surface

Introduce immutable advanced contracts under a Riesz namespace.

| contract | what it declares |
| --- | --- |
| `FunctionalStage` | the regression, plug-in map, representer problem, intervention evaluation, residual source, identification metadata, targeting map, stage name, and history |
| `NestedFunctional` | ordered stages, evidence ID, output shape, parameter key, and causal metadata |
| `AnalyticRepresenter`, `DirectRiesz`, `ProvidedRepresenter`, internal `ComposedRepresenter` | the mechanism-derived, directly learned, externally supplied, and cumulative stagewise representers |
| fitted stage artifacts | observed and intervention regression predictions, `alpha`, `alpha_star`, component and cumulative products, residual sources, masks, folds, targeting coefficients, losses, balance diagnostics, row identity, and training provenance |
| `RieszTMLEMethod` | a scalar `RieszResult` satisfying `CausalResult` and the shared inference, contrast, identification, assessment, persistence, and provenance protocols |

A custom statistical functional may use `NestedFunctional`. Causal prose and intervals need a
registered evidence ID and a complete influence construction. `RieszTMLEMethod` never fabricates a
propensity object for a direct fit.

A provided representer must establish row identity, folds, training provenance, and
counterfactual evaluation. Missing `alpha_star` is refused rather than replaced by observed
`alpha`. Built-in estimands construct stages internally; advanced custom builders validate roles,
shapes, histories, intervention evaluation, and identification metadata before fitting.

#### Folds, fitting, and targeting

Use one validated outer fold plan for regression and representer fits. Each row is predicted only
by models not trained on it; learner tuning remains inside the outer training fold; supplied folds
must prove coverage, disjointness, grouping, cluster integrity, and row identity. Independent
nuisance folds remain unavailable until their theory and diagnostic implications are specified.

For stored stages `0..J-1`, innermost first, construct:

```text
omega_observed[j] = product(alpha[k], k=j..J-1)
omega_plugin[j]   = alpha_star[j] * product(alpha[k], k=j+1..J-1)
```

The plug-in product changes only the current stage to its intervention state; it is not a product
of every `alpha_star`. Store all components separately. Fit initial regressions from the observed
outcome outward, fit or construct each stage's observed and intervention representers on the same
training fold, and pass each plug-in output outward as the next target.

A direct strategy solves an explicit empirical Riesz moment problem. A generic supervised
regression of an estimated inverse propensity score is not a direct-Riesz implementation and is
not labelled one. It is the nearest wrong construction, and it fits and predicts without
complaint.

Target innermost first. Solve the fluctuation with `omega_observed[j]`, update observed predictions
with that product and plug-in predictions with `omega_plugin[j]`, and pass the targeted plug-in
outward. Identity and logistic fluctuations are supported initially. Logistic targeting requires
a validated scaler and preserved bounds; a degenerate direction raises `CleverlyError`.

The targeted uncentered curve is:

```text
outermost_targeted_plugin
+ sum_j omega_observed[j] * (
      targeted_residual_source[j] - targeted_observed_regression[j]
  )
```

Center it at the targeted plug-in estimate, then apply declared Hájek weights, outcome scaling,
cluster aggregation, covariance, smooth contrasts, and simultaneous inference through existing
infrastructure. Recompute stage scores and the complete influence-curve mean from stored arrays.
Do not use an untargeted curve for a targeted estimate's standard error.

Repeated splitting follows only after a single split is correct. Take marginal medians over
repeats with the registered split-dispersion variance. Retain repeat IDs on artifacts. Refuse joint
covariance where coordinatewise medians break identities. Reject equal-fold averaging with an
unequal-fold-size test.

#### Initial catalog and refusals

The implementation PR must support each initial cell with full evidence or retain its named
pre-fit refusal.

| typed estimand or design | analytic Riesz | direct Riesz | decision |
| --- | --- | --- | --- |
| point `CounterfactualMean` | yes | yes | canonical single-stage functional |
| point `ATE` | yes | yes | joint contrast of counterfactual means |
| point `RiskRatio` and `OddsRatio` | yes | yes | existing joint delta method; binary outcome only |
| point outcome missing at random | yes | yes | missingness stage composed with each treatment-specific mean |
| longitudinal static and dynamic `RegimeMean` | yes | yes | evidenced sequential instances after history and arm mutations pass |
| longitudinal `RegimeContrast` | yes | yes | joint contrast of evidenced regimen means |
| point `ModifiedTreatmentPolicy` | yes | yes | evidenced invertible policies with target-specific moment map |
| smooth contrasts of initial rows | yes | yes | existing delta-method infrastructure |
| `ATT` and `ATC` | gated | gated | ratio or conditional-functional stage and variance audit required |
| `NaturalCourseMean`, `PopulationAttributableRisk`, and `PopulationAttributableFraction` | gated | gated | composition and parameter-key audit required |
| one-point stochastic `RegimeMean` | gated | gated | density-valued direct-loss audit required |
| `IncrementalMean` and `IncrementalEffect` | refused | refused | intervention depends on the treatment mechanism |
| point or longitudinal `MSMProjection` | gated | gated | target-specific projection and coefficient-EIF adapter required |
| mediation and `ControlledDirectEffect` | refused | refused | outside the first catalog |
| survival and competing-risk results | refused | refused | separate at-risk/event audit required |
| arbitrary custom nonlinear functional | refused as causal | refused as causal | statistical label only after validation |

Analytic Riesz is the existing evidenced mechanism-derived estimator expressed through the new
contract. Its gate is identity with the normalized existing engine for targeted predictions,
influence curve, standard error, and interval. Point estimate parity alone is insufficient.

#### Diagnostics and persistence

Default cheap validation covers stage scores, influence-curve mean, direct objective improvement,
held-out moments, representer tails and leverage, regression loss, and fold/provenance integrity.
Analytic fits may report mechanisms and truncation; direct fits report representer loss, balance,
tails, leverage, regularization, and clipping; composed fits report components and products.
Direct-only fits must not reconstruct a propensity model for diagnostics or sensitivity.

Store ordered stage definitions and evidence IDs, every fitted array, reconstructible strategy
configuration, fingerprints, settings, structured identification and parameter keys, cached reports,
and replayability. Unknown types fail allowlist decoding. Custom callables are descriptive and
non-reconstructible; cache-only operations remain possible only when stored predictions and
provenance suffice. Losing `alpha_star`, stage order, or component products is a load error. Round
trips compare every artifact, score, diagnostic, estimate, influence curve, metadata record, and
cached assessment.

#### Evidence and implementation sequence

Every registered functional needs the full instrument set. That is exact finite-support laws,
Gateaux derivatives, and product remainders with both nuisance errors nonzero. It is also score
checks, union-model witnesses where derived, and nonzero targeting. The mutations are sign,
`alpha_star`, stage and product order, mask, and counterfactual. Leakage spies, analytic
regressions, and a pinned secondary fixture complete it.
Direct fits additionally require loss improvement, held-out moments, same-fold observed and
intervention predictions, provenance, and extreme-representer diagnostic response.

Nested evidence uses nonconstant two- and three-stage laws and rejects reversed stages, reversed
products, all-starred products, wrong signs, and masks. Missingness evidence makes observation
depend on treatment/history and proves unobserved rows receive the correct plug-in update.
Longitudinal evidence covers dynamic-history restriction, third-arm categorical mutations,
treatment/censoring products, clusters, repeats, and persistence. Registered studies cover point
means/ATE under both union-model halves, direct consistency and coverage, static/dynamic regimen
means, and weak overlap with thresholds fixed before the final run.

Implement in one review PR with gated commits: contracts and pre-fit refusals; analytic
single-stage parity; direct single-stage learning; nested and missingness composition;
longitudinal adapters; then persistence, assessment, documentation, and the secondary fixture. No
commit merges independently. Handoff requires the enabled/refused catalog, source locators,
implementation revision and discrepancies, evidence instruments, local commands/results, and
path-based reasons for omitted validation studies.

### R2. Evidence-gated Riesz catalog expansion

Expand target by target after the engine lands. Mediation, additional longitudinal targets,
sampling designs, and other nested functionals each require a governing derivation, typed adapter,
registry entry, evidence row, refusal boundary, documentation, and applicable statistical study.
Do not expose a generic engine capability as a certified causal estimand.

### F8. Multi-arm simulated-confounding stress surface

The cited DoWhy papers support a qualitative stress analysis. They do not define a multi-arm
treatment perturbation. The pinned implementation supports a binary tail flip and a continuous
linear change only. It has no category-valued branch.

Hu et al. (2022), DOI 10.1214/21-AOAS1530, keep treatment fixed and adjust outcomes through
directional confounding functions. That method does not supply the category-valued refit law that
this surface needs. A pairwise label swap or a map over numeric arm codes would therefore be a new
scientific construction.

Wait for a published law that maps a shared latent variable and a $K$-level treatment to the same
declared support. The law must name the assessed contrast, remain invariant to label order, and
define the achieved treatment-confounder association. Until then, retain the multi-arm pre-fit
refusal. Do not reuse a binary flip by choosing two arm codes.

### F9. Clustered simulated-confounding stress surface

The pinned DoWhy implementation draws one independent latent value per row. It does not accept a
cluster identifier or define a cluster-level perturbation. Pinned `lmtp` preserves identifiers in
folds and inference, but it does not define a simulated-common-cause surface.

Ou, Tang and Chang (2023), arXiv:2301.12396v1, model unmeasured cluster effects through mixed
models and derive a different bias correction. Their construction does not perturb treatment and
outcome before a complete TMLE refit. It does not choose between row-level, cluster-level, and
mixed latent causes for this surface.

Wait for a published or canonical construction that defines that choice and its interpretation.
Until then, retain the clustered pre-fit refusal. Do not infer a shared cluster draw from grouped
folds or cluster-robust variance, because those contracts govern estimator dependence only.

### F10. Logical categorical confounder calibration

The pinned DoWhy implementation calibrates one encoded coordinate at a time. Its
[calibration helpers, lines 213-340](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/add_unobserved_common_cause.py#L213-L340)
hold both rules. The binary rule zeros one standardized column. The continuous rule uses one
column's correlation times the perturbed variable's standard deviation. Neither rule defines a
whole categorical covariate on the same scale.

Wait for a source that maps a logical categorical covariate to binary flip probabilities or signed
continuous perturbation strengths. The benchmark must state how labels, reference levels, and
multiple encoded columns affect it. A coefficient norm, grouped deletion, or permutation score
does not supply that mapping by itself. Keep the categorical refusal before every random draw
and refit until this contract exists.

### F11. Estimated-weight simulated-confounding replay

The fitted result stores estimated weights but not the model that produced them. Replay also needs
the target population and the variables that the model can read after perturbation.

Hartman and Huang (2024) bound bias from an omitted variable in survey weights. Their method does
not define weight regeneration inside this refit surface. Wait for a source-backed regeneration
rule, then store enough model provenance to reproduce it. The rule must state whether the latent
cause enters the weight model and which population each cell targets.

### F12. Missing-outcome simulated-confounding replay

The pinned DoWhy refuter perturbs treatment and outcome only. It defines no replacement law for the
response indicator. Díaz and van der Laan (2017) define randomized missing-outcome DR-TMLE under
missing at random. They do not define this diagnostic's joint perturbation law.

Do not hold the response indicator fixed. A treatment perturbation can break missing at random
conditional on the perturbed treatment. Wait for a joint observation, treatment, and outcome law
with identified refit semantics. The law must cover ordinary TMLE and the randomized missing-outcome
DR-TMLE construction separately.

### F13. Longitudinal simulated-confounding replay

The pinned DoWhy law is a single-time-point perturbation. It does not define shared latent causes
across treatment, censoring, history, and outcome nodes. Tan (2025) supplies longitudinal
sensitivity bounds, not a complete-refit perturbation surface.

Wait for a time-indexed latent law that preserves temporal order and names the assessed contrast.
It must define each node's perturbation, history update, censoring behavior, and induced association.

### F15. Controlled-direct-effect simulated-confounding replay

A controlled direct effect orders treatment, intermediate, observation, and outcome mechanisms.
The pinned DoWhy law has no intermediate branch and no contrast rule for a fixed intermediate.

Wait for a source-backed latent law that respects this order. The law must define each perturbed
mechanism, the response indicator, and the controlled contrast before complete refits can begin.

### F23. Simulated confounding on a declared outcome scale

`simulated_confounding` perturbs a Gaussian outcome by subtracting a strength times the shared
standard normal latent value (`_gaussian_outcome`,
`src/cleverly/sensitivity/simulated_confounding.py`). That perturbation is additive on the
outcome's own scale, and the latent value is unbounded. A fit that declares `q_bounds` therefore
refuses the refit, because the perturbed outcomes leave the declared support
(`src/cleverly/utils/bounds.py`). Every nonzero outcome strength fails, and a cross-fitted
continuous fit must declare `q_bounds`, so the outcome axis of the surface is unavailable to every
such fit.

A result must supply a latent perturbation law for an outcome confined to a known support, and the
reading of its strength parameter. A declared support is a statement about the measurement, so a
perturbation that leaves the support reports a different measurement rather than a stronger
confounder. Clipping the draw back to the support is not that law, because clipping changes the
induced association by an amount the strength parameter no longer names.

The treatment axis is unaffected, and the surface records the failure of each refused cell rather
than abandoning the run. `docs/examples/interventions.ipynb` shows two refused cells and states
this reason. The generated-outcome refutation refuses the same composition for the same reason, in
`_validate_generated_eligibility` (`src/cleverly/validation/refute.py`).

### F25. E-value for a controlled direct effect

A controlled direct effect is identified under two assumptions of no unmeasured confounding.
Assumption 2 covers the treatment and the outcome. Assumption 3 covers the intermediate variable
and the outcome (`src/cleverly/estimators/direct_effect.py`). The E-value of VanderWeele and Ding
(2017) inverts a risk-ratio bounding factor for a stated exposure contrast.

The [E-value paths](technical-reference/validation-methods.md#e-value) record the sources read
and their coverage. Ding and VanderWeele (2016, *Epidemiology*) allow a multivariate confounder
and two levels of a general exposure. This does not itself map a composite $(A,Z)$ exposure onto
the package's population contrast between $A$ arms at one fixed $Z=z$. The natural-effect
results of Ding and VanderWeele (2016, *Biometrika*) and Smith and VanderWeele (2019) use
different targets and confounding models. Gilbert, Fong, Kenny and Carone (2023) study a
controlled effect, but their E-value compares marker interventions within a vaccination arm.

A source-backed specialization of an existing bound is eligible; no paper has to name this
package's API. Before implementation, specify the fitted target, the target population, and the
outcome scale.

Map each allowed unmeasured confounder to its time in the treatment-intermediate-
outcome sequence. Define the treatment-confounder and outcome-confounder strengths under the
intervention on $Z$, including any conditioning on $Z$ or joint $(A,Z)$ assignment. Show that
the observed-data contrast and its standardization match the source's bounding factor for this
target. Then derive the inversion for the point estimate and the reported interval. Validate
the mapping with a nonzero law and controls that fail for a wrong arm, level, or confounding path.

### F26. Confidence limits of the plug-in omitted-variable bound

`nu2_estimator="plugin"` estimates the Riesz second moment as $E_n[\hat\alpha^2]$. That value moves
at first order with the fitted treatment mechanism, and its curve $\hat\alpha^2 - \hat\nu^2$ has no
term for that fit. On `tests/discrete_law.py` with the oracle nuisances, that curve differs from the
Gateaux derivative of $\nu^2$ by up to 21.3 for the ATE and 45.9 for the ATC. Chernozhukov,
Cinelli, Newey, Sharma and Syrgkanis (2026) estimate $\nu^2$ through the orthogonal score of their
Lemma 3 only. DoubleML calls its plug-in fallback non-orthogonal. No read source gives the influence
function of the plug-in bound for every mechanism learner this API accepts.

Saul and Hudgens (2020) give the stacked estimating-equation and sandwich-variance method for
smooth, finite-dimensional estimators. That method supplies a route for a specified parametric
mechanism: include its score, the second-moment equation, and their joint Jacobian. It does not
provide one curve for the arbitrary learners accepted here. A specialization needs a nuisance-score
interface, a derived joint curve, stated model and rate conditions, and a validation study. Open
`ci_lower`, `ci_upper`, `robustness_value_ci` and `rva` only for a specialization that meets them.

## Longitudinal contracts

The four core LTMLE evidence rows are implemented and registered in the
[validation grid](technical-reference/method-evidence/validation-grid.md). They separate
end-of-study and survival parameters from ordinary and cross-fitted nuisance estimation. The
remaining items below are proposed extensions to that core.

### F1. Stochastic categorical policies at a longitudinal node

The implemented surface assigns one category per unit. A distribution-valued policy changes the
intervention density and replaces selected probabilities with cumulative density ratios.
Implementation waits for published identification, longitudinal influence function, remainder,
and interval rate conditions; a point-treatment stochastic regime is not sufficient evidence.

### F2. Targeted bootstrap inference

Wait for a source specifying what is fixed, resampled, refitted, and retargeted and which sampling
law the interval estimates. Resampling stored curves, retargeting cached arrays, and refitting the
complete estimator are distinct procedures and must not be inferred from the name.

### F16. Longitudinal sensitivity-bound estimation

Tan (2025) derives population sensitivity bounds for a terminal outcome under binary, static
longitudinal strategies. Section 3.2 leaves estimation with sample data to future work. Section
6.3 asks for specialized algorithms, and for sample estimation of the ICE and IPW functionals. The
paper reports no sampling inference for any bound. Wait for those three contracts before adding a
sample-data operation.

### F3. Additional longitudinal estimands

Competing-event interventions and other longitudinal estimands wait for their own identification
assumptions, influence functions, targeting construction, and inference conditions. Add accepted
targets in both directions to the oracle registry and evidence gates rather than treating them as
options on an existing cause-specific estimand.

### F7. Time-respecting cross-fitting

Blocked-temporal and rolling-origin folds wait for a published TMLE result whose dependence
assumptions match the supported data. The result must specify which rows may train each prediction
and which asymptotic argument licenses the interval. Ordered indices passed through iid fold
machinery are not sufficient.

## Collaborative and DR-TMLE investigation contracts

### F18. Selector-path C-TMLE inference

The package keeps the greedy, ordered, and discrete point estimates and path diagnostics.
[RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism) delivered the refusal
that this item asked for. These paths now refuse confidence intervals and p-values, and report the
fixed-working-mechanism plug-in standard error under an explicit noninferential diagnostic name.
That computation is ordinary EIF covariance after the fit selects one candidate. It does not
establish conditional-on-selection or unconditional post-selection coverage. F18 reopens inference
on these paths only when a published result supplies the missing influence function and covariance.

A 2026-09-21 reading found that this item does not qualify as a natural extension under the
[Eligibility](#eligibility) rule. The selector is a data-adaptive selection step, which the rule
names as one with no established argument. Van der Laan and Gruber (2010), Section 4.3, records
cross-validation over-selection as "an area of study". No published base exists for one stopping
index shared across arms. The item stays in this grid.

Van der Laan and Gruber (2010), Section 4, derive an abstract adaptive-mechanism contribution under
fixed-limit and regularity assumptions. Their candidate-selection discussion does not derive the
influence function after the shipped stopping-index procedure selects a candidate.

Ju et al. (2018) permit a continuous-path contribution, but their Lemma 2 makes it zero in one
correct-outcome, product-rate regime. Their derivative equations and undersmoothing conditions do
not describe a discrete stopping index. Ju, Gruber et al. (2019), *SMMR* 28(2), report ordinary
EIF intervals for binary ATE fits. The
[source audit](references.md#collaborative-tmle) records the exact limits.

A 2026-09-21 search added five sources on this path and found no result for it. Cui and Tchetgen
Tchetgen (2024) select nuisance learners by a cross-validated pseudo-risk. In arXiv:1911.02029v6,
Theorems 5.1 and 5.2 give an oracle inequality and the consistency of the selector, and no limit
law. Section 6 says a Wald interval at the selected learners "is completely blind to the model
selection step".

Ju, Schwab and van der Laan (2019), *SMMR* 28(6), and Ju, Wyss et al. (2019), *SMMR* 28(4), state
no post-selection theorem. The first reports standard errors smaller than the sampling spread for
C-TMLE in its experiments.

Liu's 2018 Harvard dissertation, Chapter 2, derives asymptotic laws for C-TMLE coefficients,
squared errors, prediction risks and M-fold cross-validation risks, and studies post-selection
AMSE. It assumes the propensity candidates, outcome variance and covariate law are known, treats
one binary treatment-specific mean, and obtains the practical post-selection AMSE by Monte Carlo.
It therefore sharpens the selector literature without covering the shipped learned nuisances,
discrete stopping rule or joint targets.

Four further sources bound the available routes. The cross-validation oracle inequality controls
selector risk but supplies no limit law. Leeb and Pötscher rule out locally uniform distribution
estimation in finite-dimensional regression model selection, conditionally in 2006 and
unconditionally in 2008; nobody has transferred either theorem to this selector. Loftus (2015)
handles squared-error cross-validation when a Gaussian regression selection event is quadratic.
Markovic, Xia and Taylor (2017) give selective pivots when the criterion vector and target
statistics have a joint asymptotic Gaussian law, optionally after randomization. The shipped
targeted-loss selector has neither the required quadratic Gaussian reduction nor a randomized or
joint-Gaussian criterion-and-target result.

Adaptive debiased machine learning is the closest positive route. Its v2 theorem permits
model-based inference after selection when the learned working model approximates a fixed oracle
model and the stated approximation and remainder conditions hold. For a cross-validated sieve,
the limiting dimension must exist and be nonrandom. The package instead selects one global depth
from nested targeted-loss folds. A proof must establish that this depth and its induced model meet
those conditions.

[Van der Laan et al. (2026)](https://arxiv.org/html/2501.11868v3), Section 5.2 and Appendix C,
make the selected-model obligations explicit. Theorem 5 requires a linear expansion,
influence-curve stabilization, and model-approximation rates. Section 5.1, Corollary 1, covers its
own autoTML construction under Conditions B1 to B5. Neither result verifies those conditions for
the shipped stopping rule.

The oracle projection's efficiency bound is typically, not invariably, smaller
than the nonparametric bound. Applicability could therefore ratify the current curve or require a
different one; it does not predetermine the covariance verdict.

Read data-adaptive-target inference as a second option. Honest splitting permits arbitrary target
generation; the same-sample theorem instead needs a uniform expansion, Donsker control, and
influence-curve convergence. Both concern a data-adaptive estimand rather than automatically the
fixed target reported by the package. The shipped selector builds its path on the full sample and
picks one global index without checking either regime.

A direct positive route needs no selector influence term. If every candidate has the uniform
vector expansion

$$
\widehat\psi_{n,k}-\psi_0=(P_n-P_0)D_0+r_{n,k},\qquad
\max_{k\leq K_n}\lVert r_{n,k}\rVert=o_p(n^{-1/2}),
$$

with the same deterministic influence curve $D_0$, substitution of any data-dependent
$\widehat k$ preserves that expansion. For a fixed target-vector dimension, and therefore a fixed
number of treatment arms, uniform covariance consistency then validates the ordinary vector EIF
covariance by Cramér--Wold. Growing dimension would instead require a high-dimensional Gaussian
approximation. This elementary extension is plausible when every candidate converges fast enough
to the same nuisance limits, but it is not automatic under one-sided robustness where candidate
propensity limits and influence curves may differ.

Qiu, Luedtke and Carone (2021) publish one instance of this route. Theorem 4, Section 4.4 of
arXiv:2003.01856v2, keeps the ordinary influence function after cross-validation selects a sieve
dimension. It needs its conditions at a deterministic dimension, and Condition C5 at every
candidate. Their Section 3.3 shows by simulation that a cross-validated HAL bound does not give an
efficient plug-in estimator. The theorem is a template for this route, and it is not a result for
a targeting step along a propensity path.

A second route is selector stability: a unique oracle candidate, a risk margin, uniform risk
convergence, and stability of the learned covariate identity at a given depth reduce the fit to a
fixed candidate with probability tending to one. An oracle risk inequality alone proves none of
those facts; Shao's linear-model result is a concrete warning that prediction-efficient fixed-fold
cross-validation need not consistently select a model. A third route is a construction change:
select depth wholly within each outer training fold, then evaluate only on its held-out rows.
Standard orthogonal-score arguments can then treat the complete selector as a nuisance-learning
algorithm, subject to its rate conditions. The current global risk aggregation does not have that
independence.

Dang, Tarp, Abrahamsen et al. (2025), *Journal of Causal Inference* 13:20240041, DOI
10.1515/jci-2024-0041, give the closest published form of the third route. Their ES-CVTMLE trains
the experiment selector inside each fold and derives a nonstandard limit with Monte Carlo
intervals. It does not cover the shipped criterion, target or shared stopping index. Bibaut and van
der Laan, Theorems 1 and 2 of arXiv:1706.07408v2, also select a scalar index on subsamples separate
from the estimation sample. This search found no journal version of that paper.

A third 2026-09-21 search found four neighbors for F18, and RM18 at commit `dea3297e` records
its log. The nearest is
Rothenhäusler (2024), *Electronic Journal of Statistics*, DOI 10.1214/24-EJS2308. Its Section 3.4,
Theorem 2, gives intervals after a choice among a fixed finite set of estimators. The preprint
arXiv:2008.12892v2 numbers it Section 3.5, Theorem 4. The search read the other three from their
abstracts. They are Schnitzer, Sango, Ferreira Guerra and van der Laan (2020), Zrnic and Jordan
(2023), and Van Lancker, Díaz and Vansteelandt (2024), arXiv:2404.11150v2.

This review also read Schnitzer, Lok and Gruber (2016), Section 5.3, Table 3 and discussion.
Their simulations show ordinary influence-curve undercoverage for TMLE and C-TMLE with Super
Learner in some settings. This cautions against treating a plug-in interval as validated, but
their construction does not supply the selector-path expansion F18 needs.

The theorem needs a joint normal limit at a fixed parameter and an asymptotically unbiased
baseline estimator. The published text calls the inference "usually
not uniformly valid" and asks readers to use it "with caution". The shipped selector minimizes a
cross-validated targeted loss along a path the data build. No result shows that this path meets
the theorem's conditions, so the theorem does not close F18.

The fixed-candidate curve is itself unproved when the candidate's mechanism limit differs from the
treatment law. [RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism) records an
instrument law where the intercept-only curve gives a standard error of 0.0441. The sampling
standard deviation of the numerically identical least-squares coefficient is 0.0538. A result must
settle that fixed-candidate case before it addresses selection.

One published result shows the form a correction can take when the mechanism limit differs from
the treatment law. Ju, Benkeser and van der Laan (2020) fit an intentionally inconsistent
propensity. Their Theorem 1, in Section 3.4 of arXiv:1806.06784v3, gives the influence function as
the ordinary curve at that propensity limit minus a first-order term $D_r$. The term vanishes when
the limit equals the true propensity. Their construction differs from the shipped selector, so the
theorem does not supply the missing curve.

The registered evidence now measures that gap on two studies. The
[selector-based point-treatment study](technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md)
passes 11 of 14 property cells, and the
[selector-based multi-arm study](technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md)
passes 5 of 12. The [red-cell ledger](technical-reference/method-evidence/red-cells.md) lists each
red cell with its interval and its owner, and
[RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) records each attribution.
No result here identifies a selection contribution, so each study publishes under a `reporting`
policy.

Accept a result only when it covers the selected index and the selector's three split layers.
Those layers are outer nuisance folds, selection folds, and inner selection-training folds. The
result must establish whether the current curve suffices or an additional contribution is
required. It must state its remainder, rate conditions, and covariance for every supported target
vector. Learner folds and repeat draws are separate split layers.

Treat analysis regimes as separate proof cells rather than automatic transports of the first
selector result. An iid, complete-outcome, unweighted, unstratified, single-repeat result certifies
only that cell.

Missing-outcome fits additionally need the response-mechanism terms. Fixed probability weights
need a declared weighted empirical-law result. Estimated weights also need their first-stage
influence contribution.

Cluster-robust fits need a cluster-level expansion. Stratified estimates need stratum-specific
expansions and their joint covariance. Repeated cross-fitting needs the package's median and
split-dispersion aggregation. Keep each extension reporting-only until its result and
repeated-sampling evidence are registered here.

One structural point governs the multi-arm case. The package picks one stopping index for every
arm together. Any non-negligible selection contribution is driven by that shared rule but may
enter each estimand through a different sensitivity. Its full vector influence function and its
cross-covariance with the ordinary EIF are needed before changing the covariance or simultaneous
critical value. A per-arm copy of a binary correction does not generally recover those terms. The
selection criterion also reduces a target vector to one scalar risk, and any derivation must
differentiate that reduction.

If the result requires a new contribution, propagate it through pointwise and simultaneous
inference. Acceptance needs a fixed-candidate reduction and repeated-sampling coverage where the
chosen candidate changes. Retain each repeat's path and selected index, and define how selection
enters the repeated estimate. Use an independently calibrated rule before stating that selection
uncertainty is negligible.

### F19. Outcome-adaptive C-TMLE generated-design inference

Outcome-adaptive C-TMLE selects no candidate. It fits the categorical treatment mechanism on the
estimated vector of arm-specific outcome predictions. The retained diagnostic uses ordinary
adaptive-propensity EIF plug-in covariance, and the package reports no interval. The cross-fitted
implementation now follows the paper's fold-local nuisance nesting: one outcome model fitted on fold
`v`'s training rows creates both sides of that fold's generated design, the adaptive propensity is
fitted only on its training side, and both nuisances are evaluated on its held-out side. Under the
paper's six regularity conditions, the ordinary adaptive-propensity curve needs no extra first-order
generated-design term for one binary treatment-specific mean. Appendix D constructs a binary ATE
with both arm predictions and one signed fluctuation coefficient, and outlines pooled-validation
CV-C-TMLE.

This item names each appendix by its letter in the arXiv preprint 1901.05056v1. There, Appendix D
holds the additional estimators, and Appendix F holds the details for Theorem 1. The preprint text
cites those details as Appendix G. The [references](references.md) entry records the same Appendix
G and Appendix F mismatch between the published article and Supplement A.

The archived `ctmle3` source supplies implementation provenance but no inference derivation.
Benkeser, Cai and van der Laan (2020) prove a binary treatment-specific-mean result under explicit
score, rate, smoothness, and empirical-process conditions. Appendix D explicitly uses both binary
arm predictions in one adaptive propensity for the ATE and sketches a cross-validated C-TMLE. The
generated design is therefore part of that paper rather than an omitted nuisance.

The paragraphs above describe the nuisance nesting, and the fluctuation needs its own statement.
`CTMLE` refuses `targeting_scheme="fold"` (`src/cleverly/estimators/ctmle.py:758-763`), so the
outcome-adaptive fluctuation is one pooled epsilon on the stacked out-of-fold rows
(`src/cleverly/estimators/tmle.py:395-401`, `:3154-3160`). Appendix D outlines that pooled
fluctuation. The shipped update therefore does not diverge from the paper on this axis. Two
divergences are genuine, and the table below states each one.

| divergence | what ships | what Appendix D outlines |
| --- | --- | --- |
| the final average | the stacked whole-sample plug-in (`src/cleverly/estimators/tmle.py:429-439`), because `CTMLE` refuses `cv_evaluation=True` (`src/cleverly/estimators/ctmle.py:753-757`) | the `(1/V) sum_v` fold average. Equal fold weight mass guarantees agreement, but unequal mass can also agree. With fixed $V$, unweighted near-balanced folds, and bounded predictions, the difference is $O(V/n)$ |
| the fluctuation dimension | a joint fluctuation with one column for each arm | one signed coefficient for the binary ATE |

The package instead jointly targets both arm means with two fluctuation columns and derives means,
ATE, RR, and OR from that fit. A fixed-dimensional Cramér--Wold extension would be elementary once
the scalar expansions for that exact shared design and fluctuation were established, but the paper
does not state those expansions. Its model is also iid, complete-outcome, and unweighted.

The implementation previously cross-fit the adaptive propensity on globally assembled out-of-fold
outcome predictions. A propensity-training feature could then depend on outcomes from that
propensity fold's evaluation rows. The fold-local replacement removes that path and has a mutation
test that changes every evaluation outcome without moving either nuisance on those rows.

The `drtmle` `adapt_g` option supplies multi-arm implementation provenance and reports ordinary
influence-curve covariance, but it is not a multi-arm inference theorem.

Two newer frameworks sharpen that boundary. DOPE allows a finite treatment set and fixed
contrasts, and Theorem 4.3 proves ordinary-influence-function inference centered at a
data-adaptive target conditional on a representation learned on an independent sample.
Proposition 4.4 shows that root-rate representation learning can instead add a first-order
delta-method variance when inference is for a fixed target. Its appendix gives a cross-fitted
algorithm but explicitly leaves the dependent fold-oracle proof open. Outcome-adapted AutoDML
also decomposes sampling and representation errors and proves a sample-split result; it does not
prove the cross-fitted construction. Neither result covers this package's same-fold design and
mechanism fitting, shared multinomial propensity, joint targeting, and simultaneous covariance.

Generic generated-regressor orthogonality does not close the gap. Escanciano and
Pérez-Izquierdo (2023) show that second-step orthogonality removes the indirect first-step effect,
while the direct effect of learning the generated regressor remains and may require correction. A
mixed-bias nuisance-product remainder is likewise not the first-order expansion of a learned
representation. Whether the ordinary curve suffices must be derived for this estimator and target,
not inferred from either generic result.

Two sufficient-dimension-reduction papers point in both directions, and this search read their
abstracts only. Zhang, Shao, Yu and Wang (2018) state that an estimated reduction changes the
asymptotic variance unless it keeps superfluous covariates. Ma, Zhu, Zhang, Tsai and Carroll
(2019) state that an efficient-influence-function estimator with reduction-estimated nuisances
stays efficient. Neither applies a targeting step or a shared multinomial mechanism.

A third 2026-09-21 search ran two queries on the generated design, and RM18 at commit
`dea3297e` records its log. It
found no result for a shared multinomial mechanism fitted on estimated outcome columns, for its
joint targeting, or for its cross-fitted form. It recorded one neighbor from its abstract:
Schnitzer, Talbot, Liu et al. (2026), *Statistics in Medicine* 45(1-2):e70316, DOI
10.1002/sim.70316. That paper selects covariates for a longitudinal treatment model with an
outcome-adaptive LASSO. It fits no mechanism on estimated outcome predictions.

A 2026-09-21 reading applied the natural-extension exception of the [Eligibility](#eligibility)
rule and provisionally accepted the full-sample fit by invoking a vector generated regressor and
Cramér--Wold. The published *Statistical Science* article and supplement are open, and the checked
result is Theorem 1. That theorem proves one binary treatment-specific mean. The supplement's
binary direct-ATE algorithm uses two outcome-prediction columns and one signed fluctuation
coefficient, but states no theorem for it.

The provisional reading is rejected. The package fits one shared multinomial mechanism on `K`
estimated columns and uses a `K`-column joint fluctuation. Cramér--Wold converts already-proved
scalar joint expansions into a vector limit; it does not prove those expansions, their remainders,
or the covariance induced by the shared learned mechanism. The full-sample multi-arm fit therefore
does not qualify as a natural extension. The cross-fitted default remains further outside Theorem
1 because Appendix D only outlines its proof.

| item | what it requires |
| --- | --- |
| the natural-extension verdict | closed as a source-reading question: the shared-multinomial joint construction does not qualify, for the reasons above. The estimator result remains open |
| the base result | Theorem 1 of the open *Statistical Science* article, checked against the article and Supplement A. The article points to Appendix G for conditions while the supplement labels them Appendix F |
| the written record | a contract that states each step, its argument, and every inherited condition |
| its own evidence | property cells at `cross_fit=False`. The multi-arm primary block already fits `cross_fit=False` (`tests/studies/canonical_multi_arm_ctmle_oat.py`), and it is a parity comparison with `ctmle3`. Every property cell fits `cross_fit=True` (`tests/studies/multi_arm_ctmle_oat_properties.py`) |
| a nonzero witness | one for each step that can vanish at the truth, as the [Eligibility](#eligibility) rule requires |

Five items stay open. State each verdict separately.

| open item | what a result must settle |
| --- | --- |
| one shared multinomial | one categorical fit on `K` estimated columns supplies every arm's clever covariate, so an inconsistent column for one arm enters the mechanism of every other arm |
| vector target and simultaneous inference | the joint covariance and the simultaneous critical value, not the per-arm variance alone |
| uniformity | the estimator is deliberately superefficient, so a pointwise limit law does not give locally uniform coverage |
| the registered measurement | at `n = 1,000`, the point-treatment generated-design pair passes its own coverage and standard-error calibration bands. Its paired standard-error-ratio difference runs -0.0471 to -0.0209 and is descriptive. The binary OAT study passes 14 of 14 property cells on the bounded law it now runs, and its sharp-null cell passes at the unchanged 800-replication budget, at a rejection rate of 0.0712 with a 99% upper endpoint of 0.0980. The multi-arm pair passes its coverage band but both standard-error intervals cross the 1.07 upper bound, so the multi-arm study passes 10 of 12 property cells. None of these results identifies a first-order term |
| transport beyond the source law | missing-outcome fits need the response-mechanism expansion. Fixed probability weights need a weighted empirical-law result; estimated weights also need a first-stage contribution. Cluster-robust and stratified fits need dependence- and stratum-specific expansions. Repeated cross-fitting needs a result for the package's median and split-dispersion aggregation |

Accept a result only when it covers the exact cross-fitted, multi-arm construction and distinguishes
a fixed target from a target conditional on the learned design. It must establish whether the
current curve suffices or an additional representation contribution is required. The result must
cover the requested joint means or contrasts and their covariance. It must also state its
remainder and nuisance-rate conditions, and it must state a uniformity claim.

If the result requires a new contribution, propagate it through pointwise and simultaneous
inference. Acceptance needs a fixed-design reduction, a generated-design comparison, and
registered coverage evidence for every claimed treatment and target dimension.

[RM20](#rm20-intervals-outside-every-claimed-contract) decided the status. Every
`strategy="oat"` fit takes `generated_design_plugin`, and that includes a fit with `delta=` and an
`ey1`-only request. `ci`, `pvalue`, and `std_error` refuse, and `plugin_std_error` and
`plugin_interval` keep the diagnostic. F19 now holds the route that reopens the interval. For one
binary treatment-specific mean, that route is the literal scalar design that Theorem 1 covers. The
package does not build that design today. For the shipped joint fit, the route is the result that
the acceptance above describes.

### F20. Missing-outcome attributable effects

Wait for a published derivation of PAR and PAF under a declared outcome-response process. The
result must cover the package's observational reference intervention and natural-course mean in
one observed-data law.

The result must give both parent influence curves and their joint targeting equations. It must
also give the exact remainder, nuisance-rate conditions, and treatment and response positivity
conditions. Inference must use same-row covariance for the natural-course and reference parents.

The PAF result must define zero and near-zero denominator behavior. It must also establish an
interval scale under a denominator bounded away from zero.

A future implementation must reduce exactly to complete-data PAR and PAF. It needs nonzero
response-score witnesses for the natural and reference paths. It also needs denominator and
covariance mutation controls plus repeated-sampling evidence on both attributable scales.

Audit every post-fit capability before changing the refusal. Keep unsupported assessments
unavailable with a target-specific reason. Do not infer this construction from existing ATE,
natural-course, or arm-specific results.

### F21. Other missing-outcome CV-TMLE variants

Keep fold-targeted and repeated-split missing-outcome fits refused. The natural-course mean and the
arm-indexed means and contrasts refuse both variants before any learner call. The 2026-09-12 and
2026-09-14 audits found no direct interval result for either composition.

A fold-targeted extension needs a theorem for one response-weighted fluctuation coefficient in
each validation fold and for the resulting stitched influence curve. zEpid code corroborates that
control flow for a related estimator, but code is not an inference result.

The fixed-repeat result in Chernozhukov et al. (2018) applies to the paper's DML estimators under
its DML assumptions. The audit found no result that transports its coordinatewise median and
split-dispersion variance to the targeted MAR plug-in. A future result must cover the dependence
created by targeting before it can justify the package's repeated report for this target.

The stacked contracts retain three follow-ups that are not part of this hard stop. The
[natural-course contract](technical-reference/cv-tmle.md#missing-outcome-natural-course-mean) and
the [arm-indexed contract](technical-reference/cv-tmle.md#missing-outcome-arm-indexed-means-and-contrasts)
refuse each one today. Each needs its own contract and registered evidence. Do not use the evidence
of one extension for another.

| follow-up | missing work |
| --- | --- |
| fold-evaluated construction, for the natural-course mean and the arm-indexed means and contrasts | it has published support in Zheng and van der Laan (2011), Sections 2 and 2.1; it needs an implementation review of its fold plug-in and variance law, which define a separate estimator |
| supplied split plans | an audit of their balance and weighting requirements. Every cross-fitted fit now refuses a plan that carries no package generator record, which the [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) state |
| bounded-continuous stacked natural-course mean | an exact contract for scaling the fluctuation, score, point, and influence curve |

The arm-indexed audit did not examine two sibling surfaces.
[RM20](#rm20-intervals-outside-every-claimed-contract) decided both.

| surface | decision | reopen route |
| --- | --- | --- |
| shift, incremental, regime, MSM, and controlled-direct-effect targets, cross-fitted with missing outcomes | `TMLE` raises `CapabilityError` before any learner. The message names F21, and its remedy is the in-sample fit. A continuous-dose MSM meets the [X10](#x10-continuous-dose-msm-with-a-second-mechanism) refusal first, which names no remedy | F21. Each target needs its own source audit and contract, cross-fitted with missing outcomes |
| in-sample C-TMLE with missing outcomes | the fit reports the status of its path: `working_mechanism_plugin` on a selector path, and `generated_design_plugin` for `strategy="oat"` | [F18](#f18-selector-path-c-tmle-inference) and [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference). F5 holds the missing collaborative theory |

This item asked for the removal of `TestTheMnarTiltFollowsTheDraws` when RM20 closed the first
surface. The refusal deleted that class and its helper from `tests/unit/test_repeated_crossfit.py`,
because the class fitted a cross-fitted controlled direct effect with `delta=` and `repeats=2`. No
admitted composition now fits repeated draws with missing outcomes. So no admitted fit reaches the
multi-draw branch of `missingness_tilt` (`src/cleverly/sensitivity/missingness.py:223-236`). That
branch takes the median of the tilted estimate over the draws, and with one draw it returns the
estimate of that draw. A future repeated-split contract for missing outcomes makes the branch live
again, and it needs its own witness then.

### F22. Grouped cross-fitting beyond point-treatment TMLE

The package draws whole-cluster outer folds for the ordinary cross-fitted point-treatment TMLE and
DR-TMLE. Every other cross-fitted surface refuses `id=`, each for its own stated reason. Two
compositions need their own result before that refusal can be lifted.

| composition | what ships | what a result must supply |
| --- | --- | --- |
| C-TMLE with `id=` | refused at every `cross_fit` setting (`CTMLE._resolve_estimands_for_data`) | a split law for the selection folds and the nested selection folds under clustering, and the cluster-robust variance of the candidate the search stops at. [F18](#f18-selector-path-c-tmle-inference) is open for iid rows, so a clustered result needs that one first |
| cross-fitted longitudinal TMLE with `id=` | refused above one fold (`LTMLE._refuse_cross_fitted_design`). The package permits the in-sample clustered fit. Below 40 positive-mass clusters it takes `few_cluster_plugin` ([RM26](#rm26-longitudinal-clustered-intervals-at-few-clusters)) | the cluster-robust variance of the targeted sequential recursion under a grouped draw. The audit read no source for it |

The grouped point-treatment split itself is supported for the partition alone. Wang, Park, Small
and Li (2024), Section 4.2 and Theorem 4(b), prove a cross-fitted result under a random, roughly
equal partition of the clusters. Their estimator is AIPW-type with a cluster-level treatment. The
rest is the package's own estimating-equation argument, which the
[fold and outcome-scale rules](technical-reference/cv-tmle.md#grouped-folds) state with its four
conditions: independent clusters, equal cluster sizes, no interference, and the usual remainder
rates. The registered
[clustered point-treatment CV-TMLE study](technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md)
is its only empirical witness.

Two boundaries of the shipped grouped split are open.
[RM20](#rm20-intervals-outside-every-claimed-contract) decided both, and F22 holds the route that
reopens each one.

| boundary | decision | reopen route |
| --- | --- | --- |
| the cross-fitted interval at unequal cluster sizes or weight masses | a cross-fitted `TMLE` or `DRTMLE` fit whose clusters differ in row count or weight mass, overall or in a reported stratum, takes `unequal_cluster_plugin`. Its point estimator remains row weighted. The in-sample fit keeps its interval with 40 or more contributing clusters | an expansion and variance result for the current cross-fitted estimator, plus a registered study at unequal sizes |
| the normal reference interval with few clusters, which no source read here supports | a clustered `TMLE` or `DRTMLE` fit with fewer than 40 positive-mass clusters, in the fit or in one reported baseline stratum, in sample or cross-fitted, takes the `few_cluster_plugin` status. So does an in-sample `LTMLE` fit with `id=` and fewer than 40 positive-mass clusters | a $t$-reference claim, with a registered study at few clusters |

Each status keeps the point estimate, and `ci`, `pvalue`, and `std_error` refuse.
[RM26](#rm26-longitudinal-clustered-intervals-at-few-clusters) applied the few-cluster decision to
the in-sample longitudinal fit.

The later source audit found related work, but no derivation for this composition. Nugent et al.
(2024) group folds and aggregate cluster influence curves in partially clustered trials. Balkus,
Laith and Hejazi (2026) show that splitting correlated units can still remove an empirical-process
term under their conditions. Karim (2026) studies point-treatment survey TMLE.

The JSS `ltmle` article covers longitudinal software. Zeileis, Köll and Graham (2020) cover
clustered sandwich covariances for regression models. None supplies this estimator's grouped
longitudinal variance.

[Grouped folds and clustered cross-fitting](references.md#grouped-folds-and-clustered-cross-fitting)
gives every source the audit read, with the version whose locators it used.

### F4. Multi-arm missing-outcome DR-TMLE

`delta=` under `guard=("Q", "g")` continues to refuse more than two treatment arms
(`src/cleverly/estimators/drtmle.py:997-1003`). Díaz and van der Laan (2017), Theorem 2, page 20,
is a scalar result for one arm indicator. Their application estimates each arm mean with its own
indicator (Figure 1, pages 9–10). The paper gives no joint or contrast inference across arms. It
also does not show that a separate logistic tilt of `P(A = a | W)` for each arm (Step 3, page 19;
Equation (6), page 15) stays compatible with one multinomial mechanism. Page 26 mentions
cross-fitting only as a development that "would follow from trivial extensions".

Begin only when a source supplies those results. Existing binary evidence is the regression
surface the extension must preserve.

### F5. Other refused C-TMLE and DR-TMLE compositions

Continue pre-fit refusals for ATT, ATC, PAR, PAF, regimes, incremental interventions, shifts, MSMs,
mediation, and missing treatment where each variant lacks evidence. Ordinary-TMLE implementations do
not establish collaborative or doubly robust inference for these compositions.

C-TMLE with cross-fitted arm-indexed missing outcomes is refused before any learner call. F5 tracks
the missing collaborative theory as a hard stop. In-sample C-TMLE with missing outcomes still fits,
and it reports no interval. [RM20](#rm20-intervals-outside-every-claimed-contract) gave a selector
path the `working_mechanism_plugin` status and `strategy="oat"` the `generated_design_plugin`
status. [F21](#f21-other-missing-outcome-cv-tmle-variants) records the decision.

Each C-TMLE extension needs its target-specific collaborative score and selection-risk contract.
Each DR-TMLE extension needs reduced regressions, a correction, a remainder, and rate conditions.
PAR and PAF also need the joint observed-mean curve and covariance. Complete simulated-confounding
replay receives its own audit only after the estimator can fit the target.

Omitted-variable bounds are now one of these refused compositions.
[RM11](#rm11-sensitivity-bounds-outside-their-derivation) shipped that refusal, and F5 holds the
derivation that would reopen it. A bound on either fit needs an estimate of $\nu^2$ that stays
valid when the estimator does not assume a consistent treatment mechanism. It also needs the
influence curve of the bound under that estimator. A separately fitted full mechanism is a new
estimator of the bound, and it needs the same derivation.

An estimated-weight DR-TMLE fit is not refused.
[RM20](#rm20-intervals-outside-every-claimed-contract) decided that a `DRTMLE` fit with a non-empty
`guard` and varying weights declared estimated (`weights_estimated=True`) reports its point
estimate under the `estimated_weight_plugin` status. `ci`, `pvalue`, and `std_error` refuse.
`guard=()` fits the ordinary TMLE and keeps its interval. So do constant weights, which fit the
unweighted estimator. F5 now holds the route that reopens the interval: the influence contribution of the weight
estimate to the reduced regressions. This inference gap is separate from F11, which tracks
weight-model replay after a perturbation.

## Other extension and investigation contracts

### X2. Replicate-weight designs

Rust and Rao (1996) govern replication variance for complex surveys. Add BRR, jackknife, or
another replicate design only after a source audit matches its construction to this package's
weighted-law estimands and inference conventions.

### F17. Joint point-treatment parameter axes

One ordinary point-treatment fit carries one parameter axis. A working model summarises the
counterfactual means with one score equation per term. A known regime, a modified treatment policy,
or an incremental intervention replaces what those means are. One fluctuation cannot solve both
sets of score equations. The `TMLE` constructor refuses two intervention keywords together, or one
of them with `msm=`. `CausalStudy.identify` refuses a typed estimand whose set holds two kinds
([RM14](#rm14-intervention-refusals-at-identification)). Both refusals come before any model is
fitted.

Wait for a published targeting and inference result for each proposed composition, including its
joint score and covariance. Do not infer the construction from the existing single-axis
implementations. An accepted composition must reduce exactly to each standalone fit, preserve
parameter names and policy definitions, and expose the full cross-axis influence covariance.
Register nonzero controls for every cross-axis block, and repeated-sampling evidence for
simultaneous inference if it is claimed.

### F6. MNAR and incremental-intermediate compositions

An MNAR tilt for continuous-dose shifts and intermediate variables with incremental interventions
wait for identification and influence-function results covering those exact compositions.

The incremental-intermediate refusal raises `CapabilityError` before any learner
([RM24](#rm24-refusals-after-the-nuisance-fit)). The tilt on a
shift fit read available in its capability row and then declined.
[RM23](#rm23-capability-rows-that-read-available-and-then-refuse) corrected that row. Both tilt
rows of a shift fit now read `unavailable` with the sentence that the call raises.

The four items below add methods rather than studies. Each item names the maintained implementation
that a paired study would use. A named comparator is provenance for the construction. It is not the
derivation, and it is not the acceptance gate.

### X4. Sequential doubly robust longitudinal estimation

`lmtp_sdr` implements the sequentially doubly robust estimator of Díaz, Williams, Hoffman and
Schenck (2023). That estimator is consistent when either the outcome regression or the treatment
mechanism is consistent at each time point. It is a second estimator over registered longitudinal
targets, so it adds no estimand. The comparator is the pinned R `lmtp` 1.5.4 that the longitudinal
rows already use, and no new container is needed. Read the rate conditions its interval claims
first, because they differ from the sequential regression conditions.

Theorem 4 of the same article certifies a fold-local recursion for the SDR estimator. Theorem 3
certifies the pooled fluctuation that the cross-fitted TMLE now runs. This item therefore adds a
second estimator, and it no longer answers an open targeting question.

### X5. Natural and interventional mediation effects

`cleverly` reports controlled direct effects only. Natural and interventional direct and indirect
effects are separate estimands, and each carries its own identification assumptions. Díaz, Hejazi,
Rudolph and van der Laan (2021) derive the interventional effects and their efficient influence
function under an intermediate confounder. R `medoutcon` implements a cross-fitted one-step
estimator and a cross-validated TMLE for them, and it pins by commit. Read the identification, the
influence function, the targeting construction, and the interval conditions first-hand. Add each
accepted target to the oracle registry and the evidence gates in both directions.

### X6. Continuous-time survival and competing risks

The shipped survival and competing-risk estimators use discrete time nodes. Rytgaard, Gerds and van
der Laan (2022) derive the continuous-time construction, which changes the intensity model, the
targeting step, and the remainder. CRAN `concrete` 1.0.5 is the comparator, and it implements the
one-step form of Rytgaard, Eriksson and van der Laan (2023).

That comparator takes a binary baseline treatment under a static or dynamic intervention, which
bounds the paired cells a first study can claim. A discrete-time study is not evidence for a
continuous-time interval, so the existing longitudinal rows do not transfer.

### X7. Two-phase and outcome-dependent sampling

A two-phase design measures some variables on a subsample only. An outcome-dependent design samples
on the outcome itself. Each design changes the observed-data likelihood, so each needs its own
influence-function correction.

Hejazi, van der Laan, Janes, Gilbert and Benkeser (2021) derive the
two-phase correction, and R `txshift` 0.3.8 implements it. Van der Laan (2008) derives case-control
weighting under a known prevalence. Fixed observation weights do not replace either correction.
The comparator survey rejects `txshift` as a second opinion on
continuous shifts. That verdict does not carry here, because the two-phase correction is a
different feature.

### X8. Stratified incremental and MSM targeting

Ordinary TMLE refuses stratified incremental targets and stratified nonlinear or continuous MSMs.
DR-TMLE refuses a baseline stratum at a non-empty `guard`, because its reduced regressions add a
second targeting equation for the `mean` group. Each refusal raises `CapabilityError` before any
learner, and `CausalStudy.identify` refuses the incremental and MSM compositions
([RM24](#rm24-refusals-after-the-nuisance-fit)). A post-fit surface cannot repair these upstream
estimator limits.

This item is unwritten work rather than a hard stop. Each parameter is well posed, and the package
already fluctuates baseline strata for arm, regime, and shift targets. The refusal taxonomy
records these refusals as
[not written yet](technical-reference/scope-and-refusals.md#not-written-yet).

Match the stratum-indexed targeting construction to a published derivation before implementation.
Continuous MSMs also need dose-indexed strata semantics. Add the targeting equations and their
validation evidence next. Complete simulated-confounding replay receives its own audit last.

### X9. Omitted-variable bounds on the other linear functionals

[RM11](#rm11-sensitivity-bounds-outside-their-derivation) refuses the omitted-variable bound on
five kinds of fit whose parameter is a linear functional of an outcome regression. Each refusal is
correct, and each message says that the bound is well posed and not implemented. Theorem 2 of
Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026) gives the bias of such a functional, and
Equation (14) in their Section 4 gives the bounds.
[RM22](#rm22-standard-error-of-the-omitted-variable-bound) checked those locators against the
published online appendix and arXiv v6. This project has not read the published main text, so the source
read of this item still needs it.

| fit | the functional | what the contract must add |
| --- | --- | --- |
| `regime` axis | a regime mean, linear in the regression of $Y$ on $(A, W)$ | the representer of each regime, from the fit's clever covariate |
| `shift` axis | a shifted-dose mean, linear in the same regression | the representer of each shift, which reads the treatment density |
| `msm` axis | an MSM coefficient, linear in the same regression | the representer of each coefficient through the projection |
| a response mechanism | the mean of the regression of $\Delta Y$ on $(A, \Delta, W)$ | a representer that carries $\Delta$, and $\sigma^2 = E[\Delta (Y - \bar Q)^2]$ |
| an intermediate variable | a mean at the level $z$ of the regression of $Y$ on $(A, Z, W)$ | a representer with the weight $1\{Z = z\} / P(Z = z \mid A, W)$ |

The response and intermediate representers carry a second mechanism. On those fits the strength
$c_D$ measures a joint strength over the treatment mechanism and that second mechanism. Each
contract must state that reading, and its benchmark must measure the same joint strength.

Each contract names the representer, the $\nu^2$ estimator, and its influence curve. The evidence
needs an exact finite-support law for each representer. It also needs a nonzero witness for each
weight that is zero off its support, as RM11 has for the intermediate representer. DR-TMLE and
C-TMLE fits stay in [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions), and longitudinal fits
stay in [F16](#f16-longitudinal-sensitivity-bound-estimation). The `ipsi` axis stays refused,
because its functional depends on the treatment mechanism.

### X10. Continuous-dose MSM with a second mechanism

[RM32](#rm32-continuous-dose-msm-fit-with-missing-outcomes) refuses a continuous-dose MSM fit with a
missing outcome or an intermediate variable, in sample and cross-fitted. The clever covariate that
the package builds divides by the treatment density at the observed dose and at each dose of the
integration grid. `build_submodel` in `src/cleverly/estimators/targeting.py` builds it. In that
construction, a second mechanism must also divide the covariate at each of those doses. The package
does not predict either mechanism at a grid dose. `_mechanism_designs` and `_mechanism_columns` in
`src/cleverly/estimators/_nuisance.py` predict a mechanism at each arm, or at the observed and
shifted doses, and they have no branch for the grid.

The need for the grid doses belongs to that construction, not to the parameter. A weighted
fluctuation takes the regression weights $\Delta / (g \pi)$ at the observed dose and the covariate
$\varphi(a)$. It needs the second mechanism at the observed dose only. It is the candidate design.

This item is unwritten work rather than a hard stop. The parameter is well posed. The table gives
the sources that the source audit starts from, and what this project has checked in each.

| mechanism or step | source | what this project checked |
| --- | --- | --- |
| the MSM parameter | Neugebauer and van der Laan (2007) define it as a projection of the dose-response curve onto a working model | only at second hand. Section 1 of Kennedy, Ma, McHugh and Small (2017), *Journal of the Royal Statistical Society Series B* 79(4), 1229–1245, cites them for that projection. This project read that section in the author manuscript, PMC5627792 |
| the response weight | Díaz and van der Laan (2017), Section 2.1 and Equation (1), give it for one arm | the [source audit](references.md#point-treatment-and-stochastic-interventions) of the stacked arm-indexed contract records that locator |
| the intermediate weight | none. The package's [controlled direct effect](technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) cites no published derivation. Its evidence is the R `tmle` 2.1.1 comparison of the [controlled direct-effect study](technical-reference/method-evidence/controlled-direct-effect-tmle.md) | the source audit must find a derivation, or record this gap in the contract |
| the coarsening-at-random mapping of a full-data influence function | Kennedy (2020, arXiv:1802.08952) cites Theorem 7.2 of Tsiatis (2006). Baer et al. (arXiv:2306.16571) cite Theorem 8.3 of Tsiatis (2006) and Theorem 1.3 of van der Laan and Robins (2003) | the citing sentences only. This project has not read either book, so each locator is a candidate for the audit |

Match the composition to one published result, or show that it meets the
[Eligibility](#eligibility) conditions. If neither holds, move this item to the future
investigations grid.

Acceptance needs two exact-law Gateaux witnesses: one with a nonzero response weight, and one with
a nonzero intermediate weight. For each weight, a mutation control must drop that weight and fail.
The `continuous` rule of `fit_wide_tilt_refusal` must then name the MSM fit. The cross-fitted fit
stays in [F21](#f21-other-missing-outcome-cv-tmle-variants).

### X11. Learned-policy follow-ups

The [RM30](#rm30-learned-policy-value-evaluation) contract estimates one target by one
construction. Each part below is a published follow-up outside that contract. For parts (a) to
(e) and (h), the RM30 refusal of the request cites the part. No request reaches parts (f) and
(g). Each part needs its own contract, witnesses and registered study
before it ships, as RM30 did.

| part | the request that RM30 refuses | published source | readiness |
| --- | --- | --- | --- |
| (a) the value of one rule fitted on all rows | `cross_fit=False`, or one fold | van der Laan and Luedtke (2015), Section 6 and Theorem 5, under an empirical process condition | published support; pending source read |
| (b) learned longitudinal regimens | `DynamicRegimen(..., rule_kind="estimated")` | van der Laan and Luedtke (2015), Sections 2 and 7, at two time points | published support; pending source read |
| (c) a contrast of the learned-rule value with a known regime or an arm | `learned_rule=` beside `interventions=` or an arm estimand | Theorem 6 of the same paper for each value. The contrast follows by linearity of the fold-local curves, an [Eligibility](#eligibility) step that the contract must record | source audit |
| (d) categorical treatments | a treatment with more than two arms | Nordland and Holst (2026), Section 3.4, for discrete actions | source audit |
| (e) the stacked doubly robust score evaluation | `cv_evaluation=False` | Nordland and Holst (2026), Algorithm 4, which pools the scores and centres the variance at the pooled estimate | published support; pending source read |
| (f) the value of the optimal rule | no request reaches it | van der Laan and Luedtke (2015), Section 7.2, last paragraph; Luedtke and van der Laan (2016), *Annals of Statistics* | published support; pending source read |
| (g) blip and weighted-classification rule learners | no request reaches it, because RM30 has one rule learner | Luedtke and van der Laan (2016), *International Journal of Biostatistics*, Sections 4.1 to 4.3, read first-hand in the author manuscript | published support |
| (h) fold-specific targeting | `targeting_scheme="fold"` | Montoya, van der Laan, Skeem and Petersen (2023), *International Journal of Biostatistics* 19(1):239–259, Section 3.2, Step 2(c), and Section 4.2, read first-hand in the publisher's version | published support |

Part (h) is not the training-fold update that the [Eligibility](#eligibility) section names as new
theory. Montoya and co-authors fit the update on the validation rows, and the package already ships
that update for arm targets.

### F27. Learned-policy value outside the published conditions

The [RM30](#rm30-learned-policy-value-evaluation) contract refuses each request below before any
learner, with three exceptions. The
[boundary study](technical-reference/method-evidence/learned-rule-cvtmle-boundary.md) measures
the exceptional-law row and the `weak_blip` row, and the assessment row reads `unavailable`.
F27 owns every red cell of the boundary study. The table states, row by row, whether a
published result is missing.

| request | missing published result |
| --- | --- |
| an interval at an exceptional law, where the limiting-rule condition C3 of RM30 fails | a CV-TMLE interval for the fold-average target without a limiting rule. Luedtke and van der Laan (2016), *Annals of Statistics*, Section 4.2, name inverse weighting by the standard deviation and a central limit theorem for triangular arrays. Section 5 applies them to the optimal value. No reviewed source applies those tools to this target |
| an interval at the `weak_blip` law of the boundary study | no missing result. That law is within C3, and Theorem 6 of van der Laan and Luedtke (2015) covers it. A red cell there is a finite-sample limit at n = 2,000, and F27 holds it as the owner of the reporting study |
| `repeats` above 1 | each split defines a different target, and no source aggregates over targets |
| the full-refit bootstrap | each resample relearns the rules, so each draw has a different target |
| a continuous treatment | no reviewed source defines a learned rule over a dose for this target |
| `intermediate=` | no reviewed source gives a learned rule for a controlled direct effect |
| `weights=` or `id=` | van der Laan and Luedtke (2015) treat unweighted iid rows, and no source gives a grouped-split learned-rule result |
| `strata=` | no reviewed source gives a stratified learned-rule fluctuation |
| `CTMLE` or `DRTMLE` with `learned_rule=` | no collaborative or doubly robust learned-rule result was reviewed |
| the E-value, the omitted-variable bound, simulated confounding and the missingness tilt | no derivation for this target was reviewed |

## Reading a gap correctly

Not every absence is missing package functionality. The refusal taxonomy in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal) distinguishes an unimplemented
well-posed feature from a different causal question and a method that would be wrong by
construction. Only the first belongs on this roadmap.
