# Roadmap

This is the single planning contract for `cleverly`. It contains proposed work only. Implemented
capabilities belong in the [user guide](user-guide/index.md), scientific contracts in the
[technical reference](technical-reference/index.md) and [DR-TMLE contract](technical-reference/dr-tmle/index.md), validation results in
[evidence manifest](technical-reference/evidence.md), and cross-module standing decisions in the
[architecture invariants](architecture-invariants.md).

The main grid is one binding sequence. Complete lower numbers before higher numbers. Items with no
published theory do not enter this sequence.

## Remediation

This queue is the triage set that must be complete before a beta release. It holds shipped
behavior that needs a correction or a recorded decision, and a published method needed to
resolve a shipped refusal. That behavior includes wrong numbers,
intervals that no claimed contract covers, capability rows that do not match their calls, and late
refusals. Other new capabilities and work that waits on published theory stay in the main roadmap
or in the future investigations grid. Deliver the rows in priority order, and complete every row before
main-roadmap priority 1.

A new capability still needs its own contract and evidence, even in this queue.

The "next action" column states the remediation work. It is not a readiness label.

The 2026-09-18 source audit found no result for the shipped data-dependent outer folds or for the
continuous outcome scale. That work is delivered. The
[fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) hold
the audit record and the rules the package now ships.

This project then regenerated sixteen registered studies under those rules. The pooled
longitudinal update then replaced the fold-local one. RM18 records what each change moved, and
the diagnostics that attribute each move.

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) is the current record.
It lists every red verdict that a registered study publishes. It names the ask that owns each
one, by the `id` in the "What this row asks for" table of RM18.
`tests/unit/test_red_cell_ledger.py` checks the ledger against the committed results. Each red
verdict stays red under a `reporting` policy, so no verdict is hidden and no margin moved.

The ledger delivers the reporting action that RM18 names. RM18 is not complete. Five of its
follow-ups ask for a design declared before its run, and none of those designs is declared or
has run. RM18 lists them in "Deferred findings and review resolutions". The cells that F18 and F19 own
stay red until F18 or F19 meets its acceptance. That needs a published result, or a natural
extension that meets the [Eligibility](#eligibility) conditions.

The 2026-09-13 review of the example notebooks exposed RM11 to RM16. These rows correct defects in
shipped estimators, diagnostics, and messages. Each detail section names its source evidence and
the probe that measured it. RM11, RM12, and RM13 are delivered. Their detail sections record what
shipped.

The review of pull request 223, which delivered RM11 and RM12, recorded problems that it did not
fix. A survey of this roadmap for shipped behavior then found more in other items. RM20 to RM24
hold both sets, and three smaller findings extend RM16. Each detail section gives its probe and
the measured result. RM20, RM21 and RM22 are delivered, and each detail section records what
shipped.

The 2026-09-22 plan for RM20 and RM13 found two sibling surfaces that neither row names.
[RM25](#rm25-declared-stochastic-regime-densities) holds a stochastic regime density that closes
over an estimate. RM25 is delivered, and its detail section records what shipped.
[RM26](#rm26-longitudinal-clustered-intervals-at-few-clusters) holds the few-cluster interval of
the longitudinal path. RM26 is delivered, and its detail section records what shipped. The
delivery of RM20 and RM13 found more late refusals and refusal messages, which RM24 and RM16 now
hold.

The review of that delivery found a third sibling surface.
[RM27](#rm27-declared-msm-design-functions) holds an MSM design function that closes over a sample
statistic. RM27 is delivered, and its detail section records what shipped.

The 2026-09-23 plan for RM25 found a fourth sibling surface.
[RM28](#rm28-declared-densities-of-user-written-interventions) held the density of a
user-written `Intervention` class. RM28 is delivered, and its detail section records what shipped.

The review of the RM26 delivery found a saved-result surface.
[RM29](#rm29-saved-cross-fitted-clustered-longitudinal-results) held a cross-fitted clustered
`LTMLE` result that release 0.1.0 or 0.1.1 saved. RM29 is delivered. The table below lists the
rows that remain.

| priority | item | next action | problem | details |
| ---: | --- | --- | --- | --- |
| 0.31 | Capability rows that read available and then refuse | make each declared row match its call, and add a witness that sweeps the kinds of fit | four rows on three kinds of fit read available, and the call then refuses or raises. On one of them, `assess(include_refits=True)` raises `ValueError` and returns no report | [RM23](#rm23-capability-rows-that-read-available-and-then-refuse) |
| 0.32 | Intervention refusals at identification | refuse mixed intervention kinds in `CausalStudy.identify`, and name the typed estimands in each message. Refuse a zero-dimensional regimen plan by name | a mixed request passes identification and then fails at estimation, once with an `AttributeError` | [RM14](#rm14-intervention-refusals-at-identification) |
| 0.33 | Refusals after the nuisance fit | raise each refusal as `CapabilityError` before any learner call | four well-posed requests refuse after 2 to 20 learner fits. Three of them raise `NotImplementedError` or `ValueError` | [RM24](#rm24-refusals-after-the-nuisance-fit) |
| 0.41 | Calibration-slope warning rule | replace the fixed band with a rule that a registered calibration study supports | the band flagged 14 of 40 fits of a correctly specified weak-signal propensity model | [RM15](#rm15-calibration-slope-warning-rule) |
| 0.42 | Summary and error-message accuracy | correct six display surfaces, three data error messages, one refusal remedy, and two refusal reasons, add a fingerprint-only protocol option, and decide what a bootstrap summary publishes on a non-inferential fit. The simultaneous-request policy is settled below | each surface omits, misstates, or repeats a fact that the fit records | [RM16](#rm16-summary-and-error-message-accuracy) |
| 0.51 | Red property cells after the fold, scale and law changes | keep each red verdict under `reporting` with its interval, and admit inference only when F18 or F19 supplies the exact result. The [red-cell ledger](technical-reference/method-evidence/red-cells.md) delivers this. Then declare and run the five open RM18 follow-up designs, each declared before its run | registered studies publish red verdicts after the fold, scale and law changes and the pooled update. The ledger lists each one and the ask that owns it. Five RM18 follow-up designs are not declared and have not run | [RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) |
| 0.52 | One-sided robustness bias increment in DR-TMLE | investigate the exploratory between-implementation increment on binary `treatment_correct`, under a design declared before it runs | the RM18 reading is `mixed` on that configuration. The unadjusted paired 99% interval of `cleverly` minus R `drtmle` runs 0.000068 to 0.001942, while the Bonferroni interval for that comparison covers zero. No implementation defect is established | [RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle) |
| 0.61 | Learned-policy value evaluation | implement a typed learned-policy target with fold-local training and evaluation, using a published estimating and inference contract | fixed-rule paths refuse a rule learned from the analysis sample; published learned-policy methods give distinct targets and inference | [RM30](#rm30-learned-policy-value-evaluation) |

The 2026-09-22 re-triage ranked the rows by the harm that each defect does to a user today. The
table gives the tiers, from the most harmful. Inside a tier, a row with a wider reach comes first.
A row that another row depends on comes before that row.

| tier | reason | rows |
| --- | --- | --- |
| a | a published number that is wrong, or that no derivation or read source covers. An anti-conservative number ranks above a conservative one | none. RM20, RM13, RM21 and RM22 held it, and all four are delivered |
| b | a crash, an exception that is not a refusal, a capability row that reads available and then raises, or an assessment that returns no report | RM23, RM14 |
| c | a correct refusal that arrives late or as the wrong type | RM24 |
| d | a diagnostic or a warning that misleads | RM15 |
| e | a display or a message that misstates a fact that the fit records | RM16 |
| f | an investigation or a declared design that moves no verdict | RM18, RM19 |
| g | a published method needed to resolve a shipped refusal | RM30 |

The order of the tiers follows the reason that put RM11 and RM12 first. Each of those rows
published a number that no derivation covers. The table gives the reason for each place inside a
tier.

| row | reason for its place |
| --- | --- |
| RM23 | one fit loses the whole assessment report, and four rows on three kinds of fit read available and then refuse |
| RM14 | one mixed request raises an `AttributeError`. It also has a late refusal of tier c, so it takes the higher tier |
| RM24 | four refusals arrive after 2 to 20 learner fits, and three of them have the wrong type |
| RM15 | the warning flagged 14 of 40 fits of a correct model |
| RM16 | each surface misstates or repeats a recorded fact, and no number changes |
| RM18 | five open designs, which read 19 red rows in six studies. The ledger already publishes each of those verdicts |
| RM19 | one configuration, which RM18 opened. Its Bonferroni interval covers zero, and it moves no verdict |
| RM30 | a published learned-policy method can resolve the current refusal, but requires a distinct target, fold-local evaluation, and inference validation |

No open row waits on another open row. RM13, RM20, RM21, RM22, RM25, RM26, RM27, RM28 and RM29
are delivered. The RM23 sweep fits only the kinds of fit that succeed, and the RM14 and RM24 requests
produce no fit.

Main-roadmap X9 depended on RM22, which checked the locators X9 cites. RM22 is delivered.

Use four delivery groups for these eight rows and the two investigations that RM18 waits on.
Keep each item's acceptance criteria separate inside its group.

| delivery group | items | shared boundary |
| --- | --- | --- |
| refusal surfaces | RM23, RM14 and RM24 | a refusal reaches the caller where its declaration says, before the work that it refuses |
| diagnostic reports | RM15 and RM16 | one assessment and summary surface, with one documentation pass |
| red property cells | RM18 and RM19, and the F18 and F19 derivations that RM18 waits on | the recorded rule that a red cell is reporting evidence, designs declared before their runs that move no verdict, and two exact derivations that would close the inferential gaps |
| learned-policy evaluation | RM30 | a typed target and a published fold-local learning, evaluation, and inference contract with validation |

Four groups left this table when their rows were delivered. The sensitivity refusals group held
RM11 and the F5 refusal boundary, and the collaborative inference group held RM12 and the F18
audit. The inference claims group delivered RM20, RM13, RM25, RM26, RM29, RM27 and RM28. Its
boundary was one decision rule for an interval that no derivation covers. The sensitivity
outputs group held RM21 and RM22. Its boundary was the E-value and omitted-variable reports, and
one reading of the published sources they cite.

[F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) now holds the derivation that would
reopen an omitted-variable bound on a DR-TMLE or C-TMLE fit.
[F18](#f18-selector-path-c-tmle-inference) now holds the influence curve that would reopen
selector-path inference.

The re-triage removed two more groups. The pre-fit declarations group held RM13 and RM14. RM13
published a standard error that no derivation covers, so it joined RM20. RM14 refuses a request
late, so it joined the refusal surfaces. The bias localization group held RM19 alone. RM19 joined
RM18, because an RM18 owner holds its three red cells.

Each group holds consecutive priorities, so the group order is the priority order. Deliver the
items inside a group in priority order. The first decimal digit of a priority names its group, and
the second digit orders the rows inside that group. Each remediation priority is below 1, so it
does not collide with a main-roadmap priority.

Priorities give the current delivery order. This project reassigns them when it re-triages the
queue. The RM IDs and their anchors never change, so a commit names a row by its ID. A delivered
row takes its priority with it, and the other rows keep theirs. RM20, RM13, RM25, RM26 and RM27
held 0.11 to 0.15, in that order. RM28 held 0.16, RM29 held 0.17, RM21 held 0.21, and RM22 held
0.22.

Main-roadmap priority 1 waits until every remediation row is complete, as the rule above states.
The queue holds eight rows. Five rows need their corrections: RM14 to RM16, RM23 and RM24.
RM18 has five follow-up designs that are not declared and have not run. RM19 has no declared design. RM30 holds the published learned-policy implementation.

The F18 and F19 derivations do not block priority 1, because an item with no published theory does
not enter the sequence. Their cells stay red under `reporting` until F18 or F19 meets its
acceptance.

The source audit found no result for the shipped global selector or for the complete jointly
targeted outcome-adaptive inference surface. Selector post-selection inference remains in
[F18](#f18-selector-path-c-tmle-inference). The fold-local binary nuisance replacement now has
direct support, while the finite-dimensional joint binary extension and shared-multinomial vector
extension remain in
[F19](#f19-outcome-adaptive-c-tmle-generated-design-inference).

Four additional gaps already have full line items. Keep them there instead of creating duplicate
contracts: competing-event intervention targets in [F3](#f3-additional-longitudinal-estimands), a
multi-arm stress surface in [F8](#f8-multi-arm-simulated-confounding-stress-surface), longitudinal
refutation replay in [F13](#f13-longitudinal-simulated-confounding-replay), and longitudinal
sensitivity bounds in [F16](#f16-longitudinal-sensitivity-bound-estimation).

[F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) tracks the collaborative and doubly robust
families. The RM8 audit moved ordinary TMLE under `missingness=` to
[F20](#f20-missing-outcome-attributable-effects).

The 2026-09-12 and 2026-09-14 audits did not find a direct interval result for a fold-targeted
update or for the package's repeated-split report. The natural-course mean and the arm-indexed
means and contrasts refuse both variants. [F21](#f21-other-missing-outcome-cv-tmle-variants)
tracks them. It also records the retained follow-ups of the arm-indexed stacked contract.

[RM20](#rm20-intervals-outside-every-claimed-contract) decided the two sibling surfaces that the
audit did not examine. The cross-fitted shift, incremental, regime, MSM, and
controlled-direct-effect fits with missing outcomes now refuse before any learner. The in-sample
C-TMLE fit with missing outcomes reports the non-inferential status of its path.

The examples also expose the joint point-treatment parameter axes. No published targeting and
inference result covers that composition, so it is a hard stop in
[F17](#f17-joint-point-treatment-parameter-axes) rather than a row in this queue.

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
- reader-facing behavior, migration, methodology, references, and evidence are updated without
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
[RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) carries the property cells
that went red when sixteen registered studies were regenerated under it.

Replicate-weight designs are the next source-audit item in the main grid. Implement them only after
the remediation rows are complete and that audit supports the planned variance construction.
The [Remediation](#remediation) section states what still blocks them.

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
An estimate of $\nu^2$ is only as good as the fitted representer. The bound has one treatment-side
strength and no response mechanism.

The review found the defects below. A symbol in this section is in
`src/cleverly/sensitivity/omitted_variable.py` unless the text names another file. Before the
refusal shipped, `sensitivity_elements` built $\sigma^2$, $\nu^2$, and the representer from the
nuisances of the reported repeat, in `_elements_for`. `omitted_variable_bounds`, `benchmark`,
`robustness_value`, and `contour_data` all called it. The assessment declared these operations for
`tmle`, `collaborative_tmle`, and `drtmle`, through `_capability` and `SensitivityFacade._declared`
in `src/cleverly/assessment.py`. A probe on a DR-TMLE fit reported each capability as available and
returned a robustness value with no warning.

| defect | mechanism | evidence | consequence for users |
| --- | --- | --- | --- |
| DR-TMLE $\nu^2$ | the default estimator $E[2 m(\hat\alpha) - \hat\alpha^2]$ equals $\nu_0^2 - \lVert \hat\alpha - \alpha_0 \rVert^2$ by the Riesz identity. It is low when the fitted mechanism is wrong, which is the case DR-TMLE guards against. Only the bound's standard error adds the corrected curve (`omitted_variable_bounds` through `_bound_std_error`), and no derivation covers that sum | on the `dr-tmle` notebook draw the review read, the reported $\nu^2$ was 4.34. The sample value of $E[1/g_0 + 1/(1 - g_0)]$ is 7.75 | the bounds are too narrow, and the robustness value is too large |
| C-TMLE $\nu^2$ | a C-TMLE fit holds the selected working mechanism in `repeat.nuisance`. The intercept-only representer $A/p - (1 - A)/(1 - p)$ equals $E[\alpha_W \mid A]$, so its second moment cannot exceed that of $\alpha_W$. $\sigma^2$ still comes from a regression on every covariate | on `make_instrument(n=2000, seed=44)`, the plain fit gives $\nu^2 = 11.77$ and a robustness value of 0.229. The C-TMLE fit gives 4.00 and 0.381 | the collaborative robustness value is optimistic by construction, and $\sigma^2 \nu^2$ belongs to no single conditioning set |
| refusal reason for a non-arm axis | the message says that a fit "whose counterfactuals are not arms does not have" a Riesz representer (`resolve_parameter`) | a regime mean, a shift mean, and a point-treatment MSM coefficient are linear functionals of the regression, and each has a representer | a user reads a missing implementation as a mathematical limit |
| suggestion for an MSM coefficient | a name outside `LINEAR_ESTIMANDS` receives "For a risk ratio or odds ratio use sensitivity.evalue()" (`resolve_parameter`) | a probe on `msm[a]` returns that text. `evalue` refuses an `msm` target (`_select_evalue` in `src/cleverly/sensitivity/evalue.py`) | the message sends the user to a second refusal |
| missing-outcome fits | `sensitivity_elements` has no response-mechanism check. In `_elements_for`, the representer carries the response weight through `nuisance_bound`, $\sigma^2$ averages respondents only, and the strength factor has no response-side term. The assessment offers the bounds, the robustness value, and the E-value on a `PointTreatment(missingness=...)` fit | a probe on a simulated fit with a response indicator reports all three capabilities as available and returns a robustness value of 0.040. The default $\nu^2$ estimator was not positive there, so the code fell back to `"plugin"` without a warning. The `survey-nonresponse` notebook printed a bias-adjusted interval of [0.9567, 1.414] | a user reads a bound that no derivation covers for a fit that models response |
| standardized E-value on a missing-outcome fit | the Gaussian conversion divides by the standard deviation of the observed outcomes (`_standardising_sd` in `src/cleverly/sensitivity/evalue.py`). Under missing at random, the respondents' standard deviation estimates a different quantity from the population standard deviation | the same probe returns an E-value of 1.97 with `sd(Y)` computed from respondents | the standardized scale belongs to the respondents, not to the population the estimate targets |
| documented estimator names | the `omitted_variable_bounds` docstring lists `"auto"`, `"analytic"`, and `"riesz"` | the code accepts `"auto"`, `"doubly_robust"`, and `"plugin"`, and raises `ValueError` for any other value | a documented argument fails |

Apply these corrections:

1. Refuse every omitted-variable operation on a `drtmle` fit, a `collaborative_tmle` fit, or a fit
   with a response mechanism, before any computation. Put the refusal in `sensitivity_elements`,
   so that every entry point shares it. Mark the matching assessment rows `unavailable` with the
   same reason.
2. State the missing result in the reason. For DR-TMLE and C-TMLE, the package has no estimate of
   $\nu^2$ that stays valid when the estimator does not assume a consistent treatment mechanism.
   For a missing-outcome fit, no derivation gives the bound with a response mechanism. The
   missingness tilt remains the sensitivity analysis for response.
3. Refuse the standardized E-value conversion on a missing-outcome fit, and name the population
   standard deviation that it lacks. Keep an E-value computed from a reported ratio, because that
   formula reads only the estimate and its interval.
4. Remove the silent fallback to `"plugin"`. Raise a refusal that names the estimator whose value
   was not positive.
5. Do not substitute a separately fitted full mechanism. That substitution is a new estimator of
   the bound, and [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) records its stop.
6. Rewrite the non-arm refusal. Name the fit's parameter axis. For a regime, shift, or MSM axis,
   state that the bound is well posed and not implemented. For an incremental axis, state that the
   functional depends on the mechanism, so the regression-only bound does not cover it.
7. Suggest `evalue()` only for a `rr` or `or` request.
8. Correct the documented `nu2_estimator` values.

These corrections shipped as follows. One ordered rule table, `_FIT_WIDE_BOUND_RULES`, refuses
every omitted-variable operation on seven kinds of fit. The table gives them in its order.

| rule | the fit it refuses |
| --- | --- |
| `longitudinal` | a longitudinal fit |
| `drtmle` | a DR-TMLE fit |
| `collaborative_tmle` | a C-TMLE fit |
| `response_mechanism` | a fit with a response mechanism |
| `intermediate` | a fit with an intermediate variable |
| `parameter_axis` | a fit whose parameter axis is not the arm |
| `repeats` | a fit that combines more than one cross-fitting draw |

The five entry points share the table. `sensitivity_elements` raises its refusal before any
computation. `omitted_variable_bounds`, `benchmark`, `robustness_value`, and `contour_data` call
`sensitivity_elements` first, so `benchmark` hears the refusal before its refit. Each reason names
the missing result. The response reason points at the missingness tilt, which is the registered
sensitivity analysis for response. The assessment rows are the five that
`OMITTED_VARIABLE_OPERATIONS` names, and each row declares the same reason.

`_select_evalue` refuses the standardized E-value on a missing-outcome fit, so the capability row
reads `unavailable` rather than raising mid-computation. An E-value from a reported ratio still
answers. A nonpositive $\nu^2$ now raises and names the estimator whose value was not positive.
Each per-axis message now names the fit's parameter axis, the `evalue` pointer survives only for an
`rr` or `or` request, and five docstrings now list the accepted estimator names. They are the
docstrings of the five entry points, and each also has a `Raises` section.

The review of this branch changed four rules of the table.

| rule | change | reason |
| --- | --- | --- |
| `intermediate` | added after `response_mechanism` | before the change the bound returned a number on a controlled-direct-effect fit. Theorem 2 of Chernozhukov et al. (2026) covers each fixed-level estimand. The representer carries the weight $1\{Z = z\} / P(Z = z \mid A, W)$, so $c_{f,d}$ would measure a joint strength over the treatment and intermediate mechanisms |
| `repeats` | moved into the table, last | before the move, each capability row read available on a repeated fit and each call raised. A refit with one split lifts this rule and no other rule, so a fit that two rules refuse never sends its reader to that refit |
| `response_mechanism` | reason reworded | the reason said that no derivation covers the case. Theorem 2 covers the regression of $\Delta Y$ on $(A, \Delta, W)$, so the case is well posed. The reason now names the three missing pieces: the representer omits $\Delta$, $\sigma^2$ averages the respondents, and $c_{f,d}$ would be a joint strength |
| `collaborative_tmle` | reason reworded | the reason wrote the representer as $E[\alpha_W \mid A]$, which holds only for an empty selection. It now writes $E[\alpha_W \mid A, V]$. $V$ is the selected set $W_S$ on a selector path and the fitted outcome regression under `oat` |

The same review found an older defect in the doubly robust $\nu^2$ for the ATT and the ATC. Since
commit `6096dbc`, `_m_alpha` weighted the contrast by the fitted propensity $g_c(W) / P(A = c)$. The
score of the functional weights it by the observed $1\{A = c\} / P(A = c)$, which `_elements_for`
now passes. Example 2 of the online appendix of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis
(2026) gives that score, and the `_m_alpha` docstring cites DoubleML's form of it. With a fitted
propensity in that place, the Riesz identity fails and the estimate can exceed $\nu_0^2$. The table
gives the effect on `tests.discrete_law` with $\hat g = 1/2$, where $s$ is the share of the
conditioning arm.

| estimand | $\nu_0^2$ | before, $1/s^2$ | after, $4/s - 1/s^2$ |
| --- | --- | --- | --- |
| ATT | 4.5971 | 5.4083 | 3.8940 |
| ATC | 4.7707 | 3.0779 | 3.9397 |

The ATE and counterfactual-mean values did not move. No registered study imports
`cleverly.sensitivity`, so the fix moves no committed artifact.

No study was regenerated. The refusal exits before estimator construction, so it moves no fitted
array, no influence curve, and no published estimate. The bound, the benchmark, the robustness
value, and the contour remain supported on an ordinary TMLE fit with an arm parameter axis.
[F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) holds the derivation that would reopen them
on a DR-TMLE or C-TMLE fit.

The silent fallback to `"plugin"` reached shipped pages, which this row understated before. The
missing-outcome probe above is one case. Two ordinary TMLE tutorials are the others. The doubly
robust $\nu^2$ is -7.96655 on `point-treatment-tmle.ipynb` and -9.03194 on `cross-fitting.ipynb`.
Each page published a robustness value, a benchmark, and a bias-adjusted interval that the plug-in
estimator computed under a name the page never gave.

Re-executing both pages with `nu2_estimator="plugin"` stated in the code reproduces every stored
number byte for byte, which is the evidence that the fallback produced them. Both pages now state
the estimator and its condition.

`tests/unit/test_omitted_variable_refusals.py` and `tests/e2e/test_sensitivity_and_validation.py`
carry the witnesses. Each witness fails when a component is wrong:

- a test pins each refusal and its message at all five entry points and all five capability rows.
  The fits are DR-TMLE, C-TMLE, missing-outcome, intermediate, `ipsi`, shift, MSM, and repeated
  fits;
- a missing-outcome test pins the refused standardized E-value and the retained ratio E-value;
- a test that a nonpositive $\nu^2$ raises, rather than returning the plug-in value;
- a nonzero witness pins $\nu^2_{\text{intercept}} < \nu^2_{\text{full}}$ on one instrument law.
  A mutation that removes the C-TMLE refusal then reports the smaller value, and the test fails;
- a finite-support law with the wrong mechanism $\hat g = (0.5, 0.3, 0.6)$ shows the DR-TMLE
  shortfall that the refusal prevents. The doubly robust $\nu^2$ is 3.202522675737, which equals
  $\nu_0^2 - E[(\hat\alpha - \alpha_0)^2]$ from the law. The plug-in reads 5.321286848073. The
  earlier form used $\hat g = 1/2$, where both estimators read 4, so it could not fail;
- the ATT and ATC doubly robust $\nu^2$ equal $4/s - 1/s^2$ at $\hat g = 1/2$, and meet the same
  identity. A mutation that reads the fitted propensity again fails four tests;
- a nonzero witness that the intermediate representer is zero off the level $z$ and nonzero on
  it;
- a nonzero witness that the response representer is nonzero on rows with no outcome;
- a message test for each parameter axis, including that an `msm` refusal does not name `evalue`;
- a test that each documented `nu2_estimator` value is accepted.

### RM12. Collaborative intervals at an inconsistent working mechanism

The greedy, ordered, and discrete C-TMLE paths report ordinary EIF plug-in covariance at the
selected working mechanism. The
[technical reference](technical-reference/collaborative-tmle.md) says that the interval treats the
selected candidate as fixed. The measured gap does not come from selection alone.

With a correct linear outcome regression and an intercept-only working mechanism, the C-TMLE
estimate equals the least-squares coefficient on the treatment given every covariate. The reported
variance is then $\sigma^2 \nu^2 / n$ for the intercept-only representer. That variance ignores the
association between the treatment and the covariates.

| measurement | law and fits | result |
| --- | --- | --- |
| one draw | `make_instrument(n=2000, seed=44)`, greedy path | reported standard error 0.0441. The HC0 standard error of the same least-squares coefficient is 0.0545 |
| least-squares sampling spread | `make_instrument(n=2000)`, seeds 1000 to 1999 | empirical standard deviation 0.0538 |
| greedy C-TMLE repeated sampling | `make_instrument(n=2000)`, seeds 3000 to 3299, three outer folds, three selection folds | mean reported standard error 0.0454 against an empirical standard deviation of 0.0537, a ratio of 0.844. Nominal 95% intervals cover in 0.92 of fits. The selector chose the intercept-only mechanism in 172 of 300 fits |

A user reads the smaller standard error as a precision gain, and the interval undercovers.

The registered selector study now points the same way. Its forced-selection cell reports a
standard-error ratio of 0.8011 under unstratified selection folds, where the treatment-stratified
folds it used before reported 1.2539. The study reports that ratio and does not gate on it
(`tests/studies/ctmle_selector_properties.py`).
[RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) records the change.

The path record has a second defect. The `CTMLESelection.train_risk` docstring and the module
docstring of `src/cleverly/estimators/ctmle.py` said that the in-sample risk does not increase.
The search in `_Selector._forward_path` takes one more fluctuation step when no addition lowers
the risk. If the next addition still raises it, the path adds that covariate anyway. The
`collaborative-tmle` notebook output shows `risk` rising from 6.56613 to 6.56746.

Apply these corrections:

1. When a result exposes the ordinary curve at the selected working mechanism as a diagnostic,
   label it in the result summary, in `to_frame`, and in the assessment. The label says that no
   result shows this curve is the estimator's influence curve when that mechanism is not consistent
   for the treatment law. It must not render as a confidence interval or p-value.
2. Refuse confidence intervals and p-values for the greedy, ordered and discrete paths until F18
   supplies the estimator's influence curve. Keep the point estimate and path diagnostics. A
   clearly named working-mechanism plug-in standard error may remain only as a diagnostic, not as
   inferential output. Van der Laan and Gruber (2010), Theorem 4, assumes rather than proves the
   expansion needed for the selected mechanism, and Section 4.3 leaves random cross-validation
   over-selection open. Cui and Tchetgen Tchetgen (2024), Section 6 of arXiv:1911.02029v6, make a
   related point: a Wald interval at selected learners is blind to model selection, although they
   do not treat an inconsistent working mechanism. Ju, Benkeser and van der Laan (2020), Theorem
   1, show that an intentionally inconsistent propensity adds a first-order term $D_r$.
3. The audit found that the forced addition matches van der Laan and Gruber (2010), Section 2.3,
   and the pinned R `ctmle` construction. The source and both docstrings now say that the
   unpenalized likelihood fluctuation cannot worsen its own loss while the recorded penalized
   selection risk can rise. The path itself does not change.

Corrections 1 and 2 shipped as follows. `ParameterEstimate` carries an `inference` field. `CTMLE`
sets it to `working_mechanism_plugin` for the greedy, ordered, and discrete paths, and leaves it at
`influence_curve` for `oat`. [RM20](#rm20-intervals-outside-every-claimed-contract) later gave every
`oat` fit the `generated_design_plugin` status. `ci`, `pvalue`, and `std_error` then raise
`CapabilityError`. `plugin_std_error` and `plugin_interval` report the retained diagnostic, from the
same private body the refused accessors call, so the numbers did not move. `summary()`, the
`to_frame` columns, the assessment detail, and the nuisance verdict all carry the label. A contrast,
a simultaneous band, every E-value branch, `variable_importance()`, and `tipping_gamma(use_ci=True)`
refuse on these paths. `variable_importance()` refuses at its entry point, because it adjusts one
p-value per candidate and had fitted every candidate before it read one. The two point-estimate
sweeps keep running instead. `truncation_curve()` and `missingness_tilt()` rename their three spread
columns, and the default `tipping_gamma()` searches the point estimate.

The refusal keys on the strategy, as correction 2 requires. A `discrete` fit with a single
full-adjustment candidate is therefore refused although it is bit-identical to a plain TMLE fit.
The [technical reference](technical-reference/collaborative-tmle.md) records that over-refusal and
names `TMLE` as the workaround.

The review of this branch found surfaces that the refusal did not reach. One table,
`influence.spread_name`, now gives the name that each spread takes under each status. It raises
for an inferential name that has no diagnostic name. The table gives each surface on a selector fit.

| surface | behavior on a selector fit |
| --- | --- |
| bootstrap percentile limits | `bootstrap_range_lower` and `bootstrap_range_upper`, because `summary()` calls them a range |
| repeat spread of `nuisance_diagnostics` | `plugin_standard_error` and `ratio_to_plugin_standard_error` |
| `RefutationTest` frame | `plugin_std_error`. `RefutationTest` carries `inference` |
| `CoverageStudy` summary | `mean_plugin_std_error`, and each summary row carries `inference`. The verdict states that the coverage column certifies no confidence interval |
| split-noise lines of `summary()` | each share is "of plugin_std_err" |
| a fit saved before the `inference` field | `TMLEResult.__setstate__` re-stamps each estimate from the estimator that the artifact carries, and drops the saved bands and assessment answers. The loaded fit refuses `ci` |
| `tipping_gamma(use_ci=True)` capability row | resolved per request. It reads `unavailable` with the refusal sentence, where it read available and then raised |
| E-value | `_select_evalue` refuses a selector fit after its key and axis checks, so an `ey1` request reports `not_applicable` |
| `variable_importance()` | asks the estimator's `_inference_status()` rather than the strategy |

`TMLEResult.inference_status` is the fit-level status. `summary()`, the band decision in `fit`, and
`nuisance_diagnostics` read it. Only names moved, and only on selector fits. Every renamed column
reads the private body that it read before, so no number and no committed artifact moved.

No study was regenerated. The reframing changed no arithmetic, so the two registered selector rows
keep their committed replicate files and report what those columns now measure. The ratio E-value
that [RM11](#rm11-sensitivity-bounds-outside-their-derivation) correction 3 keeps is
unavailable on a selector path, because it reads an interval that these paths do not supply. RM11's
sentence therefore means "keep it for a fit that has an interval".

This row planned three witnesses. Each had to fail when a component is wrong. The table gives the
state of each.

| witness | state |
| --- | --- |
| 1. a discrete fit with only the intercept-only candidate, on a finite-support law with a correct outcome regression. The reported curve variance and the exact variance of that fixed-candidate estimator differ by a stated margin | delivered as `TestTheWorkingMechanismDiagnosticMissesTheExactVariance` in `tests/unit/test_ctmle.py` |
| 2. a greedy path on a law where `train_risk` rises, which pins the documented behavior | delivered as `TestTheRecordedRiskMayRiseWhileTheLossMayNot` in the same file, on `make_instrument(n=800, seed=11)` |
| 3. a registered coverage cell on the instrument law, with its thresholds fixed before the final run | retired. The reasons follow |

Witness 1 uses a variant of `tests.discrete_law` that 1,000 rows realise exactly. $W$ moves the
treatment probability from 0.2 to 0.8. With saturated trees, the fit equals the saturated plug-in
on the exact frame and on a resample. Its exact variance is therefore $P_0 D^{*2} = 1.4020$, from
the Gateaux oracle `law.eif`. The reported diagnostic is $P_0 D^*(\bar Q_0, \pi)^2 = 0.8896$, a
ratio of 0.634522 against a declared bound of 0.8. The saturated full-adjustment candidate is the
control, and it reads a ratio of 1.

Two mutations fail witness 1. A discrete path that fits every covariate fails two tests. A status
of `influence_curve` on the selector paths fails one. The HC0 comparison on
`make_instrument(n=2000, seed=44)` stays in `tests/e2e/test_ctmle.py` as corroboration on a
continuous law.

Witness 3 is retired for three reasons.

| reason | detail |
| --- | --- |
| no interval to test | the refusal removed the interval that the cell would test. A selector fit publishes no `ci` |
| no failure mode | the cell can measure only the retained diagnostic. A mutation that removes the refusal leaves the diagnostic unchanged, so the cell cannot fail on the defect |
| existing measurement | the registered cells that the F18 acceptance names already measure the calibration of the diagnostic |

The table gives those cells. Each has $n = 1500$. The SE ratio is the mean reported diagnostic
standard error over the empirical standard deviation of the estimates.

| study | cell | replicates | SE ratio | coverage |
| --- | --- | --- | --- | --- |
| `ctmle_selector` | `selector_necessity/collaborative` | 800 | 0.8011 | 0.9500 |
| `multi_arm_ctmle_selector` | `selector_necessity/greedy` | 400 | 0.5913 | 0.7750 |
| `multi_arm_ctmle_selector` | `selector_necessity/ordered` | 400 | 0.6245 | 0.8425 |

Coverage alone does not detect the gap. The point-treatment cell covers at 0.95 while its SE ratio
is 0.8011, so the SE ratio carries the evidence. The values come from `properties.csv` in
`tests/canonical/ctmle_selector/` and `tests/canonical/multi_arm_ctmle_selector/`.

### RM13. Estimated MSM projection weights

The [MSM technical reference](technical-reference/msm-projections.md#variations) listed a
projection weight derived from the estimated mechanism as refused. The same row said that such a
weight "does not fail" and gives a standard error that is too small. Before this row shipped, the
code checked only that `weights` was callable (`src/cleverly/msm.py:373-378` at 47e5648), and a coverage
pragma marked that branch as unreachable. `tests/unit/test_msm.py` reached it.

A callable receives an arm label and a covariate frame, and it can close over any estimate. A
callable over sample shares or a fitted mechanism therefore fitted with no message. The weight is
then a functional of $P$, and the influence curve omits its pathwise derivative. The standard error
can be wrong. The exact-law witness below gives a case where it is too small.

The package cannot inspect a closure, so the status of the weight is now a declaration. The row
asked for three corrections:

1. Require an explicit known-weight declaration on `MSM` whenever `weights` is set. Refuse
   `weights` without it before any nuisance fit. `PointTreatment.weights_estimated` is the
   precedent for declared observation weights (`src/cleverly/study.py:407`).
2. Refuse a weight declared as estimated. The message names the missing pathwise-derivative term.
3. Correct the reference row, so that it describes the declaration and both refusals.

All three shipped. One function, `cleverly.msm.refuse_projection_weights`, runs the checks in the
order of the table. [RM27](#rm27-declared-msm-design-functions) later renamed it
`refuse_msm_functions` (commit 03917d1), and it now checks the design first.
`MSM.__post_init__` calls it. `TMLE._resolve_estimands_for_data` and
`LTMLE.fit` call it again before any learner call, because a restored or modified model can carry
a declaration that this version refuses. The RM27 review added calls in `MSMSet.evaluate` and
`evaluate_regimen_msm` (commit e360555).

| part | what shipped |
| --- | --- |
| declaration | a field `weights_kind` on `MSM`. It takes `"known"`, `"estimated"`, or `None`, and it defaults to `None`. `MSM.linear` gains the same argument and forwards it |
| field order | `weights_kind` is the last field, after `doses`, so `link` stays the fourth positional argument. The delivery first put the field after `weights`. `MSM(design, terms, None, "log")` then raised `DataError`, because `"log"` landed in `weights_kind`. The review moved the field last (commit 8a33661), and `tests/unit/test_msm_projection_weights.py` pins the order and the positional link |
| an inconsistent declaration | `weights_kind="estimated"` with no `weights` raises `DataError`, and so does a value other than `"known"`, `"estimated"`, or `None`. `None` or `"known"` with no `weights` passes, because uniform weights are known |
| an array in place of a callable | `CapabilityError`. An array is one evaluation of the weight, and nothing shows that the sample did not set it. The message names both callables: `(arm_label, covariate_frame) -> (n,)` for a point treatment, and `(regimen_label, horizon, baseline_frame) -> (n,)` for a longitudinal regimen MSM |
| undeclared weight | a callable `weights` with `weights_kind=None` raises `CapabilityError`. The message asks for `weights_kind="known"` |
| estimated weight | `weights_kind="estimated"` raises `CapabilityError`. The message names the missing pathwise-derivative term |
| exception type | `CapabilityError`, which the [architecture invariants](architecture-invariants.md#public-causal-workflow) give to a well-posed composition that the package refuses by name. `refuse_unsupported("estimated_weights")` raised `NotImplementedError`. It now raises the same `CapabilityError` text |
| fit-layer check | `TMLE` and `LTMLE` call the function again before any learner call. `TMLE._retarget_detailed` calls it first too, so every recomputation checks it (commit a2f62a6) |
| a result saved before the field existed | it loads undeclared. Before RM28, loading checked nothing, so its stored estimates answered as saved. Every recomputation refuses: `retarget()`, each sweep that calls it, such as `truncation_curve()`, and `refute()`, which refits. `LTMLE` has a truncation sweep, `longitudinal_truncation_curve`, which commit 4a8b09b added on 2026-09-10. A saved `LTMLE` MSM result holds arrays and no `MSM` object, so that sweep calls no design and no weight, and its curve publishes no spread. `tests/unit/test_longitudinal_truncation_refit.py` pins the columns of that curve. This record first said that `LTMLE` has no retarget sweep, and [RM27](#rm27-declared-msm-design-functions) corrected it. The review probe was `make_binary_outcome(n=400, seed=3)` with `MSM.linear` and the weight $1 + a$, with `weights_kind` then set to `None`. Before the correction, its `truncation_curve` published `std_err`, `ci_lower`, and `ci_upper`. It now raises `CapabilityError`. At RM27 the stored `msm[a].ci` stayed (0.07496, 0.25730). [RM28](#rm28-declared-densities-of-user-written-interventions) later gave such a `TMLE` result the `undeclared_function_plugin` status. Its point estimates answer as saved, and `ci`, `pvalue` and `std_error` refuse (commit ed8e73e) |
| replay | `_freeze_msm` (`src/cleverly/sensitivity/_simulated_confounding_fixed.py`) replaces `weights` with a callable over frozen arrays, even when the model had no weights. It passes `weights_kind="known"` when the model had no weights, and the model's own declaration otherwise. A legacy callable-weight model with no declaration therefore still refuses at replay. RM27 moved that refusal before `MSMSet.evaluate` runs the design or the weight (commit 67ab7fc). The RM27 review then moved the check into `MSMSet.evaluate` itself and removed the replay's own call (commit e360555). The `replace` call still passes the model's own declaration, so it refuses as a backstop |
| reference | the [MSM reference](technical-reference/msm-projections.md#variations) tabulates the declaration, both refusals, and the ratios below. The [scope page](technical-reference/scope-and-refusals.md) marked the `Stochastic` row "not refused yet" and linked [RM25](#rm25-declared-stochastic-regime-densities). RM25 has since shipped that refusal |

The declaration is a three-state `Literal` and not a bool. Item 1 needs an undeclared state that
refuses. `PointTreatment.weights_estimated` is a bool with the default `False`
(`src/cleverly/study.py:407`). That default is safe there only because the conditional-on-weights
argument covers ordinary TMLE. A projection weight has no safe default. A `bool | None` field
would let `if model.weights_estimated:` accept `None`, which is the bypass that this row closes.

The default of the field is a plain class attribute. A model pickled before the field existed
therefore loads as undeclared, and `dataclasses.replace` still works on it. The name does not
collide with `WeightKind` in `src/cleverly/data/weighting.py`, which describes observation
weights.

The row first asked for a repeated-sampling control with weights equal to the sample arm shares.
An exact-law witness replaced it. The [testing strategy](development/testing-strategy.md) keeps a
claim that needs repeated sampling out of the fast suite, and it admits exact-law and Gateaux
instruments there. The exact-law gap is the asymptotic form of "a reported standard error below
the sampling standard deviation", and it is deterministic.

The witness law has $W$ on three levels with probabilities 0.4, 0.2, and 0.4. The treatment
probability is 0.5 at every level. The binary outcome means are (0.05, 0.05), (0.3, 0.7), and
(0.05, 0.95) for arms 0 and 1. The fit is in sample, with oracle nuisances and the identity-link
design $[1, a, W]$. The weight callable returns the sample arm share and declares it `"known"`. A
declaration cannot detect that false statement, so the witness measures what it costs.

| term | reported standard error over the exact standard error | role |
| --- | --- | --- |
| `msm[W]` | 0.7417 | the witness. The test asserts a bound of 0.8 |
| `msm[(intercept)]` | 0.9126 | recorded |
| `msm[a]` | 1.0000, to 1e-9 | the negative control. The derivative of the `a` coefficient in the arm share is zero on this design |

The exact standard error comes from the Gateaux derivative of the projection with
$h(a) = P(A = a)$ recomputed from the perturbed law. The estimated-share curve differs from the
fixed-weight curve by exactly $\partial\beta/\partial\pi \, (1\{A = 1\} - \pi)$, with
$\partial\beta/\partial\pi = (-0.45, 0, 0.45)$. A second control checks that the reported curve
equals the fixed-weight Gateaux curve to 1e-10. The measured difference is 6e-15, so the curve is
right for the functional that the user declared. The ratio uses the second moment of the curve.
`std_error` divides by $n - 1$, which would move the ratio by a factor of 1.0005.
`tests/discrete_law.py` now computes the identity-link projection through `msm_beta`, and the
oracle values of every `msm` name stayed bitwise identical.

The row planned three witnesses. The table gives the state of each. All of them are in
`tests/unit/test_msm_projection_weights.py`.

| witness | state |
| --- | --- |
| 1. a pre-fit test pins both refusals and their messages, and a spy learner shows that no nuisance fit ran | delivered. `TMLE` and `LTMLE` refuse a model whose `weights_kind` is set to `None` or `"estimated"` after construction. `NeverFit.calls` is 0, and a spy weight is never called. A control with `"known"` reaches the first learner |
| 2. a repeated-sampling control with weights equal to the sample arm shares | replaced by the exact-law witness above |
| 3. the exact-law witness | delivered. `msm[W]` reads 0.7417 against the bound of 0.8 |

Three committed mutations fail as they must. Removing the declaration check fails every
declaration witness. Removing the `TMLE` and `LTMLE` fit-layer calls fails the fit witnesses, and
the spy learners record fits. A replay that drops the declaration refuses. Since commit e360555,
the fit-layer mutation also removes the evaluator checks. A `TMLE` point fit evaluates the weight
before its first learner, so `MSMSet.evaluate` would refuse first.

No study was regenerated. The generators `tests/studies/canonical_point_msm.py` and
`tests/studies/canonical_longitudinal_msm.py`, and every other MSM call site that passes
`weights=`, now declare `weights_kind="known"`. That change moves no fitted number. The
`msm-projections` notebook `--check` reproduced every stored output.

### RM14. Intervention refusals at identification

The typed estimands do not check the kinds of their elements. `RegimeMean.regimens` and
`RegimeContrast.regimens` are typed `Any` (`src/cleverly/study.py:955`, `:993`), and
`IncrementalMean.interventions` and `IncrementalEffect.interventions` declare
`Sequence[Incremental]` with no runtime check (`:1064`, `:1106`). The refusal
happens only when the estimator runs `as_interventions` (`src/cleverly/interventions/base.py:558-584`).

| request | behavior | evidence |
| --- | --- | --- |
| `RegimeContrast(regimens=(Static(1), Incremental(2.0)))` | `identify` returns an `IdentifiedEffect`. `estimate` raises the `ValueError` from `refuse_unsupported` | probe, and `base.py:347-358` |
| `IncrementalEffect((Incremental(1.0), Static(1)))` | `identify` succeeds. `estimate` raises `AttributeError: 'Static' object has no attribute 'delta'` | probe |
| a `Shift` in a regimen set | the message tells the user to pass it to `TMLE(shifts=...)` | `base.py:359-366` |

The messages name the low-level keywords `TMLE(incremental=...)` and `TMLE(shifts=...)`. The top
level does not export `TMLE` (`src/cleverly/__init__.py:105-166`). A `CausalStudy` user declares
`IncrementalMean`, `IncrementalEffect`, `ModifiedTreatmentPolicy`, or
`ModifiedTreatmentPolicyEffect`. The incremental message also says that the object is "not an
intervention", although the user imported it from `cleverly.interventions`.

A user who declares a mixed request learns of the refusal only after the study is built. An
`AttributeError` is a crash, not a refusal.

The review of RM28 found a longitudinal sibling. `LTMLE({"x": np.array(1)}).fit(...)` raises
`TypeError: iteration over a 0-d array` from `refuse_regimen_rules`, before any learner. A probe
gave the same error at a0e93bb6 and at commit 75e86be. The review fixes of
[RM28](#rm28-declared-densities-of-user-written-interventions) record it.

Apply these corrections:

1. Check the element kinds when `CausalStudy.identify` builds the functional. Refuse a `Shift` or an
   `Incremental` in a regimen set. Refuse any element other than `Incremental` in an incremental
   estimand.
2. Name the typed estimand for each kind in the message. Cite
   [F17](#f17-joint-point-treatment-parameter-axes) for a joint request.
3. Keep `as_interventions` as a second guard for direct estimator use, and name the typed estimands
   there too.
4. Refuse a `regimens=` plan that is a zero-dimensional array with `DataError`. Name the sequence
   form and `DynamicRegimen` in the message.

Put the check in `_identify_point`, not in `_point_functional`.
`_matches_registered_point_identification` calls `_point_functional` inside a `try` that catches
`CapabilityError` and returns `False`. A refusal there would become a silent mismatch.

The witnesses must fail when a component is wrong:

- a pre-fit test for each row above pins the refusal, its message, and its raise at `identify`;
- for the `IncrementalEffect` row, a spy learner shows that no nuisance fit ran. A 2026-09-22
  probe recorded one learner fit before the present `AttributeError`, so this spy can fail. The
  `RegimeContrast` refusal already runs before any learner fit, so a spy cannot fail on that
  row, and only the raise at `identify` witnesses it;
- a mutation that removes the identification check makes the test fail at `identify`, rather than
  pass at `estimate`;
- a pre-fit test pins the refusal of the zero-dimensional plan, its message, and zero learner fits.

### RM15. Calibration-slope warning rule

`NuisanceDiagnostics.findings` warns when a calibration slope falls outside [0.7, 1.4]. The finding
says that the model "biases the weights" (`src/cleverly/validation/nuisance.py:571-575`). The
assessment turns any finding into a `nuisance_models` warning (`src/cleverly/assessment.py:2653-2658`).

The band ignores the spread of the true probabilities. With a weak signal, noise in the out-of-fold
coefficients spreads the predictions more than the truth does. A correct model then gives a slope
below 1. The finding also asserts an effect on the weights that the rule does not measure.

| law | fits | result |
| --- | --- | --- |
| point treatment, $n = 2000$, $\operatorname{logit} g_0 = 0.15 W_1$, correct logistic learner, three folds | 40 seeds | mean slope 0.647, median slope 0.738, mean AUC 0.537. The rule flagged 14 fits |
| longitudinal censoring at node 2, `make_longitudinal(n=8000)`, correct logistic model, a separate three-fold split | 60 seeds | out-of-fold slope mean 0.767, standard deviation 0.114 |

The longitudinal nuisance report applies no band (`assessment.py:2594-2604`), so the second row
raises no warning. Its `calibration_slope` column also holds two statistics under one name. Binary
rows use a logistic recalibration slope, and pseudo-outcome rows use a linear regression slope
(`nuisance.py:824` and `:859`).

A user who sees the warning respecifies a correct model. The warning does not say how to tell weak
signal from miscalibration.

Apply these corrections:

1. Do not tune a new constant to these draws.
2. Register a calibration study before a new rule ships. One law is correct with weak signal and
   measures the false-warning rate. One law is miscalibrated and measures the detection rate. Fix
   the thresholds before the final run.
3. Until that study exists, word the finding as a prompt to review the model. Report the AUC and the
   largest weight beside the slope, and do not assert that the weights are biased.
4. Give the linear slope in the longitudinal report its own column name.

The witnesses must fail when a component is wrong:

- a unit test on the correct weak-signal law pins the outcome of the registered rule;
- a mutation back to the fixed band makes that test fail;
- a miscalibrated law shows that the rule still warns.

### RM16. Summary and error-message accuracy

| surface | defect | evidence | correction |
| --- | --- | --- | --- |
| longitudinal `RegimeMean` result summary | prints `reference: always`, although the result keeps no contrast | `src/cleverly/study.py:2883-2890` drops the contrasts. `src/cleverly/longitudinal/estimator.py:608-611` prints the reference unconditionally. A probe prints `reference: always` beside `ey_regimen[always]` and `ey_regimen[never]` only | clear the reference when the study keeps no contrast, and print the line only for a contrast |
| longitudinal identification summary | the `adjustment/history` line lists only the baseline covariates | `study.py:2270` sets `adjustment=tuple(design.baseline)`, and `:2586` prints it. A probe prints `['W1', 'W2']` for a design with `L2` at node 2 | print the history for each node |
| DR-TMLE result summary | names neither DR-TMLE, the guard, nor the reduction | `src/cleverly/estimators/base.py:1367` fixes the header, and `config.describe()` (`:195-263`) has no method line. A probe finds `fitted_method` equal to `drtmle` and none of `drtmle`, `DR-TMLE`, or `guard` in the summary | add the method, the guard, and the reduction |
| regime support table | `min g`, `max ratio`, and `ratio effective n` use the mechanism before truncation. `score load` uses the truncated weights. The table does not say so | `src/cleverly/interventions/support.py:276-294` and the header at `:244-260`. In the `interventions` notebook, offer to all prints a `min g` of 0.001463, below the fit's bound of 0.0114 | label the basis of each column. The source chooses the untruncated mechanism on purpose, so this is a display gap and not an estimator defect |
| identification and result summaries | each summary reprints all 11 protocol lines, so a notebook that prints the protocol, the identified effect, and the result shows the record three times | `StudyProtocol.summary_lines` returns 11 lines (`src/cleverly/protocol.py:180-200`). `IdentifiedEffect.summary` and `IdentifiedEffect.summary_lines` append them (`src/cleverly/study.py:2569-2589`, `:2591-2610`), and the result summary appends `summary_lines` (`src/cleverly/estimators/base.py:1377-1378`). The `dr-tmle` notebook shows the block three times | add a summary option that prints only the `causal study protocol: schema N; fingerprint` line. Keep the full record as the default, so that a summary read alone stays complete |
| missing-outcome `DataError` | tells a study user to pass `delta=<column>` | `src/cleverly/data/validate.py:311`. `PointTreatment` names the field `missingness` (`src/cleverly/study.py:403`). A probe through `CausalStudy` returns the `delta=` text | name `missingness=` for a study design. Keep `delta=` only where the low-level `CausalData` constructor raises the error |
| split-spread fact of the `nuisance_models` assessment row on a selector fit | says "largest sd/se", while `summary()` of the same report says "sd/plugin se" | `_nuisance_item` in `src/cleverly/assessment.py` writes fixed text. A greedy fit of `make_instrument(n=600, seed=44)` with `repeats=2` prints "largest sd/se 0.0743 for ate" beside the working-mechanism note | read the name from `influence.spread_name`, as the other surfaces do |
| an explicit `simultaneous=True` on a fit that supplies no inference | the fit builds no band and raises no warning. Only `summary()` states the omission | a greedy fit of the same law with three estimands records no warning. Its summary prints "no simultaneous bands: a band is a joint confidence statement, and this fit reports none." The default is `True`, so the fit cannot tell an explicit request from the default | keep the present behavior. The fit makes no band and its summary names the omission. A warning would fire on every default non-inferential fit. The [collaborative reference](technical-reference/collaborative-tmle.md) and [inference status](technical-reference/inference.md#inference-status) record this policy |
| in-sample C-TMLE selection-fold refusal | tells the caller to fit in sample with `cross_fit=False`, and the fit is already in sample. The downstream nuisance error says the same | `CTMLE(strategy="greedy", cross_fit=False, selection_folds=6, q_bounds=(0, 1))` with `delta=` on `respondents_in_one_fold()` from `tests/unit/test_fold_policy_rules.py`, at `random_state=1146`. The fit raises `DataError` "C-TMLE selection cannot fit its nuisances because repeat 0, fold 2's training complement contains no row with an observed outcome", which ends "Either fit in sample with cross_fit=False on the engine". With `TMLE._check_training_support` patched out, the nuisance fold loop raises `ValueError` "a cross-fitting fold has no trainable rows for a nuisance model", which says "Fit in sample instead (cross_fit=False on the engine ...)". The remedy is `_IN_SAMPLE_REMEDY` in `src/cleverly/learners/crossfit.py` and the text in `src/cleverly/estimators/_nuisance.py` | name the selection folds as the split that failed, and give a remedy that is true for an in-sample fit |
| `DRTMLE` class docstring | describes an open centring defect on a quarter of splits, and names the test class `TestTheReportedCurveIsNotAlwaysCentred` | no test class has that name. `TestTheReportedCurveIsCentredWhereTheBoundBinds` in `tests/unit/test_drtmle_fit.py` records the fix, which solves the score at the truncated tilt. The docstring of `TestEachDrawSolvesItsOwnEquations` names the old class too | describe the fixed state, and name the present class in both docstrings |
| bootstrap summary on a non-inferential fit | `estimate.bootstrap.ci` answers, and `to_dict` emits `bootstrap_std_err`, under inferential names on a fit whose status supplies no inference | `BootstrapSummary` in `src/cleverly/inference/influence.py` is a plain dataclass, and `ParameterEstimate.to_dict` writes `bootstrap_std_err` whatever the status. `to_dict` already renames the percentile limits to `bootstrap_range_lower` and `bootstrap_range_upper`. RM12 kept the `bootstrap_std_err` name on purpose, and the [collaborative reference](technical-reference/collaborative-tmle.md) says so | decide what the bootstrap publishes on such a fit, and record the decision in the collaborative reference and in [inference status](technical-reference/inference.md#inference-status). A refusal at `.bootstrap.ci` would break the `ci=` keyword of the `BootstrapSummary` constructor, every reader of that field, and results pickled before the change |
| continuous-treatment `DataError` | the error for a continuous treatment with no `shifts=` and no `msm=` offers `Shift(0.0, cap=None)` as "the natural course". A cross-fitted fit with `delta=` then meets the F21 refusal of a shift target, whose remedy is the in-sample fit | `TMLE._check_shifts` in `src/cleverly/estimators/tmle.py`. The review probe of the F21 refusal (commit 45f072b) fitted `make_missing_outcome_binary(n=400, seed=4)` with a continuous dose and `Shift(0.0, cap=None)`, cross-fitted with `delta=`. It refused before any learner. Its in-sample fit reports `ey_shift[natural course]` 0.51256 | on a cross-fitted fit with `delta=`, name the in-sample fit beside the natural course, so that the suggested request does not meet a refusal |
| longitudinal plan written as a mapping | `{"x": {"t1": 1, "t2": d}}` resolves to `Regimen('x', t1/t2)`, with the dictionary keys as arms. The fit then raises `DataError`: "regimen 'x' assigns 't1' at time 1". The message names a label that the user did not mean as an arm, and nothing checks or calls `d` | `_plan_nodes` in `src/cleverly/longitudinal/regimen.py` reads any iterable that is not an iterator as a tuple of its items. The review of RM28 probed the plan at a0e93bb6 and at commit 75e86be, with the same result | refuse a mapping plan by name, and name the sequence form and `DynamicRegimen` in the message |
| derived risk-ratio refusal on a controlled-direct-effect fit | `_risk_ratio_refusal` says that "no controlled direct risk-ratio target is registered". A binary fit with `intermediate=` reports `rr` and `or` at each level. The [scope page](technical-reference/scope-and-refusals.md) row on a controlled direct risk ratio repeats the claim | `src/cleverly/sensitivity/_derived.py`. The RM21 probe fitted `make_cde(n=400, seed=3)` with Y split at its median, and read `rr` at both levels. `_derived_risk_ratio` calls `retarget` with no `intermediate_value`. Mutation M5 of RM21 removes the rule, and the direct call then raises `DataError`: "the data carries an intermediate variable but no intermediate_value was supplied" | name the missing part, which is a retarget at the level of the fit. The E-value no longer reaches this rule, because RM21 refuses first |
| E-value reason for a level | a request for `ey1` on a fit without an intermediate variable reads "an E-value needs an unconditioned two-arm contrast, not axis 'arm'". The level has the arm axis. It lacks a reference arm | `_select_evalue` in `src/cleverly/sensitivity/evalue.py` prints `key.axis` for each of three failed conditions. The RM21 probe read this text on the Gaussian fit | name the condition that failed |

Each correction needs a unit test that fails without it. Three tests are nonzero witnesses. A
`RegimeContrast` summary keeps its reference line. A design with a time-varying covariate prints it
at its node. A truncated fit labels a `score load` that differs from its `ratio effective n`.

The review of pull request 223 found the split-spread and simultaneous rows. The split-spread test
fails on a selector fit with `repeats=2`, and it passes on an ordinary fit. The docstring test
asserts that each test class that a docstring names exists. The RM26 review settled the
simultaneous policy without changing the default.

The delivery of RM20 found the selection-fold row. Its test must fit the probe above and assert
that neither message offers `cross_fit=False` to an in-sample fit. A control must keep that remedy
on a cross-fitted fit.

The review of that delivery found the bootstrap and continuous-treatment rows. The bootstrap row
needs a decision before a test. The test of the `DataError` row must fit the suggested shift, cross-fitted with `delta=`,
and assert that the message names the in-sample fit.

The delivery of [RM26](#rm26-longitudinal-clustered-intervals-at-few-clusters) widened the
simultaneous row. `LTMLE` skips its default band below 40 clusters, as `TMLE` does under each
non-inferential status. An explicit `simultaneous=True` also builds no band and raises no warning.
The result summary names the omission for both estimators.

The plan and the delivery of [RM21](#rm21-e-value-on-a-controlled-direct-effect-fit) found the
derived risk-ratio row and the E-value-reason row. Each test must read the message on the RM21
probe fit. A control must keep each other reason of the same function.

The PR 229 review resolved these two message rows. `_risk_ratio_refusal` now names the missing
intermediate intervention level in cached retargeting. A binary controlled-direct-effect fit
reports both `rr` and `or`, and the direct helper still refuses. `_select_evalue` now gives
separate reasons for a non-arm axis, a level without a reference arm, and a baseline stratum.
`test_evalue_direct_effect_refusals.py` checks the fitted ratio and each refusal reason.

### RM18. Red property cells after the fold, scale and law changes

Sixteen registered studies moved to unstratified folds. Six of them also moved to a bounded
outcome law with a declared support, and one moved to a binary clustered law. The six are the
manifests that record `tests/studies/bounded_cv_laws.py`. The multi-arm selector row, the
multi-arm outcome-adaptive row and both DR-TMLE rows stay on a binary outcome and declare no
`q_bounds`. Six property cells go
red for the first time after that move, across five of those studies. Every cell that was already
red stays red, and two cells turn green.

This row asks for the missing results. It does not ask for greener numbers. Every gate named here
is interval-shaped (`tests/studies/evidence/registry.py`). A larger budget narrows Monte Carlo
uncertainty around the truth and can change an unresolved verdict. Three post-run remedies are
therefore refused for every entry below. They are raising a budget, moving a margin, and
re-declaring a positive cell's law, learner or size after seeing its verdict.

#### The red-cell ledger

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) holds the current state.
It lists every red verdict that a registered study publishes, with the endpoints that verdict
failed on. It names the `id` of the ask in "What this row asks for" that owns each row.
`python -m tests.studies.evidence.red_cells` writes the ledger from the committed results.
`tests/unit/test_red_cell_ledger.py` fails on four states: a red row with no owner, an owner with
no red row, an owner that the `id` column does not list, and a red row in a `gated` study.

The ledger delivers the reporting end state. Every red verdict stays red under `reporting` at its
registered budget and margin. RM18 itself stays open. Five of its follow-ups ask for a design
declared before its run, and none of those designs is declared or has run. "What this row asks
for" marks each one `open`, and "Deferred findings and review resolutions" says why each one waits.

The rows that F18 and F19 own stay red until F18 or F19 meets its acceptance.
[RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle) carries the one measured increment
that the one-sided robustness reading opened.

The tables below are history. Each one records the values at the commits it names. Read the
ledger for a cell's current verdict.

#### The cells that went red in the regeneration merged at `2e95bcd`

| study | cell | statistic | before | after | margin | reported attribution |
| --- | --- | --- | --- | --- | --- | --- |
| [selector-based point-treatment C-TMLE](technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md) | `selector_necessity/collaborative`, positive | standardized bias | 0.1412 | 0.2173 | the 99% bias upper endpoint is 0.0037 against an equivalence margin of 0.0030, which is 0.3086 against 0.25 on the standardized scale | the external 2x2 report attributes most of the move to the bounded law and a little to the fold policy |
| the same study | `type_i_error/sharp_null`, positive | rejection rate | 0.0275 | 0.0700 | the 99% upper endpoint is 0.1095 against the 0.10 ceiling. Coverage is 0.9300, whose 99% lower endpoint is 0.8905 against the 0.90 floor | the external 2x2 report attributes the move to the bounded law alone |
| [selector-based multi-arm C-TMLE](technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md) | `root_n_and_efficiency/n_500`, positive | exact 99% coverage lower endpoint | 0.9057 | 0.8965 | the endpoint against the 0.90 floor. Coverage itself moved 0.9425 to 0.9350 | not isolated. One covered replication of 400 separates the two endpoints |
| [DR-TMLE for binary complete data](technical-reference/method-evidence/canonical-dr-tmle.md) | `double_robust_contraction/rate_outcome_correct`, positive | 99% contraction-slope interval | -3.52 to -0.06 | -3.04 to +0.12 | the interval must stay below zero | not isolated. The arm's bias is 0.0036 at the first rung. A later declared rung design resolved this cell, and "What the two readings found" gives its interval |
| [multi-arm point-treatment DR-TMLE](technical-reference/method-evidence/multi-arm-dr-tmle.md) | `root_n_and_efficiency/n_500`, positive | exact 99% coverage lower endpoint | 0.9087 | 0.8965 | the endpoint against the 0.90 floor. Coverage itself moved 0.9450 to 0.9350 | not isolated. The same cell and the same new endpoint as the multi-arm selector row above |
| [cross-fitted end-of-study longitudinal TMLE](technical-reference/method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md) | `crossfit_overfitting/cross_fitted_ltmle`, positive | reported SE over empirical SD | 1.172522, 99% upper 1.196518 | 1.176650, 99% upper 1.201555 | the shared `se_ratio_sanity` ceiling of 1.2000, exceeded by 0.001555 | the external 2x2 report attributes the move to the fold policy alone. The pooled update resolves the cell at 1.016107, with a 99% interval from 0.996092 to 1.036997. "What the pooled update found" gives the numbers |

Two of those rows carry a second cell with them. `selector_necessity/empty_control` passes its own
control rule, and it is red through the family's joint clause.
`crossfit_overfitting/in_sample_control` passed its own rule at a standard-error ratio of 0.353193
against a 0.75 ceiling, and the pair's coverage gain passed at a 99% lower endpoint of 0.453375
against a 0.15 floor. Both families were red because their positive arm was. The selector family
stayed red. The overfitting family turned green under the pooled update.

#### The cells that were red before it

| study | cell | state after the regeneration |
| --- | --- | --- |
| selector-based multi-arm C-TMLE | the greedy and ordered `selector_necessity` cells | still red on their bias endpoints. The greedy 99% bias interval runs 0.0279 to 0.0443 against a margin of 0.0158, and the ordered one runs 0.0154 to 0.0304 against 0.0145. Their RMSE ratios moved 0.4470 to 0.4415 and 0.3736 to 0.3781, and both stay inside the family's 0.80 ceiling |
| the same study | `selector_necessity/discrete` | still red. The discrete selector still stops at the empty candidate, so its RMSE ratio is 1 |
| the same study | `interval_calibration/correctly_specified` | still red |
| the same study | `type_i_error/sharp_null` | 0.0700 to 0.0625, whose 99% upper endpoint is 0.1004 against the 0.10 ceiling. Still red, and closer |
| [outcome-adaptive multi-arm C-TMLE](technical-reference/method-evidence/outcome-adaptive-multi-arm-c-tmle.md) | the `generated_design` pair | still red. The oracle SE-ratio interval runs 0.9797 to 1.1151, the estimated-design interval runs 0.9694 to 1.1026, and both leave the calibration band. The paired difference, retained as a reporting diagnostic, runs -0.0240 to 0.0016 |
| [cross-fitted weighted end-of-study longitudinal TMLE](technical-reference/method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | `double_robustness/static__both_wrong`, a control | still red. Its standardized bias moved -0.2983 to -0.2988, and to -0.3008 under the pooled update |

Nine more cells were red at commit `0b75095`, before the regeneration. The table above omitted
them. The values come from each study's `properties.csv` at that commit. The
[red-cell ledger](technical-reference/method-evidence/red-cells.md) gives each cell's current
endpoints.

| study | cell | rule it fails | at `0b75095` |
| --- | --- | --- | --- |
| [DR-TMLE for binary complete data](technical-reference/method-evidence/canonical-dr-tmle.md) | `double_robust_contraction/treatment_correct_n1500` | the 0.90 coverage floor | coverage 0.9200, lower endpoint 0.892086 |
| the same study | `double_robustness/outcome_correct` | bias equivalence | 0.002521 to 0.007468 against 0.006774 |
| the same study | `double_robustness/treatment_correct` | bias equivalence | 0.005012 to 0.010355 against 0.007315 |
| [multi-arm point-treatment DR-TMLE](technical-reference/method-evidence/multi-arm-dr-tmle.md) | `double_robust_contraction/outcome_correct_n4000` | the 0.90 coverage floor | coverage 0.9300, lower endpoint 0.898771 |
| the same study | `double_robust_contraction/rate_outcome_correct` | a slope interval below zero | -1.180439 to 3.678456 |
| the same study | `double_robust_contraction/rate_treatment_correct` | a slope interval below zero | -3.380274 to 3.278706 |
| the same study | `double_robustness/treatment_correct` | bias equivalence | 0.001071 to 0.006821 against 0.006813 |
| the same study | `interval_calibration/correctly_specified` | the SE-ratio band of 0.93 to 1.07 | 0.924604 to 1.010659 |
| selector-based multi-arm C-TMLE | `selector_necessity/empty_control`, a control | the family's joint clause | the cell passes its own rule, and the discrete, greedy and ordered paths fail theirs |

#### What each attribution rests on

A 2x2 diagnostic is a controlled run over a study's own registered seeds. It holds the law, the
learners, the sample size, the budget and the margins fixed, and it varies one axis at a time.

One of those axes is a policy the package no longer offers, so the diagnostic reaches it through a
study-only seam rather than through a setting. The point-treatment arms come from
`tests.studies.bounded_cv_laws.FoldPolicyTMLE`, and the longitudinal arm from
`tests.studies.ltmle_crossfit_properties.FirstNodeStratifiedLTMLE`. Both are committed, and
`tests/unit/test_fold_policy_rules.py` witnesses that each draws the split it names and that the
shipped estimator draws a different one. The longitudinal seam was added after the run it
reproduces, which used the same override applied by hand.

| finding | diagnostic | what it establishes |
| --- | --- | --- |
| `selector_necessity/collaborative` | an externally reported law by fold-policy 2x2 at commit `4ca7a15`, over the study's 800 registered seeds | the report attributes most of the move to the bounded law. Bounded folds with treatment strata give 0.2211, and bounded folds without give 0.2173, so the fold policy adds about 0.004. The anchor Gaussian design also misses under unstratified folds, at a standardized 0.2562 against 0.25 |
| `type_i_error/sharp_null`, selector | the same externally reported 2x2 | the report attributes the move to the bounded law alone. The cell measures 0.0700 under both fold policies, against 0.0375 on the Gaussian law without strata |
| `crossfit_overfitting/cross_fitted_ltmle` | an externally reported fold-policy 2x2 at commit `eeaa1ce`, over the study's 8,000 registered paired draws | the report attributes the move to the fold policy alone. The first-node-stratified arm gives 1.172543 with a 99% upper endpoint of 1.196538, which reproduces the committed row to 2e-5 relative. The single-fold control arm has no outer split for either policy to change, and it gives 0.353193 under both. These studies never moved to a bounded twin, so there is no second axis |
| both `n_500` coverage endpoints | a registered `fold_policy` pair, two split policies over 8,000 paired draws of each study | a boundary resolution at 400 replications. Each study now covers 374 replications of 400, where the two covered 377 and 378 before. The paired intervals include zero: -0.006875 to +0.001625 for the selector and -0.005625 to +0.004500 for DR-TMLE. The instrument therefore does not attribute the boundary crossing to the fold policy; it records that the registered verdict changed while the paired effect remains unresolved. The same cell moved the other way on outcome-adaptive multi-arm C-TMLE, where it turned green at 0.9057 |
| the two newly red weighted cells and the green end-of-study overfitting cell | a declared code-by-runtime diagnostic over each study's registered seeds, four arms per study | the pooled code changes all three verdicts at both runtimes, and the runtime changes none. "[What the runtime isolation found](#what-the-runtime-isolation-found)" gives the rows |
| `double_robust_contraction/rate_outcome_correct` | a declared rung design, run at 2,400 replications on each outer rung | three rungs of a small quantity gave a wide slope, and the width was Monte Carlo error at the top rung. The raised rungs put the interval below zero. That run also moved the interpreter, SciPy and the `src/` tree. The committed history separates the budget from those changes. At 800 replications per rung, the source and runtime changes together move each slope interval by less than 2e-5. The declared budget therefore moved this interval. "[What the committed history already separates](#what-the-committed-history-already-separates)" gives the rows |

The fold policy has no general direction on the overfitting statistic. `crossfit_overfitting`
shares one family, one statistic and one ceiling across the four cross-fitted longitudinal studies.
The same fold-policy change moved its 99% upper endpoint in both directions: by +0.005037 on
end-of-study, -0.005036 on categorical, +0.003560 on survival, and -0.001343 on competing risks.
Each of those four is the difference of the two published six-decimal endpoints, not of the raw
ones. The end-of-study move is +0.005036 before rounding.

Nothing here establishes that unstratified folds inflate this statistic. The externally reported
end-of-study 2x2 attributes the whole end-of-study move to the fold policy on that law. This branch
commits no run log for that diagnostic, so the repository does not independently verify the
attribution.

The end-of-study breach is in the conservative direction. Coverage moved 0.975375 to 0.976000 as
the standard-error ratio rose, so the interval is wide rather than invalid. The cell's own page
carried the forecast before the run, at commit `0b75095`: "The gate is close. The 99% interval
reaches 1.1965 against a ceiling of 1.2000, so a change of learner, sample size, or fold count
could move this cell across it."

#### One result flipped direction, and the study reports it rather than gating it

The selector study's `selector_necessity/collaborative` standard-error ratio was 1.2539 under
treatment-stratified selection folds. It is 0.8011 under unstratified ones. The committed 1.2539
was the artefact of the stratified split. The reported interval at this cell is therefore
anti-conservative rather than conservative, and the cell's evidence-page limitation row now says
so. The new value points the same way as
[RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism)'s own probe of the
instrument law, which measured a ratio of 0.844 and a coverage of 0.92. The study reports this
statistic and does not gate on it (`tests/studies/ctmle_selector_properties.py`).

#### The fold-local longitudinal targeting question

This question had no item of its own. The retired RM17 row held it, and the retired F24 row then
carried its contract. The pooled update below settles it.

Díaz, Williams, Hoffman and Schenck (2023), Section 5.2 and Theorem 3, define a random
near-balanced row partition for a longitudinal TMLE. The theorem's targeting construction is one
pooled all-row fluctuation per node after untargeted fold regressions. Before the change, the
package fitted fold `k`'s epsilon on fold `k`'s training rows, as the authors' own `lmtp` 1.5.4
does. Upstream commit
[`9996b04`](https://github.com/nt-williams/lmtp/commit/9996b04dcbb3ae0b1ef8862097c36d95e9f2fcf9)
calls that behavior a bug because the EIF was not mean zero. It now fits the fluctuation on
validation rows, which the forthcoming 1.5.5 release records. Theorem 3 certifies neither
fold-local update.

The end-of-study overfitting cell was where that gap showed as a number. One reading was that a
fold-local update leaves a stitched score. That score is a mean-zero residual rather than a solved
equation, so the reported standard error would run above the sampling spread. No result read here
established that reading.

Four routes could close the question. They are a theorem for the fold-local update, a
validation-fold update with its own result, the pooled update, and the SDR estimator of
[X4](#x4-sequential-doubly-robust-longitudinal-estimation). The pooled update is the one route
with a published result for the shipped plug-in estimator, so this row took it. The
[cross-fitting section](technical-reference/longitudinal-tmle.md#cross-fitting-the-recursion) of
the technical reference holds its contract, its locators, and the natural extensions it carries.

#### The pooled update, declared before it runs

This subsection declares the pooled update and the rule that publishes its studies. It precedes
every regeneration of those studies.

The cross-fitted per-regimen fit now follows Section 5.2, Steps 1 to 4, of Díaz, Williams, Hoffman
and Schenck (2023). Those steps are on journal pages 852 and 853. The functions
`_fit_regimen_crossfit` and `_pooled_targeting` in `src/cleverly/longitudinal/sequential.py`
implement them.

| part | what the fit does |
| --- | --- |
| fold regressions | each outer fold runs an untargeted backward regression sequence on its training rows |
| stitching | the held-out predictions of the folds form one out-of-fold initial estimate per node |
| pooled fluctuation | one fluctuation per node, from the horizon back to node 1, targets the stitched estimate over every follower |
| loss weight | the observation weight times the inverse of the out-of-fold cumulative mechanism |
| one fold | `n_folds=1` keeps the canonical single-fold path, which carries each targeted prediction back |

Theorem 3, on journal page 853, states the result for this construction. Its proof in the
supplement conditions on each fold's initial nuisance fit being fixed given its training rows;
the pooled coefficients depend on the full sample and are handled as a low-dimensional class.
That condition covers untargeted fold regressions followed by one pooled fluctuation per node. It
does not cover a pooled epsilon carried back into later fold regressions.
`tests/unit/test_pooled_longitudinal_targeting.py`
checks each part of the table against a longhand recomputation.

The change moves five registered studies, so each one regenerates.

| study | artifacts | publication policy before the run |
| --- | --- | --- |
| `canonical-ltmle-crossfit` | `tests/canonical/lmtp_ltmle` | `reporting` |
| `weighted-ltmle-crossfit` | `tests/canonical/weighted_lmtp_ltmle` | `reporting` |
| `canonical-categorical-ltmle-crossfit` | `tests/canonical/categorical_ltmle_crossfit` | `gated` |
| `canonical-ltmle-survival-crossfit` | `tests/canonical/lmtp_ltmle_survival` | `gated` |
| `canonical-ltmle-competing-crossfit` | `tests/canonical/lmtp_ltmle_competing_crossfit` | `gated` |

Each study compares with `lmtp` 1.5.4, and its manifest records that version. That comparator
runs a training-fold fluctuation and carries its targeted prediction into the fold's next
regression. After the change, the paired comparison therefore reads two different constructions.
Each paired margin stays as registered.

The run follows one declared rule.

| item | rule |
| --- | --- |
| margins, budgets, laws, learners, sizes, cells and seeds | unchanged in every study |
| publication as `gated` | only when every independent, paired and property verdict of the study passes |
| publication as `reporting` | when any verdict fails. RM18 names each failed cell |
| `crossfit_overfitting/cross_fitted_ltmle` | read at its registered 8,000 draws against the unchanged 1.20 ceiling |
| the published result | what the run produces, whichever way each verdict falls |

A timing probe runs before the regeneration. It sizes the run and reads no verdict.

#### What the pooled update found

The five studies regenerated once, under the rule the subsection above declared. Each publishes
what it produced. The values below come from each study's committed `properties.csv` and
`equivalence.csv`, against the rows they replaced.

| study | policy | property cells | paired comparisons |
| --- | --- | --- | --- |
| `canonical-ltmle-crossfit` | `reporting`, as declared | 32 of 32, against 30 before | 5 of 5 |
| `weighted-ltmle-crossfit` | `reporting`, as declared | 34 of 36, against 35 before | 2 of 5, against 3 before |
| `canonical-categorical-ltmle-crossfit` | `gated` | 36 of 36 | 9 of 9 |
| `canonical-ltmle-survival-crossfit` | `gated` | 52 of 52 | 8 of 8 |
| `canonical-ltmle-competing-crossfit` | `gated` | 38 of 38 | 16 of 16 |

The three gated studies pass every verdict, so each keeps `gated`. The end-of-study study now passes
every verdict too. The declared rule makes `reporting` mandatory only when a verdict fails, and it
does not move a study to `gated`. That study therefore keeps the `reporting` policy the registry
declares, and a move to `gated` is a separate registry decision.

The overfitting cell moved in each of the four studies that carry it. It is read at the registered
draws against the unchanged 1.20 ceiling.

| study | cross-fitted SE ratio before | after, with its 99% interval | coverage before and after |
| --- | --- | --- | --- |
| end-of-study | 1.176650, upper 1.201555 | 1.016107, from 0.996092 to 1.036997 | 0.976000 to 0.951000 |
| survival curve | 1.102157 | 0.983738, from 0.963806 to 1.003666 | 0.967250 to 0.943000 |
| competing risks | 1.092332 | 0.987990, from 0.968219 to 1.009389 | 0.964625 to 0.946000 |
| categorical, at 40,000 draws | 1.179559 | 0.982141, from 0.973370 to 0.991073 | 0.976825 to 0.942225 |

All four reported standard-error ratios are lower and near one in the pooled-code regeneration.
That is the direction the stitched-score reading in the fold-local targeting question above
predicted. The runtime changed at the same time. The runtime isolation below attributes the
end-of-study verdict change to the code. It does not test the stitched-score mechanism itself. The
survival-curve, competing-risk and categorical drops stay unisolated, because that diagnostic
deferred those three studies.

Two cells of the weighted study are newly red. They appeared in the pooled-code, new-runtime
regeneration. The runtime isolation below attributes both to the code and not to the runtime.

| cell | statistic | before | after | margin |
| --- | --- | --- | --- | --- |
| `interval_calibration/static__correctly_specified`, positive | empirical efficiency ratio, 99% interval | 1.053468, from 1.013022 to 1.092339 | 1.060724, from 1.019495 to 1.100601 | the band upper edge of 1.10, exceeded by 0.000601 |
| paired `ey_regimen[never]` | coverage difference, 99% lower endpoint | -0.00875, lower -0.01875, equivalent | -0.0175, lower -0.030, underpowered | the non-inferiority margin of -0.025 |

The paired mean difference on `ey_regimen[never]` stays small, at -0.000043 against a margin of
0.004659. The failure is in coverage. `cleverly` covers 0.93125 of draws, and the unchanged `lmtp`
rows cover 0.94875 with a native standard-error ratio of 1.0316. The paired `ey_regimen[always]`
and `ey_regimen[treat then continue if l2 positive]` comparisons concluded underpowered before the
change too, and they still do. The study publishes both cells
under the `reporting` policy it declared before the run. Its
[evidence page](technical-reference/method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md)
gives the complete tables.

The paired comparator now reads a different construction. On the competing-risk study the two
implementations had agreed to within 2e-9 on every paired mean. The largest paired mean difference
is now -0.000195, for `cif_regimen[continue_if_l2, relapse @ t=2]`, against a margin of 0.004029.
No paired verdict changed on the end-of-study, survival-curve, competing-risk or categorical study.

The single-fold control rows moved on three studies. The in-sample control has no outer split, and
the change does not touch its code path.

| study | control rows that differ from the committed rows | largest change in the estimate | largest change in the standard error |
| --- | --- | --- | --- |
| end-of-study | 2 of 8,000 | 0.002 | 0.0001 |
| survival curve | 4 of 8,000 | 0.003 | 0.0001 |
| competing risks | 5 of 8,000 | 0.003 | 0.0002 |
| categorical | 0 of 40,000 | none | none |

No covered flag changed. A refit of two differing end-of-study rows under the source of the
previous commit reproduces the new values, not the committed ones. The committed rows therefore
came from a different environment, and the manifests record one. The rows they replaced were
generated under Python 3.11.13 and SciPy 1.17.1. The new rows ran under Python 3.13.7 and SciPy
1.18.0. The end-of-study control ratio moved in the sixth decimal, from 0.353193 to 0.353196.

#### The runtime isolation, declared before it runs

This subsection declares a diagnostic and the rule that reads it. It precedes every run of that
diagnostic.

The pooled-code regeneration changed the code and the runtime together. Two weighted cells went
red, and the end-of-study overfitting cell went green. The diagnostic separates the two changes on
the two studies that carry those cells. It crosses two code states with two runtimes.

| axis | label | what it pins |
| --- | --- | --- |
| code | F | commit `7d5485a`, which fits the fold-local update |
| code | P | the commit that carries this declaration. Its `src/` is identical to `0e03a15`, which fits the pooled update |
| runtime | R11 | Python 3.11.13 and SciPy 1.17.1 |
| runtime | R13 | Python 3.13.7 and SciPy 1.18.0 |

Both runtimes install numpy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.0. Each study therefore runs
four arms: `F-R11`, `F-R13`, `P-R11` and `P-R13`.

| study | registry name | how each arm gets the `lmtp` rows |
| --- | --- | --- |
| `tests/canonical/weighted_lmtp_ltmle` | `weighted-ltmle-crossfit` | the first arm runs `lmtp`, and every later arm reuses that result through `--cache`. This study writes `reference-inference.csv.gz`, so the harness refuses `--skip-reference` for it |
| `tests/canonical/lmtp_ltmle` | `canonical-ltmle-crossfit` | every arm reads the committed rows through `--skip-reference` |

The committed `lmtp` rows of both studies are identical at `7d5485a` and `0e03a15`. So is the
committed `reference-inference.csv.gz`.

| held fixed | value |
| --- | --- |
| laws, learners, sizes, budgets, seeds, cells and margins | as each study registers them |
| the `lmtp` comparator rows | identical in the four arms of a study |
| numpy, pandas and scikit-learn | identical in both runtimes |
| the machine | one runner. The arms run one at a time, and no other study or test suite runs with them |

Two preconditions validate the harness. Each one also tests a claim that the committed history
makes.

| arm | must reproduce | which also tests |
| --- | --- | --- |
| `F-R11` | the rows committed at `7d5485a`. They came from `f0110bc` for the weighted study and from `eeaa1ce` for the end-of-study study | that the source changes between those commits and `7d5485a` do not move a row of either study |
| `P-R13` | the rows committed at `0e03a15`. They came from `656674c` | that the refactor in `f8ad497` does not move a row of either study |

A precondition holds when every estimate and every standard error agrees within 1e-9, and no
verdict differs. The four arms of a study must also draw identical samples. The diagnostic
compares the decompressed `samples.csv.gz` of each arm, because the gzip header records a time.

When a precondition fails, the diagnostic records the count of rows outside the tolerance and the
largest difference. It then marks the study "harness not validated, no attribution" and reads no
attribution from it.

| study | cell | statistic read |
| --- | --- | --- |
| weighted | `interval_calibration/static__correctly_specified`, positive | the empirical efficiency ratio and its 99% interval, against the band upper edge of 1.10 |
| weighted | paired `ey_regimen[never]` | the coverage difference and its 99% lower endpoint, against the non-inferiority margin of -0.025 |
| end-of-study | `crossfit_overfitting/cross_fitted_ltmle`, positive | the reported SE over the empirical SD and its 99% upper endpoint, against the 1.20 ceiling |
| end-of-study | `crossfit_overfitting/in_sample_control` | the count of rows that differ from the committed rows, and the largest change in the estimate and in the standard error |

The reading rule uses each verdict alone. It makes four comparisons per cell. The code axis
compares F with P at R11 and at R13. The runtime axis compares R11 with R13 at F and at P.

| reading | condition |
| --- | --- |
| code | the code axis changes the verdict at both runtimes, and the runtime axis changes it at neither code state |
| runtime | the runtime axis changes the verdict at both code states, and the code axis changes it at neither runtime |
| both | each axis changes the verdict at least once |
| neither | no comparison changes the verdict |

A verdict is pass or fail, so a change occurs in zero, two or four of the four comparisons. The
four readings therefore cover every outcome. The magnitudes and the count of changed rows on each
axis are descriptive. They do not change a reading.

The diagnostic changes no verdict, publication policy, budget, margin or law. Each arm writes to
a scratch directory through `--output`, and no committed study artifact changes. The repository
commits three kinds of output under `tests/diagnostics/rm18_runtime/`: the summary, the run log
and one manifest per arm.

This diagnostic leaves three studies and two earlier 2x2 reports outside its scope.

| item | status | reason |
| --- | --- | --- |
| the survival-curve, competing-risk and categorical studies | deferred | every verdict passes at `7d5485a` and at `0e03a15`, so the rule has no changed verdict to attribute. No committed log times their arms. Each end-of-study arm took 291 to 386 seconds in `run.log` |
| the law by fold-policy 2x2 at `4ca7a15`, for the selector study | stays unverified | the repository commits no run log for it, and this diagnostic does not rerun it |
| the fold-policy 2x2 at `eeaa1ce`, for the end-of-study study | stays unverified | the same |

(what-the-committed-history-already-separates)=
#### What the committed history already separates

This subsection reads committed artifacts alone, with `git show` and pandas. It fits no
estimator. It was computed on 2026-09-21 during the scoping of the diagnostic above, and before
that declaration was written.

Three point-treatment studies changed runtime between two commits. They moved from Python 3.11.13
and SciPy 1.17.1 to Python 3.13.7 and SciPy 1.18.0. Each study's `manifest.json` records both runtimes, and
the numpy and pandas versions are identical. The table compares each replication row of the older
commit with the same row at `0e03a15`.

The `src/` tree also changed between each older commit and `0e03a15`. `git diff --stat` reports
22 changed files from `99d238c` and from `6933968`, and 23 from `2049349`. Each list includes
`src/cleverly/learners/crossfit.py`. Each comparison below therefore bounds the source and runtime
changes together. It does not separate the source from the runtime.

| study | older commit | rows compared | largest change in an estimate | rows that moved more than 1e-6 | covered flags that changed |
| --- | --- | --- | --- | --- | --- |
| multi-arm point-treatment DR-TMLE | `6933968` | 14,400 primary and 10,600 property | 2.1e-12 | 0 | 0 |
| selector-based multi-arm C-TMLE | `2049349` | 21,600 primary and 5,200 property | 4e-15 | 0 | 0 |
| DR-TMLE for binary complete data, `cleverly` rows | `99d238c` | 7,200 primary | 4.3e-9 | 0 | 0 |
| the same, property rows | `99d238c` | 15,200 | 0.0047 | 6 | 0 |
| the same, `drtmle` R comparator rows | `99d238c` | 7,200 primary | 0.00062 | 6 | 0 |

The rows compared are the rows present at both commits. `0e03a15` adds 16,000 `fold_policy` rows to
each multi-arm study and 9,600 contraction rows to DR-TMLE.

The source and runtime changes together moved no row of either `n_500` study by more than 1e-6,
and they changed no covered flag. Each committed `n_500` endpoint therefore reads the same rows
under both commits.

The contraction ladder shares 7,200 rows between `99d238c` and `0e03a15`. They are the first 800
replications of each rung, and they agree within 7.3e-5. The study's slope rule, applied to those
800 replications of each rung at `0e03a15`, gives the intervals below.

| cell | `99d238c`, 800 per rung | `0e03a15`, first 800 per rung | `0e03a15`, as committed |
| --- | --- | --- | --- |
| `rate_outcome_correct` | -3.040132 to +0.123221 | -3.040131 to +0.123221 | -1.534453 to -0.482268 |
| `rate_treatment_correct` | -2.207877 to -0.880196 | -2.207860 to -0.880192 | -1.756754 to -1.022105 |
| `rate_both_wrong`, the control | -0.003031 to +0.014273 | -0.003031 to +0.014273 | +0.003726 to +0.013538 |

At the declared budget of 800, the source and runtime changes together leave each interval within
2e-5. The move of `rate_outcome_correct` below zero therefore comes from the extra outer-rung
replications that the rung design declared. It does not come from the source or the runtime.

(what-the-runtime-isolation-found)=
#### What the runtime isolation found

The diagnostic ran on 2026-09-21 under the declaration above. Its eight full arms ran one at a
time on one runner. Every value below comes from `isolation.csv` in
[`tests/diagnostics/rm18_runtime/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_runtime).
The `run.log` in that directory records each arm's command, runtime, `pip freeze` digest, wall
time and exit code.

Each runtime label also stands for the rest of its virtual environment. `run.log` records one
`pip freeze` digest per runtime, over 132 lines at R11 and 131 at R13. The two records differ in
the interpreter, SciPy, the editable install path and `backports-tarfile`, which only R11 installs.

Both preconditions held for both studies. The harness is therefore validated, and the rule reads
an attribution from each study.

| study | arm | reproduces the rows at | rows compared | rows that differ | verdicts that differ |
| --- | --- | --- | --- | --- | --- |
| weighted | `F-R11` | `7d5485a` | 8,000 primary, 42,400 property and 12,000 reference-inference | 0 | 0 |
| weighted | `P-R13` | `0e03a15` | the same | 0 | 0 |
| end-of-study | `F-R11` | `7d5485a` | 16,000 primary and 51,200 property | 0 | 0 |
| end-of-study | `P-R13` | `0e03a15` | the same | 0 | 0 |

The four arms of each study drew identical samples. Each study's decompressed `samples.csv.gz`
has one SHA-256 digest in all four arms. The preconditions also confirm two claims of the committed
history. The source changes from `f0110bc` and `eeaa1ce` to `7d5485a` move no row, and the
refactor in `f8ad497` moves no row.

The code axis changes the verdict of three cells at both runtimes. The runtime axis changes no
verdict.

| study | cell | `F-R11` | `F-R13` | `P-R11` | `P-R13` | reading |
| --- | --- | --- | --- | --- | --- | --- |
| weighted | `interval_calibration/static__correctly_specified`, positive | pass. Efficiency ratio 1.053468, from 1.013022 to 1.092339 | the same | fail. 1.060724, from 1.019495 to 1.100601 against 1.10 | the same | code |
| weighted | paired `ey_regimen[never]` | pass. Equivalent, 99% lower endpoint -0.01875 | the same | fail. Underpowered, lower endpoint -0.030 against -0.025 | the same | code |
| end-of-study | `crossfit_overfitting/cross_fitted_ltmle`, positive | fail. SE ratio 1.176650, 99% upper 1.201555 | fail. 1.176637, upper 1.201516 | pass. 1.016107, from 0.996092 to 1.036997 | the same | code |
| end-of-study | `crossfit_overfitting/in_sample_control` | pass. SE ratio 0.353193 | pass. 0.353196 | pass. 0.353193 | pass. 0.353196 | neither |

The two weighted cells therefore went red with the pooled code, not with the runtime. At each code
state, the two runtimes give the same weighted statistics to every printed decimal. This
diagnostic attributes the change and does not by itself judge the estimator. The fixed-known-weight
analysis under `RM18-fixed-weights` in "[What each owner holds](#what-each-owner-holds)" supports
the pooled estimating equation for this iid baseline-selection law. Both cells stay red as
finite-sample reporting evidence. Broader weighted laws remain open, and no margin, budget or law
moved.

The end-of-study overfitting cell turned green with the pooled code. At `F`, the runtime moves its
upper endpoint from 1.201555 to 1.201516, and both values exceed the 1.20 ceiling.

The in-sample control rows move with the runtime and not with the code. Each R11 arm reproduces
the control rows at `7d5485a`, and each R13 arm reproduces the control rows at `0e03a15`. The two
runtimes differ on 2 of 8,000 control rows at each code state. The largest change is 0.0020 in an
estimate and 0.0001 in a standard error, and no covered flag changes. This confirms the
environment reading of the control table in "What the pooled update found".

The declared rule treats the size of each change as descriptive. The table below gives it for
each axis. The code axis gives the same counts at R11 and at R13.

| study | comparison | artifact | rows compared | rows that differ | rows moved more than 1e-6 | covered flags changed | largest change in an estimate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weighted | code, F against P | primary | 8,000 | 4,000 | 3,999 | 28 | 0.012 |
| weighted | code, F against P | property | 42,400 | 42,400 | 37,598 | 682 | 0.43 |
| weighted | code, F against P | reference-inference | 12,000 | 0 | 0 | 0 | none |
| weighted | runtime at F | primary | 8,000 | 939 | 0 | 0 | 3.3e-16 |
| weighted | runtime at P | primary | 8,000 | 1,017 | 0 | 0 | 3.9e-16 |
| weighted | runtime at F and at P | property and reference-inference | 54,400 | 0 | 0 | 0 | none |
| end-of-study | code, F against P | primary | 16,000 | 8,000 | 7,998 | 54 | 0.0084 |
| end-of-study | code, F against P | property | 51,200 | 43,200 | 40,785 | 835 | 0.15 |
| end-of-study | runtime at F | primary | 16,000 | 1,800 | 0 | 0 | 3.3e-16 |
| end-of-study | runtime at F | property | 51,200 | 110 | 33 | 0 | 0.0070 |
| end-of-study | runtime at P | primary | 16,000 | 2,057 | 0 | 0 | 3.9e-16 |
| end-of-study | runtime at P | property | 51,200 | 945 | 1 | 0 | 0.0020 |

The table gives no standard-error column. The largest code-axis change in a weighted standard
error is 49,702, in `isolation.csv`. It comes from a standard-error pathology that both code
states share. In each `learner_weight_necessity` cell, 13 of 1,200 replicates report a standard
error above 1 in all four arms. The change of 49,702 is replicate 1099 of the control. Its
standard error falls from 49,757.77 at `7d5485a` to 55.32 at `0e03a15`, with the same estimate of
0.229714.

Those replicates set the published SE ratio of both cells. It is 2,758 and 2,825 at `0e03a15`, and
6,772 and 6,925 at `f0110bc`. The verdicts of both cells read the bias intervals and the paired
displacement, and not the reported standard error. `RM18-learner-weight-se` owns this pathology.
"[What the learner-weight diagnostic found](#what-the-learner-weight-diagnostic-found)" gives its
reading.

One definition in `compare.py` changed after the smoke arms and before the full arms. Each smoke
arm ran 4 replications at n = 2,000 without property cells.

At commit `97a3687`, the cell verdict was the cell's `passed` column and its `property_passed`
column together. The declaration says
the rule "uses each verdict alone". This record reads a cell's verdict as its own `passed` column.
Commit `65e319b` makes that change and records `property_passed` beside it. The committed rows at
`7d5485a` already showed that this choice moves the in-sample control from code to neither.

Commit `65e319b` is dated 22:42:33Z on 2026-09-21, and `run.log` starts the first full arm at
22:42:47Z. Neither `97a3687` nor `65e319b` was pushed before the full arms. Their order therefore
rests on the local commit timestamps and the `run.log` timestamps alone.

The change moves the reading of one cell. Under the earlier definition, the in-sample control
fails at F with its family, and it reads code. The other three cells have equal `passed` and
`property_passed` values in every arm, or no `property_passed` column. Their readings therefore
do not depend on the definition. Only `F-R13` and `P-R11` carry new information, because `F-R11`
and `P-R13` reproduce committed rows.

The survival-curve, competing-risk and categorical studies stay outside this result. The
declaration deferred them, so their overfitting drops in "What the pooled update found" stay
unisolated.

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
| `RM18-pooled` | a targeting result for the fold-local longitudinal recursion | delivered by the pooled route. The package now implements Section 5.2, Steps 1 to 4, which Theorem 3 certifies. `crossfit_overfitting/cross_fitted_ltmle` sits inside the shared ceiling at 8,000 draws, with a 99% interval from 0.996092 to 1.036997. "What the pooled update found" gives every moved cell, including two weighted cells that went red |
| `RM18-n500` | a reading of the two `n_500` coverage endpoints | delivered. A registered fold-policy diagnostic reads both endpoints as a boundary resolution. "What the two readings found" gives the numbers |
| `RM18-binary-slope` | a reading of the DR-TMLE contraction slope | delivered. A declared rung design resolves the slope, and its interval now sits below zero. "What the two readings found" gives the numbers |
| `RM18-attribution` | an attribution of the verdicts that changed with the pooled update | delivered for the end-of-study and weighted studies. A declared code-by-runtime diagnostic reads the two weighted cells and the end-of-study overfitting cell as code changes. "What the runtime isolation found" gives the numbers. The weighted cells stay red as finite-sample reporting evidence |
| `RM18-fixed-weights` | a fixed-known-weight follow-up for the two weighted cross-fitted cells that went red with the pooled update | open: the design this row asks for is not declared, and it has not run. The row closes when a design declared before its run reads the two cells, or when a published result for the pooled update under fixed known weights applies |
| `RM18-one-sided-bias` | a reading of the three one-sided-robustness bias rows | delivered. The reading that "[The one-sided robustness reading, declared before it is computed](#the-one-sided-robustness-reading-declared-before-it-is-computed)" declares ran once. "[What the one-sided reading found](#what-the-one-sided-reading-found)" gives its result. The three cells stay red under `reporting` |
| `RM18-slopes` | a reading of the two multi-arm DR-TMLE contraction slopes | open: the design this row asks for is not declared, and it has not run. This row computed the cost of that design. The row closes when a rung design declared before its run reads both slopes |
| `RM18-ordinary-weighted` | an owner for the six red rows of the ordinary weighted longitudinal study | open: the design this row asks for is not declared, and it has not run. The row closes when a design declared before its run reads the calibration row, the sharp-null row and the `targeting_necessity` family |
| `RM18-learner-weight-se` | a diagnostic for the standard-error explosions in the cross-fitted `learner_weight_necessity` rows | delivered. A complete review declaration was pushed before a clean rerun that reproduced the refit table. "[What the learner-weight diagnostic found](#what-the-learner-weight-diagnostic-found)" gives its reading. This row owns no red cell |
| `RM18-boundary` | an owner for the red cells that sit at a finite-budget boundary | open: no re-read design is declared. Each cell stays red under `reporting` at its declared budget. A re-read needs a design declared before its run, and this row refuses an undeclared budget |
| `RM18-comparator-density` | an attribution of the red paired row of the continuous modified treatment policy study | open: the design this row asks for is not declared, and it has not run. The row closes when a design declared before its run attributes the calibration excess to `cleverly` or to the comparator |

(what-each-owner-holds)=
##### What each owner holds

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) lists every red cell by
its owner. This subsection gives the reason for each assignment.

`F18` holds the red cells of both selector studies, except the multi-arm
`root_n_and_efficiency/n_500` endpoint. On both laws of the point-treatment study, the population
one-step remainder is exactly zero at the nuisance limits. The residual there is the selector's
stopping behaviour and not a nuisance rate. On the multi-arm study, the greedy and ordered paths
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

`RM18-slopes` holds `double_robust_contraction/rate_outcome_correct` and `rate_treatment_correct`
of `canonical-multi-arm-drtmle`. A rung design declared before its run can read them, as
`RM18-binary-slope` did for the binary slope.
"[What the multi-arm rung design would cost](#what-the-multi-arm-rung-design-would-cost)" gives its
cost. The design needs at least 34,267 replications at each outer rung by the delta method,
against the 600 that each rung runs now.

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

The second acceptance row previously required a negative paired standard-error deficit. That gate
was not consistent with its own sources, and it is no longer an acceptance condition.

| part of the conflict | what it states | where |
| --- | --- | --- |
| the only published result | no first-order generated-design term for one binary treatment-specific mean, so no first-order standard-error deficit | Theorem 1 of Benkeser, Cai and van der Laan, arXiv:1901.05056 v1 |
| the historical `estimated` control | passed only when the 99% upper endpoint of the paired deficit was at or below -0.01 | the removed `GENERATED_DESIGN_DEFICIT` and the historical `generated_design` verdict in `tests/studies/multi_arm_ctmle_oat_properties.py` |
| the F19 acceptance | a result that establishes whether the current curve suffices or a representation contribution is required | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |

A derivation that extends Theorem 1 would predict no first-order deficit. The registered gate now
requires each design's coverage and standard-error intervals to fit inside their calibration
bands. The point-treatment pair passes that rule. The multi-arm pair passes its coverage band but
fails because both standard-error intervals cross the 1.07 upper bound. The paired difference is
descriptive rather than an acceptance condition.

A future oracle comparison may use a two-sided equivalence margin or a predeclared sample-size
ladder that contracts the difference towards zero. A deliberately invalid generated design may
serve as a negative control; the learned design itself may not. The multi-arm `oracle_design`
SE-ratio interval runs 0.979701 to 1.115078, while its coverage interval runs 0.928173 to 0.968751.
The width of the SE interval describes the resolution of the instrument at the registered 800
replications; its coverage interval is already inside the 0.92 to 0.98 band.

#### The two readings, declared before they run

This subsection is the declaration that the `RM18-n500` and `RM18-binary-slope` rows ask for. It
precedes the run that measures either reading.

##### Why the `n_500` reading needs more than 400 replications

The two policies agree about coverage on almost every draw. A pilot on a throwaway
`fold_policy_pilot` stream measured that rate over 600 draws of each study. It is 0.0117 on the
selector study and 0.0250 on the DR-TMLE study. The committed before-and-after pair gives 0.0175
and 0.0250 over its own 400 draws. The paired difference is nonzero only on a discordant draw, so
its standard deviation is about the square root of that rate.

The quantity the reading has to resolve is small. The gated endpoint crosses the 0.90 floor
between 375 and 376 covered replications of 400, so the gate-relevant difference is 0.005. The
studies moved 0.0075 and 0.0100.

| paired replications | 99% half-width, selector | 99% half-width, DR-TMLE | resolves 0.005 |
| --- | --- | --- | --- |
| 400 | 0.0139 to 0.0170 | 0.0204 | no |
| 1,600 | 0.0070 to 0.0085 | 0.0102 | no |
| 4,000 | 0.0044 to 0.0054 | 0.0064 | marginal |
| 8,000 | 0.0031 to 0.0038 | 0.0046 | yes |

At 400 replications the interval is three to four times wider than the effect. The reading would
report "indeterminate" whatever the fold policy does, so 400 replications cannot separate the two
explanations. The committed pair says the same thing. An exact McNemar test on its discordant
draws gives 0.453 on the selector study and 0.344 on the DR-TMLE study. Neither observed move is
distinguishable from a coin flip.

The diagnostic therefore runs at 8,000 paired replications. The gated `root_n_and_efficiency/n_500`
cell keeps its law, its learners, its size and its 400-replication budget.

##### Why that budget is not the refused remedy

This row refuses an undeclared larger budget for a gated cell. A larger budget narrows the interval
around the cell's true performance and can resolve an inconclusive verdict. The fold-policy budget
was declared before the run, and its diagnostic states no verdict.

| property | a gated cell | the fold-policy diagnostic |
| --- | --- | --- |
| declares a margin | yes | no |
| states a verdict | yes | no. `passed` and `property_passed` are set for every row |
| counts toward the published pass fraction | yes | no. `tests/studies/evidence/claims.py` drops a diagnostic row |
| what a larger budget buys | a narrower interval around the cell's true performance | a narrower interval on a paired difference |

The diagnostic's budget buys resolution of a difference. It does not alter the gated cell, which
keeps the budget it published under.

##### What each outcome means

The published interval sits on the `treatment_stratified` row. It covers the coverage of that arm
minus the coverage of the `unstratified` arm. A positive value means the
retired stratified policy covered more.

| outcome | condition | reading |
| --- | --- | --- |
| a fold-policy effect | the lower endpoint is above zero | the stratified policy raises coverage at `n = 500` on this law. A lower endpoint at or above 0.005 accounts for the endpoint crossing the floor |
| a boundary resolution | the interval covers zero, and its upper endpoint is below 0.005 | no policy effect large enough to move the endpoint across the floor. RM18's attribution stands |
| a reversal | the upper endpoint is below zero | the unstratified policy covers more, and the move belongs to something else |
| indeterminate | the interval covers zero, and its upper endpoint reaches 0.005 | the measurement resolved nothing |

The diagnostic reports its own coverage at 8,000 draws. That number does not re-read the gated
cell. The gated cell publishes 400 replications under its own budget, and a tighter interval on a
different budget is not evidence about it.

##### The DR-TMLE contraction rung design

The slope is fitted over three log-equally-spaced rungs, so the middle rung's centred weight is
exactly zero. The slope is therefore the difference of the outer two rungs, divided by the log of
four.
Curvature therefore cannot widen the interval, and the committed rungs are log-linear to 0.002.
The span is already the cost-optimal one, because the resolution of a rung falls as its size
rises. The width comes from Monte Carlo noise at the top rung, where the bias is 2.46 times its
own standard error.

The design therefore raises replications at the two outer rungs and leaves the sizes alone. The
rule reads two kinds of quantity and no others. It reads the bias scale this row already
publishes, which is 0.0036 at the first rung, with the decay a second-order remainder predicts. It
reads the `both_wrong` control arm's empirical spread. It does not read the slope, the interval or
the verdict of either positive rate cell.

The ladder stays at 1,500, 3,000 and 6,000. The two outer rungs run 2,400 replications and the
middle rung keeps 800, because a rung with zero weight buys no resolution. The design is declared
to resolve a slope of magnitude one. That is the separation this family exists to distinguish,
between the second-order prediction of minus one and the non-contraction alternative of zero.

The claim is about the instrument and not about the verdict. The run publishes what it produces. A
`rate_outcome_correct` interval that still covers zero is a result this ladder reports, and not a
failure of this declaration.

#### What the two readings found

Both readings ran once, under the rules the subsection above declared before them. Each publishes
what it produced.

Both runs also moved the environment and the `src/` tree. SciPy moved from 1.17.1 to 1.18.0, and
Python moved from 3.11.13 to 3.13.7, on the runs that produced these rows. Each study's
`manifest.json` records the versions its committed rows came from, and its git history records the
versions they replaced. On all three studies, the committed history separates the design change
from the source and environment changes together. It does not separate the source from the
environment.
"[What the committed history already separates](#what-the-committed-history-already-separates)"
gives the comparison.

| study | what the source and environment changes moved together |
| --- | --- |
| both multi-arm `n_500` studies | no row by more than 1e-6, and no covered flag |
| DR-TMLE contraction ladder | each slope interval by less than 2e-5 at 800 replications per rung |

##### The `n_500` endpoints are a boundary resolution

| study | paired coverage gain | reading |
| --- | --- | --- |
| selector-based multi-arm C-TMLE | -0.006875 to +0.001625 | the interval covers zero and its upper endpoint is below 0.005 |
| multi-arm point-treatment DR-TMLE | -0.005625 to +0.004500 | the same, and the upper endpoint sits near 0.005 |

The gain is the stratified arm's coverage minus the unstratified arm's, on one law and one set of
draws. Neither study shows a fold-policy effect large enough to move the endpoint across the 0.90
floor. The attribution in the table above therefore rests on a measurement rather than on the
absence of one. The DR-TMLE reading has little room, and its evidence page says so.

The pilot under-estimated the discordance rate on both studies. The selector study declared 0.0117
and ran at 0.0214. The DR-TMLE study declared 0.0250 and ran at 0.0301. A paired interval's
half-width grows with that rate, so both readings came out wider than the declaration predicted.

The selector reading still resolves 0.005. Its declared half-width was 0.0031 to 0.0038 and its
realized half-width is 0.0043. The DR-TMLE reading does not resolve 0.005. Its declared half-width
was 0.0046 and its realized half-width is 0.0051. The "resolves 0.005: yes" row of the declaration
table above is therefore false as run. This row records that outcome and leaves the declaration
standing.

The DR-TMLE reading is carried by where its point estimate fell, and not by the resolution it
achieved. Its paired gain is -0.000625, so its upper endpoint reaches 0.004500 instead of the
0.0051 a zero-centred interval of this width would reach. A gain 0.0005 larger would have put that
endpoint at 0.005 and made the reading indeterminate. At the realized discordance rate, about
8,200 paired draws would have brought the half-width below 0.005. The selector reading does not
depend on where its point estimate fell, because its half-width alone resolves 0.005.

Nothing else moved. No verdict changed on either study, over 12 cells on the selector row and 22
on the DR-TMLE row. Both gated `n_500` cells keep their 400-replication budget.

##### The DR-TMLE contraction slope resolves

| cell | before | after |
| --- | --- | --- |
| `rate_outcome_correct` | -3.040132 to +0.123221 | -1.534453 to -0.482268 |
| `rate_both_wrong`, the control | -0.003031 to +0.014273 | +0.003726 to +0.013538 |

The positive cell's interval now sits below zero, and the control still fails to contract. The
fitted slope is -0.917626, near the -1 a second-order remainder predicts.

The declaration made one falsifiable prediction, and the run confirms it. The control's half-width
had to narrow by about the square root of three, from 0.008652 to about 0.004995. It measures
0.004906. The source and the environment also moved on the same run. At 800 replications per
rung, the two changes together leave the control interval unchanged to six decimals. The narrowing
therefore comes from the budget.

##### One correction the run forced

The two outer rungs carry their own coverage verdicts, and raising their replications raised those
gates too. `double_robust_contraction/treatment_correct_n1500` turned green when only its budget
moved. It covers 732 of 800 at the declared budget, for a 99% lower endpoint of 0.886433. It
covers 2,220 of 2,400 at the raised one, for 0.910081. Nobody declared that budget, so this row
refuses it.

Each rung's coverage verdict is therefore read at its declared 800 replications, and the extra
draws serve the slope alone. `treatment_correct_n1500` is red again, exactly as it was.

What separates the two budgets is registration, and not a difference between the gates.

A larger budget walks either endpoint toward the truth rather than toward the margin. It buys a
pass exactly when the truth already satisfies the gate. This cell is the demonstration. Its
coverage is 0.9150 at 800 replications and 0.9250 at 2,400, and both sit above the 0.90 floor, so
the extra draws resolved a cell that the declared budget could not read. The slope behaves the
same way. Neither gate is immune, and the table that once claimed otherwise here was wrong.

The protection this row offers is therefore procedural.

| budget | how it was set | what that buys |
| --- | --- | --- |
| the two outer rungs | declared before the run, under a rule that reads the law's bias scale and the control's spread, and that names the verdicts it refuses to read | a design that could have failed. Its one falsifiable prediction is checked above |
| each rung's own coverage | never declared. It rose as a side effect of the rung budget | nothing. An undeclared budget cannot be distinguished from one chosen after the verdict |

A budget nobody declared is the fishing RM18 refuses, whatever the statistics say. So the rung
verdicts return to the budget they published, and the slope keeps the budget it declared.

The study now passes 19 of 22 property cells, against 18 before. The one verdict that changed is
`rate_outcome_correct`.

(the-one-sided-robustness-reading-declared-before-it-is-computed)=
#### The one-sided robustness reading, declared before it is computed

This subsection declares the reading that `RM18-one-sided-bias` asks for. It precedes the
computation of every statistic below. `tests/diagnostics/rm18_one_sided_bias/` will hold the code
and its output.

Three cells fail bias equivalence while one nuisance is correct. The reading asks whether R
`drtmle` reproduces each finite-sample bias on the same rows. Agreement shows that a result is
shared by these two implementations at this size. It does not distinguish the simulation law from
an algorithmic feature or error that both implementations share.

| study | rows that R `drtmle` fits | what the reading can use |
| --- | --- | --- |
| `canonical-drtmle` | the primary `replicates.csv.gz` at n = 3,000, on the `outcome_correct`, `treatment_correct` and `both_correct` scenarios | paired rows for both red configurations, on the primary scenario and not on the property cell |
| `canonical-multi-arm-drtmle` | the primary scenario at n = 2,000, where both nuisances are correct | no paired row for the `treatment_correct` configuration |

The binary property cells run at n = 1,500, so the binary reading reads the primary rows at
n = 3,000. The multi-arm property cell and the multi-arm primary scenario both run at n = 2,000.

The reading computes four statistics. Each interval is a 99% Student interval, the level every
bias gate in these studies uses.

| statistic | what it measures | binary | multi-arm |
| --- | --- | --- | --- |
| (i) | the `cleverly` bias on the primary rows of the configuration | `ate`, per red configuration | `ate[medium vs high]`, both nuisances correct |
| (ii) | the R `drtmle` bias on the same rows | yes | no |
| (iii) | the paired difference, `cleverly` minus R, per replication | yes, on each red configuration and on `both_correct` | no |
| (iv) | a Welch interval for the property-cell bias minus statistic (i) | no, because the sizes differ | yes, on `double_robustness/treatment_correct` |

The binary rule reads each red configuration separately. An interval is on the side of (i) when
it excludes zero with the sign of the (i) point estimate.

| reading | condition |
| --- | --- |
| shared | (ii) excludes zero on the side of (i), and (iii) covers zero |
| estimator-specific | (ii) covers zero, and (iii) excludes zero on the side of (i) |
| mixed | (ii) and (iii) both exclude zero on the side of (i) |
| unresolved | any other pattern. This includes an (i) interval that covers zero, and a `both_correct` (iii) interval that excludes zero |

A `both_correct` paired interval that excludes zero means the two implementations differ where
neither nuisance is wrong. An increment on a one-sided configuration is then not specific to
one-sided robustness, so the rule reads it as unresolved.

The multi-arm rule reads statistic (iv) alone. A review corrected the labels after the first run.
A fixed-size contrast cannot establish asymptotic consistency.

| reading | condition | what it means |
| --- | --- | --- |
| finite-sample excess not detected | (iv) covers zero | this comparison does not distinguish the two streams at this size |
| finite-sample excess detected | (iv) excludes zero | the property-cell stream and fully specified primary stream have different mean biases under this comparison. The statistic does not identify the cause or establish asymptotic inconsistency |

The `treatment_correct` configuration has a correct treatment mechanism and a misspecified outcome
regression. `_nuisances` in `tests/studies/multi_arm_drtmle_properties.py` builds that pair. The
review correction changes no statistic.

The reading states a result and changes no verdict. The three cells stay red under `reporting`. A
`mixed` or `estimator-specific` binary reading opens a new row that localizes the increment.

The planning of this row read some of these data first, so this declaration discloses them. The
planning read found these values.

| cell | what the planning read computed | what it saw |
| --- | --- | --- |
| binary `outcome_correct`, n = 3,000 | the paired difference, as in (iii) | a 99% interval that covers zero |
| binary `treatment_correct`, n = 3,000 | the paired difference, as in (iii) | a 99% interval of about 0.00007 to 0.00194, which excludes zero |
| multi-arm `treatment_correct` | a Welch 99% interval | about -0.0003 to 0.0077, which covers zero |

The planning read expected a `mixed` binary reading for `treatment_correct`. It did not compute
statistic (ii) for either configuration. The rule above does not depend on those values, and the
declared statistics govern where they differ from the planning read.

(what-the-one-sided-reading-found)=
#### What the one-sided reading found

The reading ran once, under the declaration above. Commit `7ac37dc` recorded it at
2026-09-22 05:21 UTC, which is 2026-09-21 in the local time zone (UTC-7). Every value below
comes from `readings.csv` in
[`tests/diagnostics/rm18_one_sided_bias/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_one_sided_bias).
`tests/unit/test_rm18_one_sided_bias_diagnostic.py` rebuilds that file from the committed
artifacts. Each binary interval is a 99% Student interval over 800 replications at n = 3,000.

| configuration | (i) `cleverly` bias | (ii) R `drtmle` bias | (iii) `cleverly` minus R | reading |
| --- | --- | --- | --- | --- |
| `outcome_correct` | 0.000105 to 0.003504 | 0.000204 to 0.003588 | -0.000295 to 0.000111 | `shared` |
| `treatment_correct` | 0.001284 to 0.004921 | 0.000277 to 0.003917 | 0.000068 to 0.001942 | `mixed` |
| `both_correct` | not read | not read | -0.000081 to 0.000078 | enters both readings |

The `both_correct` paired interval covers zero. The two implementations therefore agree where
neither nuisance is wrong, and neither binary reading is `unresolved`.

On `outcome_correct`, R `drtmle` shows the same signed bias on the same rows. The paired difference
covers zero. The bias is shared by these two implementations at this size, and this comparison
does not identify its source.

On `treatment_correct`, R `drtmle` also shows a bias above zero. The unadjusted paired comparison
finds a between-implementation increment, with a point value of 0.001005 against the `cleverly`
bias of 0.003102. The signal is exploratory because the declaration adjusted for no multiplicity.
It does not establish an implementation defect or name the step of the fit that produces it.
[RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle) gives the adjusted interval. RM19
also carries the question of the step.

| statistic | rows | 99% interval |
| --- | --- | --- |
| (i) | the primary `ate[medium vs high]` rows at n = 2,000, both nuisances correct, 800 replications | -0.003016 to 0.001850 |
| the property-cell bias | `double_robustness/treatment_correct` at n = 2,000, 600 replications | 0.001469 to 0.007176 |
| (iv) | the property-cell bias minus (i), by Welch | 0.001160 to 0.008650 |

Statistic (iv) excludes zero, so the multi-arm reading is `finite-sample excess detected`. One
property-cell stream differs from the fully specified primary stream at this size. The outcome
regression is misspecified and the treatment mechanism is correct. No R comparator fits this
configuration, and the statistic does not identify the source of the difference.

The declaration disclosed a planning-read Welch interval of about -0.0003 to 0.0077. That read
used `double_robust_contraction/treatment_correct_n2000` as its second sample. That rung carries
the same configuration at the same size, so it is not the fully specified sample that the
declaration names. The declaration states that the declared statistics govern, so the reading
uses (iv).

A review then asked how the rung, the other sample of the same configuration, compares.
`readings.csv` now carries these rows with `scope` set to `supplementary`. They are not part of
the declared reading, and no rule reads them.

| supplementary row | 99% interval, by Welch |
| --- | --- |
| the rung bias minus (i) | -0.002581 to 0.004917 |
| the property-cell bias minus the rung bias | -0.000296 to 0.007771 |
| both samples, 1,200 replications, minus (i) | -0.000123 to 0.006196 |

The independent rung does not reproduce the excess. The two samples of the configuration are not
told apart, and the pooled comparison against the primary rows covers zero. The overall evidence
is inconclusive. It does not establish that the configuration adds bias or that it adds none.

No verdict, margin, budget or artifact moved. The three cells stay red under `reporting`, and
the ledger keeps `RM18-one-sided-bias` as their owner.

(the-learner-weight-standard-error-diagnostic-declared-before-it-runs)=
#### The learner-weight standard-error diagnostic, declared before it runs

This subsection was the original declaration for the diagnostic that `RM18-learner-weight-se`
asks for. A review found that it did not fix the P1-to-P4 regimen scope, the P3 floored-set union or
the relative-difference formula before the first run. The review rerun declaration in
`tests/diagnostics/rm18_learner_weight_se/README.md` fixes those choices and was committed and
pushed before the rerun. The directory holds the code, refit rows and run log.

The cross-fitted `weighted-ltmle-crossfit` study reports very large standard errors on 13 of its
1,200 `learner_weight_necessity` replicates. The same 13 replicates exceed 1 in both arms. Their
standard errors run from 25.57 to 47,459.22. The largest standard error below 1 is 0.3472, and the
arm medians are 0.0739 and 0.1013. The ordinary `weighted-ltmle` study has no standard error above
1 in either arm. All of these values come from each study's committed `property-replicates.csv.gz`.

The diagnostic tests one mechanism. Both arms fit the treatment and censoring mechanisms with the
saturated `CellMeans` learner on each fold's training rows. A training complement can hold no row
with the action a held-out follower took. The cell mean is then zero, and `g_bounds` replaces the
cumulative probability with the floor of 1e-8. The inverse of that floor is 1e8, so one unit can
dominate the influence curve.

| item | rule |
| --- | --- |
| selection | every replicate whose committed `std_error` exceeds 1 in either `learner_weight_necessity` arm. The expected set is 17, 20, 258, 278, 343, 487, 533, 711, 889, 978, 981, 998 and 1099 |
| comparison | the 13 lowest replicate indices outside the selected set, which are 0 to 12 |
| refit | each selected and comparison replicate in both arms, through `_fit_replication` in `tests/studies/weighted_longitudinal_properties_common.py` with `cross_fit=True`, on the payload `_payloads` builds |
| reproduction control | each refit separately satisfies `abs(refit - committed) / abs(committed) <= 1e-9` for `estimate` and `std_error` |
| run | one process that uses every core, with no other study or diagnostic on the machine |

The output contains exactly every declared group, replicate, arm and fitted regimen once, with
finite statistics. P1 through P4 read `always` and `never`, the two fits of the reported contrast.
P3 defines its floored set as the union of their floored followers. P5 reads all three fitted
regimens, including `treat_if_l2`.

If the selection rule gives a different set, the run uses the set the rule gives and reports the
difference. If any refit misses the reproduction control, the diagnostic states "harness not
validated, no reading" and reads no prediction.

A follower is a unit with a nonzero final-node clever covariate for the regimen. A floored follower
is a follower whose `cumulative_unbounded` prefix at the final node is below the lower `g_bounds`
limit of 1e-8. The diagnostic records these statistics from each refit.

| level | statistic | source |
| --- | --- | --- |
| replicate and arm | the contrast `std_error` | the refit result |
| replicate and arm | the share of the sum of squared contrast influence-curve values that floored followers carry | `influence_curve_scaled` of the `always` and `never` fits |
| replicate, arm and regimen | the count of followers, of floored followers, and of followers whose raw prefix is exactly zero | `cumulative_unbounded` and the final `SequentialStep` |
| replicate, arm and regimen | the smallest raw prefix among followers | `cumulative_unbounded` |
| replicate, arm and regimen | `max_weight` and `effective_n` | the `RegimenFit` properties of those names |

The diagnostic reads five predictions.

| prediction | statement | threshold |
| --- | --- | --- |
| P1 | every selected replicate has at least one floored follower in each arm | one floored follower |
| P2 | no comparison replicate has a floored follower in either arm | zero floored followers |
| P3 | in every selected replicate and arm, floored followers carry most of the squared influence curve | a share of at least 0.9 |
| P4 | in every selected replicate and arm, each floored follower has a raw prefix of exactly zero | exactly zero |
| P5 | in every comparison replicate, arm and regimen, `max_weight` stays below the size an explosion needs | 2,000 |

The thresholds came from a read of the committed standard errors, and from nothing else. A
standard error above 1 against a median near 0.1 needs the excess to carry about 99% of the
variance. A share of 0.9 therefore leaves room for several floored rows. The outcome is binary, so
one unit adds at most about its weight divided by n to the standard error. At n = 2,000, a
standard error of 1 therefore needs a weight near 2,000. The read touched no mechanism statistic.

| reading | condition |
| --- | --- |
| finite-sample empty-cell instability in the out-of-fold mechanism estimate | P1, P3 and P4 hold |
| the floor, not the saturated fit | P1 and P3 hold, and P4 fails |
| unresolved | P1 holds, and P3 fails |
| not the floor | P1 fails |
| harness not validated, no reading | a refit misses the reproduction control |

P2 and P5 do not enter the reading. The diagnostic reports each of them beside the reading. A P2
failure means a floored follower does not always make a standard error explode. The diagnostic
changes no verdict and no committed row.

(what-the-learner-weight-diagnostic-found)=
#### What the learner-weight diagnostic found

The review rerun started at 2026-09-22 14:47 UTC from clean pushed commit `c825ad5`, after the
complete rerun declaration was committed and pushed at `ad05b33`. Its 52 refits completed in 4.9
seconds and reproduced `refit.csv` byte for byte. The reading label alone changed as declared.
`run.log` records the two commit hashes, command, thread limits, output hashes and clean-tree
checks.

The original diagnostic ran from the clean tree at `ff6106b`. It started
at 2026-09-22 05:42 UTC, which is 2026-09-21 22:42 in the local time zone (UTC-7). Every value
below comes from `reading.csv` and `refit.csv` in
[`tests/diagnostics/rm18_learner_weight_se/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_learner_weight_se).
`run.log` in that directory records the command, the runtime and the exit code.
`tests/unit/test_rm18_learner_weight_se_diagnostic.py` checks that `reading.csv` follows from
`refit.csv`.

`run.log` also records a first attempt at 05:34 UTC, at commit `ceb0ce2`. That attempt stopped
before any fit returned, because `map_parallel` split each payload into seven arguments.
`ceb0ce2` was then amended into `ff6106b`, which carries the fix. `ff6106b` was committed 6
seconds before the run started, and it reached the remote in the same push as the results. The
local commit times and `run.log` attest that order, and the remote does not.

P1 to P4 read the `always` and `never` fits, the two regimens of the reported contrast. The
declaration states no regimen for them. P5 reads every fitted regimen, because the declaration
states that scope for it. The run first read P5 over the two contrast regimens alone. A review
found the mismatch, and `reading.csv` was rebuilt from the unchanged `refit.csv`.

The selection rule gave the declared set of 13 replicates. The comparison set is replicates 0 to
12. The committed standard errors of the selected set run from 25.57 to 47,459.22.

| item | result | value |
| --- | --- | --- |
| reproduction control | holds | largest relative difference 1.317e-15 on `estimate` and 1.386e-15 on `std_error`, over 52 refits, against 1e-9 |
| P1 | holds | 1 to 3 floored followers in each of the 26 selected replicate and arm pairs |
| P2 | holds | no floored follower in the 26 comparison replicate and arm pairs |
| P3 | holds | floored followers carry from 0.999923 to 1.000000 of the squared contrast influence curve |
| P4 | holds | each floored follower has a raw prefix of exactly zero |
| P5 | holds | comparison `max_weight` from 9.51 to 373.78 over all 78 regimen fits, against 2,000. Over the two contrast regimens it runs from 13.87 to 373.78 |
| reading | finite-sample empty-cell instability in the out-of-fold mechanism estimate | P1, P3 and P4 hold |

The selected fits count 44 floored followers, and all of them are in the `always` regimen. No
`never` or `treat_if_l2` fit has a floored follower.

Each floored follower has a raw cumulative prefix of exactly zero. The saturated `CellMeans` fit
gives that value when the training complement holds no row with the follower's action in its
cell. The bound then replaces the prefix with 1e-8, and that one unit carries almost all of the
squared influence curve.

The reading names the mechanism and changes nothing. No verdict reads these standard errors, so
no ledger row moves. This row proposes no change to the learner, the bound or the study.

(what-the-multi-arm-rung-design-would-cost)=
#### What the multi-arm rung design would cost

`RM18-slopes` sanctions a rung design for the two multi-arm contraction slopes and defers its run.
This subsection computes its cost from the committed rows. It applies the sizing rule that
`CONTRACTION_REPLICATES` in `tests/studies/drtmle_properties.py` declares for the binary ladder.
The rule reads the first-rung bias and the `both_wrong` control's spread. It does not read the
slope, the interval or the verdict of either positive rate cell.

The multi-arm ladder runs n = 2,000, 4,000 and 8,000, with 600 replications on each rung. Every
input below comes from `tests/canonical/multi_arm_drtmle/properties.csv`.
[`tests/diagnostics/rm18_rung_cost/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_rung_cost)
holds the computation and its output, `cost.csv`.
`tests/unit/test_rm18_rung_cost_diagnostic.py` recomputes each delta-method figure from the
committed rows.

| input | value | source |
| --- | --- | --- |
| first-rung bias, `outcome_correct_n2000` | 0.001127, with a 99% interval from -0.001651 to 0.003905 | the `bias`, `bias_ci_lower` and `bias_ci_upper` columns |
| first-rung bias, `treatment_correct_n2000` | 0.000585, with a 99% interval from -0.002274 to 0.003444 | the same columns |
| control spread `c` | 1.1660, the mean of 1.1473, 1.1397 and 1.2108 | `empirical_se` times the square root of `n`, on the three `both_wrong` rungs |

The rule carries the bias down the ladder as `1/n` and the spread as `1/sqrt(n)`. The middle rung
has zero weight, so the slope is the difference of the outer rungs divided by `log 4`. At `R`
replications on each outer rung, the delta-method 99% half-width is
`2.5758 * c * sqrt(n1 + n3) / (log 4 * b1 * n1 * sqrt(R))`. Here `b1` is the first-rung bias,
`n1` is 2,000 and `n3` is 8,000. The target is a half-width of one, as the binary rule declares.

| arm | smallest `R` by the delta method | smallest `R` by a surrogate simulation |
| --- | --- | --- |
| `outcome_correct` | 9,240, from 9,239.22 rounded up | 19,000 to 20,000 |
| `treatment_correct` | 34,267, from 34,266.75 rounded up | 70,000 to 73,000 |

The surrogate draws each rung's bias from a normal law with that mean and spread. It reads the
99% range of the fitted slope over 40,000 draws. It searches in steps of 1,000 at three fixed
seeds, and the table gives the range over those seeds.

The binary rule quoted a surrogate of this kind. On the binary inputs, this surrogate gives 1.93,
1.00 and 0.84 at 800, 2,000 and 2,400 replications. Each is within 5% of the projection that the
rule records. The delta method gives 0.64 at the binary 2,400. It understates the width when a
rung's bias is near its Monte Carlo error, so its figure is a lower bound.

Both slopes need a resolved interval, so the `treatment_correct` arm sets the cost. The design
needs at least 34,267 replications at each outer rung, against 600 now. The surrogate puts the
figure at 70,000 to 73,000. The middle rung keeps 600, the floor its coverage row needs.

The cost has a second limit. Both first-rung bias intervals cover zero, so the input the rule
reads is not resolved. The needed `R` grows as the inverse square of the true bias. A true bias
below the point estimate raises the cost without a bound. `RM18-slopes` therefore records the
cost and does not run the design.

#### What the source search found for the first three asks

The first three asks each need a derivation. The [Eligibility](#eligibility) rule asks this
repository to locate published theory, and not to create it here. A 2026-09-20 source search read
the sources below and found no result for any of the three. A 2026-09-21 follow-up also checked the
latest upstream `lmtp` correction. The pooled update has since closed the third ask. The first two
acceptance rows remain open, and the investigation row named below carries each contract.

| ask | verdict | where the contract lives |
| --- | --- | --- |
| an inference result for the shipped selector path | no published result | [F18](#f18-selector-path-c-tmle-inference) |
| an inference result for the generated design | a proved binary scalar result, and no result for the shipped construction | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| a targeting result for the fold-local longitudinal recursion | no result for the fold-local update. Theorem 3 certifies the pooled update, which now ships | the [cross-fitting section](technical-reference/longitudinal-tmle.md#cross-fitting-the-recursion) of the technical reference |

| ask | sources read, and their exact limits |
| --- | --- |
| the shipped selector path | van der Laan and Gruber (2010), *IJB* 6(1), DOI 10.2202/1557-4679.1181: Theorem 4 assumes the expansion that defines the adaptive-mechanism contribution, and Section 4.3 records cross-validation over-selection as an open irregularity. Gruber and van der Laan (2010), *IJB* 6(1), DOI 10.2202/1557-4679.1182, apply the method and state no post-selection inference result. Ju, Chambaz and van der Laan (2018), arXiv:1804.00102: Theorem 1 permits an extra contribution along a twice-differentiable continuous nuisance path for a binary scalar target, and Lemma 2 zeroes it in a correct-outcome product-rate regime. A discrete stopping index is not a differentiable path. Ju et al. (2019), *SMMR* 28(2), DOI 10.1177/0962280217729845, Section 7.4, forms intervals from the ordinary EIF, which is the fixed-candidate curve. Adaptive debiased machine learning, arXiv:2307.12544v2, is the closest positive route, and it needs the working model to approximate a fixed, nonrandom oracle model. Nobody has proved that for a global depth chosen from nested targeted-loss folds. Leeb and Pötscher (2006, *AoS* 34(5); 2008, *ET* 24(2)), Loftus (arXiv:1511.08866), and Markovic, Xia and Taylor (arXiv:1703.06559) supply no transfer, because this selector has no Gaussian quadratic reduction and no randomized or jointly Gaussian criterion-and-target limit. The multi-arm obligation has no published treatment at all |
| the generated design | Benkeser, Cai and van der Laan, *Statistical Science* 35(3), DOI 10.1214/19-STS735, preprint arXiv:1901.05056: Theorem 1 proves, for the binary treatment-specific mean, the expansion with the ordinary adaptive-propensity curve and no separate generated-design term. Theorem 1's estimator is a full-sample procedure, and Section 3.1 supplies a cross-validated variance alone. The cross-fitted point estimator appears in Appendix D, where the authors call its proof completely analogous to Zheng and van der Laan (2011) and outline it. Appendix D's binary ATE, which uses both arm predictions with one signed coefficient, is an algorithm with no theorem. No result covers a shared-multinomial vector extension. Four sources do not close it. Ju, Benkeser and van der Laan (2020), *Biometrics* 76(1):109-118, DOI 10.1111/biom.13121, build a different construction, in which outcome information enters through HAL penalty weights. Shortreed and Ertefaie (2017), *Biometrics* 73(4):1111-1122, DOI 10.1111/biom.12679, select variables and prove no inference theorem. Escanciano and Pérez-Izquierdo (2023), arXiv:2301.10643, remove the indirect first-step effect, and the direct effect of learning the generated regressor remains. DOPE, arXiv:2402.12980v2, centers Theorem 4.3 at a data-adaptive target conditional on a representation learned on an independent sample, adds a fixed-target delta-method variance in Proposition 4.4, and leaves the cross-fitted proof open in its appendix |
| the fold-local longitudinal recursion | Díaz, Williams, Hoffman and Schenck (2023), Section 5.2, Step 3, journal page 852, fits the fluctuation "using all the data points in the sample", and Theorem 3, page 853, certifies that pooled update. Zheng and van der Laan (2011) and Levy (2018), arXiv:1811.04573, both describe the targeting step as a pooled regression over validation folds. Chernozhukov et al. (2018) certifies the cross-fitted orthogonal moment, and not a plug-in of a train-fold-targeted regression. Williams and Díaz (2025), *Observational Studies* 11(3):365-367, correct Assumption 2 and the positivity statement, and say nothing about cross-fitting or targeting. Upstream `lmtp` commit `9996b04` calls its 1.5.4 training-fold fluctuation a bug because the EIF was not mean zero, and it moves the fluctuation to validation rows. The [cross-fitting section](technical-reference/longitudinal-tmle.md#cross-fitting-the-recursion) of the technical reference carries the published locators |

A second 2026-09-21 search looked for C-TMLE sources. It found no result that closes F18, the
cross-fitted multi-arm part of F19, or the fold and outcome-scale rules. The
[Collaborative TMLE references](references.md#collaborative-tmle) give each locator and state
which version this search read. Each locator below is an arXiv-version locator, and the published
section and theorem numbers may differ.

| ask | source | what it supplies, and its limit |
| --- | --- | --- |
| the shipped selector path | Cui and Tchetgen Tchetgen (2024), *Biometrika* 111(2), DOI 10.1093/biomet/asad055 | Theorems 5.1 and 5.2 of arXiv:1911.02029v6 give an oracle inequality and the consistency of a cross-validated learner selector. Section 6 says a Wald interval at the selected learners "is completely blind to the model selection step" and "may not yield uniformly valid confidence intervals". It describes two alternatives and leaves their formal comparison outside the paper |
| the shipped selector path | Qiu, Luedtke and Carone (2021), *Bernoulli* 27(4), DOI 10.3150/20-BEJ1309 | Theorem 4 of arXiv:2003.01856v2, Section 4.4, keeps the ordinary influence function after cross-validation selects a sieve dimension. Its conditions must hold for a deterministic dimension, and Condition C5 must hold for every candidate. This is a template for the uniform-expansion route of F18, and not a result for a targeting step |
| the shipped selector path | Bibaut and van der Laan, arXiv:1706.07408v2 | Theorems 1 and 2 select a scalar index on separate subsamples for a possibly non-regular target. The package selects on the full sample |
| the shipped selector path | Ju, Schwab and van der Laan (2019), *SMMR* 28(6), DOI 10.1177/0962280218774817, and Ju, Wyss et al. (2019), *SMMR* 28(4), DOI 10.1177/0962280217744588 | neither states a post-selection theorem. The first reports standard errors smaller than the sampling spread for C-TMLE in its Section 4.5 experiments. The second forms its data-analysis intervals from the analytic influence curve |
| the shipped selector path | Liu (2018), Harvard dissertation, Chapter 2 | derives asymptotic laws for C-TMLE coefficients, squared errors, prediction risks and M-fold cross-validation risks, and studies post-selection AMSE. It assumes the propensity candidates, outcome variance and covariate law are known; its target is one binary treatment-specific mean, and its practical post-selection AMSE is simulated rather than available analytically. It does not cover the shipped learned nuisances, stopping rule or joint targets |
| the shipped selector path | Dang, Tarp, Abrahamsen et al. (2025), *Journal of Causal Inference* 13:20240041, DOI 10.1515/jci-2024-0041, preprint arXiv:2210.05802 | derives a nonstandard limit and Monte Carlo intervals for an experiment selector trained separately inside each fold. It is positive precedent for F18's fold-local redesign route. It does not cover the shipped targeted-loss criterion, target or global shared stopping index |
| the shipped selector path | van der Laan, Qiu, Tarp and van der Laan (2026), *Journal of Causal Inference* 14:20240025, DOI 10.1515/jci-2024-0025; Karim (2026), arXiv:2607.02787 | the first proves asymptotic normality for an adaptive working-model RCT augmentation estimator. The second is a simulation caution about selection-unaware standard errors. Neither treats the shipped selector |
| the generated design | Ju, Benkeser and van der Laan (2020), *Biometrics* 76(1) | Theorem 1, Section 3.4 of arXiv:1806.06784v3, adds a first-order term to the influence function when the propensity limit is intentionally inconsistent |
| the generated design | Zhang, Shao, Yu and Wang (2018), and Ma, Zhu, Zhang, Tsai and Carroll (2019) | this search read the abstracts only. The first states that an estimated sufficient dimension reduction changes the asymptotic variance unless the reduction keeps superfluous covariates. The second states that an efficient-influence-function estimator with reduction-estimated nuisances stays semiparametrically efficient. Neither applies a targeting step |

On 2026-09-21 the [Eligibility](#eligibility) rule gained its natural-extension exception. The same
day, this row read F18 and F19 against that exception.

| ask | verdict under the exception | why |
| --- | --- | --- |
| the shipped selector path | does not qualify | the selector is a data-adaptive selection step, and the rule names that step as one with no established argument. Van der Laan and Gruber (2010), Section 4.3, "Irregular C-TMLE and super efficiency", records cross-validation over-selection as "an area of study" ([PMC2898626](https://pmc.ncbi.nlm.nih.gov/articles/PMC2898626/)). No published base exists for a stopping index shared across arms |
| the generated design | does not qualify | Theorem 1 of Benkeser, Cai and van der Laan proves one full-sample binary treatment-specific mean. The supplement gives a binary direct-ATE algorithm with two outcome-prediction columns and one signed fluctuation coefficient, but no theorem. The package instead fits one shared multinomial mechanism on `K` estimated columns and a `K`-column joint fluctuation. Cramér--Wold combines already-established scalar expansions; it does not establish those expansions or their remainders. The exact full-sample construction therefore remains open alongside the cross-fitted default |

F18 and F19 record the same verdicts.

A 2026-09-21 search and this review found no source that closes F18 or F19. The source-by-source
record, including whether only an abstract was read, lives once in
[Collaborative TMLE references](references.md#collaborative-tmle). The closest constructive result
is Dang, Tarp, Abrahamsen et al. (2025): it supports the fold-local redesign route, not inference
for the shipped global stopping index. No reviewed source establishes the generated design's
shared multinomial mechanism, joint targeting or cross-fitted covariance.

#### Witnesses and evidence

| claim | evidence |
| --- | --- |
| every published verdict is recomputed from the committed replication rows | `tests/unit/test_method_evidence.py::test_paper_property_verdicts_are_recomputed_from_the_replication_rows` |
| the shipped fold and scale rules have mutation-controlled witnesses | `tests/unit/test_fold_policy_rules.py` |
| the pooled longitudinal update matches a longhand recomputation, and four mutations break it | `tests/unit/test_pooled_longitudinal_targeting.py` |
| the runtime isolation attributes each changed verdict to the code or the runtime | `tests/diagnostics/rm18_runtime/compare.py` reads the four arms of each study and writes `isolation.csv` in that directory. `run.log` records each arm's command, runtime, `pip freeze` digest and exit code, and `arms/` holds each arm's manifest |
| the committed history separates the runtime from the code on three point-treatment studies and on the single-fold control rows | `tests/diagnostics/rm18_runtime/row_drift.py` reads the committed rows with `git show` and writes `row-drift.csv` in that directory |
| the end-of-study fold-policy 2x2 | a controlled run at commit `eeaa1ce` over the study's own 8,000 registered seeds, with the law, learners, size, budget and margins held fixed. The attribution table above gives both arms |
| a reported acceptance sweep found no refusal in its selected cells | a sweep at commit `6c91a48` reported that 340,600 primary and property replicates from all sixteen cross-fitted studies reached the first nuisance fit, with zero refusals. It counted one cell and one replication index as one replicate, including paired cells that share a sample. An earlier sweep at commit `5f32c14` reported 115,400 replicates from the eleven point-treatment studies |
| a reported replay matched the sampled committed rows | a replay at commit `5f32c14` reported matches for the first three replicates of every primary and property cell in the eleven point-treatment studies. Its worst relative difference was 1.5e-13, against a 1e-12 tolerance. The replay did not check every committed row |

The last three rows report acceptance runs made outside the repository. This branch commits no
script or run log for them, so the committed artifacts do not independently verify their results.
Each row names the commit, setup and reported result, as the
[fold and outcome-scale probes](technical-reference/cv-tmle.md#what-the-probes-measured) do.

Two cells turned green under the same changes, and this row records them so the direction is not
read as one-sided.
[Outcome-adaptive point-treatment C-TMLE](technical-reference/method-evidence/outcome-adaptive-point-treatment-c-tmle.md)
passes `type_i_error/sharp_null` at 0.0712, with a 99% upper endpoint of 0.0980, at the unchanged
800-replication budget. Outcome-adaptive multi-arm C-TMLE passes `root_n_and_efficiency/n_500` at
0.9057, which its page records as one Monte Carlo resolution at 400 replications.

#### Deferred findings and review resolutions

| finding | why it was deferred | what it needs |
| --- | --- | --- |
| `tests/canonical/lmtp_crossfit_adapter.R` screened a supplied density-ratio matrix against one on the cumulative product, although later columns can be structurally zero after an event | resolved in this review. The shared adapter now screens the first node at five standard errors and both competing adapters use that one helper. Ordinary, weighted, clustered and competing mutation smokes pass | validation changes before the same supplied matrix enters the fit, so the numeric artifacts remain valid under the result-neutral provenance ledger; eight study regenerations were not scientifically justified |
| Two property cells of `selector-based point-treatment C-TMLE` shared one law and seed: `double_robustness/both_correct` and `root_n_and_efficiency/n_500` | resolved in this review. `both_correct` moved from seed `12_100` to the next unused local seed, `12_104`; a registered-study test now rejects cross-family collisions while allowing intentionally paired arms | the declaration and test were pushed before regeneration. The independent redraw changed the `both_correct` standardized bias from 0.0139 to -0.0301; it still passes, and the red-cell ledger is unchanged |
| `simulated_confounding` cannot perturb an outcome on a fit that declares `q_bounds`. `_gaussian_outcome` subtracts the strength times a standard normal latent value from the outcome, so every nonzero outcome strength sends the perturbed outcome outside the declared support and the refit refuses it | a cross-fitted continuous fit must declare `q_bounds`, so the outcome axis of the surface is unavailable to every such fit. No perturbation on the declared scale exists to put in its place | [F23](#f23-simulated-confounding-on-a-declared-outcome-scale) |
| the multi-arm DR-TMLE contraction rung design. `double_robust_contraction/rate_outcome_correct` and `rate_treatment_correct` of `canonical-multi-arm-drtmle` stay red at 600 replications per rung, and a rung design declared before its run can read them | the binary sizing rule asks for at least 34,267 replications at each outer rung by the delta method, against 600 now. A surrogate puts the figure at 70,000 to 73,000. The first-rung bias intervals it reads both cover zero, so that cost has no upper bound. "[What the multi-arm rung design would cost](#what-the-multi-arm-rung-design-would-cost)" gives the arithmetic | one task that declares the rung budget from the rule, runs it once, and publishes the slope it produces. Each rung's own coverage verdict stays at its declared 600 replications, as `RM18-binary-slope` required for the binary ladder |
| the fixed-known-weight design that `RM18-fixed-weights` asks for. Its two cells stay red under `reporting` | the ledger work from `cdd1b33` gave each red cell an owner and ran the two declared diagnostics. It declared no design for these cells, and a design needs its own declaration before its run | one task that declares the design, pushes the declaration, runs it once, and publishes its reading. A published result for the pooled update under fixed known weights would also close the row |
| the design that `RM18-ordinary-weighted` asks for, which reads the calibration row, the sharp-null row and the `targeting_necessity` family of `weighted-ltmle` | the same. No design for these six rows is declared | one task that declares the design, pushes the declaration, runs it once, and publishes its reading |
| the re-read design that `RM18-boundary` asks for. Its eight cells sit at a finite-budget boundary and stay red at their declared budgets | the same. A larger budget chosen after a verdict is the remedy that RM18 refuses, so a re-read needs a declared design first | one task that declares a re-read budget from a rule that reads no verdict of these cells, runs it once, and publishes what it produces |
| the attribution design that `RM18-comparator-density` asks for. The calibration leg of the paired `ate_shift[+0.25 vs natural course]` row fails at a 99% upper endpoint of 0.065856 against 0.05 | the same. No design that separates `cleverly` from the comparator is declared | one task that declares the attribution, pushes the declaration, runs it once, and publishes its reading |

### RM19. One-sided robustness bias increment in DR-TMLE

RM18's one-sided robustness reading opened this row, because its declared rule reads `mixed` on
one configuration. On the binary `treatment_correct` configuration of `canonical-drtmle`, the
treatment mechanism is correct and the outcome regression is not. `cleverly` and R `drtmle` fit
the same 800 primary replications at n = 3,000.

| statistic | 99% interval | source |
| --- | --- | --- |
| `cleverly` bias | 0.001284 to 0.004921 | `tests/diagnostics/rm18_one_sided_bias/readings.csv`, statistic (i) |
| R `drtmle` bias | 0.000277 to 0.003917 | the same file, statistic (ii) |
| `cleverly` minus R, paired by replication | 0.000068 to 0.001942 | the same file, statistic (iii). It equals the `ate` row of `tests/canonical/drtmle/equivalence.csv` |
| `cleverly` minus R on `both_correct` | -0.000081 to 0.000078 | the same file |
| `cleverly` minus R, at the Bonferroni level over the three paired intervals | -0.000063 to 0.002073 | the same file, a supplementary row |

Both implementations show a bias above zero, and the declared paired interval excludes zero.
Three qualifications apply.

| qualification | what it shows |
| --- | --- |
| no multiplicity adjustment | the declaration adjusted for none over its three paired intervals. At the Bonferroni level of 1 - 0.01/3, the `treatment_correct` interval runs -0.000063 to 0.002073 and covers zero |
| the registered paired verdict | the same `ate` row of `tests/canonical/drtmle/equivalence.csv` concludes `equivalent`. Its point difference is 0.336 of its similarity margin of 0.002990, and its upper endpoint is 0.649 of it |
| the other configurations | the `treatment_correct` paired differences exceed those of `outcome_correct` by 0.000138 to 0.002055, and those of `both_correct` by 0.000067 to 0.001947. Each is a Welch 99% interval, supplementary and unadjusted |

The unadjusted comparisons place the increment on this one-sided configuration. No result names
its source.

RM19 shares the red property cells group with RM18. The [Remediation](#remediation) section
gives its priority and the reason for its place.

| question | state |
| --- | --- |
| does `cleverly` add bias over R `drtmle` on this configuration | measured by the declared unadjusted interval, which excludes zero. The Bonferroni interval covers zero |
| which step of the `cleverly` fit produces the increment | open. No diagnostic has read it |
| does the increment change with the sample size | open. The paired rows exist at n = 3,000 alone |
| the multi-arm `double_robustness/treatment_correct` excess | one stream reads `finite-sample excess detected`. The independent rung does not reproduce it, and the pooled comparison covers zero. The overall evidence is inconclusive. No comparator fits that configuration, so the source is not identified |

Next action: declare a design that localizes the binary increment, before any run of it. The
declaration names each step it tests, the statistic each step reads, and the rule that reads the
result. It reads the committed paired rows as they stand.

Acceptance:

- The declared design runs once and publishes its reading, whichever way the reading falls.
- The three one-sided cells stay red under `reporting`. `RM18-one-sided-bias` stays their ledger
  owner, and this row moves no margin, budget or law.
- A reading that names a defect in `cleverly` opens a separate fix. That fix needs a nonzero
  witness that fails without it, and each affected study regenerates under the RM18 rule.
- The declaration states whether its design can read the multi-arm excess. If it cannot, the row
  records that, and the multi-arm cell keeps its owner.

### RM20. Intervals outside every claimed contract

Before this row shipped, each surface in the table below reported `ci`, `pvalue`, and `std_error`
under the `influence_curve` status. For each one, the technical reference said that no claim
covers the interval, or no audit had read a source for it.
[RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism) found the same pattern
on the selector paths and refused their intervals. The 2026-09-22 probes used linear learners, and
every fit in them was in sample unless the row says otherwise.

| surface | probe | what the record said |
| --- | --- | --- |
| DR-TMLE with `weights_estimated=True` | `DRTMLE(estimands=("ate",))` on `make_binary_outcome(n=500, seed=3)`, with weights drawn uniform on [0.5, 2]. The fit reported `ci` (0.1161, 0.2841) under the `influence_curve` status, and the summary marked the weights as estimated | the `DRTMLE` docstring and [supported estimands](technical-reference/dr-tmle/supported-estimands.md) said that no interval claim covers the case. The conditional argument concerns $D^*$, not the reduced regressions |
| outcome-adaptive C-TMLE beyond one binary treatment-specific mean | `CTMLE(strategy="oat")` on `make_multi_arm(n=600, seed=5)` reported each `ate[...]` and `ey[...]` interval under the `influence_curve` status. The binary joint fit on `make_binary_outcome(n=500, seed=3)` did the same | Benkeser, Cai and van der Laan (2020), Theorem 1, proves one binary treatment-specific mean. [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) rejects the natural extension to the shared multinomial fit |
| cross-fitted clustered TMLE at unequal cluster sizes | `make_clustered(n=400, cluster_size=10, seed=7)`, with rows removed from half the clusters, which leaves 315 rows in 40 clusters of 2 to 10 rows. The cross-fitted fit with `id=` reported an interval and no warning | the [grouped folds](technical-reference/cv-tmle.md#grouped-folds) argument and registered study cover equal cluster sizes. The point estimator stays row weighted, but its interval lacks a result at unequal sizes |
| clustered fit with few clusters | the same construction with 6 clusters and 40 rows reported a normal-reference interval. No warning named the cluster count | Nugent et al. (2024), Section 2.2, recommend a $t$ reference with $J - 2$ degrees of freedom below 40 clusters. Benitez et al. (2023), Section 3.1.2, paragraph on inference, and Section 3.2.1, last paragraph, recommend it at every cluster count. The package applies neither rule |
| in-sample outcome-adaptive C-TMLE with missing outcomes | `CTMLE(strategy="oat")` with `delta=` on `make_missing_outcome(n=500, seed=4)` reported `ci` (1.0090, 1.4173) under the `influence_curve` status. The selector paths reported the non-inferential status on the same law | the arm-indexed audit read no source for C-TMLE with missing outcomes. [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) holds the cross-fitted refusal |
| shift, incremental, regime, MSM, and controlled-direct-effect targets, cross-fitted with missing outcomes | a cross-fitted shift fit with `delta=` on the same law, with the dose built as the arm plus standard normal noise, reported an interval for `ey_shift[+0.5]`. With linear learners, `q_bounds=(-5, 8)`, and `random_state=0`, it read 1.767366, `ci` (1.5881, 1.9467) | these fits ran outside the arm-indexed contract. `TestTheMnarTiltFollowsTheDraws` in `tests/unit/test_repeated_crossfit.py` was the only fast test that fitted repeated draws on this surface. It fitted a controlled direct effect with `repeats=2` |

Until this row, F5 said that the estimated-weight DR-TMLE interval conditions on the supplied
weights. The `DRTMLE` docstring states why that argument does not reach the reduced regressions,
so F5 no longer says it. The last two rows came from F21, and the two clustered rows came from
F22. Each of those items keeps the derivation that would support a claim.

The row asked for one decision for each surface. The table gives the three options.

| decision | what it needs |
| --- | --- |
| register a claim | a source that covers the surface, its exact locator, and a registered study with a nonzero witness for each step that can vanish at the truth. The [Eligibility](#eligibility) rule applies |
| a non-inferential status | an `inference` status for the surface, with a name and a reason, through the table that RM12 added. `ci`, `pvalue`, and `std_error` then refuse, and the plug-in values keep diagnostic names |
| a refusal | a pre-fit `CapabilityError` that names the missing result and the roadmap item that holds it |

The existing evidence limits a claim. The binary outcome-adaptive study passes 14 of 14 property
cells on the bounded law it runs. The registered clustered study fixes ten rows per cluster.
Neither study covers the multi-arm fit, unequal sizes, or few clusters. So no surface registers a
claim.

The 2026-09-22 plan recorded one decision for each surface, and each one shipped as planned. The
table gives each decision, the control that keeps its interval, and the item that holds the reopen
route.

| surface | decision that shipped | control that keeps its interval | reopen route |
| --- | --- | --- | --- |
| DR-TMLE with `weights_estimated=True` | the status `estimated_weight_plugin`, when `guard` is not empty and the weights vary | the same weights declared fixed. Also `guard=()` with the weights declared estimated, because that setting fits the ordinary TMLE, whose interval conditions on the weights. Also constant weights declared estimated, which fit the unweighted estimator (commit fbee899). Also the ordinary `TMLE` | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| outcome-adaptive C-TMLE, with complete or missing outcomes | the status `generated_design_plugin`, on every `strategy="oat"` fit | the ordinary `TMLE` on the same law | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| cross-fitted clustered `TMLE` and `DRTMLE` at unequal cluster sizes: row counts, or weight mass on a weighted fit, overall or within a reported stratum | the status `unequal_cluster_plugin`. It includes `cv_evaluation=True` and fold targeting | equal cluster sizes and masses within every reported stratum, cross-fitted, at 40 clusters. Also the same unequal clusters, fitted in sample, on `TMLE` and `DRTMLE` | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| clustered `TMLE` and `DRTMLE` with fewer than 40 positive-mass clusters in the fit or in one reported baseline stratum, in sample and cross-fitted | the status `few_cluster_plugin`. `FEW_CLUSTER_THRESHOLD` in `src/cleverly/_inference_status.py` holds the threshold of 40 | a fit with 40 contributing clusters, which pins `<` against `<=`. Also 80 clusters in two strata of 40 | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| shift, incremental, regime, MSM, and controlled-direct-effect targets, cross-fitted with `delta=` | a pre-fit `CapabilityError` that names F21. Its remedy is the in-sample fit | each composition in sample. Also the cross-fitted `ate` with `delta=`, which reaches the arm-indexed contract, and `par` with `delta=`, which keeps its F20 refusal | [F21](#f21-other-missing-outcome-cv-tmle-variants) |

The delivery review found three defects: two in the cluster rules and one in the coverage study.
This review found two more cluster-rule gaps. The table gives each finding and its probe.
The tests pair each new witness with a control and a mutation where the old rule is represented.

| finding | defect | correction | probe |
| --- | --- | --- | --- |
| C1 | the few-cluster rule counted the clusters of the whole fit. A stratum reads only the clusters with a row in it | the rule also counts the distinct clusters inside each reported baseline stratum. `fewest_clusters` in `src/cleverly/inference/cluster.py` returns the smallest count. The fit keeps one status, so one stratum below 40 withholds every interval (commit e480843) | `make_clustered(n=500, cluster_size=10, seed=7)` gives 50 clusters. A cluster-level stratum `S` holds 6 of them. In sample, with `id="cluster"` and `strata=["S"]`, the fit reported `ate[S='small']` 3.0331 with `ci` (1.3797, 4.6866) and a p-value of 0.0003. It now takes `few_cluster_plugin`, and no point estimate or plug-in standard error moved |
| C2 | the unequal-size rule read row counts only. A weighted fit also needs an argument for varying cluster weight mass | on a weighted fit, `unequal_cluster_sizes` reads the weight mass of each cluster too. Two masses are equal within `WEIGHT_MASS_RTOL`, 1e-9, of the largest (commit 2b05c78) | `make_clustered(n=400, cluster_size=10, seed=7)` gives 40 clusters of 10 rows. With cluster weights 0.5 or 2, cross-fitted, `id="cluster"`, and `weights="w"`, the fit took `influence_curve` and reported `ci` (-0.451, 1.653). It now takes `unequal_cluster_plugin` |
| C3 | `summarize_replications` raised `ValueError` when the records of one estimand mixed statuses. A `CoverageStudy` whose cluster count straddles 40 ran every replicate and then stopped | it summarizes a mix under `precedent_status`, and it names the statuses and their counts. [Validation methods](technical-reference/validation-methods.md) gives the columns (commit 3380131) | an in-sample clustered `TMLE` with the cluster count drawn from 38 to 42, 8 replicates, seed 0. It raised "the records for 'ate' declare different inference statuses". It now reports bias -0.0863, a mean plug-in standard error of 0.4253, and coverage 0.750 under `few_cluster_plugin` |
| C4 | the unequal-size rule checked only the whole fit; equal whole-cluster sizes and masses can hide unequal counts or masses within a reported baseline stratum | `cluster_inference_status` also checks each reported stratum. The point remains row weighted, but its interval lacks validation there | 40 ten-row clusters have three versus seven rows in one stratum across cluster groups; the old rule reported intervals, and the corrected fit takes `unequal_cluster_plugin`. A second witness holds five rows in each stratum but moves weight mass between them while keeping whole-cluster mass equal. Equal within-stratum sizes and masses retain intervals |
| C5 | the few-cluster rule counted zero-weight clusters as contributors | `fewest_clusters` counts positive-weight rows only, including within strata | 40 ten-row clusters with positive weight in only six formerly reported intervals. The corrected fit takes `few_cluster_plugin`; an otherwise identical fit with positive weight in all 40 retains intervals |

The review also recorded one over-refusal, and this row keeps it. A fit with `id=` and one row in
each cluster counts each row as a cluster. Below 40 rows it takes `few_cluster_plugin`, and the
same rows without `id=` keep their interval. The status withholds a number and publishes none, so
the review made no change.
[RM26](#rm26-longitudinal-clustered-intervals-at-few-clusters) applied the same rule to an
in-sample `LTMLE` fit, so the over-refusal now reaches a longitudinal fit with one unit in each
cluster.

A status keeps the point estimate. `ci`, `pvalue`, and `std_error` raise `CapabilityError` with
the reason of the status. `plugin_std_error` and `plugin_interval` keep the diagnostic, as RM12 set
up. `summary()` prints the label and the reason of the status, the nuisance note marks it, and the
E-value row reads `unavailable`. `variable_importance()` refuses before its first fit. A result
saved before the status existed loads re-stamped.

The estimated-weight decision is a status and not a refusal. The `weights_estimated` flag changes
no number (`src/cleverly/data/weighting.py`). A caller could therefore drop the flag to bypass a
refusal and receive the same interval. A status records what the flag declares. No registered
study covers the surface. `data.weight_report()` now names that fit as the one whose interval
the declaration withholds. The bootstrap warning for estimated weights no longer compares the
bootstrap with an influence-curve interval.

No shipped outcome-adaptive construction is the one that Theorem 1 of Benkeser, Cai and van der
Laan (2020) covers. The theorem covers one binary treatment-specific mean with a scalar design.
The package fits the mechanism on the outcome predictions of all $K$ arms, and it fluctuates the
$K$ arm means jointly. An `ey1`-only request uses the same design, so it is the joint fit. Missing
outcomes are also outside Theorem 1, as the transport row of F19 records.

Until the plan, the witness list below named the binary treatment-specific mean as a control. That
control has no shipped instance. The literal scalar design for one mean is the route that would
reopen it, and F19 holds that route. This row did not build it.

For unequal clusters, size means the row count, and on a weighted fit the weight mass too. The
equal-size condition belongs to the package's cross-fitted argument
([grouped folds](technical-reference/cv-tmle.md#grouped-folds)). `summary()` now prints the range
of row counts when they differ, for example "clusters = 40, sizes 2 to 10". It prints the range of
weight mass when only the mass differs, and the within-stratum range when that is the failing
condition. It names the number of positive-mass clusters when zero-weight clusters are present,
and the fewest contributors in one stratum on a stratified fit.

The [workflow page](workflow.md) runs `result.sensitivity.run_all()` on a cross-fitted clustered
fit. `TestTheWorkflowPageRunsOnUnequalClusters` in `tests/unit/test_cluster_status.py` runs the
calls of that page on an unequal cross-fitted weighted clustered study, so that page keeps
working. Its section 7 now says when that fit withholds the interval.

The few-cluster threshold rests on the two sources in the table. Both were read first-hand on
2026-09-22. $J$ is the cluster count, which Nugent et al. write as $N$.

| source | locator | what it recommends | threshold |
| --- | --- | --- | --- |
| Nugent, Marquez, Charlebois, Abbott and Balzer (2024), *Biostatistics* 25(3):599-616 | Section 2.2, last paragraph | a Student's $t$ reference with $J - 2$ degrees of freedom "In CRTs with fewer than 40 clusters randomized (N < 40)". It cites Hayes and Moulton (2009) | 40 clusters |
| Benitez et al. (2023), *Statistics in Medicine* 42(19):3443-3466. [References](references.md#grouped-folds-and-clustered-cross-fitting) gives the ten authors | Section 3.1.2, paragraph on inference, and Section 3.2.1, last paragraph | a $t$ reference with $J - 2$ degrees of freedom "As a finite sample approximation to the normal distribution" | none. It applies at every cluster count |

Nugent et al. give the only explicit threshold in a read source. Neither paper compares the
normal reference with the $t$ reference. Hayes and Moulton (2009) were not read. The surface
table above said "below about 30 to 40 clusters" until the plan, and no read source states that
range. The same wording in `cv-tmle.md` and `references.md` is corrected.

Nugent et al. mention 30 clusters only in a discussion of GEE and GLMM, in Section 1. Benitez et
al. mention 30 clusters twice. Section 3.1.2 recommends leave-one-cluster-out cross-validation
"for small trials (eg, J≤30)". Section 6 cites a warning against GEE with fewer than 30 clusters.
Neither passage concerns the reference distribution.

The package keeps its normal reference. A switch to $t$ would move every clustered interval and the
registered clustered study. That study runs 200 equal clusters, so it keeps the `influence_curve`
status. F22 holds the reopen route: a $t$-reference claim with a registered study at few clusters.

`TMLE._refuse_cross_fitted_missing_off_contract` holds the pre-fit refusal. It runs in
`TMLE._resolve_estimands_for_data`, after the arm-indexed contract and before the outcome-scale
rule. So it runs before fold generation, before `_preflight_training_support`, and before any
learner. `fit()` reaches it before the shared nuisances on the intermediate path. It applies to
ordinary TMLE only. `DRTMLE` and `CTMLE` refuse these compositions before any learner already.
A requested `par` or `paf` keeps its F20 sentence, because the in-sample remedy would be false for
them.

The review corrected the first clause of the refusal (commit 45f072b). It now says that both audited
contracts apply to a discrete treatment only. On a continuous dose the natural course is `Shift(0.0,
cap=None)`, a shift target that this refusal meets.

After the refusal, no admitted composition fits repeated draws with missing outcomes. The refusal
therefore deleted `TestTheMnarTiltFollowsTheDraws` and its helper `_binary_cde_with_missingness`
from `tests/unit/test_repeated_crossfit.py`, as F21 asked. The respondent check that the fixture
also carried moved to an in-sample greedy C-TMLE with selection folds
(`TestARespondentlessComplementIsRefusedOnEveryScale` in `tests/unit/test_fold_policy_rules.py`).

The refusal also closes a bypass. A `Static` regime or a saturated MSM, cross-fitted with `delta=`,
reproduced the arm-indexed fit bit for bit. The table gives a configuration that reproduces the
equality, with the refusal patched out. The value that the plan recorded here came from a
configuration that the plan did not record, and no later run reproduced it.

| configuration | reading |
| --- | --- |
| `make_missing_outcome(n=500, seed=4)`, cross-fitted at the default folds, `LinearRegression` for the outcome, `LogisticRegression` for the treatment and the response, `q_bounds=(-5, 8)`, `random_state=0` | `ey_regime[always 1]` 2.225003 and `ey1` 2.225003. `msm[a]` 1.234562 and `ate` 1.234562 |

Those fits escaped the refusals of the arm-indexed contract: `repeats=2`, fold targeting,
`cv_evaluation=True`, the linear fluctuation, and `id=` each fitted and reported an interval. Each
of them now refuses.

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
| 3 | `undeclared_function_plugin` | the saved curve treats a restored function as fixed, and no code can check that it was fixed |
| 4 | `estimated_weight_plugin` | the same premise as order 1 |
| 5 | `cross_fitted_longitudinal_plugin` | no read source covers the cluster-robust variance of the sequential recursion under grouped folds |
| 6 | `unequal_cluster_plugin` | the cross-fitted interval lacks validation at unequal cluster sizes or masses |
| 7 | `few_cluster_plugin` | no read source supports the reference distribution |

RM20 shipped orders 1, 2, 4, 6 and 7. [RM29](#rm29-saved-cross-fitted-clustered-longitudinal-results)
added order 5, and [RM28](#rm28-declared-densities-of-user-written-interventions) added order 3.
`PRECEDENCE` in `tests/unit/test_inference_status_registry.py` pins `NON_INFERENTIAL` to this
order.

Among the five RM20 statuses, only three pairs can meet. On DR-TMLE, the estimated-weight status
meets each clustered status. On
TMLE and DR-TMLE, the two clustered statuses meet each other. C-TMLE refuses `id=` at every
setting.

On a restored result, order 3 can meet each clustered status, orders 5 to 7, and it takes
precedence. `test_the_undeclared_status_precedes_the_cluster_status`, in
`tests/unit/test_rule_and_intervention_declarations.py` and
`tests/unit/test_regimen_rule_declarations.py`, restores a real few-cluster fit and reads order 3.
Order 3 cannot meet order 4, because `DRTMLE` refuses `interventions=` and `msm=`.

Each status reads the estimator configuration and the prepared data alone. No status
reads a fitted quantity, so the applicable status can be determined from the prepared data and
configuration alone. The estimator stamps the status after nuisance fitting.

The plan found two surfaces that published an inferential number under any non-inferential status.
The table gives each one and its correction.

| surface | what it published | correction that shipped |
| --- | --- | --- |
| `CVTargeting` fold reports | `cv_std_err` and `pooled_std_err`, from `pooled` and `canonical` estimates that no stamp reached | `_retarget_detailed` stamps the fit and both fold-level reports through `stamp_inference()`. `CVTargeting.inference` derives from the reports, `std_error` refuses, and the columns take their names through `spread_name` |
| the omitted-variable bound, `SensitivityBounds` | one-sided limits `ci_lower` and `ci_upper`, and the confidence-limit robustness value. A plain clustered TMLE fit, forced to a status, read "confidence-limit value 0.04412" | `SensitivityBounds` carries the status. `to_dict`, `summary`, `robustness_value()`, and the robustness row of the assessment rename each limit through `spread_name` |

RM12 did not meet the second surface. RM11 refuses the omitted-variable bound on C-TMLE and
DR-TMLE fits, so no selector fit reaches it.

Two neighboring surfaces keep their intervals without a registered study. This row records each
one and adds no row for it.

| surface | why it keeps its interval | evidence |
| --- | --- | --- |
| in-sample clustered point-treatment fit with 40 or more clusters, in the fit and in each reported stratum | Benitez et al. (2023), Section 3.2.1, support the cluster-sum aggregation for a row-weighted estimand | a recorded source. The registered clustered study is cross-fitted, so no registered study covers the in-sample fit |
| in-sample shift and incremental fits with `delta=` | exact-law instruments check the influence curve | `tests/unit/test_influence_gateaux_shift_cde.py` and `tests/unit/test_influence_gateaux_ipsi_mar.py`, which the [evidence manifest](technical-reference/evidence.md) lists for `ey_shift` and `ey_ipsi`. No registered study covers either fit |

The row planned five witnesses. Each had to fail when a component is wrong. The table gives the
state of each.

| witness | state |
| --- | --- |
| 1. for each surface that takes a status or a refusal, a test fits it and pins the status, the refusal, and the message | delivered in `tests/unit/test_estimated_weight_status.py`, `test_outcome_adaptive_status.py` (five `oat` cases: binary, three-arm, `delta=`, `ey1` only, and cross-fitted), `test_cluster_status.py`, and `test_cross_fitted_missing_off_contract.py`. `test_inference_status_reach.py` forces each status on one clustered `cv_evaluation=True` fit and walks every report |
| 2. a mutation that restores the `influence_curve` status on that surface makes its test fail | delivered. The next table gives each committed mutation |
| 3. for each surface that keeps its interval, the registered study and its nonzero witnesses | no surface registered a claim. The two neighbors above keep their intervals without a registered study |
| 4. a control shows that the neighboring supported surface keeps its interval | delivered. Each test fits the controls that the decision table names. The ordinary `TMLE` replaces the binary treatment-specific mean, which has no shipped instance |
| 5. a test pins the precedence order on a fit where several statuses apply | delivered as `TestThePrecedence` in `tests/unit/test_cluster_status.py`. A cross-fitted unequal fit with 39 clusters takes `unequal_cluster_plugin`. A guarded DR-TMLE fit with estimated weights on the same data takes `estimated_weight_plugin` |

Each mutation below is a committed test that patches the source and requires the named check to
fail.

| mutation | check that fails |
| --- | --- |
| the `CTMLE` hook before RM20 | the status check on all five `oat` cases |
| the ordinary hook on `DRTMLE` | the status check on the estimated-weight fit |
| a `DRTMLE` hook that ignores `guard` | the `guard=()` control. The mutant still passes the status fit |
| a cluster rule that ignores the sizes | the unequal-size check |
| a cluster rule that ignores `cross_fit` | the in-sample unequal-size control. The mutant still passes the status fit |
| `FEW_CLUSTER_THRESHOLD` patched to 0 | the check at 39 clusters, and the nuisance-note, assessment, and E-value checks on that fit |
| a cluster count that ignores the strata | the check on a stratum of 6 clusters inside 50 clusters |
| a size rule that reads the row counts only | the check on unequal weight mass at equal row counts |
| `WEIGHT_MASS_RTOL` patched to 0 | the tolerance check. The same ten weights, permuted inside each cluster, give masses that differ by 3.6e-15 |
| a coverage summary that labels a mix `influence_curve` | the check on a study whose cluster count straddles 40 |
| a `<=` comparison with the threshold | the control at 40 clusters |
| a stamp that skips the fold-level reports | the fold-report checks under `cv_evaluation=True` and fold targeting |
| an omitted-variable module that ignores the status | the name check of the bound |
| the off-contract refusal made a no-op | the witness for each of the five compositions. `NeverFit` records a fit |

No study was regenerated, and no fitted number moved. The stamp replaces the `inference` field only,
and `plugin_std_error`, `plugin_interval`, and the p-value body are the bodies that the refused
accessors call. The two probe intervals of the DR-TMLE and `oat` surfaces reproduce as
`plugin_interval` to four decimals. The study generators that read `ci`, `std_error`, or `pvalue`
now read them through the status-aware readers in `tests/studies/evidence/schema.py`. On one
`canonical_ctmle_oat` replicate the new rows are bit-identical to the old reads.

The registered clustered study runs 200 equal clusters, so it keeps
`influence_curve`. The only studies that fit the refused targets with `delta=`
(`canonical_cde_tmle` and `cde_tmle_properties`) fit in sample. The two `oat` method-evidence pages
now describe their cells as a diagnostic, as RM12 did for the selector pages.

The [technical reference](technical-reference/index.md) states the decision for each surface.
[Inference status](technical-reference/inference.md#inference-status) lists each status, and
the [scope page](technical-reference/scope-and-refusals.md), the
[collaborative reference](technical-reference/collaborative-tmle.md),
[DR-TMLE supported estimands](technical-reference/dr-tmle/supported-estimands.md), and
[CV-TMLE](technical-reference/cv-tmle.md) state each surface. Each status is reported at `ci`, in
`summary()`, and in the assessment. The off-contract refusal raises before any learner.
`tests/unit/test_cluster_status.py` reads the nuisance note from `diagnostics.run_all()` and from
`result.assess()`, and the E-value row, on a fit under each clustered status (commit 31fa6cb).

### RM21. E-value on a controlled-direct-effect fit

`_select_evalue` in `src/cleverly/sensitivity/evalue.py` returns the Gaussian-difference branch
before any check for an intermediate variable. The reported-ratio branches return before that
check too. The derived-ratio branch refuses the same fit in `_risk_ratio_refusal`
(`src/cleverly/sensitivity/_derived.py`). Its former reason claimed that no controlled direct
risk-ratio target was registered. [RM16](#rm16-summary-and-error-message-accuracy) records why
that reason misstated the fit. The fixed-baseline branch refuses it as well.

| fit | request | result |
| --- | --- | --- |
| `make_cde(n=2000, seed=3)`, Gaussian outcome, level $z = 0$ | `evalue(result, "ate")` | E-value 2.3563 from a ratio of 1.4955. The capability row reads available |
| the same fit, level $z = 1$ | `evalue(result, "ate")` | E-value 3.5149 from a ratio of 2.0489 |
| the same law with the outcome split at its median, level $z = 0$ | `ate` | refused: "derived risk ratios are unavailable for controlled direct effects" |
| the same binary fit | `rr` | E-value 3.4858 from the reported ratio 2.0348 |
| the same binary fit | `or` | E-value 2.6318 |

Both Gaussian levels divide by the observed outcome's sd(Y) = 1.907. That arithmetic does not
supply a source-backed confounding bound for the controlled-direct-effect target. Identification
also assumes no unmeasured confounding of the intermediate variable and the outcome. The ordinary
E-value conversion does not by itself describe both sources of
confounding for this fitted target. [F25](#f25-e-value-for-a-controlled-direct-effect) records
the mapping needed before a supported branch can open.

Apply these corrections:

1. Search for a source-backed bound and a mapping to the fitted controlled direct effect.
   A natural specialization can qualify. Record the locator, confounding model, and outcome scale.
2. Refuse each branch without a verified mapping, before any computation. Put the check in
   `_select_evalue` ahead of the branch selection, so that the capability row reads `unavailable`.
   The reason names the missing result.

The witnesses must fail when a component is wrong:

- a test on a Gaussian and a binary controlled-direct-effect fit pins the result of each branch;
- a mutation that moves the intermediate check back below the Gaussian branch makes that test
  fail;
- a control shows that a fit without an intermediate variable keeps each branch.

#### RM21 plan

The source search found no verified mapping that covers a branch of this fitted target.
[F25](#f25-e-value-for-a-controlled-direct-effect) records what each read source covers. So the
plan refuses every branch. The 2026-09-24 plan fixes the decisions in the table below. Line
numbers are at 44f44998.

| part | decision |
| --- | --- |
| predicate | a new `declares_intermediate(result)` in `src/cleverly/estimators/direct_effect.py` reads `data.has_intermediate or intermediate_value is not None`. A package fit sets both or neither, because `clever_covariate_inputs` raises `DataError` otherwise. `dataclasses.replace` can set one alone. The E-value, the omitted-variable bound, the simulated-confounding replay, and `_risk_ratio_refusal` call this one predicate |
| placement | `_select_evalue` runs the check after the longitudinal and continuous-treatment checks, and before `arm_parameter_keys`. Like those two checks, it is fit-wide. A check after the default estimand would leave a multi-arm controlled-direct-effect fit `deferred` |
| status and reason | `unavailable`, with the constant `_DIRECT_EFFECT_REFUSAL`. The reason names the missing result and ends "docs/roadmap.md F25 tracks this stop", as the simulated-confounding stops do |
| cost | on such a fit, an explicit `ey1` or an alias that the fit did not report reads `unavailable` with this reason. On a fit without an intermediate variable, the same request reads `not_applicable` or "was not requested". A continuous-dose fit with an intermediate variable stays `not_applicable`, because the continuous check runs first |
| dead branch | the `intermediate_value` check before the fixed-baseline branch (`evalue.py:395-396`) becomes unreachable. The plan deletes it |
| derived helper | `_risk_ratio_refusal` keeps its text, because a direct call of `_derived_risk_ratio` still needs it |

The plan measured each branch on `make_cde(n=400, seed=3)`, fitted in sample by
`fast_tmle(**IN_SAMPLE)` of `tests/conftest.py` with the covariates W1 to W3. The binary law splits
Y at its median. The table gives the result at 44f44998.

| fit | request | branch | E-value |
| --- | --- | --- | --- |
| Gaussian, $z = 0$ | default, `ate`, `att`, `atc` | `gaussian_difference` | 2.6947, 2.6947, 2.8060, 2.5774 |
| Gaussian, $z = 1$ | default, `ate`, `att`, `atc` | `gaussian_difference` | 3.3650, 3.3650, 3.3351, 3.4013 |
| binary with `rr`, `or` and `ate`, $z = 0$ | default and `rr`, then `or` | `reported_rr`, then `reported_or` | 3.7159, then 2.7570 |
| binary with `rr`, `or` and `ate`, $z = 1$ | `rr`, then `or` | `reported_rr`, then `reported_or` | 2.6570, then 3.9461 |
| binary with `rr`, `or` and `ate`, both levels | `ate` | refused by `_risk_ratio_refusal` | none |
| binary with `or` and `ate` only, $z = 0$ | default | `reported_or` | 2.7570 |
| the same fit, read as CV-evaluated | `ate` | refused with the CV text, through `evalue.py:395-396` | none |

The last two rows add a leak that the row did not name. A default request meets the odds-ratio
fallback at `evalue.py:393-394`, which returns before the intermediate check. Both Gaussian levels
divide by one observed value, sd(Y) = 1.942.

The same fits without `intermediate=` reach all five branches. The Gaussian fit reads
`gaussian_difference`. The binary fit with `rr` reads `reported_rr` and `reported_or`, and its
`ate` reads `derived_rr`. The binary fit without `rr` reads `derived_rr` by default. Read as
CV-evaluated, its default reads `reported_or` and its `ate` reads `fixed_baseline_ate`.

The new file `tests/unit/test_evalue_direct_effect_refusals.py` drives every witness from one
table of eleven requests. Each row names its law, its edit, its estimand, and the branch that the
fit without an intermediate variable takes.

| witness | test |
| --- | --- |
| 1. each branch refuses | each request, at both levels, reads an `unavailable` row with the RM21 reason. The free function and the facade raise it. A multi-arm fit, `ey1`, the assessment battery, and a `CausalStudy` fit refuse too |
| 2. the check moved below the Gaussian branch fails the test | the mutation tests rebuild `_select_evalue` from its own source with the check moved, and assert the exact set of requests that leak |
| 3. a fit without an intermediate variable keeps each branch | each request on that fit keeps its branch, and the rows reach all five branches |
| nonzero witness | each blocked branch computes a finite E-value above 1 on the same fit, so the refusal withholds a number |

The mutation tests commit six mutations. The plan predicts each set. The delivery measures it.

| mutation | predicted requests that leak |
| --- | --- |
| M0. the rebuilt function, unchanged | none. The control also keeps each branch |
| M1. the check below the Gaussian branch | the four Gaussian requests, the three ratio requests except `ate`, `ey1`, and the multi-arm default |
| M2. the check at its place before RM21, above the fixed-baseline branch | the M1 set, and the two odds-ratio defaults |
| M3. the check beside the status refusal | `ey1` and the multi-arm default |
| M4. the predicate in `evalue.py` returns `False` | every request |
| M5. the predicate in `_derived.py` returns `False` | none. The direct test of `_derived_risk_ratio` fails |

Five hand mutations then run one at a time, each on a copy with its sha256 recorded. H1 inlines
`intermediate_value is not None`. H2 reads `data.has_intermediate` alone. H3 reports
`not_applicable`. H4 restores `evalue.py` at 44f44998. H5 removes the rule from
`_risk_ratio_refusal`.

#### RM21 delivery

Both corrections shipped as commit f97076f planned them. The review found no verified mapping
that covers a branch of this fitted target, so every branch refuses. Each row ends with the commit
that shipped it.

| part | what shipped |
| --- | --- |
| predicate | `declares_intermediate(result)` in `src/cleverly/estimators/direct_effect.py`. `_refuse_intermediate` in `omitted_variable.py` and in `_simulated_confounding_request.py`, and `_risk_ratio_refusal` in `_derived.py`, call it. Commit a4378d2 |
| refusal | `_select_evalue` raises `_DIRECT_EFFECT_REFUSAL` with the `unavailable` status. The check runs after the longitudinal and continuous-treatment checks, and before `arm_parameter_keys`. The reason cites F25. Commit 581bdc0 |
| dead branch | the check at `evalue.py:395-396` at 44f44998 is deleted. Commit 581bdc0 |
| docstrings | the module docstring of `evalue.py` names the refusal, and `evalue()` gains a `Raises` section. Commit 581bdc0 |
| existing tests | the battery drops its `intermediate_value` case, which the new level-only test covers. Its cached-refusal test matches the full reason. `test_documentation_links.py` checks that the reason cites F25. Commit 581bdc0 |
| reference | the [E-value paths](technical-reference/validation-methods.md#e-value) gain the refusal row, the order of the fit-wide checks, and the table of read sources. The [scope page](technical-reference/scope-and-refusals.md) gains the refusal row. The [references](references.md#sensitivity-analysis) gain five entries. Commit 95e934e |
| missing result | [F25](#f25-e-value-for-a-controlled-direct-effect) and its future-grid row. Commit f97076f |

The [E-value paths](technical-reference/validation-methods.md#e-value) hold the table of read
sources. Each row gives the locator, the version read, and what the source does not cover. The
delivery checked the Biometrika volume and pages of Ding and VanderWeele (2016) at the publisher.
It read that paper as arXiv 1601.05155 only.

The table gives the outcome of the two surfaces that the plan found, and of one that the delivery
found.

| surface | outcome |
| --- | --- |
| a multi-arm controlled-direct-effect fit, found by the plan | its default row read `deferred`, and a check after the default estimand would keep it so. The check runs before the estimand, so the row reads `unavailable`. `test_a_multi_arm_fit_does_not_defer` pins the default and each `ate[...]` alias. Commit 581bdc0 |
| the text of `_risk_ratio_refusal`, found by the plan | its former text said that no controlled direct risk-ratio target was registered, while the binary fit reported `rr` at each level. Mutation M5 measured the limit: a direct call without the rule raises `DataError`, because `retarget` receives no level. [RM16](#rm16-summary-and-error-message-accuracy) records its later correction |
| the reason for `ey1` on a fit without an intermediate variable, found by the delivery | its former text named the arm axis, which the level has. RM16 records its later correction |

The row named three witnesses. The table gives the state of each. They are in
`tests/unit/test_evalue_direct_effect_refusals.py`, which held 69 tests at commit 581bdc0. The table `REQUESTS`
drives each witness.

| witness | state |
| --- | --- |
| 1. a test on a Gaussian and a binary fit pins the result of each branch | delivered in `TestAControlledDirectEffectRefusesEveryBranch`. Eleven requests at both levels read the row, the free function, and the facade. The same class covers `ey1`, the multi-arm fit, `assess()`, a `CausalStudy` fit, a level alone, and a column alone. `test_no_branch_computes` replaces each helper after the check with a function that raises, and every request still refuses |
| 2. the mutation that moves the check below the Gaussian branch fails that test | delivered as M1 below, with M0 to M5 |
| 3. a fit without an intermediate variable keeps each branch | delivered in `TestAFitWithoutAnIntermediateKeepsEachBranch`. The eleven requests reach all five branches, and `test_the_controls_reach_every_branch` checks that against `_Branch` |
| the nonzero witness | delivered in `TestTheRefusalBlocksANumber`. Each branch that reads a stored estimate computes a finite E-value above 1 on the same fit, at both levels. Both Gaussian levels and the fit without an intermediate variable divide by one sd(Y) |

Each committed mutation either compiles `_select_evalue` again from its own source with the check
moved, or replaces a predicate through `monkeypatch`. The measured sets equal the plan's
predictions.

| mutation | requests that leak, measured | the test that fails |
| --- | --- | --- |
| M0. the source compiled unchanged | none | none. `test_m0_the_rebuilt_function_refuses_and_keeps` is the control |
| M1. the check below the Gaussian branch | the four Gaussian requests, `ratio-default`, `ratio-rr`, `ratio-or`, `ey1`, and the multi-arm default. The leaked Gaussian `ate` at $z = 0$ reads 2.694664, the value before RM21 | `test_m1_the_check_below_the_gaussian_branch_leaks` |
| M2. the check at its place before RM21 | the M1 set, `odds-default`, and `odds-cv-default` | `test_m2_the_check_at_its_place_before_rm21_leaks` |
| M3. the check beside the status refusal | `ey1` and the multi-arm default | `test_m3_the_check_after_the_estimand_resolves_leaks` |
| M4. the predicate in `evalue.py` returns `False` | all thirteen requests | `test_m4_a_predicate_that_reads_no_fit_leaks_every_request` |
| M5. the predicate in `_derived.py` returns `False` | none. A direct call of `_derived_risk_ratio` raises `DataError` | `test_m5_the_derived_rule_removed_fails_its_own_witness` |

Six mutations ran by hand, one at a time. Each run copied the source file to a backup with its
sha256, applied one mutation, and ran the new file, the battery file,
`tests/unit/test_omitted_variable_refusals.py`, and `tests/unit/test_simulated_confounding.py`.
That is 552 tests. Each run then restored the file, the restored file matched its hash, and
`git status` was clean. The last two files failed no test in any run.

| mutation | file | failures |
| --- | --- | --- |
| H1. the check reads `result.intermediate_value is not None` | `src/cleverly/sensitivity/evalue.py` | 6 in the new file: the column-alone test, and M0 to M4 |
| H2. the predicate reads `data.has_intermediate` alone | `src/cleverly/estimators/direct_effect.py` | 2 in the new file, 1 in the battery file |
| H3. the check reports `not_applicable` | `src/cleverly/sensitivity/evalue.py` | 36 in the new file, 1 in the battery file |
| H4. the whole file at 44f44998 | `src/cleverly/sensitivity/evalue.py` | the new file and the battery file fail to import `_DIRECT_EFFECT_REFUSAL` |
| H4b. the file at 44f44998, with the constant and the import added | `src/cleverly/sensitivity/evalue.py` | 36 in the new file, 1 in the battery file |
| H5. `_risk_ratio_refusal` loses the rule | `src/cleverly/sensitivity/_derived.py` | 2 in the new file: the helper witness, and M2, where the two derived requests then raise `DataError` |

The delivery departed from the plan in seven places. The table gives each one.

| deviation | reason |
| --- | --- |
| the row for an explicit estimand comes from `_capability_for_arguments` | that is the row that `assess()` builds for a request. The plan named `_evalue_row`, which it calls |
| M1 compares the leaked value with the blocked branch, not with 2.6947 | the equality measures the same claim, and it does not pin learner output across platforms |
| M5 asserts `DataError` | the measured result. Without the rule, `_derived_risk_ratio` reaches `retarget`, which receives no level |
| `test_no_branch_computes` replaces six helpers and runs all eleven requests | the plan named three helpers and two requests. `arm_parameter_keys` and `_default_estimand` witness that the check runs before the estimand resolves |
| the nonzero witness takes its requests from `REQUESTS` | one table drives every witness. It covers the four branches that read a stored estimate |
| H4b | H4 as planned stops the two test files at import, so it measures no behavior |
| the literature table is in the E-value paths only | one table serves the reader and this record |

This delivery regenerated no study. The refusal sits in the selection, before any computation. No
registered study, notebook, or executed documentation example calls the E-value on a fit with
`intermediate=`. The shared predicate changes no package fit, because such a fit sets both halves
or neither. No container or R-runner file changed, so `tests/canonical/provenance-revisions.md`
needs no row.

The full fast suite gave 11556 passed and 87 skipped at 44f44998, and 11627 passed and 87
skipped at commit 63d8862.

#### RM21 review fixes

Two reviews of this delivery found one defect in the source, C1, and three in the tests, C2 to C4.
They also found six defects in the documentation, D1 to D6. The table gives each one, with the
commit that fixed it.

| finding | fix |
| --- | --- |
| C1. a combined report saved before RM21 still publishes its E-value | `_run_all` caches the report under `sensitivity.run_all`, and `_cached` serves a hit without a new gate. A controlled-direct-effect result that ran `run_all()` or `assess()` before RM21 kept a completed E-value row after `load`, and its capability row read `unavailable`. The generation of that key moves from 2 to 3, so the older entry is a cache miss. `test_a_battery_saved_before_rm21_does_not_publish_its_e_value` writes the stale report at generation 2, and a control reads it back at that generation. With the bump reverted, that test fails and the rest of its file and of `tests/unit/test_assessment_contract.py` passes. Commit 9c9f0ba |
| C1, the sibling check | no other cache key serves an E-value. `validate` and `diagnostics.run_all` hold diagnostics only. The single `sensitivity.evalue` entry is safe, because `_invoke` selects before it reads the cache. `sensitivity.derived_risk_ratio` held nothing on such a fit, because `_risk_ratio_refusal` refused it before RM21. Commit 9c9f0ba |
| C2. three copies of the probe law and of the edits that only `dataclasses.replace` makes | `tests/unit/_direct_effect_support.py` holds them once. The RM21 file, the intermediate fit of `tests/unit/test_omitted_variable_refusals.py`, and both intermediate edits of `tests/unit/test_simulated_confounding.py` use it. The four affected files passed 601 tests before and after. Commit 669663e |
| C3. the facade message matched by containment | `assert_refused` compares it exactly. A default request raises `sensitivity 'evalue' is unavailable:` before the reason, and an explicit estimand raises the reason alone. Commit 669663e |
| C4. `cv_evaluated` did not say that it refits nothing | its docstring says that the two requests on it witness the routing only. Commit 669663e |
| D1. the Smith and VanderWeele (2019) row called an author manuscript the published article | the row links the DOI of the published article. Its page locators are published page numbers. Commit 7a2ff41 |
| D2. the surface table of the delivery miscounted its sources | the sentence names two surfaces from the plan and one from the delivery. Commit 7a2ff41 |
| D3. two RM16 paragraphs each claimed "the last two rows" | each paragraph names its rows. Commit 7a2ff41 |
| D4. the comment on `_DIRECT_EFFECT_REFUSAL` sent the reader to RM21 for the sources | it names the source table in the E-value paths and F25. Commit 9c9f0ba |
| D5. four texts said that every request reads `unavailable` | a continuous-treatment fit with an intermediate variable reads `not_applicable`, because that check runs first. The E-value row, the scope row, and the F25 grid row now name a discrete treatment, and the E-value paths state the order. `test_a_continuous_dose_with_an_intermediate_is_not_applicable` pins the status on a shift fit. The reason text stays, because it says that the package refuses every request, and `not_applicable` is a refusal too. Commits 9c9f0ba and 7a2ff41 |
| D6. six smaller corrections | F25 names the *Biometrika* paper. The E-value paths give the title of each source that the search did not read, and say in the active voice which version of Ding and VanderWeele (2016) was read. The RM21 problem text points at the RM16 row, the suite line is wrapped, and the comment before the status refusal names the checks it follows. Commits 9c9f0ba and 7a2ff41 |

At commit 7a2ff41 the RM21 file held 71 tests. The full fast suite gave 11631 passed and 87
skipped. The four new tests are the two RM21 tests above, and the two source scans of
`tests/unit/test_documentation_links.py` on the new support file.

The PR 229 review corrected two further claims. The refusal now states the package's missing
implemented bound for its fitted controlled direct effect and confounding model. It no longer
infers that no such bound can exist from the two identification assumptions. The source table
distinguishes a natural effect, a marker contrast within one vaccination arm, and the fitted
population controlled direct effect at fixed $z$. The E-value summary now names the
equal-strength threshold and the tradeoff between unequal strengths. For $RR=2$, strengths
$3$ and $4$ give bounding factor $2$, although the equal-strength E-value is $3.414$.
`test_sensitivity_units.py` checks this arithmetic and the summary text.

### RM22. Standard error of the omitted-variable bound

`_elements_for` in `src/cleverly/sensitivity/omitted_variable.py` builds `psi_nu2` as the centred
element $\nu^2_i - \hat\nu^2$. For the ATT and the ATC, the representer and $m(\alpha)$ both
divide by the share $p = P(A = c)$ of the conditioning arm. The estimate is therefore a sample mean
divided by $\hat p^2$. Its influence curve has the term $-2 \nu^2 (1\{A = c\} - p) / p$. The code
omits the term, as does DoubleML's `DoubleMLIRM`. The function `_bound_std_error` adds the bias
curve to the curve of the estimate, so the standard error of the bound omits the term as well.

The 2026-09-22 probe fitted each law in sample with linear learners. The large strength is
$c_Y = 0.5$, $c_D = 0.3$, and $\rho = 1$. A ratio is the mean reported standard error over the
sampling standard deviation of the bound, for the lower bound and then the upper bound.

| law | $\nu^2$ estimator | fits | without the term | with the term |
| --- | --- | ---: | --- | --- |
| `att_result` in `tests/unit/test_assessment_contract.py`: `make_linear_ate(n=350, seed=11)`, ATT | doubly robust | 1 | standard deviation of `psi_nu2` 9.387, and a lower-bound standard error of 0.1325 at the large strength | 3.321, and 0.1206 |
| `make_linear_ate(n=1000)`, ATT | doubly robust | 500 | 1.118, 1.132 | 1.021, 1.033 |
| the same law, ATC | doubly robust | 500 | 1.098, 1.111 | 1.003, 1.015 |
| the same law, ATT | plug-in | 500 | 1.015, 1.026 | 1.113, 1.125 |
| the same law, ATC | plug-in | 500 | 0.992, 1.010 | 1.088, 1.108 |
| residual standard deviation 0.2 in the treated arm and 3 in the untreated arm, $n = 1000$, ATT | doubly robust | 500 | 0.971, 0.977 | 0.959, 0.966 |

At the default strength $c_Y = c_D = 0.03$, no ratio moves by more than 0.001.

A calculation in the probe explains the direction for the doubly robust estimator of a binary
ATT. Let $B = E[g^2 / (1 - g)]$. At the truth, the term lowers the variance of the $\nu^2$ curve by
$4 \nu^2 B / p^3$. An arm-dependent residual variance adds a cross term with $\sigma^2$. Jensen's
inequality, $B \ge p^2 / (1 - p)$, keeps the net change in the variance of the bound's curve at or
below zero. The omission therefore widens the doubly robust limits of a binary ATT.

The calculation is not a published result, and it does not cover more than two arms. The plug-in
estimator has no such result. Its $\nu^2$ is first-order sensitive to the fitted mechanism, and its
curve carries no term for that. On the linear law, the plug-in limits without the share term read
close to the sampling spread, and the share term alone overshoots. No derivation covers either
form. The `point-treatment-tmle` and `cross-fitting` tutorials request
`nu2_estimator="plugin"` and print its confidence-limit robustness value and one-sided limits.

The locators are not checked against the published article. `doi.org/10.1162/REST.a.1705`
redirects to `direct.mit.edu`, which returned HTTP 403 on 2026-09-22. The package cites Theorem 2,
Theorem 5(2), and Example 2 of the online appendix. The arXiv version 2112.13398v6, dated
2026-09-21, carries the journal reference. No record compares it with the published text.

Apply these corrections:

1. Add the term to `psi_nu2` for each parameter that conditions on an arm, under the doubly robust
   estimator. Derive the multi-arm form from the same share.
2. Decide what the plug-in confidence limits and `rva` claim. Derive the plug-in curve with its
   mechanism term, report the limits under a diagnostic name, or refuse them. Update the two
   tutorials to match the decision.
3. Read the published article and its online appendix. Correct each locator in the source, the
   refusal messages, and the reference pages.

The witnesses must fail when a component is wrong:

- on an exact finite-support law, the reported curve of $\hat\nu^2$ equals its Gateaux derivative,
  including the derivative through $p$. The term has mean zero, so a check of the mean cannot see
  it. The test compares the curves row by row, and a mutation that drops the term fails it;
- a repeated-sampling control, with its thresholds fixed before its run, reads the
  standard-error ratio of the doubly robust limits of the ATT;
- a test pins the ATE and counterfactual-mean curves, which the correction must not move.

#### RM22 plan

The 2026-09-24 plan fixes the decisions in the table below. Line numbers are at 29e25599. OV is
`src/cleverly/sensitivity/omitted_variable.py`.

| part | decision |
| --- | --- |
| share term | the doubly robust curve of $\nu^2$ gains $-2 \nu^2 w (1\{A = c\} - p) / p$ for each parameter whose `conditions_on` is not `None`. That is the ATT and the ATC, at two arms and at $K$ arms, weighted or not. Clusters need no change, because `influence_variance` sums the rows of a cluster. One private helper, `_conditioning_share_influence` in OV, computes the term |
| point $\nu^2$ | unchanged. The package divides by the full-sample weighted share, so $E_n^w[2 \ell - 1] = 1$ exactly and the division of `dml.sensemakr` is the identity here |
| ATE and means | unchanged. A new exact-law witness pins their curves |
| plug-in limits | refused. Under `nu2_estimator="plugin"` the accessors `ci_lower`, `ci_upper`, `robustness_value_ci`, the three `plugin_interval_*` accessors, and `rva` raise `CapabilityError`. `lower`, `upper`, `max_bias`, `rv`, `benchmark()` and `contour()` stay. `SensitivityElements.psi_nu2` and `psi_max_bias` read `None` under the plug-in. A bound saved before RM22 reads `nu2_estimator="unrecorded"` and refuses its limits too |
| missing result | [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) and its future-grid row |
| locators | keep Theorem 2 for the bias. Tie the bounds to Equation (14) in Section 4. Cite Lemma 3 and Theorem 4 for the score and the limits, and Theorem 5(2) only for $m$ and the representer. Cite Online Appendix A, "Statistical Inference", Equation (15), for the share term of $\theta_s$. State that the paper gives no share term for $\nu^2$ |
| cached reports | the generation of `sensitivity.run_all` moves from 3 to 4. `sensitivity.omitted_confounding`, `sensitivity.robustness_value` and `sensitivity.elements` gain generation 2 |
| studies | no existing study imports `cleverly.sensitivity`, so none is regenerated. One new study is declared below |

The derivation takes four steps. The ATT estimate is $\hat F / \hat p^2$, with
$\hat F = E_n^w[T(O; \hat g)]$ and $\hat p = E_n^w[1\{A = c\}]$. At a fixed $p$, the Riesz identity
makes $\hat F$ orthogonal in $g$, so its curve is the score of Lemma 3. The curve of $\hat p$ is
$w (1\{A = c\} - p)$, and the derivative of $F / p^2$ in $p$ is $-2 \nu^2 / p$. The ATC and each
contrast at $K$ arms take the same steps with their own conditioning arm.

The plan probes are read-only. The first probe fits the two laws of `tests/discrete_law.py` and
`tests/discrete_law_multi.py` with their oracle nuisances. It compares the curve of $\hat\nu^2$ with
the complex-step Gateaux derivative of $\nu^2$, written from the cell probabilities. The table gives
the largest error over the support points.

| law | parameters | without the term | with the term |
| --- | --- | ---: | ---: |
| two arms, unweighted | `att`, `atc` | 12.19, 9.541 | 1.2e-14, 1.2e-14 |
| two arms, weights $1 + 0.5a + 0.8y$ | `att`, `atc` | 11.9, 13.28 | 1.2e-14, 2.8e-14 |
| three arms, unweighted | the four ATT and ATC contrasts | 22.44 to 42.29 | at most 2.5e-14 |
| three arms, weights of $W$ | the four ATT and ATC contrasts | 56.78 to 84.75 | at most 6.0e-14 |
| both laws, both weightings | `ey1`, `ey0`, `ate`, and each `ate[...]` | at most 1.4e-14 | unchanged |

The plug-in curve $\hat\alpha^2 - \hat\nu^2$ differs from the same derivative by 15.1 to 45.9 on
the two-arm law, and by 21.3 for the ATE. The share term does not repair it.

The second probe applies the term to six fits. The first row is the `att_result` row of the
contract.

| fit | parameter | sd of `psi_nu2` | lower-bound standard error at the large strength |
| --- | --- | --- | --- |
| `make_linear_ate(n=350, seed=11)` | `att` | 9.3871 to 3.3206 | 0.1325 to 0.1206 |
| `make_linear_ate(n=1000, seed=1)` | `att`, `atc` | 9.0683 to 2.9203, 9.0106 to 2.7020 | 0.0750 to 0.0687, 0.0745 to 0.0678 |
| the same fit | `ate`, `ey[1]`, `ey[0]` | unchanged | unchanged |
| `make_clustered(n=400, cluster_size=10, seed=7)` | `att`, `atc` | 14.0813 to 9.2066, 13.1080 to 7.9655 | 0.5339 to 0.5040, 0.5593 to 0.5048 |

The third probe writes the whole lower bound $\theta - s \sqrt{\sigma^2 \nu^2}$ from the cell
probabilities. Its Gateaux derivative differs from the reported curve by 0.6202 for the ATT and
0.4766 for the ATC, and by at most 8.9e-16 after the fix. The probe also clusters the exact
sample, with the cluster of a row equal to its index modulo 50. The reported lower-bound standard
error of the ATT is 0.02129 without the term and 0.02303 with it. So on that sample the omission
narrows the ATT limit, and the contract's statement that the omission widens the limits does not
hold there.

##### Witnesses

A new file `tests/unit/test_omitted_variable_standard_error.py` holds the exact-law witnesses. Its
tolerance is `atol=1e-11`. Each comparator is a Gateaux derivative of a functional in
`tests/discrete_law.py` or `tests/discrete_law_multi.py`, which import no package code.

| witness | assertion |
| --- | --- |
| conditional curve | twelve ATT and ATC cases, at two and three arms, weighted and unweighted, equal the Gateaux derivative row by row |
| unconditional curve | the ATE, the means and each `ate[...]` contrast equal the Gateaux derivative row by row, and the helper is not called for them |
| lower-bound curve | the curve of the lower bound equals the derivative of the whole bound for the ATE, the ATT and the ATC |
| clustered standard error | the reported lower-bound standard error equals the cluster sum of the exact curve |
| plug-in refusal | each refused accessor raises with the F26 reason. The same fit under the doubly robust estimator reports finite limits |

The file commits these mutations. The predicted deviation is the smallest one that the probe
measured.

| mutation | predicted deviation | predicted failures |
| --- | --- | --- |
| M0. the helper, wrapped and unchanged | none | none |
| M1. the helper returns zeros | 9.54 | every conditional case, the lower-bound curve of the ATT and the ATC, and the clustered standard error |
| M2. the sign of the term flips | 19.1 | every conditional case |
| M3. an unweighted share | 3.98 | the six weighted cases only |
| M4. the weights dropped from the term | 3.64 | the six weighted cases only |
| M5. the ATT conditions on the reference | 37.9 at three arms | the ATT cases |
| M6. the term applied to every parameter | 3.70 | the unconditional cases |

The delivery then runs eight hand mutations of the source, one at a time. H1 deletes the helper
call, H2 flips its sign, H3 uses an unweighted share, H4 drops the weights, and H5 applies the
term to every parameter. H6 removes the plug-in guard, H7 reverts the cache generations, and H8
restores OV at 29e25599 with the new constants added.

##### Repeated-sampling control

A registered study reads the standard-error ratio. The declaration below is fixed before any run.

| field | value |
| --- | --- |
| name, slug | `omitted-variable bound standard error`, `omitted-variable-bound-se` |
| law | `make_linear_ate(n=1000)`. The four covariates are standard normal, $g = \operatorname{expit}(0.3 W_1 - 0.2 W_2 + 0.1 W_3)$, the noise is standard normal, and the effect is a constant 1.5 |
| fit | in-sample `TMLE` with `LinearRegression()` and an unpenalized main-effects `LogisticRegression`, `simultaneous=False`, and the estimands `ate`, `att` and `atc`. Both models are correctly specified. The estimator of $\nu^2$ is the default |
| strength | $c_Y = 0.5$, $c_D = 0.3$, $\rho = 1$ |
| truth | $\nu^2 = 4 e^{0.07}$ for the ATT and the ATC, $\nu^2 = 2 + 2 e^{0.07}$ for the ATE, $\sigma^2 = 1$, and $s = \sqrt{0.15 / 0.7}$. The six bounds are 0.541202 and 2.458798 for the ATT and the ATC, and 0.557547 and 2.442453 for the ATE |
| primary estimands | `att_lower`, `att_upper`, `atc_lower`, `atc_upper`, `ate_lower`, `ate_upper`, in the scenario `linear` |
| primary row | the estimate is the bound. The standard error is `(lower - ci_lower) / z_0.95` for a lower bound and `(ci_upper - upper) / z_0.95` for an upper bound. The interval is the estimate plus or minus 1.959964 standard errors |
| property cells | `interval_calibration` only. The six positive cells `<bound>__correctly_specified`, and two controls, `att_lower__inflated_se_control` and `att_upper__inflated_se_control` |
| control | the same fits. The standard error comes from the curve with the share term removed, which is the curve before RM22 |
| replicates, n | 10,000 primary and 10,000 per property cell, at $n = 1000$. The property cells share their draws |
| seeds | 20262201 for the samples, 20262202 for resampling |
| margins | `Margins()` unchanged |
| publication policy | `reporting` |
| reference | none. DoubleML omits the term, and `dml.sensemakr` is R code with a cross-fitted share and a different point estimate |

A positive cell passes when the 99% interval of its SE ratio lies inside (0.93, 1.07) and its
coverage interval inside (0.92, 0.98). A control passes when its SE-ratio interval lies wholly above
1.07. The 500-fit probe of the contract gave 1.033 for the ATT upper bound with the term and 1.118
for the ATT lower bound without it. At 10,000 replicates the half-width of the interval is about
0.02, so the plan predicts that every cell passes. The ATC has no control, because its ratios
without the term, 1.098 and 1.111, sit too close to 1.07 to predict a failure.

The control needs one change to the framework. `calibration_verdicts` reads the side of the band
that an SE control must leave from the kind of the cell. `shrunken_se_control` must fall below the
band and `inflated_se_control` above it, and one function holds that rule for both kinds.

The red-cell route is fixed now. If a verdict is red, it stays red at this budget and these
margins. The delivery then adds an ask to the "What this row asks for" table of RM18, names it in
the ledger, and states which witness failed. It does not run the study again at a larger budget,
and it moves no margin.

#### RM22 delivery

The three corrections shipped as commit e251457 planned them. Each row ends with the commit that
shipped it.

| part | what shipped |
| --- | --- |
| shared oracles | `tests/discrete_law.py` gains `contamination_eif`, `tilt_on`, `residual_variance`, `riesz_second_moment` and the shared `WEIGHT_FUNCTIONS`. `tests/discrete_law_multi.py` gains the two functionals. `tests/unit/_exact_sensitivity_support.py` holds the oracle fits that four test modules built by hand. The seven affected files ran 497 tests before and after. Commit c4ff6d3 |
| share term | `_conditioning_share_influence` in `src/cleverly/sensitivity/omitted_variable.py` computes $-2 \nu^2 w (1\{A = c\} - p) / p$, and `_elements_for` adds it under the doubly robust estimator for every parameter that conditions on an arm. The point $\nu^2$ does not move. The module docstrings carry the corrected locators. Commit 1581f25 |
| plug-in limits | `SensitivityBounds` records `nu2_estimator`. Under `"plugin"` the six limit accessors raise `CapabilityError` with the F26 reason, after the status guard. `to_dict()`, `summary()` and `robustness_value()` publish no limit, and the plug-in elements carry no curve. A bound pickled before RM22 reads `"unrecorded"` and refuses its limits. The assessment row names the stop. The two plug-in tutorials print the refusal. Commit bbf7d0d |
| cached reports | `sensitivity.run_all` moves from 3 to 4. `sensitivity.elements`, `sensitivity.omitted_confounding` and `sensitivity.robustness_value` gain generation 2. Commit a4a2d37 |
| reference | the section [standard error of the omitted-variable bound](technical-reference/validation-methods.md#standard-error-of-the-omitted-variable-bound), the scope row, the inference note, and the reference entries. Commit 9205aa8 |
| framework | `SE_CONTROL_SIDES` and `se_ratio_leaves_band` in `tests/studies/evidence/property_verdicts.py`. The shrunken and the inflated control share one rule. Commit ec15c85 |
| study | the declared modules, commit 8db70ff. The artifacts, the registration, the [study page](technical-reference/method-evidence/omitted-variable-bound-standard-error.md) and the grid row, commit cee8142 |
| missing result | [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) and its future-grid row. Commit e251457 |

The repository keeps no changelog and no release-notes file. The GitHub release notes are
drafted at release time. The plug-in refusal is a direct change with no deprecation step, so the
pull request states it for the release notes.

The contract named three witnesses. The table gives the state of each.

| witness | state |
| --- | --- |
| 1. the curve of $\hat\nu^2$ equals its Gateaux derivative, row by row, including the derivative through $p$ | delivered in `tests/unit/test_omitted_variable_standard_error.py`. Before the fix the twelve ATT and ATC curves missed by 9.541 to 84.75. After it each row agrees within 1e-11. The file also checks the curve of the whole lower bound, which missed by 0.6202 for the ATT and 0.4766 for the ATC, and the clustered lower-bound standard error, which missed by 0.00174 and 0.00159 |
| 2. a repeated-sampling control reads the standard-error ratio of the doubly robust ATT limits | delivered by the registered study. Every primary test and every property cell passed. The inflated-SE controls read 1.0987 (1.0791 to 1.1189) for the ATT lower bound and 1.0922 (1.0732 to 1.1129) for the upper bound, both above 1.07. The positive cells read 0.9970 to 1.0091. The run took 163 s on 16 cores |
| 3. a test pins the ATE and counterfactual-mean curves | delivered in the same file. Sixteen unconditional cases match the Gateaux derivative row by row, and a spy shows that the helper never runs for them |

The RM22 review fixes below correct this record on one point. The 200-replication
smoke run of the study was not discarded: it computed the property verdicts before commit 8db70ff.

The file commits seven mutations. The table compares each measured minimum with the plan.

| mutation | predicted | measured minimum | cases that fail |
| --- | ---: | ---: | --- |
| M0. the helper, wrapped and unchanged | none | at most 6.0e-14 | none |
| M1. the helper returns zeros | 9.54 | 9.541 | the twelve conditional cases, the two conditional bound curves, and the clustered standard error |
| M2. the sign flips | 19.1 | 19.08 | the twelve conditional cases |
| M3. an unweighted share | 3.98 | 3.977 | the six weighted cases. The six unweighted cases pass |
| M4. the weights dropped | 3.64 | 3.638 | the six weighted cases. The six unweighted cases pass |
| M5. the ATT conditions on the reference | 37.9 at three arms | 6.864 at two arms, 20.63 at three | the six ATT cases. The ATC cases pass |
| M6. the term on every parameter | 3.70 | 3.700 | the sixteen unconditional cases |

Eight hand mutations ran one at a time. Each run copied the source file to a backup with its
sha256, applied one mutation, and ran the new file, `tests/unit/test_omitted_variable_refusals.py`,
`tests/unit/test_sensitivity_multi_arm.py`, `tests/unit/test_inference_status_reach.py` and
`tests/unit/test_assessment_contract.py`. That is 541 tests. Each run then restored the file, the
restored file matched its hash, and `git status` was clean.

| mutation | file | failures |
| --- | --- | --- |
| H1. the helper call deleted | `omitted_variable.py` | 36: 35 in the new file, and the cache witness |
| H2. the sign of the term flipped | `omitted_variable.py` | 23: 22 in the new file, and the cache witness |
| H3. an unweighted share inside the helper | `omitted_variable.py` | 15 in the new file: the weighted cases, and M0, M5 and M6 |
| H4. the weights dropped inside the helper | `omitted_variable.py` | 15 in the new file, the same set as H3 |
| H5. `conditioning = parameter.arm` for every parameter | `omitted_variable.py` | 43: 39 in the new file, and 4 in the multi-arm file |
| H6. the estimator guard removed | `omitted_variable.py` | 6 in the refusal file |
| H7. the cache generations reverted | `_assessment_cache.py` | 1: the cache witness |
| H8. the module at 29e25599, with the two refusal constants added | `omitted_variable.py` | 53: 36 in the new file, 15 in the refusal file, 1 in the multi-arm file, and the cache witness |

The contract says that the omission widens the doubly robust limits. That holds on an iid fit: on
the two-arm exact law the lower-bound standard error of the ATT is 0.03882 without the term and
0.03544 with it. It does not hold in general. On the same sample in 50 clusters of 20 rows, the
standard error is 0.02129 without the term and 0.02303 with it, so the omission narrowed the
limit.

For the ATC on the same clustered sample the omission widened it, 0.02572 against 0.02413.
The tier reasoning that ranked RM22 as conservative rested on one exact law and the contract's
calculation for a binary ATT without clusters. That evidence does not cover a clustered fit, where
the shipped ATT limit could be too narrow.

The delivery departed from the plan in these places.

| departure | reason |
| --- | --- |
| the tutorial re-execution landed in the refusal commit, not in a commit of its own | the fast suite re-executes the tutorial code, so the refusal alone fails it. Each commit passes the suite |
| `tests/discrete_law_multi.py` and `tests/discrete_law_mar.py` do not import the generic derivative and tilt | registered studies import both modules, and a new import would open a gap in their recorded module lists. The support module applies the generic functions to the three-arm law |
| `WEIGHT_FUNCTIONS` moved into `tests/discrete_law.py`, and the verbatim copy in `tests/unit/test_remainder_drtmle.py` now reads it | one declaration for two modules. The plan named a support module and one copy |
| M5 measured 6.864 and 20.63, not the predicted 37.9 | the committed mutation patches `ArmParameter.conditions_on`, which also moves the ATT score $m(\alpha)$. The probe moved the term alone |
| the reference section is a level-3 heading, "Standard error of the omitted-variable bound" | the documentation build gives no anchor to a level-4 heading, so the planned `#### Standard error of the bound` could not be linked |
| the study's treatment model is an unpenalized `LogisticRegression(C=1e6)` | the committed declaration names an unpenalized model, so the GLM is the maximum-likelihood fit that "correctly specified" claims. `plan.md` named the default penalty |
| the study registration moved from the declaration commit to the artifacts commit | the register's tests read the artifacts, so a registered study without them fails the suite |
| a bound pickled before RM22 also omits its limit keys from `to_dict()` and names the reason in `summary()` | the plan defined its accessors only. A mapping that raised would break a saved report |
| the tutorials name the handler variable `limit_error` | `except ... as refusal` deletes a `refusal` name that an earlier cell of `cross-fitting.ipynb` defines |

This delivery regenerated no existing study. No registered study imports `cleverly.sensitivity`.
One new study was generated. No container or R-runner file changed, so
`tests/canonical/provenance-revisions.md` needs no row. `--check` reproduces every cell of
`point-treatment-tmle`, `cross-fitting`, `collaborative-tmle`, `msm-projections` and `dr-tmle`.
It reports three cells of `twins-causal-inference` outside its sensitivity step, from the Python
version of the stored run: solver residuals near 1e-18 and the float display of one table.

The full fast suite gave 11631 passed and 87 skipped at 29e25599, and 11778 passed and 96 skipped
at commit cee8142.

#### RM22 review fixes

Three reviews read the RM22 delivery: the code and its derivation, the study and its framework,
and the documents, notebooks and citations. None found a defect in a published number. All 14
study verdicts recompute from the artifacts. The reviews found five defects in the study and its
record, S1 to S5, four in the code and tests, C1 to C4, and eight in the documents, D1 to D8. A
verification of those fixes found four more, V1 to V4. The table gives each commit. "This record"
is the commit that adds a row, bfe6488 for the reviews and its successor for V1 to V4.

| commit | what it changed |
| --- | --- |
| 8a00b84, S1 | the shared driver passed `--replicates` to the primary phase only. Each property module fixes its own budget, so a smoke run ran the declared property study in full. `_arguments` now treats a run whose count differs from the declared primary budget, in either direction, as a smoke run, and it skips the property study. `test_a_run_at_another_budget_skips_the_property_study` pins the rule below, at and above the declared count. The record of the RM22 smoke run is in the paragraph below the table |
| this record, S2 | the working tree held CRLF copies of 18 files that this branch added or changed, although `.gitattributes` sets `eol=lf`. The study ran from two of them. Its manifest records `9f900e4b...` for `tests/studies/omitted_variable_bound_properties.py` and `335c61d3...` for `tests/studies/evidence/property_verdicts.py`. The committed LF blobs hash to `f5666660...` and `c8a60ed7...`. The content is identical and only the line endings differ. Every working file is now LF. The recorded hashes stay, because they are the bytes that ran. The table "Study module edits after a run" in `tests/canonical/provenance-revisions.md` holds one row for each of the two files. Each row shows that the recorded hash is the sha256 of the committed LF blob written with CRLF line endings |
| this record, S3 | the study page gains three limitations: the narrow pass of the `att_upper` control, the reason the ATC has no control, and the symmetric law. The grid row counts ten limits |
| f0de42f, S4 | `bound_standard_errors` rebuilds the bias curve with the term added back. It raises unless the rebuild equals the package's `psi_max_bias` and gives each reported standard error within 1e-10. For the ATE the control curve must equal the package's curve outright. `fit()` suppresses no warning, because the declared fit raises none. Both edits are result-neutral: with warnings raised as errors, the first five primary and five property replications refit to a maximum difference of 0.0. The manifest keeps the recorded hashes, and the table "Study module edits after a run" in `tests/canonical/provenance-revisions.md` records the edit with that evidence |
| f0de42f, S5 | the generic kind `inflated_se_control` carried text about $\nu^2$ and the share term. Its `CELLS` entry is now generic. `descriptions.ARM_CELLS` holds this study's text for `att_lower` and `att_upper`, so the published page reads the same |
| c54252f, C1 | M5 moves `conditions_on`, which also moves `_m_alpha` and the point $\nu^2$. M7 computes the term alone on the complement of the conditioning arm. It fails the twelve conditional cases by 21.38 to 120.2, and the sixteen unconditional cases pass. The M5 docstring says what M5 moves |
| ee5a77f, C2 | the comment on `_PLUGIN_LIMITS_REFUSAL` claimed one text, and `summary()` and the robustness row wrote their own. `limit_refusal_reason` now builds the one text for the six accessors, `summary()` and the robustness row |
| ee5a77f, C3 | a plug-in bound stored NaN limits, so two identical plug-in bounds compared unequal, and so did a pickle round trip. It now stores `None`. `test_identical_plugin_bounds_compare_equal` is the witness |
| none, C4 | declined. The private helpers beside `_conditioning_share_influence`, such as `_m_alpha` and `_riesz_representer`, carry prose docstrings with no Parameters section, and this one matches them |
| this record, D1 | the witness row said "Seven committed mutations fail it". M0 is a passing control. With M7 the row reads "Seven committed mutations fail it, and an unchanged control (M0) passes" |
| this record, D2 | the reference said "The term does not always widen the limits", which reverses the direction. A table now gives the lower-bound standard error for the ATT without clusters, the ATT in clusters, and the ATC in clusters. The omission widened the first and the third and narrowed the second. The delivery record now scopes the tier reasoning to its evidence |
| this record, D3 | the queue said that tier a held RM21 and RM22 only, and it dropped the rule that an anti-conservative number ranks above a conservative one. The tier row is back with its reason, and it reads "none. RM20, RM13, RM21 and RM22 held it" |
| this record, D4 | X9 said "Nobody has read the published main text". It now says "This project has not read" it |
| this record, D5 | the study page and the grid row say that `dml.sensemakr` divides the point estimate at `58ac44d`, not at `d5293ecb`. The references entry calls it "the implementation this package found with that term". The module docstring spells the DoubleML commit `b69f86ef`, as the other pages do |
| ee5a77f, D6 | the point-treatment tutorial's refusal paragraph now follows the review paragraph and opens "Reading `ci_lower` refuses". Its 26-word sentence is two sentences |
| ee5a77f, D7 | the plug-in refusal reads "no derivation in a source this package cites" and names `benchmark()` and `contour()`. The unrecorded refusal names `result.sensitivity.omitted_confounding()` too. Both tutorials print the plug-in text, so both were re-executed. `--check` reproduces every cell, and only the refusal line of each moved |
| this record, V1 | a verification found that the S2 and S4 rows above named no ledger row, and the S2 row said that none was needed. The table "Study module edits after a run" asks for a row for each result-neutral edit to a study module after its run. It now holds three rows for `omitted-variable-bound-se`, with the full recorded and current hashes |
| this record, V2 | the driver skips the property study when the count differs from the declared count in either direction. The comment, the S1 row and the test said "below". They now say "differs", and the test also runs the declared count and a count above it |
| this record, V3 | the "35 to 40%" failure rate of the `att_upper` control did not reproduce. The paragraph below now states the method and both assumptions: about 34% if the true ratio is the cell's 1.0922, and about 20% if it is the mean 1.0954 of the two controls |
| this record, V4 | five comments and docstrings in `omitted_variable.py`, `inference.md`, `scope-and-refusals.md` and the point-treatment tutorial kept the unqualified "No derivation gives". Each now reads "no derivation in a source this package cites". The tutorial edit is markdown, which the notebook stamp does not hash, so no notebook was re-executed |
| this record, D8 | Theorem 5(2) "writes $m$ and the representer", which keeps "score" for Lemma 3. The F26 grid row and the plan name `contour()`, the public method. The study page says "moves no ratio by more than 0.001". Two long queue lines are reflowed, and the sensitivity-outputs sentence joins the paragraph on the groups that left |

The RM22 smoke run showed the property verdicts before the study modules were committed. The
commit message of 8db70ff calls that run disposable and discarded. It was not discarded. The
smoke run discarded its 200 primary replications. It also ran the full property study at 10,000
replications, and it wrote the eight property verdicts at 11:44:58 on 2026-09-24.

The working tree was then at ec15c85, with the study modules not yet committed. Commit 8db70ff followed at
11:51:48, about seven minutes later.

The declaration still binds. The plan commit e251457, at 10:24:09, fixed the law, the budget, the
seeds, the margins and the red-cell route before either run. Nothing moved after it. The smoke and
published manifests record the same module hashes, and their configurations differ only in the
primary replication count. The two runs wrote byte-identical property artifacts: `properties.csv`
hashes to `f140e0c3...` and `property-replicates.csv.gz` to `585e6480...` in both.

The `att_upper` control passed narrowly. Its ratio is 1.0922, and its 99% interval runs from 1.0732
to 1.1129 against the band edge of 1.07. The failure rate at another seed was estimated as
follows.

| step | value |
| --- | --- |
| Monte Carlo standard deviation of the ratio | 0.0077, the interval width over $2 z_{0.995}$ |
| distance from the ratio to the lower edge | 0.0190, taken as fixed |
| failure | a new ratio at or below $1.07 + 0.0190 = 1.0890$, so the lower edge falls at or below 1.07 |
| failure rate, true ratio 1.0922, the cell's own estimate | about 34% |
| failure rate, true ratio 1.0954, the mean of the two controls | about 20% |

The plan sized the budget on the 500-fit ratio of 1.118. That estimate's error did not cover the
measured inflation of about 1.0954.

The verdict stands under the declared route. A future design would pair each control with its positive cell and read the ratio of their standard
errors.

At this record the new witness file holds 59 tests and commits eight mutations, M0 to M7. The driver test of V2 holds five cases. The
refusal file holds 115 tests. The hand-mutation counts of the delivery record were measured at
cee8142 and were not run again. The full fast suite gave 11784 passed and 96 skipped at commit
f0de42f, and 11786 passed and 96 skipped with the V1 to V4 fixes.

### RM23. Capability rows that read available and then refuse

The assessment declares each operation in a capability row before any call. The rows below read
`available`, and the call then refuses or raises. A 2026-09-22 probe fitted 16 kinds of fit with
linear learners. It compared each declared row with the status that
`assess(include_refits=True, include_retargets=True)` reports. Three kinds of fit disagree.

| fit | row | what the call does |
| --- | --- | --- |
| ordered `CTMLE` with an explicit `ordering=`, `make_instrument(n=500, seed=44)` | `refute` | `random_common_cause` adds the column `_noise_0`, and the refit raises `ValueError`: "ordering must cover every covariate; missing ['_noise_0']". `assess(include_refits=True)` raises the same error and returns no report. The same fit with the default `preorder` completes |
| shift fit with missing outcomes, `make_missing_outcome(n=400, seed=4)`, with the dose built as the arm plus standard normal noise | `missingness` | declines: "missingness_tilt is written for the arm-indexed estimands" (`src/cleverly/sensitivity/missingness.py:167`) |
| the same fit | `tipping_gamma` | declines with the same reason |
| incremental fit, `make_linear_ate(n=400, seed=2)` | `truncation_curve` | declines: "the propensity g is *inside* the estimand for an incremental intervention" (`src/cleverly/assessment.py:2313`) |

The other 13 kinds of fit agree with their rows. They are ordinary, binary, missing-outcome, shift,
controlled-direct-effect, multi-arm, outcome-adaptive C-TMLE, MSM, regime, weighted, stratified,
clustered, and DR-TMLE fits. The ordered fit loses the whole report, because a `ValueError` is not
the refusal type that `run_all` catches.

The 2026-09-23 review of RM25 found the same mismatch one level lower, in `replayability()`
(`src/cleverly/assessment.py:1146-1161`). For a point-treatment result that holds an estimator, the
function returns `retarget_cached_nuisances=True` and `refit_nuisances=True`. It reads no
declaration and no fold policy. A probe fitted each case in sample with linear learners on the
default law of `tests/discrete_law.py`, and restored it through `serialize.dumps` and `loads`. The
cross-fitted case used two folds.

| restored result | `replayability()` | `retarget()` | `refit()` |
| --- | --- | --- | --- |
| a `Stochastic` regime fit, restored without `density_kind` (RM25) | both slots `True` | `CapabilityError`: the undeclared density | `CapabilityError`: the undeclared density |
| an MSM fit with a callable weight, restored without `weights_kind` (RM13) | both slots `True` | `CapabilityError`: the undeclared weight | `CapabilityError`: the undeclared weight |
| a cross-fitted fit whose restored estimator carries `stratify_folds="treatment"` | both slots `True` | runs | `ValueError` from the fold-policy check |

Each declared control, restored with its declaration and its fold policy, runs both calls. In the
two declaration cases, the `refute` and `truncation_curve` rows read `available`.
`assess(include_refits=True, include_retargets=True)` still returns a report. Its
`truncation_curve` row then reads `unavailable` and quotes the refusal.

[RM28](#rm28-declared-densities-of-user-written-interventions) left the same mismatch on a
restored result whose rule is undeclared. A restored `Rule` result reads `available` in its
`truncation_curve` and `refute` rows. A restored `LTMLE` result reads `available` in its
`truncation_curve` row. Each call then raises `CapabilityError`. A 2026-09-24 probe on commit
faa0a5f measured these rows, and `assess(include_refits=True, include_retargets=True)` still
returned a report.

The later [RM28 review](#rm28-review-saved-longitudinal-msm-provenance-and-malformed-plans)
closes the declaration cases in these probes. Point-result `replayability()` now checks the
same function declarations as `retarget()` and `refit()`. An undeclared rule, regime density,
or MSM function makes both replay slots false and the dependent capability rows unavailable.
Longitudinal replayability checks saved rules and the evaluated MSM marker. Either refusal
makes `refit_nuisances` false and the truncation row unavailable. The original three fit kinds
in the first table and the restored cross-fitted fold-policy case remain in RM23.

Apply these corrections:

1. Resolve each of the three declined rows from the predicate that its call uses, as RM12 did for
   `tipping_gamma(use_ci=True)`. The later RM28 review resolves the restored-function rows.
2. Make the `random_common_cause` refit of an explicit ordering run, or declare the row
   `unavailable` with a reason. A refit that adds the noise column to the ordering changes the
   declared ordering, so the contract must state where the column goes.
3. Derive the two `replayability()` slots from the checks that `retarget()` and `refit()` run.
   The later RM28 review does this for function declarations. The restored cross-fitted fold
   policy in the table above still needs its own check.

The witnesses must fail when a component is wrong:

- a sweep test fits each kind of fit in a fixed list, and asserts that every row declared
  `available` runs without a refusal. The list includes the three kinds above.
  `TestNoAvailableRowDeclinesOnASelectorPath` is the precedent on selector fits;
- a mutation that restores an unconditional row makes the sweep fail on that kind of fit;
- a test that the ordered fit with an explicit ordering completes `assess(include_refits=True)`;
- a test that restores each of the three results in the second table, and asserts that
  `replayability()` agrees with what `retarget()` and `refit()` do;
- a test that restores the `Rule` and `LTMLE` results that RM28 records, and asserts that each
  replay-dependent row reads `unavailable` when its function declaration is missing.

#### RM23 plan

The 2026-09-24 plan fixes the decisions in the tables below. Its probes ran at 0cd452b4.

A capability row and its call read one predicate, as RM12 did for `tipping_gamma(use_ci=True)`.
The call raises the sentence that the predicate returns. The row quotes the same sentence.

| part | predicate | call | row |
| --- | --- | --- | --- |
| tilt | `fit_wide_tilt_refusal` in `sensitivity/missingness.py` | `missingness_tilt` and `tipping_gamma` raise from it first | both tilt rows take its reason for the whole fit |
| truncation | `truncation_refusal` in `sensitivity/positivity.py` | the module `truncation_curve` and the facade raise from it | the row resolves for each request |
| refute | `refute_refusal` in `validation/refute.py` | `refute()` raises from it before any refit | the row resolves for each request |
| replay | `TMLE._refit_configuration_refusal` | `refit()` raises from the same checks, as `CapabilityError` | `refit_nuisances` reads false |

The tilt table is ordered. Each rule assumes that its predecessors returned `None`.

| order | rule | refuses | status |
| --- | --- | --- | --- |
| 1 | `longitudinal` | a longitudinal result. The reason stays byte-identical, because two notebooks print it | `unavailable` |
| 2 | `natural_course` | a missing-outcome `NaturalCourseMean` fit | `unavailable` |
| 3 | `missing_outcome` | a fit with no missing outcome | `not_applicable` |
| 4 | `continuous` | a continuous dose, with the current sentence | `unavailable` |
| 5 | `incremental` | an incremental fit, with the current sentence | `unavailable` |
| 6 | `tiltable_parameters` | a fit that reports no arm-indexed mean or linear contrast | `unavailable` |

The response refusal of the omitted-variable bound points at the tilt only when this table returns
`None`. The default truncation axis is `not result.nuisance.fits_treatment`, which the module
uses now.

| order | truncation rule | refuses |
| --- | --- | --- |
| 1 | `observation_axis` | `mechanism=True` on a fit with no missingness and no intermediate mechanism |
| 2 | `incremental` | the treatment axis of an incremental fit |
| 3 | `natural_course` | the treatment axis of a fit that fits no treatment mechanism |

| order | refute rule | refuses |
| --- | --- | --- |
| 1 | `estimator` | a result with no estimator |
| 2 | `estimand` | an estimand that the fit did not report |
| 3 | `placebo_level` | `placebo` on `NaturalCourseMean` |
| 4 | `row_set_under_plan` | `subset` or `bootstrap_measurement_error` on a fit with `split_plan=` |

The truncation and refute rows resolve each request by one rule.

| request | row |
| --- | --- |
| the argument is omitted, and the default runs | unchanged |
| the argument is omitted, the default is refused, and another value runs | `deferred`, naming the argument, with the call's sentence |
| the argument is omitted, and no value runs | `unavailable`, with the sentence of the default |
| the argument has a refused value | `unavailable`, with the call's sentence |

`_CapabilityFacade` keeps the method and replay gates in `_gated_map`. `_request_gated` then
resolves one request. `capability()` resolves the bare request, and `_require` takes the request
of a direct call.

`TMLE.refit` asks `_configured_for_refit(data)` before it fits.

| estimator | what the refit runs |
| --- | --- |
| `TMLE` | the estimator |
| `CTMLE` with an explicit `ordering=` | the declared ordering, then each added covariate |
| `DRTMLE` with `evaluation=` | the estimator without its companion, when the companion lacks a refit covariate |

An explicit ordering ranks covariates by their prior relevance. A column drawn as independent
noise has none, so it goes last. `with_extra_covariate` appends it last. A five-draw probe left
the selection at `('W1',)`, and the mean moved by 0.0006 against a standard error of 0.098.

The companion enters no fit, fold, or score. `tests/unit/test_drtmle_companion.py` checks that
bit for bit. So a refit without it reports the same estimate.

An estimator that release 0.1.1 saved has no `split_plan` attribute. A class default of `None` on
`TMLE` restores it, because that release drew its folds from the data.

The refit slot reads the fold policy and four data contracts. These are the checks that
`_resolve_estimands_for_data` runs before a learner. A refused configuration keeps
`retarget_cached_nuisances` true. It reports the code `point_replay_refit_configuration`.

##### Probes

A 2026-09-24 probe filled each deferred argument and ran
`assess(include_refits=True, include_retargets=True)`. The table gives each fit that declines.

| fit | what the call does at 0cd452b4 |
| --- | --- |
| shift with missing outcomes | `missingness` and `tipping_gamma` decline |
| incremental, `make_linear_ate(n=400, seed=2)` | `truncation_curve` declines. The module call returns a flat curve at 2.96262 |
| incremental with missing outcomes | the three rows decline. `mechanism=True` runs |
| a regime, an MSM, or a ratio-only fit with missing outcomes | both tilt rows decline |
| a cross-fitted fit with a supplied `split_plan` | `refute` declines on `subset` |
| `NaturalCourseMean` with `estimand="ey_obs"` | `refute` declines on `placebo` |
| ordered `CTMLE` with an explicit ordering | `assess` raises `ValueError` |
| `DRTMLE` with `evaluation=` | `assess` raises `DataError` |

The other 13 kinds of fit decline no row. A second probe restored each result below through
`dumps` and `loads`.

| restored result | `replayability()` | `refit()` |
| --- | --- | --- |
| cross-fitted, `stratify_folds="treatment"` or `"treatment+outcome"` | both slots true | `ValueError` |
| `n_folds=1`, `repeats=0`, or `repeats=2` in sample | both slots true | `ValueError` |
| collaborative in sample, `stratify_folds="treatment"` | both slots true | `ValueError`. `assess` raises |
| the 0.1.1 cross-fitted artifact | both slots true | `AttributeError` on `split_plan` |
| cross-fitted continuous outcome, `q_bounds=None` | both slots true | `CapabilityError` |

##### Witnesses

| witness | assertion |
| --- | --- |
| sweep | 27 kinds of fit each answer every row that a request resolves as available. No row declines, and none stays deferred |
| nonzero | each kind runs the rows that it lists, and each live fit keeps `refit_nuisances` |
| tilt, truncation, refute | each module call raises the row's sentence exactly |
| ordering | the refit path ends with the declared ordering and then `_noise_0` |
| companion | refute gives the same draws with and without a companion |
| replay | each slot equals whether `retarget()` or `refit()` runs, for eleven restored results |
| cache | a report cached at the old generation is recomputed |

| mutation | the sweep then fails on |
| --- | --- |
| M0. every hook wrapped and unchanged | nothing |
| M1. the tilt rows ignore the predicate | five tilt kinds |
| M2. the truncation row ignores the predicate | two incremental kinds |
| M3. the refute row ignores the predicate | the split-plan and natural-course kinds |
| M4. `CTMLE` ignores added covariates | the ordered kind |
| M5. `DRTMLE` keeps its companion | the companion kind |
| M6. the refit slot ignores the preflight | three restored kinds |
| M7. no `split_plan` class default | the 0.1.1 kind |

`diagnostics.run_all` moves from generation 10 to 11, and `sensitivity.run_all` moves from 5 to 6.
No study calls these paths, so none is regenerated. `docs/examples/interventions.ipynb` is
executed again, because its incremental truncation row changes.

### RM24. Refusals after the nuisance fit

The [Definition of done](#definition-of-done) asks for a pre-fit test for every well-posed
composition that is still refused. The refusals below run after learner fits. Only the last one
raises `CapabilityError`. A 2026-09-22 probe counted the `fit` calls of spy learners in the first
three rows. The delivery of RM20 found the fourth row.

| request | where it raises | learner fits before the raise | exception |
| --- | --- | ---: | --- |
| `TMLE(incremental=...)` with `strata=`, `make_linear_ate(n=400, seed=2)` | the strata guard in `_retarget_detailed`, `src/cleverly/estimators/tmle.py:2736` | 2 | `NotImplementedError` |
| `DRTMLE` with `strata=`, the same law | the same guard, for the `mean` group | 8 | `NotImplementedError` |
| `TMLE(incremental=...)` with `intermediate=`, `make_cde(n=400, seed=3)` | `_check_incremental`, `tmle.py:2148`. `fit` fits the shared nuisances first | 3 | `ValueError` |
| `TMLE(estimands=["par"])` with `intermediate=` and `delta=`, on `binary_cde_frame()` from `tests/unit/test_cross_fitted_missing_off_contract.py`, with `LogisticRegression` in every learner role and `random_state=0` | the F20 refusal in `_fit_single`. On the intermediate path, `fit` calls `_prepare_shared` first, and that call fits the shared nuisances | 4 in sample. 20 cross-fitted at `n_folds=5` | `CapabilityError` |

The fourth row contradicts the comment in `TMLE.fit`, which says that "every unsupported
intermediate composition fails before any learner is fitted". The population-intervention
refusals run in `_fit_single`, and `_resolve_estimands_for_data` does not raise them.
`test_par_beside_an_intermediate_keeps_its_f20_refusal` pins the message and fits real learners.

A second finding concerns the order of two refusals. `DRTMLE(cross_fit=True)` with `delta=`
refuses with `NotImplementedError` in `_check_drtmle`, which runs at the start of the nuisance fit.
The fold preflight runs earlier, so a sample with no respondent in a training complement gets a
`DataError` instead. No learner is fitted in either case. The table gives the probes.

| probe | result |
| --- | --- |
| `respondents_in_one_fold()` from `tests/unit/test_fold_policy_rules.py`, `DRTMLE(cross_fit=True, n_folds=6, random_state=1146, q_bounds=(0, 1))`, at `guard=("Q", "g")` and at `guard=()` | `DataError` "cross-fitted DR-TMLE cannot fit its nuisances because repeat 0, fold 2's training complement contains no row with an observed outcome". The composition is refused at every sample, so the data error names a problem that no other sample would fix |
| `binary_missing_frame()` from `tests/unit/test_cross_fitted_missing_off_contract.py`, `DRTMLE(cross_fit=True, n_folds=5, random_state=0)` with `NeverFit` learners | `guard=()` raises `NotImplementedError` "the published missing-outcome DR-TMLE theorem uses Donsker conditions and does not establish its cross-validated extension". `guard=("Q", "g")` raises the randomized-trial `NotImplementedError` first. `NeverFit.calls` is 0 in both |

[X8](#x8-stratified-incremental-and-msm-targeting) holds the stratified targeting construction,
and [F6](#f6-mnar-and-incremental-intermediate-compositions) holds the incremental-intermediate
composition. This row changes only when each refusal reaches the caller, and its type.

Apply these corrections:

1. Raise each refusal as `CapabilityError` before any learner call. Decide it from the estimator
   configuration and the data declaration alone. A refusal of the composition comes before a
   preflight that reads the sample.
2. Keep each message, and its pointer to X8, F6, or F20.
3. Search the other `NotImplementedError` and `ValueError` refusals in `src/cleverly/estimators/`
   for the same order. List each refusal that this row moves.

The witnesses must fail when a component is wrong:

- for each request, a spy-learner test asserts that no learner call ran, and pins the message;
- a mutation that moves a guard back after the nuisance fit makes its test fail;
- a control shows that stratified arm, regime, and shift targets still fit.

### RM25. Declared stochastic regime densities

The [scope page](technical-reference/scope-and-refusals.md#wrong-by-construction) listed "a
`Stochastic` regime whose density came from the estimated mechanism" as refused
(`docs/technical-reference/scope-and-refusals.md:293` at a937a20). Nothing refused it. The
`Stochastic` docstring asked for a known density (`src/cleverly/interventions/base.py:274-295` at
a937a20), and no code checked that word.

A `density_fn` receives the covariate frame, and it can close over any estimate. For the
population-law target $\psi(P;q_\delta(g_P))$, the odds-tilt density depends on $P$ through its
treatment mechanism. The regime curve has no term for that derivative. A realized learned density
$\hat q$ instead defines a data-adaptive target $\psi(P;\hat q)$. Its fixed-density curve has no
population-mechanism derivative, but same-sample inference needs separate conditions that this API
does not check.

Van der Laan, Luedtke and Díaz (2014), Section 5, and Hubbard et al. (2016)
distinguish these targets. The [source audit](references.md#point-treatment-and-stochastic-interventions)
records both sources.

| probe | result |
| --- | --- |
| `make_linear_ate(n=400, seed=7)`, in sample, linear learners. `density_fn` returns a logistic mechanism fitted on the same frame, plus 0.1, clipped to [0, 1] | `ey_regime[fitted]` reported `ci` (2.5883, 2.9354) under the `influence_curve` status. No warning and no message named the density |

[RM13](#rm13-estimated-msm-projection-weights) added the same kind of declaration for an MSM
projection weight. The row asked for three corrections on that pattern:

1. Require a declaration that a `Stochastic` density is known. Refuse an undeclared density before
   any nuisance fit.
2. Refuse a density declared as estimated. The message distinguishes the population-law target's
   missing derivative from the unverified inference conditions of a realized learned-policy target.
3. Correct the scope-page row, so that it describes the declaration and both refusals.

All three shipped. One function, `cleverly.interventions.base.refuse_regime_densities`, runs four
checks in the order that the table lists them. The method `Stochastic.__post_init__` calls it. The
fit layer calls it again in `TMLE._resolve_estimands_for_data` and `TMLE._retarget_detailed`,
before any learner call. That second call catches a restored or modified regime, which can carry a
declaration that this version refuses. Each row of the table ends with the commit that shipped it.

| part | what shipped |
| --- | --- |
| declaration | a field `density_kind` on `Stochastic`. It takes `"known"`, `"estimated"`, or `None`, and it defaults to `None`. It is the last field, so `Stochastic(density_fn, name)` keeps its positional order. Commit 7ef1f57 |
| a density that is not callable | `DataError` when the regime is built. Before the change, `Stochastic(0.6, "coin")` built, and an in-sample fit on the default law of `tests/discrete_law.py` raised `TypeError` after two learner fits. Commit 7ef1f57 |
| an unknown value | a value other than `"known"`, `"estimated"`, or `None` raises `DataError`. Commit 7ef1f57. The review found that the check compared a value that is not a `str`. So `np.array(["known"])` built, a longer array raised `ValueError`, and `pandas.NA` raised `TypeError`, for RM13 and RM25 both. The shared check now refuses any value that is not `None` or a `str` with the same `DataError`. A `numpy.str_` is a `str` and passes |
| undeclared density | `density_kind=None` raises `CapabilityError`. The message asks for `density_kind="known"`. Commit 7ef1f57 |
| estimated density | `density_kind="estimated"` raises `CapabilityError`. The reviewed message now distinguishes the population-law derivative from the realized learned-policy target. It sends a population odds tilt to `TMLE(incremental=...)`, whose curve carries the mechanism term. Commit 7ef1f57 and this review |
| shared declaration | a private leaf module, `cleverly._declarations`, holds `FunctionDeclaration`. That class holds the three-state check and the texts of its refusals. The RM13 weight declaration became its first user. [RM27](#rm27-declared-msm-design-functions) later made the MSM design its third user (commit 268dc62). A snapshot of 246 RM13 cases recorded the same exception type and message before and after, byte for byte. `cleverly.msm.MSMWeightsKind` stays public. Commit 83c6d72 |
| fit-layer check | `TMLE._resolve_estimands_for_data` and `TMLE._retarget_detailed` call the function after the MSM weight check. So `fit`, `refit`, `CausalStudy.estimate`, `retarget`, and every sweep refuse a restored regime before any learner or density call. The check selects regimes with `isinstance`, so a subclass that skips `__post_init__` refuses at the fit. Commit 7ef1f57 |
| replay | `_freeze_regimes` (`src/cleverly/sensitivity/_simulated_confounding_fixed.py`) calls the function on the source regimes before `RegimeSet.evaluate` runs any density. A `_FrozenRegime` is not a `Stochastic`, so the refit admits it. Commit 7ef1f57. The RM27 review moved the check into `RegimeSet.evaluate`, which checks every regime before the first density, and into `Stochastic.density`. It removed the call in `_freeze_regimes` (commit e360555) |
| a result saved before the field existed | it loads undeclared. Before RM28, loading checked nothing, so its stored estimates answered as saved. `truncation_curve()`, `retarget()`, and `refit()` refuse. Commit 7ef1f57. [RM28](#rm28-declared-densities-of-user-written-interventions) later gave such a result the `undeclared_function_plugin` status. Its point estimates answer as saved, and `ci`, `pvalue` and `std_error` refuse (commit ed8e73e) |
| call sites | each `Stochastic` in the two executed documentation fences, `tests/regimes.py`, the registered generator `tests/studies/canonical_stochastic_regimes.py`, and the unit tests declares `density_kind="known"`. Commit 7ef1f57 |
| reference | the [scope page](technical-reference/scope-and-refusals.md#wrong-by-construction), the [user guide](user-guide/estimands.md#known-regimes), the [point-treatment reference](technical-reference/point-treatment-tmle.md#known-regimes), the [evidence table](technical-reference/evidence.md#the-table), and the [architecture invariants](architecture-invariants.md#public-causal-workflow) describe the declaration and both refusals. Commit df9326a |
| other policy callables | not in this row. [RM28](#rm28-declared-densities-of-user-written-interventions) delivered the declarations of user-written `Intervention` classes, `Rule`, and callable `DynamicRegimen` nodes |

The default of the field is a plain class attribute, as in RM13. A regime pickled before the field
existed therefore loads as undeclared, and `dataclasses.replace` still works on it.

Commit 77b9fe1 declared the exact-law witness before its test landed. The law is that of
`tests/discrete_law.py` with `p_w=[0.4, 0.4, 0.2]`, `g=[0.25, 0.5, 0.75]`, and
`q=[[0.05, 0.95], [0.1, 0.9], [0.1, 0.9]]`. Its 1000 rows realise the law exactly, and the fit is
in sample with oracle nuisances.

The density is the odds tilt $\delta g_n(w) / (\delta g_n(w) + 1 - g_n(w))$. Here $g_n(w)$ is the
sample treated share in stratum $w$. The density is declared `"known"`, and on this sample it
equals the tilt of the true mechanism.

The exact curve is the Gateaux derivative of the population-law tilted mean with $g$ recomputed
from the perturbed law. It differs from the curve for a fixed density by exactly one term:

$$
T = \frac{\delta \{\bar Q(1, W) - \bar Q(0, W)\}}{D^2} \{A - g(W)\},
\qquad D = \delta g(W) + 1 - g(W).
$$

`TestAnEstimatedTiltUnderstatesTheVariance` in `tests/unit/test_stochastic_regime_densities.py`
holds the witness for $\psi(P;q_\delta(g_P))$. Kennedy (2019), Section 3.3 and Appendix Corollary 2,
derives its mechanism term. The table gives the values measured on the delivered code. Each ratio
is the reported standard error over that target's exact standard error, from second moments.

| quantity | measured value | role |
| --- | --- | --- |
| ratio at $\delta = 2$ | 0.6226 | the witness. The test pins 0.6226 to 1e-4 and asserts a bound of 0.7 |
| ratio at $\delta = 0.5$ | 0.6853 | recorded. The test pins it to 1e-4 and claims no bound |
| reported curve against the Gateaux curve with $g^\star$ fixed | equal to 7.8e-16 at $\delta = 2$ and 1.1e-15 at $\delta = 0.5$ | the control. The test tolerance is 1e-12 |
| exact curve minus fixed curve minus $T$ | at most 5.0e-16 | the form of $T$. The test tolerance is 1e-12 |
| $E[T^2]$ | 0.1603 at $\delta = 2$ and 0.1158 at $\delta = 0.5$ | the nonzero witness. The test asserts that it exceeds 0.1 |
| $E[D_{\text{fixed}} T]$, with $D_{\text{fixed}}$ the curve with $g^\star$ fixed | 8.7e-19 and 3.5e-18 | orthogonality. The exact variance equals the fixed variance plus $E[T^2]$ to 1.4e-17. The test tolerance is 1e-12 |
| the reported curve against the fixed curve, as a ratio | 1.0000 | the mutation control. An oracle that froze $g$ would read 1 and fail the bound |
| the `ey_ipsi[odds x2]` curve of `TMLE(incremental=)` minus the reported regime curve | $T$, to the test tolerance of 1e-12 | the incremental axis carries the term that the regime curve omits |

The probe measured 0.6226 before the plan chose the bound. For the population-law target, the
bound 0.7 claims an understatement of more than 30 percent. A curve that carried $T$ would read 1.
The bound sits 0.077 above the
measured value, so it pins no digit. The pinned value records the measurement, as RM13 records
0.7417.

The plan also chose the law after the probe. On the default law of `tests/discrete_law.py`, the
ratio is 0.9667 at $\delta = 2$ and 0.9733 at $\delta = 0.5$. The $P(W)$ and $\bar Q$ of the
witness law, with $g = 0.5$ at every level, give 0.6060 at both values. That law cannot separate
$\delta$ from $1 / \delta$. The witness shows that the population-law target needs the term, and
it measures the omission on one law. It does not estimate a typical size.

For the population-law odds tilt, $T$ is orthogonal to the curve with $g^\star$ fixed. $T$ has mean zero given
$W$. That curve is a function of $W$ plus a residual with mean zero given $A$ and $W$. So the
omission understates the variance by exactly $\operatorname{Var}(T)$ at the truth. The probe
checked this on the witness law and on the default law, at both values of $\delta$, to 5.2e-17.
The row claims no direction for other densities.

The row planned four witnesses. The table gives the state of each. All of them are in
`tests/unit/test_stochastic_regime_densities.py`, which holds 86 tests. RM27 added four of them,
and the RM27 review added the seven tests of `TestTheEvaluatorsCheckFirst` (commit e360555).

| witness | state |
| --- | --- |
| 1. a pre-fit test pins both refusals and their messages, and a spy learner shows that no nuisance fit ran | delivered. `TestTheDeclarationIsRequired` pins the four refusals. `TestTheFitRefusesARestoredRegime` sets `density_kind` to `None` or `"estimated"` after construction. `fit`, `CausalStudy.estimate`, and `refit` then refuse, `NeverFit.calls` is 0, and a spy density is never called. A control with `"known"` reaches the first learner |
| 2. a mutation that removes the refusal makes that test fail | delivered. The test file commits four mutations in five tests, and nine more ran by hand. The text below gives them |
| 3. the exact-law witness | delivered. The ratio reads 0.6226 against the bound of 0.7 |
| 4. a control shows that a density declared known keeps its interval | delivered. `TestAKnownDensityKeepsItsInterval` fits the tilt of the true mechanism. The estimate keeps the `influence_curve` status and a finite `ci`, and its curve equals the fixed-density curve to 1e-12 |

Five committed tests apply four mutations, and each mutation replaces a function with a no-op
through `monkeypatch`. Removing the declaration check in `cleverly.interventions.base` fails every
declaration witness. Removing the check in `cleverly.estimators.tmle` fails the fit witnesses, and
`NeverFit` records fits. The same removal lets `truncation_curve()`, `retarget()`, and `refit()` run
on a legacy result. Removing the replay check lets the replay evaluate the density. Since commit
e360555, `test_removing_the_evaluator_check_evaluates_the_density` removes the check that
`RegimeSet.evaluate` and `Stochastic.density` run.

The fourth committed mutation replaces `FunctionDeclaration.refuse` with a no-op. Two RM13
witnesses and three RM25 witnesses then fail, because each one depends on that shared refusal.
RM27 added three design witnesses to that mutation test and renamed it
`test_removing_the_shared_declaration_fails_each_of_its_users`. Since commit 983b743, the test
draws its eight witnesses from `SHARED_REFUSALS`, and it runs each one as a control before the
mutation.

A further test checks that the three sites call one function object. Since commits e360555 and
983b743, it checks only `cleverly.estimators.tmle`. The replay no longer imports the function, and
the `cleverly.interventions.base` line compared the function with itself.

Nine more mutations ran by hand after commit 7ef1f57. Each run copied the file to a backup,
applied one mutation, and ran the RM25 file `tests/unit/test_stochastic_regime_densities.py`. That
file then held 63 tests. The ninth mutation also ran the RM13 file
`tests/unit/test_msm_projection_weights.py`, which then held 46 tests, as a separate run.

Each run then restored the file, and the restored file matched its blob at HEAD. No mutation
survived. Each count in the table is for the RM25 file, unless the row names the RM13 file.

| mutation | file | tests that failed |
| --- | --- | --- |
| remove the `Stochastic.__post_init__` call | `src/cleverly/interventions/base.py` | 16: each declaration refusal, the third positional argument, the `replace` of a legacy regime, and the refusal on the witness law |
| remove the call in `_resolve_estimands_for_data` | `src/cleverly/estimators/tmle.py` | 8: each fit entry for both restored states, the subclass that skips the declaration, and a legacy regime at the fit |
| remove the call in `_retarget_detailed` | `src/cleverly/estimators/tmle.py` | 2: `truncation_curve()` and `retarget()` on a legacy result |
| remove the call in `_freeze_regimes` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1: a legacy regime at replay |
| `isinstance` changed to `type(item) is Stochastic` | `src/cleverly/interventions/base.py` | 2: the subclass that inherits the declaration, and the subclass that skips it |
| move the `_freeze_regimes` check after `RegimeSet.evaluate` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1: a legacy regime at replay. The density ran two times before the refusal |
| remove the callable check | `src/cleverly/interventions/base.py` | 6: each density that is not callable, under each declaration |
| swap the undeclared and estimated texts of `_DENSITY_DECLARATION` | `src/cleverly/interventions/base.py` | 18: each test that matches either text |
| invert the membership test of `FunctionDeclaration.check` | `src/cleverly/_declarations.py` | 33 failures and 16 errors in the RM25 file. 26 failures and 14 errors in the RM13 file. Four failures in each file are its unknown-value witnesses |

The two replay rows no longer apply as written, because commit e360555 removed the call in
`_freeze_regimes`. At that commit, a deletion of the check in `RegimeSet.evaluate` failed 1 test,
and a deletion of the check in `Stochastic.density` failed 2. Commit 983b743 repeated the first
deletion, and it still failed 1 test.

No study was regenerated. Only `refuse_regime_densities` reads `density_kind`. `RegimeSet.evaluate`,
the influence code, and the estimator arguments did not change. So the change is result-neutral
under [what makes a study stale](development/method-benchmarking.md#what-makes-a-study-stale), and
no test gates the module hashes of a manifest.

A bitwise check supports that judgment. A script fitted `canonical_stochastic_regimes.fit_cleverly`
on replicates 0 and 1 of `draw_scenario`, once with the declared density and once with the uniform
density. Its 12 rows of `psi`, `std_error`, and `ci` were bitwise identical before and after the
change.

### RM26. Longitudinal clustered intervals at few clusters

[RM20](#rm20-intervals-outside-every-claimed-contract) gives a point-treatment clustered fit with
fewer than 40 clusters the `few_cluster_plugin` status. The longitudinal path had the same normal
reference and no status. An in-sample `LTMLE` fit with `id=` passed the cluster labels to
`make_estimate` (`src/cleverly/longitudinal/estimator.py:1546-1553` at dee5a5e). The estimate then
reported a normal-reference Wald interval (`src/cleverly/inference/influence.py:329-339` at
dee5a5e). No check read the cluster count, apart from the two-cluster minimum in
`src/cleverly/inference/cluster.py:150-151`.

`LongitudinalResult` (`src/cleverly/longitudinal/estimator.py:637` at dee5a5e) had no status
machinery, so the RM20 decision could not reach it. The cross-fitted clustered fit refuses `id=`
([F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle)), so the surface was the in-sample
fit alone. The summed-incidence table of a competing-risk fit also reported a cluster-robust
`std_err` (`src/cleverly/longitudinal/estimator.py:997` at dee5a5e).

| probe | result |
| --- | --- |
| `multivalue_panel(n=300, seed=43)` and `COLUMNS` from `tests/unit/test_sequential_design.py`. The cluster label `np.arange(300) // 30` gives 10 clusters of 30 consecutive rows, passed as `id=` to `LongitudinalData.from_frame`. `LTMLE({"never": 0, "always": 1}, reference="never", n_folds=1, simultaneous=False, random_state=0)`, with `LinearRegression` for the outcome and the pseudo-outcome, and `LogisticRegression(max_iter=1000)` for the treatment | `ate_regimen[always vs never]` read -0.216460, and it reported `ci` (-0.3045, -0.1284) under the `influence_curve` status. No warning named the cluster count |

The plan recorded `ci` (-0.3023, -0.1287) from a configuration that it did not record. The
configuration in the table does not reproduce it. The cluster assignment moves the interval. The
label `np.arange(300) % 10` gives `ci` (-0.3920, -0.0409) at the same point estimate. The review
re-ran the table row on 2026-09-23, with scikit-learn 1.9.0 and NumPy 2.4.6.

The sources and the threshold are those of RM20. Nugent et al. (2024), Section 2.2, recommend a
$t$ reference below 40 clusters. Benitez et al. (2023), Section 3.1.2, paragraph on inference, and
Section 3.2.1, last paragraph, recommend it at every cluster count. No registered study fits a
clustered longitudinal design, so the correction moves no study.

The row asked for three corrections:

1. Give `LongitudinalResult` an inference status through the table that RM20 generalizes. Stamp
   each mean, contrast, and MSM coefficient.
2. Apply the RM20 few-cluster decision to the in-sample clustered fit, with the same threshold
   constant.
3. Rename the summed-incidence `std_err` under that status, as RM12 renamed the point-treatment
   spread columns.

The 2026-09-23 plan measured three facts on the code at dee5a5e. The end-of-study fit used
`multivalue_panel(n=400, seed=43)`, the competing-risk fit `make_longitudinal_competing(n=400,
seed=3)`, and the MSM fit `make_longitudinal(n=400, seed=0)`. Each fit passed the labels
`np.arange(400) * k // 400` as `id=`, and it had one fold.

| probe | result |
| --- | --- |
| an end-of-study, a competing-risk, and an MSM fit at 39 and at 40 clusters | `cluster_inference_status` gave `few_cluster_plugin` at 39 clusters and `influence_curve` at 40, for each kind of fit. The point estimates at 39 and 40 clusters were bitwise identical |
| the 39-cluster competing-risk fit, with `stamp_inference` applied to its estimates by hand | `summary()` and `curve()` raised `CapabilityError`, because they read `ci` and `std_error`. `to_frame()` and `incidence_total()` answered |
| the same fit, with the stamp on the fit alone | `truncation_curve` raised `CapabilityError` with the code `longitudinal_replay_fitted_bound_mismatch`. The replay compares every field of each estimate, `inference` included |

All three corrections shipped, with the decisions that commit 7aa7fbb recorded. One private
function, `_inference_status(data, folds)` in `cleverly.longitudinal.estimator`, decides the
status. The RM20 code supplies each part, unless the row says otherwise. Each row of the table ends
with the commit that shipped it.

| part | what shipped |
| --- | --- |
| status rule | `_inference_status` calls `cluster_inference_status` on the prepared cluster labels, and on the weights of a weighted fit. `LongitudinalData` has no strata, so the count is the positive-mass cluster count of the whole fit. The function reads nothing fitted, and it passes `folds.n_folds > 1` as `cross_fit`. A live fit refuses `id=` above one fold, so it can take `few_cluster_plugin` only. Commit a41848d |
| stamp site | `_estimates` and `_msm_estimates` take a required `inference` argument and pass it to each `make_estimate` call. `LTMLE.fit` and the replay function `_refit_bound` both compute the status and call them, so the replay stays equal to the fit. Commit a41848d |
| result status | `LongitudinalResult.inference_status` returns the one status of the estimates, as `TMLEResult.inference_status` does. It returns `influence_curve` on a fit with no estimate. Commit a41848d. Both properties and `CVTargeting.inference` now read that rule from `reported_status` in `src/cleverly/inference/results.py` (commit e12c545) |
| `summary()` | a fit that supplies no inference prints the `normal-reference se` column, the reason of the status, and no interval or p-value column. The ordinary branch is byte-identical. Commit a41848d |
| `curve()` | the three spread columns take the names `plugin_std_err`, `plugin_interval_lower`, and `plugin_interval_upper` through `spread_name`, and an `inference` column is added, as `to_frame()` does. The values come from `plugin_std_error` and `plugin_interval`, the bodies that `std_error` and `ci` read. Commit a41848d |
| `incidence_total()` | `std_err` takes the name `plugin_std_err` through `spread_name`, as RM12 renames the spread columns of a sweep. Commit a41848d |
| simultaneous bands | `LTMLE._bands` returns `None` on a fit that supplies no inference, as `TMLE.fit` does. It reads the status that `LTMLE.fit` stamped (commit e12c545). `summary()` prints the RM20 sentence on a fit with two or more estimates. `simultaneous=True` is the default, so a raise would stop each default fit. The RM26 review kept this behavior for an explicit request too. Commit a41848d |
| shared texts | `StatusRecord.summary_note()` and `NO_SIMULTANEOUS_BANDS` in `src/cleverly/_inference_status.py` hold the two texts that both summaries print. `TMLEResult.summary()` reads them and stays byte-identical. Commit a41848d |
| nuisance note | the longitudinal branch of `_nuisance_item` in `src/cleverly/assessment.py` adds the `assessment_note` of the status. It reads the status of the result. A call with no result adds no note (commit e12c545). Commit a41848d |
| zero-mass clusters | the cluster line of `summary()` adds `positive weight mass in N` when a cluster has zero weight mass, as the point-treatment line does. A weighted fit with 40 or more positive-mass clusters and one zero-mass cluster now prints it too. Commit a41848d. Both lines now read the clause from `positive_mass_clause` in `src/cleverly/inference/cluster.py` (commit e12c545) |
| a result saved before the status | `LongitudinalResult.__setstate__` recomputes the status from the saved data and folds. When that status supplies no inference and the saved estimates declare another, it stamps them again. It drops the bands and the assessment cache, as RM20 does. The replay then stays equal on a restored artifact. Commit a41848d. A later repair in this branch makes `LongitudinalData.__setstate__` restore unit weights and their declaration on development artifacts saved before commit 16d2643. It recovers the backend name from the old frame field, then drops that frame. Releases 0.1.0 and 0.1.1 carry the weight fields. An unclustered result loads as saved (commit e12c545) |
| E-value | no change. Each longitudinal fit reports the E-value row `unavailable` already |
| test support | `at_or_below` moved to `tests/unit/_inference_status_support.py`, and `legacy_copy` reads `cv_targeting` with `getattr`, so both serve the longitudinal tests. Commit a41848d |
| reference | the [data design guide](user-guide/data-design.md) and [results and assessment](user-guide/results-assessment.md#contrasts-and-simultaneous-inference) describe the status on the longitudinal fit. So do [inference status](technical-reference/inference.md#inference-status), [clusters](technical-reference/inference.md#clusters), the [scope page](technical-reference/scope-and-refusals.md), [CV-TMLE](technical-reference/cv-tmle.md), the [longitudinal reference](technical-reference/longitudinal-tmle.md), and the [architecture invariants](architecture-invariants.md). Commit f309679 |
| a cross-fitted clustered result saved before commit 5f32c14 refused the fit | not in this row. [RM29](#rm29-saved-cross-fitted-clustered-longitudinal-results) now gives it a status that names the refused composition |

The row planned three witnesses, and the plan added two. The table gives the state of each. All of
them are in `tests/unit/test_longitudinal_cluster_status.py`, which held 39 tests at RM26 delivery.

| witness | state |
| --- | --- |
| 1. a test fits an in-sample `LTMLE` with `id=` and 39 clusters, and pins the status, the refusal, and the message | delivered. `TestFewClustersWithholdTheLongitudinalInterval` fits an end-of-study, a survival, a competing-risk, and an MSM fit. It pins each refusal message whole, the summary note, and the renamed columns of `curve()`, `incidence_total()`, and `coefficients()`. On the survival fit it reads both views of `curve()`, and it checks how each view maps the plug-in bounds. It also checks the skipped bands, the nuisance note, the truncation replay, and the `CausalStudy` workflow |
| 2. a control at 40 clusters keeps its interval | delivered. `TestFortyClustersKeepTheLongitudinalInterval` fits the same rows in 40 clusters. They keep `influence_curve`, the inferential column names, and the default bands. Their point estimates equal those at 39 clusters |
| 3. a mutation that restores the `influence_curve` status makes the first test fail | delivered as `test_restoring_the_influence_curve_status_fails` in `TestTheMutationsFailTheWitness`, on each kind of fit |
| 4. a rule that ignores the weights fails a fit with 40 clusters and zero weight on one of them | delivered. `test_zero_mass_clusters_do_not_count` is the witness, and `test_all_positive_weights_keep_the_interval` is its control |
| 5. a replay that drops the status fails the truncation curve | delivered as `test_a_replay_without_the_status_fails_the_truncation_check`. It raises `CapabilityError` with the code `longitudinal_replay_fitted_bound_mismatch` |

`TestAnOlderLongitudinalArtifact` restores a 39-cluster fit saved without the status, through
`pickle` and through `serialize`. The result loads under `few_cluster_plugin`, and its truncation
curve answers. A 40-cluster artifact loads as saved. The class also deletes the `weights` field
of the saved data, on an unclustered fit and on a 39-cluster fit. The first loads under
`influence_curve`, and the second loads under `few_cluster_plugin`.

`TestTheMutationsFailTheWitness` commits five mutations in eight tests, through `monkeypatch`.
Restoring the `influence_curve` status fails the witness on each kind of fit. A threshold of 0
fails the witness and the nuisance note. An at-or-below comparison still passes the witness, and
it fails the 40-cluster control. A rule that ignores the weights fails the zero-mass witness. A
replay without the status fails the truncation check.

Fifteen more mutations ran by hand at commit 3eef4a4, after the review fixes to the code and the
tests. Each run applied one mutation to the file at HEAD and ran
`tests/unit/test_longitudinal_cluster_status.py`. It then restored the file, and the restored file
matched its blob at HEAD. Each mutation failed at least one test. The table gives them. The row for
the direct read of `data.weights` comes from a later run, which the paragraph after the table
explains.

| mutation | file | tests that failed |
| --- | --- | --- |
| `_inference_status` returns `influence_curve` | `src/cleverly/longitudinal/estimator.py` | 18: the four withholding witnesses, the four report checks, the bands, the note, the zero-mass witness, the study workflow, both legacy routes, both 39-cluster loads without `weights`, the at-or-below control, and the replay mutation test |
| `_inference_status` reads `data.weights` directly | `src/cleverly/longitudinal/estimator.py` | 4: the unclustered and the 39-cluster loads without `weights`, on both routes |
| `_refit_bound` stamps `influence_curve` | `src/cleverly/longitudinal/estimator.py` | 3: the truncation replay and both legacy routes |
| `_bands` builds bands at any status | `src/cleverly/longitudinal/estimator.py` | 4: the skipped default bands, the study workflow, and both legacy routes |
| `__setstate__` skips the re-stamp | `src/cleverly/longitudinal/estimator.py` | 4: both legacy routes and both 39-cluster loads without `weights` |
| the re-stamp keeps the bands | `src/cleverly/longitudinal/estimator.py` | 4: both legacy routes and both 39-cluster loads without `weights` |
| `curve()` keeps the inferential names | `src/cleverly/longitudinal/estimator.py` | 2: the survival and competing-risk report checks |
| `curve()` adds no `inference` column | `src/cleverly/longitudinal/estimator.py` | 2: the survival and competing-risk report checks |
| `incidence_total()` keeps `std_err` | `src/cleverly/longitudinal/estimator.py` | 1: the competing-risk report check |
| `summary()` drops the positive-mass fact | `src/cleverly/longitudinal/estimator.py` | 1: the zero-mass witness |
| `summary()` drops the no-bands line | `src/cleverly/longitudinal/estimator.py` | 1: the skipped default bands |
| `summary()` drops the reason of the status | `src/cleverly/longitudinal/estimator.py` | 6: the four withholding witnesses, the zero-mass witness, and the at-or-below control |
| the nuisance item drops the note | `src/cleverly/assessment.py` | 1: the nuisance and assessment note |
| `_msm_estimates` drops the status | `src/cleverly/longitudinal/estimator.py` | 2: the MSM witness and the MSM report check |
| `_estimates` drops the status of the contrasts | `src/cleverly/longitudinal/estimator.py` | 11: the end-of-study, survival, and competing-risk witnesses, the survival and competing-risk report checks, the bands, the note, the zero-mass witness, the at-or-below control, and both legacy routes |

A sixteenth mutation at commit 3eef4a4 removed the early return of the re-stamp on unclustered
data, and no test failed. On unclustered data, `_inference_status` returns `influence_curve`, and
it reads the weights with `getattr`. So the early return was redundant, and a later commit removed
it. After that removal, four mutations on the load path ran again. The direct read of
`data.weights` now fails four tests, not two. The rows for the `influence_curve` return, the
skipped re-stamp, and the kept bands did not change.

The row probe, re-run on the delivered code, reads `few_cluster_plugin`. The table gives the
estimates.

| estimate | point estimate | `plugin_interval` |
| --- | --- | --- |
| `ey_regimen[never]` | 0.626042 | (0.5231, 0.7289) |
| `ey_regimen[always]` | 0.409581 | (0.3154, 0.5037) |
| `ate_regimen[always vs never]` | -0.216460 | (-0.3045, -0.1284), which equals the old `ci` |

`ci`, `pvalue`, and `std_error` raise `CapabilityError` on each estimate. With the default
`simultaneous=True`, the fit builds no band, and `summary()` prints the no-bands line.

No study was regenerated. No registered study fits a clustered longitudinal design. The stamp
replaces the `inference` field only, and the renamed columns read the bodies that the refused
accessors call. So the change is result-neutral under
[what makes a study stale](development/method-benchmarking.md#what-makes-a-study-stale).

A bitwise check supports that judgment. A script hashed 70 outputs before and after the change,
and every hash matched. It covered `fit_cleverly` of `canonical_ltmle`,
`canonical_ltmle_competing`, and `canonical_longitudinal_msm`, with their frames, summaries,
curves, incidence totals, coefficients, and truncation curves. It also covered unclustered and
100-cluster end-of-study, survival, competing-risk, and MSM fits with default bands. It covered
the point-treatment `TMLEResult` `summary()` and `to_frame()` at 39 and 40 clusters. The check ran
again after commit f309679 and after the review fixes in commit e12c545, and all 70 hashes
still matched.

The delivery found four more surfaces. The table gives where each one is held.

| finding | decision | where it is held |
| --- | --- | --- |
| a cross-fitted clustered `LTMLE` result saved by release 0.1.0 or 0.1.1, before commit 5f32c14 refused the fit | the RM26 rule alone left an equal 40-cluster artifact under `influence_curve` and an unequal artifact under `unequal_cluster_plugin` | [RM29](#rm29-saved-cross-fitted-clustered-longitudinal-results) now stamps `cross_fitted_longitudinal_plugin` at every count and size. [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) holds the route that reopens the fit |
| an explicit `simultaneous=True` on a few-cluster `LTMLE` fit | the fit builds no band and raises no warning, as a point-treatment fit does. The summary names the omission. The RM26 review kept this policy for both estimators | [RM16](#rm16-summary-and-error-message-accuracy) records the policy |
| a fit with `id=` and one unit in each cluster | the over-refusal that RM20 recorded now reaches `LTMLE`. The status withholds a number and publishes none, so no change was made | [RM20](#rm20-intervals-outside-every-claimed-contract) |
| `LongitudinalNuisanceDiagnostics.summary()` | it carries no status note. It publishes no spread, so it needs none | this row. No change |

### RM27. Declared MSM design functions

The [MSM reference](technical-reference/msm-projections.md#the-algorithm-as-implemented) states
that "The working design and the weights are known functions". RM13 made the weight a declaration.
Nothing declared the design. A `design` callable receives an arm label and a covariate frame, and
it can close over any statistic of the sample. A design that centres a covariate at its sample
mean is a common example. The projection is then a functional of $P$ through the design, and the
reported influence curve omits that pathwise derivative.

The published MSM derivations cover a user-supplied working design.
[Petersen et al. (2014)](https://doi.org/10.1515/jci-2013-0007), Section 3, derive the pooled
longitudinal TMLE for that target. [Martin, Santacatterina and Díaz (2024, arXiv v1)](https://arxiv.org/html/2409.18782v1),
Definition 1, also use a user-given transformation. Their Section 1 says MSM model selection lacks
general post-selection inference.

The software articles document inputs and variance calculations. They do not derive the curve
for this sample-centred design.

| source | relevant scope |
| --- | --- |
| JSS [`tmle`](https://www.jstatsoft.org/article/view/v051i13), Section 2.6, pages 8-9; Section 2.7, page 10; Section 3.6, pages 20-21 | the working MSM is user-specified, and the printed influence curve uses that design. `hAVform` can estimate the projection weight $h(A,V)$. That weight option does not derive a curve for an estimated design |
| JSS [`ltmle`](https://www.jstatsoft.org/article/view/v081i01), Sections 4.1-4.3, pages 13-18 | the working model uses specified functions $\phi_j(d,t)$ and a `summary.measures` array. `msm.weights` can use empirical regime shares. That weight option does not derive a curve for an estimated design |
| R Journal [`CIMTx`](https://journal.r-project.org/articles/RJ-2022-058/), Section 3.1 | its multi-treatment TMLE calls `tmle` and uses bootstrap intervals for treatment effects. It does not study a learned MSM design |

The omitted term concerns a population-law target whose design uses $c(P)=E_P[W]$. A fixed
design with the observed sample centre defines a different, sample-dependent target. The
fixed-design curve alone does not validate an interval for that target when the same rows set the
centre and estimate the projection.

The review of the RM20 and RM13 delivery raised this surface as a plausible finding. A 2026-09-23
probe measured it on an exact law, as RM13 measured the weight.

The probe used the law of `tests/discrete_law.py` with `q=[[0.1, 0.2], [0.5, 0.6], [0.8, 0.9]]` and
the default `P_W` and `G`. Its 1000 rows realise the law exactly. The fit was in sample, with oracle
nuisances, the identity link, and uniform weights. The design was $[1, a, W - c]$, where $c$ is the
sample mean of $W$, 0.7. On this sample, $c$ equals $E_P[W]$.

| term | reported standard error over the exact standard error | role |
| --- | --- | --- |
| `msm[(intercept)]` | 0.8927 | the witness |
| `msm[a]` | 1.0000 | a negative control. A shift of the $W$ column moves only the intercept |
| `msm[W]` | 1.0000 | a negative control, for the same reason |

The exact curve comes from the Gateaux derivative of the projection with $c = E_P[W]$ recomputed
from the perturbed law. It differs from the curve with $c$ fixed by exactly $\beta_W (W - E_P[W])$,
to 2.2e-16. On the probe's oracle, the reported curve equals the fixed-$c$ Gateaux curve to
1.6e-13. The delivered test measures 1.93e-13 against its own oracle. So the reported curve is right
for the design that the user fixed, and it omits the term of the design that the user estimated.
The ratio uses the second moment of the curve, as RM13 does.

The omitted term changes the variance by $\beta_W^2 \operatorname{Var}(W) + 2 \beta_W
\operatorname{Cov}(D, W)$, where $D$ is the reported intercept curve. That change can have either
sign, and it vanishes at $\beta_W = 0$. On the default law of `tests/discrete_law.py`, $\beta_W$ is
0.034, and the intercept ratio is 0.9980 with uniform weights. With the known weight
$1 + 0.5 a + 0.25 W$, $\beta_W$ is 0.046 and the ratio is 0.9976.

The row asked for three corrections:

1. Decide how a `design` declares that it is a known function. RM13 put the weight declaration on
   `MSM`. `MSM.linear` builds a known design, so it can carry the declaration itself. RM25 added
   `cleverly._declarations.FunctionDeclaration`, which holds the check and the texts of one
   declaration, so a design declaration can be a third instance. The review later removed the
   declaration from `MSM.linear`, and the exact type of its design now reads as known (commit
   a237257).
2. Refuse a design declared estimated before any nuisance fit. The message names the missing
   pathwise-derivative term.
3. Correct the [MSM reference](technical-reference/msm-projections.md) and the
   [scope page](technical-reference/scope-and-refusals.md#wrong-by-construction), so that each
   one describes the declaration and the refusal.

The `MSM` docstring listed its fields under `Attributes` (`src/cleverly/msm.py:434` at 0e09dd6).
`CLAUDE.md` puts the fields of a frozen dataclass under `Parameters`, because numpydoc reads the
generated signature. The fields `from_linear` and `doses` had no entry. The RM25 plan found this
and left it for this row, because item 1 changes that docstring.

All three corrections shipped, with the decisions that commit 1d0aff6 recorded. One function,
`cleverly.msm.refuse_msm_functions`, checks the design and then the weight. The RM13 and RM25
patterns supply each part, unless the row says otherwise. Each row that shipped a change names its
commit. Later commits changed nine rows, and each of those rows names the later commit too. The
review fixes at the end of this section give the reason for each change.

| part | what shipped |
| --- | --- |
| declaration | a field `design_kind` on `MSM`. It takes `"known"`, `"estimated"`, or `None`, and it defaults to `None`. Its annotation is the one that `density_kind` has: `FunctionKind` or `None`. No new public alias names it. Commit 268dc62 |
| field order | `design_kind` is the last field, after `weights_kind`. So `link` stays the fourth positional argument, as the RM13 review required in commit 8a33661. Commit 268dc62 |
| a model saved before the field existed | the default is a plain class attribute, and no `__setstate__` was added. An old model loads with `design_kind` read as `None`, and `dataclasses.replace` still works on it. Commit 268dc62 |
| shared declaration | a third `FunctionDeclaration`, `_DESIGN_DECLARATION`, sits beside `_WEIGHTS_DECLARATION` in `src/cleverly/msm.py`. Its meaning text says that the field declares whether the working design is a fixed function or one computed from the sample. The docstring of `cleverly._declarations` names the MSM design as the third user. Commit 268dc62 |
| `MSM.linear` | it passed `design_kind="known"` itself, because `_LinearDesign` is known by construction. Commit 268dc62. Commit a237257 removed that keyword, so the model holds `None`, and the exact-type rule reads its design as known. It takes no `design_kind` argument |
| exact-type rule | a private helper, `_design_kind(model)`, reads `"known"` when `design_kind` is `None` and `type(model.design) is _LinearDesign`. Otherwise it reads the field. Only the refusal and the replay call it. Commit 268dc62 added it for a model saved before the field existed. Since commit a237257, it is the only way that any `MSM.linear` model reads as known |
| why the exact type | a pickled `MSM.linear` model keeps the type `_LinearDesign`. A user can set `from_linear=True` on any design, and a subclass of `_LinearDesign` passes `isinstance`. Neither of those two shows a known design, so each one refuses when it is undeclared. Commit 268dc62 |
| undeclared design | a callable `design` with `design_kind=None`, outside the exact-type rule, raises `CapabilityError`. The message asks for `design_kind="known"`. Commit 268dc62. Commit a237257 changed its last sentence to "MSM.linear needs no declaration: the design it builds is known by construction." |
| estimated design | `design_kind="estimated"` raises `CapabilityError`. The message names the pathwise derivative through the sample statistic. It asks the user to fix the statistic before the fit, for example to centre at a stated constant. Commit 268dc62 |
| an unknown value | a value other than `"known"`, `"estimated"`, or `None` raises `DataError`, from the shared check. Commit 268dc62 |
| a design that is not callable | `DataError`, as before. The message now names both signatures: `(arm_label, covariate_frame) -> (n, p)` for a point treatment, and `(regimen_label, horizon, baseline_frame) -> (n, p)` for a longitudinal regimen MSM. The check moved from `MSM.__post_init__` into the shared function. Commit 268dc62 |
| check function | `refuse_projection_weights` became `refuse_msm_functions`, with no alias. Only `cleverly.msm.__all__` lists the name, and no API page does. The body, each message, and each check site stayed the same. Commit 03917d1 |
| check order | 1. a design that is not callable. 2. the design declaration: an unknown value, then `None` or `"estimated"`. 3. the RM13 weight checks, unchanged. RM25 uses the same order for the density. Commit 268dc62 |
| check sites | the four RM13 sites call the renamed function: `MSM.__post_init__`, `TMLE._resolve_estimands_for_data`, `TMLE._retarget_detailed`, and `LTMLE.fit`. So each one checks the design too. Commit 268dc62. Commit e360555 added `MSMSet.evaluate` and `evaluate_regimen_msm`, which check before they call the design or the weight |
| replay check | `_freeze_msm` called `refuse_msm_functions` before `MSMSet.evaluate` ran any design or weight. Before the change, a result whose callable weight lost its declaration refused at replay only after the design and the weight had each run two times. Each then ran zero times. Commit 67ab7fc. Commit e360555 removed that call, because `MSMSet.evaluate` now runs the same check on the same argument first. Each count stays zero |
| replay carry | both `replace` calls in `_freeze_msm`, for an arm replay and for a dose replay, pass `design_kind=_design_kind(model)`. Neither passes the literal `"known"`. Commit 268dc62. Commit 28df2fb added the witnesses of the dose call, and the text below gives the reason. Since commit e360555, these calls are a backstop: they refuse an undeclared model if the evaluator check is removed |
| remedy texts | three messages send the user from `MSM.linear` to a written design: the `MSM.linear` docstring, the `_numeric_level` message, and the `from_linear` message in `cleverly.longitudinal.msm`. Each one now names `design_kind='known'`. Commit 268dc62 |
| call sites | 65 written `MSM(design=...)` calls in `tests/` declare `design_kind="known"`. They include the two registered generators and the tutorial semantics. The two code cells of the `msm-projections` notebook and the `MSM` docstring example declare it too. A call to `MSM.linear` needed no change. Commit 268dc62 |
| test oracle | `msm_beta` in `tests/discrete_law.py` takes an optional `design`. `gateaux_eif` in `tests/unit/_declaration_support.py` computes the Gateaux curve for the RM13, RM25, and RM27 witnesses. Commit 268dc62. Commit 16b88e3 moved `gateaux_eif` to `tests/discrete_law.py`, where `eif` and `weighted_eif` call it |
| `refuse_unsupported("estimated_design")` | not added. No caller needs a named refusal for the design |
| docstrings | the `MSM` docstring lists all eight fields under `Parameters`, and it absorbs the `#:` comments of the fields. `MSM.linear` and `refuse_msm_functions` have full numpydoc sections. `python -m numpydoc validate` reports no finding of the checks that `docs/conf.py` enables on the three objects. Commit 268dc62. Commit e360555 gave `MSMSet.evaluate` and `evaluate_regimen_msm` their `Parameters`, `Returns`, and `Raises` sections |
| reference | the [MSM reference](technical-reference/msm-projections.md#variations) and the [scope page](technical-reference/scope-and-refusals.md#wrong-by-construction) describe the declaration and both refusals. So do the [evidence table](technical-reference/evidence.md#the-table), the [architecture invariants](architecture-invariants.md#public-causal-workflow), the [user guide](user-guide/estimands.md#marginal-structural-models), and the [validation methods](technical-reference/validation-methods.md#simulated-common-cause-stress-surface). The markdown cells of the notebook explain the declaration. Commit 4b693b4 |
| a false declaration | not detected. A design computed from the sample and declared `"known"` fits. The exact-law witness below measures what the false statement costs |
| `dataclasses.replace(model, design=new)` | it keeps the old `design_kind`, as `weights_kind` does for a new weight. The MSM reference records this limit. Commit 4b693b4. Since commit a237257, an `MSM.linear` model holds `None`, so `replace` refuses a new design that has no declaration |

The plan probes found two more surfaces on 0e09dd6. The table gives the decision for each.

| surface | probe | decision |
| --- | --- | --- |
| replay of a result whose callable weight lost its declaration | `validate_fixed_replay` refused with the RM13 message. Before the refusal, `MSMSet.evaluate` called the design two times and the weight two times. RM25 checks before its `RegimeSet.evaluate`, and RM13 did not | in this row. The replay check moved before `MSMSet.evaluate`, and both counts are now zero. Commit 67ab7fc. Commit e360555 moved the check into `MSMSet.evaluate`, and both counts stay zero |
| a saved `LTMLE` MSM result | the pickled result carries no `MSM` object. After a load, its truncation curve calls no design and no weight, and it has no standard-error column. `refute` refuses a longitudinal fit (`src/cleverly/validation/refute.py:1037` at 0e09dd6) | not in this row. No user function runs on a recomputation, and no standard error is published. `tests/unit/test_longitudinal_truncation_refit.py` pins the column list |

The RM13 record said that "`LTMLE` has no retarget sweep". That was not accurate.
`longitudinal_truncation_curve` arrived in commit 4a8b09b on 2026-09-10, before RM13. This
delivery corrected the [RM13 record](#rm13-estimated-msm-projection-weights).

Commit 1d0aff6 declared the exact-law witness before its test landed. The law and the fit are the
2026-09-23 probe above. The test design is $[1, a, W - c]$, where $c$ is the sample mean of $W$.
The test declares that design `"known"`, which is false. On this sample, $c$ is 0.7, so the
fixed-centre design is $[1, a, W - 0.7]$.

The omitted term is $T = \beta_W (W - 0.7)$ in the intercept curve.
`TestAnEstimatedCentreMisstatesTheVariance` in `tests/unit/test_msm_design_declaration.py` holds
the witness. The table gives the values measured on the delivered code. Each ratio is the reported
standard error over the exact standard error, from second moments.

| assertion | tolerance or bound | measured value |
| --- | --- | --- |
| the premise: $c = 0.7$, and each coefficient equals the fixed-centre projection | 1e-12 | at most 1.5e-14 |
| the reported curve equals the fixed-centre Gateaux curve, term by term | 1e-10 | at most 1.9e-13 |
| the exact curve minus the fixed-centre curve equals $T$ for the intercept, and zero for `a` and `W` | 1e-12 | 2.2e-16 and 4.4e-16 |
| $E[T^2]$, the nonzero witness | above 0.05 | 0.0779 |
| the intercept ratio, the witness | pinned at 0.8927 to 1e-4, and below 0.95 | 0.8926840770 |
| the `msm[a]` and `msm[W]` ratios, the negative controls | within 1e-9 of 1 | within 3.8e-15 and 2.0e-14 of 1 |
| a design declared `"estimated"` on this law | refused before any learner | refused, in `test_an_estimated_declaration_is_refused_on_this_law` |
| the intercept ratio with an oracle that freezes the centre, mutation M6 | fails the bound of 0.95 | 1.000000000000006 |

The probe measured 0.8927 before the plan chose the bound. The bound 0.95 claims an
understatement of more than 5 percent. A curve that carried $T$ would read 1. The bound sits 0.057
above the measured value, so it pins no digit. RM13 left a margin of 0.058 between 0.7417 and 0.8.
The control tolerance of 1e-9 sits at least four orders of magnitude above the measured error.

`TestAKnownDesignKeepsItsInterval` holds two controls that keep their intervals. The first is the
fixed centre 0.7, declared `"known"`. Its curve equals the fixed-centre Gateaux curve to 1.9e-13,
against the tolerance 1e-10, and it is bitwise equal to the witness curve. The second is
`MSM.linear(modifiers=("W",), interaction=False)`. Its curve equals its Gateaux curve to 1.6e-14,
against the same tolerance. Each control keeps the `influence_curve` status and a finite `ci`.

The row planned four witnesses. The table gives the state of each. All of them are in
`tests/unit/test_msm_design_declaration.py`, which holds 88 tests at commit 5e055cc and 100 at
commit 97cc57e.

| witness | state |
| --- | --- |
| 1. a pre-fit test pins the refusal and its message, and a spy learner shows that no nuisance fit ran | delivered. `TestTheDeclarationIsRequired` pins each refusal and its message. `TestTheFitRefusesARestoredModel` sets `design_kind` to `None` or `"estimated"` after construction. `fit`, `CausalStudy.estimate`, `refit`, and `LTMLE.fit` then refuse, `NeverFit.calls` is 0, and a spy design is never called. A control with `"known"` reaches the first learner. The review added a malformed model at each of the four entries, which raises `DataError` before any call (commit 9258d5a). `TestTheEvaluatorsCheckFirst` hands a restored model to `MSMSet.evaluate` and `evaluate_regimen_msm` (commit e360555). `TestTheDeclarationRefusalComesFirst` shows that a restored model meets its own refusal before a refusal of the fit configuration (commit 97cc57e) |
| 2. a mutation that removes the refusal makes that test fail | delivered. The test files commit six mutations, and commit 97cc57e added M7. Eighteen more ran by hand in the first pass, and eight more after the review. The text below gives them |
| 3. the exact-law witness, with a bound fixed before its run, and `msm[a]` and `msm[W]` as negative controls | delivered. The intercept ratio reads 0.8927 against the bound of 0.95, and each control is within 2.0e-14 of 1 |
| 4. a control shows that `MSM.linear`, and a design with a fixed centre, keep their intervals | delivered in `TestAKnownDesignKeepsItsInterval` |

Three more classes cover the exact-type rule and results saved before the field existed.
`TestTheExactTypeRuleReadsTheShorthand` shows that an `MSM.linear` model passes the check, whether
it is new or saved before the field existed. A design put into it with `dataclasses.replace`, a
legacy written design, a forged `from_linear=True`, and a subclass of `_LinearDesign` refuse. The
class `TestALegacyResultKeepsItsNumbersAndRefusesARecomputation` kept the stored interval of a
legacy result with a written design, and `truncation_curve()`, `retarget()`, and `refit()` refuse. A
legacy `MSM.linear` result still recomputes all three. `TestTheReplayChecksAndCarriesTheDeclaration`
covers the replay of an arm fit and of a dose fit.

[RM28](#rm28-declared-densities-of-user-written-interventions) reversed the stored interval
(commit ed8e73e). A legacy result with a written design now takes the `undeclared_function_plugin`
status, and its point estimates stand. `ci`, `pvalue` and `std_error` refuse, and the old interval
remains as `plugin_interval`. The class is now
`TestALegacyResultKeepsItsPointEstimatesAndRefusesARecomputation`.

Each committed mutation replaces a function through `monkeypatch`. The table gives the test that
each one fails.

| mutation | the test that fails |
| --- | --- |
| M1. `refuse_msm_functions` in `cleverly.msm` becomes a no-op | `test_removing_the_declaration_check_fails_every_declaration_witness`. Commit a237257 added `test_replacing_the_shorthand_design_drops_its_declaration` to its witnesses |
| M2. the same in `cleverly.estimators.tmle` and `cleverly.longitudinal.estimator` | `test_removing_the_fit_layer_check_fails_the_fit_witnesses`, where `NeverFit` records fits, and `test_removing_the_fit_layer_check_fails_the_recomputation_refusal`, for `truncation_curve()`, `retarget()`, and `refit()` on a legacy result. Since commit e360555, M2 also removes the checks of `cleverly.msm` and `cleverly.longitudinal.msm` for a fit and a refit. A `TMLE` point fit evaluates the design before its first learner, so `MSMSet.evaluate` would refuse first. A sweep and a retarget reuse the stored arrays, so their M2 removes only the `TMLE` check |
| M3. `MSMSet.evaluate` runs without its check. Before commit e360555, M3 removed the check in the replay module | `test_removing_the_evaluator_check_evaluates_both_and_still_refuses`, once for `weights_kind` and once for `design_kind`. The design and the weight each run two times. Then `replace` refuses the undeclared function, which shows that the replay carries the declaration and does not forge it. Commit 983b743 made the RM13 twin of this test its `weights_kind` case |
| M4. the replay `replace` drops `design_kind` | `test_a_replay_that_drops_the_declaration_refuses`, on an `MSM.linear` fit. The table of deviations below gives the reason |
| M5. `FunctionDeclaration.refuse` becomes a no-op | the RM13, RM25, and RM27 witnesses, in `test_removing_the_shared_declaration_fails_each_of_its_users` of `tests/unit/test_stochastic_regime_densities.py`. Since commit 983b743, it draws its eight witnesses from `SHARED_REFUSALS`, and it runs each one as a control before the mutation |
| M6. the oracle freezes the centre | `test_mutation_a_frozen_oracle_loses_the_witness`. The ratio reads 1.000000000000006 and fails the bound of 0.95 |
| M7. the same in `cleverly.estimators.tmle` alone | `test_removing_the_fit_layer_check_alone_lets_the_later_refusal_answer`, once for each of three refusals of the fit configuration. The restored model then meets the refusal of the configuration. Commit 97cc57e |

`test_every_site_calls_the_one_refusal` checks that `cleverly.estimators.tmle`,
`cleverly.longitudinal.estimator`, and `cleverly.longitudinal.msm` hold the function that
`cleverly.msm` defines. It also checks that the replay module and `cleverly.msm` hold one
`_design_kind`. The replay module no longer imports the function (commit e360555).

The first pass ran eighteen more mutations by hand, against commit 268dc62. At that commit, the
exact-type rule was called the legacy rule. The plan named H1 to H12. The delivery added H13, the
pair H2 and H3, and one mutation for each declaration on each `replace` call of the replay.

Each run copied the file to a backup with its sha256, applied one mutation, and ran three files.
These are the RM27 file, the RM13 file `tests/unit/test_msm_projection_weights.py`, and the RM25
file. Each run then restored the file, and the restored file matched its blob at that commit.

| mutation | file | failures in the RM27, RM13, and RM25 files | the tests that failed |
| --- | --- | --- | --- |
| H1. remove the `MSM.__post_init__` call | `src/cleverly/msm.py` | 18, 15, and 9 | the declaration witnesses |
| H2. remove the call in `_resolve_estimands_for_data` | `src/cleverly/estimators/tmle.py` | 7, 3, and 0 | the `fit`, `CausalStudy.estimate`, and `refit` entries, and the estimated declaration on the witness law |
| H3. remove the call in `_retarget_detailed` | `src/cleverly/estimators/tmle.py` | 2, 2, and 0 | `truncation_curve()` and `retarget()` on a legacy result |
| H2 and H3 together | `src/cleverly/estimators/tmle.py` | 10, 5, and 0 | the failures of H2 and H3, and `refit()` on a legacy result, which each of the two calls refuses alone |
| H4. remove the call in `LTMLE.fit` | `src/cleverly/longitudinal/estimator.py` | 2, 2, and 0 | the `LTMLE` fit witnesses |
| H5. move the `_freeze_msm` check after `MSMSet.evaluate` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1, 1, and 0 | the replay witness of each file, because the design and the weight run before the refusal |
| H6. `_design_kind` ignores the legacy rule | `src/cleverly/msm.py` | 5, 0, and 0 | the legacy `MSM.linear` controls |
| H7. the legacy rule reads `from_linear` | `src/cleverly/msm.py` | 2, 0, and 0 | the forged `from_linear=True` witness |
| H8. the legacy rule uses `isinstance` | `src/cleverly/msm.py` | 1, 0, and 0 | the subclass witness |
| H9. swap the undeclared and estimated texts | `src/cleverly/msm.py` | 20, 0, and 0 | each test that matches either text |
| H10. remove the callable check | `src/cleverly/msm.py` | 6, 0, and 0 | each design that is not callable, under each declaration |
| H11. `MSM.linear` stops passing `"known"` | `src/cleverly/msm.py` | 1, 0, and 0 | the witness that reads `design_kind` on `MSM.linear` |
| H12. the replay passes the literal `"known"`, and its check is removed | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 2, 1, and 0 | the M3 test, in the RM27 file |
| H13. `refuse_msm_functions` skips the weight checks | `src/cleverly/msm.py` | 0, 22, and 6 | the weight witnesses in the RM13 and RM25 files |
| H14a. the dose `replace` drops `design_kind` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1, 0, and 0 | `test_a_legacy_shorthand_dose_fit_replays`. The mutation survived at commit 268dc62 |
| H14b. the arm `replace` drops `design_kind` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1, 0, and 0 | `test_a_legacy_shorthand_fit_replays` |
| H15a. the dose `replace` drops `weights_kind` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1, 1, and 0 | `test_a_legacy_shorthand_dose_fit_replays` and `test_a_uniform_weight_dose_fit_replays`. The mutation survived at commit 268dc62 |
| H15b. the arm `replace` drops `weights_kind` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1, 1, and 0 | one RM27 replay test and `test_a_legacy_uniform_weight_fit_replays`. Two tests in `tests/unit/test_simulated_confounding_policies.py` fail too |

H14a and H15a survived the first pass. Each continuous replay test used a declared weight, which
`replace` copies on its own. No test replayed a dose fit with uniform weights or a legacy design.
H15a was also a gap of RM13 for current users. A declared uniform-weight dose fit would refuse on
replay if that call dropped `weights_kind`. Commit 28df2fb added a uniform-weight dose fit to the
RM13 and RM27 files, and both mutations now fail.

The review fixes changed what six of these rows test. The table gives each one at commit 5e055cc.
H2, H3, and H4 were run again there. Each run restored the file, and the restored file matched its
blob at HEAD.

| mutation | failures in the RM27, RM13, and RM25 files at commit 5e055cc | reading |
| --- | --- | --- |
| H2 | 0, 0, and 0 | the mutation survives. `MSMSet.evaluate` now refuses a point fit before its first learner, so no test separates the fit-layer call from the evaluator check. The call is still in `_resolve_estimands_for_data`. Commit 97cc57e added its witness, and the text after the review fixes gives the new counts |
| H3 | 2, 2, and 0 | unchanged. A sweep and a retarget reuse the stored arrays, so the call in `_retarget_detailed` is the only check on their path |
| H4 | 4, 2, and 0 | two more than the first pass. The two added failures are the malformed models at the `LTMLE.fit` entry (commit 9258d5a) |
| H5 | not run | it no longer applies, because commit e360555 removed the `_freeze_msm` check. Mutation e below deletes the check in `MSMSet.evaluate` instead |
| H11 | not run | it is inverted, because commit a237257 made `MSM.linear` leave `design_kind` as `None`. Mutation d below adds the keyword back |
| H12 | not run | it no longer applies as written, because the replay check that it removed is gone |

Commit 983b743 ran eight hand mutations across the three files. It ran each one before its test
changes, at commit 87f70ed, and after them. Each run restored the file to its blob. Two mutations
gained one failure. That failure is the control run of the shared-declaration witnesses.

| mutation | file | failures before the test changes | failures after the test changes |
| --- | --- | --- | --- |
| a. remove the `MSM.__post_init__` refusal | `src/cleverly/msm.py` | 43 | 44 |
| b. `_design_kind` reads `None` as known for any type | `src/cleverly/msm.py` | 17 | 18 |
| c. `_design_kind` keys on `isinstance` | `src/cleverly/msm.py` | 1 | 1 |
| d. `MSM.linear` writes `design_kind="known"` | `src/cleverly/msm.py` | 4 | 4 |
| e. remove the check in `MSMSet.evaluate` | `src/cleverly/msm.py` | 6 | 6 |
| f. remove the check in `RegimeSet.evaluate` | `src/cleverly/interventions/base.py` | 1 | 1 |
| g. the dose `replace` drops `design_kind` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 2 | 2 |
| h. the dose `replace` drops `weights_kind` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 2 | 2 |

Commit e360555 also deleted each new evaluator check by hand. The deletion failed 6 tests in
`MSMSet.evaluate`, 4 in `evaluate_regimen_msm`, 1 in `RegimeSet.evaluate`, and 2 in
`Stochastic.density`. Commit a237257 ran mutation d first, and it failed 4 tests, including the new
witness.

The delivery departed from the plan in four places. The table gives each one.

| deviation | reason |
| --- | --- |
| M4 uses an `MSM.linear` fit, not a declared written design | a declared design keeps `"known"` through `replace` on its own. So only a declaration that the replay supplies can be dropped, and the exact-type rule supplies it for `MSM.linear`. The delivery used a legacy fit. Since commit a237257, a new fit holds `None` too, and M4 uses one. The RM13 twin of the test keeps the uniform-weight fit |
| `gateaux_eif` in `tests/unit/_declaration_support.py` | it replaced the `beta_gateaux` function and the `eif` loop of the RM13 file, and the loop in `fixed_eif` of the RM25 file. The three witness files now share one contamination path. Commit 16b88e3 moved it to `tests/discrete_law.py` |
| the RM25 shared-declaration test | it covers three users, so its name changed from `..._fails_both_of_its_users` to `..._fails_each_of_its_users` |
| the notebook code cells | commit 268dc62 changed them, not the documentation commit. The fast suite runs the notebook, so an undeclared design would fail it. The markdown cells followed in commit 4b693b4 |

No study was regenerated. The generators of `tmle3_msm` and `ltmle_msm` gained only the
`design_kind="known"` keyword. `MSMSet.evaluate`, `evaluate_regimen_msm`, `solve_projection`, and
the influence code did not change in the delivery. The review added only a declaration check at
the start of the two evaluators, which returns `None` on a declared model. So each change is
result-neutral under
[what makes a study stale](development/method-benchmarking.md#what-makes-a-study-stale). No test
gates the module hashes of a manifest, and no container or R-runner file changed.

A bitwise check supports that judgment. For replicates 0 and 1 of each generator, a script took
the sha256 of the expression below.

```text
json.dumps(
    cleverly_rows(*draw_scenario(SCENARIO, PRIMARY_N, r), SCENARIO, r),
    sort_keys=True,
    default=repr,
)
```

The table gives the first 16 hexadecimal digits. They were identical at commits 67ab7fc, 268dc62,
and 5e055cc. The discrete-law oracle over the nine `msm` names hashed to `03b16bf86c5b9478` before
and after the `msm_beta` extension. Commit 16b88e3 records the same hash after its change.

| study | replicate 0 | replicate 1 |
| --- | --- | --- |
| `tmle3_msm`, point treatment, $n = 2000$ | `d7253da81059a0fd` | `bc47a47180d6d695` |
| `ltmle_msm`, longitudinal, $n = 2500$ | `734dafff18c8b28f` | `8fdeeca3cd3b7c91` |

#### RM27 review fixes

The review of this delivery found five defects, F1 to F5. Three more commits then removed repeated
test code, and one more added a witness. The table gives each commit.

| commit | what it changed |
| --- | --- |
| a237257, F1 | `MSM.linear` stopped writing `design_kind="known"`. Before it, `dataclasses.replace(MSM.linear(...), design=user_fn)` copied that field. So the package accepted a user design that nobody had declared, and the library had forged the declaration. The model now holds `None`, and the exact-type rule is the only way that its design reads as known. The undeclared message now ends "MSM.linear needs no declaration: the design it builds is known by construction." `test_replacing_the_shorthand_design_drops_its_declaration` is the witness |
| e360555, F2 | `MSMSet.evaluate` and `evaluate_regimen_msm` are public, and they ran the design and the weight with no check. A restored model with `design_kind=None` ran its design two times in `MSMSet.evaluate`, and nothing refused. Both now call `refuse_msm_functions` first. The RM25 evaluators had the same gap, so `RegimeSet.evaluate` and `Stochastic.density` now call `refuse_regime_densities` first. `TestTheEvaluatorsCheckFirst` in the RM13, RM25, and RM27 files holds the witnesses |
| 9258d5a, F3 | the fit-entry witnesses restored only `None` and `"estimated"`. `test_every_entry_refuses_a_malformed_model_before_any_call` restores `design_kind="Known"` or an array design at `fit`, `CausalStudy.estimate`, `refit`, and `LTMLE.fit`. Each entry raises `DataError` with zero learner calls and zero design calls |
| a94ce9f, F4 | the `_retarget_detailed` docstring said that an `MSM.linear` result "still recomputes". That holds only when its weight is uniform or declared `"known"`. The two `TMLE` docstrings now point at `refuse_msm_functions` and `refuse_regime_densities`. No code changed |
| 87f70ed, F5 | a comment in `_freeze_msm` said that the replay refuses an undeclared weight there. The comment now calls the `replace` calls a backstop to the check in `MSMSet.evaluate`. No code changed |
| 16b88e3 | `gateaux_eif` moved from `tests/unit/_declaration_support.py` to `tests/discrete_law.py`, and `eif` and `weighted_eif` now call it. The oracle hash `03b16bf86c5b9478` and a broader hash of the law, `fe962fd144da740d`, did not change |
| 983b743 | `tests/unit/_msm_declaration_support.py` holds the MSM builders of the RM13 and RM27 files. The three test files import support modules, and not one another. Each spy is a `Counter`. `CentredDesign` and `sample_centre` replace two fixed-centre classes. The RM13 file went from 55 to 52 tests, and the RM27 file from 87 to 88. Each of 32 exact-law arrays stayed equal |
| 5e055cc | `dose_msm` in `tests/e2e/test_ltmle_msm.py` builds the dose working model at eight sites, and `_fit_continuous` passes its weight keywords explicitly. Each construction builds the same `MSM` |
| 97cc57e | H2 survived at commit 5e055cc. `TestTheDeclarationRefusalComesFirst` in the RM27 file is now its witness, and mutation M7 is its control. The docstring of `_resolve_estimands_for_data` states the order of the refusals. The text below gives the probe |

F1 closes the hole that the review found. Before commit a237257, a declaration that the user never
wrote could reach a user design. F2 applies the same fix to the RM25 evaluators, because a restored
regime could reach `RegimeSet.evaluate` and `Stochastic.density` in the same way. No other public
evaluator runs a declared user function. `Static` and `Rule` hold none, and a user-written
`Intervention` carries no declaration, which RM28 holds. RM28 later added those declarations and
the evaluator checks of `Rule.density` and `DynamicRegimen.assignment` (commits 7c74997 and 7f87570).

Commit e360555 removed the explicit checks in `_freeze_msm` and `_freeze_regimes`. Each one stood
directly before an evaluator call that now runs the same check on the same argument. The commit
replaced both calls with `pass` while the evaluator checks were in place. The RM13, RM25, RM27, and
replay-policy files then gave the same 13 failures as with the calls in place. All 13 were mutation
controls that patched a module name. Every replay witness that counts user-function calls passed
with zero calls.

The `replace` calls in `_freeze_msm` stay. They pass the source declarations, so they refuse an
undeclared model if the evaluator check is removed. Mutation M3 depends on that backstop.

The evaluator check changed which check a point fit depends on. `MSMSet.evaluate` now refuses a
point fit before its first learner. So M2 removes the evaluator checks for `fit`,
`CausalStudy.estimate`, and `refit`, and only the `TMLE` check for `truncation_curve()` and
`retarget()`. H2 at commit 5e055cc shows that no test now fails without the call in
`_resolve_estimands_for_data`. H3 and H4 show that the calls in `_retarget_detailed` and `LTMLE.fit`
are still required.

Commit 97cc57e resolved H2. A probe replaced the name in `cleverly.estimators.tmle` with a no-op in
memory. It then ran each fit entry with and without the call. The table gives the result.

| fit | with the call | without the call |
| --- | --- | --- |
| `fit`, `CausalStudy.estimate`, `refit`, and `tmle(Y, A, W)`. Also `fit` with `intermediate=`, `repeats=3`, or `n_bootstrap=5` | the declaration refusal | the same exception and message, from `MSMSet.evaluate` |
| `fit` with `cv_evaluation=True` | the declaration refusal | the `cv_evaluation` refusal of an MSM, a `ValueError` |
| a cross-fitted `fit` of a continuous outcome with `q_bounds=None` | the declaration refusal | the outcome-scale refusal, a `CapabilityError` |
| a cross-fitted `fit` with `delta=` | the declaration refusal | the F21 missing-outcome refusal, a `CapabilityError` |

The probe used three restored models: an undeclared design, an estimated design, and an undeclared
weight. It also used four malformed models: `design_kind="Known"`, an array design,
`weights_kind="Known"`, and `weights_kind="estimated"` with no weight. Every model gave the row
above. No run called a learner, the design, or the weight.

So no path runs a learner or a user function before `MSMSet.evaluate`. The call decides only which
refusal answers. Each later refusal names a remedy, and no remedy lets an undeclared or estimated
design fit. The call stays, because removing it would make the refusal depend on the configuration.
Removing it would also have left each remaining check witnessed.
`TestTheDeclarationRefusalComesFirst` pins the order, and a declared model under each
configuration is its control.

The RM25 call at the same site already had a witness. A fit evaluates its regimes after its
learners, so that call is the only check before the first learner. The table gives each run at
commit 97cc57e. Each run restored the file, and the restored file matched its blob at HEAD.

| mutation | failures in the RM27, RM13, and RM25 files at commit 97cc57e | reading |
| --- | --- | --- |
| H2 | 6, 0, and 0 | the two restored states under the three configurations of `TestTheDeclarationRefusalComesFirst` |
| H3 | 2, 2, and 0 | unchanged |
| H4 | 4, 2, and 0 | unchanged |
| R2. remove the `refuse_regime_densities` call in `_resolve_estimands_for_data` | 0, 0, and 8 | each RM25 fit entry for both restored states, the subclass that skips the declaration, and a legacy regime at the fit |
| R3. remove the same call in `_retarget_detailed` | 0, 0, and 2 | `truncation_curve()` and `retarget()` on a legacy result |

At commit 5e055cc, `tests/unit/test_msm_design_declaration.py` holds 88 tests,
`tests/unit/test_msm_projection_weights.py` holds 52, and
`tests/unit/test_stochastic_regime_densities.py` holds 86. The full fast suite gave 11223 passed
and 87 skipped. At commit 97cc57e, the RM27 file holds 100 tests, and the full fast suite gave
11235 passed and 87 skipped.

### RM28. Declared densities of user-written interventions

The `Intervention` protocol is public, and the [interventions API page](api/interventions.md)
lists it first. Its docstring asks a user to implement `density` and carry a `name`
(`src/cleverly/interventions/base.py:84-87` at 40b8e643). `as_interventions` accepts any object that
does both (`src/cleverly/interventions/base.py:722` at 40b8e643). RM25 declares the density of
`Stochastic` alone. A user-written class carries no declaration, and its `density` receives the
`CausalData` of the fit. At 40b8e643 the docstring said so and named this row
(`src/cleverly/interventions/base.py:89-90`).

The 2026-09-23 plan for RM25 found this surface. A probe measured it on the RM25 witness law.
The exact standard error in this probe belongs to the population-mechanism odds-tilt target.
It does not assess inference for the realized learned-density target.

| probe | result |
| --- | --- |
| the RM25 witness law, in sample, with oracle nuisances. A user-written class has the `name` `"tilt"`. Its `density` returns the odds tilt at $\delta = 2$ of the sample treated share in each stratum. It reads `data.treatment` and `data.covariates` | `ey_regime[tilt]` reports its interval under the `influence_curve` status. Its standard error is 0.6226 of the population-law odds-tilt target's exact standard error. No warning and no message names the density |

The source review resolves the policy question. `Intervention.density`, `Rule.rule`, and callable
nodes of `DynamicRegimen` all admit functions learned from the analysis sample. A realized learned
policy has a different target from a population-indexed policy. Nordland and Holst (2026),
Section 3.4 and Algorithm 4, give an estimator and interval for a learned policy's value. The
CV-TMLE of van der Laan and Luedtke (2015), Section 7 and Appendix B, targets the average
value of rules learned on training folds. The current regime paths implement neither contract;
[RM30](#rm30-learned-policy-value-evaluation) records the published follow-up.

Luedtke and van der Laan (2016) show that the value of a
population optimal rule can have a regular first-order curve away from exceptional laws. That
result does not cover the sample-threshold rule below, which is not optimal on its witness law.
RM25's missing-term statement therefore cannot be copied to every learned rule. The
[source audit](references.md#point-treatment-and-stochastic-interventions) records these results.

The decision is to require a known-function declaration for all three surfaces. A function
trained independently of the analysis sample may be declared fixed for an evaluation conditional
on that training. A function learned from the analysis sample is refused by these generic paths
until the package implements a separate learned-policy estimand and inference method. This is a
conservative API decision. It does not claim that every learned rule's fixed-rule influence curve
is mathematically wrong. The declaration remains user-supplied because code cannot inspect a
closure.

Apply these corrections in RM28:

1. Require a `density_kind` protocol member on each user-written `Intervention`. Reuse the
   `FunctionDeclaration` states and refusal behavior of `Stochastic`.
2. Require the same three-state declaration for `Rule.rule` and each callable
   `DynamicRegimen` plan. Keep static labels exempt. Preserve the choice of inline rule syntax
   only if it can carry an explicit declaration.
3. Refuse undeclared and estimated functions before any learner call. Correct each API docstring,
   the [scope page](technical-reference/scope-and-refusals.md#wrong-by-construction), and the
   inference status of any restored result that cannot recompute.

The witnesses must fail when a component is wrong:

- a pre-fit test pins each refusal and its message, and a spy learner shows that no nuisance fit
  ran;
- a mutation that removes the refusal makes that test fail;
- the RM25 exact-law witness, on a user-written class that item 1 admits;
- a separate nonzero witness for a sample-learned `Rule` and a callable `DynamicRegimen` node;
- a control shows that a user-written class with a known density keeps its interval.

The 2026-09-23 plan fixes the decisions in the table below. The RM25 and RM27 patterns supply each
one, unless the row says otherwise.

| part | decision |
| --- | --- |
| rule declaration | `Rule` and `DynamicRegimen` each gain a last field `rule_kind`: `"known"`, `"estimated"`, or `None`, with the default `None`. `Rule(rule, name)` keeps its positional order. The default is a plain class attribute, so an old pickle loads with `None` |
| one declaration per regimen | the `rule_kind` of a `DynamicRegimen` covers every callable node of its plan. A plan with no callable node is exempt from the refusal. An unknown value is still refused |
| protocol member | `Intervention` gains a read-only `density_kind`. `Stochastic` keeps its field, and `Rule` returns its `rule_kind`. `Static` returns `"known"` only when its exact type is `Static`, by the RM27 exact-type rule. The replay's `_FrozenRegime` gains a `density_kind` field |
| admission | `as_interventions` stops using the runtime protocol check. A string is a level. An object with a callable `density` is taken as is, and one with no `name` raises `DataError`. Any other callable raises `DataError`. Every other value becomes `Static` |
| shared declarations | `_RULE_DECLARATION` serves `Rule` and `DynamicRegimen`. `_INTERVENTION_DECLARATION` serves a user-written class. Both sit beside `_DENSITY_DECLARATION` in `src/cleverly/interventions/base.py`, and each estimated text names the pathwise derivative |
| point check | `refuse_regime_densities` keeps its name and checks every item. A `Stochastic` meets the density declaration, a `Rule` meets the rule declaration, and any other object meets the intervention declaration |
| longitudinal check | a new `refuse_regimen_rules` in `cleverly.longitudinal.regimen` reads the raw `regimens=` value. It checks each plan that holds a callable node |
| check sites | `Rule.__post_init__`, `Rule.density`, and the three RM25 sites for a point fit. `DynamicRegimen.__post_init__`, `DynamicRegimen.assignment`, `LTMLE.fit` before `_prepare`, and the first statement of `longitudinal_truncation_curve` for a longitudinal fit |
| no constructor check | `as_interventions` and `TMLE.__init__` check no declaration. So the RM25 mutation of the fit-layer check still finds a fit witness that fails |
| carry | `_resolve_one` keeps `rule_kind` when it rebuilds a `DynamicRegimen`. The replay builds each `_FrozenRegime` with the `density_kind` of its source, and never with the literal `"known"` |
| inline syntax | a callable written inline in a `regimens=` mapping carries no declaration, and the fit refuses it. The message asks for `DynamicRegimen(label, plan, rule_kind='known')`. A mapping still takes static plans and declared `DynamicRegimen` values. No estimator keyword declares a rule |
| restored results | a new non-inferential status, `undeclared_function_plugin`, is third in the precedence. A restored `TMLE` or `LTMLE` result takes it when a rule, a user-written class, a `Stochastic` density, or a written MSM function is not declared `"known"`. The first delivery gave the MSM case to a restored `TMLE` result only. The [later review](#rm28-review-saved-longitudinal-msm-provenance-and-malformed-plans) closes that gap |
| what the status keeps | the point estimates stand, and `ci`, `pvalue` and `std_error` refuse. The plug-in standard error remains a diagnostic. This reverses the RM13, RM25 and RM27 decisions that a restored interval answers as saved |
| remedy for a user-written tilt | the estimated text sends the user to `TMLE(incremental=...)` for the population odds tilt of the mechanism |
| a false declaration | not detected. A rule computed from the sample and declared `"known"` fits. The witness below measures what the false statement costs |
| `dataclasses.replace(rule, rule=new)` | it keeps the old `rule_kind`, as RM27 keeps `design_kind`. The reference records this limit |

The plan probed six more surfaces on 40b8e643. The table gives the decision for each.

| surface | probe | decision |
| --- | --- | --- |
| a bare callable in `interventions=` | `as_interventions` wraps it as `Static` with the name `"always <function ...>"`. The fit then raises `DataError`, because the function is not a level of `A` | in this row. `as_interventions` refuses it with a message that names `Rule(rule, name, rule_kind='known')` |
| a restored `Rule` result | a module-level rule survives a pickle round trip. The result keeps its three intervals under `influence_curve`, and `truncation_curve` recomputes | in this row. The result takes the new status, and each recomputation refuses |
| an inline rule in `LTMLE` | a `regimens=` mapping with a lambda fits under `influence_curve`. A module-level rule survives a pickle round trip, keeps its intervals, and recomputes `longitudinal_truncation_curve` | in this row. The fit refuses the inline rule, and a restored result takes the new status |
| capability rows of a restored result | on a restored `Rule` result, the `truncation_curve` and `refute` rows read available. On a restored `LTMLE` result, the `truncation_curve` row reads available | not in the first delivery. The calls then refused. [RM23](#rm23-capability-rows-that-read-available-and-then-refuse) records that finding, and the [later review](#rm28-review-saved-longitudinal-msm-provenance-and-malformed-plans) corrects these declaration-specific rows |
| `CTMLE` | `CTMLE` runs its own estimand checks before the inherited declaration check (`src/cleverly/estimators/ctmle.py:1186`). A regime fit meets a C-TMLE refusal first, as under RM25 | not in this row. Every C-TMLE status is already non-inferential, so `CTMLE._inference_status` does not change |
| `dataclasses.replace` | `replace` copies every field that the call does not name | not in this row. `replace` keeps the declaration, and the reference records the limit |

The plan declares three exact-law witnesses here, before their tests land. The threshold law
uses a continuous covariate, which gives the policy value a nonzero derivative through its
threshold. On finite support, a threshold between support points stays locally constant and its
derivative is zero. A threshold at a support point can instead be discontinuous. The witness
rule is not optimal, so the optimal-rule result of Luedtke and van der Laan (2016) does not
establish its curve.

| part | the point threshold witness | the regimen threshold witness |
| --- | --- | --- |
| law | $W \sim U(0, 1)$, $A \mid W \sim \mathrm{Bern}(1/2)$, $Y = 3W + A + \varepsilon$ | $W \sim U(0, 1)$, $A_1$ and $A_2 \sim \mathrm{Bern}(1/2)$, $Y = 3W + A_2 + \varepsilon$ |
| rule | `Rule` with $d(w) = 1\{w \le c\}$, where $c$ is the sample mean of $W$, declared `"known"` | the regimen $(1, d)$ as a `DynamicRegimen` declared `"known"` |
| sample | $W_j = (j - 1/2)/200$. Four rows at each $W_j$: $A \in \{0, 1\}$ and $\varepsilon = \pm 1/4$. 800 rows | eight rows at each $W_j$: $A_1$, $A_2$ and $\varepsilon$. 1600 rows |
| fit | in sample, with `tests.discrete_law_longitudinal.CellMeans` as each learner | the same, with `n_folds=1` |
| omitted term | $T = W - c$, because $\partial\psi/\partial c = f(c)\{Q(1, c) - Q(0, c)\} = 1$ | the same $T$. The node-one term is zero |
| exact moments | $E[D^2] = 3/8$, $E[T^2] = 1/12$, $E[DT] = 1/8$ | $E[D^2] = 1/2$, and the same $T$ moments |
| exact ratio | $3/\sqrt{17} = 0.727607$ | $\sqrt{3/5} = 0.774597$ |

Here $D$ is the fixed-rule curve and $T$ the omitted term. The positive cross moment means that
the reported standard error understates the exact one. Each sample ratio is
$\sqrt{E_n[D^2] / E_n[(D + T)^2]}$. The third witness is the RM25 witness on a user-written class
`DataTilt`. Its `density` returns the odds tilt at $\delta = 2$ of the sample treated share in each
stratum, and it declares `density_kind = "known"`. The plan measured each value on 40b8e643, with
the `Rule` and the inline regimen `{"x": (1, d)}` in place of the declared objects.

| assertion | tolerance or bound | point | regimen | `DataTilt` |
| --- | --- | --- | --- | --- |
| $c = 0.5$, and the learned rule equals the fixed threshold 0.5 on each row | exact | yes | yes | not applicable |
| $\psi = 2$ | 1e-12 | 4.4e-16 | 0 | not applicable |
| each targeting coefficient | exactly 0 | 0 | 0 at both nodes | 0 |
| the reported curve minus the fixed-rule curve, row by row | 1e-10 | 1.6e-15 | 3.6e-15 | not applicable |
| $E_n[T^2]$, the nonzero witness | above 0.05 | 0.08333125 | 0.08333125 | not applicable |
| $E_n[DT]$, recorded | none | 0.12499375 | 0.12499375 | not applicable |
| the ratio, the witness | pinned to 1e-4, and below the bound | 0.727606, pin 0.7276, bound 0.8 | 0.774598, pin 0.7746, bound 0.85 | 0.622637, pin 0.6226, bound 0.7 |
| the curve against the `Stochastic(SampleTilt)` curve of RM25 | 1e-12 | not applicable | not applicable | 0 |
| `"estimated"` on this law | refused before any learner | none yet | none yet | none yet |

The sample moments differ from the exact ones only through $\mathrm{Var}_n(W) = 1/12 - 1/(12 \cdot
200^2)$. So $E_n[DT] = 1/8 - 1/(4 \cdot 200^2)$, and each ratio moves by less than 3e-5. The
measured moves are 1.1e-6 and 9.7e-7. Each bound sits at least 0.07 above its measured ratio, so
it pins no digit. A curve that carried $T$ would read 1.

Three controls keep their intervals. The fixed threshold 0.5, declared `"known"`, keeps
`influence_curve` and a finite `ci`. Its curve is bitwise equal to the witness curve, for the
`Rule` and for the regimen. A user-written class that returns the tilt of the true mechanism keeps
its interval, and its curve equals the RM25 fixed-density curve to 1e-12.

The plan commits ten mutations. Each one replaces a function through `monkeypatch`, except R10.

| mutation | the witness that must fail |
| --- | --- |
| R1. `refuse_regime_densities` in `cleverly.interventions.base` becomes a no-op | the `Rule` declaration witnesses, and the evaluator witnesses of a `Rule` and a user-written class |
| R2. the same in `cleverly.estimators.tmle` | the fit witnesses of a restored `Rule` and an undeclared class, where `NeverFit` records fits, and the recomputations of a legacy result |
| R3. `refuse_regimen_rules` in `cleverly.longitudinal.regimen` becomes a no-op | the `DynamicRegimen` declaration and evaluator witnesses |
| R4. the same in `cleverly.longitudinal.estimator` | the order witnesses, where a later refusal answers, and the legacy truncation witness. With R3 it also fails the `LTMLE` fit witnesses |
| R5. `_resolve_one` drops `rule_kind` | the witness that a declared `DynamicRegimen` in a mapping keeps its declaration |
| R6. the replay passes `None` for `density_kind` | the replay witnesses |
| R7. `FunctionDeclaration.refuse` becomes a no-op | each row of the extended RM25 table of shared refusals |
| R8. each status predicate returns `"influence_curve"` | the witnesses of a restored result's status |
| R9. `as_interventions` uses the protocol check again | a class with no `density_kind` becomes `Static`, and its refusal witness fails |
| R10. the oracle freezes the threshold, for the point and the regimen | the ratio reads 1 and fails its bound |

The plan also runs fourteen mutations by hand, H1 to H14, with a hash-checked backup of each file.
The delivery record gives each one and its counts. The plan regenerates no study. A bitwise
check hashes the replicate 0 and 1 rows of twelve canonical generators and eleven property
generators, at 40b8e643 and after the change.

#### RM28 delivery

All three corrections shipped, with the decisions that commit d23ab9f recorded. Commit 65f407c first
moved the RM25 tilt law into `tests/unit/_tilt_law_support.py`. The RM25, RM13 and RM27 files
passed the same 86, 52 and 100 tests before and after that move. The RM25 and RM27 patterns supply
each part, unless the row says otherwise. Each row ends with the commit that shipped it.

| part | what shipped |
| --- | --- |
| rule declaration | `Rule` gains a last field `rule_kind`: `"known"`, `"estimated"`, or `None`. Its default `None` is a plain class attribute, so an old pickle loads undeclared. `Rule(rule, name)` keeps its positional order. `Rule.__post_init__` and `Rule.density` run the check. Commit 7c74997 |
| protocol member | `Intervention` gains a read-only `density_kind`. `Stochastic` keeps its field, and `Rule.density_kind` returns `rule_kind`. `Static.density_kind` reads `"known"` only when the exact type is `Static`, so a subclass reads `None`. Commit 7c74997 |
| admission | `as_interventions` takes any object with a callable `density` as is, and it runs no runtime protocol check. It raises `DataError` for such an object with no `name`, and for a bare callable. Commit 7c74997 |
| point check | `refuse_regime_densities` checks each item. A `Stochastic` meets `_DENSITY_DECLARATION`, and a `Rule` meets `_RULE_DECLARATION`. Every other object needs a `density` method, and it meets `_INTERVENTION_DECLARATION`. Commit 7c74997 |
| shared declarations | `_RULE_DECLARATION` and `_INTERVENTION_DECLARATION` are the fourth and fifth users of `FunctionDeclaration`. Each estimated text names the pathwise derivative. `DynamicRegimen` uses `_RULE_DECLARATION` too. Commits 7c74997 and 7f87570 |
| replay carry | `_freeze_regimes` builds each `_FrozenRegime` with the `density_kind` of its source, and never with the literal `"known"`. Commit 7c74997 |
| regimen declaration | `DynamicRegimen` gains a last field `rule_kind`. A plan of labels alone is exempt from the refusal, and its value is still checked. Commit 7f87570 |
| longitudinal check | `refuse_regimen_rules` in `cleverly.longitudinal.regimen` reads the raw `regimens=` value. `DynamicRegimen.__post_init__`, `DynamicRegimen.assignment`, `LTMLE.fit` before `_prepare`, and the first statement of `longitudinal_truncation_curve` call it. `_resolve_one` carries `rule_kind`. Commit 7f87570 |
| inline callables | a callable in a `regimens=` mapping is refused, bare or in a tuple. The broadcast form `{label: fn}` is refused too, and a user writes `(fn,) * T` inside a `DynamicRegimen`. The message names `DynamicRegimen(label, plan, rule_kind='known')`. Commit 7f87570 |
| call sites | each `Rule` and each callable plan in `tests`, the documentation fences, and both notebooks declares `"known"`. `tests/studies/canonical_ltmle.py` gains `declared_regimens(spec)` for the oracle plans. No study's required-module set changed. Commits 7c74997 and 7f87570 |
| restored results | a new status, `undeclared_function_plugin`, is third in the precedence, after the two collaborative statuses. `TMLEResult` and `LongitudinalResult` re-stamp on load. The point estimates stand, and `ci`, `pvalue` and `std_error` refuse. The old interval remains as `plugin_interval`. The [status table](technical-reference/inference.md#inference-status) gains its row. Commit ed8e73e |
| one point refusal | `TMLE._refuse_undeclared_functions` runs the MSM check and the regime check. `fit`, `retarget`, the status predicate, and `variable_importance` share it. Commit ed8e73e |
| longitudinal status | `_declared_regimen_status` runs `refuse_regimen_rules` on the regimens of the fit, of a truncation refit, or of the restored configuration. Commit ed8e73e. Commit 0d3ce32 moved the rule that reads a refusal as the status into `cleverly._declarations.declaration_status`, which this predicate and `TMLE._declared_function_status` both call |
| legacy tests | the RM13, RM25 and RM27 legacy tests now assert that the stored interval becomes a diagnostic. Each class is now `TestALegacyResultKeepsItsPointEstimatesAndRefusesARecomputation`. Commit ed8e73e |
| reference | the [scope page](technical-reference/scope-and-refusals.md#wrong-by-construction) marks both rows refused, and five rows now share the pathwise mechanism. The rule row claims that mechanism for a population-indexed rule only. The architecture invariants, the point-treatment, longitudinal and MSM references, the user guide, the references page, the evidence table, and both notebooks describe the declaration. `DynamicRegimen` gains an API page. Commit faa0a5f |
| a false declaration | not detected. A rule computed from the sample and declared `"known"` fits. The witnesses below measure what the false statement costs |
| `dataclasses.replace(rule, rule=new)` | it keeps the old `rule_kind`, as RM27 keeps `design_kind`. `test_replacing_the_function_keeps_the_declaration` pins the limit. Commit 7c74997 |

The table gives the outcome of each surface that the plan probed, and of two that the delivery
found.

| surface | outcome |
| --- | --- |
| a bare callable in `interventions=` | refused with `DataError`. The message names `Rule(rule, name, rule_kind='known')`. Commit 7c74997 |
| a restored `Rule` result | it takes `undeclared_function_plugin`. `truncation_curve()`, `retarget()` and `refit()` refuse. Commits 7c74997 and ed8e73e |
| an inline rule in `LTMLE` | the fit refuses it. A restored result with an undeclared node takes the status, and `longitudinal_truncation_curve` refuses. Commits 7f87570 and ed8e73e |
| capability rows of a restored result | not in the first delivery. The rows read `available`, and the call then refused. [RM23](#rm23-capability-rows-that-read-available-and-then-refuse) records the finding. The later review corrects the declaration-specific rows |
| `CTMLE` | not in this row. `CTMLE._check_estimands` refuses every regime estimand first, so no C-TMLE regime result exists |
| `dataclasses.replace` | not in this row. `replace` keeps the declaration, and the reference records the limit |
| the MSM of a restored `LTMLE` result, found by the first delivery | the result kept only the evaluated arrays of its MSM, so the status could not read the source declaration. Commit ed8e73e recorded that limit. The [later review](#rm28-review-saved-longitudinal-msm-provenance-and-malformed-plans) adds a marker at evaluation and treats an old result with no marker conservatively |
| `variable_importance`, found by the delivery | it asked for the status of a live template before any refusal ran. So a live undeclared estimator met the restored-result reason. It now runs the declaration refusals first, and `test_a_live_configuration_meets_the_declaration_refusal` pins that. Commit ed8e73e |

The tests measured each witness on the delivered code. A 2026-09-24 probe on commit faa0a5f
measured the same values. The ratio of each witness is the reported standard error over the exact
one. Commit 16f9302 changed where the threshold witnesses take the exact one. `exact_rule_curve`
and `exact_regimen_curve` build $D + T$ from the closed forms alone, and never read the reported
curve. `plugin_se` writes out $\sqrt{\mathrm{Var}_n(\cdot) / n}$ with `ddof=1`. The measured
ratios did not move.

| assertion | tolerance or bound | point `Rule` | regimen $(1, d)$ | `DataTilt` |
| --- | --- | --- | --- | --- |
| $c = 0.5$, and the learned rule equals the fixed threshold 0.5 on each row | exact | yes | yes | not applicable |
| $\psi = 2$ | 1e-12 | 4.4e-16 | 0 | not applicable |
| each targeting coefficient | exactly 0 | 0, and `fluctuations` holds the `"regime"` step alone | 0 at both nodes | not tested |
| the reported curve minus the fixed-rule curve, row by row | 1e-10 | 1.6e-15 | 3.6e-15 | not applicable |
| $\partial\psi/\partial c$ by quadrature | 1e-8 | 1 | not tested | not applicable |
| $E_n[T^2]$, the nonzero witness | above 0.05 | 0.08333125 | 0.08333125 | not applicable |
| $E_n[DT]$ against $1/8 - 1/(4 \cdot 200^2)$ | 1e-12 | 0.12499375 | 0.12499375 | not applicable |
| the reported `std_error` against `plugin_se` of the reported curve | relative 1e-12 | 0.021663638 | 0.017682865 | not tested |
| the ratio, the witness | pinned to 1e-4, within 3e-5 of the exact ratio, and below the bound | 0.7276058, pin 0.7276, exact 0.7276069, bound 0.8 | 0.7745976, pin 0.7746, exact 0.7745967, bound 0.85 | 0.6226371, pin 0.6226, bound 0.7. No exact ratio is pinned |
| the curve against the `Stochastic(SampleTilt)` curve of RM25 | 1e-12 | not applicable | not applicable | 0 |
| the ratio with an oracle that freezes the threshold or the mechanism, mutation R10. For a threshold, the oracle is the closed-form fixed-rule curve | fails the bound | 1 | 1 | 1 |
| the ratio of the reported curve with $T$ added, mutation R10 | fails the bound | 1 | 1 | not applicable |
| the ratio of the reported curve with $T$ subtracted, mutation R10 | passes the bound and fails the pin | $\sqrt{5/17} = 0.5423$ | $\sqrt{2/5} = 0.6325$ | not applicable |
| `"estimated"` on this law | refused before any learner | refused | refused | refused |

Three controls keep `influence_curve` and a finite `ci`. The fixed threshold 0.5, declared
`"known"`, reports a curve bitwise equal to the witness curve, for the `Rule` and for the regimen.
The class `KnownUserTilt` returns the tilt of the true mechanism. Its curve equals the RM25 fixed-density
curve to 3.3e-16, against the tolerance 1e-12.

The row named five witnesses. The table gives the state of each. They are in
`tests/unit/test_rule_and_intervention_declarations.py`, the rule file, and
`tests/unit/test_regimen_rule_declarations.py`, the regimen file.

| witness | state |
| --- | --- |
| 1. a pre-fit test pins each refusal and its message, and a spy learner shows that no nuisance fit ran | delivered. `TestTheFitRefusesAnUndeclaredIntervention`, `TestTheFitRefusesARestoredRule` and `TestTheFitRefusesARestoredRegimen` run `fit`, `CausalStudy.estimate`, and `refit` or `ltmle`. `NeverFit.calls` is 0, and a `Counter` spy records no rule or density call. A point fit evaluates its densities only after its first learner. So at a point entry, the spy adds no check beyond `NeverFit`. A control declared `"known"` reaches the first learner. `TestTheDeclarationRefusalComesFirst` and `TestTheEvaluatorsCheckFirst` pin the order |
| 2. a mutation that removes the refusal makes that test fail | delivered. The files commit mutations R1 to R10, and seventeen more ran by hand. The tables below give them |
| 3. the RM25 exact-law witness on a user-written class | delivered in `TestAUserWrittenTiltUnderstatesTheVariance`. The ratio reads 0.6226 against the bound of 0.7 |
| 4. a separate nonzero witness for a sample-learned `Rule` and a callable `DynamicRegimen` node | delivered in `TestASampleThresholdRuleMisstatesTheVariance` and `TestASampleThresholdNodeMisstatesTheVariance`. The ratios read 0.7276 and 0.7746 against the bounds of 0.8 and 0.85 |
| 5. a control shows that a user-written class with a known density keeps its interval | delivered in `TestAKnownUserWrittenDensityKeepsItsInterval`. `TestAKnownRuleKeepsItsInterval` and `TestAKnownNodeKeepsItsInterval` add the fixed threshold |

Each committed mutation replaces a function through `monkeypatch`, except R10, which mutates the
test oracle. R7 is an RM25 test that this row extends.

| mutation | the test that fails |
| --- | --- |
| R1. `refuse_regime_densities` in `cleverly.interventions.base` becomes a no-op | `test_removing_the_check_fails_every_declaration_and_evaluator_witness`, for a `Rule` and a user-written class |
| R2. the same in `cleverly.estimators.tmle` | `test_removing_the_fit_layer_check_fails_the_fit_witnesses`, where `NeverFit` records fits. Also `..._fails_the_recomputation_refusal` and `..._alone_lets_the_later_refusal_answer` |
| R3. `refuse_regimen_rules` in `cleverly.longitudinal.regimen` becomes a no-op | `test_removing_the_regimen_check_fails_every_declaration_and_evaluator_witness` |
| R4. the same in `cleverly.longitudinal.estimator` | `test_removing_the_fit_check_alone_lets_a_data_refusal_answer` and `test_removing_the_fit_check_fails_the_legacy_truncation_witness`. `..._alone_still_refuses_before_any_learner` is its control |
| R4 with R3 | `test_removing_both_checks_fails_the_fit_witnesses` and `test_removing_both_checks_lets_every_later_refusal_answer` |
| R5. `_resolve_one` drops `rule_kind` | `test_dropping_the_resolve_carry_fails_the_declared_mapping_witness`, through the helper `drop_the_resolve_carry` |
| R6. the replay passes `None` for `density_kind` | `test_dropping_the_replay_carry_fails_the_replay_witness`, through the helper `drop_the_replay_carry`, and `test_a_replay_that_drops_the_declaration_refuses` |
| R7. `FunctionDeclaration.refuse` becomes a no-op | `test_removing_the_shared_declaration_fails_each_of_its_users` in the RM25 file. Its table `SHARED_REFUSALS` grows from 8 to 16 witnesses |
| R8. each status predicate returns `"influence_curve"` | `test_a_status_that_ignores_the_declarations_fails_the_restored_witnesses` and `test_a_status_that_ignores_the_regimens_fails_the_restored_witnesses` |
| R9. `as_interventions` uses the protocol check again | `test_the_protocol_check_admits_a_bare_class_as_a_level`. `BareTilt` becomes `Static`, and a learner runs |
| R10. the oracle freezes the threshold or the mechanism | `test_mutation_a_frozen_threshold_oracle_loses_the_witness`, in each file, and `test_mutation_a_frozen_oracle_loses_the_witness` for `DataTilt`. Each ratio reads 1 and fails its bound. Commit 16f9302 added `test_mutation_a_curve_with_the_term_fails_the_witness` in each file. A curve with $T$ added reads 1 and fails the bound, and a curve with $T$ subtracted fails the pin |

`test_every_site_calls_the_one_refusal` in each file checks that the fit layer holds the function
that the evaluators call. It also checks that `DynamicRegimen` and `Rule` share one
`_RULE_DECLARATION`.

Seventeen mutations ran by hand. Each run copied the source file to a backup with its sha256, and
applied one mutation. It ran the declaration files with the nearby regime, replay and status files.
Each run then restored the file, and the restored file matched its hash. A file that the table does
not name failed no test.

The plan named H11 and H12 for the two text swaps, and they ran as H11a and H11b. The delivery
added H13b, H15 and H16.

| mutation | file | failures |
| --- | --- | --- |
| H1. remove the `Rule.__post_init__` check | `src/cleverly/interventions/base.py` | 16 in the rule file, 4 in the RM25 file |
| H2. remove the `Rule` branch, so a `Rule` meets the generic branch | `src/cleverly/interventions/base.py` | 35 in the rule file, 4 in the RM25 file |
| H3. the generic branch passes every other object | `src/cleverly/interventions/base.py` | 22 in the rule file, 5 in the RM25 file |
| H4. `Static.density_kind` reads `"known"` for any subclass | `src/cleverly/interventions/base.py` | 1 in the rule file |
| H5. the replay passes the literal `"known"` | `src/cleverly/sensitivity/_simulated_confounding_fixed.py` | 1 in the rule file, 1 in the RM25 file |
| H6. remove the `LTMLE.fit` check | `src/cleverly/longitudinal/estimator.py` | 1 in the regimen file: the missing-column order witness |
| H7. remove the `longitudinal_truncation_curve` check | `src/cleverly/longitudinal/estimator.py` | 3 in the regimen file |
| H8. remove the `DynamicRegimen.__post_init__` check | `src/cleverly/longitudinal/regimen.py` | 16 in the regimen file, 4 in the RM25 file |
| H9. remove the `DynamicRegimen.assignment` check | `src/cleverly/longitudinal/regimen.py` | 4 in the regimen file |
| H10. a `DynamicRegimen` of labels loses its exemption | `src/cleverly/longitudinal/regimen.py` | 2 in the regimen file |
| H11a. swap the undeclared and estimated texts of `_RULE_DECLARATION` | `src/cleverly/interventions/base.py` | 24 in the rule file and 1 in the RM25 file at commit 7c74997. At commit 7f87570, 43 in the regimen file too |
| H11b. swap the same texts of `_INTERVENTION_DECLARATION` | `src/cleverly/interventions/base.py` | 20 in the rule file, 2 in the RM25 file |
| H13. the longitudinal status ignores the regimens | `src/cleverly/longitudinal/estimator.py` | 4 in the regimen file |
| H13b. the longitudinal re-stamp passes no regimens | `src/cleverly/longitudinal/estimator.py` | 4 in the regimen file |
| H14. the new status moves first in the precedence | `src/cleverly/_inference_status.py` | 2 in `tests/unit/test_inference_status_registry.py` |
| H15. the point status predicate ignores the MSM | `src/cleverly/estimators/tmle.py` | 1 in the RM13 file, 1 in the RM27 file |
| H16. the point status predicate ignores the interventions | `src/cleverly/estimators/tmle.py` | 5 in the rule file, 1 in the RM25 file |

Commit ed8e73e also ran an instrument that is not a mutation. An assertion at each live stamp site
raised if a new fit took `undeclared_function_plugin`. The full fast suite raised it only in
`tests/unit/test_inference_status_reach.py`, which forces each status on the hook. So no real fit
reached the status.

The delivery departed from the plan in six places. The table gives each one.

| deviation | reason |
| --- | --- |
| the `REGIMEN_SPEC` oracle constants stay plain tables | `tests/unit/test_oracle_independence.py` forbids a `cleverly` import in `tests/discrete_law*.py`. So `declared_regimens(spec)` in `tests/studies/canonical_ltmle.py` wraps each plan for its consumers. Every study that fits an oracle plan already imports that module (commit 7f87570) |
| H6 fails one test only | `resolve_regimens` rebuilds each regimen, and `DynamicRegimen.__post_init__` refuses it before any learner. So the `LTMLE.fit` check decides only that the declaration refusal comes before `_prepare`. The missing-column order witness pins it, and R4 alone is its control |
| a restored `Rule` at the replay refit meets the text of a user-written class | the replay refit sees a `_FrozenRegime`, which the generic branch checks. Only a mutation that removes the evaluator check opens this path. `test_removing_the_evaluator_check_still_refuses_through_the_carry` pins the text |
| the point order witnesses use two later refusals, not three as in RM27 | a declared regime fit with `cv_evaluation=True` reaches its first learner. So no refusal follows the declaration check on that path |
| the MSM of a restored `LTMLE` result | the first delivery gave the status to a written MSM function on every restored result. A `LongitudinalResult` kept no `MSM` object, so the status covered the `TMLE` MSM alone. The later review records the persisted marker needed to cover the longitudinal result |
| `variable_importance` | the plan did not name it. It read the status before any refusal ran, so commit ed8e73e made it run the declaration refusals first |

This delivery regenerated no study. Each check returns `None` on a declared function, and the call
sites gained only the `"known"` declaration. After their checks, `RegimeSet.evaluate`,
`resolve_plans`, the targeting code, and the influence code compute the same numbers. So each
commit is result-neutral under
[what makes a study stale](development/method-benchmarking.md#what-makes-a-study-stale). No test
gates the module hashes of a manifest. No container or R-runner file changed, so
`tests/canonical/provenance-revisions.md` needs no row.

A bitwise check supports that judgment. For replicates 0 and 1, a script took the sha256 of the
RM27 expression for each canonical generator. For each property generator, it hashed the output of
each row function on every replicate 0 or 1 payload. The 316 hashes were identical at 40b8e643,
at commit 7f87570, and at commit ed8e73e. Commit faa0a5f changed only docstrings in the source,
and this record changes only the roadmap. The review fixes below changed source files, and the
316 hashes were identical again at commit 75e86be.

| generators | modules |
| --- | --- |
| canonical, 24 hashes | `canonical_deterministic_regimes`, `canonical_ltmle`, `canonical_ltmle_crossfit`, `canonical_ltmle_survival`, `canonical_ltmle_survival_crossfit`, `canonical_ltmle_competing`, `canonical_ltmle_competing_crossfit`, `canonical_longitudinal_msm`, `canonical_weighted_ltmle`, `canonical_weighted_ltmle_crossfit`, `canonical_categorical_ltmle`, and `canonical_categorical_ltmle_crossfit` |
| property, 292 hashes | `ltmle_properties`, `ltmle_crossfit_properties`, `ltmle_survival_properties`, `ltmle_survival_crossfit_properties`, `ltmle_competing_properties`, `ltmle_competing_crossfit_properties`, `longitudinal_msm_properties`, `weighted_ltmle_properties`, `weighted_ltmle_crossfit_properties`, `categorical_ltmle_properties`, and `categorical_ltmle_crossfit_properties` |

At commit faa0a5f the rule file holds 130 tests and the regimen file 96. The RM25 file holds 98,
the RM13 file 53, and the RM27 file 101. The RM13 and RM27 files each gained a legacy control that
keeps its interval. The full fast suite gave 11237 passed and 87 skipped at 40b8e643, and 11506
passed and 87 skipped at commit faa0a5f and at this record.

#### RM28 review fixes

The review of this delivery found five defects in the source, S1 to S5, and seven in the tests,
T1 to T7. A review of the documentation then found twelve more, D1 to D12, and five sentences
that broke the writing standard. The table gives each commit. "This record" is the commit that
adds this table.

| commit | what it changed |
| --- | --- |
| c11564d, S1 | a `DynamicRegimen` kept its plan as given. So `DynamicRegimen("x", (n for n in [1, d]))` with no `rule_kind` built: the check skipped the iterator, and `assignment` later called the undeclared rule. `__post_init__` now reads an iterator into a tuple and stores the plan as a tuple before any check. A plan that is one label or one callable raises `DataError`: "regimen ... needs a plan with one entry per treatment node ... Write one rule for every node as (rule,) * T". `TestAPlanIsReadOnce` in the regimen file holds the witnesses |
| c11564d, S2 | three helpers read the shape of a plan, and they disagreed about an iterator. An iterator plan in a `regimens=` mapping passed the `LTMLE.fit` check and was refused only after `_prepare`. A second fit of the same mapping read an empty plan. `_plan_nodes` is now the one reader of a plan's shape. It raises `DataError` for an iterator, unread: "the plan of regimen ... is an iterator ... Pass the plan as a tuple, or as a DynamicRegimen, which stores its plan". A static generator plan that fitted once now refuses |
| 0d3ce32, S3 | `TMLE._declared_function_status` and the longitudinal `_declared_regimen_status` each held the rule that reads a refusal as `undeclared_function_plugin`. `cleverly._declarations.declaration_status` now holds it once, and both predicates call it. It reads `CapabilityError` and `DataError` as the status, and any other error propagates. `TestTheStatusPredicateIsShared` is the witness |
| b563720, S4 | a subclass of `Static` reads `density_kind` `None`, so its fit meets `_UNDECLARED_INTERVENTION`. That text ended "Static, Rule and Stochastic carry their own declarations", which such a subclass reads as a contradiction. The text and the `Intervention` docstring now add "A subclass of Static can override density, so it declares density_kind itself." `TestTheFitRefusesAStaticSubclass` drives a counting subclass through `fit`, `CausalStudy.estimate`, `refit` and `tmle` |
| b563720, S5 | the assessment note of `undeclared_function_plugin` said that RM28 "is the condition that reopens it", although RM28 is delivered. Its reason listed an MSM function for every restored result, although a `LongitudinalResult` keeps no `MSM` object. The reason now opens "A restored result whose regime rule, regime density, or, on a TMLE result, MSM function is not declared 'known'". The note says that RM28 records the rule, and that a refit with each function declared known reports an interval. `reopened_by` stays `"RM28"` |
| 16f9302, T1 | the frozen-oracle form of mutation R10 was a tautology. It computed `sample_ratio(curve, 0)`, which reads 1 for any curve, so no package could fail it. The frozen-threshold oracle now divides by the closed-form fixed-rule curve and reads 1. R10 gained `test_mutation_a_curve_with_the_term_fails_the_witness`, and `sample_ratio` is removed |
| 16f9302, T2 | each threshold witness built its exact denominator from the reported curve, as $\sqrt{E_n[c^2] / E_n[(c + T)^2]}$, and not independently. `exact_rule_curve` and `exact_regimen_curve` now build $D + T$ from the closed forms alone, and never read the reported curve |
| 16f9302, T3 | the loop over the point fit's targeting coefficients had no length guard, so it passed on an empty `fluctuations` mapping. The test now asserts that the mapping holds the `"regime"` step alone |
| 16f9302, T4 | no test read the reported `std_error`. The tests stated the ratio on the reported curve. `test_the_reported_se_is_the_plugin_se_of_the_curve` pins `std_error` to `plugin_se` of the reported curve, $\sqrt{\mathrm{Var}_n / n}$ with `ddof=1`, to a relative 1e-12. The values are 0.021663638 for the point and 0.017682865 for the regimen. The witness ratio is `std_error` over `plugin_se` of the exact curve |
| 16f9302, T5 | `UNDERSTATEMENT_BOUND` meant 0.8 in `_policy_declaration_support` and 0.7 in `_tilt_law_support`. The policy bound is now `THRESHOLD_UNDERSTATEMENT_BOUND`, and the rule file imports the tilt bound under its own name |
| 4d8a83b, T6 | the test scaffolding repeated itself. `status_fit` and `cell_mean_learners` rebuilt the `LTMLE` fit of `regimen_fit`. `PANEL_COLUMNS` and `never_fit_longitudinal_learners` lived away from `panel()`. `longitudinal_entries` and its entries took parameters that no caller passed. The rule file built the same fixture three times. The helper `drop_the_carry` named mutation R6 in the rule file and R5 in the regimen file. `regimen_fit` now takes any `regimens=` value, and `status_fit` calls it. `PANEL_COLUMNS` and `never_fit_longitudinal_learners` moved beside `panel()` in `tests/unit/_declaration_support.py`. The rule file gained `referenced_rule` and `rule_fit`. The helpers are now `drop_the_replay_carry` for R6 and `drop_the_resolve_carry` for R5 |
| b563720 and 75e86be, T7 | four entries had no test: `refute()` and `diagnostics.refute()` on a legacy point result, the one-call `tmle(Y, A, W, ...)`, a restored `CausalStudy` point result, and a `Static` subclass at a fit entry. Commit b563720 added `TestTheFitRefusesAStaticSubclass`. Commit 75e86be added `"tmle"` to the rule file's `ENTRIES`, and the fixture `study_rule_result` drives the refutation and the restored study result. The R8 control now also requires the study witness to fail |
| 29b7658, D1 | the status table of [inference](technical-reference/inference.md#inference-status) named RM28 as the item that reopens the status. The row now says that a new fit with each function declared `"known"` reports an interval. Its reopened-by cell still links RM28, which `tests/unit/test_inference_status_registry.py` requires |
| this record, D2 and D3 | two locators pointed at the wrong line. `base.py:701` at 40b8e643 is a blank line, and the `isinstance` check is at `:722`. The quoted `tmle.py` docstring is at `:1522-1523` at commit 75e86be |
| this record, D4 and D5 | the RM20 record now says that `PRECEDENCE` pins `NON_INFERENTIAL` to the order. It adds that order 3 can meet orders 5 to 7 on a restored result, and that it takes precedence |
| this record, D6 | two RM13 and RM25 rows said in the present tense that a legacy result answers as saved. They now say "Before RM28" and "At RM27" |
| 29b7658 and this record, D7 | the references page said that RM28 requires every `DynamicRegimen` to declare its rule. A plan of labels alone is exempt. It also said that the effect is 1 in the witness law, which holds at the rule's node in both laws. It now marks the witness reasoning as the package's own. The plan above distinguishes the threshold witness from the optimal-rule result |
| 29b7658, D8 | the [scope page](technical-reference/scope-and-refusals.md#wrong-by-construction) said "Five of these share one mechanism" without naming them. It now names the first three rows, and the two MSM rows that apply the same argument to a working-model function. The shift row in fifth place does not share the mechanism |
| this record, D9 | the RM20 record said that the status table lists "the five statuses". It now says "each status" |
| 29b7658, D10 | the [longitudinal reference](technical-reference/longitudinal-tmle.md#variations) gains two refusal rows, a callable node that is not declared `"known"` and an inline callable. The iterator and single-entry refusals are `DataError` statements about the input, so a paragraph gives them, not the table of kinds. A sentence gives the restored-result status. "Fixed", the word of the docstrings, replaces "prespecified" |
| this record, D11 | the RM23 corrections and witnesses now include the restored `Rule` and `LTMLE` results that RM28 records |
| 29b7658, D12 | the `LTMLE` docstring mixed the numpydoc form with the loose `name:` form. Every parameter of `LTMLE` and `LTMLE.fit` now has its own `name : type` entry, and `LTMLE.fit` has a Returns section. numpydoc validation reports no finding on either |
| 29b7658 and this record, the writing standard | the inference row, the evidence row, "No study was regenerated", "This path opens only when ...", and "RM28 holds the density" |

The hand mutations of the review fixes ran as the delivery's did, with a sha256 backup of each
file. Each run used the rule, regimen and RM25 files, and each restored file matched its hash.

| mutation | file | failures |
| --- | --- | --- |
| S1. `DynamicRegimen` does not store its plan as a tuple | `src/cleverly/longitudinal/regimen.py` | 3 in the regimen file |
| S2. `_plan_nodes` skips an iterator, as the old helper did | `src/cleverly/longitudinal/regimen.py` | 6 in the regimen file |
| S3. `declaration_status` catches every exception | `src/cleverly/_declarations.py` | 1 in the rule file |
| S4. `Static.density_kind` reads `"known"` for any subclass. H4 above ran the same mutation and failed 1 test | `src/cleverly/interventions/base.py` | 5 in the rule file |
| T4. the unclustered plug-in variance uses `ddof=0` | `src/cleverly/inference/cluster.py` | 4, two in each of the rule and regimen files |

The first review left two plan shapes that no fix changed. Each one behaved the same at a0e93bb6
and at commit 75e86be. The later review below addresses both.

| plan | behavior | row |
| --- | --- | --- |
| `{"x": np.array(1)}` | `refuse_regimen_rules` raises `TypeError: iteration over a 0-d array` before any learner. That is a crash, not a refusal | [RM14](#rm14-intervention-refusals-at-identification), whose tier covers an exception that is not a refusal |
| `{"x": {"t1": 1, "t2": d}}` | the plan resolves to `Regimen('x', t1/t2)`, with the dictionary keys as arms. The fit then raises `DataError`: "regimen 'x' assigns 't1' at time 1". Nothing checks or calls `d` | [RM16](#rm16-summary-and-error-message-accuracy), because the message misstates the request |

At commit 75e86be the rule file holds 162 tests and the regimen file 109. The full fast suite gave
11551 passed and 87 skipped at commit 75e86be.

(rm28-review-saved-longitudinal-msm-provenance-and-malformed-plans)=

#### RM28 review: saved longitudinal MSM provenance and malformed plans

The first delivery withheld inference on a restored point-treatment MSM when its source
functions lacked declarations. A longitudinal result stored evaluated MSM arrays, without the
source `MSM` object or a declaration marker. Loading therefore treated an old longitudinal MSM
interval as inferential even when the source design or projection weight was unknown.

The review assigns `functions_kind="known"` to a `RegimenMSM` only after
`evaluate_regimen_msm` accepts both source function declarations. An older saved projection has
`functions_kind=None` through the plain class default. That absence cannot establish that the
source functions were fixed. The restored result keeps its point estimates and evaluated arrays,
but takes `undeclared_function_plugin`; its interval, p-value, and standard error refuse. A
truncation replay also refuses before it uses the saved MSM arrays. Refit from the original
declared `MSM` to recover inference.

The marker records the evaluation check, not a proof that a user's `"known"` declaration was
truthful.

The review also makes capability metadata follow the same declaration guards as recomputation.
On a restored point result, a missing rule, density, or MSM declaration makes
`retarget_cached_nuisances` and `refit_nuisances` false. On a restored longitudinal result, an
undeclared regimen rule or MSM marker makes `refit_nuisances` false. The dependent assessment
rows read `unavailable` with a declaration reason before a replay starts. A result whose
functions passed their declarations retains its replay capability. [RM23](#rm23-capability-rows-that-read-available-and-then-refuse)
still holds the other row mismatches and the saved fold-policy case.

The review also assigns `DataError` before any learner to a zero-dimensional array or a mapping
used as one regimen plan. The message for an array asks for one node per treatment time. The
message for a mapping asks for a sequence or `DynamicRegimen`. The two probes above show the
first-delivery behavior that these checks replace.

### RM29. Saved cross-fitted clustered longitudinal results

Releases 0.1.0 and 0.1.1 predate commit 5f32c14, which refused `id=` on a cross-fitted `LTMLE` fit.
Those releases drew grouped folds from the cluster labels
(`src/cleverly/longitudinal/estimator.py:1634-1640` at v0.1.1). The fit then reported a
cluster-robust interval. [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) records
that no source read here covers the cluster-robust variance of the targeted sequential recursion
under a grouped draw.

Under the RM26 rule, `LongitudinalResult.__setstate__` recomputed status from saved data and folds.
With 40 or more equal clusters, that rule returned `influence_curve`. The saved interval, p-value,
and bands then stood. At unequal sizes or few clusters, the status named a different condition.

The review of the RM26 delivery raised this surface. A 2026-09-23 probe measured it with the
source of tag v0.1.1, scikit-learn 1.9.0, and NumPy 2.4.6. Each fit used
`make_longitudinal(n=400, seed=0)` and
`LTMLE({"always": 1, "never": 0}, reference="never", n_folds=5, random_state=0)`. It used
`LinearRegression` for the outcome and the pseudo-outcome, and `LogisticRegression(max_iter=1000)`
for the treatment. Each result was pickled, and then loaded at commit 3eef4a4.

| probe | result |
| --- | --- |
| v0.1.1, with the labels `np.arange(400) * 40 // 400` passed as `id=` | the result loads under `influence_curve` with 5 folds. `ate_regimen[always vs never]` reads 0.486285, with `ci` (0.3374, 0.6352) and a p-value of 1.55e-10. It keeps its simultaneous bands, and `summary()` prints the 95% CI table |
| v0.1.1, with the labels `np.arange(400) * 39 // 400`, and with 40 clusters of 5 and 15 rows in turn | both load under `unequal_cluster_plugin`, and `ci` refuses. The 39 clusters differ in size by one row, so the unequal-size status takes precedence |
| commit 3eef4a4, with `LTMLE._refuse_cross_fitted_design` patched to do nothing and 40 equal clusters | the same shape. The result loads under `influence_curve` with `ci` (0.2608, 0.6292) and its bands. A test can build the witness this way |

The repair loaded an actual v0.1.1 pickle from the first row. It kept the point estimate 0.486285,
took `cross_fitted_longitudinal_plugin`, and withheld the old interval, p-value, and bands. It also
cleared the saved assessment cache.

The decision is a dedicated non-inferential status. A saved cross-fitted clustered `LTMLE` result
loads under `cross_fitted_longitudinal_plugin` at every cluster count and size. It keeps its point
estimates and plug-in spread as diagnostics.

Its `ci`, `pvalue`, and `std_error` refuse with a reason that names the grouped longitudinal design
and F22. Loading drops saved bands and assessment
answers. The truncation replay reads the same status. New fits still refuse the design before
learner fitting.

`tests/unit/test_longitudinal_cluster_status.py::TestASavedCrossFittedClusteredResult` builds a
five-fold result with the old entry rule restored in the test only. It saves the pre-status shape,
then loads it through `pickle` and the package serializer. The test checks 40 equal clusters, 20
equal clusters, and 40 clusters alternating between 5 and 15 rows. Every load takes the same
status and withholds all three inferential accessors. A mutation that skips the grouped decision
fails at 40 equal clusters. An in-sample 40-cluster result and an unclustered cross-fitted result
keep their intervals.

The same review found that a development artifact saved before observation weights existed could
load but then fail in `summary()`, validation, assessment, diagnostics, and truncation replay.
`LongitudinalData.__setstate__` now restores unit weights and their declaration. It also recovers
the backend name from the old input frame, then drops that frame. A four-case test checks both
restoration routes, with and without cluster labels. Releases 0.1.0 and 0.1.1 already carry the
weight fields.

[Inference status](technical-reference/inference.md#inference-status) and F22 record the restored
status. The new source audit read Nugent et al. (2024) on partial-cluster trials, Balkus et al.
(2026) on cross-fitting correlated units, and the JSS `ltmle` and clustered-sandwich articles.
Those sources do not derive the cluster-robust variance of this targeted sequential recursion under
grouped folds. [References](references.md#grouped-folds-and-clustered-cross-fitting) gives their
scope and links.

### RM30. Learned-policy value evaluation

The current fixed-rule paths refuse a `Rule` or `DynamicRegimen` learned from the analysis
sample. Add a typed learned-policy estimand and result whose target states which fitted policy's
value is evaluated. Do not reinterpret a learned function as a declared fixed rule.

Van der Laan and Luedtke (2015), Section 7, Theorem 6, and Appendix B, give a CV-TMLE for the
average value of rules learned on training folds. The paper derives its targeting equation,
expansion, and interval conditions for two treatment points. Its Section 6 separately treats
the value of one rule learned on the full sample, under stronger empirical process conditions.
Nordland and Holst (2026), Section 3.4 and Algorithm 4, give a cross-fitted, doubly robust
estimator and interval for a learned policy's value with finite stages and discrete actions.
Both methods train each evaluation fold's policy on other rows.

Use the JCI CV-TMLE as the first supported method. Target the average value of training-fold
learned rules, not the value of one rule fitted on all rows. Match its fold-local policy training,
nuisance fitting, targeting, influence variance, and interval to that target. State the bounded
outcome, positivity, score convergence, nuisance-convergence, and remainder assumptions that
apply. Specify whether its limiting-rule conditions hold for the supported learner class.

The JSS score and Wald interval are a separate possible extension, not a substitute for the JCI
targeting step. Neither source licenses an arbitrary population-dependent intervention density.

Acceptance requires a typed public target, fold-local learner and evaluator contracts, saved
policy and nuisance provenance, an influence-variance check, and nonzero exact-law witnesses.
Mutation controls must detect using evaluation rows for policy training, dropping a targeting or
influence-score term, and changing the variance. Register a
repeated-sampling coverage study under the published conditions, including a limiting-rule
setting, before publishing an interval. Document the target, assumptions, refusals, public
result, and persistence round trip; keep unsupported learned-function routes refused.

A population optimal value requires its own contract under Luedtke and van der Laan (2016).
The [RM28](#rm28-declared-densities-of-user-written-interventions) refusals remain in force until
this method and its validation are delivered. The JSS result gives no pathwise derivative for an
arbitrary user-written odds tilt. The [source audit](references.md#point-treatment-and-stochastic-interventions)
records the published boundaries.

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

Increment the persistence format. Store ordered stage definitions and evidence IDs, every fitted
array, reconstructible strategy configuration, fingerprints, settings, structured identification
and parameter keys, cached reports, and replayability. Unknown types fail allowlist decoding.
Custom callables are descriptive and non-reconstructible; cache-only operations remain possible
only when stored predictions and provenance suffice. Losing `alpha_star`, stage order, or component
products is a load error. Round trips compare every artifact, score, diagnostic, estimate,
influence curve, metadata record, and cached assessment.

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

A third 2026-09-21 search found four neighbors for F18, and RM18 records its log. The nearest is
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
| the final average | the stacked whole-sample plug-in (`src/cleverly/estimators/tmle.py:429-439`), because `CTMLE` refuses `cv_evaluation=True` (`src/cleverly/estimators/ctmle.py:753-757`) | the `(1/V) sum_v` fold average. The two agree only at equal fold weight mass |
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

A third 2026-09-21 search ran two queries on the generated design, and RM18 records its log. It
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
| shift, incremental, regime, MSM, and controlled-direct-effect targets, cross-fitted with missing outcomes | `TMLE` raises `CapabilityError` before any learner. The message names F21, and its remedy is the in-sample fit | F21. Each target needs its own source audit and contract, cross-fitted with missing outcomes |
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

Releases 0.1.0 and 0.1.1 predate commit 5f32c14, which refused `id=` on a cross-fitted `LTMLE`
fit. They could save that fit. On load, every such result now takes
`cross_fitted_longitudinal_plugin`, including equal, unequal, and few-cluster data. It keeps the
point estimate and a diagnostic spread, but no interval, p-value, standard error, or band.
[RM29](#rm29-saved-cross-fitted-clustered-longitudinal-results) records the probe and the repair.
The F22 route that reopens the fit would also cover saved results.

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
sets of score equations. The refusal comes from `TMLEMethod`, before any model is fitted.

Wait for a published targeting and inference result for each proposed composition, including its
joint score and covariance. Do not infer the construction from the existing single-axis
implementations. An accepted composition must reduce exactly to each standalone fit, preserve
parameter names and policy definitions, and expose the full cross-axis influence covariance.
Register nonzero controls for every cross-axis block, and repeated-sampling evidence for
simultaneous inference if it is claimed.

### F6. MNAR and incremental-intermediate compositions

An MNAR tilt for continuous-dose shifts and intermediate variables with incremental interventions
wait for identification and influence-function results covering those exact compositions.

The incremental-intermediate refusal runs after the shared nuisance fit.
[RM24](#rm24-refusals-after-the-nuisance-fit) moves it before any learner call. The tilt on a
shift fit reads available in its capability row and then declines, and
[RM23](#rm23-capability-rows-that-read-available-and-then-refuse) holds that row.

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
DR-TMLE refuses a baseline stratum, because its reduced regressions add a second targeting equation
for the `mean` group. Each estimator raises `NotImplementedError` during the fit. A post-fit
surface cannot repair these upstream estimator limits.

This item is unwritten work rather than a hard stop. Each parameter is well posed, and the package
already fluctuates baseline strata for arm, regime, and shift targets. The refusal taxonomy records
each row as [not written yet](technical-reference/scope-and-refusals.md#not-written-yet).

Match the stratum-indexed targeting construction to a published derivation before implementation.
Continuous MSMs also need dose-indexed strata semantics. Add the targeting equations and their
validation evidence next. Complete simulated-confounding replay receives its own audit last.

The ordinary TMLE and DR-TMLE refusals run after 2 and 8 learner fits on the probes that
[RM24](#rm24-refusals-after-the-nuisance-fit) records. RM24 moves them before any learner call.
This item holds the targeting construction.

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

## Reading a gap correctly

Not every absence is missing package functionality. The refusal taxonomy in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal) distinguishes an unimplemented
well-posed feature from a different causal question and a method that would be wrong by
construction. Only the first belongs on this roadmap.
