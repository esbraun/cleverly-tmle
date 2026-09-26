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

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) is the current record.
It lists every red verdict that a registered study publishes. It names the ask that owns each
one, by the `id` in the "What this row asks for" table of RM18.
`tests/unit/test_red_cell_ledger.py` checks the ledger against the committed results. Each red
verdict stays red under a `reporting` policy, so no verdict is hidden and no margin moved.

The detail section of a delivered row keeps a short record of what shipped. Commit `dea3297e`
holds the full plan, probe and review record of each row that was delivered before RM33. Read it
with `git show dea3297e:docs/roadmap.md`.

Commit `cf914b61` holds the full RM33 record. Read it with `git show cf914b61:docs/roadmap.md`.
Commit `4befdaa6` holds the full RM34 record. Read it with `git show 4befdaa6:docs/roadmap.md`.
The table below lists the rows that remain.

| priority | item | next action | problem | details |
| ---: | --- | --- | --- | --- |
| 0.04 | Inference status of saved bands and E-values outside their result | decide the status that a `SimultaneousBands` or `EValue` takes when it loads without its result | the object records no status, and it publishes its bands or its E-value limit. No field tells an old object from a current one | [RM35](#rm35-inference-status-of-saved-bands-and-e-values-outside-their-result) |
| 0.32 | Intervention refusals at identification | refuse mixed intervention kinds in `CausalStudy.identify`, and name the typed estimands in each message. Refuse a zero-dimensional regimen plan by name | a mixed request passes identification and then fails at estimation, once with an `AttributeError` | [RM14](#rm14-intervention-refusals-at-identification) |
| 0.33 | Continuous-dose MSM fit with missing outcomes | refuse the composition by name before any learner, as `CapabilityError` | an in-sample continuous-dose MSM fit with `delta=` raises `ValueError` from the nuisance fit, after two learner fits | [RM32](#rm32-continuous-dose-msm-fit-with-missing-outcomes) |
| 0.34 | Refusals after the nuisance fit | raise each refusal as `CapabilityError` before any learner call | two stratified requests refuse with `NotImplementedError` after 2 and 8 learner fits. An incremental request with an intermediate variable refuses with `ValueError`. Two refusals in the refit replay chain raise a plain `ValueError` or `NotImplementedError` | [RM24](#rm24-refusals-after-the-nuisance-fit) |
| 0.41 | Calibration-slope warning rule | replace the fixed band with a rule that a registered calibration study supports | the band flagged 14 of 40 fits of a correctly specified weak-signal propensity model | [RM15](#rm15-calibration-slope-warning-rule) |
| 0.42 | Summary and error-message accuracy | correct six display surfaces, three data error messages and one refusal remedy, add a fingerprint-only protocol option, and decide what a bootstrap summary publishes on a non-inferential fit. The simultaneous-request policy is settled below. Correct two `benchmark` argument checks, and make one truncation row agree with its call | each surface omits, misstates, or repeats a fact that the fit records. Three more surfaces misstate what a call accepts or needs | [RM16](#rm16-summary-and-error-message-accuracy) |
| 0.51 | Red property cells after the fold, scale and law changes | keep each red verdict under `reporting` with its interval, and admit inference only when F18 or F19 supplies the exact result. The [red-cell ledger](technical-reference/method-evidence/red-cells.md) delivers this. Then declare and run the five open RM18 follow-up designs, each declared before its run | registered studies publish red verdicts after the fold, scale and law changes and the pooled update. The ledger lists each one and the ask that owns it. Five RM18 follow-up designs are not declared and have not run | [RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) |
| 0.52 | One-sided robustness bias increment in DR-TMLE | investigate the exploratory between-implementation increment on binary `treatment_correct`, under a design declared before it runs | the RM18 reading is `mixed` on that configuration. The unadjusted paired 99% interval of `cleverly` minus R `drtmle` runs 0.000068 to 0.001942, while the Bonferroni interval for that comparison covers zero. No implementation defect is established | [RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle) |
| 0.61 | Learned-policy value evaluation | implement a typed learned-policy target with fold-local training and evaluation, using a published estimating and inference contract | fixed-rule paths refuse a rule learned from the analysis sample; published learned-policy methods give distinct targets and inference | [RM30](#rm30-learned-policy-value-evaluation) |

Each row takes a tier by the harm that its defect does to a user today. The table gives the tiers,
from the most harmful. Inside a tier, a row with a wider reach comes first. A row that another row
depends on comes before that row.

| tier | reason | rows |
| --- | --- | --- |
| a | a published number that is wrong, or that no derivation or read source covers. An anti-conservative number ranks above a conservative one | RM35 |
| b | a crash, an exception that is not a refusal, a capability row that reads available and then raises, or an assessment that returns no report | RM14, RM32 |
| c | a correct refusal that arrives late or as the wrong type | RM24 |
| d | a diagnostic or a warning that misleads | RM15 |
| e | a display or a message that misstates a fact that the fit records. By extension, an argument check or a capability row that misstates what a call accepts or needs, when no number moves and nothing raises that is not a refusal | RM16 |
| f | an investigation or a declared design that moves no verdict | RM18, RM19 |
| g | a published method needed to resolve a shipped refusal | RM30 |

The table gives the reason for each place inside a tier.

| row | reason for its place |
| --- | --- |
| RM35 | a band or an E-value that a caller saved apart from its result publishes a limit that this version can withhold. Only a caller who pickles a band or E-value alone meets it |
| RM14 | one mixed request raises an `AttributeError`. It also has a late refusal of tier c, so it takes the higher tier |
| RM32 | one composition raises a `ValueError` that is not a refusal, after two learner fits. RM14 reaches every mixed intervention request, so RM14 comes first |
| RM24 | two refusals arrive after 2 and 8 learner fits, and the refusals of the row have the wrong type |
| RM15 | the warning flagged 14 of 40 fits of a correct model |
| RM16 | each surface misstates or repeats a recorded fact, or misstates what a call accepts or needs, and no number changes |
| RM18 | five open designs, which read 19 red rows in six studies. The ledger already publishes each of those verdicts |
| RM19 | one configuration, which RM18 opened. Its Bonferroni interval covers zero, and it moves no verdict |
| RM30 | a published learned-policy method can resolve the current refusal, but requires a distinct target, fold-local evaluation, and inference validation |

No open row waits on another open row. The RM14, RM24 and RM32 requests produce no fit, and RM35
reads an artifact that an earlier version saved.

Use five delivery groups for these nine rows and the two investigations that RM18 waits on.
Keep each item's acceptance criteria separate inside its group.

| delivery group | items | shared boundary |
| --- | --- | --- |
| saved-result inference | RM35 | the status that a saved artifact takes when this version refuses the configuration that produced its interval, or cannot read that configuration |
| refusal surfaces | RM14, RM32 and RM24 | a refusal reaches the caller where its declaration says, before the work that it refuses |
| diagnostic reports | RM15 and RM16 | one assessment and summary surface, with one documentation pass |
| red property cells | RM18 and RM19, and the F18 and F19 derivations that RM18 waits on | the recorded rule that a red cell is reporting evidence, designs declared before their runs that move no verdict, and two exact derivations that would close the inferential gaps |
| learned-policy evaluation | RM30 | a typed target and a published fold-local learning, evaluation, and inference contract with validation |

Each group holds consecutive priorities, so the group order is the priority order. Deliver the
items inside a group in priority order. The first decimal digit of a priority names its group, and
the second digit orders the rows inside that group. Each remediation priority is below 1, so it
does not collide with a main-roadmap priority.

Priorities give the current delivery order. This project reassigns them when it re-triages the
queue. The RM IDs and their anchors never change, so a commit names a row by its ID. A delivered
row takes its priority with it, and the other rows keep theirs.

Main-roadmap priority 1 waits until every remediation row is complete, as the rule above states.
The queue holds nine rows. Six rows need their corrections: RM14 to RM16, RM24, RM32 and
RM35. RM18 has five follow-up designs that are not declared and have not run. RM19 has no
declared design. RM30 holds the published learned-policy implementation.

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
| saved results | `TMLEResult.__setstate__` re-stamps each estimate from the estimator that the artifact carries |
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

The longitudinal path has a sibling. `LTMLE({"x": np.array(1)}).fit(...)` raises
`TypeError: iteration over a 0-d array` from `refuse_regimen_rules`, before any learner. A probe
gave the same error at a0e93bb6 and at commit 75e86be.

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
| fold-policy refusal on a restored continuous-dose fit | the old message said `stratify_folds='treatment'` "balances the outer folds on the treatment", although the saved continuous-dose split used no treatment strata | `TMLE._fold_strata` returns `None` for a continuous treatment. A copied or restored estimator can carry the old policy into `refit`, where `fold_strata_refusal` refuses it before a learner runs. `tests/unit/test_saved_fold_policy_status.py::test_restored_dose_refit_names_the_requested_strata` checks the dose and a discrete-treatment control | delivered: the refusal now says the policy "requests stratification" and describes what a split drawn from those strata would do. It retains the unstratified and in-sample remedies |
| `DRTMLE` class docstring | describes an open centring defect on a quarter of splits, and names the test class `TestTheReportedCurveIsNotAlwaysCentred` | no test class has that name. `TestTheReportedCurveIsCentredWhereTheBoundBinds` in `tests/unit/test_drtmle_fit.py` records the fix, which solves the score at the truncated tilt. The docstring of `TestEachDrawSolvesItsOwnEquations` names the old class too | describe the fixed state, and name the present class in both docstrings |
| bootstrap summary on a non-inferential fit | `estimate.bootstrap.ci` answers, and `to_dict` emits `bootstrap_std_err`, under inferential names on a fit whose status supplies no inference | `BootstrapSummary` in `src/cleverly/inference/influence.py` is a plain dataclass, and `ParameterEstimate.to_dict` writes `bootstrap_std_err` whatever the status. `to_dict` already renames the percentile limits to `bootstrap_range_lower` and `bootstrap_range_upper`. RM12 kept the `bootstrap_std_err` name on purpose, and the [collaborative reference](technical-reference/collaborative-tmle.md) says so | decide what the bootstrap publishes on such a fit, and record the decision in the collaborative reference and in [inference status](technical-reference/inference.md#inference-status). A refusal at `.bootstrap.ci` would break the `ci=` keyword of the `BootstrapSummary` constructor, every reader of that field, and results pickled before the change |
| continuous-treatment `DataError` | the error for a continuous treatment with no `shifts=` and no `msm=` offers `Shift(0.0, cap=None)` as "the natural course". A cross-fitted fit with `delta=` then meets the F21 refusal of a shift target, whose remedy is the in-sample fit | `TMLE._check_shifts` in `src/cleverly/estimators/tmle.py`. The review probe of the F21 refusal (commit 45f072b) fitted `make_missing_outcome_binary(n=400, seed=4)` with a continuous dose and `Shift(0.0, cap=None)`, cross-fitted with `delta=`. It refused before any learner. Its in-sample fit reports `ey_shift[natural course]` 0.51256 | on a cross-fitted fit with `delta=`, name the in-sample fit beside the natural course, so that the suggested request does not meet a refusal |
| longitudinal plan written as a mapping | `{"x": {"t1": 1, "t2": d}}` resolves to `Regimen('x', t1/t2)`, with the dictionary keys as arms. The fit then raises `DataError`: "regimen 'x' assigns 't1' at time 1". The message names a label that the user did not mean as an arm, and nothing checks or calls `d` | `_plan_nodes` in `src/cleverly/longitudinal/regimen.py` reads any iterable that is not an iterator as a tuple of its items. The review of RM28 probed the plan at a0e93bb6 and at commit 75e86be, with the same result | refuse a mapping plan by name, and name the sequence form and `DynamicRegimen` in the message |
| `benchmark(covariates=[])` | runs a refit that drops nothing. The report reads "implied cf_y = 0.0000, cf_d = 0.0000" and "the estimate moved by +0" | `benchmark_refusal` in `src/cleverly/sensitivity/omitted_variable.py` returns `None` for an empty request, because a covariate remains. A 2026-09-24 probe on the RM23 sweep's `ordinary` kind, `make_linear_ate(n=400, seed=2)` in sample, returned that report | refuse an empty `covariates` as a malformed argument, before the refit |
| `benchmark` covariate names on a fit with an encoded categorical covariate | accepts the indicator column `V__low`, and rejects the logical name `V` with `DataError`: "unknown covariates ['V']; this fit adjusts for ['W1', 'W2', 'W3', 'W4', 'V__low']". The fit adjusts for `V` | the same probe on the sweep's `stratified` kind, which adds a two-level `V` as a stratum and a covariate. `simulated_confounding` refuses an encoded column by name, because zeroing one encoded column does not define a logical-covariate benchmark | accept the logical name and drop its whole encoded block, or refuse an indicator column by name as `simulated_confounding` does. Name the logical column in the message |
| `truncation_curve` row of a restored guarded DR-TMLE result under a refused fold policy | reads `unavailable`, although the module call `truncation_curve(result, [0.05])` runs and returns a curve | the same probe restored the sweep's DR-TMLE kind, `make_binary_outcome(n=240, seed=11)`, with `stratify_folds="treatment"`. `assessment_capabilities` makes the guarded row require `refit_nuisances`, which reads false. The curve refits the reduced regressions inside `retarget`, and does not call `refit()` | decide which replay slot the guarded curve needs, and make the row and the call agree |

Each correction needs a unit test that fails without it. Three tests are nonzero witnesses. A
`RegimeContrast` summary keeps its reference line. A design with a time-varying covariate prints it
at its node. A truncated fit labels a `score load` that differs from its `ratio effective n`. The
table gives what the other tests need.

| row | what its test needs |
| --- | --- |
| split-spread fact | the test fails on a selector fit with `repeats=2`, and it passes on an ordinary fit |
| `DRTMLE` class docstring | the docstring test asserts that each test class that a docstring names exists |
| in-sample C-TMLE selection-fold refusal | the test fits the probe above, and asserts that neither message offers `cross_fit=False` to an in-sample fit. A control keeps that remedy on a cross-fitted fit |
| fold-policy refusal on a restored continuous-dose fit | `test_restored_dose_refit_names_the_requested_strata` refits a restored dose result and a discrete-treatment control under each refused policy. It checks that the message describes a request, not observed balance |
| bootstrap summary | a decision before a test |
| continuous-treatment `DataError` | the test fits the suggested shift, cross-fitted with `delta=`, and asserts that the message names the in-sample fit |
| the two `benchmark` rows | the test fits the probe kind of the RM23 sweep. The control is a proper subset of the covariates, which runs |
| `truncation_curve` row | the test compares the row with the module call on the restored result |

The simultaneous row needs no change. `LTMLE` skips its default band below 40 clusters, as `TMLE`
does under each non-inferential status. The result summary names the omission for both estimators.
Tier e holds the two `benchmark` rows and the truncation row by the extension that its reason
states.

Pull request 229 corrected two refusal reasons that this row held. `_risk_ratio_refusal` names the
missing intermediate intervention level, and `_select_evalue` gives a separate reason for each
failed condition. `tests/unit/test_evalue_direct_effect_refusals.py` checks both.

### RM18. Red property cells after the fold, scale and law changes

Sixteen registered studies moved to unstratified folds. Six of them also moved to a bounded
outcome law with a declared support, and one moved to a binary clustered law. The six are the
manifests that record `tests/studies/bounded_cv_laws.py`. The multi-arm selector row, the
multi-arm outcome-adaptive row and both DR-TMLE rows stay on a binary outcome and declare no
`q_bounds`. The pooled longitudinal update then regenerated five cross-fitted longitudinal
studies. Property cells went red after each change.

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

#### What the earlier readings established

Each reading below ran once, under a declaration written before its run. Commit `dea3297e` holds
each declaration and each full result, under the subsection title in the first column. Read it
with `git show dea3297e:docs/roadmap.md`. The ledger gives the current verdict of each cell.

| subsection at `dea3297e` | ask | result |
| --- | --- | --- |
| "What each attribution rests on" | the fold and law attributions | an external 2x2 at commit `4ca7a15` attributes most of the selector `selector_necessity/collaborative` move to the bounded law, and the `type_i_error/sharp_null` move to the bounded law alone. An external 2x2 at `eeaa1ce` attributes the end-of-study overfitting move to the fold policy alone. The repository commits no run log for either one, so neither is independently verified |
| "The pooled update, declared before it runs" and "What the pooled update found" | `RM18-pooled` | the cross-fitted per-regimen fit follows Section 5.2, Steps 1 to 4, of Díaz, Williams, Hoffman and Schenck (2023). Five studies regenerated under a rule that moved no margin, budget, law, cell or seed. `crossfit_overfitting/cross_fitted_ltmle` reads 1.016107, with a 99% interval from 0.996092 to 1.036997. Two cells of `weighted-ltmle-crossfit` went red |
| "The runtime isolation, declared before it runs" | `RM18-attribution` | "[What the runtime isolation found](#what-the-runtime-isolation-found)" gives the result |
| "The two readings, declared before they run" and "What the two readings found" | `RM18-n500` and `RM18-binary-slope` | a pilot measured discordance rates of 0.0117 on the selector study and 0.0250 on the DR-TMLE study, so the fold-policy diagnostic declared 8,000 paired draws. The realized half-widths are 0.0043 and 0.0051. The paired coverage gains run -0.006875 to +0.001625 and -0.005625 to +0.004500. Each covers zero with an upper endpoint below 0.005, which is a boundary resolution. The declared rung design raised the two outer rungs to 2,400 replications, and `rate_outcome_correct` now runs -1.534453 to -0.482268. Each rung's coverage verdict stays at its declared 800 replications |
| "The one-sided robustness reading, declared before it is computed" | `RM18-one-sided-bias` | "[What the one-sided reading found](#what-the-one-sided-reading-found)" gives the result |
| "The learner-weight standard-error diagnostic, declared before it runs" and "What the learner-weight diagnostic found" | `RM18-learner-weight-se` | 13 of 1,200 `learner_weight_necessity` replicates report a standard error above 1. Predictions P1 to P5 hold, and the reading is finite-sample empty-cell instability in the out-of-fold mechanism estimate. `tests/diagnostics/rm18_learner_weight_se/` holds the refits and the reading. No verdict reads those standard errors |
| "What the source search found for the first three asks" | `F18`, `F19` and `RM18-pooled` | no published result covers the shipped selector path or the shipped generated design. Theorem 3 of Díaz, Williams, Hoffman and Schenck (2023) certifies the pooled update. [Collaborative TMLE references](references.md#collaborative-tmle) give each locator, and F18 and F19 carry the contracts |
| "Deferred findings and review resolutions" | two review findings | resolved. `tests/canonical/lmtp_crossfit_adapter.R` screens the first node of a supplied density-ratio matrix. The `double_robustness/both_correct` cell of the selector study moved from seed `12_100` to `12_104`, and a registered-study test rejects a seed collision across families |

(what-the-committed-history-already-separates)=
#### What the committed history already separates

Three point-treatment studies changed the runtime and the `src/` tree between an older commit and
`0e03a15`. The runtime moved from Python 3.11.13 and SciPy 1.17.1 to Python 3.13.7 and SciPy
1.18.0. Each comparison therefore bounds the source and runtime changes together.
`tests/diagnostics/rm18_runtime/row_drift.py` reads the committed rows with `git show` and writes
`row-drift.csv`.

| study | older commit | rows compared | largest change in an estimate | rows that moved more than 1e-6 | covered flags that changed |
| --- | --- | --- | --- | --- | --- |
| multi-arm point-treatment DR-TMLE | `6933968` | 14,400 primary and 10,600 property | 2.1e-12 | 0 | 0 |
| selector-based multi-arm C-TMLE | `2049349` | 21,600 primary and 5,200 property | 4e-15 | 0 | 0 |
| DR-TMLE for binary complete data, `cleverly` rows | `99d238c` | 7,200 primary | 4.3e-9 | 0 | 0 |
| the same, property rows | `99d238c` | 15,200 | 0.0047 | 6 | 0 |
| the same, `drtmle` R comparator rows | `99d238c` | 7,200 primary | 0.00062 | 6 | 0 |

The study's slope rule, applied to the first 800 replications of each rung, gives the intervals
below.

| cell | `99d238c`, 800 per rung | `0e03a15`, first 800 per rung | `0e03a15`, as committed |
| --- | --- | --- | --- |
| `rate_outcome_correct` | -3.040132 to +0.123221 | -3.040131 to +0.123221 | -1.534453 to -0.482268 |
| `rate_treatment_correct` | -2.207877 to -0.880196 | -2.207860 to -0.880192 | -1.756754 to -1.022105 |
| `rate_both_wrong`, the control | -0.003031 to +0.014273 | -0.003031 to +0.014273 | +0.003726 to +0.013538 |

At the declared budget of 800, the source and runtime changes together leave each interval within
2e-5. The move of `rate_outcome_correct` below zero therefore comes from the extra outer-rung
replications that the rung design declared.

(what-the-runtime-isolation-found)=
#### What the runtime isolation found

The diagnostic crossed two code states with two runtimes on the weighted and end-of-study studies.
Code state F is commit `7d5485a`, which fits the fold-local update. Code state P has the `src/`
tree of `0e03a15`, which fits the pooled update. Runtime R11 is Python 3.11.13 with SciPy 1.17.1,
and runtime R13 is Python 3.13.7 with SciPy 1.18.0. Both preconditions held, so the harness is
validated. `isolation.csv` and `run.log` in
[`tests/diagnostics/rm18_runtime/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_runtime)
hold every value.

| study | cell | F, both runtimes | P, both runtimes | reading |
| --- | --- | --- | --- | --- |
| weighted | `interval_calibration/static__correctly_specified`, positive | pass. Efficiency ratio 1.053468, from 1.013022 to 1.092339 | fail. 1.060724, from 1.019495 to 1.100601 against 1.10 | code |
| weighted | paired `ey_regimen[never]` | pass. Equivalent, 99% lower endpoint -0.01875 | fail. Underpowered, lower endpoint -0.030 against -0.025 | code |
| end-of-study | `crossfit_overfitting/cross_fitted_ltmle`, positive | fail. SE ratio 1.176650, 99% upper 1.201555 at R11 and 1.201516 at R13 | pass. 1.016107, from 0.996092 to 1.036997 | code |
| end-of-study | `crossfit_overfitting/in_sample_control` | pass. SE ratio 0.353193 at R11 and 0.353196 at R13 | pass. The same | neither |

The two weighted cells went red with the pooled code, and not with the runtime.
`RM18-fixed-weights` owns them, and they stay red as finite-sample reporting evidence. The
survival-curve, competing-risk and categorical studies stay outside this result, because the
declaration deferred them.

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
| `RM18-one-sided-bias` | a reading of the three one-sided-robustness bias rows | delivered. The reading that "The one-sided robustness reading, declared before it is computed" declares ran once. "[What the one-sided reading found](#what-the-one-sided-reading-found)" gives its result. The three cells stay red under `reporting` |
| `RM18-slopes` | a reading of the two multi-arm DR-TMLE contraction slopes | open: the design this row asks for is not declared, and it has not run. This row computed the cost of that design. The row closes when a rung design declared before its run reads both slopes |
| `RM18-ordinary-weighted` | an owner for the six red rows of the ordinary weighted longitudinal study | open: the design this row asks for is not declared, and it has not run. The row closes when a design declared before its run reads the calibration row, the sharp-null row and the `targeting_necessity` family |
| `RM18-learner-weight-se` | a diagnostic for the standard-error explosions in the cross-fitted `learner_weight_necessity` rows | delivered. A complete review declaration was pushed before a clean rerun that reproduced the refit table. "What the learner-weight diagnostic found" gives its reading. This row owns no red cell |
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
exploratory. [RM19](#rm19-one-sided-robustness-bias-increment-in-dr-tmle) gives the adjusted
interval and carries the question of the step.

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

(what-the-multi-arm-rung-design-would-cost)=
#### What the multi-arm rung design would cost

`RM18-slopes` sanctions a rung design for the two multi-arm contraction slopes and defers its run.
This subsection applies the sizing rule that `CONTRACTION_REPLICATES` in
`tests/studies/drtmle_properties.py` declares for the binary ladder. The rule reads the first-rung
bias and the `both_wrong` control's spread. It does not read the slope, the interval or the verdict
of either positive rate cell.

The multi-arm ladder runs n = 2,000, 4,000 and 8,000, with 600 replications on each rung.
[`tests/diagnostics/rm18_rung_cost/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_rung_cost)
holds the computation and its output, `cost.csv`.
`tests/unit/test_rm18_rung_cost_diagnostic.py` recomputes each delta-method figure from
`tests/canonical/multi_arm_drtmle/properties.csv`.

| input | value | source |
| --- | --- | --- |
| first-rung bias, `outcome_correct_n2000` | 0.001127, with a 99% interval from -0.001651 to 0.003905 | the `bias`, `bias_ci_lower` and `bias_ci_upper` columns |
| first-rung bias, `treatment_correct_n2000` | 0.000585, with a 99% interval from -0.002274 to 0.003444 | the same columns |
| control spread `c` | 1.1660, the mean of 1.1473, 1.1397 and 1.2108 | `empirical_se` times the square root of `n`, on the three `both_wrong` rungs |

At `R` replications on each outer rung, the delta-method 99% half-width is
`2.5758 * c * sqrt(n1 + n3) / (log 4 * b1 * n1 * sqrt(R))`. Here `b1` is the first-rung bias,
`n1` is 2,000 and `n3` is 8,000. The target is a half-width of one, as the binary rule declares.

| arm | smallest `R` by the delta method | smallest `R` by a surrogate simulation |
| --- | --- | --- |
| `outcome_correct` | 9,240, from 9,239.22 rounded up | 19,000 to 20,000 |
| `treatment_correct` | 34,267, from 34,266.75 rounded up | 70,000 to 73,000 |

The delta method understates the width when a rung's bias is near its Monte Carlo error, so its
figure is a lower bound. The `treatment_correct` arm sets the cost. The design needs at least
34,267 replications at each outer rung, against 600 now. The middle rung keeps 600, the floor its
coverage row needs.

Both first-rung bias intervals cover zero, so the input that the rule reads is not resolved. The
needed `R` grows as the inverse square of the true bias. A true bias below the point estimate
raises the cost without a bound. `RM18-slopes` therefore records the cost and does not run the
design.

#### Witnesses and evidence

| claim | evidence |
| --- | --- |
| every published verdict is recomputed from the committed replication rows | `tests/unit/test_method_evidence.py::test_paper_property_verdicts_are_recomputed_from_the_replication_rows` |
| the shipped fold and scale rules have mutation-controlled witnesses | `tests/unit/test_fold_policy_rules.py` |
| the pooled longitudinal update matches a longhand recomputation, and four mutations break it | `tests/unit/test_pooled_longitudinal_targeting.py` |
| the runtime isolation attributes each changed verdict to the code or the runtime | `tests/diagnostics/rm18_runtime/compare.py` reads the four arms of each study and writes `isolation.csv` in that directory. `run.log` records each arm's command, runtime, `pip freeze` digest and exit code |
| the committed history separates the runtime from the code on three point-treatment studies | `tests/diagnostics/rm18_runtime/row_drift.py` writes `row-drift.csv` in that directory |

#### Deferred findings and review resolutions

| finding | why it was deferred | what it needs |
| --- | --- | --- |
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
E-value row reads `unavailable`. `variable_importance()` refuses before its first fit. A result
saved before the status existed loads re-stamped.

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
| 3 | `undeclared_function_plugin` | the saved curve treats a restored function as fixed, and no code can check that it was fixed |
| 4 | `estimated_weight_plugin` | the same premise as order 1 |
| 5 | `cross_fitted_longitudinal_plugin` | no read source covers the cluster-robust variance of the sequential recursion under grouped folds |
| 6 | `stratified_fold_plugin` | no shipped result covers a saved split that read the treatment |
| 7 | `undeclared_scale_plugin` | no shipped result covers an outcome scale that the held-out rows set |
| 8 | `unequal_cluster_plugin` | the cross-fitted interval lacks validation at unequal cluster sizes or masses |
| 9 | `few_cluster_plugin` | no read source supports the reference distribution |
| 10 | `unrecorded_status_plugin` | the saved estimate records no status, and holds no configuration to read one from |

RM20 shipped orders 1, 2, 4, 8 and 9.
[RM29](#rm29-saved-cross-fitted-clustered-longitudinal-results) added order 5, and [RM28](#rm28-declared-densities-of-user-written-interventions) added order 3.
[RM31](#rm31-inference-status-of-a-saved-stratified-cross-fitted-result) added order 6, and
[RM33](#rm33-inference-status-of-a-saved-undeclared-scale-cross-fitted-result) added order 7,
and [RM34](#rm34-inference-status-of-a-saved-estimate-outside-its-result) added order 10.
`PRECEDENCE` in `tests/unit/test_inference_status_registry.py` pins `NON_INFERENTIAL` to this
order.

Among the five RM20 statuses, only three pairs can meet. On DR-TMLE, the estimated-weight status
meets each clustered status. On
TMLE and DR-TMLE, the two clustered statuses meet each other. C-TMLE refuses `id=` at every
setting.

On a restored result, order 3 can meet each clustered status, the saved split and the saved
scale, orders 5 to 9, and it takes precedence.
`test_the_undeclared_status_precedes_the_cluster_status`, in
`tests/unit/test_rule_and_intervention_declarations.py` and
`tests/unit/test_regimen_rule_declarations.py`, restores a real few-cluster fit and reads order 3.
Order 3 cannot meet order 4, because `DRTMLE` refuses `interventions=` and `msm=`.

On a restored result, order 6 can meet orders 3, 4, 5, 7, 8 and 9. Orders 3, 4 and 5 take
precedence over it, and it takes precedence over the saved scale and the cluster statuses, orders
7 to 9.
`test_the_status_comes_before_a_cluster_status`, in `tests/unit/test_saved_fold_policy_status.py`,
restores a stratified fit of 39 clusters and reads order 6. A restored `LTMLE` split with
`id=` meets order 5, which comes first.

On a restored result, order 7 can meet orders 3, 4, 6, 8 and 9. It cannot meet order 5, which only
`LTMLE` takes, because the `LTMLE` path has no scale rule. Orders 3, 4 and 6 take precedence over
it, and it takes precedence over the cluster statuses, orders 8 and 9.
`test_a_saved_stratified_split_comes_first`, `test_an_undeclared_function_comes_first` and
`test_the_status_comes_before_a_cluster_status`, in `tests/unit/test_saved_scale_status.py`, read
orders 6, 3 and 7 on restored fits.

Order 10 meets no other order. No hook returns it, and each result re-stamp replaces it with
the status of the configuration that the result holds.

Each status of orders 1 to 9 reads the estimator configuration and the prepared data alone, so it
is known before a learner runs. Order 10 reads the saved state of an estimate. No status reads a
fitted quantity. The estimator stamps the status after nuisance fitting.

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
| plug-in limits | `SensitivityBounds` records `nu2_estimator`. Under `"plugin"` the six limit accessors raise `CapabilityError` with the reason that cites [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound). A bound pickled before RM22 reads `"unrecorded"` and refuses its limits |
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
request resolves as available answers it. The delivery routed its other findings to RM16, RM24,
RM31 and RM32.

### RM24. Refusals after the nuisance fit

The [Definition of done](#definition-of-done) asks for a pre-fit test for every well-posed
composition that is still refused. The table records the original late-refusal probes.
Only the last one raised `CapabilityError` then. A 2026-09-22 probe counted the `fit` calls
of spy learners in the first three rows. The delivery of RM20 found the fourth row.

| request | where it raised at the probe | learner fits before the raise | exception |
| --- | --- | ---: | --- |
| `TMLE(incremental=...)` with `strata=`, `make_linear_ate(n=400, seed=2)` | the strata guard in `_retarget_detailed`, `src/cleverly/estimators/tmle.py:2736` | 2 | `NotImplementedError` |
| `DRTMLE` with `strata=`, the same law | the same guard, for the `mean` group | 8 | `NotImplementedError` |
| `TMLE(incremental=...)` with `intermediate=`, `make_cde(n=400, seed=3)` | `_check_incremental`, `tmle.py:2148`. `fit` fits the shared nuisances first | 3 | `ValueError` |
| `TMLE(estimands=["par"])` with `intermediate=` and `delta=`, on `binary_cde_frame()` from `tests/unit/test_cross_fitted_missing_off_contract.py`, with `LogisticRegression` in every learner role and `random_state=0` | the F20 refusal in `_fit_single`. On the intermediate path, `fit` calls `_prepare_shared` first, and that call fits the shared nuisances | 4 in sample. 20 cross-fitted at `n_folds=5` | `CapabilityError` |

At the probe revision, the fourth row contradicted the comment in `TMLE.fit` about refusing
unsupported intermediate compositions before a learner. The shared
`_preflight_fit_configuration` now runs before `_prepare_shared`, so the third and fourth
requests refuse before any nuisance learner. The first two stratified requests still reach
targeting after their learner fits. `test_par_beside_an_intermediate_keeps_its_f20_refusal`
pins the fourth refusal's F20 message.

A second historical finding concerns the order of two refusals. At the probe revision,
`DRTMLE(cross_fit=True)` with `delta=` reached `_check_drtmle` at the start of the nuisance fit.
The fold preflight ran earlier, so a sample with no respondent in a training complement got a
`DataError` instead. The shared configuration preflight now runs `_check_drtmle` before drawing
folds, so that composition raises `NotImplementedError` first. No learner runs in either order.
The table records the original probes.

| probe | result |
| --- | --- |
| `respondents_in_one_fold()` from `tests/unit/test_fold_policy_rules.py`, `DRTMLE(cross_fit=True, n_folds=6, random_state=1146, q_bounds=(0, 1))`, at `guard=("Q", "g")` and at `guard=()` | `DataError` "cross-fitted DR-TMLE cannot fit its nuisances because repeat 0, fold 2's training complement contains no row with an observed outcome". The composition is refused at every sample, so the data error names a problem that no other sample would fix |
| `binary_missing_frame()` from `tests/unit/test_cross_fitted_missing_off_contract.py`, `DRTMLE(cross_fit=True, n_folds=5, random_state=0)` with `NeverFit` learners | `guard=()` raises `NotImplementedError` "the published missing-outcome DR-TMLE theorem uses Donsker conditions and does not establish its cross-validated extension". `guard=("Q", "g")` raises the randomized-trial `NotImplementedError` first. `NeverFit.calls` is 0 in both |

The [RM23](#rm23-capability-rows-that-read-available-and-then-refuse) review found one more
consumer of this row, which the table gives. It also found a late `refute()` argument check. That
check is resolved: `_validated_tests` checks a missing `negative_control_outcome=` before any
refit.

| finding | measured | action and status |
| --- | --- | --- |
| the refit replay slot reads a plain `ValueError` or `NotImplementedError` as a refusal | `TMLE._refit_configuration_refusal` runs the chain that `refit()` runs before a learner. `CTMLE._check_estimands` refuses `att` with a plain `ValueError`, and `DRTMLE._check_drtmle` raises `NotImplementedError`. Both run in that chain, before any learner | **Pending.** Move those refusals to `CapabilityError`, then narrow the catch of the slot to `CapabilityError` and `DataError` |

[X8](#x8-stratified-incremental-and-msm-targeting) holds the stratified targeting construction,
and [F6](#f6-mnar-and-incremental-intermediate-compositions) holds the incremental-intermediate
composition. The stratified requests still need earlier refusals. The incremental-intermediate
request still needs its `CapabilityError` type; its timing is already corrected.

Apply these corrections:

1. Raise each refusal as `CapabilityError` before any learner call. Decide it from the estimator
   configuration and the data declaration alone. A refusal of the composition comes before a
   preflight that reads the sample.
2. Keep each message, and its pointer to X8, F6, or F20.
3. Search the other `NotImplementedError` and `ValueError` refusals in `src/cleverly/estimators/`
   for the same order. List each refusal that this row moves.
4. Apply the pending correction of the RM23 review table above.

The witnesses must fail when a component is wrong:

- for each request, a spy-learner test asserts that no learner call ran, and pins the message;
- a mutation that moves a guard back after the nuisance fit makes its test fail;
- a control shows that stratified arm, regime, and shift targets still fit.

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
`LongitudinalResult.__setstate__` re-stamps a saved result. `tests/unit/test_longitudinal_cluster_status.py`
holds the witnesses.

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
| restored results | the non-inferential status `undeclared_function_plugin`, third in the precedence. A restored `TMLE` or `LTMLE` result takes it when a rule, a user-written class, a `Stochastic` density, or a written MSM function is not declared `"known"` |
| witnesses | `tests/unit/test_rule_and_intervention_declarations.py` and `tests/unit/test_regimen_rule_declarations.py`. `tests/unit/_policy_declaration_support.py` holds the exact-law values that the plan in the record at commit `dea3297e` declares |

### RM29. Saved cross-fitted clustered longitudinal results

Releases 0.1.0 and 0.1.1 predate commit 5f32c14, which refused `id=` on a cross-fitted `LTMLE` fit.
Those releases drew grouped folds from the cluster labels, and the fit reported a cluster-robust
interval. [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) records that no source read
here covers that variance. A v0.1.1 result with 40 equal clusters loaded under `influence_curve`.
Its `ate_regimen[always vs never]` read 0.486285, with `ci` (0.3374, 0.6352).

Pull request 226 delivered this row. A saved cross-fitted clustered `LTMLE` result now loads under
`cross_fitted_longitudinal_plugin` at every cluster count and size. It keeps its point estimates,
and `ci`, `pvalue`, and `std_error` refuse with a reason that names F22. Loading drops the saved
bands and assessment answers.
`tests/unit/test_longitudinal_cluster_status.py::TestASavedCrossFittedClusteredResult` holds the
witness. `LongitudinalData.__setstate__` also restores unit weights on an artifact saved before
observation weights existed.

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

### RM31. Inference status of a saved stratified cross-fitted result

Releases 0.1.0 and 0.1.1 cross-fitted under `stratify_folds="treatment"` by default. This version
refuses that policy, because no shipped result covers a partition read off the data that the fit
then conditions on. The
[fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) give
the audit. A result that either release saved under that policy reported its interval.

Pull request 232 delivered this row. The table gives what shipped.

| part | what shipped |
| --- | --- |
| status | `"stratified_fold_plugin"` in `src/cleverly/_inference_status.py`, sixth in the precedence. Its record names RM31 |
| point-treatment rule | `TMLE._saved_fold_policy_status` in `src/cleverly/estimators/tmle.py` reads `crossfit_plan(data).stratify_by`. `DRTMLE` reaches it through `super()` |
| longitudinal rule | `_saved_split_status` in `src/cleverly/longitudinal/estimator.py`. Only `LongitudinalResult._restamp_inference_status` reads it |
| variable importance | `VariableImportanceResult.__setstate__` gives each entry the estimate of its re-stamped fit, and withholds `adjusted_pvalue` as `None` on a non-inferential status. `variable_importance` raises the fold-policy refusal before any learner |
| tests | `tests/unit/test_saved_fold_policy_status.py` |

The re-stamp reaches the estimates that a result holds, and nothing that a caller saved apart from
its result. [RM34](#rm34-inference-status-of-a-saved-estimate-outside-its-result) delivered the
estimate saved alone. [RM35](#rm35-inference-status-of-saved-bands-and-e-values-outside-their-result) holds bands and E-values.

### RM32. Continuous-dose MSM fit with missing outcomes

An in-sample `TMLE` fit of a continuous-dose MSM with `delta=` raises a `ValueError` that is not a
refusal. A cross-fitted fit meets the F21 refusal before any learner, as the
[scope page](technical-reference/scope-and-refusals.md#not-written-yet) states.

A 2026-09-24 probe at commit 3a9e429a fitted `make_missing_outcome(n=400, seed=4)`. The dose was
the arm plus standard normal noise from `numpy.random.default_rng(0)`. The fit used
`treatment_kind="continuous"`, `MSM.linear(doses=(-1.0, 0.0, 0.5, 1.0, 2.0))`, `density_bins=6`,
`LinearRegression` for the outcome, and `LogisticRegression(max_iter=1000)` for the treatment and
the missingness learners.

| request | result |
| --- | --- |
| `TMLE(...).fit(..., delta="Delta")` | `ValueError`: "need at least one array to concatenate", from `_mechanism_columns` in `src/cleverly/estimators/_nuisance.py`, after two learner fits |
| the same request through `CausalStudy`, with `PointTreatment(missingness="Delta", treatment_kind="continuous")` and `MSMProjection` | the same `ValueError` |
| the same fit without `delta=`, with each missing outcome set to 0 | runs, and reports `msm[(intercept)]` and `msm[a]` |

The `continuous` rule of `fit_wide_tilt_refusal` names `shifts=` and `ey_shift/ate_shift`. Only a
shift fit reaches that rule now, because this fit raises first.

Refuse the composition by name before any learner, as `CapabilityError`. A future implementation
of the fit needs its own contract and evidence, and the `continuous` tilt sentence must then name
the MSM fit too.

The witness fits the probe request with spy learners. It asserts the named refusal and zero learner
fits. A control keeps the in-sample shift fit with `delta=` running.

<a id="rm33-inference-status-of-a-saved-unbounded-scale-cross-fitted-result"></a>

### RM33. Inference status of a saved undeclared-scale cross-fitted result

Releases 0.1.0 and 0.1.1 set `q_bounds=None` by default (`src/cleverly/estimators/tmle.py:382`
at v0.1.1). A continuous outcome then took its scale from every observed outcome, held-out rows
included. Commit 5f32c149 refused that scale under cross-fitting, because no shipped result covers
it. The [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
give the audit. A continuous dose drew an unstratified split, so
[RM31](#rm31-inference-status-of-a-saved-stratified-cross-fitted-result) left the interval of such a
result.

The RM33 pull request delivered this row. The table gives what shipped. Commit `cf914b61` holds
the full RM33 plan and delivery record. Read it with `git show cf914b61:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| status | `"undeclared_scale_plugin"` in `src/cleverly/_inference_status.py`, seventh in the precedence. Its record names RM33 |
| rule | `TMLE._outcome_scale_refusal` in `src/cleverly/estimators/tmle.py` holds the predicate and the sentence. `_refuse_unbounded_cross_fitted_scale` raises the sentence at fit time. `TMLE._saved_scale_status` reads it as a status. `DRTMLE` reaches it through `super()`. `LTMLE` has no saved-scale rule. Its legacy cross-fitted results have the `"stratified_fold_plugin"` saved-fold status, subject to earlier statuses |
| variable importance | `variable_importance` raises `_refuse_unbounded_cross_fitted_scale` on each candidate's prepared data before it asks the status. The sentence names the remedy `Targeting(q_bounds=(lower, upper))` |
| tests | `tests/unit/test_saved_scale_status.py` holds the witnesses, three precedence witnesses, the controls and four mutations. `tests/unit/test_inference_status_registry.py` pins the precedence |

The witness loads a cross-fitted continuous-dose fit that reads `ey_shift[+0.5]` 3.798198 with
`ci` (3.3419, 4.2545). The restored result reads `undeclared_scale_plugin`, and `plugin_interval`
keeps (3.3419, 4.2545). `ci`, `pvalue`, and `std_error` raise `CapabilityError` with the status
reason. No registered study moves, because no live fit reaches the status.

### RM34. Inference status of a saved estimate outside its result

`TMLEResult.__setstate__` and `LongitudinalResult.__setstate__` re-stamped the estimates that the
result held, from the estimator, the prepared data and the folds. An estimate pickled apart from
its result held none of them. Release 0.1.1 wrote no `inference` field
(`src/cleverly/inference/influence.py:94` at v0.1.1). The class default `"influence_curve"` filled
it, so `ci`, `pvalue` and `std_error` answered.

The RM34 pull request delivered this row. The table gives what shipped. Commit `4befdaa6` holds
the full RM34 plan and delivery record. Read it with `git show 4befdaa6:docs/roadmap.md`.

| part | what shipped |
| --- | --- |
| status | `"unrecorded_status_plugin"` in `src/cleverly/_inference_status.py`, tenth in the precedence, with the constant `UNRECORDED_STATUS`. Its record names RM34 |
| load rule | `ParameterEstimate._PICKLE_BACKFILL` in `src/cleverly/inference/influence.py` gives the status to a state without the `inference` key, through `_DefaultingUnpickle` |
| re-stamp | `restamp_restored` in the same module is the one re-stamp of both results. `stamp_inference` returns the status to `"influence_curve"` when the configuration supplies inference, and `_stamp_cached` does the same for an estimate that the assessment cache saved |
| variable importance | `VariableImportanceEntry.__setstate__` withholds `adjusted_pvalue` when its estimate refuses a p-value. `_readjusted` in `src/cleverly/variable_importance.py` computes the adjustment again when every re-stamped entry supplies inference |
| tests | `tests/unit/test_saved_bare_estimate_status.py` holds the witnesses, the controls and the mutations. `assert_restamped` in `tests/unit/_inference_status_support.py` also loads the shape without the key |

The witness is the release 0.1.1 `ate` estimate saved alone. It reads `psi` 0.219215 and
`plugin_interval` (0.158065, 0.280365), and `ci` refuses. No registered study moves, because no
committed artifact is a pickle.

### RM35. Inference status of saved bands and E-values outside their result

RM34 gave a status to a `ParameterEstimate` saved without one. Two other objects publish a limit
and record no status.

| object pickled alone | what it loads as | evidence |
| --- | --- | --- |
| `SimultaneousBands` | its `critical_value` and `bands` | the fields at `src/cleverly/inference/multiplier.py:182-186` match `v0.1.1:src/cleverly/inference/multiplier.py:181-185`. The class has no status field and no `__setstate__` |
| `EValue` | its `risk_ratio_ci` and `limit` | the fields at `src/cleverly/sensitivity/evalue.py:185-194` match `v0.1.1:src/cleverly/sensitivity/evalue.py:134-143`. The class has no status field and no `__setstate__` |

A current object of either class exists only on a fit that supplies inference, because
`simultaneous_bands` and the E-value refuse every other status. A load rule cannot tell an old
object from a current one. A rule that withheld an old object would therefore withhold a current
one too. One option adds a recorded status field to each class. The missing-key rule of RM34 then
separates an object that an earlier version saved from a current one.

The witness pickles each object without the decided field, and loads it. It asserts the decided
status and each withheld limit. A mutation that skips the decision must fail. A current object
that records its status loads as saved.

`BootstrapSummary` is outside this row, because
[RM16](#rm16-summary-and-error-message-accuracy) decides what a bootstrap summary publishes on a
non-inferential fit. `ScoreCheck`, `RefutationTest`, `NuisanceDiagnostics`, `ReplicationRecord`
and `EstimandSummary` are outside it too. Their `inference` field picks a name or a wording, and
no accessor refuses on it.

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
