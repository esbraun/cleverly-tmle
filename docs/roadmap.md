# Roadmap

This is the single planning contract for `cleverly`. It contains proposed work only. Implemented
capabilities belong in the [user guide](user-guide/index.md), scientific contracts in the
[technical reference](technical-reference/index.md) and [DR-TMLE contract](technical-reference/dr-tmle/index.md), validation results in
[evidence manifest](technical-reference/evidence.md), and cross-module standing decisions in the
[architecture invariants](architecture-invariants.md).

The main grid orders open work by beta milestone. Complete a milestone's dependencies before its
rows. An item enters the grid through one of the acceptance routes in [Eligibility](#eligibility).
An item that needs a new identification, regularity, or asymptotic argument stays in the future
investigations grid.

`cleverly` is alpha, and it keeps no backward compatibility. `cleverly.load` warns with
`VersionMismatchWarning` when a different version saved the artifact, and it does not migrate the
artifact. This roadmap therefore carries no compatibility or migration items. The "Compatibility"
section of `CLAUDE.md` states the rule.

This file lists open work only. A delivered row leaves the file. Code comments and evidence pages
still name delivered rows, RM8 to RM33, by their IDs.
[The roadmap at commit `4ce96cda`](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md)
holds the record of each one.

The natural-extension review left the file when it was
complete. [The roadmap at commit `6a55f038`](https://github.com/esbraun/cleverly-tmle/blob/6a55f0388b703765eab79c78efec82474e67ab8e/docs/roadmap.md#x18-natural-extension-reviews)
holds its questions, and the [natural-extension verdicts](technical-reference/natural-extension-verdicts.md) hold its answers.
[The roadmap at commit `287a1fb5`](https://github.com/esbraun/cleverly-tmle/blob/287a1fb5385ad2cc0faa75e05672a3233012c236/docs/roadmap.md#rm40-correct-the-dr-tmle-inference-claim-when-one-nuisance-is-wrong)
holds the record of RM40, the DR-TMLE one-wrong inference correction.
[The roadmap at commit `2a8ad955`](https://github.com/esbraun/cleverly-tmle/blob/2a8ad9552e7aeb19f171643da5311eb0c6b03a91/docs/roadmap.md#rm39-keep-a-study-run-alive-through-a-failed-replicate)
holds the record of RM39, the failure record and accounting of a study run.

## Remediation

This queue holds the remediation work that must be complete before a beta release. It holds
shipped behavior that needs a correction or a recorded decision, and a published method needed to
resolve a shipped refusal. Examples are wrong numbers, intervals that no claimed contract covers,
capability rows that do not match their calls, late refusals, and shipped outputs that no
registered study measures. The queue also holds a defect of the validation harness that can lose
or invalidate a registered study run. Deliver the rows in priority order, and complete every row
before the first beta release.

A new capability still needs its own contract and evidence, even in this queue.

No remediation row is open.

Each row takes a tier by the harm that its defect does to a user today. The table gives the tiers,
from the most harmful. Inside a tier, a row with a wider reach comes first. A row that another row
depends on comes before that row.

| tier | reason | rows |
| --- | --- | --- |
| a | a published number that is wrong, or that no derivation or read source covers. An anti-conservative number ranks above a conservative one | no open row |
| b | a crash, an exception that is not a refusal, a capability row that reads available and then raises, or an assessment that returns no report | no open row |
| c | a correct refusal that arrives late or as the wrong type | no open row |
| d | a diagnostic or a warning that misleads | no open row |
| e | a display or a message that misstates a fact that the fit records. By extension, an argument check or a capability row that misstates what a call accepts or needs, when no number moves and nothing raises that is not a refusal | no open row |
| f | an investigation, a declared design that moves no verdict, or a shipped output that no registered study measures | no open row |
| g | a published method needed to resolve a shipped refusal | no open row |

A remediation row carries a priority below 1. The first decimal digit names a delivery group, and
the second digit orders the rows inside it. A main-roadmap row carries a milestone instead. This
project reassigns priorities and milestones when it re-triages the queue. The IDs and their anchors never
change, so a commit names a row by its ID. A new row takes the next number that the history of this
file has not used.

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) lists every red verdict
that a registered study publishes. [Red-cell owners](#red-cell-owners) names the item that owns
each one. `tests/unit/test_red_cell_ledger.py` checks the ledger against the committed results.
Each red verdict stays red under a `reporting` policy, so no verdict is hidden and no margin moves.

## Main roadmap

### Beta objective

The first beta makes `cleverly` a practical Python package for the effects of decisions made over
time. One documented workflow answers each question in the table below.

| question | required analysis |
| --- | --- |
| What happens if customers receive different sequences of offers? | static categorical regimen means and contrasts |
| What happens if the next offer depends on engagement? | prespecified dynamic categorical policies |
| What happens if prices are reduced at several decision times? | longitudinal continuous modified treatment policies |
| What happens if an offer is randomized with a specified probability? | known stochastic treatment policies |
| How sensitive is the result to modeling and weak support? | several estimators, support reports, and descriptive stability checks |

The workflow stays `CausalStudy -> identify -> estimate -> assess`. Methods and policies attach to
the causal question, not to a separate API for each implementation. Prespecified policies come
first. Policy learning follows after honest evaluation is available.

This section is the plan for future work. The package already ships much of each milestone, and
the [user guide](user-guide/index.md) and [technical reference](technical-reference/index.md)
describe that work.

LTMLE and `SDR` already fit static, dynamic, categorical, known stochastic, and
modified treatment policies, including continuous treatment nodes. The package also ships
clustered and cross-fitted longitudinal TMLE with the pooled update. It ships the cross-fitted
longitudinal MSM, simultaneous bands over declared policies, and the full-refit bootstrap. The
cross-fitted and cluster bootstrap kinds stay diagnostic. The rows below hold the open work only.

### Milestones

The milestones order work by dependency and release role. Documentation, interface prototypes, and
comparator preparation can proceed while an affected estimator is under repair. The milestones set
scope, not dates.

| milestone | deliverable | dependency | release role |
| --- | --- | --- | --- |
| B0 | the acceptance policy, the status and refusal contract, and accurate refusals | none | the foundation for each beta claim |
| B1 | the trajectory contract, the panel adapter, and evidence over longer horizons | B0 | a reliable sequential core |
| B2 | validated continuous longitudinal modified treatment policies | B1 | required for the pricing beta |
| B3 | sequential baselines and the two applied tutorials | B1; B2 for the continuous examples | completes the first beta |
| B4 | categorical incremental policies and repeated longitudinal cross-fitting | the relevant B1 to B3 engine | the prioritized beta expansion |
| BP | the beta parity group: the former priority-1 rows that this reorganization does not place in B0 to B4 | their own dependencies | required for the first beta |
| BD | the worked examples and the documentation, reviewed against the methods that ship at that time | every row of milestones B0 to B4 and BP, merged | the last work before the first beta |
| B5 | honest learned-policy evaluation and selected production extensions | a stable evaluation engine | later beta releases |

The first beta needs B0 to B3, BP, and BD. It does not need every estimator to support every policy,
outcome, weighting, or dependence combination. It needs a usable path for each promised question,
and honest boundaries for the rest. The release process sets the beta version. This roadmap does
not change the package version.

| milestone | item | readiness | dependency | details |
| --- | --- | --- | --- | --- |
| B0 | Experimental status and support rows | no new theory | none | [RM42](#rm42-experimental-status-and-support-rows) |
| B1 | Longitudinal evidence over three to five decisions | published support | the shipped `LTMLE` and `SDR` | [X39](#x39-longitudinal-evidence-over-three-to-five-decisions) |
| B1 | Targeting and fitting options | no new theory; each option keeps its default bit-identical | none | [X22](#x22-targeting-and-fitting-options) |
| B2 | Continuous vector components at a node | published support; pending source read | shipped longitudinal modified treatment policies | [X31](#x31-continuous-vector-components-at-a-node) |
| B2 | Continuous-policy validation and support checks | published support | shipped longitudinal modified treatment policies | [X40](#x40-continuous-policy-validation-and-support-checks) |
| B3 | Applied tutorials for offers and pricing | no new theory | X40; the shipped `SequentialGComputation` and `SequentialIPW` | [X42](#x42-applied-tutorials-for-offers-and-pricing) |
| B4 | Categorical incremental policies | natural extension candidate; source audit | the shipped binary `IPSI` estimator | [X43](#x43-categorical-incremental-policies) |
| B4 | Repeated cross-fitting for longitudinal fits | published support; pending source read | the shipped point-treatment repeats and trajectory-level splits | [X46](#x46-repeated-cross-fitting-for-longitudinal-fits) |
| BP | Repeated outcomes for SDR and IPSI | published support for the incremental case; a per-horizon reduction for `SDR` | the shipped `LTMLE` repeated outcome | [X32b](#x32b-repeated-outcomes-for-sdr-and-ipsi) |
| BP | Adaptive-propensity IPTW | published support; pending source read | none | [X21](#x21-adaptive-propensity-iptw) |
| BP | C-TMLE candidate sequences | published support; pending source read | shipped selector paths | [X16](#x16-c-tmle-candidate-sequences) |
| BP | Plug-in omitted-variable limits with a declared parametric mechanism | [published support; stacked-equation extension](technical-reference/natural-extension-verdicts.md) | the shipped plug-in bound | [X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism) |
| BP | Per-arm outcome-adaptive C-TMLE intervals with a parametric outcome | [natural extension](technical-reference/hard-stop-review.md) candidate; oracle diagnostic | RM42; the shipped per-arm design | [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) |
| BD | Worked examples for the shipped methods | no new theory | every row of milestones B0 to B4 and BP | [X68](#x68-worked-examples-for-the-shipped-methods) |
| BD | Documentation review for new and changed methods | no new theory | every row of milestones B0 to B4 and BP; X68 | [X69](#x69-documentation-review-for-new-and-changed-methods) |
| B5 | Learned-policy follow-ups | published support for parts (g) and (h); natural extension for parts (i) and (k) to (o); published support; pending source read for parts (a), (b), (e) and (f); source audit for parts (c), (d) and (j) | the shipped learned-rule value | [X11](#x11-learned-policy-follow-ups) |
| B5 | Structural dynamic treatment effects | source audit | X41 | [X44](#x44-structural-dynamic-treatment-effects) |
| B5 | Parametric forward-simulation g-formula | source audit | X41 | [X45](#x45-parametric-forward-simulation-g-formula) |
| B5 | Longitudinal simulated-confounding replay with a time-fixed latent | natural extension | the shipped point-treatment replay | [X48](#x48-longitudinal-simulated-confounding-replay-with-a-time-fixed-latent) |
| after beta | Replicate-weight designs | source audit | weighted-law variance construction | [X2](#x2-replicate-weight-designs) |
| after beta | Efficient known-score ATT and ATC | published support; pending source read | the shipped known treatment mechanism | [X35](#x35-efficient-known-score-att-and-atc) |
| after beta | Natural and interventional mediation effects | published support; pending source read | target-specific identification and evidence | [X5](#x5-natural-and-interventional-mediation-effects) |
| after beta | Continuous-time survival and competing risks | published support; pending source read | continuous-time intensity and targeting contracts | [X6](#x6-continuous-time-survival-and-competing-risks) |
| after beta | Hazard targeting beyond a held baseline treatment | published support; pending source read | the shipped hazard targeting on a held baseline treatment | [X70](#x70-hazard-targeting-beyond-a-held-baseline-treatment) |
| after beta | Two-phase and outcome-dependent sampling | published support; pending source read | observed-data likelihood and influence correction | [X7](#x7-two-phase-and-outcome-dependent-sampling) |
| after beta | Omitted-variable bounds on the other linear functionals | published support; pending source read | the shipped arm-axis bound | [X9](#x9-omitted-variable-bounds-on-the-other-linear-functionals) |
| after beta | Continuous-dose MSM with a second mechanism | source audit | implemented continuous-dose MSM targeting, missing-outcome arm targeting, and controlled-direct-effect targeting | [X10](#x10-continuous-dose-msm-with-a-second-mechanism) |
| after beta | Fold-evaluated CV-TMLE with baseline strata | [published support; finite-partition extension](technical-reference/natural-extension-verdicts.md) | the shipped fold-evaluated estimate and the shipped stratified fluctuation | [X28](#x28-fold-evaluated-cv-tmle-with-baseline-strata) |
| after beta | Simulated-confounding laws for point-treatment fits | [natural extension](technical-reference/hard-stop-review.md); no new theory for part (d) | the shipped point-treatment replay | [X47](#x47-simulated-confounding-laws-for-point-treatment-fits) |
| after beta | E-value for a controlled direct effect | [natural extension](technical-reference/hard-stop-review.md) | the shipped E-value conversions | [X49](#x49-e-value-for-a-controlled-direct-effect) |
| after beta | Further DR-TMLE compositions | [natural extension](technical-reference/hard-stop-review.md); no new theory for parts (d) and (e) | the shipped `DRTMLE` | [X50](#x50-further-dr-tmle-compositions) |
| after beta | C-TMLE point estimates for further targets | [natural extension](technical-reference/hard-stop-review.md); no new theory for part (f) | the shipped selector and outcome-adaptive paths | [X51](#x51-c-tmle-point-estimates-for-further-targets) |
| after beta | Parametric C-TMLE intervals | [natural extension](technical-reference/hard-stop-review.md) | the shipped selector path; X58 for part (b) | [X52](#x52-parametric-c-tmle-intervals) |
| after beta | Missing-outcome CV-TMLE variants | [natural extension](technical-reference/hard-stop-review.md); pending source read for the incremental and tilt axes of part (d) | the shipped stacked missing-outcome contracts | [X53](#x53-missing-outcome-cv-tmle-variants) |
| after beta | Small-sample evidence for clustered intervals and boundary nodes | [no new theory](technical-reference/hard-stop-review.md) | the shipped clustered intervals and `constant_node_plugin` status | [X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) |
| after beta | Bootstrap extensions | [published support](technical-reference/hard-stop-review.md); the book chapter of part (c) is not read | the shipped full-refit bootstrap; a HAL learner for part (b) | [X55](#x55-bootstrap-extensions) |
| after beta | Competing-event intervention targets | [published support](technical-reference/hard-stop-review.md) for part (a); source audit for parts (b) and (c) | the shipped censored `LTMLE` | [X56](#x56-competing-event-intervention-targets) |
| after beta | Joint intervention axes and tilt compositions | [natural extension](technical-reference/hard-stop-review.md) | the shipped single-axis fits and the MNAR tilt | [X57](#x57-joint-intervention-axes-and-tilt-compositions) |
| after beta | Settings of the stacked and natural-course contracts | no new theory for part (a); natural extension for parts (b) to (d) | the shipped stacked and natural-course contracts; the shipped controlled direct effect for part (d) | [X60](#x60-settings-of-the-stacked-and-natural-course-contracts) |
| after beta | Fold targeting and fold evaluation | natural extension for parts (a) to (c) | the shipped fold-specific update and fold-evaluated report | [X61](#x61-fold-targeting-and-fold-evaluation) |
| after beta | Other DR-TMLE and C-TMLE compositions | natural extension for parts (a), (c) and (d); no new theory for part (b) | the shipped `DRTMLE` and `CTMLE`; X50 part (c) and X51 part (a) for part (d) | [X62](#x62-other-dr-tmle-and-c-tmle-compositions) |
| after beta | Missing data, intermediates and designs | natural extension for parts (a) and (b); no new theory for parts (c) and (d) | the shipped controlled direct effect and known treatment mechanism | [X63](#x63-missing-data-intermediates-and-designs) |
| after beta | Intervention and policy declarations | natural extension for parts (a) to (d); no new theory for part (e) | the shipped intervention declarations | [X64](#x64-intervention-and-policy-declarations) |
| after beta | Sensitivity analyses beyond the shipped scope | natural extension for parts (a), (b), (f), (h) and (i); no new theory for parts (c) to (e), (g) and (j) | the shipped sensitivity analyses; X61 part (c) and X63 part (c); a continuous natural-course mean for part (a) | [X65](#x65-sensitivity-analyses-beyond-the-shipped-scope) |
| after beta | Simulated confounding beyond X47 and X48 | natural extension | X47; X57 for part (c); X50, X51 and X62 for part (h); F5 for a continuous fit in part (f) | [X66](#x66-simulated-confounding-beyond-x47-and-x48) |
| after beta | Refutation and assessment adapters | no new theory | the shipped `refute` catalog | [X67](#x67-refutation-and-assessment-adapters) |
| after beta | Repeated outcomes for the sequential baselines | no new theory | the shipped `LTMLE` repeated outcome; X41 | [X71](#x71-repeated-outcomes-for-the-sequential-baselines) |
| after beta | EP learner | published support; pending source read | shared study, fold, learner, and assessment contracts; a concrete analysis need | [P1](#p1-ep-learner) |
| after beta | Nested Riesz engine and initial catalog | published support; source audit complete | typed study, identification, result, and assessment contracts; a concrete analysis need | [R1](#r1-nested-riesz-engine-and-initial-catalog) |
| after beta | Evidence-gated Riesz catalog expansion | source audit for each target | R1 and a target-specific derivation | [R2](#r2-evidence-gated-riesz-catalog-expansion) |

Rows in progress keep their order within a milestone. The BP rows close estimator gaps against the
pinned comparators `ctmle` 0.1.2, `ctmle3` 0.1.0, `tmle3` 0.2.0, `drtmle` 1.1.2 and `lmtp` 1.5.4.
They also close gaps against the survival packages `survtmle`, `concrete` and MOSS.
X2 moves forward only if a pilot analysis needs survey weights. P1, R1, and R2 move forward only
when they reduce a concrete implementation burden of a beta row.

### Validate the claim that a release makes

Each released configuration has a support row and an evidence status that matches its output.
Declare the supported configuration cells first, then choose the studies that cover them. Reuse
existing evidence for unchanged components, and add checks for each changed construction.

| check | purpose |
| --- | --- |
| an exact finite-law or analytic continuous-law calculation | verifies the intended functional independently of the estimator |
| influence-function derivative and remainder checks | verify the changed derivative terms and the stated robustness conditions |
| nonzero updates and deliberate mutations | detect mistakes that vanish under oracle nuisances |
| acceptable comparator runs | establish parity where appropriate, or give informative component and statistical comparisons with documented differences |
| end-to-end learned-nuisance studies | evaluate the user pipeline, not only supplied oracle predictions |
| repeated sampling | measures bias, coverage, SE calibration, and behavior under weak support |
| executable applied examples | verify the public workflow, messages, assumptions, and result persistence |

When no comparator exists, the functional, derivative, mutation, and statistical checks carry the
claim, as [Theory and comparators](architecture-invariants.md#theory-and-comparators) states.
Record comparator commits, dependencies, learner settings, supplied predictions, trimming, and
variance conventions. Numerical parity and statistical performance are separate claims. When a
study compares oracle mechanisms, add learned-mechanism evidence before a production-readiness
claim. Keep historical results and preset margins, and publish failed or inconclusive results
under the [red-cell rule](development/method-benchmarking.md#red-cells).

The first beta is ready when each statement below holds:

- the remediation queue is empty;
- B0 is adopted, and any known incorrect output or uncertainty is fixed or outside the supported
  claims;
- categorical `LTMLE` and `SDR` configurations, with paired effects and censoring where
  validated, carry evidence over three to five decisions;
- continuous longitudinal modified treatment policies pass their support checks and carry
  explicit support and identification assumptions;
- known stochastic binary and finite-category policies carry evidence over three to five
  decisions ([X39](#x39-longitudinal-evidence-over-three-to-five-decisions));
- sequential g-computation and IPW are available over a declared subset, with correctly scoped
  uncertainty;
- both applied tutorials run through the public API and explain method choice and causal
  assumptions;
- the BP rows are delivered; and
- the code, documentation, distribution, and affected-study checks of the release process pass.

## Future investigations

This grid is the research queue. Each row holds a claim that needs an argument of a new kind. Move
a claim into the main roadmap when published theory supplies that argument, or when the
[Eligibility](#eligibility) rule qualifies it as a natural extension. A missing comparator never
keeps an item here, and a comparator never removes a stop. The
[hard-stop review](technical-reference/hard-stop-review.md) read each earlier row under that rule.
It moved each part that an established method or a natural extension carries to a main-roadmap row.

| kind of missing argument | what it means |
| --- | --- |
| identification | a law or an assumption that identifies the target from the observed data |
| regularity | a pathwise derivative of the target, or a model in which the estimator is regular |
| asymptotic | a limit theorem for the estimator as the package builds it |
| adaptive-optimum inference | a limit law after a data-adaptive selection, or for a target with no fixed limit |
| arbitrary dependence | an argument for a fixed number of dependent units |
| serial dependence | a limit theorem for split-based TMLE under mixing |

| investigation | kept claim | kind of missing argument | current boundary | details |
| --- | --- | --- | --- | --- |
| Missing-outcome simulated-confounding replay | a replay of the stored estimator when the response or treatment-observation mechanism reads the treatment | identification | complete outcomes only. [X47](#x47-simulated-confounding-laws-for-point-treatment-fits) holds a response mechanism that reads the covariates only | [F12](#f12-missing-outcome-simulated-confounding-replay) |
| Longitudinal simulated-confounding replay | censoring or survival outcomes; a time-varying covariate after the first treatment node; a time-varying latent with treatment feedback | identification | point-treatment results only. [X48](#x48-longitudinal-simulated-confounding-replay-with-a-time-fixed-latent) holds one time-fixed latent on uncensored terminal-outcome fits | [F13](#f13-longitudinal-simulated-confounding-replay) |
| Controlled-direct-effect simulated-confounding replay | the treatment axis on a fit with an intermediate | identification | fits without an intermediate only. [X47](#x47-simulated-confounding-laws-for-point-treatment-fits) holds the outcome axis and an intermediate axis that holds the treatment | [F15](#f15-controlled-direct-effect-simulated-confounding-replay) |
| Plug-in omitted-variable confidence limits | confidence limits of the plug-in $E_n[\hat\alpha^2]$ for a learned mechanism other than the declared model of [X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism) | regularity | the plug-in bounds, `rv`, `max_bias`, `benchmark()` and `contour()`. The one-sided limits and `rva` refuse under `nu2_estimator="plugin"`. `nu2_estimator="doubly_robust"` gives them for any mechanism learner | [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) |
| Targeted bootstrap inference | full-refit bootstrap consistency with data-adaptive or cross-fitted nuisances; a row-resampling (nonparametric) targeted bootstrap for learners other than HAL | asymptotic | the in-sample point-treatment `TMLE` bootstrap publishes its percentile interval, with a data-adaptive learner too, although the registered study measured only correctly specified nuisances. A cross-fitted point bootstrap, and every `DRTMLE` and `CTMLE` bootstrap, prints as a diagnostic. [X55](#x55-bootstrap-extensions) holds the extensions that published support carries | [F2](#f2-targeted-bootstrap-inference) |
| Longitudinal sensitivity-bound estimation | (a) a sample estimate of the bounds of Tan (2025); (b) its interval; (c) an omitted-variable bound or an E-value on a longitudinal fit | (a) asymptotic; (b) regularity; (c) identification | no sensitivity bound on a longitudinal fit | [F16](#f16-longitudinal-sensitivity-bound-estimation) |
| Other refused C-TMLE and DR-TMLE compositions | C-TMLE ATT, ATC, incremental, modified treatment policies, the risk-ratio tilt, controlled direct effects, continuous doses and longitudinal functionals; DR-TMLE ATT, ATC, modified treatment policies, the risk-ratio tilt, incremental, controlled direct effects, longitudinal functionals, continuous doses, a partial guard with missing outcomes, fold targeting and fold evaluation; DR-TMLE combined with C-TMLE | regularity for the C-TMLE targets; asymptotic for the DR-TMLE targets | named pre-fit refusals. [X50](#x50-further-dr-tmle-compositions) and [X51](#x51-c-tmle-point-estimates-for-further-targets) hold the compositions that moved, and [X62](#x62-other-dr-tmle-and-c-tmle-compositions) holds the others | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| Selector-path C-TMLE inference | inference after the shipped stopping-index selection; a fold-specific selection under `cv_evaluation=True` or `targeting_scheme="fold"` | adaptive-optimum inference | point estimates and path diagnostics. The greedy, ordered and discrete paths refuse `ci`, `pvalue` and `std_error`, except a `discrete` fit whose one declared candidate is the full adjustment set. [X52](#x52-parametric-c-tmle-intervals) holds one declared partial candidate with parametric learners | [F18](#f18-selector-path-c-tmle-inference) |
| Outcome-adaptive C-TMLE generated-design inference | a flexible outcome learner; a nonparametric shared multinomial design; the cross-fitted construction; the fold average and a fold-specific update; local uniformity over the nonparametric model | asymptotic; regularity for the shared design | point estimates only, under the `generated_design_plugin` status. The per-arm study measured the diagnostic 8 to 24 percent too small ([the applied revert](technical-reference/collaborative-tmle.md#the-applied-revert)). [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) holds the per-arm cells with a parametric outcome, and [X52](#x52-parametric-c-tmle-intervals) holds the shared and composite parametric cells | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| Other missing-outcome CV-TMLE variants | a cross-fitted `DRTMLE` with `delta=` or a missing treatment, and `reduced_crossfit="nested"` on the composite and on the randomized missing-outcome `DRTMLE` | asymptotic | the in-sample composite `DRTMLE`. [X53](#x53-missing-outcome-cv-tmle-variants) holds the `TMLE` variants | [F21](#f21-other-missing-outcome-cv-tmle-variants) |
| Grouped cross-fitting beyond point-treatment TMLE | clustered C-TMLE intervals beyond the parametric cells of X52 | asymptotic | C-TMLE refuses `id=`. [X51](#x51-c-tmle-point-estimates-for-further-targets) holds the clustered point estimates | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| Time-respecting cross-fitting | blocked-temporal and rolling-origin folds for a dependent calendar series | serial dependence | iid and grouped cross-fitting only | [F7](#f7-time-respecting-cross-fitting) |
| Learned-policy value outside the published conditions | an interval at an exceptional law; the full-refit bootstrap; a continuous treatment; a fold-evaluated `DRTMLE` learned rule; `CTMLE` with `learned_rule=`; a learned rule with missing outcomes or a missing treatment; an E-value for a learned-rule contrast | adaptive-optimum inference; asymptotic; regularity | `LearnedRuleValue` fits the fold-evaluated CV-TMLE of [learned rules](technical-reference/point-treatment-tmle.md#learned-rules) under the limiting-rule condition. Its boundary study measures under-coverage at an exceptional law. [X11](#x11-learned-policy-follow-ups) holds the compositions that moved | [F27](#f27-learned-policy-value-outside-the-published-conditions) |
| Inference at a boundary node estimate | an interval in the rare-event local regime, where the node hazard shrinks with $n$ | asymptotic | the point estimate stands under the `constant_node_plugin` status. [X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) holds the interval at a fixed law | [F31](#f31-inference-at-a-boundary-node-estimate) |
| Finite-sample limits of clustered intervals | an interval below 4 clusters | arbitrary dependence | `few_cluster_plugin` below 10 positive-mass clusters, and below 20 for `LTMLE`. [X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) holds the measurements from 4 clusters up to those floors | [F28](#f28-finite-sample-limits-of-clustered-intervals) |
| Fold policies that read the treatment or the outcome | outer folds stratified on the treatment or the outcome | asymptotic | unstratified folds, `stratify_by="none"`, which is the default. A split that reads $A$ or $Y$ refuses before any learner | [F32](#f32-fold-policies-that-read-the-treatment-or-the-outcome) |
| Other intervention and design compositions | the composite missing-treatment indicator with a controlled direct effect; an `IPSI` plan with a modified-policy or risk-ratio-tilt node; an MSM weight from a learned mechanism; automatic cumulative-bound selection for `LTMLE` | identification; regularity; adaptive-optimum inference | named pre-fit refusals. [X63](#x63-missing-data-intermediates-and-designs) and [X64](#x64-intervention-and-policy-declarations) hold the declarations that a natural extension carries | [F33](#f33-other-intervention-and-design-compositions) |
| E-values beyond the total-population arm contrast | the ATT and ATC E-value on a binomial outcome; the E-value of a regime, policy or incremental contrast, with the derived risk ratio off the arm axis | identification | E-values of total-population arm contrasts, and the standardized ATT and ATC E-value on a gaussian outcome. [X49](#x49-e-value-for-a-controlled-direct-effect) holds the controlled direct effect, and [X65](#x65-sensitivity-analyses-beyond-the-shipped-scope) the other sensitivity requests | [F34](#f34-e-values-beyond-the-total-population-arm-contrast) |
| Inference for sequential baselines with data-adaptive nuisances | an asymptotic-linearity result for longitudinal sequential g-computation or IPW whose nuisances are fitted by a data-adaptive learner | asymptotic | the point estimate stands. A fit that reads a regression fitted by a learner other than `GLM` takes the `adaptive_baseline_plugin` status and reports a plug-in diagnostic only | [F35](#f35-inference-for-sequential-baselines-with-data-adaptive-nuisances) |

## Eligibility

`cleverly` accepts established methods and natural extensions of established theory. Published
support or a natural extension supplies the theory. An acceptable comparison with an existing
implementation can then supply acceptance evidence for the claims it covers.

A natural extension
is a bounded adaptation of known results. It needs a written statement of its target,
assumptions, estimating equations, and inference. It does not need a separate published theorem
for each composition. When the remaining work is such an adaptation, implement it.

Theory and comparators are separate questions.
[Theory and comparators](architecture-invariants.md#theory-and-comparators) is the repository
standard for both. The absence of a comparator, exact or related, never blocks an item. Missing
fundamental theory blocks the affected claim when a natural extension of established results
cannot supply it. Defer that method, configuration, or inference claim, and keep every unrelated
supported capability. Parity and related comparisons do not excuse a known defect or a misleading
uncertainty estimate.

### Three acceptance routes

| route | sufficient justification | required evidence |
| --- | --- | --- |
| established method | a published method covers the target and the proposed estimator | an assumption map, implementation checks, and scoped statistical validation |
| acceptable implementation comparison | a credible reference gives informative evidence for a method that the first or the third route already covers in theory | the documented target and algorithm differences, a reproducible reference, justified comparison metrics, and independent checks of each claim the comparison does not cover |
| natural extension | existing results extend through routine finite-dimensional or algebraic arguments | a short derivation of the changed terms, explicit assumptions, and tests that expose the change |

A natural extension satisfies every condition in the table below. The
[natural-extension verdicts](technical-reference/natural-extension-verdicts.md) record earlier
readings under this rule.

| condition | what it requires |
| --- | --- |
| a published base | the extension starts from one published result, and it cites that result's exact locator |
| an established argument | each step reuses the base result's own argument, or a standard result. The next table lists the standard results |
| no new fundamental argument | no step needs a new identification, regularity, or asymptotic argument, and no source records a step as open or as a defect |
| inherited conditions | the remainder and rate conditions carry over from the base result, and the contract states each one |
| a written record | the item's contract states the base result, each step, and the argument that carries it |
| its own evidence | each step that can vanish at the truth has a nonzero witness. An interval or another repeated-sampling claim also needs a registered study |

Each step below counts as an established argument. A step of this kind needs no further source.

| standard step | what it carries |
| --- | --- |
| linearity | a sum, a difference or a fixed linear functional of asymptotically linear estimates, such as a contrast, an attributable risk or a restricted mean |
| the delta method | a smooth function of a fixed number of estimates, such as a ratio, a fraction or a transform |
| a fixed-dimension stack | estimates that are each asymptotically linear on the same rows, stacked by Cramér–Wold, with their joint covariance from the stacked curves |
| the chain rule for influence functions | a composition of differentiable functionals, including a known weight or a known design |
| an indicator reduction | a binary-treatment result applied to $1\{A = a\}$ for one arm, or to a composite indicator such as $\Delta \cdot 1\{A = a\}$, when identification holds for that indicator |
| a finite partition | a marginal result applied inside each of a fixed number of baseline strata, then stacked |
| a finite categorical sum | a result for one category applied to each of a fixed number of categories, and summed with known weights that do not depend on the law. At one node, the weights can be known functions of the history at that node, such as a known stochastic policy. Over several nodes the step applies inside each node of the sequential regression, and the chain rule carries the composition |
| a fixed full-rank projection | a fixed working model, with known projection weights, projected over a fixed finite set of jointly asymptotically linear estimates, such as an MSM over declared policies. The derivative of the projection equation is nonsingular at the truth. Estimated projection weights add a derivative term and are outside this step |
| a cluster as the unit | an iid result applied to independent clusters as the sampling units, with cluster-summed curves |
| a stacked estimating equation | a smooth finite-dimensional nuisance whose score is stacked with the target equation, as Saul and Hudgens (2020) describe |

Each adaptation explains why its assumptions and its remainder still apply. Apply the conditions
to reach a verdict, not to find a reason to wait. When every step is in the table above, the
burden falls on an objection: a source that records the step as open or as a defect, or a required
argument of a new kind. Without such an objection, the extension qualifies. A reading that
rejected an extension before this rule is open to a new reading under it.

An extension that fails one condition is new theory. It stays in the future investigations grid.
These examples fail:

| example | why it fails |
| --- | --- |
| inference after a data-adaptive selection step, such as an adaptively selected policy | no established argument carries the selection |
| a fluctuation fitted on each fold's training rows in place of a pooled one | the update reuses the rows that fit the initial nuisances, so the held-out influence function is not mean zero. Upstream `lmtp` commit `9996b04` records this as a bug |
| a growing target dimension | it needs a limit theorem of a new kind |
| arbitrary dependence between units | no independent-unit or independent-cluster argument applies |
| the mean outcome when every unit takes one fixed value of a continuous treatment, in a nonparametric model | the target is not pathwise differentiable, so regular root-n inference does not carry over |

No route accepts an influence function that omits a required term. An intervention law that
depends on the observed law, such as a propensity tilt, adds a derivative term. A comparator that
omits the term cannot justify its omission.

When a proposed algorithm needs new fundamental work, use an existing covered estimator of the same
target where one exists.

### Comparisons

An exact comparator is useful when one exists. Neither an exact nor a related comparator is an
eligibility requirement or a universal release gate. Do not defer a capability only because no
library implements the identical target, algorithm, fold construction, or composition. Use the
strongest feasible comparison and state its scope. A comparison is acceptance evidence for the
claims it covers. The evidence pages label each claim that rests on a comparison as
comparator-backed.

A planned comparator run that has not finished does not block a method either. The method ships
when its Python studies pass with their expected properties. Until the run finishes, the evidence
names the property studies and carries the label "comparator runs pending". A later change registers the
comparator results and fixes each defect that they show.

| comparison | appropriate use | claim it supports |
| --- | --- | --- |
| matching implementation | the same target, and an aligned algorithm and settings | numerical parity within declared tolerances |
| different estimator of a shared target | TMLE, SDR, IPW, or g-computation under aligned causal assumptions | target recovery, and comparative bias, coverage, efficiency, or stability |
| related target or component | an overlapping binary case, a categorical reduction, a density-ratio component, or a projection | evidence for the shared component or reduction. Checks of the extension supplement it |

Differences in learners, targeting, trimming, normalization, or finite-sample construction can be
acceptable. Document their consequences, and choose metrics that suit the comparison. Agreement on
a related target does not establish the full target. Validate the uncovered part through a written
adaptation and independent calculations or studies. Reject a comparison whose differences make
the claimed conclusion uninformative.

A comparator's bug remains a bug when `cleverly` reproduces it. Where code and paper disagree, the published derivation governs, and the discrepancy becomes a
nonzero regression or mutation test.

Record the comparator's source revision when one is available. That record identifies what ran.
It does not require an exact comparator method.

### Classify the analysis and its inference

| status | meaning | public behavior |
| --- | --- | --- |
| supported | an acceptance route and its scoped evidence are complete | normal availability, with stated assumptions and coverage limits |
| experimental | the target is justified, but an algorithm or inference extension has unresolved evidence | explicit opt-in. Each estimate and interval carries the label, and an unsupported interval is withheld |
| unavailable | the implementation is absent, the request is invalid, or a known defect makes the claimed analysis unreliable | a refusal before fitting where possible, which names the reason and an available alternative |

A supported point estimate does not make its confidence interval supported. Track support by the
combination of target, estimator, outcome, weighting, splitting, and inference. A configuration that
passed does not certify another configuration of the same method. The `experimental` status does
not exist in the package yet. [RM42](#rm42-experimental-status-and-support-rows) owns it.

The readiness labels rate method support, not programming effort:

- **published support**: a paper derives the method and inference claim;
- **natural extension**: a published base and the standard steps above carry the method and
  inference claim;
- **no new theory**: the item reuses shipped estimators and adds no statistical claim;
- **source audit**: a published paper appears to cover it, but the exact construction must be
  matched and discrepancies resolved before implementation;
- **waiting on published theory**: related methods exist, but the requested composition or
  inference claim needs a new identification, regularity, or asymptotic argument that no bounded
  extension supplies; it belongs only in the future investigations grid; and
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

## Red-cell owners

Each red cell in the [red-cell ledger](technical-reference/method-evidence/red-cells.md) names its
owner by an `id` in the table below. `tests/studies/evidence/red_cells.py` reads this table.
`tests/unit/test_red_cell_ledger.py` fails on four states: a red row with no owner, an owner with
no red row, an owner that the `id` column does not list, and a red row in a `gated` study.

Every red verdict is diagnosed first. A genuine problem (an algorithm defect, an inconsistent
estimator, or a test or design bug) is fixed, and the study re-runs under a fresh declaration that
states the change. A red cell with no defect stays red under `reporting` at its registered budget
and margin, with an owner row below. No change may make a test less meaningful or easier to pass:
moving a margin, adding replications to cross a bound, weakening the law, dropping or relabelling a
cell, or substituting an oracle. The
[red-cell rule](development/method-benchmarking.md#red-cells) gives the full text.

The `RM18-` owners hold the cells that went red when sixteen registered studies moved to
unstratified folds and bounded laws, and five longitudinal studies moved to the pooled update. A
declared diagnostic read each cell once. Each diagnostic directory and its
`tests/unit/test_rm18_*` test rebuild the reading from the committed rows. No reading names a
defect in `cleverly`.

The `strata-boundary-mean` and `band-finite-sample` owners hold joint-coverage and calibration
cells that a declared oracle-band diagnostic read as finite-sample.
`tests/unit/test_band_shortfall_reading.py` rebuilds each reading from the committed rows. Like
the `RM18-` owners, they are standing records of a reading. They do not gate the beta release.

The `X20-bootstrap` owner holds the bootstrap kinds that stay diagnostic. It does not gate the
beta release either.

The `X41-negative-control` owner holds the three `misspecification_control` cells of the
`sequential-baselines` study. They are a declared negative control and not a defect. On law `X`
every main-terms `GLM` is wrong, so the rule must fail. No work is owed.

The `F4-calibration-draws` owner holds two cells that read one draw set of the multi-arm
missing-outcome DR-TMLE study. It is a standing record of a reading, and it does not gate the
beta release.

The `composite-high-arm` owner holds the `ey[high]` truth row of both implementations in the
composite missing-data study. It is a standing record of a reading, and it does not gate the beta
release.

The `F1-power-design` owner holds the power cell of the known-policy study, whose declared size
gives it an exact power of 0.5365. It is a standing record of a reading, and it does not gate the beta
release.

The `mtp-point-calibration` owner holds one calibration cell of the point modified-treatment-policy
study. Its standard error is conservative, and the diagnosis finds no defect.

The `mtp-longitudinal-limits` owner holds one cell of the longitudinal modified-treatment-policy
study whose diagnosis finds no defect: a finite-sample calibration cell.

The `F28` owner holds the clustered cells of the unequal-size and few-cluster studies that read
red under their `reporting` policy. It owns a finite-sample limit, not a missing theorem.
[X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) is the work that
closes these cells.

The `X4-finite-sample` owner holds one coverage cell of the SDR study at n = 2,000. `LTMLE`
shares its shortfall on the same draws, so the diagnosis finds no defect in `SDR`.

The `X8-logit-small-stratum` and `X8-identity-small-stratum` owners hold cells of the two
stratified MSM studies in their smallest stratum. Each is a standing record of a reading, and
neither gates the beta release. The `X8-drtmle-one-sided-bias` and `X8-drtmle-small-stratum`
owners do the same for the stratified DR-TMLE study.

| id | work | acceptance |
| --- | --- | --- |
| `F18` | an inference result for the shipped selector path | an influence curve derived after the stopping-index selection, and a registered study whose `selector_necessity` and `type_i_error` cells pass their existing margins at their existing budgets. The multi-arm `interval_calibration/correctly_specified` cell must also pass its band, because its standard-error ratio interval, 0.8987 to 0.9862, measures the fixed-candidate covariance that F18 replaces |
| `F19` | an inference result for the generated design, the red cells of `ctmle-oat-per-arm`, and the short-SE truth rows and ladder rungs of `ctmle-oat-per-arm-strata-missing` | a published result, or an Eligibility-qualifying natural extension, that establishes whether the ordinary adaptive-propensity curve suffices for the cross-fitted shared-multinomial joint target. For the per-arm design, a curve that adds the generated-design term $c^\top \mathrm{IF}_\beta$, with $c = E[(1 - g_0 / G)\, \partial \bar Q / \partial \beta]$, or a source that removes it. Implement any contribution the result requires, then register covariance and coverage evidence at declared budgets. The per-arm revert cells must pass their declared bands under a fresh declaration. [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) plans the per-arm part for the configurations that its support predicate admits. The committed red cells keep their verdicts and stay owned here. The `generated_design` pair stays a reporting diagnostic, and acceptance does not require a negative standard-error deficit |
| `F27` | an interval for the fold-average learned-rule value outside the published conditions | [F27](#f27-learned-policy-value-outside-the-published-conditions) gives the acceptance for each request. The `exceptional` row needs a published result. The `weak_blip` row is a finite-sample limit at n = 2,000 inside condition C3 |
| `RM18-fixed-weights` | the `interval_calibration/static__correctly_specified` cell and the paired `ey_regimen[never]` row of `weighted-ltmle-crossfit` | reading `finite-sample, contracting` and `equivalent at the declared budget` ([`tests/diagnostics/rm18_fixed_weights/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_fixed_weights)). The pooled update applies the weighted targeting equation, so the reading is no reason to revert it. A result for complex surveys, estimated or calibrated weights, clustering or other longitudinal weight laws is still open. Karim (2026), arXiv:2606.30918v2, is adjacent point-treatment survey theory, and it does not cover this construction |
| `RM18-boundary` | cells at a finite-budget boundary: `root_n_and_efficiency/n_500` of `canonical-multi-arm-ctmle-selector`; `double_robust_contraction/treatment_correct_n1500` of `canonical-drtmle`; `double_robust_contraction/outcome_correct_n4000`, `interval_calibration/correctly_specified` and `root_n_and_efficiency/n_500` of `canonical-multi-arm-drtmle`; `double_robustness/static__both_wrong` and the paired `ey_regimen[always]` and `ey_regimen[treat then continue if l2 positive]` rows of `weighted-ltmle-crossfit` | reading `resolved: truth satisfies the gate` for the six cells, and `comparator SE convention` for the two paired rows ([`tests/diagnostics/rm18_boundary/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_boundary)). Before a future regeneration of `weighted-ltmle-crossfit`, declare whether it adopts the `lmtp` standard-error convention for the two paired rows |
| `RM18-one-sided-bias` | `double_robustness/outcome_correct` and `double_robustness/treatment_correct` of `canonical-drtmle`, and `double_robustness/treatment_correct` of `canonical-multi-arm-drtmle` | reading `shared` and `mixed` on the binary rows, and `finite-sample excess detected` on the multi-arm row ([`tests/diagnostics/rm18_one_sided_bias/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_one_sided_bias)). A declared localization design on 2,000 fresh draws read `no increment at the declared resolution` ([`tests/diagnostics/rm19_one_sided_increment/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm19_one_sided_increment)). No validated R comparator fits the multi-arm configuration, so the multi-arm excess has no attribution |
| `RM18-ordinary-weighted` | `interval_calibration/static__correctly_specified`, `type_i_error/static__sharp_null` and the four `targeting_necessity` cells of `weighted-ltmle` | reading `finite-sample, contracting`, `resolved within gate`, `control underpowered by design` and `family resolved` ([`tests/diagnostics/rm18_ordinary_weighted/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_ordinary_weighted)). The static untargeted control discriminates with a probability near 0.72 at the registered 1,200 replicates. Re-declare that control's law before any regeneration of the study, as the control precedent at `69de6f8` allows |
| `RM18-comparator-density` | the paired `ate_policy[+0.25 vs natural course]` row of `shift-policies` | reading `cleverly density representation` ([`tests/diagnostics/rm18_comparator_density/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_comparator_density)). The calibration leg fails with a 99% upper endpoint of 0.064762 against a margin of 0.05, so the conclusion is `inconclusive`. The modified-treatment-policy work kept the density representation. The design ran again after the estimand rename, which draws a new bootstrap stream, and the reading stands |
| `strata-boundary-mean` | the two primary truth rows of `ey[1][V=2]` and the `interval_calibration/v2_ey1__correctly_specified`, `simultaneous_coverage/strata__simultaneous_band` and `simultaneous_coverage/crossfit_strata__simultaneous_band` cells of `canonical-stratified-tmle` | reading `finite-sample Wald interval`: the treated mean of stratum V = 2 is 0.9358, and at n = 2,000 the stratum holds about 157 treated rows and about 9 expected non-events. R `tmle3` shows the same coverage on the same draws, the bias is inside its margin, and the oracle band at the exact critical value also under-covers ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then each cell stays published red under `reporting` |
| `band-finite-sample` | the `simultaneous_coverage/all_reported__simultaneous_band` cell of `canonical-ltmle-survival`, the `categorical_ltmle__simultaneous_band`, `categorical_ltmle_crossfit__simultaneous_band` and `cde_z0__simultaneous_band` cells of `default-simultaneous-bands`, and the `clustered_regimens__simultaneous_band` cell of `clustered-cross-fitted-ltmle`, and the `six_regimens__simultaneous_band` cell of `longitudinal-sdr` | reading `finite-sample, pointwise shortfall inherited`: the oracle band at the design critical value also covers below 0.95, and the source study's pointwise calibration of the same parameters sits near the lower edge of its band. The package critical value averages a little below the design one, so the multiplier explains a small part of each shortfall ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then each cell stays published red under `reporting` |
| `F4-calibration-draws` | the `interval_calibration/ate__correctly_specified` and `simultaneous_coverage/arms__simultaneous_band` cells of `multi-arm-mar-drtmle`, which read one set of 2,400 fits | reading `finite-sample, one draw set`. The SE-ratio interval ends at 0.9294 against a floor of 0.93, and the joint-coverage interval at 0.9196 against 0.92. The calibration fits have an empirical SD 1.037 times their mean standard error. The same configuration gives 0.997 at `n_2000` and 0.989 at `l3_ate_low__both_correct`, on independent draws. `simultaneous=True` changes no estimate and no standard error. The band critical value averages 2.500 against the oracle 2.508, and the oracle band covers 0.9342 against the package's 0.9337, so the band construction explains none of the shortfall ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). Pooled over the 4,400 independent both-correct fits of `ate[low vs high]` at n = 2,000, the spread is 1.020 times the mean standard error (bootstrap 99% interval 0.991 to 1.047), and a Bartlett test across the three cells gives p = 0.20, so the draw set differs from its siblings by no more than chance. No defect was found in the K-arm code. The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then both cells stay published red under `reporting` |
| `composite-high-arm` | the `ey[high]` truth rows of `cleverly-composite-drtmle` and `drtmle-r-composite` on the three-arm scenario of `composite-missing-drtmle` | reading `finite-sample, shared with the comparator` ([`tests/unit/test_composite_high_arm_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_composite_high_arm_reading.py)). Coverage is 0.9225 in the package and 0.9237 in R `drtmle`, and the 99% intervals end at 0.8949 and 0.8963 against a floor of 0.90. The paired mean difference is 4e-6. The mean standard error is 0.0342 in both, against an efficiency-bound SD of 0.0344 at n = 2,000, so the variance estimate is not low. The empirical SD of the estimates is 0.0367, 1.066 times the bound. The 47 package fits that reached `max_outer` cover at 0.957, against 0.921 for the fits that met the tolerance, so the cap is not the cause. `high` has the smallest composite mechanism of the law, 0.070 where `W = 0`, which holds half the mass. Every other arm and contrast of the scenario passes. The owner closes when a re-declared cell, with its law or size declared before its run, passes, or when a small-sample interval correction with its own registered study does. Until then both rows stay published red under `reporting` |
| `X4-finite-sample` | the `sequential_double_robustness/both_correct__n_2000` cell of `longitudinal-sdr` | reading `finite-sample, shared with LTMLE`. Coverage is 0.911, with a 99% interval of 0.885 to 0.933 against a floor of 0.90. The bias interval is inside its margin, and the SE ratio is 0.946. The law's smallest path probability is near 0.012, so an inverse weight can reach about 85. A diagnostic refitted the cell's fit and the shipped `LTMLE` with the same learners on 1,000 fresh draws ([`tests/diagnostics/x4_both_correct/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x4_both_correct)). Both cover 0.903, with SE ratios 0.918 and 0.917. Each standard error correlates with its own error, at 0.301 and 0.321. The same cell at n = 8,000 covers 0.950 with an SE ratio of 0.999 ([`tests/unit/test_sdr_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_sdr_shortfall_reading.py)). The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then the cell stays published red under `reporting` |
| `X8-logit-small-stratum` | the `interval_calibration/logit_v1_a__correctly_specified` and `interval_calibration/logit_v2_a__correctly_specified` cells of `canonical-stratified-incremental-msm` | reading `finite-sample, inherited from the unstratified fit` for stratum 2. The SE ratios are 0.967 and 0.862, and the lower ends of their intervals are 0.927 and 0.782 against a floor of 0.93. Coverage is 0.943 and 0.944. Stratum 1 holds about 600 of the 2,000 rows, and stratum 2 about 400. A declared diagnostic read stratum 2 only. It fitted the shipped unstratified logit MSM on samples of that size from the law given $V = 2$. That fit under-reports as well, 0.896 against the stratified fit's 0.867 on fresh draws, and both are calibrated at four times the size, 0.997 and 1.006 ([`tests/diagnostics/x8_logit_small_stratum/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x8_logit_small_stratum)). No diagnostic reads stratum 1, whose interval misses the floor by 0.003. The stratified fit equals the subset fit exactly under fixed nuisances (`tests/unit/test_stratified_msm_exact.py`), so no defect of the stratified construction was found. The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then both cells stay published red under `reporting` |
| `X8-identity-small-stratum` | the `interval_calibration/identity_v2_a__correctly_specified` cell and the paired `msm[W][V=2]` row of `canonical-stratified-msm-identity` | reading `finite-sample, smallest stratum`. Stratum 2 holds about 400 of the 2,000 rows. The calibration cell's SE-ratio interval is 0.926 to 1.004 against a floor of 0.93, and on the primary draws the same coefficient reads 0.983 in `cleverly` and 0.985 in R `tmle3`. The paired row is inconclusive on its calibration leg alone: the upper limit is 0.0523 against a margin of 0.05, at a resolution of 0.0476. The two estimates differ by 6e-6 on average with the same spread, and R reports a standard error 2.2% above its own spread, consistent with the declared nuisance difference: R fits a Gaussian `Lrnr_glm` inside each subset ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). No defect was found. The owner closes when a re-declared cell, with its law or size declared before its run, passes. Until then both stay published red under `reporting` |
| `X8-drtmle-one-sided-bias` | the four `double_robustness/*__treatment_correct` cells of `canonical-stratified-drtmle`: the marginal ATE and each stratum ATE | reading `shared` in strata 0 and 1 and `mixed` in stratum 2, in the vocabulary of `RM18-one-sided-bias` ([`tests/diagnostics/x8_drtmle_treatment_correct/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x8_drtmle_treatment_correct)). The refit reproduces the committed estimates exactly. On each stratum's rows, the shipped unstratified `DRTMLE` and R `drtmle` 1.1.2, handed the same initial arrays, carry a positive bias too. In stratum 2 the package's bias, 0.0237, exceeds R's, 0.0157: the paired difference is 0.0080 with a standard error of 0.0018. The unstratified fit on the stratum's rows matches the package there to 2e-5, so the excess belongs to `DRTMLE` at about 400 rows and not to the strata. In stratum 1 the stratified and subset fits differ by 0.0057 (standard error 0.0013). The stratified marginal is the mixture of the stratum estimates, so it inherits their bias: 0.56 of its spread at n = 2,000 (99% interval 0.43 to 0.69) and 0.43 at n = 8,000 (0.25 to 0.61). The intervals overlap, so contraction faster than the spread is not established. No defect of the stratified construction was found. The owner closes when a re-declared cell, with its law or size declared before its run, passes, or when `RM18-one-sided-bias` closes with a correction that also covers the stratified fit. Until then the four cells stay published red under `reporting` |
| `X8-drtmle-small-stratum` | the two primary truth rows of `ey[1][V=2]` and the `interval_calibration/v2_ate__correctly_specified` cell of `canonical-stratified-drtmle` | reading `finite-sample, shared with the comparator`. Stratum 2 holds about 400 of the 2,000 rows. Coverage of `ey[1][V=2]` is 0.900 in the package and 0.8925 in R `drtmle` on the same draws, with estimates 4e-5 apart on average. The calibration cell's SE-ratio interval ends at 0.9294 against a floor of 0.93, and on the primary draws the same ATE reads 0.985 in the package and 0.973 in R ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). The owner closes when a re-declared cell, with its law or size declared before its run, passes. Until then the three stay published red under `reporting` |
| `F1-power-design` | the `power/mix__alternative` cell of `stochastic-categorical-ltmle` | reading `underpowered at its declared size`. Its n = 4,000 was copied from `canonical-categorical-ltmle`, where the contrast is 0.125. Here the contrast is -0.0488, so the exact power of a two-sided 5% test is 0.5365, and exact power 0.80 needs n = 7,460. The cell rejects at 0.5575 (99% interval 0.511 to 0.603) against a floor of 0.80, and that interval contains the exact power ([`tests/unit/test_stochastic_categorical_ltmle_design.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_stochastic_categorical_ltmle_design.py)). Its bias is inside the margin and its coverage is 0.9475. The `type_i_error` cell, on the null variant of the law, passes. The owner closes when a re-declared power cell, with its law or size declared before its run, passes. Until then the cell stays published red under `reporting` |
| `mtp-longitudinal-limits` | the `interval_calibration/categorical_mtp__correctly_specified` cell of `longitudinal-mtp` | reading `finite-sample, sparse cells`: the efficiency ratios read 1.120 and 1.098 against a band of 0.9 to 1.1, and coverage 0.9435 and the SE ratio pass. The cell fits saturated cell probabilities for the six-level mechanism. A diagnostic refitted it on fresh draws. The in-sample fit reads a reported efficiency ratio of 1.103 at n = 2,000 and 1.023 at n = 8,000, and the five-fold fit 1.389 and 1.048, on 300 fresh draws per size. The reported ratio's excess falls about four times as n grows four times, the order of the cost of estimating sparse cell probabilities, and reads 1.003 at n = 128,000. A wrong bound or an inefficient curve would keep its excess. Both outcome regressions are misspecified, and the saturated mechanism makes the fit the NPMLE whatever the outcome learner, so the estimate is unaffected ([`tests/diagnostics/longitudinal_mtp_categorical_efficiency/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/longitudinal_mtp_categorical_efficiency), [`tests/unit/test_longitudinal_mtp_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_longitudinal_mtp_reading.py)). The cell read the same in runs 2 and 3. The owner closes when a re-declared cell, with its size declared before its run, passes. Until then the cell stays published red under `reporting` |
| `mtp-point-calibration` | the `interval_calibration/halve__correctly_specified` cell of `policy-point-mtp` | reading `Monte Carlo excursion`. The SE ratio is 1.0429, with a 99% interval of 1.0011 to 1.0881 against an upper bound of 1.07. It was 1.0430 at 160 bins, so the bin count does not move it. Coverage is 0.9605, and the reported standard error matches the efficiency bound (ratio 1.0018). The influence curve with the binned ratio predicts a ratio of 0.9998. A diagnostic refitted the same configuration on 2,000 fresh draws and read 0.999, with a bootstrap 99% interval of 0.959 to 1.043. Fewer than 1% of its bootstrap ratios reach 1.0429 ([`tests/diagnostics/mtp_point_halve_excursion/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/mtp_point_halve_excursion)). The diagnostic cannot change the verdict. The owner closes when a re-declared calibration cell for this policy passes, with a budget of no more than 2,000 replications fixed before its run. Until then the cell stays published red under `reporting` |
| `X13-finite-sample` | the `interval_calibration/weighted_t5__correctly_specified` cell of `point-treatment-survival` and the `interval_calibration/three_arm_t3__correctly_specified` cell of `point-treatment-survival-crossfit` | reading `finite-sample`. The weighted cell's SE ratio is 1.027, with a 99% interval of 0.981 to 1.076 against an upper bound of 1.07. Its reported standard error equals the weighted efficiency bound (ratio 1.0007), and the interval of its empirical efficiency ratio, 0.930 to 1.020, contains one. A diagnostic on 1,000 fresh draws ([`tests/diagnostics/x13_weighted_calibration/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x13_weighted_calibration)) reads an SE ratio of 1.022 and a coverage of 0.957. The cross-fitted three-arm cell's empirical efficiency interval is 1.010 to 1.108 against 1.10, with an SE ratio of 1.008. A diagnostic on fresh draws ([`tests/diagnostics/x13_crossfit_three_arm/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x13_crossfit_three_arm)) put its reported standard error over the bound at 1.0003 in sample, 1.082 at five folds and 1.055 at ten folds at n = 2,000, and 1.0085 at five folds at n = 8,000, so the excess is the finite-sample cost of cross-fitting saturated cell means. The owner closes when a cell is re-declared with its size and replication count chosen before its run by a stated sizing rule, such as a pass probability of at least 0.90 under exact calibration, and that cell passes. A larger run that happens to pass does not close it. Until then each cell stays published red under `reporting` |
| `F28` | [X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) parts (d), (f) and (g), for the finite-sample limits of clustered intervals: the fold-evaluated covariate pair of `clustered-unequal-cvtmle`, and the fold-evaluated bias cells and cross-fitted `DRTMLE` IID controls of `clustered-few-cluster-tmle` | each group closes on its own registered study that passes. The fold-evaluated pair of `clustered-unequal-cvtmle` at 40 clusters in 10 folds: a fold-evaluated degrees-of-freedom rule $t(\min(J-2, J-V))$ at 40 clusters or more. The three `tmle_cv_evaluation__unequal_informative` cells: a cluster-size-weighted fold average or another fold-evaluated point whose bias stays inside the margin. The stacked report is the remedy today. The four `drtmle_crossfit` `iid_t_control` cells: a calibrated cross-fitted `DRTMLE` cluster variance, or an IID control that compares the two SEs and not the IID SE with the empirical SD ([X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes)) |
| `X20-bootstrap` | the `interval_calibration` cells `boot_ey_crossfit__correctly_specified`, `boot_ate_crossfit__correctly_specified` and `boot_ate_clustered__correctly_specified` of `full-refit-bootstrap-and-derived-contrasts` | the declared rule 3 keeps the `end_of_study/cross_fit` and `end_of_study/cluster` kinds out of `LICENSED_BOOTSTRAP_DESIGNS`, so their bootstrap prints as a diagnostic. A diagnostic reading, not a registered result: on this cell the cross-fitted estimator is heavy-tailed. Its 4,000 committed estimates have excess kurtosis 6.7, and about 5% of them lie at the probability boundary, because `g_bounds=(1e-8, 1)` with saturated cell means leaves sparse training cells unbounded. A bootstrap resample holds about 63% unique units, so more replicates reach the boundary and the replicate spread exceeds the sampling spread (SE ratios 1.23 and 1.27). A probe that kept each unit's copies in one fold raised the SE ratio to about 1.65, so the split of copies across folds is not the cause. The cluster contrast covers 0.924 with a calibrated SE ratio (0.977), a finite-sample shortfall of the percentile interval at 60 clusters. The owner closes when a re-declared cell for the kind, with its law, size, nuisance bounds or resampling scheme declared before its run, passes a registered cell |
| `X41-negative-control` | the three `misspecification_control/*_x_contrast__main_terms` cells of `sequential-baselines` (Hajek, Horvitz-Thompson and ICE) | none. The study declares the cells red before the run (`EXPECTED_RED` in `tests/studies/canonical_sequential_baselines.py`). A law whose nuisances are all wrong must fail the calibration rule, and `tests/unit/test_sequential_baselines_design.py` computes the power of that failure. The cells read a standardized bias of 1.7 to 2.0 and a coverage of 0.48 to 0.61, with an SE ratio near 1. A pass would mean the rule cannot detect a misspecified fit |

## Detailed implementation contracts

The sections below give one contract for each open item. Their physical order does not override
the main grid.

### RM42. Experimental status and support rows

The [Eligibility](#eligibility) rule classifies each analysis as supported, experimental, or
unavailable. The package has no `experimental` status today. Each refusal carries a reason code,
and [How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal)
defines the five codes. This row implements the status and the support rows.

| part | what it delivers |
| --- | --- |
| (a) the experimental status | an explicit opt-in for a configuration whose target is justified but whose algorithm or inference extension has unresolved evidence. It is a new record in `cleverly._inference_status` beside the `*_plugin` statuses, not a second status field. Each estimate and interval carries the label, and an unsupported interval is withheld |
| (c) configuration support rows | a support row for each combination of target, estimator, outcome, weighting, splitting, and inference, whose status matches the output |
| (d) persistence | the experimental status, its configuration key, the support row, and the inference assumptions survive `result.save()` and `cleverly.load()` |

Each inherited refusal keeps its intervention-specific support checks, the estimator-specific
requirements for policy followers, the invalid-event checks, and the identification declarations.
A test can detect an observed support failure and check a declared support restriction. It cannot
establish population positivity or the absence of unmeasured confounding. A new refusal does not
call a stochastic or continuous policy an inappropriate question.

Acceptance:

- a test that an experimental fit carries its label through the summary, the tables, and a
  save-and-load round trip, and withholds an unsupported interval;
- a test that each support row matches the status that the fit reports.

### X39. Longitudinal evidence over three to five decisions

Every registered end-of-study longitudinal study uses two treatment nodes. The ordinary and
cross-fitted categorical studies use complete outcomes, and the known-policy study uses one
two-node, three-level law. This row adds laws with three to five decisions, with
censoring and realistic support loss, for `LTMLE` and `SDR`.

The studies reproduce the existing binary and categorical targets, exercise a nonzero targeting
update, and include leakage and covariance controls for the paired policy contrasts. Each page
states the conditions for consistency apart from the conditions for valid intervals.

Use R `ltmle` for the overlapping ordinary regimen and MSM cases, and `lmtp` for the `SDR` cases.
The cross-fit studies pin `lmtp` at `f04a2b4`, which fits the fluctuation on training rows.
Upstream `9996b04` fits it on validation rows. Neither runs the pooled update, so label the
cross-fitted `LTMLE` comparison as a different estimator of a shared target. Audit `9996b04` as a
second comparator option.

Acceptance: a registered study for each estimator, with the controls above and a pre-queue design
review.

### X40. Continuous-policy validation and support checks

`LTMLE` and `SDR` fit modified treatment policies at a continuous node through `Shift`, `Scale`,
`Piecewise`, and `ModifiedPolicy`. `check_continuous_policy` already refuses a map without a
density ratio before any learner. The support report is a diagnostic, and the classifier route
reports no unsupported mass. This row adds a declared-support check and the evidence that a
pricing analysis needs. Díaz et al. (2023), *Journal of the American Statistical Association*, give the
identification, density-ratio estimation, TMLE, and SDR. The 2025 erratum of Williams and Díaz in
*Observational Studies* corrects the identification assumptions.

| part | what it delivers |
| --- | --- |
| (a) transformed-law support | a declared conditional support at each node, and a pre-fit refusal that names the node and the count of rows whose assigned dose falls outside it. The post-fit support report keeps the estimated unsupported mass |
| (b) map structure | `check_continuous_policy` already refuses a map that sends an interval to one dose, such as `max(floor, 0.95 * price)`. This part adds an explicit mixed-law contract for such a map, or records that it stays refused |
| (c) classifier ratio | the classifier route already splits by unit and balances the two copies (`cleverly.learners.density_ratio`). This part adds a mutation control that splits the copies across folds, and a witness with unequal copy weights |
| (d) assumptions | the corrected identification assumptions in the causal metadata. The MTP effect and its stochastic-intervention representation are separate interpretations, each with its own stated assumptions |
| (e) data-tuned maps | an estimated support boundary or a data-tuned modification, built on training rows only. The contract states the target: the average of the fold-specific parameters, as the learned-rule value does, or the population parameter with the derivative term of the estimated map |
| (f) outcomes | bounded binary and continuous outcomes first. A revenue outcome declares its support, and the package does not winsorize it silently |

A policy that reads the counterfactual natural treatment history needs an audit first. A fixed
continuous path, such as prices of exactly $10, $12, and $11, is a different target. It does not
inherit regular root-n inference. Offer an observed categorical grid, a supported stochastic price
distribution, or a separately labelled model-based analysis. Never replace the requested path with
a shift silently.

Acceptance:

- means and contrasts under analytic continuous laws, with nonbinding and binding policy
  boundaries;
- identity policies, non-unit Jacobians, several inverse branches, ratio normalization, and
  wrong-ratio controls;
- learned-ratio studies beside oracle-ratio parity, with price-feedback confounding and several
  follow-up lengths.

### X42. Applied tutorials for offers and pricing

Publish two tutorials that run through the public API.

| tutorial | required content |
| --- | --- |
| sequential customer offers | categorical and dynamic rules, randomized offers, treatment-confounder feedback, censoring, and paired policy comparisons |
| sequential pricing | continuous modified treatment policies, support boundaries, price-induced engagement changes, and contrasts against a declared reference policy |

Each tutorial states time zero, eligibility, treatment versions, available history, confounders,
outcome horizon, and interference assumptions. Use synthetic laws with known truth, and a held-out
applied pilot where data access permits. Explain when TMLE, `SDR`, g-computation, or IPW is
appropriate, and why their answers can differ. Keep the stagewise support and truncation
diagnostics, and label a truncation curve as a descriptive estimator change.

Acceptance: both notebooks execute in the documentation build.

### X43. Categorical incremental policies

`IPSI` fits Kennedy's odds-multiplier target at binary nodes, and its odds tilt names two arms. A
categorical incremental policy is a candidate natural extension, starting from the binary case that
Kennedy (2019) covers directly. A propensity-dependent policy changes with the observed treatment
law, so its influence function carries a derivative term that a known stochastic policy does not
have. Do not substitute the risk-ratio policy of `lmtp`.

Use the pinned `npcausal` runner where it covers the odds-based target. `imtp` 0.1.0 at `d4b5204`
omits the derivative term from its influence curve
([comparator survey](development/method-benchmarking.md#choose-the-comparator-before-you-write-the-runner)),
so it can witness a point estimate only. Re-audit a later `imtp` release, including its
longitudinal support, before using it as a comparator.

Acceptance: a written derivation of the changed terms under the [Eligibility](#eligibility)
conditions, identity and treatment-score witnesses, a dropout control, and a registered study.

### X44. Structural dynamic treatment effects

EconML's `DynamicDML` estimates sequential treatment effects under structural assumptions that
differ from general policy-value estimation. Lewis and Syrgkanis (2021), *NeurIPS*, give the
targets and the assumptions. Assess it as a separate optional method or integration. Compare a
shared target only where the structural model makes the targets equal, and do not treat a
structural coefficient as an unrestricted policy value.

Acceptance: a written map of the targets and the assumptions, and the decision whether to ship.

### X45. Parametric forward-simulation g-formula

The NICE Monte Carlo g-formula models and simulates covariate trajectories. It differs from
backward sequential regression. McGrath et al. (2020), *Patterns*, describe `gfoRmula`, and
`pygformula` implements it in Python.

Keep it as an optional integration. Compare a shared target
under correctly specified laws, and account for the simulation error. A model-based fixed
continuous path states its model reliance and its extrapolation assumptions.

Acceptance: a registered study on a shared target, with the Monte Carlo error declared.

### X46. Repeated cross-fitting for longitudinal fits

Point-treatment fits repeat the fold draw and report the median over draws, with the
split-adjusted median variance of Chernozhukov et al. (2018), Definition 3.3 and equation (3.14).
The [repeated cross-fitting study](technical-reference/method-evidence/repeated-cross-fitting.md)
validates that report. Longitudinal fits take `n_folds` only. This row extends trajectory-level
split provenance to repeated fold draws for `LTMLE` and `SDR`, and applies the same fixed-repeat
rule.

The median does not inherit an ordinary influence function. The contract states the fixed-repeat
rule that carries the variance, and it keeps the joint covariance of paired policy contrasts
within each draw.

Acceptance: a witness that each draw reproduces a single-draw fit, a mutation control that reuses
one draw, and a registered study of the median and its variance.

### X16. C-TMLE candidate sequences

The shipped `CTMLE` builds candidates from covariate sets: the greedy, ordered and discrete paths.
The pinned `ctmle` 0.1.2 builds two other kinds of sequence.

| part | what to add | published source | comparator |
| --- | --- | --- | --- |
| (a) a lasso path | candidates indexed by the penalty of an L1 propensity fit, with the C-TMLE selection over the penalty | Ju, Wyss, Franklin, Schneeweiss, Häggström and van der Laan (2019), *SMMR* 28(4) | `ctmleGlmnet` |
| (b) declared propensity candidates | an ordered sequence of propensity estimates that a user supplies, with the same selection | Ju, Gruber et al. (2019), *SMMR* 28(2); van der Laan and Gruber (2010) for the general template | `ctmleGeneral` and `build_gn_seq` |

Both parts report point estimates and path diagnostics. Both take the `working_mechanism_plugin`
status, because [F18](#f18-selector-path-c-tmle-inference) holds selector-path inference. The
contract for part (b) states whether a supplied candidate may be fitted on the analysis sample,
and which folds its predictions must respect.

Acceptance:

- an exact-law witness that each path reaches the full model at its end;
- a mutation control on the path order;
- a registered study against the pinned `ctmle` that reuses the selector study's laws and reports
  the point and the plug-in diagnostic.


### X22. Targeting and fitting options

One pull request delivers these options. Each comparator ships the option, and none needs new
theory. Each option keeps every default fit bit-identical, and each changes a fit when it binds.

| part | option | what it does | comparator |
| --- | --- | --- | --- |
| (a) best iterate | report the targeting iterate with the smallest score norm over the loop | a fit that stops at `max_iter` reports its best iterate, and its record says which one | `tmle3` `use_best` |
| (b) stopping rule | choose the score threshold: the shipped variance-scaled rule, or a threshold of $1/n$ | the two rules that `tmle3` names `scaled_var` and `sample_size` | `tmle3` `convergence_type` |
| (c) step bound | bound the fluctuation step in each iteration | a damped update for a fit whose first step overshoots | `tmle3` `constrain_step` and `delta_epsilon` |
| (d) quantile trimming | trim the cumulative density ratio at a declared quantile, beside the shipped bound `g_bounds` | the trimming rule of `lmtp`. The fit records the trimmed share, as it records truncation now. `SDR` reads the same per-node ratio | `lmtp` `.trim` |
| (e) Markov order | restrict each node's nuisance history to the last `k` nodes | an assumption about the data. The identification summary states it. `SDR` reads the same per-node ratio and regressions | `lmtp` `k` |
| (f) per-arm outcome fit | fit the outcome regression separately in each arm | a nuisance choice for `TMLE` and `DRTMLE` | `drtmle` `stratify` |
| (g) early stop on a C-TMLE path | stop the greedy search after a declared number of steps without a risk improvement | a cheaper path. The fit keeps its `working_mechanism_plugin` status | `ctmle` `patience` |
| (h) hazard and outcome bounds | bound each node's regression into a declared interval before the logistic fluctuation | a fit on a declared range of the hazard or the iterated mean | `survtmle` `bounds` |
| (i) pooled-over-time regressions | fit one hazard regression and one censoring regression over every node, with the node as a covariate, in place of one regression per node | a fit with fewer events per node. A pooled fit is a nuisance choice for `survival_model="hazard"` | `survtmle`, MOSS and `tmle3` `tmle_survival` |

At a continuous modified-treatment-policy node of `LTMLE`, the ratio that part (d) trims is the
node's density ratio, which the fit stores as `RegimenFit.node_ratio`.

Part (h) covers `survtmle` `method = "hazard"` as well as `method = "mean"`. On a fit with
`survival_model="hazard"`, it bounds each node's cause-specific hazard. Part (i) applies to a fit
with `survival_model="hazard"`, which today fits one hazard regression per node and cause and one censoring regression per
node.

Acceptance: for each part, a test that the default fit is bit-identical to the fit before the
option, and a witness on a law where the option binds. Parts (d) and (e) also name their
assumption in the identification summary, and a test pins that text.

### X21. Adaptive-propensity IPTW

`cleverly` has no inverse-probability-weighted estimator. `drtmle` 1.1.2 ships `adaptive_iptw()`,
which reports an IPTW estimate with a targeted propensity, in two forms: `iptw_tmle` and
`iptw_os`. Van der Laan (2014), *International Journal of Biostatistics* 10(1), derives it. With a
propensity fitted by a flexible learner and targeted, the IPTW estimator is asymptotically
linear, and its curve needs no outcome regression.

Read the derivation for the targeting of the propensity, the reduced regression it needs, and the
interval conditions. `drtmle` also fits it with missing outcomes and several treatment levels.
Each of those is a separate composition with its own contract.

Acceptance:

- exact-law witnesses with a nonzero targeting step;
- a mutation control that drops the propensity targeting;
- a registered study against the pinned `drtmle` that reuses the DR-TMLE laws.

### X26. Plug-in omitted-variable limits with a declared parametric mechanism

`ci_lower`, `ci_upper`, `robustness_value_ci` and `rva` refuse under `nu2_estimator="plugin"`
(`_PLUGIN_LIMITS_REFUSAL` in `src/cleverly/sensitivity/omitted_variable.py`). This item opens them
for one declared mechanism. The
[natural-extension verdicts](technical-reference/natural-extension-verdicts.md) record the review
as part (h).

| item | contract |
| --- | --- |
| base results | Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026), the bound and its Theorem 4 limits. Saul and Hudgens (2020), *Journal of Statistical Software* 92(2), for stacked estimating equations and the sandwich variance |
| step | a stacked estimating equation of (i) the mechanism score $s(\gamma) = X\{A - \mathrm{expit}(X\gamma)\}$, or the multinomial score, (ii) the second-moment equation $\alpha(O; \gamma)^2 - \nu^2 = 0$, and (iii) the curves of $\theta$ and $\sigma^2$ that ship. The curve of the plug-in $\nu^2$ is $\alpha^2 - \nu^2 + \partial_\gamma E[\alpha^2] \cdot \{-J_\gamma^{-1} s\}$. The chain rule and the delta method carry it into the bound |
| objection search | [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) records no curve for an arbitrary learner, and that case stays refused. DoubleML calls the plug-in "non-orthogonal", which describes the arbitrary-learner case and not the parametric stack |
| inherited conditions | a correctly specified mechanism model, because otherwise the plug-in estimates $\nu^2(\gamma^*) \neq \nu^2_0$; smoothness; an invertible Jacobian. `g_bounds` makes $\alpha$ differentiable almost everywhere only, so the contract either requires no active clip at the fit or states that the derivative is zero on clipped rows |
| learner rule | the bound reads the fit's own $\hat g$. So the stack is valid only when the fit's own treatment learner is the declared unpenalized logistic or multinomial model, or a declared known mechanism. A Super Learner whose library contains that model is refused |

A declared known mechanism is the degenerate case of the stack. The fit reads
$\hat\alpha = \alpha(g_0)$, so $E_n[\alpha(g_0)^2]$ is a sample mean of a known function with curve
$\alpha^2 - \nu^2$. The ATT and ATC share term follows by the delta method. The
[hard-stop review](technical-reference/hard-stop-review.md) moved this case from F26.

Acceptance:

- a mechanism-score interface for an unpenalized `LogisticRegression`, binary and multinomial;
- an exact-law Jacobian witness, and a mutation control that drops the $\gamma$ term and fails;
- the four limits open only under the learner rule, with a refusal test for a Super Learner that
  contains the model;
- a known-mechanism witness: on a finite law with a declared known mechanism, the reported curve
  of $\nu^2$ equals $\alpha(g_0)^2 - \nu^2$ row by row;
- a registered coverage study.

### X58. Per-arm outcome-adaptive C-TMLE intervals with a parametric outcome

This item gives the per-arm outcome-adaptive design a corrected interval in a restricted scope. It
takes the tractable part of [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference): a
regular parametric outcome model with a declared treatment-mechanism contract. F19 keeps the
flexible outcome learner, the shared design, local uniformity, and the compositions outside this
contract. F19 also keeps the committed red cells. Every configuration outside the support predicate
below keeps the `generated_design_plugin` status.

**What is known.** [The applied revert](technical-reference/collaborative-tmle.md#the-applied-revert)
and the [per-arm evidence page](technical-reference/method-evidence/outcome-adaptive-per-arm-c-tmle.md)
give the cells.

| finding | evidence | consequence |
| --- | --- | --- |
| a comparator exists | R `drtmle` 1.1.2 at commit `538a3a2`, with `adapt_g = TRUE`, called once per arm on a binary recode | the comparator does not block this item |
| `cleverly` agrees with the comparator | each estimate and standard error agrees to `1e-8` in `ctmle-oat-per-arm` | parity holds for the uncorrected curve |
| the reported standard error is too small | ratios from 0.76 to 0.92 over the primary estimands at $n$ = 1,500. The $n$ ladder reads 0.83, 0.86 and 0.86 at $n$ = 500, 2,000 and 8,000 | parity does not establish calibrated inference, and it cannot override a measured shortfall |
| the learned design causes the deficit | the paired cells with $\bar Q_n$ pinned to the truth read 0.997 and 1.009 | learning the predictions adds an omitted contribution |
| a correction has diagnostic support | `tests/diagnostics/x17_generated_design/run.py` adds $c^\top \mathrm{IF}_\beta$ and reads ratios of 1.03 and 0.98, at $n$ = 1,500 and 6,000 | a restricted correction is a concrete candidate |
| no production correction exists | the diagnostic reads the true $g_0$ and the outcome learner's coefficient influence function | each oracle quantity needs an estimable or declared replacement |

For arm $a$, $g_{0,a}(W) = P(A = a \mid W)$ is the full mechanism, and
$G_a(\bar Q_a(W)) = P(A = a \mid \bar Q_a(W))$ is the adaptive mechanism. The diagnosis identifies
this candidate curve for a parametric outcome model with coefficients $\beta$:

$$
D_{\mathrm{corrected},a} = D_{\mathrm{plugin},a} + c_a^\top \mathrm{IF}_\beta, \qquad
c_a = E\left[\left(1 - \frac{g_{0,a}(W)}{G_a(\bar Q_a(W))}\right)
\frac{\partial \bar Q_a(W)}{\partial \beta}\right].
$$

$\mathrm{IF}_\beta$ is the influence function of the fitted outcome coefficients. The formula
summarizes the diagnosis. It is not the inference contract. With a parametric adaptive mechanism,
the $\gamma$ and $\epsilon$ terms also enter. This item derives the complete first-order
expansion of the fitted estimator, with the outcome fit, the adaptive mechanism fit, the targeting
step and the final average.

**Readiness.** This item is a natural extension candidate with an oracle diagnostic. The
[hard-stop review](technical-reference/hard-stop-review.md) reads the in-sample parametric stack
as a natural extension. Its base is a stacked M-estimator (Saul and Hudgens 2020) with the
conditions of Appendix F of Benkeser, Cai and van der Laan, arXiv:1901.05056v1. The stack, not the
curve of Theorem 1, supplies the generated-design term. Theorem 1 needs condition (v), and
condition (v) fails here.

The steps are an indicator reduction for each arm, a stacked estimating
equation, the chain rule for influence functions, the delta method, and a fixed-dimension stack.
This item must confirm that every step is in the standard-step table of
[Eligibility](#eligibility). A step outside that table returns its claim to F19.

**First scope.** Each extension enters through its own acceptance cells.

| axis | first implementation |
| --- | --- |
| design | `strategy="oat"`, `oat_design="per_arm"` |
| data | independent rows, complete outcomes and treatment, no `strata=`, no weights |
| treatment | binary |
| outcome learner | a declared regular parametric model that exposes its score, its derivative and its coefficient influence function |
| adaptive mechanism | a declared smooth parametric fit. The contract states its assumptions and the behavior of `g_bounds` |
| target | the two arm means and the ATE, with their joint covariance |
| fitting | in sample |
| inference | a corrected pointwise covariance and a Wald interval, inside the declared model and its regularity conditions |

| part | extension | entry condition |
| --- | --- | --- |
| (a) | the first scope above | the acceptance list below |
| (b) | more than two arms, and `rr`, `or`, `par` and `paf` by the delta method | a map of the changed contributions, and their own witnesses and cells |
| (c) | cross-fitting with parametric learners, and `repeats` above 1 | an explicit map of the fold-specific coefficient contributions, and a written expansion. Without one, the claim stays in F19 |
| (d) | fixed known weights | the weight-tilted law of the shipped per-arm contract, with its own witness and cell |

The `ctmle-oat-per-arm` revert cells split by part. Part (a) takes the binary primary rows of
`ey1`, `ey0` and `ate`, `interval_calibration/binary_active__correctly_specified` and the binary
`generated_design` pair. Part (b) takes the other primary rows, the three-arm cells and the
simultaneous band. Part (c) takes `binary_repeats__correctly_specified`, and part (d) takes
`binary_weighted__correctly_specified`. A cell stays with F19 until its part passes.

**Routes to compare.** This item evaluates each route, then implements the smallest justified route
with reliable behavior.

| route | what it estimates | conditions to state |
| --- | --- | --- |
| a full mechanism | $c_a^\top \mathrm{IF}_\beta$, plus the $\gamma$ and $\epsilon$ terms of the parametric adaptive mechanism, with a declared known $g_0$ or with a correctly specified parametric mechanism whose score enters the stack. `CTMLE` refuses a declared `treatment_probabilities=` today (`KNOWN_CTMLE_REFUSAL`). This route lifts that refusal for the per-arm correction only, with $g_0$ an input to the curve and not to the design, and updates [scope and refusals](technical-reference/scope-and-refusals.md) | the full mechanism is known or consistently estimated |
| a joint estimating equation | the sandwich over $\beta$, the adaptive mechanism $\gamma$, the fluctuation $\epsilon$ and $\psi$, as [X52](#x52-parametric-c-tmle-intervals) describes | the outcome model is correct for coverage of $\psi_0$. The sample Jacobian may estimate $c_a$ from the observed treatment without a full mechanism. The derivation must confirm this |

A consistent full mechanism is an acceptable cost for valid restricted inference. Avoiding that
mechanism is a further objective, and F19 keeps it. A correct outcome model alone does not
establish calibrated uncertainty. This item claims no superefficiency and no efficiency gain over
ordinary TMLE without a separate argument.

**Implementation sequence.**

1. Derive the complete first-order expansion for the fitted estimator and the learner contract.
2. Name each oracle quantity, and specify an estimable or declared replacement.
3. Evaluate the full-mechanism route, with a known or consistently estimated $g_0$.
4. Evaluate the joint estimating equation when it accounts for every fitted stage.
5. Compare the routes, and choose one.
6. Compute the covariance from the corrected row-wise curves. Keep the cross-arm covariance for the
   ATE.
7. Implement the support predicate, and restrict the inferential status to the configurations that
   meet it.
8. Add parts (b) and (c) only after their changed contributions are mapped and tested.

**Release gate.** A support predicate replaces the global flag `OAT_PER_ARM_INFERENTIAL`. The
predicate reads the outcome learner, the mechanism learner, the design, the fitting, the target and
the inference request. It never reads a fitted array. One passed study does not open every per-arm
fit.

The corrected status, the covariance and the method declaration survive `result.save()`,
`cleverly.load()`, contrasts and summaries. [RM42](#rm42-experimental-status-and-support-rows)
part (c) supplies the support rows. `plugin_std_error`, `plugin_interval` and the bootstrap spread
stay diagnostics. Ordinary TMLE stays the inference alternative under its own assumptions.

**Acceptance.**

| criterion | evidence |
| --- | --- |
| a complete correction | a written expansion, or a standard joint-equation argument, that accounts for the outcome fit, the adaptive mechanism fit, the targeting step and the average |
| no oracle quantity | the released curve uses quantities that a user can estimate or declare known |
| an independent check | a tractable law verifies the changed contribution, its sign and the resulting covariance |
| sensitive witnesses | a mutation that removes or reverses the contribution fails the check. A fixed design removes the learned-design contribution where the derivation says so |
| the user pipeline | the studies fit the declared outcome and mechanism models. No study supplies an unknown true quantity |
| calibration | a registered repeated-sampling study measures bias, the standard-error ratio, coverage and the sample-size ladder |
| joint effects | the study validates the arm covariance and the ATE interval together |
| comparator scope | R parity stays evidence for the unchanged estimator. The corrected uncertainty is checked independently, because it can differ from R. The Python calibration study gates the merge. An R comparison is registered when it arrives, and it does not block the merge |
| numerical boundaries | probability truncation and an ill-conditioned fit have explicit behavior and evidence |
| honest output | an unsupported configuration keeps the diagnostic-only interval. A supported fit states its model and inference conditions |

Declare the corrected procedure and the study before the final validation draws. Keep the failed
results and thresholds of `ctmle-oat-per-arm`. Do not relabel an old failure as a pass, and do not
widen a margin after the outcome is known.

Run a fresh declared study, or a registered regeneration, for the changed procedure. Reuse the
evidence for unchanged components, and add checks for each changed term. Cross-fitting alone is not
the correction. A full-refit bootstrap needs its own validity argument for this estimator. Neither
removes the generated-design contribution, and neither resolves the irregularity.

When the acceptance passes, close the supported part of F19 only. F19 keeps its other claims and
its red cells.

### X68. Worked examples for the shipped methods

Review every worked example in `docs/examples/` against the methods that ship when every row
of milestones B0 to B4 and BP has merged. Start this item only then, so that it covers each method
that exists then.
A method here means one implementation family of the
[implementation matrix](technical-reference/index.md#implementation-matrix).

| task | required content |
| --- | --- |
| inventory | a table that maps each implementation family to the example page or section that shows it, or records that no example shows it |
| update | each notebook runs on the current API. Replace removed and renamed calls, refresh the stored output, and correct each claim that the current code no longer supports |
| add | an example for each family that the inventory shows without one. The [X42](#x42-applied-tutorials-for-offers-and-pricing) tutorials count as coverage |
| index | each new page has a row in the [examples index](examples/index.md), with its evaluation question, its method entry, and the failure mode it shows. The page and its method entry link to each other, as the examples contract in `docs/README.md` requires |

Use the shared program and study design of the [examples index](examples/index.md) when the method
can answer a question in that program. Otherwise, use a synthetic law with a known truth. Use
compact data and explicit learners. An example shows normal use of a method. It does not assert a
statistical guarantee from one sample.

Acceptance: the inventory shows an example for each implementation family, and every notebook
executes in the documentation build.

### X69. Documentation review for new and changed methods

Update the reader-facing documentation for each method that the rows of milestones B0 to B4 and BP
add or change. Then review all of it for quality. Start this item after X68, so that the review includes the new
example pages. The scope is `README.md` and every reader-facing source in `docs/`. This item is
the broad review that step 1 of the documentation procedure otherwise excludes.

| check | required result |
| --- | --- |
| coverage | each added or changed method has a user-guide path, a technical-reference entry, an evidence row, and its refusals. The merged pull requests of those rows give the list of methods |
| correctness | each technical claim agrees with the code, a test, a registered artifact, or a cited source. Correct or remove each claim that does not |
| clarity | each page follows the [documentation procedure](development/contributing.md#write-documentation) |
| succinctness | each sentence adds verifiable information. Remove repetition, generic introductions, restated headings, closing summaries, and adjectives with no evidence |

The succinctness check targets the patterns of unreviewed generated text. Remove them, and do not
replace them with other words of the same kind. Keep a table where the content is parallel.

Acceptance: a reviewer other than the author reads each changed page against the four checks and
records the findings in the pull request. `python -m tests.prose` has no unjudged finding, and the
documentation build passes with warnings as errors.

### X2. Replicate-weight designs

Rust and Rao (1996) govern replication variance for complex surveys. Add BRR, jackknife, or
another replicate design only after a source audit matches its construction to this package's
weighted-law estimands and inference conventions.

### X35. Efficient known-score ATT and ATC

On data that declares its treatment mechanism, the shipped ATT divides by $g_0$ and reports the
treated mean of $\bar Q^*_1-\bar Q^*_0$. That curve is not the efficient one when the propensity
score is known. Hahn (1998) gives a smaller ATT bound for a known score than for an unknown one.

By the remainder algebra of the
[known treatment mechanism](technical-reference/point-treatment-tmle.md#known-treatment-mechanism)
section, the g-weighted plug-in $P_n\{g_0(\bar Q^*_1-\bar Q^*_0)\}/P_n g_0$ has the smaller curve.
The ATC is the mirror image. No pinned comparator implements it. R `tmle` and the `tmle3`
`Param_ATT` both fluctuate $g$, so their ATT has the unknown-score curve. The item therefore follows
the beta.

| item | contract |
| --- | --- |
| base result | Hahn (1998), *Econometrica* 66(2):315–331, Theorems 1 and 2. [References](references.md) records that only the abstract was read. Read Theorem 2 first-hand before the row starts |
| step | the g-weighted plug-in at $g_n=g_0$, with the remainder algebra of the shipped known-mechanism ATT |
| estimator choice | an option on the known-mechanism ATT and ATC. The shipped treated-mean form stays the default |
| comparator | none. The study cells compare with the truth only |

Acceptance:

- an exact-law test of the curve of the g-weighted plug-in;
- a variance witness: on a law where the score carries information, the reported standard error is
  below the standard error of the shipped form;
- truth-only registered study cells.

### X5. Natural and interventional mediation effects

`cleverly` reports controlled direct effects only. Natural and interventional direct and indirect
effects are separate estimands, and each carries its own identification assumptions. Díaz, Hejazi,
Rudolph and van der Laan (2021) derive the interventional effects and their efficient influence
function under an intermediate confounder. R `medoutcon` implements a cross-fitted one-step
estimator and a cross-validated TMLE for them, and it pins by commit. Read the identification, the
influence function, the targeting construction, and the interval conditions first-hand. Add each
accepted target to the oracle registry and the evidence gates in both directions.

Mediation under an incremental tilt is a further target of this row. Hejazi, Rudolph, van der Laan
and Díaz (2023), arXiv:2009.06203v4, Lemma 3 and Theorems 3 and 4, give its identification and
efficient estimators.

### X6. Continuous-time survival and competing risks

The shipped survival and competing-risk estimators use discrete time nodes. Rytgaard, Gerds and van
der Laan (2022) derive the continuous-time construction, which changes the intensity model, the
targeting step, and the remainder. The `concrete` package implements the one-step form of
Rytgaard, Eriksson and van der Laan (2023). Its last release is 1.0.5, and CRAN archived it in
2024, so pin it by commit before it becomes a comparator.

That comparator takes a binary baseline treatment under a static or dynamic intervention, which
bounds the paired cells a first study can claim. A discrete-time study is not evidence for a
continuous-time interval, so the existing longitudinal rows do not transfer. The discrete-time
whole-curve targeting has shipped as
[hazard-based survival targeting](technical-reference/longitudinal-tmle.md#hazard-based-survival-targeting),
with `survival_model="hazard"` and `targeting="one_step"`. On a discrete grid, a `concrete` intervention on the baseline treatment is a
[held point-treatment design](technical-reference/longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes) with a plan at node 1. A stochastic intervention is a
[known stochastic policy](technical-reference/longitudinal-tmle.md#known-stochastic-policies) at
that node.

### X70. Hazard targeting beyond a held baseline treatment

`LTMLE` with `survival_model="hazard"` supports one categorical baseline treatment held over every
node, with baseline covariates only. The
[hazard and isotonic survival refusals](technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)
list each design it refuses. Each refusal names `survival_model="iterated_mean"` with
`isotonic=True` as the working alternative.

| part | what to add | published source |
| --- | --- | --- |
| (a) a treatment at several nodes | a hazard model and a fluctuation of the treatment mechanism at each decision node | Stitelman, De Gruttola and van der Laan (2012), *The International Journal of Biostatistics* 8(1) |
| (b) covariates after baseline | a model and a fluctuation of each later covariate, beside the event and censoring hazards | Stitelman, De Gruttola and van der Laan (2012) |
| (c) a continuous baseline treatment or a modified treatment policy | the hazards evaluated at the shifted dose, with the treatment density ratio in the clever covariate | Díaz and van der Laan (2012), *Biometrics* 68(2), for the point-treatment shift |
| (d) a working model over the hazard curve | MSM coefficients over regimen and horizon cells from the targeted hazards | Rosenblum and van der Laan (2010), *The International Journal of Biostatistics* 6(2) |

Acceptance:

- exact-law witnesses for each new clever covariate, with a mutation control that drops it;
- a registered study for each part, on a law where the per-horizon and the hazard fits differ.

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

### X9. Omitted-variable bounds on the other linear functionals

The package refuses the omitted-variable bound on
five kinds of fit whose parameter is a linear functional of an outcome regression. Each refusal is
correct, and each message says that the bound is well posed and not implemented. Theorem 2 of
Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026) gives the bias of such a functional, and
Equation (14) in their Section 4 gives the bounds.
The [omitted-variable bound study](technical-reference/method-evidence/omitted-variable-bound-standard-error.md) checked those locators against the
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
weight that is zero off its support, as `tests/unit/test_omitted_variable_refusals.py` has for the
intermediate representer. DR-TMLE and
C-TMLE fits stay in [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions), and longitudinal fits
stay in [F16](#f16-longitudinal-sensitivity-bound-estimation). The `ipsi` axis stays refused,
because its functional depends on the treatment mechanism.

### X10. Continuous-dose MSM with a second mechanism

`refuse_continuous_msm_mechanisms` in `src/cleverly/msm.py` refuses a continuous-dose MSM fit with a
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

The [learned-rule contract](technical-reference/point-treatment-tmle.md#learned-rules) estimates one target by one
construction. Each part below is a published follow-up outside that contract. For parts (a) to
(e) and (h), the refusal of the request cites the part. No request reaches parts (f) and
(g). Each part needs its own contract, witnesses and registered study
before it ships, as the learned-rule value did.

| part | the request that the contract refuses | published source | readiness |
| --- | --- | --- | --- |
| (a) the value of one rule fitted on all rows | `cross_fit=False`, or one fold | van der Laan and Luedtke (2015), Section 6 and Theorem 5, under an empirical process condition | published support; pending source read |
| (b) learned longitudinal regimens | `DynamicRegimen(..., rule_kind="estimated")` | van der Laan and Luedtke (2015), Sections 2 and 7, at two time points | published support; pending source read |
| (c) a contrast of the learned-rule value with a known regime or an arm | `learned_rule=` beside `interventions=` or an arm estimand | Theorem 6 of the same paper for each value. The contrast follows by linearity of the fold-local curves, an [Eligibility](#eligibility) step that the contract must record | source audit |
| (d) categorical treatments | a treatment with more than two arms | Nordland and Holst (2026), Section 3.4, for discrete actions | source audit |
| (e) the stacked doubly robust score evaluation | `cv_evaluation=False` | Nordland and Holst (2026), Algorithm 4, which pools the scores and centres the variance at the pooled estimate | published support; pending source read |
| (f) the value of the optimal rule | no request reaches it | van der Laan and Luedtke (2015), Section 7.2, last paragraph; Luedtke and van der Laan (2016), *Annals of Statistics* | published support; pending source read |
| (g) blip and weighted-classification rule learners | no request reaches it, because the contract has one rule learner | Luedtke and van der Laan (2016), *International Journal of Biostatistics*, Sections 4.1 to 4.3, read first-hand in the author manuscript | published support |
| (h) fold-specific targeting | `targeting_scheme="fold"` | Montoya, van der Laan, Skeem and Petersen (2023), *International Journal of Biostatistics* 19(1):239–259, Section 3.2, Step 2(c), and Section 4.2, read first-hand in the publisher's version | published support |

A rule learned on separate training data and frozen before evaluation is a known rule.
`Rule(..., rule_kind="known")` and `DynamicRegimen(..., rule_kind="known")` evaluate it, and the
interval targets that rule conditional on the training data. A report that selects the best of
several estimated rules changes the meaning of the reported value
([F27](#f27-learned-policy-value-outside-the-published-conditions)).

The [hard-stop review](technical-reference/hard-stop-review.md) adds the parts below. Each one
holds under condition C3 in its fixed-limit form (van der Laan and Luedtke 2015, Theorem 6), and
its refusal cites the part. Part (c) also covers a learned rule beside another parameter axis.
Under C3 the fixed-dimension stack carries that case, and it shares the source audit of part (c).

| part | the request that the contract refuses | step | readiness |
| --- | --- | --- | --- |
| (i) the mean over a fixed number of splits | `repeats` above 1 | linearity over a common limit. Under the fold-average target that the package reports, the median of split estimates estimates no defined parameter, so it stays refused | natural extension |
| (j) a learned rule with a controlled intermediate | `intermediate=` | a two-time-point rule pair: the learned rule at $A$ and the static $Z = z$ | source audit, with part (b) |
| (k) observation weights | `weights=` | the chain rule with a known weight. The contract states whether the rule learner reads the weights | natural extension |
| (l) clusters | `id=` | a cluster as the unit, with the few-cluster rules of [F28](#f28-finite-sample-limits-of-clustered-intervals) | natural extension |
| (m) baseline strata | `strata=` | a finite partition, after [X28](#x28-fold-evaluated-cv-tmle-with-baseline-strata) | natural extension |
| (n) the omitted-variable bound | the bound on a learned-rule fit | the bound of the frozen fold rules after [X9](#x9-omitted-variable-bounds-on-the-other-linear-functionals). Its limits need $\sigma^2$ and $\nu^2$ asymptotically linear under C3, or the row ships the point bound only | natural extension |
| (o) simulated confounding and `refute` | either on a learned-rule fit | a new fold-evaluated replay path with the fold rules frozen. It reads as confounding of the value of these fixed rules | natural extension |

Part (h) is not the training-fold update that the [Eligibility](#eligibility) section names as new
theory. Montoya and co-authors fit the update on the validation rows, and the package already ships
that update for arm targets.

### X28. Fold-evaluated CV-TMLE with baseline strata

`cv_evaluation=True` refuses `strata=` before any learner (`src/cleverly/estimators/tmle.py`).
The fold-evaluated estimate needs the stratum shares $P_n(S=s)$ and a stratum-indexed fold
average inside each validation fold. The pooled stratified update is unchanged. The item follows
the beta because no pinned comparator reports a fold-averaged stratified estimate. The `tmle3`
CV-TMLE evaluates a pooled mean over the validation predictions, and the shipped
`cross_fit=True` pooled fit is that estimator.

| item | contract |
| --- | --- |
| base results | the shipped fold-evaluated estimate ([CV-TMLE](technical-reference/cv-tmle.md)) and the shipped stratified fluctuation ([point-treatment TMLE](technical-reference/point-treatment-tmle.md#weights-strata-and-clusters)) |
| step | a finite partition. Each stratum estimate is the fold average of the stratum plug-ins of the validation folds. Each stratum curve is $I(S=s) D_s / P_n(S=s)$ |
| objection search | the [natural-extension verdicts](technical-reference/natural-extension-verdicts.md), part (i), found no objection to a finite partition in Kennedy (2019), Benkeser et al. (2017), or the MSM projection contract. The fold-evaluated base estimate is Zheng and van der Laan (2011), Sections 2 and 2.1, which [F21](#f21-other-missing-outcome-cv-tmle-variants) cites. No source read records the stratum restriction of that estimate as open |
| inherited conditions | the conditions of the fold-evaluated estimate inside each stratum; positivity inside each stratum; a fixed number of strata |

Acceptance:

- a fold-evaluated stratified fit whose stratum estimates equal the fold average of the per-fold
  stratum plug-ins;
- an exact-law test of each stratum curve;
- a registered calibration cell.

### X31. Continuous vector components at a node

`LTMLE` takes a vector of categorical columns as one node, and the discrete formula gives its
ratio. `LongitudinalData.from_frame` refuses a vector node with a continuous component, before any
learner. `CausalData` takes one treatment column, so a point treatment has no vector node at all.
The pinned `lmtp` 1.5.4 accepts a list of columns as one node of `trt`, so this is a parity gap.

| part | what to add | published source | comparator |
| --- | --- | --- | --- |
| (a) a point vector treatment | `CausalData(treatment=["A1", "A2"])` with a joint policy, on the routes the longitudinal vector node uses | Hoffman et al. (2024), *Epidemiology* 35(5). This locator is pending its source read | `lmtp` 1.5.4 with one node of two columns |
| (b) a continuous component | the ratio of a joint policy when one column is a dose | the same source, and Section 5.4 of Díaz, Williams, Hoffman and Schenck (2023) for the classifier route | the same |

The classifier route (`ratio="classifier"`) needs only the joint policy and the stacked fit, so it
is the first route to build. A density route needs a joint conditional density, or declared
per-component pieces whose Jacobian is the product of the component Jacobians.

Acceptance:

- exact-law Gateaux witnesses for each part on a coarse discrete law;
- a mutation control that drops one component's ratio;
- a registered study against the pinned `lmtp` 1.5.4, with the ratio supplied to both sides.

### X32b. Repeated outcomes for SDR and IPSI

`LTMLE` fits a repeated outcome, declared with `outcome=Repeated([...])`. Each horizon is the
end-of-study fit of the panel truncated after it
([repeated outcomes](technical-reference/longitudinal-tmle.md#repeated-outcomes)). `LTMLE.fit`
refuses a repeated outcome for any other estimator kind, before any learner. So `SDR` and `IPSI`
refuse it now.

| part | what to add | evidence |
| --- | --- | --- |
| (a) `SDR` | the per-horizon reduction of the sequentially doubly robust fit | the reduction test of `tests/unit/test_repeated_outcome.py` for `SDR`, and exact-law Gateaux witnesses |
| (b) `IPSI` | the curve $t \mapsto E[Y_t^{Q}]$ under a tilt. Kim, Kennedy and Naimi (2021), *Journal of Causal Inference* 9(1):302–344, Theorems 3.1, 4.1 and 5.1, cover tilts, dropout, and the joint process over $\delta$ and $t$ | the same, for `IPSI` |

Acceptance:

- `tests/unit/test_repeated_outcome_subclasses.py` passes with each estimator fitting rather than
  refusing;
- exact-law witnesses with a non-absorbing outcome that returns to zero, and a mutation control
  that seeds every horizon at $Y_T$;
- a truth-calibrated registered study per estimator, or a row in `longitudinal-repeated-outcome`
  for each. Kim's research code is unpinned, so `IPSI` has no paired comparator.

### X47. Simulated-confounding laws for point-treatment fits

`simulated_confounding` refuses seven compositions on a point-treatment fit. The [hard-stop review](technical-reference/hard-stop-review.md) found
a declarable law for each part below. Each law is a declaration with a stated reading, not an
estimator claim. The base law is the pinned DoWhy refuter, `_include_confounders_effect` at commit
`2116d5c`.

| part | request | law and reading | witness |
| --- | --- | --- | --- |
| (a) | a multi-arm treatment | the binary flip applied to the two arms of a declared pairwise contrast $a$ against $b$: $g'_a = (1-s)g_a + s g_b$. The swap is label-invariant by construction. A $K \times K$ transition is not adopted, because no source gives its reading | swapping the arm codes gives identical cells |
| (b) | a clustered fit | the caller declares `latent_level` as `"row"`, `"cluster"`, or mixed with a declared $\rho$. `"row"` is refused when the treatment is constant within clusters. `treatSens` 3.0 (`R/genU_contY.R`) is a design precedent for the level, not a base | a cluster-level latent is constant within each cluster |
| (c) | a categorical benchmark covariate | a block calibration over every encoded column of one logical covariate, fitted unpenalized. The shipped helper's default L2 penalty breaks reference invariance, so the contract changes the fit | relabelling the reference level gives the same strength |
| (d) | estimated weights | hold the supplied weights fixed. Each cell targets the fit's weighted population, $E[w\,h(W, A', Y')]$. Regenerating the weights is not adopted, because it defines no population that the fit targets | the replay at zero strength reproduces the weighted estimate. At a declared nonzero strength on a finite law, the replay cell equals $E[w\,h(W, A', Y')]$ computed exactly. A mutation that drops $w$ from the perturbed refit fails that check |
| (e) | a missing outcome or a missing treatment | outcome-axis cells, and a declared observation mechanism that reads the covariates only, so $\Delta \perp (A', Y') \mid W$ | a mutation that lets the mechanism read $A$ fails the declaration check |
| (f) | an intermediate | outcome-axis cells, and an intermediate-axis law that flips or shifts $Z$ in the latent tail and shifts $Y$ with $A$ held: $g_{Z'} = (1-s)g_Z + s(1-g_Z)$ | $A$ is unchanged. A mutation that flips $A$ fails |
| (g) | a declared `q_bounds` | an outcome shift on the logit scale of $[a, b]$. Values stay inside, and the law is the identity at zero strength. The report gives the share of rows at a bound, which never move. A normal-score option has lower confidence. That option uses $\hat F = \mathrm{rank}/(n + 1)$, so $\Phi^{-1}(\hat F)$ stays finite | each perturbed outcome stays in $[a, b]$ |

Readiness: natural extension, and no new theory for part (d). No part makes an interval claim.
[F12](#f12-missing-outcome-simulated-confounding-replay) and
[F15](#f15-controlled-direct-effect-simulated-confounding-replay) keep the parts that need
identification.

Acceptance: each part's witness; a pre-fit refusal test for each request that stays refused; the
existing refusal constants changed in the same commit.

### X48. Longitudinal simulated-confounding replay with a time-fixed latent

One time-fixed latent is shared by every treatment node, and the DoWhy law applies at each node.
The contract covers uncensored terminal-outcome fits whose panel has no time-varying covariate
after the first treatment node. That narrowing removes covariates that were generated under the
original treatment. Censoring, survival outcomes, post-treatment covariates and time-varying
latents stay in [F13](#f13-longitudinal-simulated-confounding-replay).

A later treatment node has the same structure as a post-treatment covariate when its mechanism
reads an earlier treatment. The data generated $A_2$ under the original $A_1$, so given the
misclassified $A'_1$ the node $A_2$ depends on the latent. The induced association of $A_2$ with
the latent then includes the path through the earlier misclassified treatment. The contract must
do one of two things before it ships:

| option | contract |
| --- | --- |
| define the induced association | the reading of each strength names the induced association at a later node, including the path through $A'_1$, and the replay reports it |
| restrict the mechanisms | the replay covers only fits whose declared treatment mechanism at each later node does not read an earlier treatment, and it refuses the others before any draw |

Until the contract chooses one, a later node whose mechanism reads an earlier treatment stays in
[F13](#f13-longitudinal-simulated-confounding-replay).

Acceptance: the replay at zero strength reproduces the estimate; a nonzero witness at a two-node
law; a pre-fit refusal for each panel outside the contract; for the restricted option, a pre-fit
refusal of a mechanism that reads an earlier treatment.

### X49. E-value for a controlled direct effect

The target is the package's controlled-direct-effect risk ratio between arms $a$ and $a'$ at a
fixed $Z = z$. The [hard-stop review](technical-reference/hard-stop-review.md) records the verdict.

| item | contract |
| --- | --- |
| step 1 | the bias of the ratio splits into one factor per arm. The split is an identity |
| step 2 | each factor is bounded by the bounding factor of Ding and VanderWeele (2016, *Epidemiology*), eAppendix Proposition A.2 (arXiv:1507.03984v1). The strengths are suprema over $w$. Arm $a$ uses the ratio $\sup_u P(u \mid a, z, W) / P(u \mid W)$. Arm $a'$ uses the reciprocal direction $\sup_u P(u \mid W) / P(u \mid a', z, W)$. Each arm has its own $RR_{UY \mid \cdot}$ |
| precedent | Smith, Mathur and VanderWeele (2021), arXiv:2005.02908v1, Result 1, whose $BF_s$ is a per-arm product |
| inversion | with equal strengths, the ordinary E-value of $\sqrt{RR}$. The limit uses the confidence limit nearest the null, as the shipped `rr` E-value does. The odds-ratio and Gaussian conversions keep their approximation label |
| reading | the cell ratio $\sup_u P(u \mid A = a, Z = z, W) / P(u \mid W)$ mixes the association of $U$ with $A$ and with $Z$, and the report names it so |
| inherited conditions | $Y_{az} \perp A \mid W, U$ and $Y_{az} \perp Z \mid A, W, U$, with $U$ a baseline latent that $A$ does not affect; positivity given $(W, U)$. The strengths are suprema over $w$ |
| context | VanderWeele (2010), *Epidemiology* 21(4):540-551, Theorem 1, assumes that $U$ confounds only $Z$ and $Y$. It is context, not the base |

Acceptance: a nonzero finite law where a single bounding factor fails and the product holds; a
wrong-arm control that fails.

### X50. Further DR-TMLE compositions

The base is Benkeser, Carone, van der Laan and Gilbert (2017), Theorem 1 (PMC5793673), in sample.

| part | request | step |
| --- | --- | --- |
| (a) | `interventions=` | an indicator reduction to $1\{A = d(W)\}$. A known stochastic policy is a known tilt in $W$ for each arm |
| (b) | `msm=` over discrete arms | a fixed full-rank projection of the arm or regime means |
| (c) | `NaturalCourseMean`, PAR and PAF with missing outcomes | Theorem 1 with $\Delta$ as the treatment and $(A, W)$ as the covariates, then a stack, linearity for PAR and the delta method for PAF |
| (d) | `evaluation=` on the composite | a diagnostic only. No new theory |
| (e) | the omitted-variable bound | the shipped plug-in bound recentred on the DR-TMLE point, under a consistent mechanism. No new theory. The owner chooses the scope |
| (f) | estimated weights with a guard | the conditional reading of the weights does not use efficiency, so it covers the fixed-weight DR-TMLE curve. It reopens `estimated_weight_plugin` for `DRTMLE` |

Part (c) has four steps.

| step | work |
| --- | --- |
| 1 | a realised-arm composite view in `estimators/composite.py`, with the indicator $\Delta$ and the covariates $(A, W)$, and reduced regressions over $\bar m(A, W)$ and $\pi(A, W)$ |
| 2 | the DR-TMLE natural-course mean `ey_obs` at each guard and reduction, with Gateaux and remainder witnesses for the corrected curve $D^*-D^*_Q-D^*_g$, and its post-fit readers |
| 3 | PAR and PAF through the shipped `TargetContext.natural_course` hook |
| 4 | a registered study with one-wrong and both-wrong cells, paired with `drtmle` 1.1.2 through the constant-treatment form (`A = rep(1, n)`, `W = cbind(A, W)`, `DeltaY = !is.na(Y)`). Its one-wrong cells declare their reading from the [case table](technical-reference/dr-tmle/theorem.md#the-conditioning-terms-are-linear-in-one-error) |

Acceptance: a nonzero witness for each changed term; a registered study for each interval claim.

### X51. C-TMLE point estimates for further targets

Points only. Every fit keeps `working_mechanism_plugin` or `generated_design_plugin`. The
observation factors stay full-covariate and off the selection path, as the shipped selector fits
them. A path that holds them needs its own derivation and is out of scope.

| part | request | step |
| --- | --- | --- |
| (a) | the natural course, PAR and PAF with missing outcomes, and cross-fitted arm-indexed missing outcomes, on a selector strategy or the shared design | an ordinary natural-course TMLE stacked beside the C-TMLE arm mean (van der Laan and Gruber 2009, WP 246, Section 2.4) |
| (b) | `interventions=` | an indicator reduction to $1\{A = d(W)\}$ (WP 246, Theorem 2) |
| (c) | `msm=` over discrete arms | a fixed full-rank projection of the arm means, with a declared selection criterion |
| (d) | a missing treatment on a selector strategy, in sample | the composite indicator reduction |
| (e) | `id=` | a cluster as the unit with whole-cluster folds. Each point inherits the fixed-limit assumption for the selected mechanism (WP 246, Section 4) |
| (f) | the omitted-variable bound | as X50 part (e). No new theory |

Acceptance: a nonzero witness for each changed term; a registered point study.

### X52. Parametric C-TMLE intervals

The base is a stacked estimating equation (Saul and Hudgens 2020) over every parameter of a fully
parametric fit: the outcome GLM $\beta$, the design or candidate mechanism $\gamma$, the
fluctuation $\epsilon$ and $\psi$. The sandwich supplies the generated-design term and every other
term.

| part | request |
| --- | --- |
| (a) | a selector fit with one declared partial candidate and parametric learners |
| (b) | the composite cells of the per-arm outcome-adaptive design (baseline strata, a missing outcome, a missing treatment), with a parametric outcome, in sample. It builds on [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome), which holds the plain per-arm cells |
| (c) | the shared design with a parametric outcome and a parametric multinomial design |
| (d) | the clustered versions of parts (a) to (c) and of the X58 cells, with the few-cluster rules of [F28](#f28-finite-sample-limits-of-clustered-intervals) |

| item | contract |
| --- | --- |
| learners | unpenalized parametric learners, read from the fit's own learners, as in X26 |
| bounds | `g_bounds` inactive at the limit |
| coverage, part (a) | coverage of $\psi_0$ needs the outcome GLM correct, or the candidate mechanism equal to $g_0$. Otherwise the interval covers the pseudo-true limit |
| coverage, parts (b) and (c) | coverage of $\psi_0$ needs the outcome GLM correct. Otherwise the interval covers the pseudo-true limit |
| regularity | regular in the declared outcome model. Under a correct candidate mechanism, part (a) is regular in the model where that candidate holds. No part claims regularity over the nonparametric model |

Acceptance: separate nonzero laws for the outcome and mechanism terms; a mutation that drops the
$\epsilon$ and $\gamma$ terms at a law where the design limit differs from the projection; a
registered study.

### X53. Missing-outcome CV-TMLE variants

| part | request | step |
| --- | --- | --- |
| (a) | `targeting_scheme="fold"` with missing outcomes | Montoya et al. (2023), Section 3.2, Step 2(c), with the composite indicator reduction. The coefficient is fit on the validation rows |
| (b) | `repeats` above 1 with the median | base: the shipped stacked contract (Zheng and van der Laan 2010, WP 273, Theorem 1, with the composite indicator reduction), for a fixed number of splits. Each draw is $\psi_0 + P_n D + o_P(n^{-1/2})$ with common nuisance limits, so the median is too. No independence between draws is needed. The split-dispersion term is $o_P(n^{-1})$, so its benefit is a finite-sample evidence claim |
| (c) | a cross-fitted `TMLE` with a missing treatment | a product of fold-local factors estimates $P(C_a = 1 \mid W)$ |
| (d) | shift, regime, controlled-direct-effect and MSM targets, cross-fitted with missing outcomes | Díaz, Williams, Hoffman and Schenck (2023), Theorem 3 (arXiv:2006.01366v4), with $(A, \Delta)$ as the exposure. An MSM adds the projection. The incremental and risk-ratio tilt axes are pending a source read of Kennedy (2019) and Kim et al. (2021), because Theorem 2 assumes that the regime does not depend on $P$ |
| (e) | the four stacked follow-ups | the fold-evaluated construction, supplied split plans, the bounded-continuous stacked natural-course mean, and stacked targets with baseline strata. Each has its own contract and evidence |

[F21](#f21-other-missing-outcome-cv-tmle-variants) keeps the cross-fitted `DRTMLE` compositions.
Acceptance: a nonzero witness for each changed term; a registered study for each part.

### X54. Small-sample evidence for clustered intervals and boundary nodes

The cluster-as-unit theory is the same at 4 clusters as at 10. Each part below needs a registered
measurement or a declared construction, not a theorem.
[F28](#f28-finite-sample-limits-of-clustered-intervals) stays the owner id of the clustered red
cells, and this row is the work that closes them.

| part | request | the evidence that reopens it |
| --- | --- | --- |
| (a) | an interval at 4 to 9 clusters, or at 4 to 19 for in-sample `LTMLE` | a registered measurement. At 4 clusters a pre-run probe found 0.25% to 9% of draws on which a fit cannot run, because the outcome does not vary among one plan's followers or a training fold lacks an outcome class |
| (b) | a cross-fitted `LTMLE` interval at 4 to 19 clusters | a registered measurement. `few-cluster-cross-fitted-ltmle` measures 20 and 30 clusters only |
| (c) | a small-sample reference other than $t_{J-2}$ | a reference with its registered study. Nugent et al. (2024), Section 2.2, and Benitez et al. (2023), Sections 3.1.2 and 3.2.1, state $t_{J-2}$ for cluster-randomized trials. Fang et al., arXiv:2510.08438, Section 3.3, gives another |
| (d) | a fold-evaluated interval at 40 clusters or more | a fold-evaluated degrees-of-freedom rule $t(\min(J-2, J-V))$ with its own registered study. The unequal-size study's pair at 40 clusters in 10 folds reads coverage 0.9325, with lower endpoint 0.918 against the 0.92 floor |
| (e) | the fold-evaluated variance at few clusters per fold | a calibration result for the centered fold cluster variance. The few-cluster study measures it at 2, 4 and 6 clusters per fold |
| (f) | a fold-evaluated point at informative unequal sizes with few clusters per fold | a construction whose bias at 2 to 6 clusters per fold stays inside 0.25 empirical standard deviations, such as a cluster-size-weighted fold average. The study reads a bias of 0.61, 0.43 and 0.32, where the stacked fit reads 0.14, 0.10 and 0.06 (`tests/unit/test_fold_evaluated_ratio_bias.py`) |
| (g) | a calibrated cross-fitted `DRTMLE` cluster variance at 10 to 30 clusters | a variance whose SE ratio stays inside the calibration band, or a redesigned IID control. The study reads cluster-robust SE ratios of 1.19, 1.14 and 1.12 at 10, 20 and 30 equal clusters |
| (h) | complex nuisance learners at about 20 clusters | a registered study. Wang, Park, Small and Li (2024), accepted manuscript PMC11795269, Section 4.2, Remark 3, recommend parsimonious working models at about 20 clusters. The registered evidence uses parametric nuisance learners only |
| (i) | an interval at a boundary node estimate, at a fixed law | a declared node-variance construction and a registered study where zero-event nodes are common. At a law with interior hazards, and a node risk set that grows linearly in $n$, a zero-event node has exponentially small probability. An interval that differs from the ordinary one only on that event is then valid at every fixed law under the shipped theorem. A law with a true node hazard of exactly 0 or 1 is neither interior nor in the local regime, so the construction must be conservative there, or F31 keeps that case |

[F31](#f31-inference-at-a-boundary-node-estimate) keeps the rare-event regime, and F28 keeps fits
below 4 clusters. Acceptance: each part's registered study passes under a declaration fixed before
its run.

### X55. Bootstrap extensions

| part | request | base | conditions |
| --- | --- | --- | --- |
| (a) | the natural-course, PAR and PAF bootstrap with missing outcomes, in sample | the bootstrap of a smooth M-estimator, Tang and Westling (2024), arXiv:2404.03064v2, Theorem 3.4 | finite-dimensional nuisances, read off the fit's own learners, as in X26; a registered study |
| (b) | the HAL-TMLE bootstrap | Cai and van der Laan (2020), arXiv:1905.10299, Section 3.6, Theorem 4 | in sample; a HAL learner, which does not ship; $C$ fixed at the selected value; a per-replicate update |
| (c) | the model-based targeted bootstrap | Coyle (2017), Berkeley dissertation, Section 1.4 and Appendix A.3, Theorem 9 (thesis numbering). The book chapter of Coyle and van der Laan (2018) was not read | resampling from the targeted fit, with Super Learner and TMLE refitted and retargeted on each sample. Theorem 9 needs the efficient score solved to the order it states, a negligible remainder, a uniform Donsker class, and $L_2$ convergence of $D^*$. The local read records that order as $o_P(1/n)$, and the row confirms it against Theorem 9 before it ships. The nuisance rates $o_P(n^{-1/4})$ under the targeted fit, positivity and bounded sectional variation come from Lemma 2 and Corollary 1, for the treatment-specific mean only. Other targets need their own rate lemma |

[F2](#f2-targeted-bootstrap-inference) keeps bootstrap consistency with data-adaptive or
cross-fitted nuisances. Acceptance: a registered coverage study for each part.

### X56. Competing-event intervention targets

| part | target | base | readiness |
| --- | --- | --- | --- |
| (a) | the risk under eliminated competing events | Young, Stensrud, Tchetgen Tchetgen and Hernán (2020), arXiv:1806.06136v3, Sections 5, 6.1 (Equation 23) and 8.1. The competing event is a censoring event. The shipped censored `LTMLE` fits a composite censoring indicator $1\{C_t = 1, D_t = 0\}$ placed after $D_t$ in the node order, with $C_t = 1$ meaning that the unit remains under observation | published support |
| (b) | separable effects with baseline covariates | Stensrud et al. (2022), arXiv:1901.09472v2, Section 5.2, formula (5), with the estimator of Wang et al. (2025), arXiv:2304.04904v1, under $Z_t := D_t$ | source audit |
| (c) | separable effects with time-varying common causes | Stensrud, Hernán, Tchetgen Tchetgen, Robins, Didelez and Young (2021), *Lifetime Data Analysis* 27(4):588-631, Section 6.2, Equation (19), with the Wang et al. (2025) estimator under $Z_t := (L_{A_D,t}, D_t)$ | source audit |

The source audit of parts (b) and (c) checks the node order, the user-declared partition of
$L_t$, and the efficient influence curve. No data check can verify the dismissible-component
conditions. Acceptance: oracle and evidence gates for each target in both directions.

### X57. Joint intervention axes and tilt compositions

| part | request | step |
| --- | --- | --- |
| (a) | two intervention keywords in one fit | separately targeted fits on the same rows, one per axis, then a fixed-dimension stack for the joint curve and covariance. Linearity gives cross-axis contrasts. An incremental axis does not share $g^*$ with a regime axis |
| (b) | `msm=` over declared regimes, shifts or tilts | the shipped MSM score with the arm indicator replaced by $1\{A = d(W)\}$ (an indicator reduction) or by $q(a \mid W)$ (a finite categorical sum). A tilt adds its mechanism term for each declared $\delta$. The score carries the effect modifiers $V$ |
| (c) | an arm MSM beside another axis | a fixed-dimension stack |
| (d) | the MNAR tilt on a shift fit or an incremental fit | the point curve $E[\bar Q_\gamma(d(A, W), W)]$ on a shift fit, and $E[\sum_a q_\delta(a \mid W) \bar Q_\gamma(a, W)]$ on an incremental fit. At $\gamma \neq 0$ the spread columns use the `plugin_std_err` and `plugin_interval_*` names, and no interval is claimed |
| (e) | a controlled direct effect under a tilt | a redirection to the shipped two-node `IPSI` plan, a tilt node and then a label node $Z = z$ (part (s) of the [natural-extension verdicts](technical-reference/natural-extension-verdicts.md)). The refusal names the call |

Acceptance: each composition reduces exactly to each standalone fit; nonzero controls for every
cross-axis block; a registered study for each interval claim.

### X60. Settings of the stacked and natural-course contracts

Two audited contracts refuse the requests below. The first is the cross-fitted arm-indexed
stacked contract with missing outcomes. The second is the missing-outcome contract of
`NaturalCourseMean`, PAR and PAF. That contract refuses the three targeting settings in sample and
cross-fitted. It refuses weights and clusters when it is cross-fitted. Each refusal raises before
any learner, in `estimators/tmle.py`.

| part | request | step |
| --- | --- | --- |
| (a) | `fluctuation="linear"`, one-step targeting, and `target_weights=True`, on both contracts | each setting changes the submodel only. The in-sample `TMLE` ships each setting for arm targets. [Estimator options](technical-reference/point-treatment-tmle.md#estimator-options) states that each one solves the same score. No new theory is needed |
| (b) | `att` and `atc` on the arm-indexed stacked fit | the shipped in-sample missing-outcome ATT and ATC curves, cross-fitted under Zheng and van der Laan (2011), WP 273, Theorem 2. That theorem leaves an empirical component untargeted, as the [references](references.md) record. For the ATT, that component is the treated share and the covariate law of the treated. The ATT coordinates join the joint fluctuation over the stacked validation rows |
| (c) | observation weights and clusters on both stacked fits | known weights in each targeting equation, and a cluster as the unit with whole-cluster folds. The clustered fit inherits the four conditions of [grouped folds](technical-reference/cv-tmle.md#grouped-folds): independent clusters, equal cluster sizes, no interference, and the usual remainder rates. [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) records the same conditions |
| (d) | `intermediate=` beside a natural-course target | the mean $E[\bar Q(A, z, W)]$, which sets $Z = z$ and leaves $A$ natural. It is an indicator reduction to $1\{Z = z\}$ with $(A, W)$ as the covariates, as X50 part (c) uses for $\Delta$. It holds under the controlled-direct-effect conditions $Y(a, z) \perp Z \mid A, W$ and $P(Z = z \mid A, W) > 0$. The contract states both. With missing outcomes the indicator is $\Delta \cdot 1\{Z = z\}$. PAR follows by linearity, and PAF by the delta method |

The curve of part (d) is
$1\{Z = z\}/P(Z = z \mid A, W)\,(Y - \bar Q(A, z, W)) + \bar Q(A, z, W) - \psi$. A reading of the
natural course as the observed $E[Y]$ sets no intervention. That request is `different_estimand`,
not a gap.

Acceptance needs each check below.

- unit weights reproduce the unweighted stacked fit exactly;
- one-row clusters with the same folds reproduce the iid stacked point and curve;
- part (d) at a law where every unit has $Z = z$ equals the shipped natural-course mean;
- each setting of part (a) solves the stacked score to solver tolerance at a finite law;
- a nonzero witness for each changed term;
- a pre-fit refusal test for each request that stays refused;
- a registered study for each interval claim.

### X61. Fold targeting and fold evaluation

The fold-specific update fits the fluctuation on the validation rows. Montoya, van der Laan, Skeem
and Petersen (2023), Section 3.2, Step 2(c), publish that update, as
[X11](#x11-learned-policy-follow-ups) part (h) records. Zheng and van der Laan (2011), WP 273,
choose one pooled fluctuation. The [references](references.md) record that WP 273 does not
validate a separate fluctuation in each fold. The parts below compose the fold-specific update, or the fold-evaluated report, with
settings that the package refuses.

| part | request | step |
| --- | --- | --- |
| (a) | `targeting_scheme="fold"` with `incremental=` | each fold's update solves the efficient influence function equation of Kennedy (2019), arXiv v3, Theorem 2, on its validation rows. That equation includes the mechanism term. The fold average is the estimator of Section 4.2, Algorithm 1, and Section 4.3, Theorem 3, gives its limit |
| (b) | baseline strata with `targeting_scheme="fold"` | a finite partition of the fold-specific update. Each stratum solves its own fold-specific equation, and the strata are stacked |
| (c) | `cv_evaluation=True` with `rr`, `or` or `msm=` | the delta method on the fold-evaluated arm means for a ratio, and a fixed full-rank projection of those means for an MSM. The estimator is the ratio of the fold averages, not the fold average of the fold ratios |

[F18](#f18-selector-path-c-tmle-inference) keeps a fold-specific selection on the C-TMLE selector
paths. [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) keeps the fold average of
`strategy="oat"`. [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) keeps `DRTMLE` with
fold targeting or fold evaluation.
[F32](#f32-fold-policies-that-read-the-treatment-or-the-outcome) keeps outer folds stratified on
the treatment or the outcome.

Acceptance needs each check below.

- one stratum reproduces the unstratified fold-specific fit exactly;
- the fold-evaluated ratio equals the ratio of the reported fold-evaluated arm means exactly;
- a nonzero witness for the mechanism term of part (a);
- a pre-fit refusal test for each request that stays refused;
- a registered study for each interval claim.

### X62. Other DR-TMLE and C-TMLE compositions

[F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) keeps the compositions that need a new
regularity or asymptotic argument.
[F21](#f21-other-missing-outcome-cv-tmle-variants) keeps `reduced_crossfit="nested"` on the
randomized missing-outcome `DRTMLE`. X50 and X51 hold the compositions that moved. This item holds
the other refused compositions of `DRTMLE` and `CTMLE`. `CausalStudy.available_methods()` names
this item for part (d). `estimators/drtmle.py` and `estimators/composite.py` refuse the other
parts before any learner.

| part | request | step |
| --- | --- | --- |
| (a) | observation weights on the randomized missing-outcome `DRTMLE` | known weights in each targeting equation, with the reduced regressions and $\pi$ fitted under weighted loss. The transport argument of the [DR-TMLE supported estimands](technical-reference/dr-tmle/supported-estimands.md) page states this step. The declared mechanism is the tilted-law mechanism only when the weights depend on $W$ alone. The contract requires that condition. Base: Díaz and van der Laan (2017), Section 2.1 and Equation (1) |
| (b) | `evaluation=` on the randomized missing-outcome `DRTMLE` | a diagnostic only, as X50 part (d). No new theory is needed |
| (c) | the composite missing-data route for a method other than `TMLE`, `DRTMLE` and `CTMLE` | the indicator reduction to $\Delta_A \cdot \Delta \cdot 1\{A = a\}$ for each method whose binary-treatment result ships. No shipped method reaches this refusal |
| (d) | `NaturalCourseMean`, PAR and PAF with complete outcomes under `CTMLE` or `DRTMLE` | the $\Delta \equiv 1$ case of X50 part (c) and X51 part (a). With complete outcomes the natural-course TMLE returns the sample mean and the curve $Y - E[Y]$, as the `ey_obs` row of the [evidence map](technical-reference/evidence.md) records. C-TMLE gives points only, as X51 states |

Acceptance needs each check below.

- part (d) equals X50 part (c) and X51 part (a) at $\Delta \equiv 1$, and its natural-course
  coordinate equals the sample mean exactly;
- unit weights in part (a) reproduce the unweighted fit exactly;
- a nonzero witness for each changed term;
- a pre-fit refusal test for each request that stays refused;
- a registered study for each interval claim.

### X63. Missing data, intermediates and designs

`CausalData`, the declarations of `data/validate.py` and `data/known_mechanism.py`, and the
natural-course support report refuse the requests below. A missing treatment beside a continuous
treatment or a categorical modified treatment policy is an `invalid_declaration`, not a gap. The
recording may depend on the treatment, so the law of $A$ given $W$ is not identified.
[F33](#f33-other-intervention-and-design-compositions) keeps the composite indicator with a
controlled direct effect.

| part | request | step |
| --- | --- | --- |
| (a) | a declared known density for a continuous treatment | a declared conditional density in place of the fitted one in the density ratio, for a policy whose definition does not read $g$. Base: Díaz, Williams, Hoffman and Schenck (2023), Theorem 3, whose remainder is a product of the ratio error and the outcome error. A declared $g_0$ makes the ratio error zero, as the [known treatment mechanism](technical-reference/point-treatment-tmle.md#known-treatment-mechanism) section states for discrete policies. The curve stays valid at $g_0$. The part claims no efficiency in the known-$g$ model, as that section states for the ATT |
| (b) | a multi-valued intermediate variable | the controlled direct effect at each level $z$, through the indicator $1\{Z = z\}$. The step is an indicator reduction |
| (c) | a fit with no covariates | an intercept-only nuisance design. The arm mean then reduces to the unadjusted mean. No new theory is needed |
| (d) | a support report of the response mechanism for the natural course with missing outcomes | a diagnostic over the fitted response probabilities. It makes no estimator claim |

Acceptance needs each check below.

- a two-level intermediate equals the shipped binary controlled direct effect;
- a fit with no covariates equals the unadjusted arm means exactly;
- a declared density equal to the fitted one reproduces the fitted-density fit;
- a nonzero witness for each changed term;
- a pre-fit refusal test for each request that stays refused;
- a registered study for each interval claim.

### X64. Intervention and policy declarations

`interventions/base.py`, `interventions/policy.py`, `msm.py` and `CausalStudy` refuse the
requests below before any learner.
[F33](#f33-other-intervention-and-design-compositions) keeps an `IPSI` plan with a modified-policy
or risk-ratio-tilt node, an MSM weight from a learned mechanism, and automatic cumulative-bound
selection for `LTMLE`.

| part | request | step |
| --- | --- | --- |
| (a) | `Stochastic(density_kind="estimated")`, and a user `Intervention` with an estimated density | the regime curve plus the term of Kennedy (2019), Appendix 8.2, Lemma 3. Kim, Kennedy and Naimi (2021), Appendix F, prove that term for a general intervention distribution. The term needs the derivative of $g^*$ with respect to $P$. A user-written density supplies none, so the part needs a declared family with its derivative. For a declared family it is a natural extension. For an arbitrary user density it stays refused |
| (b) | a policy with a continuous randomizer | an integral over the declared randomizer law. Base: Díaz, Williams, Hoffman and Schenck (2023), Section 2, journal page 849, and Theorem 3, as verdict (c) of the [natural-extension verdicts](technical-reference/natural-extension-verdicts.md) reads them. The law is known, so the chain rule with a known weight carries it. The contract requires the error of the numerical integral to be $o_P(n^{-1/2})$ |
| (c) | a known stochastic density on a continuous dose | the curve of Kennedy (2019), Appendix 8.2, Lemma 2: $g^*(A \mid W)/g(A \mid W)\,(Y - \bar Q(A, W)) + \int \bar Q(a, W) g^*(a \mid W)\,da - \psi$. The contract requires $g^*(\cdot \mid W)$ to be absolutely continuous with respect to $g(\cdot \mid W)$, with a bounded ratio. It states the error of the integral over $a$ |
| (d) | estimated MSM projection weights from arm shares, and an estimated MSM design such as a sample centre | a stacked estimating equation for the finite-dimensional weight or design |
| (e) | a method whose catalog does not declare the requested target | a catalog row and its evidence for each pair of method and target. The catalog row needs no new theory. Each pair inherits the readiness of the item that holds its construction. For example, `CTMLE` ATT stays in F5 |

Acceptance needs each check below.

- a continuous-randomizer declaration with two atoms equals the shipped finite randomizer;
- at an exact law, a known density equal to the true law of $A + \delta$ reproduces the shift
  target;
- a declared family that does not read $P$ reproduces the known-density fit, with a zero derivative
  term;
- a nonzero witness for each changed term, the mechanism term of part (a) included;
- a pre-fit refusal test for each request that stays refused;
- a registered study for each interval claim.

### X65. Sensitivity analyses beyond the shipped scope

The E-value, the derived risk ratio, the omitted-variable bound and the missingness tilt each
refuse the requests below.
[F34](#f34-e-values-beyond-the-total-population-arm-contrast) keeps the E-value among the treated
or the controls, and the E-value of a regime or policy contrast.

| part | request | step |
| --- | --- | --- |
| (a) | the standardized E-value with a response mechanism | a plug-in population standard deviation $\{\hat E[Y^2] - \hat E[Y]^2\}^{1/2}$. It reads two natural-course means under missing at random, of $Y$ and of $Y^2$, each with a fixed `q_bounds`. The shipped conversion uses its standard deviation as a plug-in, and this part does the same. The part depends on a continuous natural-course mean |
| (b) | the E-value, the derived risk ratio and the omitted-variable bound of a stratum-conditional estimand | the marginal result applied inside each stratum. The step is a finite partition |
| (c) | the derived risk ratio for a method other than `TMLE` | a same-seed equality witness for each method. No new theory is needed |
| (d) | the derived risk ratio after fold evaluation | the fold-evaluated ratio of X61 part (c). No new theory is needed beyond that part |
| (e) | the derived risk ratio of a controlled direct effect | the cached retargeting passes the fitted intermediate level. No new theory is needed |
| (f) | the omitted-variable bound after `repeats` above 1 | the bound computed on each draw, then the median, under the argument of X53 part (b). Each draw shares the first-order curve, so the median bound uses that curve, not a coordinatewise median of influence terms. X53 part (b) carries the TMLE point only. The contract therefore states that $\hat\sigma^2$ and $\hat\nu^2$ are each asymptotically linear, with common limits over the draws |
| (g) | a benchmark that drops every covariate | an intercept-only short model. It depends on X63 part (c) |
| (h) | the missingness tilt and `tipping_gamma` of the natural-course mean | the point curve $E[\Delta Y + (1 - \Delta)\bar Q_\gamma(A, W)]$. The spread columns take the `plugin_` names, as X57 part (d) |
| (i) | the missingness tilt of a ratio, a regime mean or an MSM coefficient | the tilted arm means, then the delta method for a ratio, an indicator reduction for a regime, and the fixed projection for an MSM. At $\gamma \neq 0$ the spread columns use the `plugin_std_err` and `plugin_interval_*` names, and no interval is claimed, as in part (h) and X57 part (d) |
| (j) | a dose-grid support diagnostic for a continuous MSM | the fitted treatment density at each grid dose, with the share of rows below a declared floor. It is a diagnostic |

Acceptance needs each check below.

- one stratum in part (b) reproduces the marginal result exactly;
- one repeat in part (f) reproduces the shipped bound exactly;
- a nonzero witness for each changed term;
- a mutation control for each part that adds a term;
- a registered study for each part that reports a confidence limit.

### X66. Simulated confounding beyond X47 and X48

The base law is the pinned DoWhy refuter, as in
[X47](#x47-simulated-confounding-laws-for-point-treatment-fits). Each part below needs a
declared reading of the strengths. No part makes an interval claim. The simulated-confounding
request checks in `sensitivity/` refuse each request before any refit.

| part | request | step |
| --- | --- | --- |
| (a) | an MSM with a link other than identity, log or logit | the replay refits the MSM under its own link. The law does not read the link |
| (b) | a regime class other than `Static`, `Rule` and `Stochastic` | the replay of a regime that is a known function of $W$. A user `Intervention` with `density_kind="known"` qualifies |
| (c) | a fixed regime or MSM beside another axis | a replay of each axis of the joint fit. It depends on X57 |
| (d) | a risk-ratio tilt fit | the contract declares one of the two shipped readings. The replay can rebuild the mechanism on each perturbed draw, as the incremental replay does. It can also apply the known rule to the perturbed treatment, as the policy replay does. `RiskRatioTilt` is a known rule: the `lmtp` `ipsi` rule, with a known two-point randomizer on the natural value |
| (e) | a binary replay under a method other than `TMLE`, `CTMLE` and `DRTMLE` | a refit of the stored estimator on each perturbed draw, with a zero-strength reproduction witness for each estimator |
| (f) | a continuous replay under a method other than `TMLE` | as part (e). A continuous fit under `DRTMLE` or `CTMLE` needs F5 first, because neither method fits a continuous dose |
| (g) | a fixed-policy replay under a method other than `TMLE`, `DRTMLE` and `CTMLE` | as part (e). X50 and X51 hold the DR-TMLE and C-TMLE regimes and MSMs |
| (h) | PAR or PAF under `DRTMLE` or `CTMLE` | as part (e). It depends on X50 part (c), X51 part (a) and X62 part (d) |
| (i) | a binary replay of a parameter off the arm axis, and a continuous replay of a parameter off the policy axis | the replay applied to the refit of that parameter, with a declared reading for each parameter. A parameter that reads $g$ follows part (d) |

Acceptance: the replay at zero strength reproduces the estimate; a nonzero witness for each part.

### X67. Refutation and assessment adapters

`refute` and `bootstrap_measurement_error` refuse the requests below. Each part needs a
generated-outcome process whose effect is exactly computable from its generated law. Such a part
needs an exact calculation and no new theory.

| part | request | step |
| --- | --- | --- |
| (a) | a longitudinal refute adapter and a longitudinal generated-outcome process | a sequential outcome process that reads the baseline covariates and the treatments only, or one that also generates each time-varying covariate. Then the sequential g-formula on the generated law gives the effect exactly |
| (b) | a generated-outcome process for another outcome family or a non-binary treatment | a registered process for each family and for each pair of arms |
| (c) | a generated-outcome process for a controlled direct effect, an MSM, an intervention, a ratio, or a functional other than `BackdoorMeanContrast` | a registered process for each target, with its effect computed exactly. An intervention that reads the propensity, such as an odds tilt, needs a process that also generates the treatment |
| (d) | a generated-outcome process with missing outcomes | a declared observation process under missing at random, and the effect computed on the complete law |
| (e) | `bootstrap_measurement_error` on a result that is not a point-treatment result, or on the strata variables | a perturbation of each node's covariates, and a declared reading that keeps the reported strata at their recorded values |

Acceptance: each process's effect checked against an exact calculation; a pre-fit refusal test
for each request that stays refused.

### X71. Repeated outcomes for the sequential baselines

`SequentialGComputation` and `SequentialIPW` refuse a repeated outcome, declared with
`outcome=Repeated([...])`, before any learner. `LTMLE` fits one: each horizon is the
end-of-study fit of the panel truncated after it
([repeated outcomes](technical-reference/longitudinal-tmle.md#repeated-outcomes)). The same
reduction applies to each baseline, and its stacked estimating equation covers several horizons
as it covers several plans.

| part | what to add | evidence |
| --- | --- | --- |
| (a) | `gcomp` and `ipw` in `REPEATED_OUTCOME_KINDS`, with the per-horizon fit of the truncated panel | `tests/unit/test_repeated_outcome_subclasses.py` requires each horizon's estimate and curve to equal the end-of-study fit of the truncated panel |

Acceptance: the exact reduction test for both baselines, and a registered property study of the
repeated-outcome cells.

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

## Future investigation contracts

Each contract below states the kept claim, the kind of argument that it needs, and the boundary
that ships. The [hard-stop review](technical-reference/hard-stop-review.md) gives the sources read
for each verdict.

### F12. Missing-outcome simulated-confounding replay

The pinned DoWhy refuter perturbs the treatment and the outcome only.
[X47](#x47-simulated-confounding-laws-for-point-treatment-fits) part (e) holds a declared response
mechanism that reads the covariates only. There $\Delta \perp (A', Y') \mid W$ holds, so missing
at random holds given $(A', W)$.

| kept claim | kind | missing argument |
| --- | --- | --- |
| a replay of the stored estimator when the response mechanism or the treatment-observation mechanism reads the treatment | identification | a flip of $A$ breaks missing at random given $A'$, so the stored estimator does not identify the perturbed law. A refit that carries the original $A$ as an auxiliary variable of the response model identifies it, because $\Delta \perp Y' \mid (A', A, W)$. That refit is a different estimator and needs its own contract |

### F13. Longitudinal simulated-confounding replay

The pinned DoWhy law is a single-time-point perturbation.
[X48](#x48-longitudinal-simulated-confounding-replay-with-a-time-fixed-latent) holds one time-fixed
latent on uncensored terminal-outcome fits with no time-varying covariate after the first
treatment node.

| kept claim | kind | missing argument |
| --- | --- | --- |
| censoring or a survival outcome | identification | the flip breaks coarsening at random at each censoring node, as in F12. An additive shift of an absorbing process is not a law on its support |
| a time-varying covariate after the first treatment node | identification | a covariate generated under the original $A_t$ depends on the latent given $A'_t$, and the sequential regression conditions on it |
| a later treatment node whose mechanism reads an earlier treatment | identification | the node was generated under the original earlier treatment, so given the misclassified one it depends on the latent. X48 moves it only when its contract defines that induced association, or restricts the mechanisms to ones that do not read an earlier treatment |
| a time-varying latent with treatment feedback | identification | a law whose strengths name the feedback. Tan (2025), arXiv:2308.15725v1, Equations (11) and (34), gives bounds and no law |

### F15. Controlled-direct-effect simulated-confounding replay

[X47](#x47-simulated-confounding-laws-for-point-treatment-fits) part (f) holds the outcome-axis
cells, and an intermediate-axis law that flips or shifts $Z$ and shifts $Y$ with $A$ held.

| kept claim | kind | missing argument |
| --- | --- | --- |
| the treatment axis on a fit with an intermediate | identification | a flip of $A$ makes the latent confound $Z$ and $Y$ through the misclassified $A$. No law names the strength of that path. `mediationsens` 0.0.3 simulates a pretreatment latent for natural effects only |

### F26. Confidence limits of the plug-in omitted-variable bound

`nu2_estimator="plugin"` estimates the Riesz second moment as $E_n[\hat\alpha^2]$. That value moves
at first order with the fitted treatment mechanism, because
$E[\hat\alpha^2] - \nu_0^2 = 2E[\alpha_0(\hat\alpha - \alpha_0)] + E[(\hat\alpha - \alpha_0)^2]$.
Its curve $\hat\alpha^2 - \hat\nu^2$ has no term for that fit.

On `tests/discrete_law.py` with the oracle nuisances, that curve differs from the Gateaux
derivative of $\nu^2$ by up to 21.3 for the ATE and 45.9 for the ATC. Chernozhukov et al. (2018),
arXiv:1608.00060v7, Section 1.1, record that bias for a non-orthogonal plug-in. DoubleML calls its plug-in fallback non-orthogonal.

| kept claim | kind | missing argument |
| --- | --- | --- |
| confidence limits of the plug-in bound for a learned mechanism other than the declared model of X26 | regularity | a learner-specific argument, such as an undersmoothing rule, that removes the first-order term. No standard step carries it |

The orthogonal estimator is the alternative. `nu2_estimator="doubly_robust"` estimates $\nu^2$
through the orthogonal score of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis,
arXiv:2112.13398v5, Section 4, Lemma 2. The repository's earlier citation of version 6, not re-read here, numbers that lemma Lemma 3. Their Theorem 4 gives the limits for
any mechanism learner at its rates. The refusal names this alternative.
[X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism) holds the
declared unpenalized logistic or multinomial mechanism and the declared known mechanism. A Super
Learner that contains that model stays here, because the bound reads the fit's own $\hat g$.

### F2. Targeted bootstrap inference

The package ships the full-refit bootstrap, in which each replicate refits the whole estimator. Its
registered study `full-refit-bootstrap-and-derived-contrasts` measures in-sample fits with
correctly specified nuisances, and it measures the point-treatment `TMLE` only. `TMLE` publishes
the percentile interval as inference for an in-sample point-treatment fit only. A cross-fitted
point-treatment fit prints its bootstrap as a diagnostic, as the cross-fitted longitudinal kinds
do. Every `DRTMLE` fit and every `CTMLE` fit prints it as a diagnostic too. An in-sample `TMLE`
fit with a data-adaptive learner still publishes the interval. The registered study measured only
correctly specified nuisances, and this row holds the theory for the data-adaptive case.

| kept claim | kind | missing argument |
| --- | --- | --- |
| full-refit bootstrap consistency with data-adaptive or cross-fitted nuisances | asymptotic | Tang and Westling (2024), arXiv:2404.03064v2, Theorem 3.4, needs condition (B2) for the bootstrap nuisance, and the paper does not treat cross-fitting. A resample can put copies of one row in a training fold and its validation fold |
| a row-resampling (nonparametric) targeted bootstrap for learners other than HAL | asymptotic | Cai and van der Laan (2020), arXiv:1905.10299, Theorem 4, covers a HAL learner with the bound $C$ fixed. Coyle (2017), Theorem 9, covers a model-based bootstrap under a uniform Donsker class. No read source covers the nonparametric bootstrap with another learner |

[X55](#x55-bootstrap-extensions) holds the in-sample bootstrap with finite-dimensional nuisances,
the HAL-TMLE bootstrap, and the model-based targeted bootstrap of Coyle (2017), Theorem 9, under a
uniform Donsker class. Resampling stored
curves, retargeting cached arrays and refitting the complete estimator are distinct procedures. Do
not infer one from the name of another.

### F16. Longitudinal sensitivity-bound estimation

Tan (2025) derives population sensitivity bounds for a terminal outcome under binary, static
longitudinal strategies. The table gives the three kept claims.

| kept claim | kind | missing argument |
| --- | --- | --- |
| (a) a sample estimate of the bounds | asymptotic | the consistency of a nested sample program with generated pseudo-outcomes at each period. Tan, arXiv:2308.15725v1, Section 3.1, leaves estimation with sample data to future work. Shen, Bacak and Kennedy, arXiv:2601.17779v1, Section 5.2, do the same for the incremental analogue |
| (b) an interval for that estimate | regularity | sampling inference for non-smooth bound functionals over periods. Bonvini et al., arXiv:2210.04681v2, Section 6.1, leave the time-varying constraint to future work |
| (c) an omitted-variable bound or an E-value on a longitudinal fit | identification | a per-period latent law that identifies the bias of a sequential g-formula target. The bound of Ding and VanderWeele (2016) reads a point exposure. Chernozhukov et al., arXiv:2112.13398v5, Section 4, bound a functional of a known regression form |

Both longitudinal refusals cite this item.

### F7. Time-respecting cross-fitting

Blocked-temporal and rolling-origin folds wait for a published TMLE result whose dependence
assumptions match the supported data. The kind of missing argument is serial dependence.
Ballinari and Wehrli (2024), arXiv:2411.10009v2, Theorem 4, and Semenova et al., arXiv:1712.09988v6,
give DML results under mixing, not TMLE results. The result must specify which rows may train each prediction
and which asymptotic argument licenses the interval. Ordered indices passed through iid fold
machinery are not sufficient.


### F18. Selector-path C-TMLE inference

The greedy, ordered and discrete paths report point estimates and path diagnostics. They refuse
`ci`, `pvalue` and `std_error` under the `working_mechanism_plugin` status. They report the
ordinary curve at the selected candidate as `plugin_std_error` and `plugin_interval`. That curve
does not establish conditional or unconditional coverage after selection. F18 reopens inference
when a published result supplies the influence function and covariance after the shipped
selection.

[X52](#x52-parametric-c-tmle-intervals) part (a) holds one declared partial candidate with
parametric learners, where a stacked estimating equation carries the fixed candidate. This item
keeps inference after the stopping-index selection. Its kind is adaptive-optimum inference.

F18 also keeps a fold-specific selection and update on the selector paths, which
`cv_evaluation=True` and `targeting_scheme="fold"` request. No read source derives one. `CTMLE`
refuses both before any learner, and the refusal names this item.

F18 is not a natural extension under the [Eligibility](#eligibility) rule. The selector is a
data-adaptive selection step, which the rule names as one with no established argument. No
published base exists for one stopping index shared across arms.

**The fixed-candidate curve comes first.** The curve is unproved when the candidate's mechanism
limit differs from the treatment law. On `make_instrument(n=2000, seed=44)`, an intercept-only
working mechanism gives a standard error of 0.0441. The sampling standard deviation of the
numerically identical least-squares coefficient is 0.0538. Over 300 greedy fits the ratio is
0.844, and nominal 95% intervals cover in 0.92 of fits.
`TestTheWorkingMechanismDiagnosticMissesTheExactVariance` in `tests/unit/test_ctmle.py` pins the
gap.

Ju, Benkeser and van der Laan (2020), Theorem 1 in Section 3.4 of arXiv:1806.06784v3, show
the form a correction can take: the ordinary curve at the propensity limit minus a term $D_r$
that vanishes at the true propensity. Their construction differs from the shipped selector. A
result must settle this fixed-candidate case before it addresses selection. X52 part (a) settles
it for parametric learners only.

**Sources read, and why none closes F18.** The [source audit](references.md#collaborative-tmle)
gives each locator.

| source | what it gives | what it lacks for F18 |
| --- | --- | --- |
| van der Laan and Gruber (2010), Section 4 | an abstract adaptive-mechanism contribution under fixed-limit and regularity assumptions. Section 4.3 leaves cross-validation over-selection open | the curve after the shipped stopping-index selection |
| Ju et al. (2018), Lemma 2 | a continuous-path contribution that is zero in one correct-outcome, product-rate regime | a discrete stopping index |
| Ju, Gruber et al. (2019), *SMMR* 28(2); Ju, Wyss et al. (2019), *SMMR* 28(4); Ju, Schwab and van der Laan (2019), *SMMR* 28(6) | ordinary EIF intervals. The last reports standard errors smaller than the sampling spread | a post-selection theorem |
| Cui and Tchetgen Tchetgen (2024), arXiv:1911.02029v6, Theorems 5.1 and 5.2 | an oracle inequality and selector consistency. Section 6 calls a Wald interval at the selected learners "completely blind to the model selection step" | a limit law |
| Liu (2018), Harvard dissertation, Chapter 2 | asymptotic laws for C-TMLE coefficients and risks, with known candidates, known outcome variance and a known covariate law, for one binary treatment-specific mean | learned nuisances, the discrete stopping rule and joint targets |
| Leeb and Pötscher (2006, 2008) | no locally uniform distribution estimation after model selection in finite-dimensional regression | a transfer to this selector |
| Loftus (2015); Markovic, Xia and Taylor (2017) | selective inference when the selection event is quadratic and Gaussian, or the criterion and target are jointly Gaussian | either property for a targeted-loss selector |
| van der Laan, Carone, Luedtke and van der Laan (2026), [arXiv:2307.12544v2](https://arxiv.org/abs/2307.12544v2), Section 2.2; adaptive debiased machine learning v2. The review did not read Theorem 5 or Corollary 1 | inference after data-driven model selection when the selected model approximates an unknown, fixed oracle submodel | a fixed oracle submodel for the shipped path, and a proof that one global depth from nested targeted-loss folds meets the conditions of that framework |
| Qiu, Luedtke and Carone (2021), arXiv:2003.01856v2, Theorem 4 in Section 4.4 | the ordinary influence function after cross-validation selects a sieve dimension, under Condition C5 at every candidate | a targeting step along a propensity path |
| Dang, Tarp, Abrahamsen et al. (2025), *Journal of Causal Inference* 13:20240041; Bibaut and van der Laan, arXiv:1706.07408v2, Theorems 1 and 2 | a selector trained inside each fold, with a nonstandard limit | this criterion, target and shared stopping index |
| Rothenhäusler (2024), *Electronic Journal of Statistics*, Section 3.4, Theorem 2. The preprint arXiv:2008.12892v2 numbers it Section 3.5, Theorem 4 | intervals after a choice among a fixed finite set of estimators, "usually not uniformly valid" | a fixed set. The shipped path is built from the data |
| Schnitzer, Lok and Gruber (2016), Section 5.3 and Table 3 | simulated under-coverage of ordinary curves for TMLE and C-TMLE with Super Learner | a construction |

Three further sources were read from their abstracts only: Schnitzer, Sango, Ferreira Guerra and
van der Laan (2020); Zrnic and Jordan (2023); and Van Lancker, Díaz and Vansteelandt (2024),
arXiv:2404.11150v2.

**Routes that a result can take.**

| route | what it needs |
| --- | --- |
| a uniform expansion | every candidate has the same deterministic curve $D_0$, with $\widehat\psi_{n,k}-\psi_0=(P_n-P_0)D_0+r_{n,k}$ and $\max_{k\leq K_n}\lVert r_{n,k}\rVert=o_p(n^{-1/2})$. Substitution of any data-dependent $\widehat k$ then keeps the expansion, and Cramér--Wold gives the vector result at a fixed number of arms. It is plausible when every candidate converges fast to the same nuisance limits, and not under one-sided robustness |
| selector stability | a unique oracle candidate, a risk margin, uniform risk convergence, and a stable covariate identity at each depth. An oracle risk inequality alone proves none of these. Shao's linear-model result shows that prediction-efficient fixed-fold cross-validation need not select consistently |
| a construction change | select the depth inside each outer training fold, and evaluate on its held-out rows only. Orthogonal-score arguments then treat the selector as nuisance learning, under its rate conditions |
| a data-adaptive target | honest splitting permits any target generation, and the same-sample theorem needs a uniform expansion, Donsker control and curve convergence. Either regime concerns a data-adaptive estimand, not the fixed target that the package reports |

The oracle projection's efficiency bound is typically smaller than the nonparametric bound. A
result could therefore confirm the current curve or require a different one.

**Acceptance.**

- The result covers the selected index and the selector's three split layers: outer nuisance
  folds, selection folds and inner selection-training folds. Learner folds and repeat draws are
  separate layers.
- It establishes whether the current curve suffices or an added contribution is needed. It
  states the remainder, the rate conditions and the covariance for every supported target vector.
- Each analysis regime is its own proof cell. An iid, complete-outcome, unweighted, unstratified,
  single-repeat result certifies only that cell. Missing outcomes need the response-mechanism
  terms, fixed weights a weighted empirical-law result, estimated weights their first-stage
  contribution, clusters a cluster-level expansion, strata stratum-specific expansions and their
  joint covariance, and repeats the median and split-dispersion aggregation.
- The multi-arm case picks one stopping index for every arm. A result gives the full vector
  influence function and its cross-covariance with the ordinary curve, and it differentiates the
  reduction of the target vector to one scalar risk. A per-arm copy of a binary correction does
  not recover those terms.
- A new contribution propagates through pointwise and simultaneous inference. The evidence needs a
  fixed-candidate reduction and repeated-sampling coverage where the chosen candidate changes.
  Each repeat keeps its path and its selected index.
- [Red-cell owners](#red-cell-owners) names the cells of the
  [point-treatment](technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md)
  and [multi-arm](technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md) selector
  studies that must pass.


### F19. Outcome-adaptive C-TMLE generated-design inference

Outcome-adaptive C-TMLE selects no candidate. Its shared design, `oat_design="shared"`, fits one
categorical treatment mechanism on the estimated vector of arm-specific outcome predictions. Its
per-arm design, the default, fits one binary mechanism per arm on that arm's own prediction.
Every `oat` fit takes the `generated_design_plugin` status, including an `ey1`-only request and
a per-arm fit with `strata=`, `delta=` or `treatment_delta=`. `ci`, `pvalue` and `std_error`
refuse, and `plugin_std_error` and `plugin_interval` keep the ordinary adaptive-propensity
diagnostic. This item holds the interval of the per-arm
[strata and missing-data fits](technical-reference/collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design)
as well. Their observation factors are generated regressors in the same sense.

The cross-fitted fit follows the paper's fold-local nuisance nesting. One outcome model, fitted on
fold `v`'s training rows, creates both sides of that fold's generated design. The adaptive
propensity is fitted on the training side only, and both nuisances are evaluated on the held-out
side. A mutation test changes every evaluation outcome without moving either nuisance on those
rows.

[X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) holds the per-arm design with a regular parametric outcome model:
a corrected curve, its covariance and its Wald interval under a declared support predicate.
[X52](#x52-parametric-c-tmle-intervals) part (b) holds the composite per-arm cells, and part (c)
holds the shared design with a parametric outcome and design. This item keeps five claims. The table gives the kind of
argument that each one needs.

| kept claim | kind of missing argument |
| --- | --- |
| a flexible outcome learner. The Riesz representer of the generated-design term reads $1/g_0$ | asymptotic |
| a nonparametric shared multinomial design | regularity |
| the cross-fitted construction. Appendix D states no theorem | asymptotic |
| the $(1/V)\sum_v$ fold average under `cv_evaluation=True`, and a fold-specific update under `targeting_scheme="fold"`. Appendix D outlines the fold average and states no theorem. `CTMLE` refuses both before any learner | asymptotic |
| local uniformity over the nonparametric model | asymptotic |

**The published base.** Benkeser, Cai and van der Laan (2020), *Statistical Science*, Theorem 1,
proves under six regularity conditions that the ordinary curve needs no first-order
generated-design term for one binary treatment-specific mean. This item names each appendix by
its letter in arXiv:1901.05056v1. Appendix D builds a binary ATE with both arm predictions and one
signed fluctuation coefficient, states no theorem for it, and outlines a pooled-validation
CV-C-TMLE. The article points to Appendix G for the conditions of Theorem 1, and Supplement A
labels them Appendix F. The [references](references.md) entry records the mismatch.

**The per-arm design's generated-design term.** The per-arm design is Theorem 1's construction
for each arm. Theorem 1 needs its condition (v), which bounds the derivative of
$P(A = a \mid \bar Q_n, \bar Q_0)$ in $\bar Q_0$. For a root-$n$ outcome regression on a
continuous $W$, that derivative grows like $n^{1/2}$, so the remainder $R_{24}$ is first
order.

For a parametric outcome model, the term is $c^\top \mathrm{IF}_\beta$, with
$c = E[(1 - g_0 / G)\, \partial \bar Q / \partial \beta]$, when the design learner is consistent
for $G = P(A = a \mid \bar Q(a, W))$. With a parametric design learner, the $\epsilon$ and
$\gamma$ terms also enter. [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) plans to derive and implement the
complete term. The x17
ratios below measure the omission on one law. They do not show that it is zero.

The registered study
`ctmle-oat-per-arm` measured standard errors 8 to 24 percent too small, flat in $n$, and its
cells with the design pinned to the truth were calibrated.
`tests/diagnostics/x17_generated_design/run.py` adds the term with the true $g_0$ and restores the
ratio to 1.03 at $n$ = 1,500 and 0.98 at $n$ = 6,000. [The applied revert](technical-reference/collaborative-tmle.md#the-applied-revert) gives the
cells. The term needs $g_0$, which the design exists to avoid, and the influence function of the
outcome learner. No source read here gives either for a flexible learner with the targeting
step.

**What ships against Appendix D.** `CTMLE` refuses `targeting_scheme="fold"`, so the fluctuation is
one pooled coefficient on the stacked out-of-fold rows, as Appendix D outlines. Two divergences
remain.

| divergence | what ships | what Appendix D outlines |
| --- | --- | --- |
| the final average | the stacked whole-sample plug-in, because `CTMLE` refuses `cv_evaluation=True` | the $(1/V)\sum_v$ fold average. With fixed $V$, near-balanced unweighted folds and bounded predictions, the difference is $O(V/n)$ |
| the fluctuation dimension | a joint fluctuation with one column for each arm, from which the means, ATE, RR and OR follow | one signed coefficient for the binary ATE |

**Why the nonparametric shared design is not a natural extension.** It fits one shared
multinomial mechanism on `K` estimated columns and uses a `K`-column joint fluctuation. Cramér--Wold turns
proved scalar joint expansions into a vector limit. It does not prove those expansions, their
remainders, or the covariance that the shared learned mechanism induces. The cross-fitted default
is further outside Theorem 1, because Appendix D only outlines its proof.

**Sources read, and why none closes F19.**

| source | what it lacks |
| --- | --- |
| archived `ctmle3` and the `drtmle` `adapt_g` option | each is implementation provenance and reports the ordinary curve. Neither is an inference theorem |
| DOPE, Theorem 4.3 and Proposition 4.4 | it proves inference for a data-adaptive target conditional on a representation learned on an independent sample. A fixed target can need a further delta-method variance. Its appendix leaves the dependent fold-oracle proof open |
| outcome-adapted AutoDML | a sample-split result, and no result for the cross-fitted construction |
| Escanciano and Pérez-Izquierdo (2023) | second-step orthogonality removes the indirect first-step effect. The direct effect of learning the generated regressor remains |
| Zhang, Shao, Yu and Wang (2018); Ma, Zhu, Zhang, Tsai and Carroll (2019), abstracts only | estimated dimension reduction can change the variance or keep efficiency. Neither has a targeting step or a shared multinomial mechanism |
| Schnitzer, Talbot, Liu et al. (2026), *Statistics in Medicine* 45(1-2):e70316, abstract only | it selects covariates for a longitudinal treatment model, and fits no mechanism on estimated outcome predictions |

**Open items.** A result states a verdict on each one.

| open item | what a result must settle |
| --- | --- |
| the per-arm generated-design term | an analogue of $c^\top \mathrm{IF}_\beta$ for a flexible outcome learner, or an estimator of the term that does not need a consistent full mechanism when no other derivation supplies one, or a source that shows the term vanishes under a stated condition. [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) may use a known or consistently estimated full mechanism, and it does not need to settle this item |
| one shared multinomial | one categorical fit on `K` estimated columns supplies every arm's clever covariate, so an inconsistent column for one arm enters every other arm's mechanism |
| vector target and simultaneous inference | the joint covariance and the simultaneous critical value, not the per-arm variance alone |
| uniformity, for both designs | each estimator is superefficient by design, so a pointwise limit law does not give locally uniform coverage. A pointwise claim under stated regularity conditions, such as the [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) claim, does not need a uniformity result. It states that it is pointwise |
| transport of the shared design beyond the source law | missing outcomes need the response-mechanism expansion. Fixed weights need a weighted empirical-law result, and estimated weights a first-stage contribution. Clusters and strata need dependence- and stratum-specific expansions. Repeats need the median and split-dispersion aggregation |

**Acceptance.**

- A result covers the exact cross-fitted multi-arm construction, and it separates a fixed target
  from a target conditional on the learned design.
- For the per-arm design, the revert cells of `ctmle-oat-per-arm` pass their declared bands under
  a fresh declaration. A support predicate for the validated configurations replaces the flag
  `OAT_PER_ARM_INFERENTIAL`. [X58](#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) plans this for its
  scope.
- It establishes whether the current curve suffices or an added representation contribution is
  needed.
- It covers the requested joint means or contrasts and their covariance. It states the remainder,
  the nuisance-rate conditions, and whether its claim is pointwise or locally uniform.
- A new contribution propagates through pointwise and simultaneous inference.
- The evidence needs a fixed-design reduction, a generated-design comparison, and registered
  coverage for every claimed treatment and target dimension.
- A contraction of the generated-design pair needs a two-sided equivalence margin or a declared
  sample-size ladder. A deliberately invalid design can serve as a negative control, and the
  learned design cannot.

The registered evidence is in the
[point-treatment](technical-reference/method-evidence/outcome-adaptive-point-treatment-c-tmle.md)
and [multi-arm](technical-reference/method-evidence/outcome-adaptive-multi-arm-c-tmle.md) studies.
The point-treatment generated-design pair passes its own calibration bands. The multi-arm pair
passes its coverage band, and both of its standard-error intervals cross the 1.07 upper bound. No
result there identifies a first-order term.


### F21. Other missing-outcome CV-TMLE variants

[X53](#x53-missing-outcome-cv-tmle-variants) holds the `TMLE` variants: fold targeting, the median
over a fixed number of splits, a cross-fitted missing treatment, the off-contract axes, and the
four stacked follow-ups. This item keeps the cross-fitted `DRTMLE` compositions.

| kept claim | kind | missing argument |
| --- | --- | --- |
| a cross-fitted `DRTMLE` with `delta=` or a missing treatment | asymptotic | the package's own argument for the cross-fitted DR-TMLE leaves the pooled reduced cross-fit term [B] open ([DR-TMLE targeting](technical-reference/dr-tmle/targeting.md)). No cross-fitting lemma reaches it |
| `reduced_crossfit="nested"` on the composite | asymptotic | the nested construction removes term [B], so it is the candidate route for a later reading. The cross-fit refusal comes first today, so no request reaches the nested composite |
| `reduced_crossfit="nested"` on the randomized missing-outcome `DRTMLE` | asymptotic | the nested construction needs the argument that the composite row lacks. No read source gives it for the randomized missing-outcome route. `DRTMLE` refuses it before any learner |

A cross-fitted `DRTMLE` fit with missing outcomes refuses before any learner, and its message
names this item. A C-TMLE selector fit takes `working_mechanism_plugin`, which
[F18](#f18-selector-path-c-tmle-inference) holds. Every `strategy="oat"` fit takes
`generated_design_plugin`, which [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference)
holds.

### F22. Grouped cross-fitting beyond point-treatment TMLE

The package draws whole-cluster outer folds for the ordinary cross-fitted point-treatment TMLE,
DR-TMLE and longitudinal TMLE. C-TMLE refuses `id=` at every `cross_fit` setting.
[X51](#x51-c-tmle-point-estimates-for-further-targets) part (e) holds the clustered point
estimates, with a cluster as the unit and whole-cluster folds. The clustered versions of the
parametric cells, X58's cells included, move with [X52](#x52-parametric-c-tmle-intervals) part (d). This item keeps every
other clustered C-TMLE interval. Its kind is asymptotic.

| composition | what ships | what a result must supply |
| --- | --- | --- |
| a C-TMLE interval with `id=` | refused at every `cross_fit` setting (`CTMLE._resolve_estimands_for_data`) | a split law for the selection folds and the nested selection folds under clustering, and the cluster-robust variance of the candidate the search stops at. [F18](#f18-selector-path-c-tmle-inference) and [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) are open for iid rows, so a clustered result needs them first |

The other grouped compositions take the cluster as the sampling unit, and the
[natural-extension verdicts](technical-reference/natural-extension-verdicts.md) qualify them as
natural extensions. The cross-fitted interval at unequal cluster sizes and the $t$ reference at
few clusters ship ([clusters](technical-reference/inference.md#clusters)), and
[F28](#f28-finite-sample-limits-of-clustered-intervals) holds their finite-sample limits.
Cross-fitted longitudinal TMLE with `id=` ships with whole-cluster folds
([longitudinal clusters](technical-reference/longitudinal-tmle.md#clusters)).

The grouped point-treatment split is supported for the partition alone. Wang, Park, Small and Li
(2024), Section 4.2 and Theorem 4(b), prove a cross-fitted result under a random, roughly equal
partition of the clusters. Their estimator is AIPW-type with a cluster-level treatment. The rest is
the package's own estimating-equation argument, which the
[fold and outcome-scale rules](technical-reference/cv-tmle.md#grouped-folds) state with its four
conditions: independent clusters, equal cluster sizes, no interference, and the usual remainder
rates. The registered
[clustered point-treatment CV-TMLE study](technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md)
is its only empirical witness.

The source audit found related work.
[Grouped folds and clustered cross-fitting](references.md#grouped-folds-and-clustered-cross-fitting)
gives every source the audit read, with the version whose locators it used.

| source | what it covers |
| --- | --- |
| Nugent et al. (2024) | grouped folds and aggregated cluster influence curves in partially clustered trials |
| Balkus, Laith and Hejazi (2026) | splitting correlated units can still remove an empirical-process term under their conditions |
| Karim (2026) | point-treatment survey TMLE |
| the JSS `ltmle` article | longitudinal software |
| Zeileis, Köll and Graham (2020) | clustered sandwich covariances for regression models |

None supplies a split law or a variance for the clustered C-TMLE search.


### F5. Other refused C-TMLE and DR-TMLE compositions

[X50](#x50-further-dr-tmle-compositions) holds the DR-TMLE regimes, discrete-arm MSMs, the natural
course with missing outcomes, `evaluation=` on the composite, the omitted-variable bound, and
estimated weights with a guard. [X51](#x51-c-tmle-point-estimates-for-further-targets) holds the
C-TMLE point estimates for the natural course, regimes, discrete-arm MSMs, a selector missing
treatment, clusters, and the omitted-variable bound.
[X62](#x62-other-dr-tmle-and-c-tmle-compositions) holds the other compositions that a natural
extension carries. This item keeps the compositions below.

| kept claim | kind | missing argument |
| --- | --- | --- |
| C-TMLE ATT and ATC | regularity | the clever covariate and the treated covariate law read $g$, so the target is not a function of $Q$ alone. van der Laan and Gruber (2009), WP 246, Theorem 2, needs that form |
| C-TMLE incremental, modified treatment policies and the risk-ratio tilt | regularity | each candidate $g$ of the collaborative search defines a different parameter, so collaborative consistency fails. The risk-ratio tilt is the `lmtp` `ipsi` rule, a known two-point randomizer on the natural value, so it joins the modified treatment policies |
| C-TMLE controlled direct effect | regularity | a collaborative path over two mechanisms. Schnitzer, Sango, Ferreira Guerra and van der Laan (2020), *Biometrics* 76(1):145-157, may give a published point method. The review read its abstract only, and its first-hand read is pending |
| C-TMLE on a continuous dose | regularity | a collaborative candidate sequence of conditional density estimators. No read source derives one |
| C-TMLE on a longitudinal functional | regularity | a sequential collaborative score. No read source derives one |
| DR-TMLE ATT, ATC, modified treatment policies, the risk-ratio tilt and incremental | asymptotic | Benkeser, Carone, van der Laan and Gilbert (2017), PMC5793673, Section 4, name other parameters as future work. Díaz et al., arXiv:2006.01366v4, Theorem 2, assumes that the regime does not depend on $P$ |
| DR-TMLE controlled direct effect and longitudinal functionals | asymptotic | Benkeser et al. (2017), Section 1, name interventions at several time points as future work |
| DR-TMLE on a continuous dose, with its evaluation companion and its nested construction | asymptotic | reduced regressions that condition on a conditional density in place of an arm probability. No read source derives them |
| DR-TMLE combined with C-TMLE | asymptotic | a reduced regression conditions on the collaborative $\hat g$. The reduced-dimension correction needs $\hat g \to g_0$, which C-TMLE does not give |
| a partial `guard` on the randomized missing-outcome DR-TMLE | asymptotic | the shipped algorithm targets the treatment, observation and outcome correction blocks jointly. No read source gives a result for a subset of the three blocks |
| DR-TMLE with `targeting_scheme="fold"` or `cv_evaluation=True` | asymptotic | each fold's reduced regressions and alternation, and a corrected curve under fold-wise evaluation. No read source gives a fold-evaluated DR-TMLE, as F27 records for the learned-rule value |

Ordinary-TMLE implementations do not establish the C-TMLE or DR-TMLE corrections for these
compositions. Each pre-fit refusal names this item and an alternative that ships, in most cases a
plain `TMLE`.

### F27. Learned-policy value outside the published conditions

The [learned-rule contract](technical-reference/point-treatment-tmle.md#learned-rules) refuses each
request below before any learner, except the exceptional-law row. The
[boundary study](technical-reference/method-evidence/learned-rule-cvtmle-boundary.md) measures the
exceptional-law row and the `weak_blip` row. F27 owns every red cell of the boundary study.
[X11](#x11-learned-policy-follow-ups) parts (c) and (i) to (o) hold the compositions that moved.

| kept claim | kind | missing argument |
| --- | --- | --- |
| an interval at an exceptional law, where the limiting-rule condition C3 of that contract fails | adaptive-optimum inference | a CV-TMLE interval for the fold-average target without a limiting rule. van der Laan and Luedtke (2015), Theorem 6, and Hubbard et al. (2016), Theorem 1, both need a limit curve. Luedtke and van der Laan (2016), *Annals of Statistics*, Section 4.2, name inverse weighting by the standard deviation and a central limit theorem for triangular arrays. Section 5 applies them to the optimal value. Fava (2026), arXiv:2610.02944v1, covers a known-propensity IPW score only |
| the full-refit bootstrap | asymptotic | each resample relearns the rules, so each draw has a different target. No read source gives the bootstrap law of a data-adaptive target |
| a continuous treatment | regularity | a deterministic dose rule gives the value of fixed doses, which is not pathwise differentiable |
| a fold-evaluated `DRTMLE` learned rule | asymptotic | no read source gives a fold-evaluated DR-TMLE. `DRTMLE` refuses `cv_evaluation=True` for that reason |
| `CTMLE` with `learned_rule=` | adaptive-optimum inference | the selector and outcome-adaptive paths have no interval for any target, as F18 and F19 record |
| missing outcomes or a missing treatment with `learned_rule=` | asymptotic | van der Laan and Luedtke (2015) treat complete outcomes, and no read source gives a cross-fitted learned-rule fit with missing outcomes or a missing treatment. The missingness tilt is unreachable for the same reason. The missing-treatment refusal (`composite.py`) cites this row |
| an E-value for a learned-rule contrast | regularity | no E-value of a regime contrast ships (`evalue.py`, the arm-contrast check). A regime contrast is not a contrast of two exposure levels, so the bound needs its own mapping |

The `weak_blip` row has no missing result. That law is within C3, and Theorem 6 of van der Laan and
Luedtke (2015) covers it. A red cell there is a finite-sample limit at n = 2,000, and F27 holds it
as the owner of the reporting study.

### F28. Finite-sample limits of clustered intervals

A clustered `TMLE` or `DRTMLE` fit reports a Student $t$ interval with $J-2$ degrees of freedom
when it reads 10 to 39 clusters with positive weight mass, and an `LTMLE` fit, in sample or
cross-fitted, when it reads 20 to 39 ([clusters](technical-reference/inference.md#clusters)). Below that floor, in the
fit or in one reported baseline stratum, the fit takes the `"few_cluster_plugin"` status. Each
floor is the smallest count the registered few-cluster study measures for that fit.

| kept claim | kind | missing argument |
| --- | --- | --- |
| an interval below 4 clusters | arbitrary dependence | a supported construction for a fixed number of dependent units. Below 4 clusters the reference has fewer than 2 degrees of freedom, and at 2 clusters or fewer no independent-cluster argument applies |

[X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) parts (a) to (h) hold
the registered measurements from 4 clusters up to each floor, and the other small-sample work. The
cluster-as-unit theory is the same at 4 clusters as at 10, so that work needs evidence and not a
theorem. F28 stays the owner id of the clustered red cells, and X54 is the work that closes them.

### F31. Inference at a boundary node estimate

A longitudinal fit reads a node regression whose fitted rows all hold 0, or all hold 1, as that
value ([a node with no event](technical-reference/longitudinal-tmle.md#a-node-with-no-event)).
The point estimate is the maximum-likelihood value, and `survtmle` reads such a node the same
way. The node's term of the influence curve is then identically zero, although the true hazard
there can be positive. So the plug-in standard error leaves that node's variance out.

[X54](#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) part (i) holds the
interval at a fixed law. There a zero-event node has exponentially small probability, so the
shipped theorem covers an interval that differs from the ordinary one only on that event.

| kept claim | kind | missing argument |
| --- | --- | --- |
| an interval in the rare-event local regime, where the node hazard $h_n$ tends to 0 | asymptotic | a limit theorem for a TMLE parameter whose node hazard shrinks with $n$. Boundary theory, such as Andrews (1997), assumes the truth on the boundary. The review read its abstract only |

### F32. Fold policies that read the treatment or the outcome

`stratify_folds` can draw the outer folds inside strata of the treatment or the outcome.
`learners/crossfit.py` refuses that split before any learner. The
[fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) give
the audit. Zheng and van der Laan (2011), WP 273, draw the split vector uniformly over a finite
support. Polley (2010) proves the Super Learner oracle result for folds drawn independently of the
learning sample.

| kept claim | kind | missing argument |
| --- | --- | --- |
| outer folds stratified on the treatment or the outcome | asymptotic | a result that keeps the cross-fitted remainder argument when the partition is a function of $A$ or $Y$. Each read result draws the split independently of the data |

### F33. Other intervention and design compositions

Each request below is well posed, and each one refuses before any learner.
[X63](#x63-missing-data-intermediates-and-designs) and
[X64](#x64-intervention-and-policy-declarations) hold the declarations that a natural extension
carries. This item keeps the compositions that need an argument of a new kind.

| kept claim | kind | missing argument |
| --- | --- | --- |
| the composite missing-treatment indicator with a controlled direct effect | identification | a stated condition that ties $Z$ to the recording of the treatment. No source read here states one. The shared preflight in `estimators/composite.py` refuses the fit |
| an `IPSI` plan with a modified-treatment-policy node or a risk-ratio-tilt node | regularity | a density ratio at each modified-policy or risk-ratio-tilt node inside the cross-fitted `IPSI` estimator, beside the mechanism term of each odds-tilt node. No read source derives the mixed plan. `longitudinal/regimen.py` refuses it. [X57](#x57-joint-intervention-axes-and-tilt-compositions) holds the point-treatment joint axes |
| an MSM projection weight from a learned mechanism | regularity | the weight is a functional of $P$, so the curve needs the pathwise derivative through the learned mechanism. No read source gives that term for the shipped projection. `msm.py` refuses it. X64 part (d) holds finite-dimensional weights and designs |
| automatic selection of the cumulative bound for `LTMLE` | adaptive-optimum inference | inference after a data-adaptive bound selector. `utils/bounds.py` refuses a named selector. The fixed heuristic (0.01, 1.0) ships as the default |

### F34. E-values beyond the total-population arm contrast

The shipped E-value converts a risk ratio of two exposure levels in the total population. Ding and
VanderWeele (2016), *Sensitivity analysis without assumptions*, give its bounding factor.
[X49](#x49-e-value-for-a-controlled-direct-effect) applies that bound to each arm of a controlled
direct effect. The requests below refuse before any computation.

| kept claim | kind | missing argument |
| --- | --- | --- |
| the ATT and ATC E-value on a binomial outcome | identification | a cited bound for the causal risk ratio among the treated or the controls, with its reading of the strengths. Among the treated only the control-arm counterfactual is confounded, so the total-population factor does not carry over by name. No source read here states that bound. `sensitivity/evalue.py` refuses the request |
| the E-value of a regime, policy or incremental contrast, and the derived risk ratio off the arm axis | identification | a regime contrast is not a contrast of two exposure levels, so the bound needs its own mapping, as [F27](#f27-learned-policy-value-outside-the-published-conditions) records for a learned rule. The derived ratio off the arm axis has no consumer until that mapping exists. `sensitivity/_derived.py` refuses it |

### F35. Inference for sequential baselines with data-adaptive nuisances

`SequentialGComputation` and `SequentialIPW` report an interval only when every regression they
read is a `GLM`. The interval is then the stacked estimating equation of the parametric fits.
With another learner the estimator solves no efficient influence-curve equation. Its first-order
bias is then of the order of the learner's error.

| kept claim | kind | missing argument |
| --- | --- | --- |
| an interval for sequential g-computation with a data-adaptive outcome learner | asymptotic | an asymptotic-linearity theorem for the iterated conditional expectation plug-in over several time points, such as an undersmoothing result. No source read here derives one |
| an interval for longitudinal IPW with a data-adaptive mechanism learner | asymptotic | the same result for the cumulative weights. Ertefaie, Hejazi and van der Laan (2023), Section 3, Theorem 1, cover IPW with an undersmoothed highly adaptive lasso propensity at a single time point only |
| a cross-fitted baseline, `n_folds` above 1 | asymptotic | the same result for cross-fitted nuisances. Cross-fitting does not remove the first-order bias of a plug-in or IPW estimate, so the in-sample result does not carry over |

## Reading a gap correctly

Not every absence is missing package functionality. The five reason codes in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal)
separate an `implementation_gap` and an `inference_gap` from an `invalid_declaration`, a
`different_estimand`, and an `api_redirection`. Only the first two belong on this roadmap.
Each `implementation_gap` refusal names the item that tracks it.
`tests/unit/test_refusal_codes.py` checks each one in the source and in each module-level record.
