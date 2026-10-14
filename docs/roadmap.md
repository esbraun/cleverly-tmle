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

## Remediation

This queue holds the remediation work that must be complete before a beta release. It holds
shipped behavior that needs a correction or a recorded decision, and a published method needed to
resolve a shipped refusal. Examples are wrong numbers, intervals that no claimed contract covers,
capability rows that do not match their calls, late refusals, and shipped outputs that no
registered study measures. The queue also holds a defect of the validation harness that can lose
or invalidate a registered study run. Deliver the rows in priority order, and complete every row
before the first beta release.

A new capability still needs its own contract and evidence, even in this queue.

| priority | item | tier | acceptance |
| ---: | --- | --- | --- |
| 0.1 | RM39. Keep a study run alive through a failed replicate | b | the contract in [RM39](#rm39-keep-a-study-run-alive-through-a-failed-replicate), its tests, and passing handoff checks |

### RM39. Keep a study run alive through a failed replicate

Status: parts a1 (resilience), a2 (failure recording and accounting) and b1 (the primary
Python phase) are delivered. Part b2 (the property phase) is open. The section leaves this file
when b2 merges.

A registered study with a comparator runs in two phases. The Python phase fits `cleverly`. The
reference phase fits the pinned comparator, usually R in its container. Before RM39, one failed
replicate stopped a phase, and a killed worker or container lost the whole phase. A run of several
hours was then lost to one draw.

Goal: one failed replicate costs one replicate. A killed worker costs the group in flight, a crash
resumes, and the summary accounts for each failure against a declared bound. The row takes tier b,
because its defect is a crash of a registered study run.

| term | meaning |
| --- | --- |
| replicate | one drawn sample and its fits |
| group | the unit that one fit call takes. A group can hold several scenarios |
| kill | a group with a start marker and no checkpoint after a pass. Its worker died |
| incomplete | a group with two kills. The run stops and names the group |
| checkpoint | the stored result of one group. In part a2, also the failure record of one group |
| resume key | the fields that two runs must share before they share checkpoints |
| failure | in part a2, an error inside the declared comparator-fit stage. The run records it and counts it against the bound |
| fatal error | any other error. It stops the phase |

#### Part a1: resilience

Every error stays fatal in part a1. No summary, published table or verdict changes.

| acceptance | status |
| --- | --- |
| each R fit runs in a forked child and writes a checkpoint. No fit runs in the parent R process | delivered |
| the run fits a killed group once more, alone, after it halves the workers. A second kill stops the run with exit 3 and names the group | delivered |
| a pass that makes no progress stops the run with exit 3 | delivered |
| a rerun resumes from the checkpoints and writes the bytes of an uninterrupted run | delivered |
| the resume key holds the declaration without its output path, `HEAD` and the tree state, the reference sources, the harness, the content of the samples and the truth, n, the replicate count and the Python versions | delivered |
| the worker count follows measured memory. Calibration fits the first group of each scenario in a fresh fork and measures its peak and its private memory | delivered |
| the driver runs a container that was killed whole once more, with half the workers. A second kill stops the run | delivered |
| a host lock and a container name refuse a second driver on one resume key | delivered |
| `ctmle3_oat`, `ctmle_selector` and `drtmle` stay off the harness. Each carries a byte-identical copy of its RM39 functions | delivered |
| each fit sets the seed that the driver declares for its replicate | delivered |
| no study is regenerated, because the edits are result-neutral. A runner that fails the determinism gate stops the row | delivered |

The [testing strategy](development/testing-strategy.md#resume-a-study-run) gives the commands, the
cache locations and the memory log. `tests/canonical/harness_fixture/determinism.csv` records the
determinism gate for each runner. `tests/canonical/provenance-revisions.md` declares the changed
hash of `study_harness.R` and of the three runners outside the harness.

#### Part a2: failure recording and accounting

| acceptance | status |
| --- | --- |
| a runner wraps its comparator-fit call, and only that call, in `study_stage`. An error inside the stage is a failure: the run records it and continues. Every other error is fatal | delivered |
| a failure record holds the group, its scenarios and replicates, the attempt, the class, the stage, the call, the message and the seed | delivered |
| when each of the first 20 groups fails, the run stops with the class `systematic` and the first message | delivered |
| `StudyRecord.failure_bound` declares a bound for each implementation before the run. The bound is between 0 and 0.05, and the default is 0 | delivered |
| `failures.csv` publishes each record, and `failure-accounting.csv` publishes the count and rate of each cell. Each file exists only when a failure exists. The published schema does not change | delivered |
| a cell is red when its failures exceed the bound times the replicate count, rounded down, or when a worst-case leg fails. The [red-cell rule](development/method-benchmarking.md#red-cells) routes it | delivered |
| a rate takes every split of the failed replicates. A bias statistic keeps its complete-case scale. A spread statistic also takes the survivor mean. A failed reference replicate in a non-inferiority leg takes the reference's best case | delivered |
| a paired statistic reads only the replicates where both implementations succeeded, and it reports the dropped replicates | delivered |
| an informative-failure witness reports the error of the surviving implementation on the failed replicates. It is a report, not the gate | delivered |
| a study hook defines `account_failures`, or its accounting row reads `failures not accountable` and is red | delivered |
| property cells, efficiency ratios, spread slopes, simultaneous coverage and the negative-control union read Python-phase rows only, so no a2 failure reaches them | moved to part b2 |

The design treats a failure conservatively. It does not rely on a test that failures are
independent of the outcome, because that test cannot see the missing outcome. The range-bounded
worst case can miss a failure whose value falls outside the observed range. A small declared bound
limits that case.

#### Part b1: the primary Python phase

| acceptance | status |
| --- | --- |
| `fit_replicates` gives every primary `draw_and_fit` loop the isolation, the records and the checkpoints of parts a1 and a2. With no failure, each study's Python phase writes the bytes it wrote before | delivered |
| these exceptions stop the phase: `CapabilityError`, `MethodConfigurationError`, `KeyboardInterrupt`, `SystemExit`, `MemoryError`, `TypeError`, `AttributeError`, `NameError`, `ImportError` and `AssertionError`. An error in the draw also stops it | delivered |
| the phase records only `DataError`, `LongitudinalError`, `numpy.linalg.LinAlgError`, `FloatingPointError`, `ZeroDivisionError`, `OverflowError`, and a plain `ValueError` or `RuntimeError` that `numpy`, `scipy`, `sklearn` or `statsmodels` raised | delivered |
| a subject row that breaks a row rule is an `invalid_result` record | delivered |
| the accounting fills a lost subject value at its worst case, counts each side against its own bound, and charges a `paired_input_missing` reference loss to the subject | delivered |
| the Python phase checkpoints each replicate under the part a1 resume key. A killed worker costs the replicates in flight, and a second kill of one replicate stops the run | delivered |
| `CoverageStudy` records the same classes and accepts a `ReplicationCheckpoint` | delivered |
| the zEpid runner records failures, stops on the systematic prefix and when every group failed, and resumes from the driver's stored reference result. It keeps one container | delivered |

#### Part b2: the property phase

| acceptance |
| --- |
| `fit_replicates` gives the property studies the records and the checkpoints of part b1 |
| property cells, efficiency ratios, spread slopes, simultaneous coverage and the negative-control union account for their lost replicates |

Each row takes a tier by the harm that its defect does to a user today. The table gives the tiers,
from the most harmful. Inside a tier, a row with a wider reach comes first. A row that another row
depends on comes before that row.

| tier | reason | rows |
| --- | --- | --- |
| a | a published number that is wrong, or that no derivation or read source covers. An anti-conservative number ranks above a conservative one | no open row |
| b | a crash, an exception that is not a refusal, a capability row that reads available and then raises, or an assessment that returns no report | RM39 |
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
| B5 | honest learned-policy evaluation and selected production extensions | a stable evaluation engine | later beta releases |

The first beta needs B0 to B3 and BP. It does not need every estimator to support every policy,
outcome, weighting, or dependence combination. It needs a usable path for each promised question,
and honest boundaries for the rest. The release process sets the beta version. This roadmap does
not change the package version.

| milestone | item | readiness | dependency | details |
| --- | --- | --- | --- | --- |
| B0 | Acceptance status and refusal reasons | no new theory | none | [RM42](#rm42-acceptance-status-and-refusal-reasons) |
| B0 | Review of the inherited hard stops | no new theory | the Eligibility rule | [RM43](#rm43-review-of-the-inherited-hard-stops) |
| B1 | Long-panel adapter and policy history | no new theory | the shipped trajectory representation | [X38](#x38-long-panel-adapter-and-policy-history) |
| B1 | Longitudinal evidence over three to five decisions | published support | the shipped `LTMLE` and `SDR` | [X39](#x39-longitudinal-evidence-over-three-to-five-decisions) |
| B1 | Targeting and fitting options | no new theory; each option keeps its default bit-identical | none | [X22](#x22-targeting-and-fitting-options) |
| B2 | Continuous vector components at a node | published support; pending source read | shipped longitudinal modified treatment policies | [X31](#x31-continuous-vector-components-at-a-node) |
| B2 | Continuous-policy validation and support checks | published support | shipped longitudinal modified treatment policies | [X40](#x40-continuous-policy-validation-and-support-checks) |
| B3 | Sequential g-computation and IPW baselines | published support; pending source read | the shipped `LTMLE` plan resolution | [X41](#x41-sequential-g-computation-and-ipw-baselines) |
| B3 | Applied tutorials for offers and pricing | no new theory | X38, X40, X41 | [X42](#x42-applied-tutorials-for-offers-and-pricing) |
| B4 | Categorical incremental policies | natural extension candidate; source audit | the shipped binary `IPSI` estimator | [X43](#x43-categorical-incremental-policies) |
| B4 | Repeated cross-fitting for longitudinal fits | published support; pending source read | the shipped point-treatment repeats and trajectory-level splits | [X46](#x46-repeated-cross-fitting-for-longitudinal-fits) |
| BP | Repeated-outcome curves | published support for the incremental case; a fixed-dimension stack for static and dynamic plans | the shipped `IPSI` estimator | [X32](#x32-repeated-outcome-curves) |
| BP | Adaptive-propensity IPTW | published support; pending source read | none | [X21](#x21-adaptive-propensity-iptw) |
| BP | C-TMLE candidate sequences | published support; pending source read | shipped selector paths | [X16](#x16-c-tmle-candidate-sequences) |
| BP | Hazard-based and monotone survival curves | published support; pending source read | the shipped point-treatment survival design | [X14](#x14-hazard-based-and-monotone-survival-curves) |
| BP | Plug-in omitted-variable limits with a declared parametric mechanism | [published support; stacked-equation extension](technical-reference/natural-extension-verdicts.md) | the shipped plug-in bound | [X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism) |
| B5 | Learned-policy follow-ups | published support for parts (g) and (h); published support; pending source read for parts (a), (b), (e) and (f); source audit for parts (c) and (d) | the shipped learned-rule value | [X11](#x11-learned-policy-follow-ups) |
| B5 | Structural dynamic treatment effects | source audit | X41 | [X44](#x44-structural-dynamic-treatment-effects) |
| B5 | Parametric forward-simulation g-formula | source audit | X41 | [X45](#x45-parametric-forward-simulation-g-formula) |
| after beta | Replicate-weight designs | source audit | weighted-law variance construction | [X2](#x2-replicate-weight-designs) |
| after beta | Efficient known-score ATT and ATC | published support; pending source read | the shipped known treatment mechanism | [X35](#x35-efficient-known-score-att-and-atc) |
| after beta | Natural and interventional mediation effects | published support; pending source read | target-specific identification and evidence | [X5](#x5-natural-and-interventional-mediation-effects) |
| after beta | Continuous-time survival and competing risks | published support; pending source read | continuous-time intensity and targeting contracts; X14 | [X6](#x6-continuous-time-survival-and-competing-risks) |
| after beta | Two-phase and outcome-dependent sampling | published support; pending source read | observed-data likelihood and influence correction | [X7](#x7-two-phase-and-outcome-dependent-sampling) |
| after beta | Omitted-variable bounds on the other linear functionals | published support; pending source read | the shipped arm-axis bound | [X9](#x9-omitted-variable-bounds-on-the-other-linear-functionals) |
| after beta | Continuous-dose MSM with a second mechanism | source audit | implemented continuous-dose MSM targeting, missing-outcome arm targeting, and controlled-direct-effect targeting | [X10](#x10-continuous-dose-msm-with-a-second-mechanism) |
| after beta | Fold-evaluated CV-TMLE with baseline strata | [published support; finite-partition extension](technical-reference/natural-extension-verdicts.md) | the shipped fold-evaluated estimate and the shipped stratified fluctuation | [X28](#x28-fold-evaluated-cv-tmle-with-baseline-strata) |
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

These items are hard stops for the claim that each row names. Move one into the main roadmap when
published theory supplies the missing result, or when the [Eligibility](#eligibility) rule
qualifies it as a natural extension. A missing comparator never keeps an item here, and a
comparator never removes a stop. [RM43](#rm43-review-of-the-inherited-hard-stops) reviews
each row under the current rule. Each row names
the current boundary, and the refusal that keeps it.

| investigation | missing result | current boundary | details |
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
| Plug-in omitted-variable confidence limits | an influence function or an inference result for the plug-in estimate of the Riesz second moment with an estimated treatment mechanism | the plug-in bounds, `rv`, `max_bias`, `benchmark()` and `contour()`. The one-sided limits and `rva` refuse for every mechanism learner except the declared parametric model that [X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism) implements | [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) |
| Targeted bootstrap inference | a construction that defines what is fixed, resampled, refitted, and retargeted, plus the sampling law of the interval | existing bootstrap inference is not this procedure | [F2](#f2-targeted-bootstrap-inference) |
| Longitudinal sensitivity-bound estimation | sample estimation of the bound functionals, a specialized algorithm, and sampling inference | no sensitivity bound on a longitudinal fit | [F16](#f16-longitudinal-sensitivity-bound-estimation) |
| Additional longitudinal estimands | target-specific identification, influence function, targeting construction, and inference conditions | existing end-of-study, survival, competing-risk, and MSM targets only | [F3](#f3-additional-longitudinal-estimands) |
| Other refused C-TMLE and DR-TMLE compositions | composition-specific score, reduced regressions, correction, remainder, and rate conditions. An observational missing outcome and a missing treatment on `DRTMLE` ship through the [composite indicator](technical-reference/dr-tmle/theorem.md#observational-missing-data-the-composite-indicator). On the composite, `evaluation=` and `reduced_crossfit="nested"` stay refused | named pre-fit refusals. `DRTMLE` refuses `NaturalCourseMean`, PAR and PAF with missing outcomes, because no DR-TMLE natural-course mean ships. A `DRTMLE` fit with a non-empty `guard` and estimated weights reports its point estimate under the `estimated_weight_plugin` status, and no interval | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| Selector-path C-TMLE inference | an influence function and covariance after the shipped data-adaptive stopping-index selection | point estimates and path diagnostics only. The greedy, ordered, and discrete paths refuse `ci`, `pvalue`, and `std_error`, except a `discrete` fit whose one declared candidate is the full adjustment set, and report a named working-mechanism plug-in diagnostic | [F18](#f18-selector-path-c-tmle-inference) |
| Outcome-adaptive C-TMLE generated-design inference | the generated-design term $c^\top \mathrm{IF}_\beta$ of the per-arm design, a vector result for the shared multinomial design, and uniformity for both designs | point estimates only. Every `strategy="oat"` fit, of either design and with or without baseline strata or missing data, refuses `ci`, `pvalue`, and `std_error` under the `generated_design_plugin` status, and reports a named generated-design plug-in diagnostic. The per-arm study measured that diagnostic 8 to 24 percent too small ([the applied revert](technical-reference/collaborative-tmle.md#the-applied-revert)) | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| Other missing-outcome CV-TMLE variants | a direct interval result for fold-specific targeting and for the fixed-repeat median and split-dispersion report after CV-TMLE targeting | the package supports the ordinary natural-course estimator, the stacked natural-course estimator for a binary outcome, the stacked arm-indexed means and contrasts, and their stacked PAR and PAF for a binary outcome. Each stacked estimator uses one repeat and pooled targeting. Shift, incremental, regime, MSM, and controlled-direct-effect targets refuse a cross-fitted fit with missing outcomes before any learner | [F21](#f21-other-missing-outcome-cv-tmle-variants) |
| Grouped cross-fitting beyond point-treatment TMLE | a split law and a cluster-robust variance for the C-TMLE selection folds and the candidate the search stops at | whole-cluster outer folds for cross-fitted point-treatment TMLE and DR-TMLE, at equal or unequal cluster sizes, with the [clustered interval rules](technical-reference/inference.md#clusters). Cross-fitted longitudinal TMLE draws whole-cluster folds too ([longitudinal clusters](technical-reference/longitudinal-tmle.md#clusters)). C-TMLE refuses `id=` | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| MNAR and incremental-intermediate compositions | identification and influence-function results for the exact compositions | point-treatment sensitivity and implemented interventions remain separate | [F6](#f6-mnar-and-incremental-intermediate-compositions) |
| Joint point-treatment parameter axes | a targeting and inference result for one fit that carries an MSM projection together with a regime, shift, or incremental intervention, including its joint score and covariance | single-axis point-treatment fits only | [F17](#f17-joint-point-treatment-parameter-axes) |
| Time-respecting cross-fitting | dependence and split-specific TMLE inference for blocked-temporal or rolling-origin folds | iid and grouped cross-fitting only | [F7](#f7-time-respecting-cross-fitting) |
| Learned-policy value outside the published conditions | an interval for the fold-average learned-rule value without a limiting rule, and results for the refused compositions | `LearnedRuleValue` fits the fold-evaluated CV-TMLE of [learned rules](technical-reference/point-treatment-tmle.md#learned-rules) under the limiting-rule condition. Its boundary study measures under-coverage at an exceptional law. The other F27 requests refuse before any learner | [F27](#f27-learned-policy-value-outside-the-published-conditions) |
| Inference at a boundary node estimate | an interval for a longitudinal TMLE parameter when a node regression reads a constant 0 or 1, such as a hazard estimated as zero at a grid node with no event, with its coverage | the point estimate stands. The parameter, and every contrast and restricted mean that reads it, take the `constant_node_plugin` status and report a plug-in diagnostic only. The band leaves them out | [F31](#f31-inference-at-a-boundary-node-estimate) |
| Finite-sample limits of clustered intervals | a small-sample correction or reference for the cluster-summed TMLE curve, at few clusters, at unequal cluster sizes, and for the fold-evaluated variance at few clusters per fold. An interval at 4 to 9 clusters (4 to 19 for `LTMLE`), which no registered study measures, and a fold-evaluated degrees-of-freedom rule at 40 clusters or more | `few_cluster_plugin` below 10 positive-mass clusters, and below 20 for `LTMLE`, the smallest counts the registered study measures. $t_{J-2}$ from 10 (20 for `LTMLE`) to 39. Fold-evaluated fits need 2 clusters per fold. The registered evidence uses parametric nuisance learners | [F28](#f28-finite-sample-limits-of-clustered-intervals) |

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
claims it covers. [RM42](#rm42-acceptance-status-and-refusal-reasons) part (e) adds the
comparator-backed label to the evidence pages.

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
not exist in the package yet. [RM42](#rm42-acceptance-status-and-refusal-reasons) owns it.

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

The `X4-finite-sample` owner holds one coverage cell of the SDR study at n = 2,000. `LTMLE`
shares its shortfall on the same draws, so the diagnosis finds no defect in `SDR`.

The `X8-logit-small-stratum` and `X8-identity-small-stratum` owners hold cells of the two
stratified MSM studies in their smallest stratum. Each is a standing record of a reading, and
neither gates the beta release. The `X8-drtmle-one-sided-bias` and `X8-drtmle-small-stratum`
owners do the same for the stratified DR-TMLE study.

| id | work | acceptance |
| --- | --- | --- |
| `F18` | an inference result for the shipped selector path | an influence curve derived after the stopping-index selection, and a registered study whose `selector_necessity` and `type_i_error` cells pass their existing margins at their existing budgets. The multi-arm `interval_calibration/correctly_specified` cell must also pass its band, because its standard-error ratio interval, 0.8987 to 0.9862, measures the fixed-candidate covariance that F18 replaces |
| `F19` | an inference result for the generated design, the red cells of `ctmle-oat-per-arm`, and the short-SE truth rows and ladder rungs of `ctmle-oat-per-arm-strata-missing` | a published result, or an Eligibility-qualifying natural extension, that establishes whether the ordinary adaptive-propensity curve suffices for the cross-fitted shared-multinomial joint target. For the per-arm design, a curve that adds the generated-design term $c^\top \mathrm{IF}_\beta$, with $c = E[(1 - g_0 / G)\, \partial \bar Q / \partial \beta]$, or a source that removes it. Implement any contribution the result requires, then register covariance and coverage evidence at declared budgets. The per-arm revert cells must pass their declared bands under a fresh declaration. The `generated_design` pair stays a reporting diagnostic, and acceptance does not require a negative standard-error deficit |
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
| `F28` | finite-sample limits of clustered intervals: the fold-evaluated covariate pair of `clustered-unequal-cvtmle`, and the fold-evaluated bias cells and cross-fitted `DRTMLE` IID controls of `clustered-few-cluster-tmle` | each group closes on its own registered study that passes. The fold-evaluated pair of `clustered-unequal-cvtmle` at 40 clusters in 10 folds: a fold-evaluated degrees-of-freedom rule $t(\min(J-2, J-V))$ at 40 clusters or more. The three `tmle_cv_evaluation__unequal_informative` cells: a cluster-size-weighted fold average or another fold-evaluated point whose bias stays inside the margin. The stacked report is the remedy today. The four `drtmle_crossfit` `iid_t_control` cells: a calibrated cross-fitted `DRTMLE` cluster variance, or an IID control that compares the two SEs and not the IID SE with the empirical SD ([F28](#f28-finite-sample-limits-of-clustered-intervals)) |
| `X20-bootstrap` | the `interval_calibration` cells `boot_ey_crossfit__correctly_specified`, `boot_ate_crossfit__correctly_specified` and `boot_ate_clustered__correctly_specified` of `full-refit-bootstrap-and-derived-contrasts` | the declared rule 3 keeps the `end_of_study/cross_fit` and `end_of_study/cluster` kinds out of `LICENSED_BOOTSTRAP_DESIGNS`, so their bootstrap prints as a diagnostic. A diagnostic reading, not a registered result: on this cell the cross-fitted estimator is heavy-tailed. Its 4,000 committed estimates have excess kurtosis 6.7, and about 5% of them lie at the probability boundary, because `g_bounds=(1e-8, 1)` with saturated cell means leaves sparse training cells unbounded. A bootstrap resample holds about 63% unique units, so more replicates reach the boundary and the replicate spread exceeds the sampling spread (SE ratios 1.23 and 1.27). A probe that kept each unit's copies in one fold raised the SE ratio to about 1.65, so the split of copies across folds is not the cause. The cluster contrast covers 0.924 with a calibrated SE ratio (0.977), a finite-sample shortfall of the percentile interval at 60 clusters. The owner closes when a re-declared cell for the kind, with its law, size, nuisance bounds or resampling scheme declared before its run, passes a registered cell |

## Detailed implementation contracts

The sections below give one contract for each open item. Their physical order does not override
the main grid.

### RM42. Acceptance status and refusal reasons

The [Eligibility](#eligibility) rule classifies each analysis as supported, experimental, or
unavailable. The package has no `experimental` status today. Its refusals are typed errors with
prose reasons and no structured reason. This row implements both contracts.

| part | what it delivers |
| --- | --- |
| (a) the experimental status | an explicit opt-in for a configuration whose target is justified but whose algorithm or inference extension has unresolved evidence. It is a new record in `cleverly._inference_status` beside the `*_plugin` statuses, not a second status field. Each estimate and interval carries the label, and an unsupported interval is withheld |
| (b) refusal reasons | a reason code on each refusal, drawn from the table below, with the failed condition and an available alternative in the message |
| (c) configuration support rows | a support row for each combination of target, estimator, outcome, weighting, splitting, and inference, whose status matches the output |
| (d) persistence | the status, the reason, the policy definitions, the folds, and the inference assumptions survive `result.save()` and `cleverly.load()` |
| (e) comparator-backed rows | the comparator-backed label on each evidence page claim that rests on a comparison, and a validation-grid row for a comparator-backed study with no property cells. Comparator-free studies already carry rows with a zero-row equivalence artifact |

| reason | meaning | handling |
| --- | --- | --- |
| invalid declaration or data | the requested analysis cannot run on the declared sample or design | keep the validation, and state the failed condition |
| different estimand | the requested option changes the causal question | name both questions, and never substitute one silently |
| implementation gap | a valid method has no implementation yet | name the roadmap item and the available alternatives |
| inference gap | the estimate exists, but the requested uncertainty calculation is incomplete | offer the supported inference, or explicit experimental estimation where defensible |
| API redirection | the feature exists through another argument or object | point to a working call. This reason does not count as a missing capability |

Part (e) keeps the checks of the target, of each changed mathematical term, and of each claimed
statistical property. It changes no committed study result.

Each inherited refusal keeps its intervention-specific support checks, the estimator-specific
requirements for policy followers, the invalid-event checks, and the identification declarations.
A test can detect an observed support failure and check a declared support restriction. It cannot
establish population positivity or the absence of unmeasured confounding. A new refusal does not
call a stochastic or continuous policy an inappropriate question.

Acceptance:

- a pre-fit test for each reason code, which pins the code, the message, and the alternative;
- a test that an experimental fit carries its label through the summary, the tables, and a
  save-and-load round trip, and withholds an unsupported interval;
- a test that each support row matches the status that the fit reports.

### RM43. Review of the inherited hard stops

The [future investigations](#future-investigations) grid holds items that an earlier rule required
a published derivation for. The [Eligibility](#eligibility) rule now admits three acceptance routes,
and it never treats a missing comparator as a gap. Review each grid row under those routes. Record
for each row the base result, each step, and the verdict:

| verdict | action |
| --- | --- |
| established method or natural extension | move the row into the main roadmap, with a contract and its evidence |
| established theory or a natural extension, with a comparison that covers part of the evidence | move the row, name the comparison, and name the claims that need independent checks |
| new identification, regularity, or asymptotic argument needed | keep the row, and scope its stop to the unsupported method, configuration, or inference claim |

Do not treat an API gap, a finite-category adaptation, or a standard projection as fundamental
research by default. Keep a research queue for new identification, interference,
adaptive-optimum inference, and arbitrary-dependence results. Each move updates the refusal that
the row names. The owner's plan places F13 and F16 in later beta work. Each lacks a time-indexed
latent law today, so they stay in the grid until this review reads them.

Acceptance: a written verdict for every grid row, and a pre-fit test for each refusal that changes.

### X38. Long-panel adapter and policy history

`LongitudinalData.from_frame` reads one wide row per unit. Customer data usually arrives as one row
per customer and decision time. This row adds a validated long-panel adapter, and it fixes the
history that each policy can read.

1. Accept a long panel with customer ID, decision time, covariate, treatment, censoring, and
   outcome roles.
2. Convert the panel into the existing trajectory representation. Preserve temporal order and the
   original treatment labels.
3. Map the panel's columns onto the shipped history frames. A rule reads `history_frame`, and a
   policy density reads `policy_frame`, which adds the earlier treatments.
4. Refuse a panel column that would let a rule read future values or final outcomes.
5. Carry the policy definitions through result persistence.

One customer trajectory is one observation for the inference model. Repeated time rows within that
trajectory do not add independent observations. Customers who depend on each other within one
account or store use the shipped cluster contract with `id=`. Calendar validation and general
time-series dependence stay with [F7](#f7-time-respecting-cross-fitting).

Acceptance:

- a round-trip test from a long panel to the trajectory representation and back, with irregular
  censoring and unsorted input;
- refusal tests for duplicate rows, a gap in decision time, and a rule that reads the future;
- equality of an `LTMLE` fit on the adapted panel and on the equivalent wide frame.

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

### X41. Sequential g-computation and IPW baselines

`cleverly` has no longitudinal plug-in or inverse-probability-weighted estimator. This row adds
sequential g-computation and IPW over the targets that `LTMLE` supports: static, dynamic, and
known stochastic policies. They let a user compare model reliance and weight instability, and they
verify policy semantics independently.

Declare the IPW normalization, the treatment and censoring weights, and the uncertainty from
nuisance estimation. An interval needs parametric nuisance models, with the stacked estimating
equation or the bootstrap. With a data-adaptive learner, each baseline reports a point estimate
only, under a status that withholds the interval.

Do not copy the TMLE uncertainty into either
baseline without justification.
Do not describe longitudinal g-computation as one regression on every treatment and confounder.
`stremr` fits static, dynamic, and stochastic regimes with categorical exposures from long-form
input ([comparator survey](development/method-benchmarking.md#choose-the-comparator-before-you-write-the-runner)).
After X38, use it for the overlapping cases, and audit its category-valued policy support before
a multi-arm parity claim. For finite-category extensions, combine the
established stochastic intervention argument with exact enumeration and independent influence
function checks.

Acceptance:

- exact-law witnesses for both baselines, including a three-arm witness with nonzero mass on the
  third arm;
- a deterministic-policy reduction to the shipped regimen targets;
- a registered study with a declared interval rule for each baseline, or a points-only status
  where no interval contract applies.

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

### X14. Hazard-based and monotone survival curves

The shipped survival fit targets each horizon in its own backward pass, so the reported curve is
not constrained to be monotone. The
[survival-curve evidence page](technical-reference/method-evidence/ordinary-survival-curve-longitudinal-tmle.md)
states this limit.

| part | what to add | published source | comparator |
| --- | --- | --- | --- |
| (a) hazard-based targeting | targeting of the discrete hazard at every horizon, so the curve is a product of targeted hazards | Benkeser, Carone and Gilbert (2018), *Statistics in Medicine* 37(2) | `survtmle` `method = "hazard"` |
| (b) one-step whole-curve targeting | one targeting step that solves the score equations of every horizon together, with a simultaneous band over the curve | Cai and van der Laan (2020), *Biometrics* 76(3), the one-step survival paper. It is a different paper from their HAL bootstrap paper | MOSS |

Both parts need a pinned commit of their comparator, because `survtmle` and MOSS are archived on
CRAN. Part (b) supplies a one-step band over the whole curve. `tmle3` `tmle_survival` at the
pinned `ed72f8a` is a second part (b) candidate. It takes constrained hazard steps, a binary
treatment, and the grid $1, \dots, \max \tilde T$. A pairing must give it the grid of the
`cleverly` fit.

Both parts start from the
[held point-treatment design](technical-reference/longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes) and from `TimeToEvent`, which carry the per-node
indicators, the masks, the censoring and the grid. The `survtmle` runner of the
`point-treatment-survival` study takes `method=`, so part (a) needs no new image. The
[survival-curve study](technical-reference/method-evidence/ordinary-survival-curve-longitudinal-tmle.md)
measures the band of the shipped per-horizon curve.

Acceptance:

- exact-law witnesses for each targeting construction;
- a test that each curve is monotone on a law where per-horizon targeting is not;
- a registered study with joint coverage over the horizons.

### X32. Repeated-outcome curves

`LongitudinalData` reads `outcome=[...]` only as an absorbing event. X32 adds a declaration of a
non-absorbing outcome $Y_t$ at each node. It also adds the curve $t \mapsto E[Y_t^{Q}]$ over a
declared set of horizons.

| item | contract |
| --- | --- |
| base | Kim, Kennedy and Naimi (2021), *Journal of Causal Inference* 9(1):302–344, Theorems 3.1, 4.1 and 5.1 (tilts, dropout, and the joint process over $\delta$ and $t$). For static and dynamic plans, each horizon is the shipped end-of-study estimand on the panel truncated after node $t$. The curve is their fixed-dimension stack |
| estimators | one backward pass per horizon, seeded at $Y_t$, with earlier $Y_s$ in the history. `IPSI`, `LTMLE` and `SDR` each fit the declaration or refuse it by name |
| inference | a joint band over plans and horizons from the stacked curve |

Today a user can fit one horizon by hand. Truncate the panel after node $t$, and declare $Y_t$ as
the outcome and earlier $Y_s$ as covariates.
[Incremental interventions over time](technical-reference/longitudinal-tmle.md#the-incremental-estimand)
states that route.

Acceptance:

- exact-law witnesses with a non-absorbing outcome that returns to zero, so an absorbing reading
  fails;
- a mutation control that treats $Y_t$ as absorbing;
- a truth-calibrated registered study. Kim's research code is unpinned, so no paired comparator
  exists.

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

At a continuous modified-treatment-policy node of `LTMLE`, the ratio that part (d) trims is the
node's density ratio, which the fit stores as `RegimenFit.node_ratio`.

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
| learner rule | the bound reads the fit's own $\hat g$. So the stack is valid only when the fit's own treatment learner is the declared unpenalized logistic or multinomial model. A Super Learner whose library contains that model is refused |

Acceptance:

- a mechanism-score interface for an unpenalized `LogisticRegression`, binary and multinomial;
- an exact-law Jacobian witness, and a mutation control that drops the $\gamma$ term and fails;
- the four limits open only under the learner rule, with a refusal test for a Super Learner that
  contains the model;
- a registered coverage study.

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

### X6. Continuous-time survival and competing risks

The shipped survival and competing-risk estimators use discrete time nodes. Rytgaard, Gerds and van
der Laan (2022) derive the continuous-time construction, which changes the intensity model, the
targeting step, and the remainder. The `concrete` package implements the one-step form of
Rytgaard, Eriksson and van der Laan (2023). Its last release is 1.0.5, and CRAN archived it in
2024, so pin it by commit before it becomes a comparator.

That comparator takes a binary baseline treatment under a static or dynamic intervention, which
bounds the paired cells a first study can claim. A discrete-time study is not evidence for a
continuous-time interval, so the existing longitudinal rows do not transfer. The whole-curve
targeting of [X14](#x14-hazard-based-and-monotone-survival-curves) comes first. On a discrete
grid, a `concrete` intervention on the baseline treatment is a
[held point-treatment design](technical-reference/longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes) with a plan at node 1. A stochastic intervention is a
[known stochastic policy](technical-reference/longitudinal-tmle.md#known-stochastic-policies) at
that node.

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

Each contract below states the missing published result, the boundary that ships, and the
acceptance that would move the item into the main grid.

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
than abandoning the run. `docs/examples/interventions.ipynb` runs a treatment-only grid and
states this reason. The generated-outcome refutation refuses the same composition for the same reason, in
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

The stacked estimating equation for a declared unpenalized logistic or multinomial mechanism moved
to [X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism). This item keeps
every other mechanism learner, a Super Learner that contains that model included. The bound reads
the fit's own $\hat g$, so only the fit's own learner can supply the stacked score.

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
assumptions, influence functions, targeting construction, and inference conditions. For the
incremental case of a non-absorbing time-varying outcome, Kim, Kennedy and Naimi (2021), Theorems
3.1, 4.1 and 5.1, supply the result. [X32](#x32-repeated-outcome-curves) owns that curve. Add accepted
targets in both directions to the oracle registry and evidence gates rather than treating them as
options on an existing cause-specific estimand.

### F7. Time-respecting cross-fitting

Blocked-temporal and rolling-origin folds wait for a published TMLE result whose dependence
assumptions match the supported data. The result must specify which rows may train each prediction
and which asymptotic argument licenses the interval. Ordered indices passed through iid fold
machinery are not sufficient.

### F18. Selector-path C-TMLE inference

The greedy, ordered and discrete paths report point estimates and path diagnostics. They refuse
`ci`, `pvalue` and `std_error` under the `working_mechanism_plugin` status. They report the
ordinary curve at the selected candidate as `plugin_std_error` and `plugin_interval`. That curve
does not establish conditional or unconditional coverage after selection. F18 reopens inference
when a published result supplies the influence function and covariance after the shipped
selection.

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
result must settle this fixed-candidate case before it addresses selection.

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
| van der Laan et al. (2026), [arXiv:2501.11868v3](https://arxiv.org/html/2501.11868v3), Section 5.2, Appendix C, Theorem 5 and Corollary 1; adaptive debiased machine learning v2 | model-based inference after selection under a linear expansion, curve stabilization and approximation rates | a proof that one global depth from nested targeted-loss folds meets those conditions |
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
$c = E[(1 - g_0 / G)\, \partial \bar Q / \partial \beta]$. The registered study
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

**Why the shared design is not a natural extension.** It fits one shared multinomial
mechanism on `K` estimated columns and uses a `K`-column joint fluctuation. Cramér--Wold turns
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
| the per-arm generated-design term | an estimator of $c^\top \mathrm{IF}_\beta$, or of its analogue for a flexible outcome learner, that does not need a consistent full mechanism, or a source that shows the term vanishes under a stated condition |
| one shared multinomial | one categorical fit on `K` estimated columns supplies every arm's clever covariate, so an inconsistent column for one arm enters every other arm's mechanism |
| vector target and simultaneous inference | the joint covariance and the simultaneous critical value, not the per-arm variance alone |
| uniformity, for both designs | each estimator is superefficient by design, so a pointwise limit law does not give locally uniform coverage. This item covers the per-arm design too |
| transport of the shared design beyond the source law | missing outcomes need the response-mechanism expansion. Fixed weights need a weighted empirical-law result, and estimated weights a first-stage contribution. Clusters and strata need dependence- and stratum-specific expansions. Repeats need the median and split-dispersion aggregation |

**Acceptance.**

- A result covers the exact cross-fitted multi-arm construction, and it separates a fixed target
  from a target conditional on the learned design.
- For the per-arm design, the revert cells of `ctmle-oat-per-arm` pass their declared bands under
  a fresh declaration, and the flag `OAT_PER_ARM_INFERENTIAL` returns to `True`.
- It establishes whether the current curve suffices or an added representation contribution is
  needed.
- It covers the requested joint means or contrasts and their covariance. It states the remainder,
  the nuisance-rate conditions and a uniformity claim.
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

Keep fold-targeted and repeated-split missing-outcome fits refused for estimators that report an
interval. The natural-course mean, PAR, PAF, and the arm-indexed means and contrasts refuse both
variants before any learner call. No read source gives a direct interval result for either
composition. The per-arm outcome-adaptive `CTMLE` reports point estimates only, and it admits
`repeats > 1` with missing outcomes.

A fold-targeted extension needs a theorem for one response-weighted fluctuation coefficient in
each validation fold and for the resulting stitched influence curve. zEpid code corroborates that
control flow for a related estimator, but code is not an inference result.

The fixed-repeat result in Chernozhukov et al. (2018) applies to the paper's DML estimators under
its DML assumptions. No read result transports its coordinatewise median and split-dispersion
variance to the targeted MAR plug-in. A future result must cover the dependence created by
targeting before it can justify the package's repeated report for this target.

A cross-fitted `DRTMLE` fit with `delta=`, and any cross-fitted fit with a missing treatment, stay
refused at every target. Each needs a fold-local composite mechanism and a
direct interval result for it.

Shift, incremental, regime, MSM and controlled-direct-effect targets, cross-fitted with missing
outcomes, raise `CapabilityError` before any learner. The message names F21, and its remedy is the
in-sample fit. A continuous-dose MSM meets the
[X10](#x10-continuous-dose-msm-with-a-second-mechanism) refusal first, which names no remedy. Each
target needs its own source audit and contract.

A C-TMLE selector fit takes
`working_mechanism_plugin`, which [F18](#f18-selector-path-c-tmle-inference) holds. Every
`strategy="oat"` fit takes `generated_design_plugin`, which
[F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) holds. The per-arm design ships
point estimates with baseline strata and missing data under
[its contract](technical-reference/collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design).

The stacked contracts have four follow-ups outside this hard stop. The
[natural-course contract](technical-reference/cv-tmle.md#missing-outcome-natural-course-mean) and
the [arm-indexed contract](technical-reference/cv-tmle.md#missing-outcome-arm-indexed-means-and-contrasts)
refuse each one today. Each needs its own contract and registered evidence. Do not use the evidence
of one extension for another.

| follow-up | missing work |
| --- | --- |
| fold-evaluated construction, for the natural-course mean and the arm-indexed means and contrasts | it has published support in Zheng and van der Laan (2011), Sections 2 and 2.1; it needs an implementation review of its fold plug-in and variance law, which define a separate estimator |
| supplied split plans | an audit of their balance and weighting requirements. Every cross-fitted fit refuses a plan that carries no package generator record, which the [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) state |
| bounded-continuous stacked natural-course mean | an exact contract for scaling the fluctuation, score, point, and influence curve |
| stacked arm-indexed targets, PAR, and PAF with baseline strata | the finite-partition step that the stacked natural-course mean ships, with a witness and a registered cell. The scalar stacked `ey_obs` admits `strata=`. The per-arm `CTMLE` ships these targets with strata for point estimates, outside this contract |

A per-arm outcome-adaptive `CTMLE` fit with missing outcomes and `repeats > 1` reaches the
multi-draw branch of `missingness_tilt` (`src/cleverly/sensitivity/missingness.py`). The test
`test_the_tilt_reads_the_full_covariate_model[3]` in
`tests/unit/test_outcome_adaptive_per_arm_extensions.py` is its witness: the tilt equals the
median over draws of a per-draw longhand. A repeated-split contract for interval estimators needs
its own witness.

### F22. Grouped cross-fitting beyond point-treatment TMLE

The package draws whole-cluster outer folds for the ordinary cross-fitted point-treatment TMLE,
DR-TMLE and longitudinal TMLE. C-TMLE refuses `id=` at every `cross_fit` setting, and this item holds that refusal.

| composition | what ships | what a result must supply |
| --- | --- | --- |
| C-TMLE with `id=` | refused at every `cross_fit` setting (`CTMLE._resolve_estimands_for_data`) | a split law for the selection folds and the nested selection folds under clustering, and the cluster-robust variance of the candidate the search stops at. [F18](#f18-selector-path-c-tmle-inference) is open for iid rows, so a clustered result needs that one first |

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

Continue pre-fit refusals for ATT, ATC, PAR, PAF, regimes, incremental interventions, shifts, MSMs,
mediation, and a missing treatment on the C-TMLE selector strategies and the shared
outcome-adaptive design, where each variant lacks evidence. Ordinary-TMLE
implementations do not establish the C-TMLE or DR-TMLE corrections for these compositions.
An observational missing outcome and a missing treatment on `DRTMLE` ship through the
[composite indicator](technical-reference/dr-tmle/theorem.md#observational-missing-data-the-composite-indicator).
On the composite, `evaluation=` and `reduced_crossfit="nested"` stay refused here. Both carry one
fold-free treatment mechanism, and the composite has up to three factors. A composite `DRTMLE`
fits `strata=` with one block per stratum, as every `DRTMLE` route does
([baseline strata](technical-reference/dr-tmle/supported-estimands.md#baseline-strata)).

The C-TMLE selector strategies and the shared outcome-adaptive design refuse cross-fitted
arm-indexed missing outcomes, and PAR and PAF with missing outcomes, before any learner call.
In-sample C-TMLE with missing outcomes still fits, and it reports no interval: a selector path
takes `working_mechanism_plugin`, and `strategy="oat"` takes `generated_design_plugin`. The
per-arm outcome-adaptive design ships point estimates with baseline strata, missing outcomes and,
in sample, a missing treatment, under
[its contract](technical-reference/collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design).

Each extension of a C-TMLE selector strategy or of the shared design needs its target-specific
collaborative score and selection-risk contract.
Each DR-TMLE extension needs reduced regressions, a correction, a remainder, and rate conditions.
PAR and PAF also need the joint observed-mean curve and covariance. Complete simulated-confounding
replay receives its own audit only after the estimator can fit the target.

`DRTMLE` refuses `NaturalCourseMean`, PAR and PAF with missing outcomes. The stack that ordinary
TMLE ships ([PAR and PAF with missing outcomes](technical-reference/point-treatment-tmle.md#par-and-paf-with-missing-outcomes))
needs a DR-TMLE natural-course mean, and none exists.

The composite construction is per arm: it evaluates the regression and the mechanism with the
arm set on every row. The natural-course mean needs them at the realised arm. Benkeser et al.
(2017), Theorem 1, applied with $\Delta$ as the treatment and $(A, W)$ as the covariates, supplies
the parent. The work has four steps.

| step | work |
| --- | --- |
| 1 | a realised-arm composite view in `estimators/composite.py`, with the indicator $\Delta$ and the covariates $(A, W)$, and reduced regressions over $\bar m(A, W)$ and $\pi(A, W)$ |
| 2 | the DR-TMLE natural-course mean `ey_obs` at each guard and reduction, with Gateaux and remainder witnesses for the corrected curve $D^*-D^*_Q-D^*_g$, and its post-fit readers (correction check, truncation, retarget, nuisance diagnostics) |
| 3 | PAR and PAF through the shipped `TargetContext.natural_course` hook, with no second spelling |
| 4 | a registered study with one-wrong and both-wrong coverage cells, paired with `drtmle` 1.1.2 through the constant-treatment form (`A = rep(1, n)`, `W = cbind(A, W)`, `DeltaY = !is.na(Y)`), after a probe confirms that `drtmle` accepts a one-level `A`. Its one-wrong cells declare their reading from the [case table](technical-reference/dr-tmle/theorem.md#the-conditioning-terms-are-linear-in-one-error) and use a law where the conditioning term does not vanish, or a known mechanism |

Omitted-variable bounds on a DR-TMLE or C-TMLE fit are refused. A bound on either fit needs an
estimate of $\nu^2$ that stays valid when the estimator does not assume a consistent treatment
mechanism. It also needs the influence curve of the bound under that estimator. A separately
fitted full mechanism is a new estimator of the bound, and it needs the same derivation.

A `DRTMLE` fit with a non-empty `guard` and varying weights declared estimated
(`weights_estimated=True`) reports its point estimate under the `estimated_weight_plugin` status,
and `ci`, `pvalue` and `std_error` refuse. The flag changes no number, so the status is not a
refusal that a caller could bypass by dropping the flag. `guard=()` fits the ordinary TMLE and
keeps its interval. So do constant weights, which fit the unweighted estimator. The interval
reopens with the influence contribution of the weight estimate to the reduced regressions. This
gap is separate from F11, which tracks weight-model replay after a perturbation.

### F17. Joint point-treatment parameter axes

One ordinary point-treatment fit carries one parameter axis. A working model summarises the
counterfactual means with one score equation per term. A known regime, a modified treatment policy,
or an incremental intervention replaces what those means are. One fluctuation cannot solve both
sets of score equations. The `TMLE` constructor refuses two intervention keywords together, or one
of them with `msm=`. `CausalStudy.identify` refuses a typed estimand whose set holds two kinds.
Both refusals come before any model is
fitted.

Wait for a published targeting and inference result for each proposed composition, including its
joint score and covariance. Do not infer the construction from the existing single-axis
implementations. An accepted composition must reduce exactly to each standalone fit, preserve
parameter names and policy definitions, and expose the full cross-axis influence covariance.
Register nonzero controls for every cross-axis block, and repeated-sampling evidence for
simultaneous inference if it is claimed.

### F6. MNAR and incremental-intermediate compositions

An MNAR tilt for continuous-dose shifts, and intermediate variables with incremental
interventions, wait for identification and influence-function results covering those exact
compositions. The incremental-intermediate refusal raises `CapabilityError` before any learner.
Both tilt rows of a shift fit read `unavailable`, with the sentence that the call raises.

### F27. Learned-policy value outside the published conditions

The [learned-rule contract](technical-reference/point-treatment-tmle.md#learned-rules) refuses each request below before any
learner, with three exceptions. The
[boundary study](technical-reference/method-evidence/learned-rule-cvtmle-boundary.md) measures
the exceptional-law row and the `weak_blip` row, and the assessment row reads `unavailable`.
F27 owns every red cell of the boundary study. The table states, row by row, whether a
published result is missing.

| request | missing published result |
| --- | --- |
| an interval at an exceptional law, where the limiting-rule condition C3 of that contract fails | a CV-TMLE interval for the fold-average target without a limiting rule. Luedtke and van der Laan (2016), *Annals of Statistics*, Section 4.2, name inverse weighting by the standard deviation and a central limit theorem for triangular arrays. Section 5 applies them to the optimal value. No reviewed source applies those tools to this target |
| an interval at the `weak_blip` law of the boundary study | no missing result. That law is within C3, and Theorem 6 of van der Laan and Luedtke (2015) covers it. A red cell there is a finite-sample limit at n = 2,000, and F27 holds it as the owner of the reporting study |
| `repeats` above 1 | each split defines a different target, and no source aggregates over targets |
| the full-refit bootstrap | each resample relearns the rules, so each draw has a different target |
| a continuous treatment | no reviewed source defines a learned rule over a dose for this target |
| `intermediate=` | no reviewed source gives a learned rule for a controlled direct effect |
| `weights=` or `id=` | van der Laan and Luedtke (2015) treat unweighted iid rows, and no source gives a grouped-split learned-rule result |
| `strata=` | no reviewed source gives a stratified learned-rule fluctuation |
| `CTMLE` or `DRTMLE` with `learned_rule=` | no collaborative or doubly robust learned-rule result was reviewed |
| the E-value, the omitted-variable bound, simulated confounding and the missingness tilt | no derivation for this target was reviewed |

### F28. Finite-sample limits of clustered intervals

A clustered `TMLE` or `DRTMLE` fit reports a Student $t$ interval with $J-2$ degrees of freedom
when it reads 10 to 39 clusters with positive weight mass, and an `LTMLE` fit, in sample or
cross-fitted, when it reads 20 to 39 ([clusters](technical-reference/inference.md#clusters)). Below that floor, in the
fit or in one reported baseline stratum, the fit takes the `"few_cluster_plugin"` status. Each
floor is the smallest count the registered few-cluster study measures for that fit. F28 owns 4
to 9 clusters, and 4 to 19 for `LTMLE`. Below 4 the reference would have fewer than 2 degrees of
freedom.

| request | missing published result |
| --- | --- |
| an interval below 4 clusters | a supported construction and registered evidence. The current package supplies neither. A $t_1$ reference has finite quantiles despite its undefined mean, so that mean does not rule out a future interval |
| a small-sample reference other than $t_{J-2}$ | a correction or reference derived for the cluster-summed TMLE curve. Nugent et al. (2024), Section 2.2, and Benitez et al. (2023), Sections 3.1.2 and 3.2.1, state $t_{J-2}$ for cluster-randomized trials. No source derives it for a row-level treatment |
| an interval at 4 to 9 clusters, or at 4 to 19 for `LTMLE` | a registered measurement, which would reopen these fits. The registered few-cluster study starts at 10 clusters, and its in-sample `LTMLE` cells at 20. At 4 clusters a pre-run probe found 0.25% to 9% of draws on which a fit cannot run, because the outcome does not vary among one plan's followers or a training fold lacks an outcome class. At 10 clusters the first full run lost 1 `LTMLE` fit in 4,000 in each size law the same way |
| a fold-evaluated interval at 40 clusters or more | a fold-evaluated degrees-of-freedom rule $t(\min(J-2, J-V))$ at 40 clusters or more, with its own registered study that passes. The unequal-size study's fold-evaluated pair at 40 clusters in 10 folds reads coverage 0.9325, with lower endpoint 0.918 against the 0.92 floor, on the normal reference that the rule keeps there |
| a cross-fitted `LTMLE` interval at 4 to 19 clusters | a registered measurement. No cross-fitted study measures 4 to 19 clusters. `few-cluster-cross-fitted-ltmle` measures 20 and 30: its declared failure-only probe of 2,000 draws found 3 and 8 draws at 10 clusters that admit no fit, so the 10-cluster cells were dropped before any run |
| the fold-evaluated variance at few clusters per fold | a calibration result for the centered fold cluster variance. The package requires 2 clusters per fold and gives the report $\min(J-2, J-V)$ degrees of freedom, the pooled within-fold count. The few-cluster study measures it at 2, 4 and 6 clusters per fold, beside a reported $J-2$ arm |
| a fold-evaluated point at informative unequal sizes with few clusters per fold | a fold-evaluated point construction whose bias at 2 to 6 clusters per fold stays inside 0.25 empirical standard deviations, such as a cluster-size-weighted fold average whose fluctuation uses the same weights, with its own registered study. The stacked report is the remedy today. The few-cluster study reads a bias of 0.61, 0.43 and 0.32 at 2, 4 and 6 clusters per fold, where the stacked fit reads 0.14, 0.10 and 0.06. The equal $1/V$ fold average causes it (`tests/unit/test_fold_evaluated_ratio_bias.py`) |
| a calibrated cross-fitted `DRTMLE` cluster variance at 10 to 30 clusters | a cluster variance for the cross-fitted `DRTMLE` curve whose SE ratio stays inside the calibration band, or a redesigned IID control, with a registered study. The few-cluster study reads cluster-robust SE ratios of 1.19, 1.14 and 1.12 at 10, 20 and 30 equal clusters. The IID SE is 23% to 28% below the cluster-robust SE, and the IID SE ratio upper endpoint reaches the 0.80 ceiling at four cells |
| complex nuisance learners at about 20 clusters | none for parametric learners. Wang, Park, Small and Li (2024), Remark 3, caution against complex working models at $m = 20$. The registered few-cluster evidence uses parametric nuisance learners only |

### F31. Inference at a boundary node estimate

A longitudinal fit reads a node regression whose fitted rows all hold 0, or all hold 1, as that
value ([a node with no event](technical-reference/longitudinal-tmle.md#a-node-with-no-event)).
The point estimate is the maximum-likelihood value, and `survtmle` reads such a node the same
way. The node's term of the influence curve is then identically zero, although the true hazard
there can be positive. So the plug-in standard error leaves that node's variance out. A risk of
zero gets a standard error of zero, and a contrast against it carries the other arm's variance
only.

| request | missing published result |
| --- | --- |
| an interval for a parameter whose recursion reads a boundary node | a construction for a TMLE parameter with a boundary node hazard, such as a one-sided exact bound for the node combined with the influence curve of the other nodes, with its coverage. No source read here derives one |
| a band that includes such a parameter | the same construction, extended to a joint statement. The fit builds its band over the other parameters |

## Reading a gap correctly

Not every absence is missing package functionality. The refusal taxonomy in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal) distinguishes an unimplemented
well-posed feature from a different causal question and a method that would be wrong by
construction. Only the first belongs on this roadmap.
