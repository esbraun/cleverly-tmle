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

This file lists open work only. A delivered row leaves the file. Code comments and evidence pages
still name delivered rows, RM8 to RM33, by their IDs.
[The roadmap at commit `4ce96cda`](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md)
holds the record of each one. The natural-extension review left the file when it was
complete. [The roadmap at commit `6a55f038`](https://github.com/esbraun/cleverly-tmle/blob/6a55f0388b703765eab79c78efec82474e67ab8e/docs/roadmap.md#x18-natural-extension-reviews)
holds its questions, and the [natural-extension verdicts](technical-reference/natural-extension-verdicts.md) hold its answers.

## Remediation

This queue holds the remediation work that must be complete before a beta release. It holds
shipped behavior that needs a correction or a recorded decision, and a published method needed to
resolve a shipped refusal. Examples are wrong numbers, intervals that no claimed contract covers,
capability rows that do not match their calls, late refusals, and shipped outputs that no
registered study measures. Deliver the rows in priority order, and complete every row before
main-roadmap priority 1.

A new capability still needs its own contract and evidence, even in this queue.

The queue has no open row.

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

A priority below 1 is a remediation priority, so it does not collide with a main-roadmap priority.
The first decimal digit names a delivery group, and the second digit orders the rows inside it.
This project reassigns priorities when it re-triages the queue. The IDs and their anchors never
change, so a commit names a row by its ID. A new row takes the next number that the history of this
file has not used.

The [red-cell ledger](technical-reference/method-evidence/red-cells.md) lists every red verdict
that a registered study publishes. [Red-cell owners](#red-cell-owners) names the item that owns
each one. `tests/unit/test_red_cell_ledger.py` checks the ledger against the committed results.
Each red verdict stays red under a `reporting` policy, so no verdict is hidden and no margin moves.

## Main roadmap

Priority 1 is the beta parity group. It closes the estimator gaps against the pinned comparators
`ctmle` 0.1.2, `ctmle3` 0.1.0, `tmle3` 0.2.0, `drtmle` 1.1.2 and `lmtp` 1.5.4, and against the
survival packages `survtmle`, `concrete` and MOSS. Complete priority 1 before the beta release.
Priorities 2 to 5 follow the beta.

| priority | item | readiness | dependency | details |
| ---: | --- | --- | --- | --- |
| 1.5 | Clustered intervals at unequal cluster sizes and at few clusters | [published support; cluster-as-unit extension](technical-reference/natural-extension-verdicts.md) | the shipped grouped point-treatment split | [X24](#x24-clustered-intervals-at-unequal-cluster-sizes-and-at-few-clusters) |
| 1.6 | Cross-fitted clustered longitudinal TMLE | [published support; cluster-as-unit extension](technical-reference/natural-extension-verdicts.md) | X24 | [X25](#x25-cross-fitted-clustered-longitudinal-tmle) |
| 1.7 | Known stochastic categorical policies at a longitudinal node | [published support; known-weight extension](technical-reference/natural-extension-verdicts.md) | shipped categorical longitudinal nodes | [F1](#f1-stochastic-categorical-policies-at-a-longitudinal-node) |
| 1.8 | Modified treatment policies beyond the additive point shift | published support; pending source read | shipped additive shift and categorical longitudinal nodes | [X12](#x12-modified-treatment-policies-beyond-the-additive-point-shift) |
| 1.9 | Point-treatment survival and time-to-event input | published support; pending source read | shipped survival and competing-risk recursion | [X13](#x13-point-treatment-survival-and-time-to-event-input) |
| 1.10 | Known treatment mechanism | source audit | shipped `treatment_probabilities=` on `DRTMLE` | [X15](#x15-known-treatment-mechanism) |
| 1.11 | Outcome-adaptive C-TMLE intervals from per-arm scalar designs | published support for one mean; natural extension for every arm and contrast | shipped `strategy="oat"` | [X17](#x17-outcome-adaptive-c-tmle-intervals-from-per-arm-scalar-designs) |
| 1.12 | C-TMLE candidate sequences | published support; pending source read | shipped selector paths | [X16](#x16-c-tmle-candidate-sequences) |
| 1.13 | Sequential doubly robust longitudinal estimation | published support; pending source read | implemented longitudinal targets | [X4](#x4-sequential-doubly-robust-longitudinal-estimation) |
| 1.14 | Hazard-based and monotone survival curves | published support; pending source read | X13 | [X14](#x14-hazard-based-and-monotone-survival-curves) |
| 1.15 | Incremental interventions over time | source audit | X12 for the longitudinal intervention density | [X19](#x19-incremental-interventions-over-time) |
| 1.16 | Targeting and fitting options | no new theory; each option keeps its default bit-identical | none | [X22](#x22-targeting-and-fitting-options) |
| 1.17 | Adaptive-propensity IPTW | published support; pending source read | none | [X21](#x21-adaptive-propensity-iptw) |
| 1.18 | Plug-in omitted-variable limits with a declared parametric mechanism | [published support; stacked-equation extension](technical-reference/natural-extension-verdicts.md) | the shipped plug-in bound | [X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism) |
| 2 | Replicate-weight designs | source audit | weighted-law variance construction | [X2](#x2-replicate-weight-designs) |
| 3.1 | Natural and interventional mediation effects | published support; pending source read | target-specific identification and evidence | [X5](#x5-natural-and-interventional-mediation-effects) |
| 3.2 | Continuous-time survival and competing risks | published support; pending source read | continuous-time intensity and targeting contracts; X14 | [X6](#x6-continuous-time-survival-and-competing-risks) |
| 3.3 | Two-phase and outcome-dependent sampling | published support; pending source read | observed-data likelihood and influence correction | [X7](#x7-two-phase-and-outcome-dependent-sampling) |
| 3.4 | Omitted-variable bounds on the other linear functionals | published support; pending source read | the shipped arm-axis bound | [X9](#x9-omitted-variable-bounds-on-the-other-linear-functionals) |
| 3.5 | Continuous-dose MSM with a second mechanism | source audit | implemented continuous-dose MSM targeting, missing-outcome arm targeting, and controlled-direct-effect targeting | [X10](#x10-continuous-dose-msm-with-a-second-mechanism) |
| 3.6 | Learned-policy follow-ups | published support for parts (g) and (h); published support; pending source read for parts (a), (b), (e) and (f); source audit for parts (c) and (d) | the shipped learned-rule value | [X11](#x11-learned-policy-follow-ups) |
| 3.7 | Fold-evaluated CV-TMLE with baseline strata | [published support; finite-partition extension](technical-reference/natural-extension-verdicts.md) | the shipped fold-evaluated estimate and the shipped stratified fluctuation | [X28](#x28-fold-evaluated-cv-tmle-with-baseline-strata) |
| 4 | EP learner | published support; pending source read | shared study, fold, learner, and assessment contracts | [P1](#p1-ep-learner) |
| 5.1 | Nested Riesz engine and initial catalog | published support; source audit complete | typed study, identification, result, and assessment contracts | [R1](#r1-nested-riesz-engine-and-initial-catalog) |
| 5.2 | Evidence-gated Riesz catalog expansion | source audit for each target | R1 and a target-specific derivation | [R2](#r2-evidence-gated-riesz-catalog-expansion) |

## Future investigations

These items are hard stops. Move one into the main roadmap only after a published paper supplies
the missing result, or after the [Eligibility](#eligibility) rule qualifies it as a natural
extension. Package code and a related estimator do not remove the stop. Each row names
the current boundary, and the refusal that keeps it.

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
| Plug-in omitted-variable confidence limits | an influence function or an inference result for the plug-in estimate of the Riesz second moment with an estimated treatment mechanism | the plug-in bounds, `rv`, `max_bias`, `benchmark()` and `contour()`. The one-sided limits and `rva` refuse for every mechanism learner except the declared parametric model that [X26](#x26-plug-in-omitted-variable-limits-with-a-declared-parametric-mechanism) implements | [F26](#f26-confidence-limits-of-the-plug-in-omitted-variable-bound) |
| Targeted bootstrap inference | a construction that defines what is fixed, resampled, refitted, and retargeted, plus the sampling law of the interval | existing bootstrap inference is not this procedure | [F2](#f2-targeted-bootstrap-inference) |
| Longitudinal sensitivity-bound estimation | sample estimation of the bound functionals, a specialized algorithm, and sampling inference | no sensitivity bound on a longitudinal fit | [F16](#f16-longitudinal-sensitivity-bound-estimation) |
| Additional longitudinal estimands | target-specific identification, influence function, targeting construction, and inference conditions | existing end-of-study, survival, competing-risk, and MSM targets only | [F3](#f3-additional-longitudinal-estimands) |
| Other refused C-TMLE and DR-TMLE compositions | composition-specific score, reduced regressions, correction, remainder, and rate conditions. An observational missing outcome and a missing treatment on `DRTMLE` ship through the [composite indicator](technical-reference/dr-tmle/theorem.md#observational-missing-data-the-composite-indicator). On the composite, `evaluation=` and `reduced_crossfit="nested"` stay refused | named pre-fit refusals. `DRTMLE` refuses `NaturalCourseMean`, PAR and PAF with missing outcomes, because no DR-TMLE natural-course mean ships. A `DRTMLE` fit with a non-empty `guard` and estimated weights reports its point estimate under the `estimated_weight_plugin` status, and no interval | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| Selector-path C-TMLE inference | an influence function and covariance after the shipped data-adaptive stopping-index selection | point estimates and path diagnostics only. The greedy, ordered, and discrete paths refuse `ci`, `pvalue`, and `std_error`, except a `discrete` fit whose one declared candidate is the full adjustment set, and report a named working-mechanism plug-in diagnostic | [F18](#f18-selector-path-c-tmle-inference) |
| Outcome-adaptive C-TMLE generated-design inference | exact scalar expansions for the shipped joint binary fit and a multi-arm vector extension of the paper-backed fold-local construction | point estimates only. Every `strategy="oat"` fit refuses `ci`, `pvalue`, and `std_error` under the `generated_design_plugin` status, and reports a named generated-design plug-in diagnostic. X17 builds the per-arm scalar construction that Theorem 1 proves and its stack | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| Other missing-outcome CV-TMLE variants | a direct interval result for fold-specific targeting and for the fixed-repeat median and split-dispersion report after CV-TMLE targeting | the package supports the ordinary natural-course estimator, the stacked natural-course estimator for a binary outcome, the stacked arm-indexed means and contrasts, and their stacked PAR and PAF for a binary outcome. Each stacked estimator uses one repeat and pooled targeting. Shift, incremental, regime, MSM, and controlled-direct-effect targets refuse a cross-fitted fit with missing outcomes before any learner | [F21](#f21-other-missing-outcome-cv-tmle-variants) |
| Grouped cross-fitting beyond point-treatment TMLE | a split law and a cluster-robust variance for the C-TMLE selection folds and the candidate the search stops at | whole-cluster outer folds for cross-fitted point-treatment TMLE and DR-TMLE only. A cross-fitted fit at unequal cluster sizes, in rows or in weight mass, overall or within a reported stratum, takes the `unequal_cluster_plugin` status. A fit with fewer than 40 positive-mass clusters, in total or in one reported stratum, takes `few_cluster_plugin`. Neither reports an interval. The point-treatment boundaries moved to [X24](#x24-clustered-intervals-at-unequal-cluster-sizes-and-at-few-clusters), and the longitudinal row to [X25](#x25-cross-fitted-clustered-longitudinal-tmle) | [F22](#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| MNAR and incremental-intermediate compositions | identification and influence-function results for the exact compositions | point-treatment sensitivity and implemented interventions remain separate | [F6](#f6-mnar-and-incremental-intermediate-compositions) |
| Joint point-treatment parameter axes | a targeting and inference result for one fit that carries an MSM projection together with a regime, shift, or incremental intervention, including its joint score and covariance | single-axis point-treatment fits only | [F17](#f17-joint-point-treatment-parameter-axes) |
| Time-respecting cross-fitting | dependence and split-specific TMLE inference for blocked-temporal or rolling-origin folds | iid and grouped cross-fitting only | [F7](#f7-time-respecting-cross-fitting) |
| Learned-policy value outside the published conditions | an interval for the fold-average learned-rule value without a limiting rule, and results for the refused compositions | `LearnedRuleValue` fits the fold-evaluated CV-TMLE of [learned rules](technical-reference/point-treatment-tmle.md#learned-rules) under the limiting-rule condition. Its boundary study measures under-coverage at an exceptional law. The other F27 requests refuse before any learner | [F27](#f27-learned-policy-value-outside-the-published-conditions) |

## Eligibility

`cleverly` implements established statistical methods; it does not use a package feature as the
place to invent one. A scientific feature enters implementation only when a published derivation
covers the estimand and requested inference regime. A new estimator or composition requires an
identified parameter, its influence function, the targeting or estimating equations, and the
remainder and rate conditions needed for the claimed interval. If one is absent, the item is to
locate published theory, not create it here.

One exception applies, and this project applies it generously. A natural extension of a
published derivation can close the gap. Such an extension needs no new fundamental proof. It must
satisfy every condition in the table below.

| condition | what it requires |
| --- | --- |
| a published base | the extension starts from one published result, and it cites that result's exact locator |
| an established argument | each step reuses the base result's own argument, or a standard result. The next table lists the standard results |
| no new fundamental proof | no step needs a new kind of limit theorem, and no source records a step as open or as a defect |
| inherited conditions | the remainder and rate conditions carry over from the base result, and the contract states each one |
| a written record | the item's contract states the base result, each step, and the argument that carries it |
| its own evidence | the extension has a registered study, and each step that can vanish at the truth has a nonzero witness |

Each step below counts as an established argument. A step of this kind needs no further source.

| standard step | what it carries |
| --- | --- |
| linearity | a sum, a difference or a fixed linear functional of asymptotically linear estimates, such as a contrast, an attributable risk or a restricted mean |
| the delta method | a smooth function of a fixed number of estimates, such as a ratio, a fraction or a transform |
| a fixed-dimension stack | estimates that are each asymptotically linear on the same rows, stacked by Cramér–Wold, with their joint covariance from the stacked curves |
| the chain rule for influence functions | a composition of differentiable functionals, including a known weight or a known design |
| an indicator reduction | a binary-treatment result applied to $1\{A = a\}$ for one arm, or to a composite indicator such as $\Delta \cdot 1\{A = a\}$, when identification holds for that indicator |
| a finite partition | a marginal result applied inside each of a fixed number of baseline strata, then stacked |
| a cluster as the unit | an iid result applied to independent clusters as the sampling units, with cluster-summed curves |
| a stacked estimating equation | a smooth finite-dimensional nuisance whose score is stacked with the target equation, as Saul and Hudgens (2020) describe |

Apply the conditions to reach a verdict, not to find a reason to wait. When every step is in the
table above, the burden falls on an objection: a source that records the step as open or as a
defect, or a required limit theorem of a new kind. Without such an objection, the extension
qualifies. The registered study and the nonzero witnesses then carry the evidence. A reading that
rejected an extension before this rule is open to a new reading under it.

An extension that fails one condition is new theory. It stays in the future investigations grid.
Three examples fail. A data-adaptive selection step has no established argument. A fold-local
update replaces a pooled one, and upstream `lmtp` commit `9996b04` records its training-fold update
as a bug. A growing target dimension needs a new kind of limit theorem.

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

## Red-cell owners

Each red cell in the [red-cell ledger](technical-reference/method-evidence/red-cells.md) names its
owner by an `id` in the table below. `tests/studies/evidence/red_cells.py` reads this table.
`tests/unit/test_red_cell_ledger.py` fails on four states: a red row with no owner, an owner with
no red row, an owner that the `id` column does not list, and a red row in a `gated` study.

Every red verdict stays red under `reporting` at its registered budget and margin. Three remedies
are refused after a run: raising a budget, moving a margin, and re-declaring a cell's law, learner
or size after its verdict is seen. A change of law, budget or convention is declared before the
regeneration that it governs.

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

The `X8-logit-small-stratum` and `X8-identity-small-stratum` owners hold cells of the two
stratified MSM studies in their smallest stratum. Each is a standing record of a reading, and
neither gates the beta release. The `X8-drtmle-one-sided-bias` and `X8-drtmle-small-stratum`
owners do the same for the stratified DR-TMLE study.

| id | work | acceptance |
| --- | --- | --- |
| `F18` | an inference result for the shipped selector path | an influence curve derived after the stopping-index selection, and a registered study whose `selector_necessity` and `type_i_error` cells pass their existing margins at their existing budgets. The multi-arm `interval_calibration/correctly_specified` cell must also pass its band, because its standard-error ratio interval, 0.8987 to 0.9862, measures the fixed-candidate covariance that F18 replaces |
| `F19` | an inference result for the generated design | a published result, or an Eligibility-qualifying natural extension, that establishes whether the ordinary adaptive-propensity curve suffices for the cross-fitted shared-multinomial joint target. Implement any contribution the result requires, then register covariance and coverage evidence at declared budgets. The `generated_design` pair stays a reporting diagnostic, and acceptance does not require a negative standard-error deficit |
| `F27` | an interval for the fold-average learned-rule value outside the published conditions | [F27](#f27-learned-policy-value-outside-the-published-conditions) gives the acceptance for each request. The `exceptional` row needs a published result. The `weak_blip` row is a finite-sample limit at n = 2,000 inside condition C3 |
| `RM18-fixed-weights` | the `interval_calibration/static__correctly_specified` cell and the paired `ey_regimen[never]` row of `weighted-ltmle-crossfit` | reading `finite-sample, contracting` and `equivalent at the declared budget` ([`tests/diagnostics/rm18_fixed_weights/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_fixed_weights)). The pooled update applies the weighted targeting equation, so the reading is no reason to revert it. A result for complex surveys, estimated or calibrated weights, clustering or other longitudinal weight laws is still open. Karim (2026), arXiv:2606.30918v2, is adjacent point-treatment survey theory, and it does not cover this construction |
| `RM18-boundary` | cells at a finite-budget boundary: `root_n_and_efficiency/n_500` of `canonical-multi-arm-ctmle-selector`; `double_robust_contraction/treatment_correct_n1500` of `canonical-drtmle`; `double_robust_contraction/outcome_correct_n4000`, `interval_calibration/correctly_specified` and `root_n_and_efficiency/n_500` of `canonical-multi-arm-drtmle`; `double_robustness/static__both_wrong` and the paired `ey_regimen[always]` and `ey_regimen[treat then continue if l2 positive]` rows of `weighted-ltmle-crossfit` | reading `resolved: truth satisfies the gate` for the six cells, and `comparator SE convention` for the two paired rows ([`tests/diagnostics/rm18_boundary/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_boundary)). Before a future regeneration of `weighted-ltmle-crossfit`, declare whether it adopts the `lmtp` standard-error convention for the two paired rows |
| `RM18-one-sided-bias` | `double_robustness/outcome_correct` and `double_robustness/treatment_correct` of `canonical-drtmle`, and `double_robustness/treatment_correct` of `canonical-multi-arm-drtmle` | reading `shared` and `mixed` on the binary rows, and `finite-sample excess detected` on the multi-arm row ([`tests/diagnostics/rm18_one_sided_bias/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_one_sided_bias)). A declared localization design on 2,000 fresh draws read `no increment at the declared resolution` ([`tests/diagnostics/rm19_one_sided_increment/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm19_one_sided_increment)). No validated R comparator fits the multi-arm configuration, so the multi-arm excess has no attribution |
| `RM18-ordinary-weighted` | `interval_calibration/static__correctly_specified`, `type_i_error/static__sharp_null` and the four `targeting_necessity` cells of `weighted-ltmle` | reading `finite-sample, contracting`, `resolved within gate`, `control underpowered by design` and `family resolved` ([`tests/diagnostics/rm18_ordinary_weighted/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_ordinary_weighted)). The static untargeted control discriminates with a probability near 0.72 at the registered 1,200 replicates. Re-declare that control's law before any regeneration of the study, as the control precedent at `69de6f8` allows |
| `RM18-comparator-density` | the paired `ate_shift[+0.25 vs natural course]` row of `shift-policies` | reading `cleverly density representation` ([`tests/diagnostics/rm18_comparator_density/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_comparator_density)). The calibration leg fails with a 99% upper endpoint of 0.065856 against a margin of 0.05, so the conclusion is `inconclusive`. If X12 changes the density representation, it re-reads this row |
| `strata-boundary-mean` | the two primary truth rows of `ey[1][V=2]` and the `interval_calibration/v2_ey1__correctly_specified`, `simultaneous_coverage/strata__simultaneous_band` and `simultaneous_coverage/crossfit_strata__simultaneous_band` cells of `canonical-stratified-tmle` | reading `finite-sample Wald interval`: the treated mean of stratum V = 2 is 0.9358, and at n = 2,000 the stratum holds about 157 treated rows and about 9 expected non-events. R `tmle3` shows the same coverage on the same draws, the bias is inside its margin, and the oracle band at the exact critical value also under-covers ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then each cell stays published red under `reporting` |
| `band-finite-sample` | the `simultaneous_coverage/all_reported__simultaneous_band` cell of `canonical-ltmle-survival`, and the `categorical_ltmle__simultaneous_band`, `categorical_ltmle_crossfit__simultaneous_band` and `cde_z0__simultaneous_band` cells of `default-simultaneous-bands` | reading `finite-sample, pointwise shortfall inherited`: the oracle band at the design critical value also covers below 0.95, and the source study's pointwise calibration of the same parameters sits near the lower edge of its band. The package critical value averages a little below the design one, so the multiplier explains a small part of each shortfall ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then each cell stays published red under `reporting` |
| `F4-calibration-draws` | the `interval_calibration/ate__correctly_specified` and `simultaneous_coverage/arms__simultaneous_band` cells of `multi-arm-mar-drtmle`, which read one set of 2,400 fits | reading `finite-sample, one draw set`. The SE-ratio interval ends at 0.9294 against a floor of 0.93, and the joint-coverage interval at 0.9196 against 0.92. The calibration fits have an empirical SD 1.037 times their mean standard error. The same configuration gives 0.997 at `n_2000` and 0.989 at `l3_ate_low__both_correct`, on independent draws. `simultaneous=True` changes no estimate and no standard error. The band critical value averages 2.500 against the oracle 2.508, and the oracle band covers 0.9342 against the package's 0.9337, so the band construction explains none of the shortfall ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). Pooled over the 4,400 independent both-correct fits of `ate[low vs high]` at n = 2,000, the spread is 1.020 times the mean standard error (bootstrap 99% interval 0.991 to 1.047), and a Bartlett test across the three cells gives p = 0.20, so the draw set differs from its siblings by no more than chance. No defect was found in the K-arm code. The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then both cells stay published red under `reporting` |
| `composite-high-arm` | the `ey[high]` truth rows of `cleverly-composite-drtmle` and `drtmle-r-composite` on the three-arm scenario of `composite-missing-drtmle` | reading `finite-sample, shared with the comparator` ([`tests/unit/test_composite_high_arm_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_composite_high_arm_reading.py)). Coverage is 0.9225 in the package and 0.9237 in R `drtmle`, and the 99% intervals end at 0.8949 and 0.8963 against a floor of 0.90. The paired mean difference is 4e-6. The mean standard error is 0.0342 in both, against an efficiency-bound SD of 0.0344 at n = 2,000, so the variance estimate is not low. The empirical SD of the estimates is 0.0367, 1.066 times the bound. The 47 package fits that reached `max_outer` cover at 0.957, against 0.921 for the fits that met the tolerance, so the cap is not the cause. `high` has the smallest composite mechanism of the law, 0.070 where `W = 0`, which holds half the mass. Every other arm and contrast of the scenario passes. The owner closes when a re-declared cell, with its law or size declared before its run, passes, or when a small-sample interval correction with its own registered study does. Until then both rows stay published red under `reporting` |
| `X8-logit-small-stratum` | the `interval_calibration/logit_v1_a__correctly_specified` and `interval_calibration/logit_v2_a__correctly_specified` cells of `canonical-stratified-incremental-msm` | reading `finite-sample, inherited from the unstratified fit`. The SE-ratio intervals end at 0.927 and 0.939 against a floor of 0.93, with coverage 0.943 and 0.944. Stratum 2 holds about 400 of the 2,000 rows. A declared diagnostic fitted the shipped unstratified logit MSM on samples of that size from the law given $V = 2$. It under-reports alike, 0.896 against the stratified fit's 0.867 on fresh draws, and both are calibrated at four times the size, 0.997 and 1.006 ([`tests/diagnostics/x8_logit_small_stratum/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x8_logit_small_stratum)). The stratified fit equals the subset fit exactly under fixed nuisances (`tests/unit/test_stratified_msm_exact.py`), so no defect of the stratified construction was found. The owner closes when a re-declared cell, with its law or size declared before its run, or a small-sample interval correction with its own registered study passes. Until then both cells stay published red under `reporting` |
| `X8-identity-small-stratum` | the `interval_calibration/identity_v2_a__correctly_specified` cell and the paired `msm[W][V=2]` row of `canonical-stratified-msm-identity` | reading `finite-sample, smallest stratum`. Stratum 2 holds about 400 of the 2,000 rows. The calibration cell's SE-ratio interval is 0.926 to 1.004 against a floor of 0.93, and on the primary draws the same coefficient reads 0.983 in `cleverly` and 0.985 in R `tmle3`. The paired row is inconclusive on its calibration leg alone: the upper limit is 0.0523 against a margin of 0.05, at a resolution of 0.0476. The two estimates differ by 6e-6 on average with the same spread, and R reports a standard error 2.2% above its own spread, because it fits a Gaussian `Lrnr_glm` inside each subset ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). No defect was found. The owner closes when a re-declared cell, with its law or size declared before its run, passes. Until then both stay published red under `reporting` |
| `X8-drtmle-one-sided-bias` | the four `double_robustness/*__treatment_correct` cells of `canonical-stratified-drtmle`: the marginal ATE and each stratum ATE | reading `shared, finite-sample, contracting` ([`tests/diagnostics/x8_drtmle_treatment_correct/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x8_drtmle_treatment_correct)). The refit reproduces the committed estimates exactly. On each stratum's rows, the shipped unstratified `DRTMLE` and R `drtmle` 1.1.2, handed the same initial arrays, carry the same positive bias: 0.0099 to 0.0237 against 0.0121 to 0.0237 for the stratified fit. This is the one-sided bias that `RM18-one-sided-bias` records for the unstratified study on the same paper law, at stratum size. The stratified marginal is the mixture of the stratum estimates, so it inherits their bias: 0.0151 against 0.0029 for the unstratified marginal at n = 2,000. At n = 8,000 it falls to 0.0058, faster than its spread, which halves from 0.0269 to 0.0135. No defect of the stratified construction was found. The owner closes when a re-declared cell, with its law or size declared before its run, passes, or when `RM18-one-sided-bias` closes with a correction that also covers the stratified fit. Until then the four cells stay published red under `reporting` |
| `X8-drtmle-small-stratum` | the two primary truth rows of `ey[1][V=2]` and the `interval_calibration/v2_ate__correctly_specified` cell of `canonical-stratified-drtmle` | reading `finite-sample, shared with the comparator`. Stratum 2 holds about 400 of the 2,000 rows. Coverage of `ey[1][V=2]` is 0.900 in the package and 0.8925 in R `drtmle` on the same draws, with estimates 4e-5 apart on average. The calibration cell's SE-ratio interval ends at 0.9294 against a floor of 0.93, and on the primary draws the same ATE reads 0.985 in the package and 0.973 in R ([`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py)). The owner closes when a re-declared cell, with its law or size declared before its run, passes. Until then the three stay published red under `reporting` |
| `X20-bootstrap` | the `interval_calibration` cells `boot_ey_crossfit__correctly_specified`, `boot_ate_crossfit__correctly_specified` and `boot_ate_clustered__correctly_specified` of `full-refit-bootstrap-and-derived-contrasts` | the declared rule 3 keeps the `end_of_study/cross_fit` and `end_of_study/cluster` kinds out of `LICENSED_BOOTSTRAP_DESIGNS`, so their bootstrap prints as a diagnostic. A diagnostic reading, not a registered result: on this cell the cross-fitted estimator is heavy-tailed. Its 4,000 committed estimates have excess kurtosis 6.7, and about 5% of them lie at the probability boundary, because `g_bounds=(1e-8, 1)` with saturated cell means leaves sparse training cells unbounded. A bootstrap resample holds about 63% unique units, so more replicates reach the boundary and the replicate spread exceeds the sampling spread (SE ratios 1.23 and 1.27). A probe that kept each unit's copies in one fold raised the SE ratio to about 1.65, so the split of copies across folds is not the cause. The cluster contrast covers 0.924 with a calibrated SE ratio (0.977), a finite-sample shortfall of the percentile interval at 60 clusters. The owner closes when a re-declared cell for the kind, with its law, size, nuisance bounds or resampling scheme declared before its run, passes a registered cell |

## Detailed implementation contracts

The sections below give one contract for each open item. Their physical order does not override
the main grid.


### X24. Clustered intervals at unequal cluster sizes and at few clusters

A cross-fitted `TMLE` or `DRTMLE` fit at unequal cluster sizes takes `unequal_cluster_plugin`. A
clustered fit with fewer than 40 positive-mass clusters, in the fit or in one reported stratum,
takes `few_cluster_plugin` (`src/cleverly/inference/cluster.py`, `src/cleverly/_inference_status.py`).
Neither reports an interval. The
[natural-extension verdicts](technical-reference/natural-extension-verdicts.md) record the review
as part (g).

| item | contract |
| --- | --- |
| base results | Wang, Park, Small and Li (2024), *Journal of the American Statistical Association* 119(548), Assumption 1(b), the $\mu_I$ definition of Section 2, Section 4.2 and Theorem 4(b). Nugent, Marquez, Charlebois, Abbott and Balzer (2024), *Biostatistics* 25(3), Section 2.2, last paragraph, for a $t$ reference with $J - 2$ degrees of freedom below 40 clusters. Benitez et al. (2023), Sections 3.1.2 and 3.2.1, give the same recommendation |
| step at unequal sizes | a cluster as the unit, with the cluster size $N_j$ a bounded random cluster attribute (Assumption 1(b)). The row-weighted estimand is $\mu_I(a) = E\{\sum_j Y_{ij}(a)\}/E(N_i)$, a ratio of two cluster-level means, so the delta method gives its curve. Wang et al. treat a cluster-level treatment with an AIPW estimator. The package's row-level treatment uses the cluster-sum estimating-equation argument of the shipped equal-size case |
| step at few clusters | the $t_{J-2}$ reference adds no limit theorem. Its quantiles converge to the normal ones, so the asymptotic claim does not change. The sources recommend it as a finite-sample approximation |
| objection search | Park and Kang (arXiv:2110.07740) note that cluster averages equal row weighting only at equal sizes. That concerns the estimand, and the package's estimand is $\mu_I$. Wang et al., Section 5, use a variance factor $m/(m-5)$ and a $t$ reference with $m - 5$ degrees of freedom, and Remark 3 recommends machine learning at about 100 clusters. These are finite-sample warnings, and the registered study measures them |
| inherited conditions | independent clusters; a bounded random cluster size; no interference between units; the remainder rates in the cluster count. A row-level treatment keeps row-level exchangeability given `W`. The cluster size enters only if it confounds, and then it must be a covariate |

The contract fixes these rules before it declares the study.

| decision | rule |
| --- | --- |
| degrees of freedom | $J - 2$ for the fit, and $J_s - 2$ for each reported stratum. A contrast uses the degrees of freedom of its own cluster count |
| minimum cluster count | below $J = 4$ the fit keeps `few_cluster_plugin` |
| simultaneous band | none below 40 clusters. The fit reports pointwise intervals only, and the summary says so |
| `plugin_interval` | keeps the normal reference, as a labelled diagnostic |

Acceptance:

- the ratio-of-cluster-sums curve for cross-fitted `TMLE` and `DRTMLE` in place of
  `unequal_cluster_plugin`, with a nonzero witness that it differs from the row curve at unequal
  sizes;
- the $t$ reference for clustered `TMLE`, `DRTMLE` and in-sample `LTMLE`;
- each status retired or narrowed, with its tests;
- a registered study at unequal sizes, informative and non-informative, and at
  $J \in \{10, 20, 30\}$. Declare the `reporting` policy and the red-cell owner before the run.

### X25. Cross-fitted clustered longitudinal TMLE

`LTMLE._refuse_cross_fitted_design` refuses `id=` above one fold. The in-sample clustered fit runs.
The [natural-extension verdicts](technical-reference/natural-extension-verdicts.md) record the
review as part (f).

| item | contract |
| --- | --- |
| base result | Díaz, Williams, Hoffman and Schenck (2023), Section 5.2, journal page 852, for the cross-fitted TMLE over a random partition, and Theorem 3, page 853 |
| step | a cluster as the unit. Whole-cluster folds make each validation fold independent of its training fold. The estimating equation summed over rows is a cluster-level estimating equation, so the curve is the cluster-summed row curve divided by $E(N)$. The in-sample clustered `LTMLE` ships this curve |
| objection search | the earlier text of F22 said that no read source gives this variance, which is an absence. Balkus, Laith and Hejazi (2026), and Nugent et al. (2024), Section 2.3, support grouped folds with aggregated cluster curves. Schnitzer, van der Laan, Moodie and Platt (2014), Section 3.4.1, give the in-sample clustered longitudinal TMLE with a sandwich variance. None records a defect |
| inherited conditions | independent clusters; no interference between units; a bounded cluster size; the Theorem 3 rates and bounded density ratios in the cluster count; the unequal-size and few-cluster rules of [X24](#x24-clustered-intervals-at-unequal-cluster-sizes-and-at-few-clusters) |

This item inherits the rules of X24, so it cannot ship before X24.

Acceptance:

- whole-cluster folds for `LTMLE` cross-fitting, and the refusal of `id=` lifted;
- a nonzero witness that the cluster-summed curve differs from the row curve under within-cluster
  correlation, and a mutation control that splits a cluster across folds and fails;
- a registered study of the clustered cross-fitted longitudinal fit.

### F1. Stochastic categorical policies at a longitudinal node

The implemented surface assigns one category per unit, because a `DynamicRegimen` node returns one
label per row. This item adds a node that holds a known policy density $q_t(\cdot \mid h_t)$. The
[natural-extension verdicts](technical-reference/natural-extension-verdicts.md) record the review
as part (c).

| item | contract |
| --- | --- |
| base result | Díaz, Williams, Hoffman and Schenck (2023), *Journal of the American Statistical Association* 118(542). Section 2, journal page 849, admits a random regime $d(a_t, h_t, \varepsilon_t)$ when the randomizer is "(i) drawn independently across units and independently of U, and (ii) its distribution does not depend on P". The section places $\varepsilon_t$ in $L_t$ "without loss of generality". Section 4, page 850, keeps the efficiency theory to a $d$ that does not depend on P. Theorem 1, page 849, Section 5.2, page 852, and Theorem 3, page 853 |
| steps | with $\varepsilon_t$ in $L_t$, the estimator that records the randomizer is Theorem 3 as stated. The package ships the integrated estimator, which has no Monte Carlo noise. To marginalize $\varepsilon_{t+1}$ in the Theorem 1 recursion is to average under a known law, so $m_t$ reads $\sum_j q_{t+1}(j \mid h)\, m_{t+1}(j, h)$ and $r_t$ becomes $q_t/g_t$. The curve without $\varepsilon$ is $E[D_{\mathrm{aug}} \mid O]$, the projection of the augmented curve onto the tangent space without $\varepsilon$. The remainder is $E_\varepsilon[\mathrm{Rem}]$, second order by linearity of expectation. The Eligibility row is the chain rule with a known weight |
| conditions met by construction | (i) no $m_t$ reads a randomizer; (ii) each $r_k$ reads $\varepsilon_k$ only; (iii) the randomizers are mutually independent. $q_t$ reads $h_t$ only |
| objection search | page 850 says multiply robust estimation "is not generally possible for random regimes d that depend on P". That applies to a policy that depends on P only, which stays out: [X19](#x19-incremental-interventions-over-time) owns the incremental tilt. The earlier sentence of this item, that a point-treatment stochastic regime is not sufficient evidence, gives way to this direct longitudinal source |
| inherited conditions | the Theorem 3 rate condition $\sum_t \lVert \hat r_t - r_t \rVert\, \lVert \tilde m_t - m_t \rVert = o_P(n^{-1/2})$; bounded ratios, $P\{r_t < c\} = 1$; positivity of $g_t$ wherever $q_t > 0$; a policy density that is known and fixed before the fit (`density_kind="known"`, as the point-treatment `Stochastic` regime requires). A policy that reads the natural value $A_t$ needs Assumption 3 and is a modified treatment policy ([X12](#x12-modified-treatment-policies-beyond-the-additive-point-shift)) |

Acceptance:

- a node with a known policy density inside `DynamicRegimen`, whose recursion evaluates
  $\sum_j q\, m(j, \cdot)$ and whose cumulative ratio uses $q/g$;
- an exact-law Gateaux witness with a nonzero, non-degenerate $q$;
- mutation controls that use a selected column in place of the ratio, or drop a node, and fail;
- an exact reduction to the deterministic regimen when $q$ is a point mass;
- the estimator that records the randomizer, as a cross-check on one law;
- a registered study. `lmtp` has no direct comparator.

### X12. Modified treatment policies beyond the additive point shift

`Shift` adds a constant to a continuous dose at a point treatment, with an optional cap
(`src/cleverly/interventions/shift.py`). The pinned `lmtp` 1.5.4 also estimates three targets that
`cleverly` does not.

| part | what is missing | published source | comparator |
| --- | --- | --- | --- |
| (a) a general point-treatment policy | a multiplicative shift, and a policy $d(a, w)$ that a user declares, invertible where the density ratio needs it | Haneuse and Rotnitzky (2013); Díaz and van der Laan (2012) | `lmtp_tmle` with one time point and a `shift` function |
| (b) a longitudinal policy on a continuous dose | a fixed policy $d(a_t, h_t)$ at each node of a longitudinal fit, additive, multiplicative or declared. `LTMLE` refuses `shifts=` today, and it reads a numeric node as unordered arms | Díaz, Williams, Hoffman and Schenck (2023), Theorem 3, journal page 853, which the [longitudinal reference](technical-reference/longitudinal-tmle.md) already cites for categorical rules | `lmtp_tmle` with `mtp = TRUE` |
| (c) several treatment columns at one node | a policy on a vector of exposures at one time point | Hoffman et al. (2024), *Epidemiology* 35(5). This locator is pending its source read | `lmtp` with `trt` as a list |

Read Theorem 3 for the density-ratio condition on a continuous dose, the policy invertibility
condition, and the remainder. The longitudinal density ratio needs a conditional density of the
dose at every node. The point-treatment shift already estimates one with `density_bins=`. Each
node's ratio enters the cumulative product, and `g_bounds` must bound the product as it bounds
the categorical one.

Each policy must be fixed before the fit and declared `"known"`, as `Rule` and `DynamicRegimen`
declare it. A policy learned from the analysis sample stays refused.

Acceptance:

- exact-law Gateaux witnesses for the point and longitudinal curves on a coarse discrete dose;
- mutation controls for the inverse policy, the density ratio at each node, and the node order;
- registered ordinary and cross-fitted studies against the pinned `lmtp` 1.5.4, with identical
  folds and exact density inputs where the comparator accepts them;
- a re-read of the `RM18-comparator-density` paired row if the density representation changes;
- an update of the `shifts` refusal in `_REFUSED`, and of the continuous-dose rows of the
  longitudinal and scope pages, which then point to the shipped fit.

### X13. Point-treatment survival and time-to-event input

The shipped survival and competing-risk fits take a wide table with one column for each node, and
one treatment column at every node. A baseline-only treatment is refused with `DataError`. The
survival packages of the TMLE ecosystem start from a different design and a different data
layout.

| part | what to add | published source | comparator |
| --- | --- | --- | --- |
| (a) a point-treatment survival curve | a baseline treatment, censoring over time, and the survival or cumulative-incidence curve at declared horizons, as a design of its own. The fit intervenes on the baseline treatment and on censoring only | Stitelman, De Gruttola and van der Laan (2012); Benkeser, Carone and Gilbert (2018), *Statistics in Medicine* 37(2), for `survtmle` | `tmle3` `tmle_survival` at the pinned `ed72f8a`; `survtmle` `method = "mean"` |
| (b) long-format input | a converter from one row per unit with an event time and an event type, the `ftime` and `ftype` layout, to the node layout at a declared time grid | no theory. It is a data transformation | `survtmle`; `concrete` |

Part (a) can reuse the sequential recursion. State in the contract whether it builds the
treatment nodes from the baseline column, or fits a separate point-treatment recursion. The two
must give the same estimate on one law, and a test pins that identity.

Part (b) must refuse an event time outside the grid and a tie that the grid cannot order. It must
state how it bins a continuous time. It must round-trip on a law whose event times sit on the
grid.

Acceptance:

- exact-law witnesses for part (a), with a nonzero censoring mechanism;
- a registered ordinary study against the pinned `tmle3`;
- a converter test for each refusal of part (b).

`survtmle`, `concrete` and MOSS are archived on CRAN, so a pinned commit is needed before any of
them becomes a comparator.

### X15. Known treatment mechanism

`TMLE` always estimates the treatment mechanism. `DRTMLE` takes `treatment_probabilities=` only
with `delta=`, and it refuses the array on complete data. A randomized trial knows its mechanism.
`tmle3` (`LF_known`) and `drtmle` (`gn`) accept it.

Work: add a declared known mechanism to `TMLE`, and to complete-data `DRTMLE`. Keep it distinct
from `randomized=True`, which estimates the mechanism for chance-imbalance adjustment. Read the
source for each inference claim before choosing the reported curve.

| question | where the audit starts |
| --- | --- |
| the curve when the mechanism is known, and the direction of its variance against the efficient curve | the audit must find and read a source. Moore and van der Laan (2009), *Statistics in Medicine* 28(1), on covariate adjustment in trials, is the first candidate |
| the `DRTMLE` complete-data refusal | the row of [DR-TMLE supported estimands](technical-reference/dr-tmle/supported-estimands.md) that states the gap |

The known array is row-aligned to the data as passed, so it keeps the shipped refusal with
`n_bootstrap=`.

Acceptance:

- a contract that cites the read source for the reported curve;
- an exact-law test where the outcome regression is wrong and the known mechanism is true, with a
  mutation control that perturbs the array;
- a test that a known mechanism and a fitted mechanism give the same point on an exact law where
  the fitted mechanism equals the known one;
- a registered study at a randomized law.

The [refusal taxonomy](technical-reference/scope-and-refusals.md) then drops the `DRTMLE` row.

### X17. Outcome-adaptive C-TMLE intervals from per-arm scalar designs

Benkeser, Cai and van der Laan (2020), *Statistical Science*, Theorem 1, proves that the ordinary
adaptive-propensity curve needs no generated-design term for one binary treatment-specific mean.
It holds under the paper's six regularity conditions. The proof uses a scalar design: the
propensity is fitted on the estimated outcome regression of that one arm. The shipped
`strategy="oat"` fit uses a different design. It fits one multinomial mechanism on every arm's
prediction and targets all arms jointly, so every fit takes `generated_design_plugin`.

This item builds the scalar design, and it extends it to every arm by the standard steps of the
[Eligibility](#eligibility) rule.

| part | construction | argument |
| --- | --- | --- |
| (a) one binary treatment-specific mean | for `ey1` or `ey0` alone, fit the propensity of that arm on that arm's outcome prediction alone, and target the mean with one coefficient | Theorem 1 directly |
| (b) every arm mean, binary or multi-arm | repeat part (a) for each arm on $1\{A = a\}$, with its own scalar design and its own coefficient | an indicator reduction for each arm, then a fixed-dimension stack of the per-arm curves on the same rows |
| (c) contrasts | ATE, RR and OR from the stacked arm means | linearity and the delta method |
| (d) the cross-fitted form | fold-local outcome and propensity fits, with one pooled coefficient per arm | Theorem 1 with cross-fitted nuisances. The author preprint (arXiv:1901.05056), Appendix D, "Cross-validated CTMLE", printed page 26, states this step: fold-trained nuisances, one $\epsilon_n$ "found by pooling over the validation samples", and sample splitting in place of the Donsker condition (iv). Its pooled coefficient matches this part. The [natural-extension verdicts](technical-reference/natural-extension-verdicts.md) record it as part (k). Take the final page numbers from the published Supplement A |

Each configuration takes the `influence_curve` status. The shared-multinomial joint design keeps
`generated_design_plugin`, and [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference)
keeps its route. The per-arm design is a different estimator, so a request names it explicitly,
and the shared design stays the default only if this item decides so in its contract.

The contract states the six conditions, the article locator, and the Appendix F and Appendix G
mismatch between the article and Supplement A.

Acceptance:

- an exact-law witness that the scalar design and the joint design differ when the arms differ;
- a mutation control that feeds another arm's prediction to an arm's design and fails;
- a mutation control that drops one arm's cross-covariance block from the stack and fails;
- a registered study of parts (a) to (c) with coverage, standard-error calibration, null-size and
  joint-coverage cells at declared budgets, on a binary and a three-arm law;
- a status test that the shared design still refuses `ci`, `pvalue` and `std_error`.

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

### X4. Sequential doubly robust longitudinal estimation

`lmtp_sdr` implements the sequentially doubly robust estimator of Díaz, Williams, Hoffman and
Schenck (2023). That estimator is consistent when either the outcome regression or the treatment
mechanism is consistent at each time point. It is a second estimator over registered longitudinal
targets, so it adds no estimand. The comparator is the pinned R `lmtp` 1.5.4 that the longitudinal
rows already use, and no new container is needed.

Theorem 4 of the same article certifies a fold-local recursion for the SDR estimator. Theorem 3
certifies the pooled fluctuation that the cross-fitted TMLE runs. Read the rate conditions that
the SDR interval claims first, because they differ from the sequential regression conditions.
After X12 ships, the SDR estimator covers the continuous-dose policies too, and its study covers
one of them.

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
CRAN. Part (b) supplies a one-step band over the whole curve. The
[survival-curve study](technical-reference/method-evidence/ordinary-survival-curve-longitudinal-tmle.md)
measures the band of the shipped per-horizon curve.

Acceptance:

- exact-law witnesses for each targeting construction;
- a test that each curve is monotone on a law where per-horizon targeting is not;
- a registered study with joint coverage over the horizons.

### X19. Incremental interventions over time

`TMLE(incremental=...)` fits Kennedy's odds tilt of a binary point treatment. `LTMLE` refuses
`incremental=`, because the tilt needs the product of tilted mechanisms and a mechanism submodel at
every node. The pinned `lmtp` 1.5.4 ships `ipsi()`, which is a risk-ratio tilt and not the odds
tilt.

| part | what to add | published source | comparator |
| --- | --- | --- | --- |
| (a) the odds tilt over time | the incremental intervention at every node of a binary longitudinal treatment | Kennedy (2019), *Journal of the American Statistical Association* 114(526), which treats time-varying treatments | `npcausal` 0.1.0, pinned for the point-treatment incremental study. The audit must confirm that its `ipsi()` fits time-varying data |
| (b) the risk-ratio tilt | `lmtp`'s risk-ratio incremental intervention, at a point treatment and over time | the source that the `lmtp` documentation cites for `ipsi()`. The audit must read it | `lmtp` 1.5.4 `ipsi()` |

Part (b) is a different parameter from part (a). Give it its own name and its own estimand, and
do not alias it to `Incremental`.

Acceptance:

- exact-law witnesses with a nonzero tilt at every node;
- a mutation control that drops one node's tilt;
- a registered study for each part against its comparator.

### X22. Targeting and fitting options

One pull request delivers these options. Each comparator ships the option, and none needs new
theory. Each option keeps every default fit bit-identical, and each changes a fit when it binds.

| part | option | what it does | comparator |
| --- | --- | --- | --- |
| (a) best iterate | report the targeting iterate with the smallest score norm over the loop | a fit that stops at `max_iter` reports its best iterate, and its record says which one | `tmle3` `use_best` |
| (b) stopping rule | choose the score threshold: the shipped variance-scaled rule, or a threshold of $1/n$ | the two rules that `tmle3` names `scaled_var` and `sample_size` | `tmle3` `convergence_type` |
| (c) step bound | bound the fluctuation step in each iteration | a damped update for a fit whose first step overshoots | `tmle3` `constrain_step` and `delta_epsilon` |
| (d) quantile trimming | trim the cumulative density ratio at a declared quantile, beside the shipped bound `g_bounds` | the trimming rule of `lmtp`. The fit records the trimmed share, as it records truncation now | `lmtp` `.trim` |
| (e) Markov order | restrict each node's nuisance history to the last `k` nodes | an assumption about the data. The identification summary states it | `lmtp` `k` |
| (f) per-arm outcome fit | fit the outcome regression separately in each arm | a nuisance choice for `TMLE` and `DRTMLE` | `drtmle` `stratify` |
| (g) early stop on a C-TMLE path | stop the greedy search after a declared number of steps without a risk improvement | a cheaper path. The fit keeps its `working_mechanism_plugin` status | `ctmle` `patience` |

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
continuous-time interval, so the existing longitudinal rows do not transfer. The point-treatment
design of [X13](#x13-point-treatment-survival-and-time-to-event-input) and the whole-curve
targeting of [X14](#x14-hazard-based-and-monotone-survival-curves) come first.

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
assumptions, influence functions, targeting construction, and inference conditions. Add accepted
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

Outcome-adaptive C-TMLE selects no candidate. It fits the categorical treatment mechanism on the
estimated vector of arm-specific outcome predictions. Every `strategy="oat"` fit takes the
`generated_design_plugin` status, including a fit with `delta=` and an `ey1`-only request. `ci`,
`pvalue` and `std_error` refuse, and `plugin_std_error` and `plugin_interval` keep the ordinary
adaptive-propensity diagnostic.

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
[X17](#x17-outcome-adaptive-c-tmle-intervals-from-per-arm-scalar-designs) builds the
per-arm scalar designs that Theorem 1 proves, and their stack.

**What ships against Appendix D.** `CTMLE` refuses `targeting_scheme="fold"`, so the fluctuation is
one pooled coefficient on the stacked out-of-fold rows, as Appendix D outlines. Two divergences
remain.

| divergence | what ships | what Appendix D outlines |
| --- | --- | --- |
| the final average | the stacked whole-sample plug-in, because `CTMLE` refuses `cv_evaluation=True` | the $(1/V)\sum_v$ fold average. With fixed $V$, near-balanced unweighted folds and bounded predictions, the difference is $O(V/n)$ |
| the fluctuation dimension | a joint fluctuation with one column for each arm, from which the means, ATE, RR and OR follow | one signed coefficient for the binary ATE |

**Why the shipped fit is not a natural extension.** The package fits one shared multinomial
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
| one shared multinomial | one categorical fit on `K` estimated columns supplies every arm's clever covariate, so an inconsistent column for one arm enters every other arm's mechanism |
| vector target and simultaneous inference | the joint covariance and the simultaneous critical value, not the per-arm variance alone |
| uniformity | the estimator is superefficient by design, so a pointwise limit law does not give locally uniform coverage |
| transport beyond the source law | missing outcomes need the response-mechanism expansion. Fixed weights need a weighted empirical-law result, and estimated weights a first-stage contribution. Clusters and strata need dependence- and stratum-specific expansions. Repeats need the median and split-dispersion aggregation |

**Acceptance.**

- A result covers the exact cross-fitted multi-arm construction, and it separates a fixed target
  from a target conditional on the learned design.
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

Keep fold-targeted and repeated-split missing-outcome fits refused. The natural-course mean, PAR, PAF,
and the arm-indexed means and contrasts refuse both variants before any learner call. No read source gives
a direct interval result for either composition.

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
target needs its own source audit and contract. An in-sample C-TMLE fit with missing outcomes
takes the status of its path, which [F18](#f18-selector-path-c-tmle-inference) and
[F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) hold.

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
| stacked arm-indexed targets, PAR, and PAF with baseline strata | the finite-partition step that the stacked natural-course mean ships, with a witness and a registered cell. The scalar stacked `ey_obs` admits `strata=` |

No admitted fit reaches the multi-draw branch of `missingness_tilt`
(`src/cleverly/sensitivity/missingness.py`), because no admitted composition fits repeated draws
with missing outcomes. A repeated-split contract for missing outcomes makes that branch live, and
it needs its own witness then.

### F22. Grouped cross-fitting beyond point-treatment TMLE

The package draws whole-cluster outer folds for the ordinary cross-fitted point-treatment TMLE and
DR-TMLE. C-TMLE refuses `id=` at every `cross_fit` setting, and this item holds that refusal.

| composition | what ships | what a result must supply |
| --- | --- | --- |
| C-TMLE with `id=` | refused at every `cross_fit` setting (`CTMLE._resolve_estimands_for_data`) | a split law for the selection folds and the nested selection folds under clustering, and the cluster-robust variance of the candidate the search stops at. [F18](#f18-selector-path-c-tmle-inference) is open for iid rows, so a clustered result needs that one first |

The other grouped compositions take the cluster as the sampling unit, and the
[natural-extension verdicts](technical-reference/natural-extension-verdicts.md) qualify them as
natural extensions. [X24](#x24-clustered-intervals-at-unequal-cluster-sizes-and-at-few-clusters)
holds the cross-fitted interval at unequal cluster sizes and the $t$ reference at few clusters.
[X25](#x25-cross-fitted-clustered-longitudinal-tmle) holds cross-fitted longitudinal TMLE with
`id=`.

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
mediation, and a missing treatment on `CTMLE`, where each variant lacks evidence. Ordinary-TMLE
implementations do not establish collaborative or doubly robust inference for these compositions.
An observational missing outcome and a missing treatment on `DRTMLE` ship through the
[composite indicator](technical-reference/dr-tmle/theorem.md#observational-missing-data-the-composite-indicator).
On the composite, `evaluation=` and `reduced_crossfit="nested"` stay refused here. Both carry one
fold-free treatment mechanism, and the composite has up to three factors. A composite `DRTMLE`
fits `strata=` with one block per stratum, as every `DRTMLE` route does
([baseline strata](technical-reference/dr-tmle/supported-estimands.md#baseline-strata)).

C-TMLE with cross-fitted arm-indexed missing outcomes is refused before any learner call.
In-sample C-TMLE with missing outcomes still fits, and it reports no interval: a selector path
takes `working_mechanism_plugin`, and `strategy="oat"` takes `generated_design_plugin`.

Each C-TMLE extension needs its target-specific collaborative score and selection-risk contract.
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
| 4 | a registered study with one-wrong and both-wrong coverage cells, paired with `drtmle` 1.1.2 through the constant-treatment form (`A = rep(1, n)`, `W = cbind(A, W)`, `DeltaY = !is.na(Y)`), after a probe confirms that `drtmle` accepts a one-level `A` |

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

## Reading a gap correctly

Not every absence is missing package functionality. The refusal taxonomy in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal) distinguishes an unimplemented
well-posed feature from a different causal question and a method that would be wrong by
construction. Only the first belongs on this roadmap.
