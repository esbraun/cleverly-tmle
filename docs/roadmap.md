# Roadmap

This is the single planning contract for `cleverly`. It contains proposed work only. Implemented
capabilities belong in the [user guide](user-guide/index.md), scientific contracts in the
[technical reference](technical-reference/index.md) and [DR-TMLE contract](technical-reference/dr-tmle/index.md), validation results in
[evidence manifest](technical-reference/evidence.md), and cross-module standing decisions in the
[architecture invariants](architecture-invariants.md).

The main grid is one binding sequence. Complete lower numbers before higher numbers. Items with no
published theory do not enter this sequence.

## Remediation

The examples are executable, but their review exposed gaps in the public study record and in
post-fit coverage. Deliver the rows in priority order. Complete every row before main-roadmap
priority 1. A new capability still needs its own contract and evidence, even in this queue.

The "next action" column states the remediation work. It is not a readiness label.

The 2026-09-18 source audit found no result for the shipped data-dependent outer folds or for the
continuous outcome scale. That work is delivered. The
[fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) hold
the audit record and the rules the package now ships.

This project then regenerated sixteen registered studies under those rules. Six property cells go
red for the first time, across five of those studies. Every cell that was already red stays red.
RM18 carries both sets. Each affected study publishes its red cell and that cell's interval under a
`reporting` policy, so no verdict is hidden and no margin moved.

The 2026-09-13 review of the example notebooks exposed RM11 to RM16. These rows correct defects in
shipped estimators, diagnostics, and messages. Each detail section names its source evidence and
the probe that measured it.

| priority | item | next action | problem | details |
| ---: | --- | --- | --- | --- |
| 0.1 | Red property cells after the fold, scale and law changes | derive the missing selector-path and generated-design results, settle the fold-local longitudinal targeting question, and publish each red cell with its interval until then | six property cells across five registered studies go red for the first time in this pull request, and the cells that were already red stay red | [RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) |
| 0.2 | Sensitivity bounds outside their derivation | refuse every omitted-variable operation on DR-TMLE, C-TMLE, and missing-outcome fits, refuse the standardized E-value conversion on missing-outcome fits, and correct the refusal messages for the other parameter axes | the bound runs where no derivation covers it, and on DR-TMLE and C-TMLE fits it understates the bias | [RM11](#rm11-sensitivity-bounds-outside-their-derivation) |
| 0.3 | Collaborative intervals at an inconsistent working mechanism | label every collaborative interval in its output, and correct the path-risk docstrings | the curve at an intercept-only working mechanism gives a standard-error ratio of 0.844 and a coverage of 0.92 over 300 draws | [RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism) |
| 0.4 | Estimated MSM projection weights | require a declaration that a projection weight is known, and refuse an estimated weight before the fit | a callable that closes over estimated weights fits without a message and reports a standard error that is too small | [RM13](#rm13-estimated-msm-projection-weights) |
| 0.5 | Intervention refusals at identification | refuse mixed intervention kinds in `CausalStudy.identify`, and name the typed estimands in each message | a mixed request passes identification and then fails at estimation, once with an `AttributeError` | [RM14](#rm14-intervention-refusals-at-identification) |
| 0.6 | Calibration-slope warning rule | replace the fixed band with a rule that a registered calibration study supports | the band flagged 14 of 40 fits of a correctly specified weak-signal propensity model | [RM15](#rm15-calibration-slope-warning-rule) |
| 0.7 | Summary and error-message accuracy | correct four display surfaces and one data error message, and add a fingerprint-only protocol option | each surface omits, misstates, or repeats a fact that the fit records | [RM16](#rm16-summary-and-error-message-accuracy) |

Use five delivery groups for these seven rows and investigations. Keep each item's acceptance
criteria separate inside its group.

| delivery group | items | shared boundary |
| --- | --- | --- |
| red property cells | RM18, and the F18, F19 and F24 derivations it waits on | one decision about what a failing positive cell is evidence of, and three derivations that would close it |
| sensitivity refusals | RM11 and the F5 refusal boundary | one capability route and one family of derivation messages |
| collaborative inference | RM12 and the F18 audit | one decision about the selected working mechanism and selector |
| pre-fit declarations | RM13 and RM14 | refuse unsupported requests before any nuisance fit |
| diagnostic reports | RM15 and RM16 | one assessment and summary surface, with one documentation pass |

Each group holds consecutive priorities, so the group order is the priority order. Deliver the
items inside a group in priority order.

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
tracks them. It also records the retained follow-ups of the arm-indexed stacked contract and two
unexamined sibling gaps.

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
| Stochastic categorical policies at a longitudinal node | longitudinal identification, influence function, remainder, and interval conditions for a distribution-valued policy | deterministic categorical regimens only | [F1](#f1-stochastic-categorical-policies-at-a-longitudinal-node) |
| Targeted bootstrap inference | a construction that defines what is fixed, resampled, refitted, and retargeted, plus the sampling law of the interval | existing bootstrap inference is not this procedure | [F2](#f2-targeted-bootstrap-inference) |
| Longitudinal sensitivity-bound estimation | sample estimation of the bound functionals, a specialized algorithm, and sampling inference | no sensitivity bound on a longitudinal fit | [F16](#f16-longitudinal-sensitivity-bound-estimation) |
| Additional longitudinal estimands | target-specific identification, influence function, targeting construction, and inference conditions | existing end-of-study, survival, competing-risk, and MSM targets only | [F3](#f3-additional-longitudinal-estimands) |
| Multi-arm missing-outcome DR-TMLE | joint and contrast inference across arms, and one treatment mechanism compatible with a separate logistic tilt for each arm | binary randomized treatment only | [F4](#f4-multi-arm-missing-outcome-dr-tmle) |
| Other refused C-TMLE and DR-TMLE compositions | composition-specific score, reduced regressions, correction, remainder, and rate conditions | named pre-fit refusals and conditional-on-weight intervals remain | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| Selector-path C-TMLE inference | an influence function and covariance after the shipped data-adaptive stopping-index selection | ordinary EIF plug-in covariance that treats the selected candidate as fixed | [F18](#f18-selector-path-c-tmle-inference) |
| Outcome-adaptive C-TMLE generated-design inference | exact scalar expansions for the shipped joint binary fit and a multi-arm vector extension of the paper-backed fold-local construction | ordinary adaptive-propensity EIF covariance with a proved binary scalar construction and open joint-target extensions | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| Missing-outcome attributable effects | a direct observed-data derivation for the joint natural-course and reference-intervention means, their remainder, and attributable-effect inference | complete-data PAR and PAF, one iid MAR natural-course mean, and arm-specific missing-outcome means remain separate | [F20](#f20-missing-outcome-attributable-effects) |
| Other missing-outcome CV-TMLE variants | a direct interval result for fold-specific targeting and for the fixed-repeat median and split-dispersion report after CV-TMLE targeting | the package supports the ordinary natural-course estimator, the stacked natural-course estimator for a binary outcome, and the stacked arm-indexed means and contrasts. Each stacked estimator uses one repeat and pooled targeting. F21 also records their follow-ups and two sibling gaps | [F21](#f21-other-missing-outcome-cv-tmle-variants) |
| MNAR and incremental-intermediate compositions | identification and influence-function results for the exact compositions | point-treatment sensitivity and implemented interventions remain separate | [F6](#f6-mnar-and-incremental-intermediate-compositions) |
| Joint point-treatment parameter axes | a targeting and inference result for one fit that carries an MSM projection together with a regime, shift, or incremental intervention, including its joint score and covariance | single-axis point-treatment fits only | [F17](#f17-joint-point-treatment-parameter-axes) |
| Time-respecting cross-fitting | dependence and split-specific TMLE inference for blocked-temporal or rolling-origin folds | iid and grouped cross-fitting only | [F7](#f7-time-respecting-cross-fitting) |
| Fold-local targeting of the longitudinal recursion | a weak-convergence result for a fluctuation fitted inside each training fold of a sequential regression | the published theorem certifies a pooled all-row fluctuation, and the shipped cross-fitted longitudinal TMLE fits each fold's fluctuation on that fold's training rows | [F24](#f24-fold-local-targeting-of-the-longitudinal-recursion) |

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

Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022) bound the omitted-variable bias of a
linear functional of the outcome regression. The bound is $\sqrt{\sigma^2 \nu^2}$ times a strength
factor. Here $\nu^2$ is the second moment of the Riesz representer for the declared adjustment set.
An estimate of $\nu^2$ is only as good as the fitted representer. The bound has one treatment-side
strength and no response mechanism.

`sensitivity_elements` builds $\sigma^2$, $\nu^2$, and the representer from the nuisances of the
reported repeat (`src/cleverly/sensitivity/omitted_variable.py:193-281`). The bounds, the
benchmark, the robustness value, and the contour all call it (`:526`, `:762`, `:842`, `:866`). The
assessment declares these operations for `tmle`, `collaborative_tmle`, and `drtmle`
(`src/cleverly/assessment.py:312`, `:3414-3429`). A probe on a DR-TMLE fit reports each capability
as available and returns a robustness value with no warning.

| defect | mechanism | evidence | consequence for users |
| --- | --- | --- | --- |
| DR-TMLE $\nu^2$ | the default estimator $E[2 m(\hat\alpha) - \hat\alpha^2]$ equals $\nu_0^2 - \lVert \hat\alpha - \alpha_0 \rVert^2$ by the Riesz identity. It is low when the fitted mechanism is wrong, which is the case DR-TMLE guards against. Only the bound's standard error adds the corrected curve (`:536-537`, `:561-564`), and no derivation covers that sum | on the stored `dr-tmle` notebook draw, the reported $\nu^2$ is 4.34. The sample value of $E[1/g_0 + 1/(1 - g_0)]$ is 7.75 | the bounds are too narrow, and the robustness value is too large |
| C-TMLE $\nu^2$ | a C-TMLE fit holds the selected working mechanism in `repeat.nuisance`. The intercept-only representer $A/p - (1 - A)/(1 - p)$ equals $E[\alpha_W \mid A]$, so its second moment cannot exceed that of $\alpha_W$. $\sigma^2$ still comes from a regression on every covariate | on `make_instrument(n=2000, seed=44)`, the plain fit gives $\nu^2 = 11.77$ and a robustness value of 0.229. The C-TMLE fit gives 4.00 and 0.381 | the collaborative robustness value is optimistic by construction, and $\sigma^2 \nu^2$ belongs to no single conditioning set |
| refusal reason for a non-arm axis | the message says that a fit "whose counterfactuals are not arms does not have" a Riesz representer (`:178-185`) | a regime mean, a shift mean, and a point-treatment MSM coefficient are linear functionals of the regression, and each has a representer | a user reads a missing implementation as a mathematical limit |
| suggestion for an MSM coefficient | a name outside `LINEAR_ESTIMANDS` receives "For a risk ratio or odds ratio use sensitivity.evalue()" (`:164-168`) | a probe on `msm[a]` returns that text. `evalue` refuses an `msm` target (`src/cleverly/sensitivity/evalue.py:342-346`) | the message sends the user to a second refusal |
| missing-outcome fits | `sensitivity_elements` has no response-mechanism check. The representer carries the response weight through `nuisance_bound` (`:216`), $\sigma^2$ averages respondents only (`:245`), and the strength factor has no response-side term. The assessment offers the bounds, the robustness value, and the E-value on a `PointTreatment(missingness=...)` fit | a probe on a simulated fit with a response indicator reports all three capabilities as available and returns a robustness value of 0.040. The default $\nu^2$ estimator was not positive there, so the code fell back to `"plugin"` without a warning (`:258-259`, marked unreachable for coverage). The `survey-nonresponse` notebook prints a bias-adjusted interval of [0.9567, 1.414] | a user reads a bound that no derivation covers for a fit that models response |
| standardized E-value on a missing-outcome fit | the Gaussian conversion divides by the standard deviation of the observed outcomes (`src/cleverly/sensitivity/evalue.py:449-460`). Under missing at random, the respondents' standard deviation estimates a different quantity from the population standard deviation | the same probe returns an E-value of 1.97 with `sd(Y)` computed from respondents | the standardized scale belongs to the respondents, not to the population the estimate targets |
| documented estimator names | the `omitted_variable_bounds` docstring lists `"auto"`, `"analytic"`, and `"riesz"` (`:518`) | the code accepts `"auto"`, `"doubly_robust"`, and `"plugin"`, and raises `ValueError` for any other value (`:251-266`) | a documented argument fails |

Apply these corrections:

1. Refuse every omitted-variable operation on a `drtmle` fit, a `collaborative_tmle` fit, or a fit
   with a response mechanism, before any computation. Put the refusal in `sensitivity_elements`,
   so that all four entry points share it. Mark the matching assessment rows `unavailable` with the
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

The witnesses must fail when a component is wrong:

- a pre-fit test pins each refusal and its message on a DR-TMLE fit, a C-TMLE fit, and a
  missing-outcome fit;
- a missing-outcome test pins the refused standardized E-value and the retained ratio E-value;
- a test that a nonpositive $\nu^2$ raises, rather than returning the plug-in value;
- a nonzero witness pins $\nu^2_{\text{intercept}} < \nu^2_{\text{full}}$ on one instrument law.
  A mutation that removes the C-TMLE refusal then reports the smaller value, and the test fails;
- a finite-support law with a wrong mechanism and a known $\nu_0^2$ shows the DR-TMLE shortfall
  that the refusal prevents;
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

The path record has a second defect. The `CTMLESelection.train_risk` docstring
(`src/cleverly/estimators/ctmle.py:309-313`) and the module docstring (`:39-41`) say that the
in-sample risk does not increase. `_forward_path` (`:1337-1348`) takes one more fluctuation step
when no addition lowers the risk. If the next addition still raises it, the path adds that covariate
anyway. The `collaborative-tmle` notebook output shows `risk` rising from 6.59305 to 6.60011.

Apply these corrections:

1. Print a label with every collaborative interval in the result summary, in `to_frame`, and in
   the assessment. The label says that the curve is the EIF at the selected working mechanism. It
   also says that no result shows this curve is the estimator's influence curve when that mechanism
   is not consistent for the treatment law.
2. Decide in the [F18](#f18-selector-path-c-tmle-inference) audit whether the interval is refused
   or kept with that label. Do not add an ad hoc variance term in the meantime.
3. Audit `_forward_path` against van der Laan and Gruber (2010) and the pinned R `ctmle`. If the
   forced addition matches both, correct the two docstrings. If it does not, change the path and
   regenerate the C-TMLE studies whose selections it moves.

The witnesses must fail when a component is wrong:

- a discrete-strategy fit with only the intercept-only candidate, on a finite-support law with a
  correct outcome regression. The test compares the reported curve variance with the exact
  variance of that fixed-candidate estimator, and the two must differ by a stated margin;
- a greedy path on a law where `train_risk` rises, which pins the documented behavior;
- a registered coverage cell on the instrument law, with its thresholds fixed before the final
  run.

### RM13. Estimated MSM projection weights

The [MSM technical reference](technical-reference/msm-projections.md#variations) lists a
projection weight derived from the estimated mechanism as refused. The same row says that such a
weight "does not fail" and gives a standard error that is too small. The code checks only that
`weights` is callable (`src/cleverly/msm.py:373-378`), and a coverage pragma marks that branch as
unreachable.

A callable receives an arm label and a covariate frame, and it can close over any estimate. A
callable over sample shares or a fitted mechanism therefore fits with no message. The weight is
then a functional of $P$, and the influence curve omits its pathwise derivative. The reported
standard error is too small, and the reference calls the composition refused.

The package cannot inspect a closure, so make the status of the weight a declaration:

1. Require an explicit known-weight declaration on `MSM` whenever `weights` is set. Refuse
   `weights` without it before any nuisance fit. `PointTreatment.weights_estimated` is the
   precedent for declared observation weights (`src/cleverly/study.py:407`).
2. Refuse a weight declared as estimated. The message names the missing pathwise-derivative term.
3. Correct the reference row, so that it describes the declaration and both refusals.

The witnesses must fail when a component is wrong:

- a pre-fit test pins both refusals and their messages, and a spy learner shows that no nuisance
  fit ran;
- a repeated-sampling control with weights equal to the sample arm shares. It records a reported
  standard error below the sampling standard deviation, which is the reason for the refusal.

### RM14. Intervention refusals at identification

The typed estimands do not check the kinds of their elements. `RegimeMean.regimens` and
`RegimeContrast.regimens` are typed `Any` (`src/cleverly/study.py:953`, `:991`), and
`IncrementalMean.interventions` and `IncrementalEffect.interventions` declare
`Sequence[Incremental]` with no runtime check (`:1062`, `:1104`). The refusal
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

Apply these corrections:

1. Check the element kinds when `CausalStudy.identify` builds the functional. Refuse a `Shift` or an
   `Incremental` in a regimen set. Refuse any element other than `Incremental` in an incremental
   estimand.
2. Name the typed estimand for each kind in the message. Cite
   [F17](#f17-joint-point-treatment-parameter-axes) for a joint request.
3. Keep `as_interventions` as a second guard for direct estimator use, and name the typed estimands
   there too.

The witnesses must fail when a component is wrong:

- a pre-fit test for each row above pins the refusal, its message, and its raise at `identify`;
- a spy learner shows that no nuisance fit ran;
- a mutation that removes the identification check makes the test fail at `identify`, rather than
  pass at `estimate`.

### RM15. Calibration-slope warning rule

`NuisanceDiagnostics.findings` warns when a calibration slope falls outside [0.7, 1.4]. The finding
says that the model "biases the weights" (`src/cleverly/validation/nuisance.py:513-517`). The
assessment turns any finding into a `nuisance_models` warning (`src/cleverly/assessment.py:2638-2645`).

The band ignores the spread of the true probabilities. With a weak signal, noise in the out-of-fold
coefficients spreads the predictions more than the truth does. A correct model then gives a slope
below 1. The finding also asserts an effect on the weights that the rule does not measure.

| law | fits | result |
| --- | --- | --- |
| point treatment, $n = 2000$, $\operatorname{logit} g_0 = 0.15 W_1$, correct logistic learner, three folds | 40 seeds | mean slope 0.647, median slope 0.738, mean AUC 0.537. The rule flagged 14 fits |
| longitudinal censoring at node 2, `make_longitudinal(n=8000)`, correct logistic model, a separate three-fold split | 60 seeds | out-of-fold slope mean 0.767, standard deviation 0.114 |

The longitudinal nuisance report applies no band (`assessment.py:2587-2597`), so the second row
raises no warning. Its `calibration_slope` column also holds two statistics under one name. Binary
rows use a logistic recalibration slope, and pseudo-outcome rows use a linear regression slope
(`nuisance.py:845-862`).

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
| longitudinal `RegimeMean` result summary | prints `reference: always`, although the result keeps no contrast | `src/cleverly/study.py:2872-2878` drops the contrasts. `src/cleverly/longitudinal/estimator.py:598-605` prints the reference unconditionally. A probe prints `reference: always` beside `ey_regimen[always]` and `ey_regimen[never]` only | clear the reference when the study keeps no contrast, and print the line only for a contrast |
| longitudinal identification summary | the `adjustment/history` line lists only the baseline covariates | `study.py:2268` sets `adjustment=tuple(design.baseline)`, and `:2574` prints it. A probe prints `['W1', 'W2']` for a design with `L2` at node 2 | print the history for each node |
| DR-TMLE result summary | names neither DR-TMLE, the guard, nor the reduction | `src/cleverly/estimators/base.py:1180` fixes the header, and `config.describe()` (`:179-200`) has no method line. A probe finds `fitted_method` equal to `drtmle` and none of `drtmle`, `DR-TMLE`, or `guard` in the summary | add the method, the guard, and the reduction |
| regime support table | `min g`, `max ratio`, and `ratio effective n` use the mechanism before truncation. `score load` uses the truncated weights. The table does not say so | `src/cleverly/interventions/support.py:276-294` and the header at `:244-260`. In the `interventions` notebook, offer to all prints a `min g` of 0.001463, below the fit's bound of 0.0114 | label the basis of each column. The source chooses the untruncated mechanism on purpose, so this is a display gap and not an estimator defect |
| identification and result summaries | each summary reprints all 11 protocol lines, so a notebook that prints the protocol, the identified effect, and the result shows the record three times | `StudyProtocol.summary_lines` returns 11 lines (`src/cleverly/protocol.py:180-200`). `IdentifiedEffect.summary` and `IdentifiedEffect.summary_lines` append them (`src/cleverly/study.py:2566-2578`, `:2587-2597`), and the result summary appends `summary_lines` (`src/cleverly/estimators/base.py:1190-1191`). The `dr-tmle` notebook shows the block three times | add a summary option that prints only the `causal study protocol: schema N; fingerprint` line. Keep the full record as the default, so that a summary read alone stays complete |
| missing-outcome `DataError` | tells a study user to pass `delta=<column>` | `src/cleverly/data/validate.py:311`. `PointTreatment` names the field `missingness` (`src/cleverly/study.py:403`). A probe through `CausalStudy` returns the `delta=` text | name `missingness=` for a study design. Keep `delta=` only where the low-level `CausalData` constructor raises the error |

Each correction needs a unit test that fails without it. Three tests are nonzero witnesses. A
`RegimeContrast` summary keeps its reference line. A design with a time-varying covariate prints it
at its node. A truncated fit labels a `score load` that differs from its `ratio effective n`.

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

#### The cells that went red in this pull request

| study | cell | statistic | before | after | margin | reported attribution |
| --- | --- | --- | --- | --- | --- | --- |
| [selector-based point-treatment C-TMLE](technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md) | `selector_necessity/collaborative`, positive | standardized bias | 0.1412 | 0.2173 | the 99% bias upper endpoint is 0.0037 against an equivalence margin of 0.0030, which is 0.3086 against 0.25 on the standardized scale | the external 2x2 report attributes most of the move to the bounded law and a little to the fold policy |
| the same study | `type_i_error/sharp_null`, positive | rejection rate | 0.0275 | 0.0700 | the 99% upper endpoint is 0.1095 against the 0.10 ceiling. Coverage is 0.9300, whose 99% lower endpoint is 0.8905 against the 0.90 floor | the external 2x2 report attributes the move to the bounded law alone |
| [selector-based multi-arm C-TMLE](technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md) | `root_n_and_efficiency/n_500`, positive | exact 99% coverage lower endpoint | 0.9057 | 0.8965 | the endpoint against the 0.90 floor. Coverage itself moved 0.9425 to 0.9350 | not isolated. One covered replication of 400 separates the two endpoints |
| [DR-TMLE for binary complete data](technical-reference/method-evidence/canonical-dr-tmle.md) | `double_robust_contraction/rate_outcome_correct`, positive | 99% contraction-slope interval | -3.52 to -0.06 | -3.04 to +0.12 at the regeneration, and -1.5345 to -0.4823 in the committed artifact | the interval must stay below zero | not isolated. The arm's bias is 0.0036 at the first rung. A declared rung design now resolves this cell, and "What the two readings found" gives its interval |
| [multi-arm point-treatment DR-TMLE](technical-reference/method-evidence/multi-arm-dr-tmle.md) | `root_n_and_efficiency/n_500`, positive | exact 99% coverage lower endpoint | 0.9087 | 0.8965 | the endpoint against the 0.90 floor. Coverage itself moved 0.9450 to 0.9350 | not isolated. The same cell and the same new endpoint as the multi-arm selector row above |
| [cross-fitted end-of-study longitudinal TMLE](technical-reference/method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md) | `crossfit_overfitting/cross_fitted_ltmle`, positive | reported SE over empirical SD | 1.172522, 99% upper 1.196518 | 1.176650, 99% upper 1.201555 | the shared `se_ratio_sanity` ceiling of 1.2000, exceeded by 0.001555 | the external 2x2 report attributes the move to the fold policy alone |

Two of those rows carry a second cell with them. `selector_necessity/empty_control` passes its own
control rule, and it is red through the family's joint clause.
`crossfit_overfitting/in_sample_control` passes its own rule at a standard-error ratio of 0.353193
against a 0.75 ceiling, and the pair's coverage gain passes at a 99% lower endpoint of 0.453375
against a 0.15 floor. Both families are red because their positive arm is.

#### The cells that were red before it

| study | cell | state after the regeneration |
| --- | --- | --- |
| selector-based multi-arm C-TMLE | the greedy and ordered `selector_necessity` RMSE ratios | 0.4470 to 0.4415, and 0.3736 to 0.3781. Both stay red, and neither moved by more than 0.006 |
| the same study | `selector_necessity/discrete` | still red. The discrete selector still stops at the empty candidate, so its RMSE ratio is 1 |
| the same study | `interval_calibration/correctly_specified` | still red |
| the same study | `type_i_error/sharp_null` | 0.0700 to 0.0625, whose 99% upper endpoint is 0.1004 against the 0.10 ceiling. Still red, and closer |
| [outcome-adaptive multi-arm C-TMLE](technical-reference/method-evidence/outcome-adaptive-multi-arm-c-tmle.md) | the `generated_design` pair | still red. The oracle interval runs 0.9797 to 1.1151, and the paired deficit runs -0.0240 to 0.0016 |
| [cross-fitted weighted end-of-study longitudinal TMLE](technical-reference/method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | `double_robustness/static__both_wrong`, a control | still red. Its standardized bias moved -0.2983 to -0.2988 |

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
| both `n_500` coverage endpoints | a registered `fold_policy` pair, two split policies over 8,000 paired draws of each study | a boundary resolution at 400 replications. Each study now covers 374 replications of 400, where the two covered 377 and 378 before. Neither paired coverage interval reaches the gate-relevant 0.005. "What the two readings found" gives both intervals, and it records the resolution the DR-TMLE instrument missed. The same cell moved the other way on outcome-adaptive multi-arm C-TMLE, where it turned green at 0.9057 |
| `double_robust_contraction/rate_outcome_correct` | a declared rung design, run at 2,400 replications on each outer rung | three rungs of a small quantity gave a wide slope, and the width was Monte Carlo error at the top rung. The raised rungs put the interval below zero. That run also moved the interpreter and SciPy, so the budget and the environment are confounded |

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

#### The open question each cell belongs to

| cells | open question |
| --- | --- |
| `selector_necessity/collaborative`, `selector_necessity/empty_control`, and the standard-error ratio reversal | [F18](#f18-selector-path-c-tmle-inference). No result derives an influence curve after the shipped stopping-index selection. The population one-step remainder is exactly zero at the nuisance limits on both laws, so the residual is the selector's stopping behaviour rather than a nuisance rate |
| the same reversal, again | [RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism). The registered row and RM12's own probe now point the same way, so RM12's labelling work covers this study's reported interval |
| the `generated_design` pair | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference). The paired standard-error deficit under an estimated outcome regression has no derivation |
| `crossfit_overfitting/cross_fitted_ltmle` | [F24](#f24-fold-local-targeting-of-the-longitudinal-recursion). No source read here certifies a fluctuation fitted inside each training fold. The fold-local longitudinal targeting question below states what the cell measures |

#### The fold-local longitudinal targeting question

This question had no item of its own, and the retired RM17 row held it.
[F24](#f24-fold-local-targeting-of-the-longitudinal-recursion) now carries its contract, its
sources and its four remediation options. RM18 keeps the measured cell.

Díaz, Williams, Hoffman and Schenck (2023), Section 5.2 and Theorem 3, define a random
near-balanced row partition for a longitudinal TMLE, and the package now draws exactly that
partition. The theorem's targeting construction is a pooled all-row fluctuation after nuisance
prediction. The shipped estimator instead fits fold `k`'s epsilon on fold `k`'s training rows
(`src/cleverly/longitudinal/sequential.py:51-56`, `:1377-1412`, `:919`), as the authors' own `lmtp`
1.5.4 does.

Upstream commit
[`9996b04`](https://github.com/nt-williams/lmtp/commit/9996b04dcbb3ae0b1ef8862097c36d95e9f2fcf9)
calls that behavior a bug because the EIF was not mean zero. It now fits the fluctuation on
validation rows, which the forthcoming 1.5.5 release records. The point-treatment
`targeting_scheme="fold"` variant also fits its epsilon on validation rows
(`src/cleverly/estimators/tmle.py:3033-3050`). Training-fold and validation-fold updates are
different departures from a pooled update. Theorem 3 certifies neither one.

The end-of-study cell above is where that gap shows as a number. One reading is that a fold-local
update leaves a stitched score. That score would be a mean-zero residual rather than a solved
equation, so the reported standard error would run above the sampling spread. No result read here
establishes that reading. The remediation is one of F24's four options. It is not a larger
budget.

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
supplement holds each fold's fit fixed given its training rows. That condition covers untargeted
fold regressions followed by one pooled fluctuation per node. It does not cover a pooled epsilon
carried back into later fold regressions. `tests/unit/test_pooled_longitudinal_targeting.py`
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

#### What this row asks for

| work | acceptance |
| --- | --- |
| an inference result for the shipped selector path | an influence curve derived after the stopping-index selection, and a registered study whose `selector_necessity` and `type_i_error` cells pass their existing margins at their existing budgets |
| an inference result for the generated-design deficit | a derivation of the paired standard-error deficit under an estimated outcome regression, and the `generated_design` pair passing at its existing budget |
| a targeting result for the fold-local longitudinal recursion | a theorem for the shipped update, a validation-fold update, a pooled update, or the SDR estimator of [X4](#x4-sequential-doubly-robust-longitudinal-estimation). The selected route needs registered evidence, and `crossfit_overfitting/cross_fitted_ltmle` must sit inside the shared `se_ratio_sanity` ceiling at 8,000 draws. [F24](#f24-fold-local-targeting-of-the-longitudinal-recursion) states what each option needs |
| a reading of the two `n_500` coverage endpoints | delivered. A registered fold-policy diagnostic reads both endpoints as a boundary resolution. "What the two readings found" gives the numbers |
| a reading of the DR-TMLE contraction slope | delivered. A declared rung design resolves the slope, and its interval now sits below zero. "What the two readings found" gives the numbers |

#### The two readings, declared before they run

This subsection is the declaration the last two acceptance rows ask for. It precedes the run that
measures either reading.

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

Both readings ran once, under the rules the subsection below declared before them. Each publishes
what it produced.

Both runs also moved the environment. SciPy moved from 1.17.1 to 1.18.0, and Python moved from
3.11.13 to 3.13.7, on the runs that produced these rows. Each study's `manifest.json` records the
versions its committed rows came from, and its git history records the versions they replaced. A
difference between a committed row and the row it replaced therefore carries the environment move
as well as the design change. Nothing here separates the two.

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
0.004906. The environment moved on the same run, so the measured width carries that move too.

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

#### What the source search found for the first three asks

The first three asks each need a derivation. The [Eligibility](#eligibility) rule asks this
repository to locate published theory, and not to create it here. A 2026-09-20 source search read
the sources below and found no result for any of the three. A 2026-09-21 follow-up also checked the
latest upstream `lmtp` correction. The source audit is complete, but these
three acceptance rows remain open. The investigation row named below carries each contract.

| ask | verdict | where the contract lives |
| --- | --- | --- |
| an inference result for the shipped selector path | no published result | [F18](#f18-selector-path-c-tmle-inference) |
| an inference result for the generated-design deficit | a proved binary scalar result, and no result for the shipped construction | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| a targeting result for the fold-local longitudinal recursion | no result for the shipped update | [F24](#f24-fold-local-targeting-of-the-longitudinal-recursion) |

| ask | sources read, and their exact limits |
| --- | --- |
| the shipped selector path | van der Laan and Gruber (2010), *IJB* 6(1), DOI 10.2202/1557-4679.1181: Theorem 4 assumes the expansion that defines the adaptive-mechanism contribution, and Section 4.3 records cross-validation over-selection as an open irregularity. Gruber and van der Laan (2010), *IJB* 6(1), DOI 10.2202/1557-4679.1182, apply the method and state no post-selection inference result. Ju, Chambaz and van der Laan (2018), arXiv:1804.00102: Theorem 1 permits an extra contribution along a twice-differentiable continuous nuisance path for a binary scalar target, and Lemma 2 zeroes it in a correct-outcome product-rate regime. A discrete stopping index is not a differentiable path. Ju et al. (2019), *SMMR* 28(6), DOI 10.1177/0962280217729845, Section 7.4, forms intervals from the ordinary EIF, which is the fixed-candidate curve. Adaptive debiased machine learning, arXiv:2307.12544v2, is the closest positive route, and it needs the working model to approximate a fixed, nonrandom oracle model. Nobody has proved that for a global depth chosen from nested targeted-loss folds. Leeb and Pötscher (2006, *AoS* 34(5); 2008, *ET* 24(2)), Loftus (arXiv:1511.08866), and Markovic, Xia and Taylor (arXiv:1703.06559) supply no transfer, because this selector has no Gaussian quadratic reduction and no randomized or jointly Gaussian criterion-and-target limit. The multi-arm obligation has no published treatment at all |
| the generated-design deficit | Benkeser, Cai and van der Laan, *Statistical Science* 35(3), DOI 10.1214/19-STS735, preprint arXiv:1901.05056: Theorem 1 proves, for the binary treatment-specific mean, the expansion with the ordinary adaptive-propensity curve and no separate generated-design term. Theorem 1's estimator is a full-sample procedure, and Section 3.1 supplies a cross-validated variance alone. The cross-fitted point estimator appears in Appendix D, where the authors call its proof completely analogous to Zheng and van der Laan (2011) and outline it. Appendix D's binary ATE, which uses both arm predictions with one signed coefficient, is an algorithm with no theorem. No result covers a shared-multinomial vector extension. Four sources do not close it. Ju, Benkeser and van der Laan (2020), *Biometrics* 76(1):109-118, DOI 10.1111/biom.13121, build a different construction, in which outcome information enters through HAL penalty weights. Shortreed and Ertefaie (2017), *Biometrics* 73(4):1111-1122, DOI 10.1111/biom.12679, select variables and prove no inference theorem. Escanciano and Pérez-Izquierdo (2023), arXiv:2301.10643, remove the indirect first-step effect, and the direct effect of learning the generated regressor remains. DOPE, arXiv:2402.12980v2, centers Theorem 4.3 at a data-adaptive target conditional on a representation learned on an independent sample, adds a fixed-target delta-method variance in Proposition 4.4, and leaves the cross-fitted proof open in its appendix |
| the fold-local longitudinal recursion | Díaz, Williams, Hoffman and Schenck (2023), Section 5.2, Step 3, journal page 852, fits the fluctuation "using all the data points in the sample", and Theorem 3, page 853, certifies that pooled update. Zheng and van der Laan (2011) and Levy (2018), arXiv:1811.04573, both describe the targeting step as a pooled regression over validation folds. Chernozhukov et al. (2018) certifies the cross-fitted orthogonal moment, and not a plug-in of a train-fold-targeted regression. Williams and Díaz (2025), *Observational Studies* 11(3):365-367, correct Assumption 2 and the positivity statement, and say nothing about cross-fitting or targeting. Upstream `lmtp` commit `9996b04` calls its 1.5.4 training-fold fluctuation a bug because the EIF was not mean zero, and it moves the fluctuation to validation rows. [F24](#f24-fold-local-targeting-of-the-longitudinal-recursion) carries the published locators and all four remediation options |

#### Witnesses and evidence

| claim | evidence |
| --- | --- |
| every published verdict is recomputed from the committed replication rows | `tests/unit/test_method_evidence.py::test_paper_property_verdicts_are_recomputed_from_the_replication_rows` |
| the shipped fold and scale rules have mutation-controlled witnesses | `tests/unit/test_fold_policy_rules.py` |
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

#### Three findings this work deferred

| finding | why this pull request does not carry it | what it needs |
| --- | --- | --- |
| `tests/canonical/lmtp_crossfit_adapter.R` screens a supplied density-ratio matrix with `abs(mean(supplied) - 1) > 0.5` on the cumulative product. A unit whose follow-up ends at the first node has structurally zero later columns, so that mean estimates the probability of reaching the later node rather than one. `tests/canonical/lmtp_competing_adapter.R` carries the corrected form, which screens the first column | eight study manifests record the adapter's bytes, and two of those studies sit outside the ones regenerated here. Correcting the anchor invalidates every one of them | one task that corrects the anchor and regenerates all eight rows together. Two fixture READMEs carry the deferral, `tests/canonical/lmtp_ltmle/README.md` and `tests/canonical/lmtp_ltmle_survival/README.md`. Both counted four manifests and both now name the eight |
| Two property cells of `selector-based point-treatment C-TMLE` share a sample. `double_robustness/both_correct` and `root_n_and_efficiency/n_500` sit on one law at seed `12_100`, so their covariates are identical row for row, and the two cells are published side by side as separate evidence. The seed-offset table in `tests/studies/bounded_cv_laws.py` separates a consumer from the block it inherits and cannot reach a collision inside one consumer. `test_the_registered_studies_do_not_share_their_samples` iterates primary scenarios alone, so nothing refuses it | the collision predates this branch, and moving either seed redraws a cell of a registered study, so the fix is a regeneration rather than an edit. Widening the test first would fail on the committed rows | one task that widens the sample-sharing check to declared property cells, with an allowance for the families whose two arms are paired on one seed by design, then moves the colliding seed and regenerates the study |
| `simulated_confounding` cannot perturb an outcome on a fit that declares `q_bounds`. `_gaussian_outcome` subtracts the strength times a standard normal latent value from the outcome, so every nonzero outcome strength sends the perturbed outcome outside the declared support and the refit refuses it | a cross-fitted continuous fit must declare `q_bounds`, so the outcome axis of the surface is unavailable to every such fit. No perturbation on the declared scale exists to put in its place | [F23](#f23-simulated-confounding-on-a-declared-outcome-scale) |

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

### F24. Fold-local targeting of the longitudinal recursion

The cross-fitted longitudinal TMLE fits each fold's fluctuation on that fold's training rows. No
source read here certifies that update. The retired RM17 row held this question, and
[RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) holds the property cell where
the gap shows as a number. Upstream `lmtp` now classifies the same training-fold behavior as a bug.
This row holds the contract.

| composition | what ships | what a result must supply |
| --- | --- | --- |
| cross-fitted longitudinal TMLE | `LTMLE` offers no `targeting_scheme` and no `cv_evaluation`, so the fold-local update is its only behaviour (`src/cleverly/longitudinal/estimator.py:1704-1729`). Fold `k` fits every mechanism, regression and fluctuation on its training complement (`src/cleverly/longitudinal/sequential.py:51-56`, `:1377-1412`, `:919`) | weak convergence for a fluctuation fitted inside each training fold of a sequential regression, with its remainder and rate conditions |
| the reported estimate | a plug-in of the targeted regression, and never an influence-function average (`src/cleverly/longitudinal/sequential.py:1172-1173`, `:1486-1487`) | the limit law of that plug-in under the fold-local update |
| the point-treatment fold variant | `targeting_scheme="fold"` fits its epsilon on each fold's validation rows (`src/cleverly/estimators/tmle.py:3033-3050`) | its own result. A training-fold fluctuation and a validation-fold fluctuation are two different departures from a pooled update |
| corrected upstream `lmtp` TMLE | commit [`9996b04`](https://github.com/nt-williams/lmtp/commit/9996b04dcbb3ae0b1ef8862097c36d95e9f2fcf9) fits the fluctuation on validation rows and adds a mean-EIF test | a direct result for the validation-fold targeted plug-in, with its remainder and rate conditions |

Díaz, Williams, Hoffman and Schenck (2023), *JASA* 118(542):846-857, govern the shipped
construction. The locators below come from the typeset published article. ArXiv v4 has the same
section numbers and algorithm details. Read the page locators against the published version.

| published locator | what it states | journal page |
| --- | --- | --- |
| Section 5.2, Step 3 | the TMLE fluctuation is fitted "using all the data points in the sample" | 852 |
| Theorem 3 | weak convergence of that pooled TMLE | 853 |
| Section 5.3, Step 2 | each SDR regression uses only the data points of one fold | 854 |
| Section 5.3, Step 3 | the SDR estimate is an average of influence-function values | 854 |
| Lemma 4 and Theorem 4 | multiple robustness and weak convergence of the SDR estimator | 854 |

Theorem 3 therefore certifies the split and a pooled fluctuation, and not the shipped update.

Four further sources bound the available routes. Zheng and van der Laan (2011) and Levy (2018),
arXiv:1811.04573, both describe the targeting step as a pooled regression over validation folds.
Chernozhukov et al. (2018) certifies the cross-fitted orthogonal moment, and not a plug-in of a
train-fold-targeted regression. Williams and Díaz (2025), *Observational Studies* 11(3):365-367,
correct Assumption 2 and the positivity statement of the 2023 paper. That correction says nothing
about cross-fitting or targeting.

Upstream `lmtp` commit `9996b04`, dated 2026-06-09, says its
training-fold fluctuation left the EIF nonzero. The patch moves that fluctuation to validation
rows and adds a mean-EIF test. The forthcoming 1.5.5 `NEWS` file records the same fix.

A result closes this row in one of four ways.

| option | what it needs |
| --- | --- |
| a theorem for the shipped update | weak convergence of the fold-local targeted plug-in, with its remainder and rate conditions |
| a validation-fold update | an implementation that matches corrected upstream `lmtp`, a direct weak-convergence result for that targeted plug-in, and registered evidence |
| a pooled update | an implementation of the certified Section 5.2 fluctuation, and its own registered evidence |
| the SDR estimator | Section 5.3, Lemma 4 and Theorem 4 certify a fold-local sequential recursion for that estimator. Its estimate is an influence-function average rather than a plug-in of a targeted regression, so it is covered theory for a different estimator and not a free substitution. [X4](#x4-sequential-doubly-robust-longitudinal-estimation) plans `lmtp_sdr`, and no SDR path ships today |

The SDR option changes the estimator and not the rate conditions. The authors write on page 854
that the rates required for root-n consistency in Theorems 3 and 4 are the same. They add that the
SDR estimator does not seem to confer asymptotic advantages with respect to the TMLE.

## Collaborative and DR-TMLE investigation contracts

### F18. Selector-path C-TMLE inference

Keep the greedy, ordered, and discrete intervals labelled as fixed-candidate plug-in intervals.
Each fit uses the ordinary EIF covariance after it selects one candidate. That computation does
not establish conditional-on-selection or unconditional post-selection coverage.

Van der Laan and Gruber (2010), Section 4, derive an abstract adaptive-mechanism contribution under
fixed-limit and regularity assumptions. Their candidate-selection discussion does not derive the
influence function after the shipped stopping-index procedure selects a candidate.

Ju et al. (2018) permit a continuous-path contribution, but their Lemma 2 makes it zero in one
correct-outcome, product-rate regime. Their derivative equations and undersmoothing conditions do
not describe a discrete stopping index. Ju et al. (2019) report ordinary EIF intervals for binary
ATE fits. The
[source audit](references.md#collaborative-tmle) records the exact limits.

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

A second route is selector stability: a unique oracle candidate, a risk margin, uniform risk
convergence, and stability of the learned covariate identity at a given depth reduce the fit to a
fixed candidate with probability tending to one. An oracle risk inequality alone proves none of
those facts; Shao's linear-model result is a concrete warning that prediction-efficient fixed-fold
cross-validation need not consistently select a model. A third route is a construction change:
select depth wholly within each outer training fold, then evaluate only on its held-out rows.
Standard orthogonal-score arguments can then treat the complete selector as a nuisance-learning
algorithm, subject to its rate conditions. The current global risk aggregation does not have that
independence.

The fixed-candidate curve is itself unproved when the candidate's mechanism limit differs from the
treatment law. [RM12](#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism) records an
instrument law where the intercept-only curve gives a standard error of 0.0441. The sampling
standard deviation of the numerically identical least-squares coefficient is 0.0538. A result must
settle that fixed-candidate case before it addresses selection.

The registered evidence now measures that gap on two studies. The
[selector-based point-treatment study](technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md)
passes 11 of 14 property cells, and the
[selector-based multi-arm study](technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md)
passes 5 of 12. [RM18](#rm18-red-property-cells-after-the-fold-scale-and-law-changes) lists each
red cell with its interval and its attribution. No result here identifies a selection contribution,
so each study publishes under a `reporting` policy.

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
estimated vector of arm-specific outcome predictions. The current interval uses ordinary
adaptive-propensity EIF plug-in covariance. The cross-fitted implementation now follows the
paper's fold-local nuisance nesting: one outcome model fitted on fold `v`'s training rows creates
both sides of that fold's generated design, the adaptive propensity is fitted only on its training
side, and both nuisances are evaluated on its held-out side. Under the paper's six regularity
conditions, the ordinary adaptive-propensity curve needs no extra first-order generated-design
term for one binary treatment-specific mean. Appendix D constructs a binary ATE with both arm
predictions and one signed fluctuation coefficient, and outlines pooled-validation CV-C-TMLE.

The archived `ctmle3` source supplies implementation provenance but no inference derivation.
Benkeser, Cai and van der Laan (2020) prove a binary treatment-specific-mean result under explicit
score, rate, smoothness, and empirical-process conditions. Appendix D explicitly uses both binary
arm predictions in one adaptive propensity for the ATE and sketches a cross-validated C-TMLE. The
generated design is therefore part of that paper rather than an omitted nuisance.

The paragraphs above describe the nuisance nesting, and the fluctuation needs its own statement.
`CTMLE` refuses `targeting_scheme="fold"` (`src/cleverly/estimators/ctmle.py:651-656`), so the
outcome-adaptive fluctuation is one pooled epsilon on the stacked out-of-fold rows
(`src/cleverly/estimators/tmle.py:372-378`, `:3061-3067`). Appendix D outlines that pooled
fluctuation. The shipped update therefore does not diverge from the paper on this axis. Two
divergences are genuine, and the table below states each one.

| divergence | what ships | what Appendix D outlines |
| --- | --- | --- |
| the final average | the stacked whole-sample plug-in (`src/cleverly/estimators/tmle.py:408-418`), because `CTMLE` refuses `cv_evaluation=True` (`src/cleverly/estimators/ctmle.py:646-650`) | the `(1/V) sum_v` fold average. The two agree only at equal fold weight mass |
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

Five items stay open. State each verdict separately.

| open item | what a result must settle |
| --- | --- |
| one shared multinomial | one categorical fit on `K` estimated columns supplies every arm's clever covariate, so an inconsistent column for one arm enters the mechanism of every other arm |
| vector target and simultaneous inference | the joint covariance and the simultaneous critical value, not the per-arm variance alone |
| uniformity | the estimator is deliberately superefficient, so a pointwise limit law does not give locally uniform coverage |
| the registered measurement | at `n = 1,000`, the point-treatment generated-design pair resolves a finite-sample standard-error-ratio deficit under a correct outcome regression without showing invalid coverage. Its paired deficit runs -0.0471 to -0.0209. The binary OAT study passes 14 of 14 property cells on the bounded law it now runs, and its sharp-null cell turned green at the unchanged 800-replication budget, at a rejection rate of 0.0712 with a 99% upper endpoint of 0.0980. The multi-arm pair does not resolve a deficit, and the multi-arm study passes 10 of 12 property cells. None of these results identifies a first-order term |
| transport beyond the source law | missing-outcome fits need the response-mechanism expansion. Fixed probability weights need a weighted empirical-law result; estimated weights also need a first-stage contribution. Cluster-robust and stratified fits need dependence- and stratum-specific expansions. Repeated cross-fitting needs a result for the package's median and split-dispersion aggregation |

Accept a result only when it covers the exact cross-fitted, multi-arm construction and distinguishes
a fixed target from a target conditional on the learned design. It must establish whether the
current curve suffices or an additional representation contribution is required. The result must
cover the requested joint means or contrasts and their covariance. It must also state its
remainder and nuisance-rate conditions, and it must state a uniformity claim.

If the result requires a new contribution, propagate it through pointwise and simultaneous
inference. Acceptance needs a fixed-design reduction, a generated-design comparison, and
registered coverage evidence for every claimed treatment and target dimension.

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

The arm-indexed audit did not examine two sibling surfaces. Each needs its own source audit before
any contract.

| sibling surface | current behavior |
| --- | --- |
| the shift, incremental, regime, MSM, and controlled-direct-effect targets under cross-fitting with missing outcomes | they fit today, outside the arm-indexed contract, and tests cover them. `TestTheMnarTiltFollowsTheDraws` in `tests/unit/test_repeated_crossfit.py` is the only fast test that fits repeated draws on this surface, a controlled direct effect with `repeats=2`. Remove or move that test when this gap closes |
| ordinary, in-sample C-TMLE with missing outcomes | it fits today; the audit read no source for it. [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) holds cross-fitted C-TMLE |

### F22. Grouped cross-fitting beyond point-treatment TMLE

The package draws whole-cluster outer folds for the ordinary cross-fitted point-treatment TMLE and
DR-TMLE. Every other cross-fitted surface refuses `id=`, each for its own stated reason. Three
compositions need their own result before that refusal can be lifted.

| composition | what ships | what a result must supply |
| --- | --- | --- |
| C-TMLE with `id=` | refused at every `cross_fit` setting (`CTMLE._resolve_estimands_for_data`) | a split law for the selection folds and the nested selection folds under clustering, and the cluster-robust variance of the candidate the search stops at. [F18](#f18-selector-path-c-tmle-inference) is open for iid rows, so a clustered result needs that one first |
| cross-fitted longitudinal TMLE with `id=` | refused above one fold (`LTMLE._refuse_cross_fitted_design`). The in-sample clustered fit is evidenced, and it stays available | the cluster-robust variance of the targeted sequential recursion under a grouped draw. The audit read no source for it |
| a row-weighted against a cluster-weighted target at unequal cluster sizes | not refused, and not claimed | the two weightings agree only at equal, or non-informative, cluster sizes. The documented scope is equal sizes, and the registered study fixes ten rows per cluster |

The grouped point-treatment split itself is supported for the partition alone. Wang, Park, Small
and Li (2024), Section 4.2 and Theorem 4(b), prove a cross-fitted result under a random, roughly
equal partition of the clusters. Their estimator is AIPW-type with a cluster-level treatment. The
rest is the package's own estimating-equation argument, which the
[fold and outcome-scale rules](technical-reference/cv-tmle.md#grouped-folds) state with its four
conditions: independent clusters, equal cluster sizes, no interference, and the usual remainder
rates. The registered
[clustered point-treatment CV-TMLE study](technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md)
is its only empirical witness.

A result must also state two further boundaries. No source read here supports a normal reference
interval with few clusters. Benitez et al. (2023) and Nugent et al. (2024) recommend a $t$
reference with $J - 2$ degrees of freedom below about 30 to 40 clusters, and the package applies
neither rule.

[Grouped folds and clustered cross-fitting](references.md#grouped-folds-and-clustered-cross-fitting)
gives every source the audit read, with the version whose locators it used.

### F4. Multi-arm missing-outcome DR-TMLE

`delta=` under `guard=("Q", "g")` continues to refuse more than two treatment arms
(`src/cleverly/estimators/drtmle.py:957-962`). Díaz and van der Laan (2017), Theorem 2, page 20,
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
the missing collaborative theory as a hard stop. In-sample C-TMLE with missing outcomes still fits.
[F21](#f21-other-missing-outcome-cv-tmle-variants) records it as an unexamined sibling surface.

Each C-TMLE extension needs its target-specific collaborative score and selection-risk contract.
Each DR-TMLE extension needs reduced regressions, a correction, a remainder, and rate conditions.
PAR and PAF also need the joint observed-mean curve and covariance. Complete simulated-confounding
replay receives its own audit only after the estimator can fit the target.

Omitted-variable bounds join these refused compositions once
[RM11](#rm11-sensitivity-bounds-outside-their-derivation) lands. A bound on either fit needs
an estimate of $\nu^2$ that stays valid when the estimator does not assume a consistent treatment
mechanism. It also needs the influence curve of the bound under that estimator. A separately fitted
full mechanism is a new estimator of the bound, and it needs the same derivation.

An estimated-weight DR-TMLE fit is not refused. Its interval conditions on the supplied weights.
An unconditional interval claim needs the influence contribution from weight estimation. This
inference gap is separate from F11, which tracks weight-model replay after a perturbation.

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

[F24](#f24-fold-local-targeting-of-the-longitudinal-recursion) names this item as one of its four
remediation routes. Theorem 4 of the same article certifies a fold-local recursion for the SDR
estimator. Theorem 3 certifies a pooled fluctuation for the shipped one. So this item answers
F24's fold-local question by shipping the estimator the theory already covers.

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

## Reading a gap correctly

Not every absence is missing package functionality. The refusal taxonomy in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal) distinguishes an unimplemented
well-posed feature from a different causal question and a method that would be wrong by
construction. Only the first belongs on this roadmap.
