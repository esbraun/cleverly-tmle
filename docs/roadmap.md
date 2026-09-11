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
post-fit coverage. Complete these rows in order before main-roadmap priority 1. A new capability
still needs its own contract and evidence, even when it appears in this top-priority queue.

The "next action" column states the remediation work. It is not a readiness label. RM8 carries an
open source audit.

| priority | item | next action | problem exposed by the examples | details |
| ---: | --- | --- | --- | --- |
| 0.1 | Missing-outcome natural-course mean | add the audited iid MAR mean | the observed-law mean is refused when outcomes are missing, even though the response process is declared | [RM7](#rm7-missing-outcome-natural-course-mean) |
| 0.2 | Missing-outcome attributable effects | complete the source audit | population attributable risk and fraction are refused when outcomes are missing | [RM8](#rm8-missing-outcome-attributable-effects) |

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

[F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) is a neighbour rather than a duplicate. It
tracks the joint observed-mean curve and covariance for the collaborative and doubly robust
families. RM8 covers the same two parameters for ordinary TMLE under `missingness=`.

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
| Stochastic categorical policies at a longitudinal node | longitudinal identification, influence function, remainder, and interval conditions for a distribution-valued policy | deterministic categorical regimens only | [F1](#f1-stochastic-categorical-policies-at-a-longitudinal-node) |
| Targeted bootstrap inference | a construction that defines what is fixed, resampled, refitted, and retargeted, plus the sampling law of the interval | existing bootstrap inference is not this procedure | [F2](#f2-targeted-bootstrap-inference) |
| Longitudinal sensitivity-bound estimation | sample estimation of the bound functionals, a specialized algorithm, and sampling inference | no sensitivity bound on a longitudinal fit | [F16](#f16-longitudinal-sensitivity-bound-estimation) |
| Additional longitudinal estimands | target-specific identification, influence function, targeting construction, and inference conditions | existing end-of-study, survival, competing-risk, and MSM targets only | [F3](#f3-additional-longitudinal-estimands) |
| Multi-arm missing-outcome DR-TMLE | arm-indexed observation, treatment, and outcome corrections, with a remainder and rate conditions | binary randomized treatment only | [F4](#f4-multi-arm-missing-outcome-dr-tmle) |
| Other refused C-TMLE and DR-TMLE compositions | composition-specific score, reduced regressions, correction, remainder, and rate conditions | named pre-fit refusals and conditional-on-weight intervals remain | [F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| Selector-path C-TMLE inference | an influence function and covariance after the shipped data-adaptive stopping-index selection | ordinary EIF plug-in covariance that treats the selected candidate as fixed | [F18](#f18-selector-path-c-tmle-inference) |
| Outcome-adaptive C-TMLE generated-design inference | exact scalar expansions for the shipped joint binary fit and a multi-arm vector extension of the paper-backed fold-local construction | ordinary adaptive-propensity EIF covariance with a proved binary scalar construction and open joint-target extensions | [F19](#f19-outcome-adaptive-c-tmle-generated-design-inference) |
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
records completed studies. The RM7 audit supports one iid missing-at-random mean. RM8 holds the
remaining remediation audit. Replicate-weight designs are the next source-audit item in the main
grid. Implement them only after that audit supports the planned variance construction.
Longitudinal sensitivity-bound estimation remains in
[F16](#f16-longitudinal-sensitivity-bound-estimation).

## Detailed implementation contracts

The sections below group contracts by subsystem. Their physical order does not override the main
grid.

### RM7. Missing-outcome natural-course mean

Add `NaturalCourseMean()` under `missingness=` for ordinary TMLE. As deliberate first-PR
engineering boundaries, cover one scalar target with binary treatment from unweighted,
unclustered, unstratified iid observations. Require `cross_fit=False`, `fluctuation="logistic"`,
`targeting="iterative"`, and `target_weights=False`. The audited source also gives the equivalent
weighted intercept fluctuation. Treat `target_weights=True` as a follow-up extension that needs its
own package tests.
Accept a binary outcome or a continuous outcome with fixed declared `q_bounds`.

Let $X=(A,W)$ contain the observed treatment and baseline covariates. Let
$m_P(x)=E_P(Y\mid\Delta=1,X=x)$ and $\pi_P(x)=P_P(\Delta=1\mid X=x)$. The target is

$$
\psi(P)=E_P\{m_P(X)\}.
$$

The target population is represented by all input rows before outcome response. It keeps the
observed joint law of treatment and covariates. Identification requires
$Y\mathrel{\perp\!\!\!\perp}\Delta\mid X$ and $\pi_0(X)>0$ almost surely.

Díaz, Carone and van der Laan (2016), Section 2 and Equations (1)–(5), govern this target. The
[source record](references.md#point-treatment-and-stochastic-interventions) gives the locators and
notation map. Its efficient influence function is

$$
D_P(O)=\frac{\Delta}{\pi_P(X)}\{Y-m_P(X)\}+m_P(X)-\psi(P).
$$

Its exact second-order remainder is

$$
R_2(P,P_0)=\int\left\{1-\frac{\pi_0(x)}{\pi_P(x)}\right\}
  \{m_P(x)-m_0(x)\}\,dP_{X,0}(x).
$$

Fit the logistic fluctuation among rows with $\Delta=1$:

$$
\operatorname{logit}\hat m_\epsilon(x)
  =\operatorname{logit}\hat m(x)+\frac{\epsilon}{\hat\pi(x)}.
$$

Solve $P_n[\Delta\{Y-\hat m^\star(X)\}/\hat\pi(X)]=0$. Report the plug-in
$P_n\hat m^\star(X)$. Do not delegate this target to the binary incremental-intervention alias
whose identity-policy influence function has the same algebra.

The treatment mechanism enters the parameter through the natural distribution of $A$ in $X$.
It is not a separately estimated nuisance and contributes no inverse treatment probability.
This parameter needs no treatment positivity or treatment exchangeability assumption.

For the first interval, require $1/\hat\pi(X)=O_P(1)$ and convergence of the estimated influence
function in $L_2(P_0)$. Require $D(\hat P)$ to belong to one Donsker class with probability tending
to one, and require
$R_2(\hat P,P_0)=o_P(n^{-1/2})$. A sufficient rate condition is

$$
\|\hat\pi-\pi_0\|_{P_0}\|\hat m^\star-m_0\|_{P_0}=o_P(n^{-1/2}).
$$

Clipping at $b$ guarantees $1/\hat\pi_b\leq1/b$. For the stated efficient interval, also require
$\hat\pi_b\to\pi_0$. A sufficient compatibility condition is uniform $\pi_0>b$ and consistency of
the raw response-score learner. If $\pi_0<b$ on a set with positive probability, clipping alone
cannot establish that convergence.

Keep `NaturalCourseMean()` refused with cross-fitting, repeated splits, observation weights,
clusters, or strata. Also refuse nonbinary treatment, joint targeting, simultaneous bands, and
bootstrap inference. Keep the current complete-outcome behavior for those compositions. Keep
C-TMLE, DR-TMLE, intermediate, shift, and incremental compositions at their existing boundaries.

An `ey_obs`-only result must mark the arm-specific missingness tilt and tipping gamma unavailable.
The combined assessment must report the same state. Add a natural-course sensitivity parameter
before either assessment can run for this result.

Acceptance requires exact-law, Gateaux, remainder, and deliberate response-score mutation checks.
Prove that no treatment learner or propensity prediction enters the fit. Add separate witnesses
with the outcome regression correct and response score wrong, and with the two roles reversed.
Add a law where the respondent-only empirical mean differs from the target. The $\pi\equiv1$ limit
must reduce exactly to the existing empirical mean and $Y-\psi$ influence function. Add
repeated-sampling coverage and standard-error calibration before exposing the interval.

The identification record must state MAR given $(A,W)$ and response positivity. It must name only
the outcome regression and response mechanism as nuisances. Pin its target expression,
double-robustness statement, typed parameter key, and current method availability in tests.

### RM8. Missing-outcome attributable effects

Add population attributable risk and population attributable fraction with missing outcomes as
their own observed-law targets. Audit each influence curve rather than composing the current ATE and
observed-mean results algebraically. The fraction must define its zero-denominator behavior and the
joint covariance of its numerator and denominator.

This row depends on RM7. Its complete-outcome reduction reads the observed-law mean that RM7
derives, so RM7 must land first. Hubbard and van der Laan (2008) govern the complete-outcome PAR
and PAF, and the audit must carry them to a declared response process.

[F5](#f5-other-refused-c-tmle-and-dr-tmle-compositions) states the same joint-curve requirement
for the collaborative and doubly robust families. This row covers ordinary TMLE under
`missingness=` only.

Acceptance requires exact reduction to the complete-outcome PAR and PAF, a nonzero response-score
witness, denominator-boundary refusals, and repeated-sampling evidence for both difference and ratio
scales. Keep the public pre-fit refusal until all pieces are registered.

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
| the registered measurement | at `n = 1,000`, the point-treatment generated-design pair resolves a finite-sample standard-error-ratio deficit under a correct outcome regression without showing invalid coverage. The binary OAT study passes 13 of 14 property cells. Its sharp-null cell is red: the 99% rejection upper endpoint is 0.1008 against 0.10, and the coverage lower endpoint is 0.8992 against 0.90. The multi-arm pair does not resolve a deficit, and the multi-arm study passes 9 of 12 property cells. None of these results identifies a first-order term |
| transport beyond the source law | missing-outcome fits need the response-mechanism expansion. Fixed probability weights need a weighted empirical-law result; estimated weights also need a first-stage contribution. Cluster-robust and stratified fits need dependence- and stratum-specific expansions. Repeated cross-fitting needs a result for the package's median and split-dispersion aggregation |

Accept a result only when it covers the exact cross-fitted, multi-arm construction and distinguishes
a fixed target from a target conditional on the learned design. It must establish whether the
current curve suffices or an additional representation contribution is required. The result must
cover the requested joint means or contrasts and their covariance. It must also state its
remainder and nuisance-rate conditions, and it must state a uniformity claim.

If the result requires a new contribution, propagate it through pointwise and simultaneous
inference. Acceptance needs a fixed-design reduction, a generated-design comparison, and
registered coverage evidence for every claimed treatment and target dimension.

### F4. Multi-arm missing-outcome DR-TMLE

`delta=` under `guard=("Q", "g")` continues to refuse more than two treatment arms. Díaz and van
der Laan's missing-outcome theorem is binary and does not provide arm-indexed observation,
treatment, and outcome correction blocks. Begin only when a source supplies the multi-arm
corrected influence curve, remainder, and rate conditions. Existing binary evidence is the
regression surface the extension must preserve.

### F5. Other refused C-TMLE and DR-TMLE compositions

Continue pre-fit refusals for ATT, ATC, PAR, PAF, regimes, incremental interventions, shifts, MSMs,
mediation, and missing treatment where each variant lacks evidence. Ordinary-TMLE implementations do
not establish collaborative or doubly robust inference for these compositions.

Each C-TMLE extension needs its target-specific collaborative score and selection-risk contract.
Each DR-TMLE extension needs reduced regressions, a correction, a remainder, and rate conditions.
PAR and PAF also need the joint observed-mean curve and covariance. Complete simulated-confounding
replay receives its own audit only after the estimator can fit the target.

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
