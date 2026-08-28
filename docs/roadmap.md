# Roadmap

This is the single planning contract for `cleverly`. It contains proposed work only. Implemented
capabilities belong in the [user guide](user-guide/index.md), scientific contracts in the
[technical reference](technical-reference/index.md) and [DR-TMLE contract](technical-reference/dr-tmle/index.md), validation results in
[evidence manifest](technical-reference/evidence.md), and cross-module standing decisions in the
[architecture invariants](architecture-invariants.md).

The tracks below are independent. Order is binding within a track, but an item in one track does
not block another track unless its dependency says so.

| track | order | item | readiness | dependency | details |
| --- | ---: | --- | --- | --- | --- |
| Validation | V1 | Categorical longitudinal studies | source audit complete; pinned R `lmtp` comparator | registered multi-arm helpers | [V1](#v1-categorical-longitudinal-studies) |
| Validation | V2 | Fold-repeat studies | source audit | V1 ordering only | [V2](#v2-fold-repeat-studies) |
| Validation | V3 | Clustered inference studies | source audit | V2 ordering only | [V3](#v3-clustered-inference-studies) |
| Validation | V4 | Point-treatment weight studies | source audit | V3 ordering only | [V4](#v4-point-treatment-weight-studies) |
| Validation | V5 | Controlled direct-effect studies | source audit | V4 ordering only | [V5](#v5-controlled-direct-effect-studies) |
| Validation | V6 | Weighted longitudinal studies | source audit | V5 establishes the fixed-weight study design | [V6](#v6-weighted-longitudinal-studies) |
| Extensibility | E1 | Nested Riesz engine and initial catalog | published support; source audit complete | typed study, identification, result, and assessment contracts | [E1](#e1-nested-riesz-engine-and-initial-catalog) |
| Extensibility | E2 | Optional DoWhy integration | source audit | E1 in the default sequence; may split if schedules diverge | [E2](#e2-optional-dowhy-integration) |
| Extensibility | E3 | EP learner | published support; pending source read | E1 in the default sequence; may split if schedules diverge | [E3](#e3-ep-learner) |
| Extensibility | E4 | Evidence-gated catalog expansion | source audit for each target | the engine and target-specific derivation | [E4](#e4-evidence-gated-catalog-expansion) |
| Longitudinal | L1 | Stochastic categorical policies at a node | waiting on published theory | none | [L1](#l1-stochastic-categorical-policies-at-a-node) |
| Longitudinal | L2 | Targeted bootstrap | waiting on a citable construction | L1 ordering only | [L2](#l2-targeted-bootstrap) |
| Longitudinal | L3 | Persistence and serialization | theory-neutral | L2 ordering only | [L3](#l3-persistence-and-serialization) |
| Longitudinal | L4 | Sensitivity analysis | source audit for each operation | L3 ordering only | [L4](#l4-sensitivity-analysis) |
| Longitudinal | L5 | Additional longitudinal estimands | waiting on published theory | L4 ordering only | [L5](#l5-additional-longitudinal-estimands) |
| Longitudinal | L6 | Time-respecting cross-fitting | source audit | L5 | [L6](#l6-time-respecting-cross-fitting) |
| DR-TMLE | D1 | Multi-arm missing-outcome DR-TMLE | waiting on published theory | a multi-arm corrected influence curve | [D1](#d1-multi-arm-missing-outcome-dr-tmle) |
| DR-TMLE | D2 | Other refused DR-TMLE compositions | waiting on published theory | composition-specific reduced regressions and corrected curve | [D2](#d2-other-refused-dr-tmle-compositions) |
| Later candidates | C1 | Replicate-weight designs | source audit | weighted-law variance construction | [C1](#c1-replicate-weight-designs) |
| Later candidates | C2 | MNAR and incremental-intermediate extensions | waiting on published theory | composition-specific identification and influence function | [C2](#c2-mnar-and-incremental-intermediate-extensions) |
| Later candidates | C3 | HAL and undersmoothed HAL learners | published support; source audit | profiling evidence before a native implementation | [C3](#c3-hal-and-undersmoothed-hal-learners) |

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
- **source audit**: theory or canonical code appears to cover it, but the exact construction must
  be matched and discrepancies resolved before implementation;
- **theory-neutral**: engineering that preserves an already-derived estimator;
- **waiting on published theory**: related methods exist, but not the requested composition or
  inference claim; this is not an active research assignment for the project; and
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

## Validation track

The [implementation validation grid](technical-reference/method-evidence/validation-grid.md)
records completed studies. This track records the sequence for implementation families that the
grid does not cover. A completed item leaves this roadmap and enters the grid with committed
artifacts.

### V1. Categorical longitudinal studies

Add ordinary and cross-fitted rows for categorical treatment nodes under static and dynamic plans.
The source audit identifies pinned R `lmtp` 1.5.4 as the canonical comparison. It supports
categorical treatment nodes in longitudinal static and dynamic plans.

#### Shared law and parameter scope

Use one two-time-point law with three labelled treatment levels at both nodes. The label order must
differ from the semantic treatment order. The first treatment must change the intermediate
covariate, and the second treatment mechanism and outcome must depend on that covariate.

Report static plans that use the reference and third arms. Report a mixed static plan and a dynamic
plan that selects two different second-node arms from the observed history. Include every regimen
mean and each non-reference contrast in the known-truth study.

Share the law, exact g-formula oracle, sampler, nuisance fixtures, plan-to-arm mapping, fit adapter,
and result-row conversion across the two rows. Keep each study's record, seed stream, fitting mode,
property adapter, and artifacts separate. Extend the existing categorical longitudinal exact law
instead of creating a second distribution with the same scientific role.

#### Canonical comparisons

Compare both rows with R `lmtp` 1.5.4 at pinned commit `f04a2b4`. Give both implementations the
same realized samples, labelled treatment columns, exact treatment mechanism, nuisance regression
family, plans, folds, and intervals. Run `lmtp` with one all-row fold for the ordinary study. Give
it the exact rowwise five-fold assignment for the cross-fitted study.

The source audit rejects R `npcausal` because its public estimators do not include deterministic
categorical treatment plans over time. It also rejects R `stremr` as the primary comparison because
its long-form data interface introduces a second representation. Keep the Poulos companion code as
supporting evidence for longitudinal multi-valued treatment, but do not use its simulation scripts
as the canonical implementation witness.

Do not split one categorical regimen into separate binary fits. Do not compare with a point-
treatment estimator or reuse the binary longitudinal rows. Keep every source-audit snapshot in the
references as comparator provenance.

#### Ordinary categorical LTMLE

Fit the shared law without cross-fitting. Give both primary implementations the same
quasibinomial outcome-regression family and the exact treatment mechanism. Use saturated cell
learners for the exact-law property instruments. Validate every reported mean and contrast against
its exact g-formula truth with pointwise 95% Wald intervals.

Give R `lmtp` one fold whose training and validation sets contain every row. Verify its categorical
density ratios against the exact law before the comparison runs.

The property study must test both halves of sequential double robustness and a both-wrong control.
It must test bias, coverage, standard-error calibration, and root-n behavior at three sizes for one
static and one dynamic contrast. Add a sharp-null size cell and a nonzero-effect power control.

Add a nonzero targeting control. Add a third-arm control that substitutes a binary complement for
the assigned probability. Add a dynamic-rule control that changes the second-node arm on one
history stratum. Each mutation must fail the same instrument that its positive cell passes.

#### Cross-fitted categorical LTMLE

Fit one declared five-fold split and preserve the rowwise fold assignment. Fit and target each
training recursion before evaluation on its held-out rows. Do not inherit the ordinary row's
finite-sample result merely because both estimators have the same limit.

Give R `lmtp` the same rowwise fold assignment and exact categorical density ratios. Keep the
fold-specific nuisance and targeting recursion inside each training and validation pair.

Repeat the ordinary row's double-robustness, three-size, calibration, null, power, targeting, third-
arm, and dynamic-rule instruments under cross-fitting. Add a flexible-learner overfitting pair that
compares the cross-fitted estimator with the same in-sample learner and the same realized samples.

Structural tests must prove that each training fold contains all three treatment levels at both
nodes. They must also prove that held-out predictions come only from the matching training
complement. A common random seed or fold count is not enough evidence for those conditions.

#### Registration and publication

Use 99% confidence bounds for every statistical verdict. Size each replication budget against the
binding endpoint before the final run. Run disposable ordinary and cross-fitted smoke studies before
the declared studies.

Treat missing rows, failed fits, non-finite results, active undeclared bounds, incomplete treatment
support, and fold leakage as hard errors. Run Python and R smoke comparisons before the declared
studies. Commit the six standard CSV artifacts and a hash-complete manifest for each row. The
seventh standard file is each manifest itself. Generate every measured table from those artifacts.

Update the validation grid, study index, evidence manifest, longitudinal method page, testing
strategy, and applicable examples. Add plain-English descriptions for every new scenario,
implementation, estimand, property family, and cell. Retire a legacy repeated-sampling check only
when one exists and both replacement rows cover its claim.

Remove V1 after both rows and every documentation gate pass. Promote and renumber the remaining
validation items so the roadmap contains only proposed work. Update their ordering dependencies at
the same time.

Each limits section must state the complete-outcome design, two-node and three-level scope,
learner class, fitting scheme, interval type, and deterministic-plan boundary. It must name excluded
censoring, missingness, stochastic categorical policies, continuous doses, survival and competing
risks, weights, clusters, fold repeats, simultaneous bands, and flexible primary learners.

### V2. Fold-repeat studies

Validate rowwise averaging across independent fold draws. The study must distinguish repeated
cross-fitting from one fixed split and from equal-fold averaging.

### V3. Clustered inference studies

Validate cluster-level covariance and fold integrity under genuine within-cluster dependence. The
negative control must analyze the same rows as independent observations.

### V4. Point-treatment weight studies

Validate fixed probability weights against the tilted population law. The negative control must
omit the weights and converge to a different parameter.

### V5. Controlled direct-effect studies

Validate each declared intermediate level against its exact controlled parameter. The study must
exercise the treatment and intermediate mechanism product with a nonzero control.

### V6. Weighted longitudinal studies

Validate fixed weights through nuisance fitting, targeting, plug-in averaging, and covariance. The
negative control must omit the weights and miss the declared longitudinal parameter.

## Extensibility track

### E1. Nested Riesz engine and initial catalog

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

Repeated splitting follows only after a single split is correct: average point estimates and
rowwise curves across repeats before variance calculation, retain repeat IDs on artifacts, and
reject equal-fold averaging with an unequal-fold-size test.

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
treatment/censoring products, clusters, repeats, and persistence. Named slow studies cover point
means/ATE under both union-model halves, direct consistency and coverage, static/dynamic regimen
means, and weak overlap with thresholds fixed before the final run.

Implement in one review PR with gated commits: contracts and pre-fit refusals; analytic
single-stage parity; direct single-stage learning; nested and missingness composition;
longitudinal adapters; then persistence, assessment, documentation, and the secondary fixture. No
commit merges independently. Handoff requires the enabled/refused catalog, source locators,
implementation revision and discrepancies, evidence instruments, local commands/results, and
path-based reasons for omitted slow studies.

### E2. Optional DoWhy integration

Add `DoWhyIdentificationProvider` behind a `dowhy` extra. It accepts supported graphs, invokes
DoWhy identification, translates supported backdoor results into `IdentifiedEffect`, preserves
the original identified estimand and graph/provider provenance, verifies treatment, outcome,
adjustment set, and population, and refuses other strategies before fitting.

A graph stays optional and no causal discovery is performed. Supplying a graph *and* an
adjustment set means "validate this proposed set". It never means "pick whichever is convenient".
A disagreement is an error. The user resolves it by naming a different valid set, and the provider
never chooses one. Front-door, IV, transport, mediation, and unidentified results stay
refused until a matching `cleverly` functional and estimator are evidenced.

The reverse adapter accepts a DoWhy `IdentifiedEstimand`, translates supported backdoor effects,
runs the ordinary `cleverly` engine, returns the generic DoWhy estimate, and attaches the native
`CausalResult`. The native result remains the complete diagnostics and provenance surface.

Keep DoWhy out of the core and initially out of `all`, but include `cleverly[dowhy]` in `dev` so
translation tests run in ordinary local and CI tiers. Pin a tested public-API version range,
isolate imports under the integration package, document translation limits, and add a no-extra
session or marker for missing-dependency errors.

Acceptance requires equivalence between graph and explicit-adjustment workflows for the same
identified functional, pre-fit refusals for unsupported results, round-trip provenance, a version
matrix, and successful core import and operation without DoWhy installed.

### E3. EP learner

After first-hand review of the governing EP derivation, add `ConditionalContrast` estimands,
modifier schema, sieve/basis strategy, efficient plug-in risk and targeting, bounded outcome
predictions, a second-stage contrast learner, out-of-fold risk/calibration, and a conditional
prediction result. Reuse study/identification objects, nuisance strategies, folds, data backends,
provenance, persistence, and capability-aware assessment.

The first catalog is paper-derived CATE and conditional relative risk. Other losses and contrasts
require their own derivations. Aggregating an EP curve is a separate parameter and receives scalar
inference only after its influence contribution is implemented and tested.

Acceptance requires exact score and risk checks, bounded predictions, out-of-fold calibration,
modifier and split/basis stability diagnostics, mutation controls for targeting sign, basis
contribution, and contrast construction, plus named slow oracle-efficiency and stability studies.

### E4. Evidence-gated catalog expansion

Expand target by target after the engine lands. Mediation, additional longitudinal targets,
sampling designs, and other nested functionals each require a governing derivation, typed adapter,
registry entry, evidence row, refusal boundary, documentation, and applicable statistical study.
Do not expose a generic engine capability as a certified causal estimand.

## Longitudinal track

The four core LTMLE evidence rows are implemented and registered in the
[validation grid](technical-reference/method-evidence/validation-grid.md). They separate
end-of-study and survival parameters from ordinary and cross-fitted nuisance estimation. The
remaining items below are proposed extensions to that core.

### L1. Stochastic categorical policies at a node

The implemented surface assigns one category per unit. A distribution-valued policy changes the
intervention density and replaces selected probabilities with cumulative density ratios.
Implementation waits for published identification, longitudinal influence function, remainder,
and interval rate conditions; a point-treatment stochastic regime is not sufficient evidence.

### L2. Targeted bootstrap

Wait for a source specifying what is fixed, resampled, refitted, and retargeted and which sampling
law the interval estimates. Resampling stored curves, retargeting cached arrays, and refitting the
complete estimator are distinct procedures and must not be inferred from the name.

### L3. Persistence and serialization

Preserve the fitted recursion, regimen and node metadata, targeting state, diagnostics, and enough
learner provenance to distinguish replayable operations from those requiring a refit. Round trips
must preserve estimates, curves, scores, and refusal behavior.

### L4. Sensitivity analysis

A sweep over prespecified nuisance bounds may refit an established estimator without defining a
new estimand. Any change to the intervention, missingness law, or reported parameter requires its
own identification and influence-function result. Rerun the full backward recursion whenever a
bound changes an earlier pseudo-outcome.

### L5. Additional longitudinal estimands

Competing-event interventions and other longitudinal estimands wait for their own identification
assumptions, influence functions, targeting construction, and inference conditions. Add accepted
targets in both directions to the oracle registry and evidence gates rather than treating them as
options on an existing cause-specific estimand.

### L6. Time-respecting cross-fitting

Audit blocked-temporal and rolling-origin splitting separately against published results whose
dependence assumptions match the supported data. Record which rows may train every prediction and
which asymptotic argument licenses its interval. Ordered indices passed through iid fold machinery
are not sufficient.

## DR-TMLE track

### D1. Multi-arm missing-outcome DR-TMLE

`delta=` under `guard=("Q", "g")` continues to refuse more than two treatment arms. Díaz and van
der Laan's missing-outcome theorem is binary and does not provide arm-indexed observation,
treatment, and outcome correction blocks. Begin only when a source supplies the multi-arm
corrected influence curve, remainder, and rate conditions. Existing binary evidence is the
regression surface the extension must preserve.

### D2. Other refused DR-TMLE compositions

Continue pre-fit refusals for `att`/`atc`, stochastic and incremental interventions, continuous
shifts, MSMs, mediation, C-TMLE, estimated weights, and missing treatment in the DR-TMLE regime.
Ordinary-TMLE implementations do not establish intervals valid when one primary nuisance is
inconsistent. Each composition waits for its reduced regressions, corrected influence curve,
remainder, and rate conditions; estimated weights also require their estimation influence term.

## Later-candidate track

### C1. Replicate-weight designs

Add BRR, jackknife, or another replicate design only after its published variance construction is
matched to this package's weighted-law estimands and inference conventions.

### C2. MNAR and incremental-intermediate extensions

An MNAR tilt for continuous-dose shifts and intermediate variables with incremental interventions
wait for identification and influence-function results covering those exact compositions.

### C3. HAL and undersmoothed HAL learners

Match published loss, basis, optimization, and undersmoothing criteria. Consider a native
implementation only after profiling shows that package-owned HAL work materially dominates
end-to-end time.

## Reading a gap correctly

Not every absence is missing package functionality. The refusal taxonomy in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal) distinguishes an unimplemented
well-posed feature from a different causal question and a method that would be wrong by
construction. Only the first belongs on this roadmap.
