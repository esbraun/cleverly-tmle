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
| Validation | V1 | Fold-repeat studies | source audit | none | [V1](#v1-fold-repeat-studies) |
| Validation | V2 | Clustered inference studies | source audit | V1 ordering only | [V2](#v2-clustered-inference-studies) |
| Validation | V3 | Point-treatment weight studies | source audit | V2 ordering only | [V3](#v3-point-treatment-weight-studies) |
| Validation | V4 | Controlled direct-effect studies | source audit | V3 ordering only | [V4](#v4-controlled-direct-effect-studies) |
| Validation | V5 | Weighted longitudinal studies | source audit | V4 establishes the fixed-weight study design | [V5](#v5-weighted-longitudinal-studies) |
| Validation | V6 | Fold-evaluated CV-TMLE comparator | source audit | V5 ordering only | [V6](#v6-fold-evaluated-cv-tmle-comparator) |
| Validation | V7 | Selector-based multi-arm C-TMLE comparator | source audit | V6 ordering only | [V7](#v7-selector-based-multi-arm-c-tmle-comparator) |
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
| Later candidates | C4 | Sequential doubly robust longitudinal estimation | published support; pending source read | C3 ordering only | [C4](#c4-sequential-doubly-robust-longitudinal-estimation) |
| Later candidates | C5 | Natural and interventional mediation effects | published support; pending source read | C4 ordering only | [C5](#c5-natural-and-interventional-mediation-effects) |
| Later candidates | C6 | Continuous-time survival and competing risks | published support; pending source read | C5 ordering only | [C6](#c6-continuous-time-survival-and-competing-risks) |
| Later candidates | C7 | Two-phase and outcome-dependent sampling | published support; pending source read | C6 ordering only | [C7](#c7-two-phase-and-outcome-dependent-sampling) |

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

### V1. Fold-repeat studies

Register one repeated point-treatment cross-fitted TMLE row. Name it
`repeated point-treatment cross-fitted TMLE`, with slug `repeated-crossfit-tmle`.

The row uses pooled stacked targeting and whole-sample plug-in evaluation. It uses five outer
folds and three independent fold draws. It averages repeat estimates and rowwise influence curves
before inference. It averages risk ratios and odds ratios on their log scales. It reports
pointwise 95% intervals.

Reuse the binary and bounded-continuous laws from the ordinary point-treatment study. Include each
applicable point-treatment estimand. Run 800 replications per law at a sample size of 1,000. Use
fixed GLM nuisance families, fixed propensity bounds, and a new study seed. Retain every
replication, and reject each failed fit.

Reuse the shared CV-TMLE property laws, cells, seeds, summaries, and verdicts. Positive cells use
three repeats. Cover double robustness, root-n convergence, interval calibration, type-I error,
power, and flexible-tree cross-fitting. Keep the existing both-wrong, invalid-inference,
nonzero-effect, and in-sample controls.

Add one repeat-specific paired property family on a fixed nonlinear sample. Use ATT so whole-sample
and equal-fold evaluation differ. Fix the sample size at 600, and vary only the fold seed across
200 paired replications.

| cell | role | construction |
| --- | --- | --- |
| `rowwise_three_draw_average` | positive | pooled CV-TMLE with three independent fold draws |
| `one_fixed_split` | control | the same estimator with one fold draw |
| `equal_fold_average` | control | one fold draw with fold plug-in evaluation |

Gate the positive cell with 99% paired-bootstrap upper bounds. Require each spread ratio below
0.80. Compare the repeated estimate's standard deviation with each control's standard deviation.
Both controls must fail the same bound near a ratio of one. The independent-draw reference is
`1 / sqrt(3)`, while the declared bound allows fold correlation and Monte Carlo error.

Add a decision study for mean and median repeat aggregation. Fit five identical fold draws for
both rules. Run 400 replications at a sample size of 1,000. The specificity control uses correct
oracle outcome and treatment nuisances on the stress law. Require the mean-to-median RMSE upper
bound below 1.10.

Use a natural split-instability law for the stress regime. A tail stratum has about 4% probability,
a treatment probability of 0.12, and strong treatment-effect modification. Other rows have a
treatment probability near 0.50. Fit a fully grown outcome tree and the exact treatment mechanism.
Require the median-to-mean RMSE upper bound below 0.95. Also report their paired 90th-percentile
absolute-error ratio without adding a second acceptance margin.

Compare two variance rules on the ordinary and stressed means. The current rule uses the variance
of the row-aligned averaged influence curve. The DML mean rule averages each draw's squared
standard error plus its squared distance from the aggregated estimate. Both rules use the same
samples, folds, point estimates, and nominal 95% interval.

Require both specificity-control intervals to clear the 0.90 coverage floor and the 0.80 to 1.20
SE screen. Require the adjusted stress interval to meet those rules. The unadjusted stress control must
resolve below nominal coverage or below the SE screen. Publish red results under reporting policy
rather than moving the law, mutation, budgets, or margins.

Keep the shared 99% Monte Carlo confidence level and the existing CV-TMLE margins. The
standardized-bias margin is 0.25. The coverage floor is 0.90. The SE sanity band is 0.80 to 1.20.
The calibration SE band is 0.93 to 1.07. The calibration coverage band is 0.92 to 0.98.

The type-I ceiling is 0.10, and minimum power is 0.80.

The root-n slope band is -0.625 to -0.375, and it excludes -0.25.

No maintained implementation matches the complete declared repeat rule. Chernozhukov et al.
(2018) define mean and median aggregation across fixed repeated partitions. Their mean variance
adds within-partition variance and between-partition squared deviation. `cleverly` instead uses
the variance of its row-aligned averaged influence curve.

R `tmle3`, `drtmle`, and `lmtp` use one split or one pooled split. Current `DoubleML` uses a median
and a between-partition adjustment. R `Crossfit` uses a median under double cross-fitting. Python
`zepid` defaults to a median for a different fold-targeted construction. Record these exclusions,
set the reference to null, and publish a zero-row equivalence artifact.

Keep the implementation DRY. Parameterize the shared CV-TMLE property runner by repeat count.
Extract shared point-treatment result conversion where needed. Put the paired spread and RMSE ratio
intervals and their verdicts in the generic evidence machinery. Reuse the fitted draw reports for
both aggregation and variance rules. Do not copy inference-scale, truth, interval, resampling,
schema, or publication code.

Publish the registered row, its method-evidence page, generated grid tables, and all seven required
artifacts. State that the row does not validate clustering, observation weights, missing outcomes,
longitudinal data, or an external implementation. Document the rejected comparator candidates in
the benchmarking survey.

Run disposable primary and fixed-sample probes before the declared run. Use them only to confirm
schema integrity, successful fitting, and nonzero fold noise. Do not move a declared margin after
the full run. Regenerate the registered artifacts and generated documentation once.

Verify the artifact hashes, registered verdicts, generated prose, and the named slow re-execution.
Run the targeted evidence tests, Ruff checks, mypy, the prose report, the complete fast tier, and
the warning-as-error documentation build.

After the row passes, remove this completed proposal and promote the remaining validation items.
Change the uncovered-family count from five to four. Remove only the replaced fold-repeat legacy
study, and update the legacy count. Keep unrelated legacy studies and existing artifacts intact.

### V2. Clustered inference studies

Validate cluster-level covariance and fold integrity under genuine within-cluster dependence. The
negative control must analyze the same rows as independent observations.

Three pinned implementations accept a cluster identifier. R `tmle` 2.1.1 and R `ltmle` 1.3-0 accept
`id=`, and R `lmtp` 1.5.4 accepts a cluster identifier column. Choose the one whose data layout
matches the study.

### V3. Point-treatment weight studies

Validate fixed probability weights against the tilted population law. The negative control must
omit the weights and converge to a different parameter.

R `tmle` 2.1.1 accepts `obsWeights=` for a biased sampling design. That package is pinned already,
so this study needs no new container.

### V4. Controlled direct-effect studies

Validate each declared intermediate level against its exact controlled parameter. The study must
exercise the treatment and intermediate mechanism product with a nonzero control.

R `tmle` 2.1.1 estimates a controlled direct effect when `Z` is a binary intermediate. It returns
one fit for `Z = 0` and one fit for `Z = 1`, which are the two declared levels. This study can
therefore carry a paired comparison rather than an empty equivalence artifact. The comparator is
binary, so a further intermediate level stays outside the paired cells.

### V5. Weighted longitudinal studies

Validate fixed weights through nuisance fitting, targeting, plug-in averaging, and covariance. The
negative control must omit the weights and miss the declared longitudinal parameter.

R `ltmle` 1.3-0 accepts `observation.weights=`, and R `lmtp` 1.5.4 accepts sampling `weights`. Both
packages are pinned already.

### V6. Fold-evaluated CV-TMLE comparator

The published fold-evaluated row claims no comparator, and that claim holds for the pooled update.
Python `zepid` ships the fold-targeted construction instead. It fits one coefficient inside each
split, and it averages the fold plug-ins by split size. Its variance is the mean of the
within-split influence-curve variances over the total sample size. That equals this package's
cross-validated variance at equal fold sizes.

Register a second study for `targeting_scheme="fold"` with `cv_evaluation=True`. A study record
names one reference, so do not move a comparator onto the published row. Two exclusions follow.
`zepid` reports the ATE, the risk ratio, and the odds ratio, so `att` and `atc` stay outside the
paired cells. The run must set the mean combination over one partition, because the median default
is the aggregation this package refuses.

### V7. Selector-based multi-arm C-TMLE comparator

The published selector row claims no comparator. R `ctmle` 0.1.2 documents its treatment as a
binary indicator, and archived R `ctmle3` ships the outcome-adaptive specification alone. Julia
`TMLE.jl` ships a greedy strategy and a pre-ordered adaptive-correlation strategy. It accepts
categorical treatment levels, it stratifies folds by treatment, and it selects a candidate on a
cross-validated loss. It composes the risk ratio and the odds ratio by the delta method over a
joint estimand.

Two consequences follow. `TMLE.jl` has no discrete ladder, so the greedy and ordered scenarios need
a separate record from the discrete scenario. This is the first comparator outside R, so it needs a
pinned Julia image beside the existing R images.

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

The four items below add methods rather than studies. Each one names the maintained implementation
that a paired study would use. A named comparator is provenance for the construction. It is not the
derivation, and it is not the acceptance gate.

### C4. Sequential doubly robust longitudinal estimation

`lmtp_sdr` implements the sequentially doubly robust estimator of Díaz, Williams, Hoffman and
Schenck (2023). That estimator is consistent when either the outcome regression or the treatment
mechanism is consistent at each time point. It is a second estimator over registered longitudinal
targets, so it adds no estimand. The comparator is the pinned R `lmtp` 1.5.4 that the longitudinal
rows already use, and no new container is needed. Read the rate conditions its interval claims
first, because they differ from the sequential regression conditions.

### C5. Natural and interventional mediation effects

`cleverly` reports controlled direct effects only. Natural and interventional direct and indirect
effects are separate estimands, and each carries its own identification assumptions. Díaz, Hejazi,
Rudolph and van der Laan (2021) derive the interventional effects and their efficient influence
function under an intermediate confounder. R `medoutcon` implements a cross-fitted one-step
estimator and a cross-validated TMLE for them, and it pins by commit. Read the identification, the
influence function, the targeting construction, and the interval conditions first-hand. Add each
accepted target to the oracle registry and the evidence gates in both directions.

### C6. Continuous-time survival and competing risks

The shipped survival and competing-risk estimators use discrete time nodes. Rytgaard, Gerds and van
der Laan (2022) derive the continuous-time construction, which changes the intensity model, the
targeting step, and the remainder. CRAN `concrete` 1.0.5 is the comparator, and it implements the
one-step form of Rytgaard, Eriksson and van der Laan (2023).

That comparator takes a binary baseline treatment under a static or dynamic intervention, which
bounds the paired cells a first study can claim. A discrete-time study is not evidence for a
continuous-time interval, so the existing longitudinal rows do not transfer.

### C7. Two-phase and outcome-dependent sampling

A two-phase design measures some variables on a subsample only. An outcome-dependent design samples
on the outcome itself. Each design changes the observed-data likelihood, so each needs its own
influence-function correction.

Hejazi, van der Laan, Janes, Gilbert and Benkeser (2021) derive the
two-phase correction, and R `txshift` 0.3.8 implements it. Van der Laan (2008) derives case-control
weighting under a known prevalence, and Julia `TMLE.jl` implements it. Fixed observation weights do
not replace either correction. The comparator survey rejects `txshift` as a second opinion on
continuous shifts. That verdict does not carry here, because the two-phase correction is a
different feature.

## Reading a gap correctly

Not every absence is missing package functionality. The refusal taxonomy in
[How to read a refusal](technical-reference/scope-and-refusals.md#how-to-read-a-refusal) distinguishes an unimplemented
well-posed feature from a different causal question and a method that would be wrong by
construction. Only the first belongs on this roadmap.
