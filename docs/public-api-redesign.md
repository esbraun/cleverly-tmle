# Public API and extensibility redesign

Status: pre-implementation design plan

Decision date: 2026-08-17

Implementation status: not started

This document is the required planning gate for the public API redesign. No implementation of the
redesign should begin until this document has been reviewed and accepted. Once accepted, changes
should follow the ordered work packages and evidence gates below rather than treating the example
syntax as permission to bypass the scientific contracts in
[`architecture-invariants.md`](architecture-invariants.md) or
[`evidence.md`](evidence.md).

The redesign is intentionally allowed to break the current alpha API. It should preserve validated
statistical behavior where the estimand and method are unchanged, but it need not preserve the
constructor layout, return containers, string-based target declarations, or top-level exports.

Every fenced example below is tagged `text` rather than `python`, and must stay that way until the
work packages land. A `python`-tagged fence anywhere in the tree has to be registered as an
executable `doc-section`/`doc-block`:
`tests/e2e/test_doc_snippets.py::test_every_python_block_has_execution_metadata` counts the raw
fence markers against the registered blocks, and it is in the **fast** tier. Registering these would
mean executing `from cleverly import ATE, CausalStudy, PointTreatment` against a package that has
none of them, and a `catalogue:` marker cannot exempt them either, because they name no receiver in
that module's `RECEIVERS`. The syntax here is a proposal, so it is quoted rather than run;
`docs/user-guide.md` is where a fence is a promise the suite keeps.

Note that the count is a plain substring search and cannot tell quotation from code, so writing the
fence marker inline in a sentence fails the test just as an unregistered example would. That is why
the paragraph above names the tag instead of showing it.

## 1. Goals

The new API should be understandable to an analyst who knows causal inference but does not yet know
TMLE terminology. The first decisions a user sees should therefore be the causal question, design,
identification assumptions, and estimand—not fluctuations, clever covariates, nuisance bounds, or
score groups.

The design must also support the following roadmap without another architectural reset:

- propensity/density-derived and directly learned Riesz representers as alternative ways to build
  the orthogonalization and targeting weights;
- nested Riesz functionals for longitudinal treatment, missingness, and other evidence-backed
  sequential parameters;
- efficient plug-in (EP) learning for heterogeneous causal contrasts;
- sensitivity analysis, validation, refutation, and simulation after fitting;
- optional DAG-based identification before estimation;
- optional integration with DoWhy without making DoWhy a core dependency;
- new estimands and estimation methods through typed public extension points;
- pandas and polars input/output behavior, influence-curve inference, persistence, provenance,
  clustering, weighting, and the repository's existing scientific evidence requirements.

### 1.1 Success criteria

The redesign is successful when all of the following are true:

1. A basic ATE analysis can be read in causal-workflow order: create a study, identify an effect,
   estimate it, then assess it.
2. The result explains the causal estimand and identification assumptions without requiring the
   user to understand the TMLE implementation.
3. An estimation method cannot silently change the adjustment set or causal estimand selected
   during identification.
4. Analytic propensity-based TMLE and direct-Riesz TMLE share downstream inference and result
   infrastructure while retaining method-appropriate diagnostics.
5. A heterogeneous EP result is not misrepresented as a scalar `ParameterEstimate`.
6. Point-treatment and longitudinal fits share a documented result protocol even when some
   operations are explicitly unavailable.
7. Every sensitivity or validation operation declares which fitted artifacts and scientific
   assumptions it requires.
8. Unsupported graph/function/method compositions fail before nuisance fitting, with an error that
   names the missing identification result, fitted artifact, or published derivation.
9. The old and new APIs do not coexist as two independent computational paths.
10. Existing numerical behavior is preserved for unchanged, already-supported estimands until an
    intentional and independently evidenced change says otherwise.

## 2. Non-goals

This redesign will not:

- implement causal discovery;
- infer a DAG from column names or observed correlations;
- make a graph mandatory when the user already supplies an identified adjustment set;
- claim that the existence of a Riesz representation is by itself sufficient evidence for a new
  causal estimand or confidence interval;
- make direct Riesz learning eliminate the need for an outcome regression;
- expose arbitrary formula evaluation or unvalidated expression strings as the principal Python
  extension API;
- automatically select an estimation method from the data;
- treat refutation tests as proofs that the identifying assumptions hold;
- preserve every current alpha spelling indefinitely;
- use parity with DoWhy, DoubleML, RieszCML, or another package as the acceptance criterion for a
  mathematical change.

## 3. Current API assessment

### 3.1 Strengths to preserve

The current implementation already has several valuable seams:

- A fitted point-treatment result retains influence curves, nuisance predictions, targeting state,
  fold plans, configuration, and provenance.
- `result.sensitivity` and `result.validation` correctly make assessment a post-fit operation.
- The target registry separates a reported functional from the fluctuation group used to target
  it.
- The submodel registry separates score equations from estimand names.
- Inference is mostly downstream of influence curves, allowing covariance, contrasts, clustered
  variance, simultaneous bands, and multiplier bootstrap to be shared.
- `TMLE.retarget` makes several cached sensitivity operations inexpensive.
- Point-treatment estimator variants reuse the common result and assessment surface when they
  change only nuisance selection.
- Dataframe backend preservation, serialization boundaries, refusal tests, and the bidirectional
  oracle/evidence gates are already explicit contracts.

These should be reused rather than replaced merely to make the public syntax look different.

### 3.2 Problems to address

The current `TMLE` constructor combines too many independent choices:

- learner selection;
- causal estimands;
- interventions, shifts, incremental interventions, and MSMs;
- cross-fitting and repeated splitting;
- fluctuation and targeting algorithms;
- clipping and bounds;
- inference and bootstrap configuration;
- screening;
- runtime and parallel settings.

Variable roles are then supplied separately to `fit()`. As a result, the complete scientific
question is not represented by any one public object before fitting.

Other usability and extensibility problems are:

- `fit().single()` exposes a return-type workaround in the ordinary quickstart.
- String estimands carry little structured information and require separate code to recover arm or
  regime metadata.
- The mutually exclusive parameter-axis mechanism is appropriate internally but hard for users to
  discover.
- Point and longitudinal estimators have different result classes and assessment support.
- Public registration begins after identification, leaving no natural place for graph-derived
  adjustment sets or a stored identification result.
- Names such as `treatment_learner`, `propensity`, and `g_bounds` assume an analytic mechanism-based
  representer.
- Advanced extension documentation starts with TMLE internals (`Target.group`, submodels, clever
  covariates) rather than with the mathematical functional and its identification.
- The functional API `tmle(Y, A, W, ...)` uses an R-oriented vocabulary that does not align with
  the rest of the Python workflow.

## 4. External design review

### 4.1 DoWhy

DoWhy's main contribution to this design is the explicit workflow split between causal model,
identification, estimation, and refutation. It also demonstrates the value of an inspectable
identified estimand and an estimator adapter ecosystem.

Advantages of adopting that workflow shape:

- identification assumptions become visible before fitting;
- a DAG can determine or validate adjustment variables;
- estimation backends do not need to own graph algorithms;
- refutation is framed as assessment of an identified estimate rather than as another estimator
  constructor option.

Disadvantages of making `cleverly` a DoWhy extension first:

- it would make a broad external framework a mandatory dependency for users who have an explicit
  adjustment set;
- DoWhy's generic estimate object cannot expose all targeting, cross-fitting, nuisance, influence,
  and replay state without a package-specific attachment;
- its dotted string method dispatch is weaker than a typed extension contract;
- its point-effect workflow does not naturally own `cleverly`'s longitudinal recursion and nested
  Riesz artifacts;
- coupling core data ingestion to DoWhy would threaten the existing narwhals/backend contract.

Decision: preserve the workflow separation, but keep `cleverly` standalone and integrate DoWhy
through optional adapters.

### 4.2 DoubleML

DoubleML demonstrates a consistent fitted-method surface for inference, bootstrap, tuning, and
omitted-confounder sensitivity. Its separation between model-specific scores and base-class
inference is close to `cleverly`'s separation between targets/targeting and influence-curve
inference.

Useful lessons:

- fitted assessment methods are highly discoverable;
- Riesz sensitivity elements belong with the fitted orthogonal score;
- a common result/inference layer can support many model classes;
- explicit score callables are a useful advanced escape hatch.

Design cautions:

- many model-specific top-level classes make method discovery harder as the catalog grows;
- methods that mutate and return the fitted object make cached analyses and serialization state
  harder to reason about;
- a score-centric API still assumes the user knows which statistical model class matches the
  causal question.

Decision: adopt fitted assessment and a shared orthogonal-result layer, but return immutable report
objects and keep the causal question ahead of the estimation method.

### 4.3 diff-diff

`diff-diff` demonstrates an approachable sklearn-style fit, statsmodels-style result summaries,
diagnostic marker types, a combined diagnostic report, and contextual practitioner next steps.

Useful lessons:

- results should use a consistent canonical output schema;
- diagnostic outputs should be distinguishable by type from estimator results;
- a combined report should say which checks ran, failed, passed, or did not apply;
- context-aware next steps are more useful than a flat catalog of every possible diagnostic.

Design cautions:

- a diagnostic-report constructor with a growing list of `run_*` flags becomes another oversized
  configuration surface;
- heterogeneous result classes can force capability routing by result-class name;
- sklearn compatibility is useful for learner components but does not by itself express causal
  identification.

Decision: adopt immutable, typed reports and structured next steps. Use capabilities rather than
class-name branches or a large boolean constructor.

### 4.4 RieszCML

RieszCML is the principal method implementation reference for Riesz integration with TMLE. Its
companion paper is the governing derivation to audit before implementation. The important
decomposition is:

- `f`: the regression evaluated on observed data;
- `h`: the plug-in transformation, often the regression evaluated under an intervention;
- `alpha`: the Riesz representer evaluated on observed data;
- `alpha_star`: the representer evaluated in the intervention/counterfactual state used to update
  plug-in predictions;
- an uncentered influence expression, commonly `h + alpha * (Y - f)`;
- stage composition for nested functionals, with cumulative products of representers.

Useful lessons:

- one functional-stage abstraction can support one-step and targeted estimators;
- analytic inverse-probability/density ratios and directly learned representers can feed the same
  functional representation;
- nested regressions compose innermost-first while residual weights accumulate representers in the
  mathematically required direction;
- `alpha_star` is separate from observed-data `alpha` and is essential to a correct plug-in update;
- a catalog of evidenced functionals should be separate from the generic engine;
- smooth contrasts can remain downstream through the delta method.

Design cautions:

- R formulas and named numeric dictionaries are not an appropriate primary Python public API;
- row-aligned external arrays require identity, fold, and counterfactual-evaluation provenance;
- a generic functional builder must not allow users to obtain an apparently certified causal
  interval from incomplete identification or influence-function information;
- RieszCML comparison is secondary evidence only.

Decision: follow its method and compositional structure, but express it through typed Python
protocols, staged capability validation, and `cleverly`'s independent evidence gates.

### 4.5 EP learner

The EP learner estimates heterogeneous causal contrasts such as the conditional average treatment
effect and conditional relative risk. It targets an efficient plug-in estimate of a population
risk, rather than merely replacing the propensity learner in an ordinary scalar TMLE.

Decision: implement EP as a distinct method for a conditional-contrast estimand. It may share
study, identification, nuisance, cross-fitting, reporting, and validation infrastructure, but its
result must expose predictions and conditional-risk diagnostics rather than pretending to be a
single scalar estimate.

## 5. Architectural decisions

### 5.1 The beginner-facing root is `CausalStudy`

Recommended basic workflow:

```text
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    data,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=["W1", "W2", "W3", "W4"],
    ),
)

effect = study.identify(ATE())
result = effect.estimate(method="tmle", random_state=0)

print(effect.summary())
print(result.summary())
print(result.validate().summary())
print(result.sensitivity.omitted_confounding().summary())
```

Why this is recommended:

- the first object represents the study rather than one computational algorithm;
- roles and data are validated once;
- the identified effect becomes inspectable before estimation;
- later estimators cannot silently select a different adjustment set;
- graph identification fits before `estimate()` without changing the estimation API;
- multiple estimation methods can be compared on the same identified effect.

Costs:

- the minimal workflow has an explicit identification step;
- data is bound to the study object, which requires careful memory and serialization handling;
- method developers must consume a richer typed contract than the current `fit(data, roles...)`.

Mitigation:

- keep only the backend name and validated arrays in persistent artifacts, not an unnecessary copy
  of the caller's full dataframe;
- provide a concise `study.estimate(ATE(), method="tmle")` convenience that internally creates and
  stores the same `IdentifiedEffect`, while documenting the explicit two-step workflow first;
- require `result.identified_effect` regardless of which convenience path was used.

### 5.2 Identification is a first-class object

`study.identify(estimand)` returns `IdentifiedEffect` with:

- the requested causal estimand;
- its identified observed-data functional;
- adjustment variables or sequential history;
- identification provider and graph provenance;
- stated assumptions;
- support/positivity requirements;
- required observed variables;
- compatible estimation-method capabilities;
- explicit refusals for known incompatible methods.

An estimator receives `IdentifiedEffect`, never only a string such as `"ate"`.

### 5.3 Graph support is optional and delegated initially

Core provider:

- `ExplicitAdjustmentProvider`, used when the design supplies an adjustment set or known randomized
  assignment.

Optional provider:

- `DoWhyIdentificationProvider`, installed through `cleverly[dowhy]`.

Rules:

- a graph is optional;
- graph-based identification occurs before fitting;
- supplying both a graph and adjustment set means "validate this proposed set" rather than "pick
  whichever is convenient";
- a disagreement is an error unless the user chooses a different valid set explicitly;
- front-door, IV, transport, mediation, or unidentified results are refused until a matching
  `cleverly` functional and estimator have been evidenced;
- no causal discovery is performed.

### 5.4 Estimands are typed

Initial public examples include:

- `ATE()`
- `ATT()` and `ATC()`
- `CounterfactualMean(treatment=...)`
- `RegimeMean(regime=...)`
- `ModifiedTreatmentPolicy(shift=...)`
- `IncrementalEffect(delta=...)`
- `MSMProjection(model=...)`
- `ConditionalContrast(kind=..., modifiers=...)`

Each estimand declares:

- a human-readable definition;
- structured parameter keys;
- applicable designs;
- identification requirements;
- the observed-data functional or nested functional stages;
- supported estimation-method capabilities;
- inference scale and transformations;
- scientific references and evidence identifier.

String aliases such as `"ate"` may exist in the convenience layer, but must resolve immediately to
a typed object. Strings must not drive internal branching throughout the estimator.

### 5.5 Public extensions use typed protocols

Public extension points:

- `Estimand`
- `IdentificationProvider`
- `EstimationMethod`
- `OutcomeStrategy`
- `MechanismStrategy`
- `RepresenterStrategy`
- `CrossFitStrategy`
- `TargetingStrategy`
- `InferenceStrategy`
- `Diagnostic`
- `SensitivityMethod`

Advantages:

- required behavior is discoverable and type-checkable;
- capability validation can happen before fitting;
- method-specific fitted artifacts have stable meanings;
- custom methods do not need to mutate global dictionaries at import time.

Costs:

- more interfaces must be designed and maintained;
- a typed protocol can become overly abstract if created before two real implementations need it;
- custom callables still need an escape hatch.

Mitigation:

- extract each protocol from at least the current analytic implementation and one planned
  alternative;
- keep protocols behavior-oriented and small;
- allow typed callables inside strategy objects;
- mark custom callable configurations as non-reconstructible for persistence rather than silently
  substituting defaults.

### 5.6 Named presets coexist with typed method objects

Ordinary selection:

```text
effect.estimate(method="tmle")
effect.estimate(method="riesz_tmle")
effect.estimate(method="ep")
```

Advanced selection:

```text
from cleverly.methods import DirectRiesz, RieszTMLEMethod

method = RieszTMLEMethod(
    outcome=outcome_strategy,
    representer=DirectRiesz(riesz_learner),
    targeting="logistic",
    cross_fitting={"folds": 10, "repeats": 2},
)

result = effect.estimate(method=method)
```

Preset definitions:

- `"tmle"`: analytic mechanism-derived representers using propensity, density, missingness,
  censoring, or related mechanism models as required by the functional;
- `"riesz_tmle"`: directly learned or explicitly provided representers plus outcome regressions and
  targeting;
- `"ep"`: efficient plug-in learning for a compatible `ConditionalContrast`;
- the collaborative and doubly-robust-inference variants become typed strategies rather than new
  study types, keeping the names -- `CollaborativeTMLEMethod`, `DRTMLEMethod` -- that already say
  which estimator they are.

No automatic method selection will be added. A method may recommend alternatives after capability
validation, but it may not silently substitute another estimator.

### 5.7 Configuration is split by concern

Replace the monolithic estimator constructor with immutable configuration groups:

- `ModelSpec`
- `CrossFitting`
- `Targeting`
- `Inference`
- `Runtime`

The convenience API may accept common keyword shortcuts, but must normalize them into these
objects before fitting. The normalized configuration is stored on the result and drives
serialization and reproducibility.

Scientific choices and runtime choices must remain distinguishable. Changing `n_jobs` must not
change the represented estimator; changing the cross-fit scheme or targeting construction may.

## 6. Proposed public objects and contracts

### 6.1 `CausalStudy`

Responsibilities:

- own validated design roles and access to the analysis data;
- select an identification provider;
- identify one or more compatible estimands;
- record graph/adjustment provenance;
- provide no estimation logic of its own.

It must not:

- choose nuisance learners;
- choose a targeting algorithm;
- run diagnostics automatically;
- mutate the input frame.

### 6.2 Design objects

`PointTreatment` fields should cover:

- outcome;
- treatment;
- adjustment variables;
- optional missingness indicator;
- optional intermediate variable only while the controlled-direct-effect path remains supported;
- observation weights and their interpretation;
- cluster identifier;
- strata;
- known assignment information;
- treatment kind.

`LongitudinalTreatment` fields should cover:

- outcome node or nodes;
- treatment nodes;
- baseline variables;
- time-varying histories;
- censoring nodes;
- identifier and cluster structure;
- observation weights;
- horizons;
- survival/competing-event metadata;
- ordering and history-availability rules.

The design owns roles. The estimand owns the intervention or causal contrast. The method owns how
the identified functional is estimated.

### 6.3 `IdentifiedEffect`

Required methods and properties:

- `estimand`
- `functional`
- `identification`
- `available_methods()`
- `summary()`
- `estimate(...)`

`available_methods()` must return structured capability records rather than only strings. An
unavailable method includes the reason, such as a missing `alpha_star` evaluation, unsupported
front-door functional, or absent conditional-contrast theory.

### 6.4 `CausalResult`

Shared result protocol:

- `estimates`
- `estimate` and `psi` for a singleton result;
- stable parameter indexing;
- `summary()`;
- `to_frame()`;
- `influence_curves` when defined;
- `covariance()`;
- `contrast()`;
- `identified_effect`;
- `method` and normalized configuration;
- `provenance`;
- `diagnostics`;
- `validate()`;
- `sensitivity`;
- `save()`.

Internally, estimates use structured `ParameterKey` objects. User-facing indexing may retain stable
aliases such as `result["ate"]`, but arm, regimen, horizon, cause, or MSM-term metadata must never
be reconstructed by parsing the alias.

An ordinary singleton fit returns `CausalResult` directly. There is no primary `ResultSet.single()`
step. A fit with several parameters is the same result type with several structured entries.

### 6.5 Conditional-effect result

An EP result satisfies the common reporting/provenance contract and additionally exposes:

- `predict(data)`;
- conditional-effect predictions;
- modifier schema;
- fitted risk and targeting artifacts;
- out-of-fold risk;
- calibration and stability diagnostics.

It must not expose scalar Wald inference unless a specific scalar functional of the learned curve
has its own supported influence-function construction.

## 7. Riesz integration design

### 7.1 Scientific boundary

Direct Riesz learning replaces the analytic construction of a representer from propensity or
density components. It does not generally replace the outcome regression. The core single-stage
form still contains:

```text
plug-in term h + representer alpha * residual (Y - f)
```

Documentation, configuration names, and errors must preserve that distinction.

### 7.2 Functional stage

Introduce an immutable internal/public-advanced `FunctionalStage` with:

- `regression`: definition and fitted artifact for observed regression `f`;
- `plugin_map`: counterfactual/intervention evaluation `h`;
- `representer_problem`: analytic, learned, or provided source for observed `alpha`;
- `intervention_evaluation`: how to evaluate `alpha_star` under the intervention defining `h`;
- `residual_source`: observed outcome or the inner stage's targeted plug-in value;
- `identification`: assumptions and references specific to this stage;
- `targeting_map`: which regression predictions are updated and how;
- structured stage name and history metadata.

Built-in estimands construct these stages internally. An advanced custom builder must validate
input roles, output shapes, counterfactual evaluation, identification metadata, and influence
construction before fitting.

### 7.3 Representer strategies

Provide:

1. `AnalyticRepresenter`
   - derives `alpha` and `alpha_star` from fitted mechanisms;
   - wraps the current propensity/density/missingness/censoring path;
   - preserves current overlap and truncation diagnostics.
2. `DirectRiesz`
   - fits the representer through a published Riesz loss or moment problem;
   - predicts both observed and intervention evaluations using the same fold-trained model;
   - stores balance/moment residuals, loss, basis/model information, and fold provenance.
3. `ProvidedRepresenter`
   - accepts an external fit or predictions only with explicit row identity, folds, training
     provenance, and counterfactual-evaluation support;
   - refuses a bare aligned array when its relationship to the analysis rows cannot be established.
4. `ComposedRepresenter`
   - internal stagewise cumulative products for nested functionals;
   - records every component separately for diagnosis rather than storing only the final product.

### 7.4 Cross-fitting requirements

Default behavior:

- one outer fold plan is shared across outcome and representer stages for reproducibility and
  stagewise diagnostics;
- every row's prediction comes from models that did not train on that row;
- learner-internal tuning remains nested inside the outer training fold;
- repeated splitting averages the same identified functional with coherent influence curves;
- longitudinal stages respect history and clustering restrictions;
- user-provided folds are validated for coverage, disjointness, grouping, and row identity.

Allow independent nuisance folds only through an explicit advanced option whose theoretical
validity and diagnostic implications are documented.

### 7.5 Nested composition

The engine is implemented broadly from its first Riesz milestone rather than being hard-coded to
point treatment. Public catalog exposure remains gated.

For stages supplied innermost-first, the engine must:

1. fit the innermost regression and its observed/counterfactual evaluations;
2. make its plug-in output the residual source for the next stage;
3. fit all remaining stages using only their permitted history;
4. evaluate observed `alpha_j` and intervention `alpha_star_j` out of fold;
5. construct the cumulative observed products in the direction specified by the derivation;
6. construct the corresponding intervention products separately;
7. target regressions innermost-first;
8. update plug-in predictions with intervention products, not the observed-data products;
9. build the final plug-in estimate from the outermost targeted stage;
10. construct the exact paper-derived influence curve and score checks;
11. store every stage's predictions, representers, cumulative products, targeting coefficients,
    folds, and convergence state.

Do not copy RieszCML's current variance or finite-sample choices without auditing them against the
paper. Record any paper/code discrepancy and add a nonzero regression or mutation test for the
chosen interpretation.

### 7.6 Initial evidence-gated catalog

The first public catalog should include only functionals for which the full contract has been
established:

- point-treatment counterfactual means;
- ATE and supported transformations/contrasts;
- ATT/ATC only after their ratio or conditional-functional construction is fully represented;
- missing-at-random means and existing supported missing-outcome effects;
- evidenced static and dynamic longitudinal regimen means;
- modified treatment policies already supported by published theory and current evidence;
- smooth contrasts through the existing delta-method layer.

The general engine may be able to express more. That is not permission to register more.

Initially gated or refused:

- mediation targets pending target-specific identification, Riesz-stage, remainder, and inference
  audits;
- new competing-event interventions;
- arbitrary two-phase sampling estimands;
- arbitrary nonlinear nested functionals;
- any custom functional lacking an evidence identifier and complete influence construction.

## 8. EP learner design

EP is a later work package built on the new contracts, not part of the Riesz-TMLE engine itself.

Shared infrastructure:

- study and identification objects;
- outcome/treatment nuisance strategies;
- cross-fitting;
- dataframe/backend handling;
- provenance and persistence;
- refutation and simulation harnesses;
- capability-aware assessment.

EP-specific infrastructure:

- `ConditionalContrast` estimands;
- modifier design and schema;
- sieve/basis strategy for infinite-dimensional targeting;
- efficient plug-in risk construction;
- bounded targeted outcome predictions;
- second-stage contrast learner;
- out-of-fold risk and calibration;
- conditional-effect prediction result.

First supported contrasts should be the paper-derived CATE and conditional relative risk. Other
losses or contrasts require their own derivation and evidence rather than a generic `kind=` branch.

An aggregate of an EP curve is a separate parameter. It receives scalar inference only when the
aggregate's influence contribution is implemented and tested.

## 9. Validation, diagnostics, sensitivity, and refutation

### 9.1 Public workflow

```text
validation = result.validate()
full_diagnostics = result.diagnostics.run_all(include_refits=True)

overlap = result.diagnostics.support()
nuisance = result.diagnostics.nuisance_models()
scores = result.diagnostics.score_equations()
refutation = result.diagnostics.refute()

confounding = result.sensitivity.omitted_confounding()
benchmark = result.sensitivity.benchmark(["W1", "W2"])
missingness = result.sensitivity.missingness()
all_sensitivity = result.sensitivity.run_all()
```

`validate()` runs inexpensive method-appropriate checks and returns an immutable
`ValidationReport`. It does not run refits or long simulations.

Every report item uses one of:

- `passed`;
- `failed`;
- `warning`;
- `not_applicable`;
- `unavailable`.

`not_applicable` means the question does not apply to this identified functional.
`unavailable` means the question is relevant but the required artifact or derivation is absent.
The distinction must appear in serialized output.

### 9.2 Capability declaration

Each operation declares:

- supported result/functional capabilities;
- required stored artifacts;
- whether it retargets cached artifacts, refits models, or only summarizes state;
- whether it is deterministic from a saved result;
- method-specific interpretation;
- expected computational cost.

The capability registry is checked in both directions: every public result family is covered by
every assessment operation, and every exemption names a live result/method and a reason.

### 9.3 Method-appropriate support diagnostics

Analytic mechanism-based TMLE:

- propensity/density distributions;
- treatment/regime support;
- effective sample size;
- mechanism truncation;
- clever-covariate leverage.

Direct-Riesz TMLE:

- representer tail behavior;
- signed and absolute weight concentration;
- effective sample size where meaningful;
- Riesz loss and moment/balance residuals;
- counterfactual representer evaluation;
- cumulative stage leverage.

A direct-Riesz-only fit must not report a propensity distribution it never estimated. Likewise, a
regularization or representer-clipping curve must not be called a propensity truncation curve.

Longitudinal/nested fits:

- stagewise score equations;
- cumulative representer products;
- history-specific support;
- follower/at-risk sample sizes;
- recursion convergence;
- node-specific nuisance quality.

EP fits:

- targeted risk equation;
- out-of-fold risk;
- boundedness;
- conditional calibration;
- modifier support;
- split and basis stability.

### 9.4 Sensitivity

The omitted-confounder analysis should consume the representer actually used by the fitted method,
whether analytic or directly learned.

Other rules:

- sensitivity is resolved against a structured parameter key, not a parsed display name;
- missingness tilts remain limited to functionals with published identification under the tilt;
- mechanism-bound curves remain distinct from direct-Riesz regularization sensitivity;
- benchmarking that refits models must require a reconstructible method recipe;
- saved results may replay cached retargeting analyses but must refuse operations requiring lost
  custom callables;
- combined reports skip inapplicable families while explaining the omission.

### 9.5 Caching and persistence

Completed assessment results are cached by operation name plus normalized arguments. Caching must
not mutate the headline estimate or silently alter its summary.

Built-in strategies are fully serializable. Custom callables are permitted but marked
non-reconstructible. A saved result records separately whether it can:

- summarize existing artifacts;
- rerun targeting from cached nuisances;
- evaluate a stored representer on existing counterfactual designs;
- refit nuisances;
- evaluate new data.

## 10. DoWhy integration

### 10.1 Identification direction

`DoWhyIdentificationProvider` should:

- accept the supported graph representation through the optional dependency;
- invoke DoWhy identification;
- translate a supported backdoor estimand into `IdentifiedEffect`;
- preserve the original DoWhy identified estimand and graph/provider provenance;
- verify treatment, outcome, adjustment, and target population;
- refuse unsupported identification strategies before fitting.

### 10.2 Estimator direction

A DoWhy-compatible `cleverly` adapter should:

- receive a DoWhy `IdentifiedEstimand`;
- translate supported backdoor effects into typed `cleverly` estimands;
- run the normal estimation engine;
- construct the generic DoWhy estimate expected by its workflow;
- attach the native `CausalResult` for full diagnostics, sensitivity, influence curves, and
  provenance.

The DoWhy wrapper is intentionally less expressive than the native result. Native `cleverly`
documentation remains primary.

### 10.3 Dependency and compatibility policy

- add a separate `dowhy` optional dependency group;
- do not add DoWhy to `all` until installation size and compatibility have been assessed;
- **but add `cleverly[dowhy]` to the `dev` extra**, so section 13.8's version matrix and
  translation-equivalence checks actually execute in the ordinary local tiers. `dev` resolves to
  `cleverly[all]` plus tooling, so an extra kept out of `all` and out of `dev` is installed by no
  session and its tests can only skip. That is exactly the `numba`/`bench` trap already recorded in
  `pyproject.toml`, whose only mitigation was a dedicated CI job -- and this repository's stated
  budget condition rules CI out as a gate, so the mitigation is not available a second time. A
  skipped correctness check reads like a passing one;
- pin a tested compatible range rather than importing unbounded private APIs;
- isolate all imports under the integration package;
- test missing-dependency errors, which requires a way to run the suite *without* the extra --
  either a separate uninstalled session or a marker, decided when the group is added;
- document the supported DoWhy version range and translation limitations.

## 11. Breaking migration

The redesign will be a clean alpha break. It should not preserve duplicate top-level paths through
indefinite aliases.

Migration mapping:

| current API | new API |
| --- | --- |
| `TMLE(...).fit(data, ...).single()` | `CausalStudy(...).identify(...).estimate("tmle")` |
| `LTMLE(...).fit(...)` | `CausalStudy(design=LongitudinalTreatment(...)).identify(...).estimate("tmle")` |
| `CTMLE(...)` | `estimate(method=CollaborativeTMLEMethod(...))` |
| `DRTMLE(...)` | `estimate(method=DRTMLEMethod(...))` |
| `estimands=("ate", ...)` | typed estimand objects passed to `identify()` |
| `interventions=`, `shifts=`, `incremental=` | typed estimand/intervention declarations |
| `TMLEResultSet.single()` | removed; the fit already returns the `CausalResult` that `.single()` used to unwrap |
| `result.validation` | `result.diagnostics` plus `result.validate()` |
| `result.sensitivity` | retained as a capability-aware facade |
| `tmle(Y, A, W, ...)` | `CausalStudy` construction or an explicit migration helper |

One row is easy to misread and the migration guide should spell it out. `CausalResult.estimate` is
the singleton **point value** of section 6.4, not the object `.single()` returned; a mechanical
rewrite of `res = fit.single()` to `res = fit.estimate` turns every subsequent `res.summary()` into
an attribute error on a float. `.single()` has no replacement call because it has nothing left to
unwrap.

Required migration aids:

- a dedicated old-to-new guide with full runnable before/after examples;
- a constructor-argument mapping table;
- an explanation of changed defaults, if any;
- an optional static migration script for straightforward call shapes;
- a stable Git tag containing the final old API;
- no runtime alias that silently changes the identified estimand or method.

## 12. Ordered implementation work packages

### Work package 0: approve the design and scientific reading list

Deliverables:

- review and accept this document;
- record unresolved choices as explicit amendments here;
- read the Riesz TMLE paper first-hand;
- pin the RieszCML revision used as implementation provenance;
- read the current EP learner revision first-hand;
- update `references.md` with precise locators;
- identify which current estimands have direct support under the Riesz construction and which need
  separate audits.

Exit gate:

- no unresolved product-level decision that would change the public object model;
- each scientific work package names its parameter, governing derivation, and evidence plan.

### Work package 1: typed study, identification, and result contracts

Deliverables:

- `CausalStudy` and typed designs;
- typed estimands and parameter keys;
- explicit-adjustment identification provider;
- `IdentifiedEffect`;
- shared `CausalResult` protocol;
- immutable configuration groups;
- capability records and errors;
- current point and longitudinal engines adapted behind the contracts without changing their
  arithmetic.

Exit gate:

- new quickstart works through existing analytic TMLE;
- ordinary fit returns no `ResultSet.single()` wrapper;
- **every** currently registered estimand is bit-for-bit unchanged where the normalized
  configuration is the same -- longitudinal and survival included, since this package adapts those
  engines too, and a shifted regimen mean is the failure this gate exists to catch;
- that invariance covers the reported *inference*, not only the point estimate: influence curve,
  standard error, and confidence interval. An engine can be re-plumbed into a new contract without
  moving `psi` and still report a different interval, and an interval is what a reader acts on;
- longitudinal results satisfy or explicitly refuse every shared result operation;
- no graph, Riesz, or EP behavior is faked by placeholders that return estimates.

### Work package 2: assessment contract

Deliverables:

- unified diagnostics facade;
- `validate()` default battery;
- immutable report types and statuses;
- capability-driven sensitivity routing;
- existing point diagnostics migrated without numerical change;
- longitudinal stagewise diagnostic adapters;
- assessment caching and replayability metadata.

Exit gate:

- every result/method/assessment cell is deliberately supported or refused;
- every migrated diagnostic returns the number it returns today, longitudinal stagewise adapters
  included -- the deliverables above claim the arithmetic is unchanged, so the gate has to be able
  to fail when it is not;
- saved results reproduce all cache-only assessments;
- no expensive refit runs from summary or default validation.

### Work package 3: general nested Riesz engine

Deliverables:

- functional-stage and nested-functional representations;
- analytic, direct, provided, and composed representer strategies;
- observed and intervention representer evaluation;
- stagewise cross-fitting;
- nested targeting;
- influence-curve and inference integration;
- stage artifacts, provenance, persistence, and diagnostics;
- analytic representation of existing TMLE targets;
- initial evidence-gated catalog.

Exit gate:

- all registered Riesz targets satisfy the registry/evidence gates in both directions;
- analytic representers reproduce the current supported estimators;
- direct representers satisfy their moment/balance checks;
- deliberate `alpha_star`, sign, stage-order, and cumulative-product mutations fail;
- a pinned RieszCML comparison is present only as secondary evidence.

### Work package 4: optional DoWhy integration

Deliverables:

- optional dependency group;
- identification provider;
- backdoor translation;
- reverse estimator adapter;
- graph provenance;
- native and DoWhy workflow documentation.

Exit gate:

- graph and explicit adjustment workflows agree after producing the same identified functional;
- unsupported graph results fail before nuisance fitting;
- the core package imports and functions without DoWhy installed.

### Work package 5: EP learner

Deliverables:

- conditional-contrast estimands;
- EP targeting/risk machinery;
- conditional-effect results and prediction;
- CATE and conditional-relative-risk implementations;
- EP diagnostics;
- slow oracle-efficiency and stability studies.

Exit gate:

- paper-derived score/risk checks pass;
- predictions respect their mathematical bounds;
- mutation tests cover the targeting sign, sieve/basis contribution, and contrast construction;
- no unsupported scalar inference is exposed.

### Work package 6: catalog expansion

Deliverables are target-specific. Mediation, additional longitudinal targets, sampling designs, and
other nested functionals each require their own source audit, registry entry, evidence row,
refusals, documentation, and simulation studies.

## 13. Testing and evidence plan

### 13.1 API and contract tests

- discover every public estimation method and ensure it declares supported estimands/designs;
- discover every result family and ensure every result operation is supported or explicitly
  refused;
- detect stale capability exemptions;
- pin the intentionally small top-level export list;
- test named presets and typed strategies normalize to the same configuration;
- test error messages at identification and pre-fit capability boundaries;
- run complete beginner workflows as documentation tests;
- preserve pandas, polars, Arrow-backed pandas, and array ingestion behavior.

### 13.2 Identification tests

- randomized design with known assignment;
- explicit backdoor adjustment;
- graph with a confounder;
- graph containing a mediator that must not be adjusted for total effect;
- graph containing a collider;
- graph with an instrument;
- redundant adjustment variables;
- multiple valid adjustment sets;
- supplied adjustment set validated against the graph;
- disagreement between supplied set and graph;
- unidentified effect;
- missing optional DoWhy installation;
- refusals for unsupported front-door, IV, mediation, and transport results;
- graph/provider/adjustment provenance round trip.

### 13.3 Riesz scientific tests

For every registered functional:

- exact finite-support law;
- Gateaux derivative check of the influence curve;
- exact remainder identity or rate study;
- score-equation check after targeting;
- nonzero targeting witness;
- outcome-correct/representer-wrong and outcome-wrong/representer-correct checks where the published
  double-robustness result applies;
- mutation control for representer sign;
- mutation control replacing `alpha_star` with observed `alpha`;
- mutation control reversing cumulative-product direction;
- mutation control reversing stage order;
- missingness/at-risk/history mask mutation;
- counterfactual-block mutation that cannot vanish at the truth;
- cross-fit leakage detection at every stage;
- binary analytic-representer regression against the existing path;
- pinned secondary RieszCML fixture with predetermined discrepancy handling.

### 13.4 Direct representer tests

- Riesz loss decreases relative to a null representer;
- empirical moment/balance residuals satisfy the specified tolerance;
- observed and intervention predictions use the same training fold model;
- external predictions with missing row identity are refused;
- regularization and clipping are recorded in provenance;
- representer tail/leverage diagnostics respond to a deliberate extreme-weight process;
- learned-representer sensitivity elements use the fitted representer rather than reconstructing a
  propensity model that was never fit.

### 13.5 Nested and longitudinal tests

- two-stage exact law with every representer factor nonconstant;
- missingness stage composed with longitudinal treatment stages;
- dynamic rule restricted to available history;
- full cumulative product on the innermost residual;
- stage-specific intervention product on targeted plug-in predictions;
- bound-active, nonzero-targeting case;
- repeated cross-fitting and clustering;
- persistence of stage ordering and regimen metadata;
- explicit refusals for unsupported survival, competing-risk, and mediation compositions.

### 13.6 Assessment tests

- method/result/diagnostic capability matrix;
- no propensity terminology on a direct-Riesz-only result;
- no direct-Riesz balance claim on an analytic method lacking the required fitted artifacts;
- cached and freshly evaluated assessment results agree;
- combined reports distinguish inapplicable from unavailable;
- refit-based analyses refuse non-reconstructible custom methods;
- sensitivity resolves structured parameter metadata without parsing labels;
- expensive operations remain opt-in;
- report frames preserve the caller's dataframe backend.

### 13.7 EP tests

- known CATE law;
- known conditional-relative-risk law;
- targeted efficient-risk score;
- prediction boundedness;
- out-of-fold risk calculation;
- leakage control;
- basis/sieve mutation;
- targeting-sign mutation;
- second-stage contrast mutation;
- slow convergence and oracle-efficiency study;
- weak-overlap stability comparison against relevant T/DR/R learner baselines.

### 13.8 Integration and release checks

- DoWhy supported-version matrix;
- native and DoWhy translation equivalence -- both of these are local gates under a `dev`
  environment that installs `cleverly[dowhy]` (section 10.3), not checks whose skip is accepted;
- missing optional dependency behavior;
- persistence round trips for every built-in method/result family;
- migration-guide transcripts;
- Ruff across the entire tree, including Markdown examples;
- mypy over `src/cleverly`;
- smallest relevant tests while iterating, then complete local fast and relevant slow tiers;
- no reliance on GitHub Actions as a correctness signal while the repository's stated budget
  condition remains in force.

## 14. Documentation plan

The eventual implementation must update:

- README quickstart and architecture overview;
- user guide organized by causal question and workflow, not estimator internals;
- technical appendix for each identified functional and method;
- evidence registry for every new registered estimand;
- architecture invariants for the study/identification/functional-stage contracts;
- roadmap statuses;
- references with precise paper locators;
- migration guide;
- DoWhy optional-integration guide;
- advanced extension guide for typed estimands, representers, and methods.

Recommended user-guide order:

1. State the causal question.
2. Describe the study design and variables.
3. Identify with an explicit adjustment set.
4. Optionally identify from a DAG.
5. Choose a supported estimand.
6. Estimate with the analytic TMLE preset.
7. Read the estimate and assumptions.
8. Validate targeting and nuisance fits.
9. Assess support and sensitivity.
10. Compare with a direct-Riesz strategy.
11. Save and reproduce the analysis.
12. Move to longitudinal and heterogeneous-effect workflows.

## 15. Risks and mitigations

### Risk: abstraction exceeds current needs

Mitigation: extract every protocol from at least two concrete paths, normally the current analytic
method and the planned Riesz alternative. Keep implementation-private abstractions private until a
real third-party use case requires them.

### Risk: a generic Riesz builder bypasses scientific certification

Mitigation: separate the computational functional object from the registered causal estimand.
Generic custom fits must be labeled custom statistical functionals unless they provide the full
identification and evidence contract. Only registered, evidenced catalog entries receive built-in
causal descriptions.

### Risk: `alpha_star` is mishandled

Mitigation: make observed and intervention evaluation separate required interfaces; never infer
one from the other by reusing a vector; add a deliberate-mutation test for every registered stage.

### Risk: direct Riesz results receive misleading propensity diagnostics

Mitigation: route assessment by fitted-artifact capabilities and terminology, not by broad
estimator family.

### Risk: clean break strands current users

Mitigation: publish before/after examples, a complete argument map, an optional static migration
tool, and a stable final old-API tag. Do not keep duplicate engines.

### Risk: DoWhy coupling leaks into core

Mitigation: isolate provider and estimator adapters under an optional integration module and test
the package without DoWhy installed.

### Risk: unified results hide legitimate method differences

Mitigation: define a small shared protocol and capability records. Do not force every result to
implement meaningless methods or fabricate scalar inference.

### Risk: nested-from-day-one grows the first Riesz milestone excessively

Mitigation: implement the general engine, but gate the public catalog. Ship the engine only when
at least the point, missingness, and evidenced longitudinal paths validate its composition; defer
mediation entries until their separate source audits pass.

## 16. Final decisions and defaults

- The primary workflow is `CausalStudy -> identify -> estimate -> assess`.
- The redesign is a clean alpha API break.
- `cleverly` remains standalone.
- DoWhy is an optional identification and estimator integration.
- Explicit adjustment sets remain a core workflow.
- Public extensibility uses typed protocols.
- Ordinary users may choose named method presets; advanced users pass strategy objects.
- Estimation methods are never selected automatically.
- Direct Riesz learning replaces the mechanism-derived representer, not the outcome regression.
- The Riesz engine supports nested composition from its first milestone.
- The public Riesz catalog remains target-by-target evidence-gated.
- EP is a separate heterogeneous-effect method and result family.
- Validation, diagnostics, refutation, and sensitivity remain post-fit result operations.
- Default validation is inexpensive and does not refit.
- Built-in configurations are reconstructible; custom callables may be non-reconstructible and are
  labeled accordingly.
- Structured parameter metadata is authoritative; display-name parsing is forbidden for routing.
- Existing uncommitted work is preserved and is not part of this redesign.

## 17. Pre-implementation checklist

Implementation may begin only after all boxes below are checked in a reviewed revision of this
document:

- [ ] Product/API design accepted.
- [ ] Clean-break release target selected.
- [ ] Riesz TMLE paper read first-hand and precise results recorded.
- [ ] RieszCML revision pinned and paper/code discrepancies recorded.
- [ ] EP paper read first-hand and first supported contrasts confirmed.
- [ ] Initial Riesz catalog enumerated with evidence status for every entry.
- [ ] DoWhy supported version and public adapter boundary confirmed.
- [ ] Old-to-new migration examples agreed.
- [ ] Work package 1 acceptance tests written or enumerated at test-case level.
- [ ] Architecture-invariant changes drafted.
- [ ] No implementation change has been mixed into the planning commit.
