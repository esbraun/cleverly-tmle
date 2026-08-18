# Nested Riesz implementation plan

**Status:** planning contract only. No runtime code is part of this document's pull request.
Implementation begins only after this plan is reviewed and merged.

**Governing work package:** logical PR 5 / work package 3 in
[`public-api-redesign.md`](public-api-redesign.md).

**Source audit date:** 2026-08-17.

## 1. Purpose and completion boundary

The next implementation pull request will add the general nested Riesz engine and the first
evidence-gated public catalog. It is a scientific review unit, not a collection of API stubs. The
PR is complete only when it contains all of the following:

1. typed functional-stage, representer-strategy, fitted-stage, and nested-functional contracts;
2. analytic, direct, provided, and composed representer implementations;
3. distinct observed-data and intervention-state representer evaluation;
4. stagewise cross-fitting with one validated fold plan shared by regression and representer fits;
5. nested targeting in the paper-derived order;
6. influence-curve inference through the existing `ParameterEstimate`, covariance, contrast,
   clustering, and simultaneous-inference infrastructure;
7. method-appropriate diagnostics, validation, persistence, provenance, and replay metadata;
8. analytic representations of the existing estimators that the first catalog claims to support;
9. direct-Riesz fits for the same catalog where the functional's Riesz moment problem is fully
   specified;
10. independent exact-law, Gateaux, remainder, score, mutation, leakage, and statistical evidence;
11. explicit pre-fit refusals for every nearby composition that is not in the first catalog; and
12. a pinned RieszCML comparison used only as secondary implementation evidence.

The PR must not ship a generic builder that can produce a causal interval from incomplete
identification metadata. It must not register a target merely because the engine can represent its
arithmetic. It must not route direct-Riesz results through propensity-only diagnostics or
sensitivity calculations.

## 2. Sources and the interpretation to implement

### 2.1 Governing nested-TMLE derivation

Balkus, Testa & Hejazi (2026), arXiv:2604.21721v1 is the governing source.

The implementation reads these results as follows:

- Equation (1), page 3, gives the single-stage centered influence function
  `h(O; eta) + alpha(X) * (Y - eta(X)) - Psi(eta)`.
- Theorem 1, pages 5–6, supplies the Riesz EIF construction, and Corollary 1 specializes it
  to a conditional mean.
- Theorem 2, pages 6–8, gives the sequential EIF. In the paper's **outermost-first** index,
  residual `t` is multiplied by the prefix product `prod(k=1..t, alpha_k)`.
- Algorithm 1, pages 9–10, fits sequential regressions in reverse time, forms those cumulative
  products, targets from the innermost regression outward, and reports the outer plug-in mean.
- Section 4 states the score solved by the targeting loss. The implementation must recompute that
  score from stored arrays rather than persist a Boolean success flag.
- Section 5.1 identifies the treatment-specific mean as a one-stage instance.
- Section 5.2 identifies static longitudinal treatment-regime means as sequential instances.

The repository will store stages **innermost first**, matching backward-recursion code and the
current longitudinal `SequentialStep` order. Under this convention, the paper's prefix product is
a suffix product in storage order:

```text
paper order:       outer 1, ..., inner T
stored order:      inner 0, ..., outer J-1
residual weight:   omega[j] = product(alpha[k], k=j..J-1)
```

Every public description, serialized manifest, test fixture, and diagnostic will name the storage
order. No API will expose a bare list called `stages` without declaring its direction.

### 2.2 Governing direct-representer loss

Chernozhukov, Newey & Singh (2022), arXiv:1809.05224v5 / Econometrica 90(3), is the
governing source for the first direct learner.

- Equation (2.4), page 12, gives the orthogonal moment
  `m(W, gamma) - theta + alpha(X) * (Y - gamma(X))`.
- Equation (2.5), pages 12–13, gives the product-bias identity that the remainder test will
  evaluate at deliberately incorrect nuisance values.
- Equations (3.1)–(3.2), pages 14–15, give the cross-fitted debiased and targeted estimators.
- Equations (3.3), (3.6), and (3.7), pages 15–17, define the dictionary moment `M`, Gram matrix
  `G`, and penalized minimum-distance objective for the Riesz coefficients.

The first direct strategy will therefore solve an explicit empirical Riesz moment problem. A
generic supervised regression of an estimated inverse propensity score is not a direct-Riesz
implementation and will not be labeled one.

The first solver may use a ridge-regularized closed form for a finite dictionary because it is the
same quadratic Riesz loss with an L2 rather than L1 penalty. Its result must record:

- dictionary construction and fitted dimension;
- centering/scaling convention;
- penalty type and value;
- fold-specific objective and null objective;
- fold-specific moment/balance residuals;
- coefficient norm;
- any clipping or bounding performed after prediction; and
- the row and fold provenance of both `alpha` and `alpha_star`.

L1/minimum-distance Lasso, neural Riesz regression, and user-defined optimization backends may be
added later through the same strategy protocol. They are not required to prove the first engine.

### 2.3 Pinned secondary implementation source

The implementation comparison is `nshlab/RieszCML` commit
[`45e8d277930cd0df4eb8a91a7c686ee4c6fdef09`](https://github.com/nshlab/RieszCML/tree/45e8d277930cd0df4eb8a91a7c686ee4c6fdef09).
The repository describes itself as work in progress, and its package metadata is incomplete, so it
is provenance and a source of deliberate mutations—not an oracle.

The audited implementation supplies several useful distinctions:

- `R/ComposedRieszCurve.R` stores stages innermost first and uses suffix cumulative products.
- `R/riesz_tmle.R` targets stages innermost first.
- `R/RieszCurve.R` and `R/catalog.R` distinguish observed `alpha` from intervention-state
  `alpha_star`.
- `tests/testthat/test-double-robustness.R` contains nonzero witnesses for reversed products and
  for updating counterfactual predictions with observed `alpha`.

The comparison also exposes choices that `cleverly` will not copy:

1. **Missing `alpha_star`.** RieszCML warns and falls back to observed `alpha`. `cleverly` will
   refuse before fitting because the fallback is not generally a valid plug-in update.
2. **Variance state.** At the pinned revision, both single and composed TMLE use the untargeted
   influence curve for the reported variance while the point estimate is targeted. `cleverly`
   will derive inference from the targeted fitted state, verify it by a Gateaux check, and retain
   the untargeted curve only as a diagnostic comparison.
3. **Cross-fitting.** The audited package evaluates nuisance functions on the supplied data but
   does not own the fold plan that proves each row was predicted out of sample. `cleverly` will
   make folds part of the stage fit and persistence contract.
4. **Formula evaluation.** R formulas and length-compatible numeric vectors do not establish
   row identity, training provenance, or intervention support. `cleverly` will use typed callables
   and fitted artifacts, and will refuse unproven aligned arrays.
5. **Generic registration.** A computational Riesz curve is not by itself an evidenced causal
   estimand. `cleverly` will keep the generic engine separate from the public catalog.

The secondary fixture will pin a small, deterministic set of inputs and expected intermediate
arrays. Numerical parity will not replace the independent mathematical evidence.

## 3. Typed contracts

The implementation PR will add a focused module namespace rather than enlarging `study.py` or the
existing analytic estimator classes with Riesz-specific state.

### 3.1 `FunctionalStage`

An immutable stage definition will declare:

- stable `name` and structured `history` metadata;
- regression training design and observed evaluation design;
- plug-in/intervention evaluation design;
- the residual source: observed outcome or preceding inner stage;
- `RepresenterProblem`, including the functional moment map;
- `RepresenterStrategy`;
- targeting map for observed and intervention predictions;
- training, at-risk, observed, and evaluation masks where applicable;
- identification assumptions, scientific references, and evidence identifier;
- required variables and expected shapes; and
- whether custom callables make the configuration non-reconstructible.

Construction validates names, roles, row counts, row identity, masks, design widths, and metadata.
The stage does not yet promise a causal parameter: that promise belongs to a registered estimand
that constructs the stage.

### 3.2 `NestedFunctional`

An immutable nested functional will hold an **innermost-first** nonempty tuple of stages and:

- validate common row identity and fold eligibility;
- require only the innermost stage to read the observed outcome;
- require each outer stage's residual source to be the preceding plug-in output;
- declare the final plug-in map and inference scale;
- carry the registered evidence identifier when it is a built-in causal functional; and
- label an advanced unregistered construction as a custom statistical functional.

A custom statistical functional may be fitted and inspected, but it will not receive built-in
causal prose or a certified causal interval unless it supplies the complete identification and
evidence contract.

### 3.3 Representer strategies

`RepresenterStrategy` will have one behavior-oriented fitting method that receives a stage, the
shared fold plan, training rows, and observed/intervention evaluation rows. It returns one fitted
artifact containing both evaluations and diagnostics.

The first implementations are:

- `AnalyticRepresenter`: constructs `alpha` and `alpha_star` from fitted treatment,
  missingness, censoring, or density components and preserves their truncation state;
- `DirectRiesz`: fits both evaluations from one fold-trained dictionary/solver;
- `ProvidedRepresenter`: accepts an external model or prediction artifact only with row IDs,
  fold IDs, training provenance, functional/evaluation metadata, and both evaluations; and
- `ComposedRepresenter`: stores component evaluations and the paper-derived cumulative products,
  never only the product.

The protocol will not infer `alpha_star` from `alpha`. An analytic strategy that has no defined
intervention evaluation and an external artifact that supplies only one vector both fail before
regression fitting.

### 3.4 Fitted artifacts

Each `RieszStageArtifact` will persist:

- initial and targeted regression predictions at observed rows;
- initial and targeted plug-in predictions;
- residual source used by the score;
- observed `alpha` and intervention `alpha_star`;
- cumulative observed and intervention update products;
- fold assignment and per-fold training fingerprints;
- targeting coefficient, score, iteration count, convergence state, and failure reason;
- regularization, clipping, masks, and representer diagnostics; and
- stage name, order, history, identification, and evidence metadata.

`RieszFit` will hold the ordered tuple, final plug-in estimate, rowwise targeted influence curve,
untargeted comparison curve, and shared fold/provenance records.

### 3.5 Method and result surface

`RieszTMLEMethod` will implement the existing `EstimationMethod` protocol and contain:

- outcome-regression strategy;
- representer strategy (direct by default for the `"riesz_tmle"` preset);
- cross-fitting configuration;
- targeting/link configuration;
- inference configuration; and
- runtime/provenance configuration.

`IdentifiedEffect.available_methods()` will declare `riesz_tmle` only for catalog cells whose
stage builder and evidence gate are implemented. The availability check must run before any
nuisance or representer model is constructed.

`RieszResult` will implement `CausalResult` without pretending its fitted artifacts are
propensities. It will reuse:

- `ParameterEstimate` and influence-curve variance;
- clustered variance and covariance;
- delta-method contrasts and transformations;
- simultaneous inference;
- dataframe-backend-preserving output;
- causal identification and structured parameter keys;
- immutable assessment reports; and
- common provenance and replayability records.

It will expose Riesz-specific stage artifacts and diagnostics in addition to the shared result
surface. A separate result type is preferable to inserting fake `Propensity` objects into
`TMLEResult.nuisance`.

## 4. Fitting and targeting algorithm

For one nested functional, the engine will execute the following order.

### 4.1 Pre-fit validation

1. Validate the identified effect against the catalog and method capabilities.
2. Validate all stage roles, histories, masks, intervention evaluations, and evidence IDs.
3. Realize one outer fold plan, respecting treatment stratification and cluster integrity.
4. Validate user-provided folds for complete coverage, exactly one held-out fold per row, no
   train/test overlap, cluster integrity, and row identity.
5. Reject unsupported result families, custom callables that conflict with requested persistence,
   and provided representers without provenance.

No learner is cloned or fitted before these checks pass.

### 4.2 Initial regression and representer fits

1. Start with the observed outcome as the innermost residual source.
2. Fit the innermost regression out of fold and predict it at observed and intervention states.
3. Fit or construct that stage's `alpha` and `alpha_star` from the same outer training fold.
4. Pass the stage's initial plug-in output outward as the next regression target.
5. Repeat until the outermost stage is fitted.
6. Retain fold-trained models only while needed for evaluation; persist predictions and provenance,
   not arbitrary executable model objects.

Learner-internal tuning occurs only within an outer training fold. The first implementation will
not offer independent regression and representer folds. That option remains gated until its theory
and diagnostics are specified.

### 4.3 Product construction

For stored stages `0..J-1`, innermost first:

```text
omega_observed[j] = product(alpha[k], k=j..J-1)
omega_plugin[j]   = alpha_star[j] * product(alpha[k], k=j+1..J-1)
```

The second expression updates only the current stage at its intervention state while retaining the
outer histories at their observed states. It is not the product of all `alpha_star` values. Every
component and product is stored separately.

The product direction and intervention expression will be checked by exact arrays whose factors
are all nonconstant. Binary indicator identities alone are insufficient because they can make
wrong all-starred and mixed products coincide.

### 4.4 Sequential targeting

1. Set the current target to the observed outcome.
2. At the innermost stage, solve the one-dimensional fluctuation score using
   `omega_observed[0]`.
3. Update observed predictions with the observed product and plug-in predictions with
   `omega_plugin[0]`.
4. Pass the targeted plug-in prediction outward as the next stage's target.
5. Repeat through the outermost stage.
6. Average the outermost targeted plug-in prediction using the declared observation weights.
7. Recompute every stage score and the complete influence-curve mean from stored arrays.

Identity and logistic fluctuations will be supported initially. Logistic targeting requires an
explicit or validated outcome scaler, carries the same bounds at every evaluation, and records
bound activity. A zero/degenerate direction is a `CleverlyError`, not a silent zero update.

### 4.5 Influence curve and inference

The targeted uncentered curve, in stored order, is:

```text
outermost_targeted_plugin
+ sum_j omega_observed[j] * (
      targeted_residual_source[j] - targeted_observed_regression[j]
  )
```

The reported influence curve subtracts the targeted plug-in estimate, then applies the declared
observation-weight/Hájek and outcome-scale transformations. Cluster aggregation, covariance,
smooth contrasts, and simultaneous bands then use the existing inference functions.

The implementation will not use an untargeted curve to report a targeted estimate's standard
error without an explicit derivation. The targeted curve is independently checked by a Gateaux
derivative at the fitted nuisance state and by exact finite-support identities.

### 4.6 Repeated splitting

Repeated cross-fitting will be added only after a single split is correct. Each repeat fits the
same identified functional, returns one rowwise influence curve, and uses the existing averaging
rule: average point estimates and rowwise curves across repeats before variance calculation.
Stage artifacts retain the repeat index. A test with unequal fold sizes will reject equal-fold
averaging.

## 5. Initial public catalog

The following matrix is the implementation boundary. “Initial” means the next implementation PR
must either support the cell with all evidence or retain a named pre-fit refusal.

| current typed estimand/design | analytic Riesz | direct Riesz | initial decision and reason |
| --- | --- | --- | --- |
| point `CounterfactualMean` | yes | yes | initial; canonical single-stage functional |
| point `ATE` | yes | yes | initial as a contrast of jointly fitted counterfactual means |
| point `RiskRatio` / `OddsRatio` | yes | yes | initial through existing joint delta-method inference; binary outcome only |
| point outcome missing at random | yes | yes | initial as a missingness stage composed with each treatment-specific mean |
| longitudinal static `RegimeMean` | yes | yes | initial; Theorem 2 / Algorithm 1 catalog anchor |
| longitudinal dynamic `RegimeMean` | yes | yes | initial after history restriction and arm-evaluation mutations pass |
| longitudinal `RegimeContrast` | yes | yes | initial as a joint contrast of evidenced regimen means |
| point `ModifiedTreatmentPolicy` | yes | yes | initial only for the already evidenced invertible policies; its moment map and intervention evaluation are target-specific |
| smooth contrasts of the rows above | yes | yes | initial through existing delta-method infrastructure |
| `ATT` / `ATC` | gated | gated | ratio/conditional-functional construction needs a separate stage and variance audit |
| `NaturalCourseMean`, `PAR`, `PAF` | gated | gated | add only after the natural-course/missingness composition and parameter keys are audited |
| stochastic `RegimeMean` at one point | gated | gated | density-valued intervention moment map needs its own direct-loss audit |
| `IncrementalMean` / `IncrementalEffect` | refused | refused | the intervention depends on the treatment mechanism; it is not the fixed linear functional used by the first direct solver |
| point or longitudinal `MSMProjection` | gated | gated | projection loss, cell weights, and coefficient EIF need a target-specific adapter |
| `ControlledDirectEffect` and mediation | refused | refused | mediation is explicitly outside the first catalog despite engine expressibility |
| survival and competing-risk regimen results | refused | refused | at-risk/event composition requires a separate Riesz-stage and inference audit |
| arbitrary custom nonlinear functional | refused as causal | refused as causal | may be labeled a custom statistical functional only after shape validation |

“Analytic Riesz” is not a new estimator result. It is the existing evidenced mechanism-derived
representer expressed through the new stage contract. Its acceptance gate is numerical identity
with the current engine for the normalized configuration, including targeted predictions,
influence curve, standard error, and interval—not merely the point estimate.

## 6. Assessment, persistence, and provenance

### 6.1 Capability-aware assessment

The assessment matrix will add a `riesz` scalar-result family. Its cheap default validation runs:

- stage score and complete influence-curve mean checks;
- direct-representer objective improvement over the null;
- held-out moment/balance residuals;
- tail, leverage, and effective-sample-size summaries from fitted representers;
- stagewise outcome-regression held-out loss; and
- fold/provenance integrity checks.

Terminology is strategy-specific:

- analytic results may report treatment/missingness probabilities and truncation;
- direct-only results report representer balance, loss, tails, leverage, regularization, and
  clipping; and
- composed results report every component and cumulative product.

No direct result will reconstruct a propensity model for positivity or omitted-confounding
sensitivity. Sensitivity methods remain unavailable until their required learned-representer
elements have a published derivation and a stored-artifact implementation.

### 6.2 Persistence format

The format version will be incremented. The manifest will encode typed allowlisted objects, while
arrays live in the NPZ payload. It will include:

- ordered stage definitions and evidence IDs;
- every fitted stage array listed in section 3.4;
- representer strategy and reconstructible configuration;
- row, data, fold, and training fingerprints;
- targeting and inference settings;
- structured identification and parameter keys;
- cached assessment reports; and
- replayability metadata.

The decoder allowlist will contain the new public dataclasses and enums. Unknown types remain an
error. Provided representer models and custom callables are written as non-reconstructible
descriptions only when their predictions and provenance are sufficient for cache-only assessment;
refit-based operations remain unavailable after restore.

Round-trip tests compare every stage array, score, diagnostic, parameter estimate, influence
curve, causal metadata record, and cached assessment. A saved file that loses `alpha_star`, stage
order, or component products must fail to load rather than silently substitute a value.

## 7. Independent evidence plan

### 7.1 Contract and registry gates

- Every public method declares support or a reasoned refusal for every typed estimand/design cell.
- Every Riesz catalog entry maps in both directions to one evidence row.
- Every result operation is supported or deliberately refused for the `riesz` family.
- The top-level export list remains intentionally small; advanced contracts live under a Riesz
  namespace unless ordinary users need them.
- Capability refusals are asserted to occur before learner construction using fail-if-fit learners.

### 7.2 Single-stage scientific evidence

For the counterfactual mean and ATE:

1. exact finite-support value and rowwise EIF;
2. complex-step or central-difference Gateaux derivative against the exact law;
3. exact product remainder with outcome and representer errors both nonzero;
4. outcome-correct/representer-wrong and outcome-wrong/representer-correct witnesses;
5. nonzero targeting coefficient and score equation;
6. sign mutation of `alpha`;
7. replacement of `alpha_star` by observed `alpha` on a law where they differ;
8. counterfactual-design mutation at nonzero epsilon;
9. cross-fit training-row spies for outcome and representer models;
10. analytic parity with the existing TMLE path; and
11. a pinned, secondary RieszCML fixture with a documented tolerance and discrepancy policy.

The direct learner additionally requires loss improvement, held-out moment residuals, same-model
observed/intervention predictions, regularization/clipping provenance, and a deliberate
extreme-representer diagnostic response.

### 7.3 Nested and missingness evidence

Use a two-stage finite-support law with both stage representers nonconstant and all residual terms
nonzero. Assert:

- the complete rowwise EIF from a hand calculation;
- the full product on the innermost residual and the outer-only product on the outer residual;
- intervention update products with `alpha_star` only at the current stage;
- innermost-first targeting and outermost plug-in evaluation;
- exact Gateaux derivative and product remainder;
- reversal-of-stage-order, reversal-of-product-direction, all-starred-product, sign, and mask
  mutations all fail; and
- a bound-active, nonzero-targeting case still solves each stored score.

Prepending a missingness stage adds tests in which missingness depends on treatment/history,
unobserved rows must receive a plug-in update, and using the observed missingness indicator in
`alpha_star` fails.

### 7.4 Longitudinal evidence

The initial longitudinal catalog uses the existing exact laws but adds new Riesz-specific
instruments:

- analytic stage artifacts reproduce existing static and dynamic regimen fits;
- direct stage moment residuals are measured at every treatment node;
- the assigned arm of a dynamic rule is a function only of available history;
- a third-arm categorical mutation rejects binary-complement logic;
- cumulative products include treatment and censoring factors in the derived order;
- cluster-respecting folds remain intact at every stage;
- repeated splitting and unequal-fold aggregation are exercised; and
- stage order, regimen metadata, and every product survive persistence.

Survival, competing risk, longitudinal MSM, and mediation requests use fail-if-fit refusals until
their separate derivations are accepted.

### 7.5 Statistical studies

The applicable slow tier is required because the new direct learner changes nuisance estimation,
targeting, and influence-curve inference. Add named studies for:

- point counterfactual means / ATE under both union-model halves;
- direct-representer consistency and interval coverage;
- static and dynamic longitudinal regimen means; and
- weak overlap, checking stability and non-collapsed coverage without redefining the target.

The studies will publish truth derivations, Monte Carlo standard errors, seed policy, failure/drop
counts, bias, empirical standard deviation, mean standard error, and coverage. Thresholds are set
before running the final study and are not tuned to one result.

## 8. Implementation sequence inside one review PR

The implementation remains one pull request so the general engine and its scientific evidence are
reviewed together. It should be built as the following individually reviewable commits.

### Commit 1: contracts and pre-fit capability boundary

- add typed stage, functional, strategy, method, fitted-artifact, and result contracts;
- add catalog declarations as unavailable-by-default records;
- add method/result/assessment matrix tests and fail-if-fit refusals; and
- add architecture invariants for stage order, `alpha_star`, evidence registration, and folds.

**Gate:** no estimate can yet be returned; every unsupported composition fails with a
`CleverlyError` before engine construction.

### Commit 2: analytic single-stage representation

- adapt point counterfactual means to build one functional stage;
- implement analytic observed/intervention evaluation;
- implement targeting, targeted EIF, and shared result inference; and
- prove bit-for-bit parity with the current analytic engine.

**Gate:** single-stage exact law, Gateaux, remainder, score, mutation, and parity tests pass.

### Commit 3: direct single-stage representer

- implement the finite-dictionary Riesz moment problem and fold artifacts;
- share the outer fold plan with the outcome regression;
- add balance/loss/tail diagnostics and provenance; and
- enable direct point means and their evidenced contrasts.

**Gate:** direct loss, held-out moments, same-fold model, row identity, union-model, leakage, and
slow point-study tests pass.

### Commit 4: nested composition and missingness

- implement cumulative observed and intervention products;
- implement innermost-first sequential targeting;
- compose missingness with point-treatment means; and
- add every product/order/`alpha_star`/mask mutation witness.

**Gate:** exact two- and three-stage laws, Gateaux, remainder, score, persistence, and mutation
tests pass.

### Commit 5: longitudinal analytic and direct adapters

- translate static and dynamic regimen nodes into functional stages;
- preserve history, at-risk, censoring, categorical-arm, cluster, and regimen metadata;
- prove analytic parity with existing `LTMLE`; and
- enable direct stage representers only for the evidenced regimen-mean cells.

**Gate:** static/dynamic/categorical exact laws, parity, direct moments, clustering, repeated
splitting, and named slow longitudinal studies pass.

### Commit 6: persistence, assessment, documentation, and secondary fixture

- complete format-version changes and round trips;
- add capability-aware diagnostics and default validation;
- add user, methodology, evidence, reference, migration, and roadmap documentation;
- add the pinned RieszCML fixture and discrepancy note; and
- update the redesign status only after all gates pass.

**Gate:** restored results reproduce all cache-only assessments, docs tests pass, and no direct
result uses propensity-only language.

No commit is merged independently. If a later gate reveals that an earlier contract is wrong, the
implementation PR changes the contract and its plan amendment together rather than preserving a
premature interface for compatibility.

## 9. Validation commands and handoff gate

During development, run the smallest named modules for the commit being changed. Before the
implementation PR is handed off, run sequentially:

```text
ruff check .
ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q
pytest <named applicable Riesz slow studies> -q
```

Fast and slow tests must not run concurrently. The complete slow tier is required only if the
shared estimation or inference changes can affect non-Riesz families; otherwise the named point
and longitudinal Riesz studies are the correct evidence scope. GitHub Actions is not a gate under
the repository's current budget condition.

The handoff report must list:

- every catalog cell enabled and every nearby cell refused;
- primary-source sections implemented;
- pinned implementation revision and recorded discrepancies;
- exact/mutation/statistical instruments added;
- local validation commands and results; and
- slow studies deliberately not run, with path-based reasoning.

## 10. Review questions that must be settled before implementation

This plan proposes answers to the following questions. Review should amend the document before
runtime work starts if any answer is rejected.

1. **Stage order:** store stages innermost first and make the order explicit everywhere.
2. **Missing `alpha_star`:** refuse; never fall back to observed `alpha`.
3. **Variance:** use the targeted fitted-state EIF, independently verified; keep the untargeted
   curve only as a diagnostic.
4. **Direct solver:** start with a finite dictionary and quadratic ridge Riesz loss; do not call
   inverse-propensity regression direct Riesz.
5. **Folds:** require one shared outer fold plan initially.
6. **Results:** add a `RieszResult` satisfying `CausalResult`; do not populate analytic-result
   fields with fabricated mechanisms.
7. **Custom functions:** allow a validated custom statistical functional, but reserve causal
   descriptions and intervals for registered evidence IDs.
8. **Catalog:** ship only the “initial” rows in section 5; keep all other rows as named pre-fit
   refusals.
9. **Implementation shape:** keep the engine and all its evidence in one implementation PR with
   reviewable commits.
10. **Start condition:** do not begin runtime implementation until this planning PR is merged.
