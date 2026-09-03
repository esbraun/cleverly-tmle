# Post-fit assessment battery: one call, real objects, next steps

Date: 2026-09-03. Baseline: `main` at `8d14e81`. Status: implemented and independently
reviewed in the working tree. All repository gates pass.

This plan lives under `plans/` rather than `docs/` on purpose. `docs/roadmap.md` is the planning
contract for source-backed scientific work, and `tests/prose.py` scans reader-facing Markdown
under `docs/`. This is an engineering plan for the assessment surface. Its one scientific change
is the E-value selection rule in D3.

The review checked the current source and tests at the stated baseline. The line numbers below are
landmarks, not a substitute for following the named symbols after the implementation moves them.

## 1. Problems, each pinned to current behavior

| # | problem | evidence at the baseline |
| --- | --- | --- |
| P1 | `run_all()` keeps a generic status row and discards the returned study object. | `_CapabilityFacade._run_all` in `src/cleverly/assessment.py` passes the return value to `_diagnostic_item`. That interpreter reads `.passed` when present, special-cases support, and otherwise emits a generic sentence. A caller must invoke the operation again to inspect its object. |
| P2 | `corrections` is declared for ordinary TMLE and can report a vacuous pass. | `_capability` stamps all point rows with `("tmle", "collaborative_tmle", "drtmle")`; `AssessmentCapability.methods` is not consulted by `capability()`. `CorrectionCheck.passed` is true for zero rows, and `tests/unit/test_drtmle_fit.py` pins `ordinary.diagnostics.corrections().rows == ()`. |
| P3 | A binary-outcome typed `ATE()` ordinary-TMLE fit cannot produce its exact post-fit risk-ratio E-value. | `CausalStudy` narrows the result to the identified target, so the result reports only `ate`. `sensitivity/evalue.py` then needs a separately reported reference-arm mean for its fixed-baseline approximation. The fit retains the per-repeat nuisances and `TMLE.retarget`, which can target the matching marginal `rr` without refitting them. A direct baseline check on this revision found identical point estimate, variance, and influence curve for cached-nuisance retargeting and a same-seed typed `RiskRatio()` ordinary-TMLE fit. Equivalent behavior has not been established for CTMLE or DR-TMLE, whose fitting paths add method-specific selection or reductions. |
| P4 | The explicit `att`/`atc` E-value path has no supported conditional conversion. | For binomial outcomes, `_baseline_mean` selects the marginal reference-arm mean. ATT and ATC average their contrast in a conditioning arm, so that marginal level is not the required conditional baseline risk. For Gaussian outcomes, the code standardizes with the population outcome SD without a conditional-effect derivation. No repository source establishes either conditional conversion. |
| P5 | `passed` can mean only that a sensitivity calculation completed. | `_diagnostic_item` maps any returned object with no `.passed` field to `PASSED` and the same generic detail. `summary()` also omits `next_steps`. |
| P6 | A combined report cannot supply required operation arguments or a common seed. | `_run_all` skips every `requires_arguments` row and invokes each other operation with no arguments. `refute`, `benchmark`, and `simulated_confounding` already resolve and record a reproducible seed; the gap is that `run_all` cannot override it or pass `covariates`, `grid`, and related choices. |
| P7 | There is no single result-level call for the applicable post-fit battery. | `validate()`, `diagnostics.run_all()`, and `sensitivity.run_all()` return separate report types. `docs/user-guide/results-assessment.md` presents them separately. |
| P8 | `Replayability.retarget_cached_nuisances` is true when the estimator is absent. | `replayability()` returns `Replayability(True, True, ...)` for `estimator is None`, while `truncation_curve` requires that estimator. `missingness_tilt` and `tipping_gamma` are different: `sensitivity/missingness.py` re-mixes stored targeted predictions and the missingness mechanism and does not use `result.estimator` or `retarget`. |

## 2. Constraints established by the review

The implementation must keep these existing contracts.

- `docs/architecture-invariants.md` requires assessment routing from declared fitted artifacts,
  not result-class names. It also requires sensitivity selection through `ParameterKey`, not
  parsed aliases. The current `_family()` type-name check is debt to remove, not a pattern to
  extend.
- `CausalResult` in `src/cleverly/study.py` is the public protocol for both scalar result classes.
  Adding `assess()` to both classes also requires adding it to this protocol.
- `_pack_cached` packs only a top-level pandas or Polars frame. Putting a raw frame inside an
  `AssessmentItem` would bypass that conversion unless the item packs its own payload.
- `_normalize` already supports mappings, sequences, NumPy scalars, and content-addressed NumPy
  arrays. New cache keys should use it rather than define a second normalizer.
- `TMLE._fit_single` rejects nonlinear `rr` and `or` targets under `cv_evaluation=True`.
  A post-fit retarget must not bypass that refusal.
- A controlled direct-effect fit has an `intermediate_value`; deriving an unregistered ratio
  there would invent a controlled direct risk ratio. This plan does not add that estimand.
- A single `BenchmarkResult` describes one aggregate set of dropped covariates. It does not hold
  one result per covariate and does not carry a robustness value.
- A truncation-curve frame has no scientific verdict or warning threshold. Simulated confounding
  deliberately has no inferential pass/fail verdict. Interpreters must not invent either one.
- VanderWeele and Ding define the E-value on a risk-ratio scale (Annals of Internal Medicine,
  2017, DOI `10.7326/M16-2607`). Chinn derives `log(OR) / 1.81` as a standardized mean difference
  (Statistics in Medicine, 2000, PMID `11113947`). Chinn does not derive an ATT baseline-risk
  conversion. Keep the repository's continuous-outcome approximation, but describe the complete
  approximation rather than attributing a direct risk-ratio formula to Chinn alone.

## 3. Design decisions

### D1. Preserve the returned object and the invocation record

Extend `AssessmentItem` with a private packed payload and non-comparing arguments:

```python
_report: Any = field(default=None, compare=False, repr=False)
arguments: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)


@property
def report(self) -> Any:
    return _unpack_cached(self._report)
```

Construct the item with `_pack_cached(returned_object, backend)`. This handles a returned frame
without changing the existing top-level cache format. `compare=False` is required for both fields:
an invocation can contain a NumPy array such as a negative-control outcome, whose mapping equality
does not produce one Boolean value.

`arguments` records the arguments actually used, including defaults. Bind defaults against the
routed operation's real signature, not the facade's `*args, **kwargs` signature. Replace a
requested `random_state=None` with the resolved seed exposed by `RefutationResult`,
`BenchmarkResult`, or `SimulatedConfoundingResult`. This makes the stochastic invocation
reproducible from the row. Inputs remain subject to the existing trusted-joblib persistence
contract; do not promise that arbitrary user objects are serializable.

`DiagnosticReport` gains:

```python
def report(self, name: str) -> Any
def reports(self) -> dict[str, Any]
def next_steps(self) -> tuple[str, ...]
```

`report(name)` raises `KeyError` if the row did not run or has no object. `reports()` returns only
rows that ran. `next_steps()` de-duplicates text in row order. `summary()` adds a fourth
`next step` column. `to_frame()` already has `next_steps`; keep its existing columns.

### D2. Make capabilities method-aware and remove type-name routing

Declare `corrections` with `methods=("drtmle",)`. Make the common `capability(operation)` narrow a
row against the fitted method. A mismatch returns `available=False`,
`status=NOT_APPLICABLE`, and reason `"the fitted method does not use the correction system"`.
A DR-TMLE fit whose guard subtracts no correction term is also `NOT_APPLICABLE`, with that fact in
the reason.

Replace `_family()` with an explicit result-owned `assessment_family` declaration implemented by
`TMLEResult` and `LongitudinalResult` and included in `CausalResult`. Do not use a registry keyed by
the Python result class: that would retain the architecture-invariant violation. Future engines
declare their own family on the fitted artifact.

Consequences:

- Plain TMLE and collaborative TMLE report `corrections: not_applicable`.
- A direct `diagnostics.corrections()` call on either method raises `CapabilityError` with the
  declared reason.
- Update the test that currently expects the empty `CorrectionCheck` on ordinary TMLE.
- Relax the contract test that assumes every point row has the same `methods`. Instead, verify that
  each method entry names a constructible method and that every public family has a declaration for
  every public operation.

This mechanism also supports future method-specific diagnostics without a branch in `run_all`.

### D3. Derive only the risk ratio needed by E-value

Add a private helper in `src/cleverly/sensitivity/_derived.py`:

```python
def _derived_risk_ratio(result, source_estimand: str) -> ParameterEstimate:
    """Retarget the source contrast's arms to a marginal risk ratio."""
```

Keep it private. A general public `derived_estimates(result, names)` surface is outside the need
demonstrated here and is ambiguous for multi-arm and non-arm targets.

The helper follows these rules.

1. Read the source alias from `result.parameter_keys`. Require target `ate` or `or`, axis `arm`, and an
   unconditioned marginal contrast. Use the key's structured value and reference to construct the
   matching risk-ratio target. Do not call `parameter_stem()` or parse bracketed aliases.
2. Require a binomial point-treatment result, ordinary TMLE, a retained estimator, and cached
   repeat nuisances. Refuse longitudinal, CTMLE, DR-TMLE, continuous-treatment, regime, shift,
   IPSI, MSM, ATT, ATC, and level parameters by naming the unsupported method, axis, or target.
   The method restriction is an evidence boundary: only ordinary TMLE has the direct equality
   witness at this baseline. Extend it only after an engine-specific same-seed equality witness.
3. Refuse `cv_evaluation=True` and `intermediate_value is not None`. These are existing estimator
   boundaries, not convenience exclusions. A marginal mean-group ratio can use the existing
   stratified fluctuation; cover that case with an equality witness rather than refusing all
   stratified fits.
4. Retarget each repeat through `result.estimator.retarget(result.data, repeat.nuisance, ...)`.
   Pass `config.g_bounds`, `config.g_bounds_conditional`,
   `nuisance_bound=config.missingness_bound`, and `config.alpha_sig` explicitly. The estimator
   already owns its `TargetingSpec`. Do not assume retarget defaults equal the original fit.
5. Combine repeat estimates with the same `median_estimates` path as the original fit. Cache the
   final `ParameterEstimate` through `_cached` under a key that includes the structured source key.
   Never merge it into `result.estimates`.

The E-value selection matrix is:

| request and fitted artifacts | result |
| --- | --- |
| reported `rr` | exact reported ratio; `scale="risk ratio"`, `approximate=False` |
| default request on eligible ordinary-TMLE binomial marginal `ate`, or explicit request for that reported `ate` | derive its matching `rr`; use the derived RR alias as `EValue.estimand`, set `scale="risk ratio"`, `approximate=False`, and name the source ATE in `note` |
| default request on eligible ordinary-TMLE binomial marginal `or` | derive the matching `rr` from cached nuisances, with the same exact metadata as above |
| explicit reported `or` | retain the square-root conversion; `scale="odds ratio"`, `approximate=True` |
| binomial marginal `ate`, no retained estimator, but a matching reported reference-arm mean | retain the fixed-baseline approximation; `scale="risk difference"`, `approximate=True` |
| binomial marginal `ate` without either derivation artifacts or its matching reported level | `UNAVAILABLE`; name the missing estimator or level |
| `att` or `atc`, any outcome family | refuse; the necessary conditional baseline or standardization and a supported conditional ratio target are absent |
| Gaussian marginal `ate` | retain the current standardized-outcome approximation; `scale="mean difference"`, `approximate=True`; report the outcome SD in the note and describe the Chinn OR-to-SMD step plus the subsequent E-value approximation accurately |
| level-only or non-arm result | `NOT_APPLICABLE`; an E-value here has no supported two-arm contrast |
| a request needing derivation on CTMLE, DR-TMLE, CV-evaluated, or controlled-direct fits; any legacy result lacking structured keys | `UNAVAILABLE`; return the specific refusal reason |

The explicit-OR row preserves the caller's chosen approximate analysis. The default is free to
prefer the exact derived ratio because the caller did not choose a reported scale. Fix the current
`EValue.scale="risk ratio"` assignment so it describes the source scale before conversion, as its
docstring promises.

`SensitivityFacade.capability("evalue")` must use the same decision function as `evalue()`. Do not
duplicate a looser approximation of the matrix in the capability code.

Default selection must refuse multiple eligible contrasts and list the candidate aliases. An
explicit alias selects only that contrast. Combined runs determine availability and execution
from the supplied alias before gating costs: derived RR uses `retarget`; reported RR, explicit
OR, Gaussian, and fixed-baseline branches use `summarize`. Bare `assess()` must never retarget.

The explicit-OR square-root transformation approximates a risk ratio for common outcomes, not
rare outcomes. For rare outcomes the OR itself approximates the RR. The square-root approximation
can lie above or below the RR; do not claim a one-sided error. This wording follows Table 2 of
VanderWeele and Ding (2017) and VanderWeele (2017), DOI `10.1097/EDE.0000000000000733`.
The Gaussian branch combines the Chinn relation with this common-outcome approximation.

Required scientific witnesses:

- On one binary marginal ATE fixture, the derived risk ratio and a separate same-seed typed
  `RiskRatio()` fit have equal point estimate, variance, and influence curve to `1e-12`.
- Repeat that equality check on a stratified marginal fixture, because the mean-group retarget path
  has explicit strata handling.
- Mutate one fold's cached propensity and show the derived ratio moves. This catches a helper that
  recomputes from levels but ignores the nuisance artifact it claims to retarget.
- Pin refusals for ATT, ATC, CTMLE, DR-TMLE, CV evaluation, controlled direct effects, legacy
  missing keys, non-arm axes, and longitudinal results.
- Pin the direct RR, explicit OR, legacy fixed-baseline ATE, and Gaussian branches so scale and
  approximation metadata cannot drift.

### D4. Interpret objects without inventing scientific rules

Replace `_diagnostic_item` and the duplicate branches in `validate_result` with a two-way-checked
interpreter registry:

```python
INTERPRETERS: dict[str, Callable[[Any, Any], AssessmentItem]]
```

Add `AssessmentStatus.COMPLETED = "completed"`. Reserve `PASSED` and `FAILED` for an operation that
has an explicit verdict. Use `WARNING` for an existing diagnostic rule or an expected refusal
raised while an aggregate report runs. The warning must name the refusal and retain a direct-call
next step. The aggregate report then continues with every accepted diagnostic. Direct standalone
calls still raise `CapabilityError` with their precise refusal. Structural exceptions such as
`KeyError` and `TypeError` still abort because they identify implementation defects, not supported
refusals. A capability known to be unavailable before invocation remains `UNAVAILABLE` or
`NOT_APPLICABLE` and appears in omissions.

| operation | status and evidence-backed detail |
| --- | --- |
| `score_equations` | Use `ScoreCheck.passed` and its row ratios; preserve the existing conditioning warning. Detail gives row count and worst `abs(score) / threshold`. |
| `support` | Reuse `_support_warning`. Detail gives the report's truncated fraction and minimum effective-sample-size ratio when those fields exist. Do not impose one shape across arm, shift, IPSI, and longitudinal support objects. |
| `nuisance_models` | Reuse the exact rules already encoded by `NuisanceDiagnostics.verdict`: propensity AUC above 0.9, missingness AUC above 0.9, calibration slope outside 0.7 to 1.4, mean-learner weight above 0.8, and outcome R2 below 0.05. Low propensity AUC is favorable overlap, not a warning. Centralize these structured findings on `NuisanceDiagnostics` so summary and interpreter cannot diverge. |
| `corrections` | Use `CorrectionCheck.passed`; detail distinguishes identity residuals from reported correction magnitude and includes `contract`. A `bound-active` contract is explanatory, not automatically a failure. |
| `truncation_curve` | `COMPLETED`. Give the evaluated bound range and estimate range. The frame defines no warning threshold. |
| `refute` | `PASSED` only when every `RefutationResult` test passed; otherwise `FAILED`. Name failed tests and direct the caller to their retained draws. |
| `omitted_confounding` | `WARNING` only when the actual returned bias-adjusted interval spans its stored null; otherwise `COMPLETED`. Detail reports stored `cf_y`, `cf_d`, `rho`, and interval. Do not describe these as defaults when callers supplied them. |
| `robustness_value` | `COMPLETED`. Give the point and confidence-limit robustness values. |
| `elements` | `COMPLETED`. Give the returned components. |
| `contour` | `COMPLETED`. Give grid dimensions and ranges and direct the caller to the retained frame. |
| `benchmark` | `COMPLETED`. Give the one aggregate covariate set and its `cf_y`, `cf_d`, `rho`, and `delta_psi`. Do not compare it to an RV that `BenchmarkResult` does not contain. |
| `simulated_confounding` | `COMPLETED` when all cells ran; `WARNING` when `failures` is nonempty. Give maximum successful displacement, failed-cell count, and corner association when present. Never attach a scientific pass/fail rule. |
| `evalue` | `COMPLETED`. Give `point`, `limit`, source scale, and exact/approximate branch. If `limit == 1`, say the interval already includes the null. |
| `missingness` | `COMPLETED`; remain `NOT_APPLICABLE` on complete outcomes. Give the gamma and estimate ranges from the returned frame. |
| `tipping_gamma` | `COMPLETED`; remain `NOT_APPLICABLE` on complete outcomes. Report the returned value or that no tip occurred in the searched interval. |

An operation can pass the capability precheck and still refuse caller-supplied arguments. In that
case, `run_all()` catches `CapabilityError`, records an informative `WARNING`, and runs later rows.
Tests must prove that one refused diagnostic does not prevent a later accepted diagnostic from
returning and retaining its report object.

Refused rows retain bound defaults and supplied arguments too. If no stochastic seed was
resolved, preserve `random_state=None` rather than generating or inventing a seed for the row.
Warnings supplement measured row counts, ratios, and support facts; they do not replace them.

Interpreters may format fields and compute descriptive minima, maxima, counts, and ranges. They may
not create a new inferential cutoff. This distinction is why no validation study changes for D4.

Reimplement `validate()` by applying the same interpreters to cached score, support, and nuisance
objects. This removes the second set of status rules from `validate_result`.

### D5. Accept analyst choices and add the result-level battery

Extend both facade `run_all` signatures:

```python
def run_all(
    self,
    *,
    include_refits: bool = False,
    include_retargets: bool = False,
    arguments: Mapping[str, Mapping[str, Any]] | None = None,
    random_state: int | None = None,
) -> DiagnosticReport:
```

- Validate all operation keys and all argument mappings before running anything. An undeclared
  operation raises `KeyError`.
- A row runs when `arguments[operation]` supplies all required arguments. Otherwise it stays
  `UNAVAILABLE` with the existing missing-argument reason.
- Add `accepts_random_state=True` to `refute`, `benchmark`, and `simulated_confounding`. Forward the
  top-level seed only to those routes.
- Reject a call that specifies both top-level `random_state` and
  `arguments[operation]["random_state"]`; silent precedence would make the recorded invocation
  surprising.
- `random_state=None` lets each operation keep its current resolution rule. A seeded fit reuses
  its fit seed. An unseeded fit draws and records a seed. Do not describe both cases as drawing
  from the fit seed.
- Include the effective arguments in the existing `_cache_key` path.

Add `TMLEResult.assess(...)` and `LongitudinalResult.assess(...)`, and add the method to
`CausalResult`. Each returns a new public `AssessmentReport`:

```python
@dataclass(frozen=True)
class AssessmentReport:
    validation: ValidationReport
    diagnostics: DiagnosticReport
    sensitivity: DiagnosticReport

    def summary(self) -> str: ...
    def to_frame(self, data=None) -> Any: ...
    def report(self, name: str, *, surface: str | None = None) -> Any: ...
    def next_steps(self) -> tuple[str, ...]: ...

    @property
    def attention(self) -> tuple[AssessmentItem, ...]: ...

    @property
    def omissions(self) -> tuple[AssessmentItem, ...]: ...
```

The combined presentation has three sections. Validation owns `score_equations`, `support`, and
`nuisance_models`; omit their duplicate diagnostic rows from the displayed diagnostics section.
`to_frame()` emits one presented row per `(surface, operation)` with a `surface` column.
`report(name)` uses that presented index. If future distinct surfaces expose the same name, require
`surface=` instead of picking silently.

`attention` contains `FAILED` and `WARNING` rows. `omissions` contains `NOT_APPLICABLE` and
`UNAVAILABLE` rows so a skipped expensive analysis is visible without being called a scientific
warning. The summary shows both sets.

With no arguments, `assess()` reads stored artifacts and may populate the assessment cache. It
does not refit or retarget because both cost flags default to false. The full call is:

```python
battery = result.assess(
    include_refits=True,
    include_retargets=True,
    random_state=SEED,
    arguments={
        "benchmark": {"covariates": ("W1", "W2", "W3")},
        "simulated_confounding": {
            "grid": grid,
            "benchmark_covariates": ("W1", "W2"),
        },
        "refute": {
            "tests": ("placebo", "random_common_cause", "subset", "negative_control_outcome"),
            "negative_control_outcome": frame["negative_control_Y"].to_numpy(),
        },
    },
)
print(battery.summary())
evalue = battery.report("evalue")
```

`assess()` does not add refutation processes. `dummy_outcome`, `simulated_outcome`, and
measurement-error refutations run only when the caller supplies their required declarations.

### D6. Fix replayability without disabling stored-artifact sensitivity

When `result.estimator is None`, set `replayability.retarget_cached_nuisances=False`.
`truncation_curve` and the D3 derived ratio should use that declaration because both call the
estimator's retarget path.

Do not apply that gate to `missingness_tilt` or `tipping_gamma`. Their implementation uses stored
targeted predictions and missingness probabilities directly. Keep their present capability on a
restored or legacy result whenever those specific artifacts are available. Their cost class may
remain `retarget` for the caller-facing opt-in, but that label does not imply an estimator
requirement.

### D7. Keep roadmap extensions declarative

| roadmap item | extension after its own source audit |
| --- | --- |
| S4 longitudinal sensitivity | Add the new Tan-model operation, its route, capability rows, and interpreter. Do not flip current point-treatment sensitivity rows to available for longitudinal results. |
| S5 simulated confounding | Change dynamic eligibility and required arguments for the existing operation. |
| R1 Riesz engine | The result artifact declares its `assessment_family`; method-specific rows use `methods=("riesz_tmle",)`. Do not add a class-name registry. |
| P1 EP learner | Its distinct fitted artifact declares its family and adds its rows. |
| I1 DoWhy | The attached native `CausalResult` obtains `assess()` through the protocol implementation. |
| X1 persistence | Round-trip `AssessmentReport` and its packed item payloads through the existing assessment cache. |

Contract tests cover both directions:

- every capability operation has an interpreter and every interpreter names a declared operation;
- every public result family declares every public operation;
- every method restriction names a constructible method;
- every `accepts_random_state=True` route has `random_state` in its real signature;
- every row declared argument-free binds and runs with no operation arguments;
- result-family routing reads the artifact's `assessment_family`, and parameter routing reads
  `ParameterKey`.

## 4. Implementation sequence in one pull request

The requested deliverable is one implementation pull request. Keep the work reviewable in these
ordered slices, with tests beside each slice. Do not open six separate pull requests.

| slice | change | focused tests | result-neutral? |
| --- | --- | --- | --- |
| 1 | D1 payload and invocation record; D6 replayability correction | raw-frame report access; save/load round-trip; equality with NumPy arguments; missing estimator versus missingness-artifact behavior | yes |
| 2 | D2 method and artifact-aware capabilities | ordinary/CTMLE correction refusals; empty-guard DR-TMLE; family and method contract tests | yes |
| 3 | D3 private RR derivation and exact E-value matrix | equality and mutation witnesses; all exact, approximate, and refusal branches | E-value output changes only for eligible ATE/OR defaults and unsupported ATT/ATC |
| 4 | D4 interpreter registry and `COMPLETED` | one fixture for every explicit verdict or warning; descriptive-only cases remain completed; two-way registry test | yes |
| 5 | D5 arguments, seeds, `AssessmentReport`, and `assess()` | required-argument rows run; seed conflict; seeded/unseeded cache separation; duplicate presentation; protocol conformance; point and longitudinal calls | yes |
| 6 | public exports, docs, and full gates | API inventory, doctests, prose review, Sphinx build | yes |

No validation study regeneration is required. The change adds post-fit callers and presentation;
it does not alter estimator inputs, targeting formulas, or existing validation verdicts. The D3
helper calls the existing retarget path. Confirm the committed evidence manifests do not name an
affected result-determining module before the PR handoff, as required by
`docs/development/method-benchmarking.md`.

## 5. Documentation and public API work

- Export `AssessmentReport` from `src/cleverly/__init__.py`.
- Add `AssessmentReport` to both `docs/api/results-assessment.md` and
  `docs/api/object-index.rst`. The latter is the public object manifest consumed by
  `tests/unit/test_documentation_api.py`.
- Add `cleverly.estimators.TMLEResult.assess` to `EXAMPLE_TARGETS`. Give both concrete `assess`
  methods numpydoc-complete docstrings; the task-spine example can live on the point result.
- Do not document `_derived_risk_ratio` as public API.
- In `docs/user-guide/results-assessment.md`, lead with `assess()`, retain the individual facade
  calls, document `arguments`, and distinguish `completed`, `passed`, `attention`, and
  `omissions`.
- In `docs/technical-reference/validation-methods.md`, document the exact E-value matrix and the
  combined-run seed behavior.
- In `docs/technical-reference/scope-and-refusals.md`, add correction, ATT/ATC E-value,
  CV-evaluated ratio, and controlled-direct-ratio refusals.
- Run `python -m tests.prose` after editing reader-facing docs. Review every new finding and record
  any accepted exception in `tests/prose-report.md`.

Run the repository gates exactly as documented:

```text
ruff check .
ruff format --check .
mypy
pytest -q -n auto --dist loadgroup
python -m tests.prose
nox -s docs
```

Formatting may run before these checks, but the handoff evidence should report the check commands,
not combine a mutating formatter with a gate.

## 6. Out of scope

- `CoverageStudy` stays in estimator validation. It needs a data-generating process with known
  truth and is not a property of one observed fit.
- Omitted-variable analyses keep refusing median-combined repeats. The median of influence terms
  is not the influence function of the median bound (`sensitivity/omitted_variable.py`).
- No interpreter invents a threshold. A report without a verdict is `COMPLETED` and descriptive.
- The derived ratio never widens `result.estimates` and is not a new public estimand API.
- This PR does not add a controlled direct risk ratio, conditional ATT/ATC risk ratio,
  longitudinal sensitivity model, or new refutation process.

## 7. Verification record

Independent review found request-specific E-value selection, refusal presentation, and contract
coverage gaps. Regression tests reproduced each behavior before its correction. The focused
assessment, E-value, and end-to-end set passes 166 tests. The full fast suite passes 7,638 tests,
with 83 declared skips. Ruff check and format, mypy, prose review, and the warning-strict Sphinx
build also pass. The affected production modules do not appear in a study manifest, so this
post-fit and presentation change requires no study regeneration.
