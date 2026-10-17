# cleverly 0.1.4

This alpha release labels each estimate as supported or experimental. It adds vector treatments
with a continuous component, repeated-outcome curves, hazard-based survival curves, and
sequential baselines. It also adds three C-TMLE paths and refusal reason codes. These are the
changes since 0.1.3.

## Upgrade notes

Review these changes before refitting an analysis under 0.1.4.

Install scikit-learn 1.8 or later. The package now requires `scikit-learn>=1.8` instead of
`>=1.3`. The L1 logistic path of `CTMLE(strategy="lasso")` needs the newer solver.
([#308](https://github.com/esbraun/cleverly-tmle/pull/308))

Read `estimate.support` before you cite an interval, a band, or a derived estimate. Each
estimate now carries the label `supported` or `experimental`, with a reason, evidence ids, and
limits. No argument requests or removes the label.
([#344](https://github.com/esbraun/cleverly-tmle/pull/344))

Replace `BootstrapSummary.inferential` with `BootstrapSummary.diagnostic`, which has the opposite
sense. `BootstrapSummary` also gains `below_floor` and `support`.
([#344](https://github.com/esbraun/cleverly-tmle/pull/344))

Stop treating `CapabilityError` as the signal for these cases. They now return labelled
results, so read `estimate.support` or `result.inference_status` instead.

| case | 0.1.3 | 0.1.4 |
| --- | --- | --- |
| `weights_estimated=True` (`estimated_weight_plugin`) | interval refused | plug-in interval, experimental |
| 3 to 9 clusters (3 to 19 for `LTMLE`) | interval refused | Student t interval on J - 2 degrees of freedom, status `few_cluster_plugin`, experimental |
| fewer than 3 clusters | interval refused | point estimate and `plugin_std_error`, status `no_cluster_variance_plugin` |
| missing-outcome compositions listed under "Experimental compositions" in the scope and refusals reference | `CapabilityError` | experimental label, old refusal text as the reason |

([#344](https://github.com/esbraun/cleverly-tmle/pull/344))

Expect a full-refit bootstrap to keep its inferential columns with the experimental label in
more cases. These cases are clustered fits with 10 or more clusters, cross-fitted point fits,
competing-event fits, and `DRTMLE` fits. A `CTMLE` bootstrap and a bootstrap below the cluster
floor stay diagnostic.
([#344](https://github.com/esbraun/cleverly-tmle/pull/344))

Update exception handling that catches `TypeError` from a refused longitudinal call. These calls
now raise a `CleverlyError` subclass, which is a `ValueError`.

- `SDR` and `IPSI` with `alpha=`, `max_iter=`, or `tol=` raise `MethodConfigurationError`.
- `IPSI.fit()` on point-treatment data raises `MethodConfigurationError`.
- A point-treatment keyword that `LTMLE` refuses by name raises `CapabilityError`.
- `TimeToEvent(time_varying=...)` raises `CapabilityError` with the code `api_redirection`.

An unknown keyword still raises `TypeError`.
([#303](https://github.com/esbraun/cleverly-tmle/pull/303))

Replace `reason=` with `refusal=` where code builds a capability row, such as
`MethodAvailability` or `AssessmentItem`. The argument takes a `Refusal` from
`cleverly.exceptions`. The `reason` property still returns the message text.
([#303](https://github.com/esbraun/cleverly-tmle/pull/303))

Update code that reads the refusal constants in `cleverly.methods`. `SDR_POINT_REFUSAL`,
`SDR_TILT_REFUSAL`, `LTMLE_TILT_REFUSAL`, `IPSI_NO_FLUCTUATION`, `IPSI_POINT_REFUSAL`, and
`REPEATED_OUTCOME_REFUSAL` are now `Refusal` records. Replace `SDR_NO_FLUCTUATION` with
`sdr_no_fluctuation_refusal(name)`.
([#303](https://github.com/esbraun/cleverly-tmle/pull/303))

Replace `cleverly.estimators.ctmle.OAT_PER_ARM_INFERENTIAL` and `per_arm_design_admits()` with
`per_arm_parametric_support(estimator, data)`. The function returns a `SupportVerdict` that
names the first unmet requirement.
([#314](https://github.com/esbraun/cleverly-tmle/pull/314))

Remove from a `truncation_curve()` grid each pair that truncates a declared known mechanism.
Such a pair now raises `CapabilityError` with the code `invalid_declaration`. The default
point-treatment grid omits these pairs.
([#311](https://github.com/esbraun/cleverly-tmle/pull/311))

Check code that reads `LongitudinalDiagnostics.rows`. A node with no fluctuation now reports
`converged=None` and `epsilon=()`. Earlier versions reported `converged=True` for SDR and IPSI
nodes.
([#318](https://github.com/esbraun/cleverly-tmle/pull/318))

Refit analyses when you move saved results to 0.1.4. Loading another version's result emits
`VersionMismatchWarning`. The loader does not migrate the result, and the load can fail.

## Added

### Support label

Each estimate carries `estimate.support`. The label is `supported` only when a registered
support row covers the fit configuration and the row's study cells passed. Otherwise the label
is `experimental`. The label makes no coverage claim of its own.
([#344](https://github.com/esbraun/cleverly-tmle/pull/344))

| surface | where the label appears |
| --- | --- |
| `repr()` | `[experimental]` |
| `summary()` and `to_frame()` | a `support` column |
| `to_dict()` | `support`, `support_reason`, `support_evidence`, and `support_limits` |
| simultaneous band | `result.simultaneous.support` |
| `result.config.support_key` | the key that classified the fit |

A ratio, contrast, or other derived estimate is `supported` only when its row lists the
`derived` kind and each input is `supported`. `result.save()` and `cleverly.load()` keep the
labels. The user guide section "Read the support label" describes each surface.
([#344](https://github.com/esbraun/cleverly-tmle/pull/344))

Registered support rows now cover the `IPSI` studies, the `vector-mtp` study, and derived
estimates on in-sample binary `TMLE`. The `IPSI` competing-risk row is supported with a stated
limit: its standard error is conservative.
([#345](https://github.com/esbraun/cleverly-tmle/pull/345),
[#346](https://github.com/esbraun/cleverly-tmle/pull/346),
[#347](https://github.com/esbraun/cleverly-tmle/pull/347))

### Refusal reason codes

Each `CleverlyError` carries `error.refusal`, a `Refusal` record with a `RefusalReason` code.
The codes are `invalid_declaration`, `different_estimand`, `implementation_gap`,
`inference_gap`, and `api_redirection`. `cleverly.exceptions` exports `Refusal`,
`RefusalReason`, and `NO_ALTERNATIVE`.
([#303](https://github.com/esbraun/cleverly-tmle/pull/303))

Capability rows carry the same record before a fit. These rows are `MethodAvailability`,
`result.diagnostics.capabilities`, and `AssessmentItem`. The row and the raised error state
one reason. A built-in `ValueError` or `TypeError` still means a malformed argument.
([#303](https://github.com/esbraun/cleverly-tmle/pull/303))

### Vector treatments with a continuous component

`cleverly.interventions.JointPolicy` declares one policy for each component of a vector
treatment. Pass `treatment=[...]` with `continuous_treatment=[...]` to `CausalData` or
`PointTreatment`. The longitudinal data classes take the same list form.
([#335](https://github.com/esbraun/cleverly-tmle/pull/335))

- Point `TMLE(policies=[JointPolicy(...)])` reports `ey_policy` and `ate_policy` with
  `ratio="density"` or `ratio="classifier"`.
- `LTMLE` and `SDR` fit a node with categorical and continuous columns through a `JointPolicy`
  in each regimen. Censoring, survival, competing risks, weights, clusters, cross-fitting, and
  `msm=` are available.
- A vector of categorical columns only is one categorical treatment over the observed tuples,
  with at most 20 levels.
- A vector with a dose has no arms. `DRTMLE`, `CTMLE`, a known treatment mechanism, and
  `survival_model="hazard"` refuse it with a stated reason.

Evidence: the `vector-mtp` study passes 20 of 20 truth tests and 36 of 36 property cells at
1,000 replications of n = 2,000. The R `lmtp` comparator runs are pending.
([#335](https://github.com/esbraun/cleverly-tmle/pull/335))

### Repeated-outcome curves

`Repeated` declares an outcome that is measured at each node and can return to zero. Use
`outcome=Repeated(["Y1", "Y2", "Y3"])` in `LongitudinalTreatment`. `LTMLE` then reports the
mean curve `ey_regimen[d @ t=h]` and its contrasts at each horizon h.
([#305](https://github.com/esbraun/cleverly-tmle/pull/305),
[#306](https://github.com/esbraun/cleverly-tmle/pull/306))

- Each horizon is the end-of-study fit of the panel cut after h.
- Only `curve(scale="mean")` is available. Risk, survival, `rmst()`, and `rmtl()` are refused.
- Encode a lost unit as `C_t = 0`. A missing `Y_t` on an uncensored row is refused.
- `SDR`, `IPSI`, and `LongitudinalData.from_long_panel()` refuse a `Repeated` outcome.
- `cleverly.datasets.make_longitudinal_repeated` returns a frame and its exact truth.

Evidence: 13 of 13 truth tests and 97 of 97 property cells pass at 2,400 replications of
n = 3,000 on one three-node law. The R `ltmle` comparison is pending. The full-refit bootstrap
is diagnostic for this outcome.
([#305](https://github.com/esbraun/cleverly-tmle/pull/305))

### Hazard-based survival curves and isotonic projection

`LTMLE` accepts `survival_model="hazard"`, `targeting="one_step"`, `step_size=`, and
`isotonic=`. The hazard model targets cause-specific discrete hazards for a survival or
competing-risk curve. It needs one categorical baseline treatment and no covariate after
baseline.
([#310](https://github.com/esbraun/cleverly-tmle/pull/310))

- `targeting="one_step"` targets one set of hazards for all horizons (Cai and van der Laan
  2020, Algorithm 2).
- `isotonic=True` projects each curve and its band limits onto a non-decreasing curve.
  `result.isotonic_moves` lists the moved levels.
- A pointwise interval at a moved level is valid only where the true curve strictly increases.
- Other designs refuse the hazard model before any learner and name
  `survival_model="iterated_mean"` with `isotonic=True` as the route.

Evidence: three Python property studies pass every truth test and cell. No registered study
measures the coverage of a hazard fit or a projected fit. The R `survtmle` and `tmle3`
comparators are pending.
([#310](https://github.com/esbraun/cleverly-tmle/pull/310))

### Sequential g-computation and IPW baselines

`cleverly.longitudinal.SequentialGComputation` and `SequentialIPW` estimate the `LTMLE`
estimands by iterated conditional expectation and by inverse-probability weighting. Use
`gcomp()`, `ipw()`, or `CausalStudy(...).estimate(method="gcomp")` and `method="ipw"`.
`SequentialIPW` takes `normalization="hajek"` (default) or `"horvitz_thompson"`.
([#309](https://github.com/esbraun/cleverly-tmle/pull/309))

`cleverly.learners.GLM` is the default learner of both baselines. A baseline reports an
influence-curve interval only when each regression it reads is a `GLM`. That interval is valid
only when those models are correctly specified. Other learners give a point estimate with the
status `adaptive_baseline_plugin`, and the interval is withheld.
([#309](https://github.com/esbraun/cleverly-tmle/pull/309))

Evidence: coverage is 0.941 to 0.961 at 1,000 replications of n = 8,000 on two correctly
specified laws. The declared misspecification controls fail as designed. Agreement with R
`ltmle` and `stremr` is checked at fixture level only.
([#309](https://github.com/esbraun/cleverly-tmle/pull/309))

### C-TMLE paths

`CTMLE` accepts `strategy="lasso"` and `strategy="learners"`, with `lambdas=`,
`treatment_candidates=`, and `selection_split=`. These follow R `ctmleGlmnet` and
`ctmleGeneral` with C-TMLE1. Both report a point estimate and plug-in diagnostics only, under
status `working_mechanism_plugin`.
([#308](https://github.com/esbraun/cleverly-tmle/pull/308))

- `cleverly.estimators.PropensityArrays` supplies precomputed propensity arrays.
- `CTMLESelection` gains `labels`, `final`, and `treatment_risk_increases`.
- Evidence: the `c-tmle-candidate-sequences` study declares a red lasso coverage cell (0.9237)
  and a red necessity family. The R `ctmle` runs are pending.

`CTMLE(strategy="derivative")` fits C-TMLE0, one TMLE whose fluctuation adds derivative
covariates along a path of treatment mechanisms. It selects nothing and reports the ordinary
TMLE interval. It also admits clustered designs through `id=`.
([#312](https://github.com/esbraun/cleverly-tmle/pull/312))

- The interval needs consistent outcome and treatment mechanism estimators.
- The source papers report undercoverage when the outcome regression is misspecified.
- Evidence: the `ctmle-derivative` study passes 3 of 3 truth tests and 21 of 21 property
  cells. The R `ctmle` type 3 comparator is pending.

`CTMLE(strategy="oat", oat_design="per_arm", cross_fit=False)` reports an influence-curve
interval with `ParametricOutcome` and `ParametricMechanism` from `cleverly.learners`. The
interval comes from one stacked Z-estimator. It covers the arm mean only if the outcome model
is correct or the clipped mechanism is the true mechanism.
([#314](https://github.com/esbraun/cleverly-tmle/pull/314),
[#330](https://github.com/esbraun/cleverly-tmle/pull/330))

- `per_arm_parametric_support()` states the admitted designs. These are binary treatment,
  complete unweighted data, and the `ate`, `rr`, `or`, `ey`, `ey0`, and `ey1` estimands.
- Other outcome-adaptive fits keep `generated_design_plugin` and withhold the interval.
- The default `simultaneous=True` builds no band for these fits, and the summary says so.
- Evidence: a Python property study over seven laws at n = 500 to 8,000 passes. Its two
  negative controls read red as designed.

### Sensitivity and datasets

`omitted_variable_bounds(result, benchmark=record)` reads `cf_y` and `cf_d` from a
`BenchmarkResult` and passes the gains unchanged. This follows Remark 8 of Appendix E.5 of
arXiv:2112.13398. `BenchmarkResult` gains the unclipped `gain_y` and `gain_d`.
([#337](https://github.com/esbraun/cleverly-tmle/pull/337))

`cleverly.datasets.make_sequential_offers` simulates a three-week offer campaign with a known
offer mechanism and an exact truth. `offers_protocol` and `offers_truth` come with it.
([#315](https://github.com/esbraun/cleverly-tmle/pull/315),
[#331](https://github.com/esbraun/cleverly-tmle/pull/331))

`cleverly.datasets.make_missing_outcome_bounded` simulates a proportion outcome on (0, 1) with
missing values.
([#329](https://github.com/esbraun/cleverly-tmle/pull/329))

## Changed

`omitted_variable_bounds` and its grid now accept `cf_y = 1`. The valid ranges are `[0, 1]` for
`cf_y` and `[0, 1)` for `cf_d`. The defaults still resolve to 0.03.
([#337](https://github.com/esbraun/cleverly-tmle/pull/337))

A longitudinal fit with `baseline=()` still raises `DataError`. The refusal now carries the
code `implementation_gap`.
([#318](https://github.com/esbraun/cleverly-tmle/pull/318))

`LongitudinalDiagnostics.to_frame()` omits the `epsilon` and `converged` columns when no node
has a fluctuation.
([#318](https://github.com/esbraun/cleverly-tmle/pull/318))

The `IPSI` studies have now run. The 0.1.3 notes said they had not. `longitudinal-incremental`
passes 36 of 36 property cells, and `longitudinal-incremental-dropout` passes 24 of 25. Its
competing-risk cell is conservative, with coverage 0.959. The R `npcausal` comparator is
pending.
([#342](https://github.com/esbraun/cleverly-tmle/pull/342))

### Documentation

The tutorials now start from one causal question and follow a shared outline. Each tutorial
links its method evidence in a "Method evidence" section.
([#319](https://github.com/esbraun/cleverly-tmle/pull/319),
[#321](https://github.com/esbraun/cleverly-tmle/pull/321),
[#323](https://github.com/esbraun/cleverly-tmle/pull/323),
[#324](https://github.com/esbraun/cleverly-tmle/pull/324),
[#325](https://github.com/esbraun/cleverly-tmle/pull/325),
[#326](https://github.com/esbraun/cleverly-tmle/pull/326),
[#327](https://github.com/esbraun/cleverly-tmle/pull/327),
[#328](https://github.com/esbraun/cleverly-tmle/pull/328),
[#334](https://github.com/esbraun/cleverly-tmle/pull/334),
[#338](https://github.com/esbraun/cleverly-tmle/pull/338),
[#339](https://github.com/esbraun/cleverly-tmle/pull/339),
[#341](https://github.com/esbraun/cleverly-tmle/pull/341))

A new sequential-offers tutorial covers a time-varying confounder with a known offer mechanism.
([#315](https://github.com/esbraun/cleverly-tmle/pull/315))

The DR-TMLE tutorial is removed. The DR-TMLE technical reference remains.
([#348](https://github.com/esbraun/cleverly-tmle/pull/348))

## Fixed

`DRTMLE`: `result.diagnostics.support()` now reports the score-load row of the mean group.
Earlier versions omitted it on fits with a non-empty `guard` and on missing-outcome fits.
Estimates and standard errors are unchanged.
([#322](https://github.com/esbraun/cleverly-tmle/pull/322))

`dataclasses.replace` now works on `LongPanel`, `PointTreatment`, `TimeToEvent`, and
`LongitudinalTreatment` with a declared mechanism.
([#318](https://github.com/esbraun/cleverly-tmle/pull/318))

When one arm of a `CTMLE` per-arm stack fails, a sound arm now keeps its `ey[<label>]`
interval. `per_arm_parametric_support()` refuses a `ParametricOutcome` whose
`arm_columns` differs from the number of arm indicator columns.
([#316](https://github.com/esbraun/cleverly-tmle/pull/316),
[#317](https://github.com/esbraun/cleverly-tmle/pull/317))

## Packaging and compatibility

cleverly supports Python 3.11, 3.12, and 3.13 with pandas or Polars dataframes. It now requires
`scikit-learn>=1.8`.
([#308](https://github.com/esbraun/cleverly-tmle/pull/308))

This remains an alpha release. The package does not promise compatibility for public APIs,
`CausalStudy` objects, pickled objects, or saved results. Follow the upgrade notes when you move
an analysis to 0.1.4.

**Full changelog:** [v0.1.3...v0.1.4](https://github.com/esbraun/cleverly-tmle/compare/v0.1.3...v0.1.4)
