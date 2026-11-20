# cleverly 0.2.7

This release changes the intervals of bounded differences and continuous outcomes, and makes the
switched standard error the default for the continuous-time survival TMLE. It also counts fits and
predicts memory before long runs, and makes many large fits faster or smaller with identical numbers.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.
- Intervals of two kinds of estimate move. A bounded difference of two binary-outcome means from
  one fit (an ATE, a stratum effect, a regimen or policy contrast) now takes a correlated MOVER
  interval built from the two means' logit-scale intervals. An unclustered continuous-outcome
  target takes a skew-widened t interval, which always contains the earlier normal interval.
  `wald_test` inverts the shipped interval, so its p-value agrees with the interval. Bounded means,
  clustered estimates, ratios, binary-outcome coefficients and bands keep their intervals. The
  registered `asymmetric-wald` study measured both constructions.
  ([#712](https://github.com/esbraun/cleverly-tmle/pull/712))
- `ContinuousSurvivalTMLE` now defaults to `variance="switched"`: when the influence-curve
  variance estimate is itself unstable (relative SD above 0.125), the standard error is raised to
  the predictable variation (Rytgaard and van der Laan). Intervals of such fits widen; pass
  `variance="influence_curve"` for the earlier interval. The refusals "no event of a cause by the
  first target or restricted horizon" and "arm followed short of the last target" now raise
  `DataError` instead of `CapabilityError`. From the registered `continuous-survival-variance`
  study, a fit without restricted horizons reads supported from 500 units; one with them keeps
  1,000.
  ([#723](https://github.com/esbraun/cleverly-tmle/pull/723))
- The fit-time record of a `StructuralNestedMeanResult` is renamed from `diagnostics` to
  `structural_diagnostics`. `diagnostics` is now the assessment facade shared with other results.
  ([#733](https://github.com/esbraun/cleverly-tmle/pull/733))
- `LTMLE`, `SDR`, IPSI and `ContinuousSurvivalTMLE` fits predict their peak memory before any
  learner runs, and refuse with `LongitudinalMemoryError` or `ContinuousSurvivalMemoryError` when
  the prediction exceeds the memory available. Pass `max_memory_bytes=` to set the limit yourself.
  Where the platform does not report available memory, nothing is checked.
  ([#687](https://github.com/esbraun/cleverly-tmle/pull/687),
  [#702](https://github.com/esbraun/cleverly-tmle/pull/702),
  [#720](https://github.com/esbraun/cleverly-tmle/pull/720),
  [#728](https://github.com/esbraun/cleverly-tmle/pull/728))

## Added

### Survival

- `LTMLE(survival_model="hazard", targeting="one_step")` accepts covariates measured after
  baseline when the treatment is given at several nodes, for static rules, dynamic rules and known
  stochastic policies. The registered `hazard-time-varying-multinode` study supports the Wald
  intervals in sample and cross-fitted, and the max-t band in sample (support rows G63.multinode).
  Modified treatment policies and tilts on this path still refuse.
  ([#694](https://github.com/esbraun/cleverly-tmle/pull/694))
- `ContinuousSurvivalTMLE` reads supported with one or three causes at n ≥ 1,000, from the
  registered `continuous-survival-causes` study. A fit with four or more causes, or one or three
  causes with a restricted horizon, reads experimental.
  ([#725](https://github.com/esbraun/cleverly-tmle/pull/725))
- `ContinuousSurvivalTMLE(n_bootstrap=..., bootstrap_interval="percentile" | "bca")` adds a
  full-refit nonparametric bootstrap, with a studentized bootstrap band under `simultaneous=True`
  and `plot(band="bootstrap")`. Every bootstrap interval reads experimental, and they are withheld
  when more than 1% of refits fail.
  ([#730](https://github.com/esbraun/cleverly-tmle/pull/730))
- `refute()` on a default `ContinuousSurvivalTMLE` fit prints a placebo verdict and a subset
  verdict inside the scope the registered `refuter-calibration-continuous-survival` study
  calibrated. ([#727](https://github.com/esbraun/cleverly-tmle/pull/727))
- `sensitivity.simulated_confounding()` runs on a `ContinuousSurvivalResult`. Its readings are
  descriptive. ([#700](https://github.com/esbraun/cleverly-tmle/pull/700))
- `ContinuousSurvivalTMLE(n_jobs=...)` runs the targeting walk's blocks on threads, with the same
  result for every `n_jobs`. ([#713](https://github.com/esbraun/cleverly-tmle/pull/713))

### Point treatment and effect modification

- The projection TMLE of a conditional contrast (`ConditionalContrast` with a fixed basis) fits a
  missing treatment in sample, `strata=`, `repeats` above 1 and the full-refit bootstrap. The
  registered `projection-compositions` study supports them (support rows T49 and T50); the
  bootstrap of a cross-fitted fit stays experimental.
  ([#701](https://github.com/esbraun/cleverly-tmle/pull/701))
- `CTMLE(l1_solver="glmstar")` solves the L1-penalized candidate path with the optional `glmstar`
  package, checking each solution's KKT residual and falling back to the built-in solver with a
  warning on failure. The default `"builtin"` path is unchanged.
  ([#726](https://github.com/esbraun/cleverly-tmle/pull/726))

### Assessment

- Effect curves, structural nested mean fits, dose-response fits and `variable_importance` runs
  gain `diagnostics`, `sensitivity`, `validate()`, `assess()` and `replayability`. Each operation
  runs or refuses with its reason. A variable-importance row combines its candidates' rows and
  returns `CandidateReports`. ([#733](https://github.com/esbraun/cleverly-tmle/pull/733))

### Planning and memory

- `IdentifiedEffect.planned_fits()` and `CausalStudy.planned_fits()` return `PlannedFits`: how
  many learner fits an `estimate` call would make, without fitting anything.
  `cleverly.variable_importance_plan()` does the same for `variable_importance`.
  ([#697](https://github.com/esbraun/cleverly-tmle/pull/697),
  [#709](https://github.com/esbraun/cleverly-tmle/pull/709))
- `variable_importance(..., keep_fits=False)` releases each candidate's data and fit before the
  next one, keeping the entries and the summary.
  ([#706](https://github.com/esbraun/cleverly-tmle/pull/706))
- `targeting_fit`, `IdentifiedEffect.targeting_curve` and `CurveDiagnostics.targeting_curve` take
  `max_memory_bytes=` and refuse before any fit when the half-sample bootstrap would not fit.
  ([#682](https://github.com/esbraun/cleverly-tmle/pull/682))
- A result's `assessment_cache` takes an optional `max_bytes` budget and then drops, least
  recently used first, only answers it can recompute bit for bit.
  ([#707](https://github.com/esbraun/cleverly-tmle/pull/707))
- `CTMLESelection` reports `candidate_fits`, `mechanism_fits` and `target_solves` per branch, and
  its summary adds a search-work row. ([#719](https://github.com/esbraun/cleverly-tmle/pull/719))

### Plots

- Plots of large data take display limits: `influence(max_points=)`, `curve(max_rug_points=)`,
  `zipper(max_intervals=)` and `balance(page=)`. Every limit defaults to the earlier figure, and
  the statistics still use every unit. ([#686](https://github.com/esbraun/cleverly-tmle/pull/686))

## Changed

- A cross-fitted, fold-evaluated clustered `TMLE` whose smallest training complement holds fewer
  than 4 clusters with positive weight reports `few_cluster_plugin`. The registered
  `few-cluster-fold-variance` study set the floor.
  ([#680](https://github.com/esbraun/cleverly-tmle/pull/680))
- A cross-fitted `TMLE` of a discrete treatment applies the constant-target rule to a training
  complement with one outcome value instead of refusing.
  ([#680](https://github.com/esbraun/cleverly-tmle/pull/680))

## Fixed

- On a hazard fit with treatment at several nodes, `nuisance_models()` and the `calibration()`
  plot scored the hazard learner at the wrong treatment path. Estimates were not affected.
  ([#683](https://github.com/esbraun/cleverly-tmle/pull/683))
- `CoxHazard` on separated or heavy-tailed data could accept a step with an infinite log partial
  likelihood and end with huge coefficients and cumulative hazards. Step halving now keeps the
  linear predictor inside the range where every term is finite.
  ([#699](https://github.com/esbraun/cleverly-tmle/pull/699))
- The simulated-confounding summary and plot of a discrete-time survival fit described an outcome
  flip; they now describe the log-odds hazard shift that was computed. The numbers were right.
  ([#700](https://github.com/esbraun/cleverly-tmle/pull/700))
- Dropping a modifier from a tensor or interaction basis in `modifier_stability` raised
  `TypeError`. ([#733](https://github.com/esbraun/cleverly-tmle/pull/733))

## Performance

- The continuous-time survival TMLE's targeting walk runs over blocks of units on the
  event-support grid instead of holding arrays of every unit by every distinct time. Risks with
  the shipped learners are bitwise equal to 0.2.6; restricted means and user learners agree to
  rounding. ([#702](https://github.com/esbraun/cleverly-tmle/pull/702))

The changes below give bitwise-identical results.

- Point TMLE builds counterfactual rows per fold, and duplicate-covariate checks no longer copy the
  covariate matrix. ([#685](https://github.com/esbraun/cleverly-tmle/pull/685),
  [#698](https://github.com/esbraun/cleverly-tmle/pull/698),
  [#705](https://github.com/esbraun/cleverly-tmle/pull/705))
- Adaptive TMLE's working-model selection runs about 5 times faster.
  ([#691](https://github.com/esbraun/cleverly-tmle/pull/691))
- The default Mammen multiplier band draws in about half the time.
  ([#692](https://github.com/esbraun/cleverly-tmle/pull/692),
  [#711](https://github.com/esbraun/cleverly-tmle/pull/711))
- Longitudinal and survival fits hold less memory: shared arrays across horizons, per-horizon
  release of hazard state, and smaller `SurvivalCTMLE`, `SurvivalDRTMLE` and `LearnedRegimen`
  fits. ([#703](https://github.com/esbraun/cleverly-tmle/pull/703),
  [#715](https://github.com/esbraun/cleverly-tmle/pull/715),
  [#716](https://github.com/esbraun/cleverly-tmle/pull/716),
  [#717](https://github.com/esbraun/cleverly-tmle/pull/717),
  [#720](https://github.com/esbraun/cleverly-tmle/pull/720),
  [#721](https://github.com/esbraun/cleverly-tmle/pull/721),
  [#722](https://github.com/esbraun/cleverly-tmle/pull/722),
  [#724](https://github.com/esbraun/cleverly-tmle/pull/724))
- `CoxHazard`, the C-TMLE search, the EP sieve size search, `tipping_gamma`, the continuous
  marginal sensitivity bounds and several plots run faster or smaller.
  ([#684](https://github.com/esbraun/cleverly-tmle/pull/684),
  [#686](https://github.com/esbraun/cleverly-tmle/pull/686),
  [#704](https://github.com/esbraun/cleverly-tmle/pull/704),
  [#714](https://github.com/esbraun/cleverly-tmle/pull/714),
  [#718](https://github.com/esbraun/cleverly-tmle/pull/718),
  [#734](https://github.com/esbraun/cleverly-tmle/pull/734))

## Packaging and compatibility

- The new `cleverly[glmstar]` extra installs `glmstar>=0.7.1` for `CTMLE(l1_solver="glmstar")`.
  It is GPL-2.0-only and is not part of `cleverly[all]`.
  ([#726](https://github.com/esbraun/cleverly-tmle/pull/726))

## Documentation

- The technical reference adds an evidence page and validation-grid rows for each new registered
  study. These studies are Python simulation evidence, and none of them has an external comparator.
  ([#680](https://github.com/esbraun/cleverly-tmle/pull/680),
  [#694](https://github.com/esbraun/cleverly-tmle/pull/694),
  [#701](https://github.com/esbraun/cleverly-tmle/pull/701),
  [#712](https://github.com/esbraun/cleverly-tmle/pull/712),
  [#723](https://github.com/esbraun/cleverly-tmle/pull/723),
  [#725](https://github.com/esbraun/cleverly-tmle/pull/725),
  [#727](https://github.com/esbraun/cleverly-tmle/pull/727))
- The user guide adds "Count the model fits before a long run", and the technical reference
  documents each memory guard. ([#687](https://github.com/esbraun/cleverly-tmle/pull/687),
  [#697](https://github.com/esbraun/cleverly-tmle/pull/697),
  [#728](https://github.com/esbraun/cleverly-tmle/pull/728))

**Full changelog:** [v0.2.6...v0.2.7](https://github.com/esbraun/cleverly-tmle/compare/v0.2.6...v0.2.7)
