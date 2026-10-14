# cleverly 0.1.3

This alpha release adds sequentially doubly robust estimation, longitudinal incremental interventions,
and long-panel input. It extends outcome-adaptive C-TMLE point estimates and narrows several
inference claims. These are the changes since 0.1.2.

## Upgrade notes

Review these changes before refitting an analysis under 0.1.3.

Set `oat_design="shared"` to retain the previous `CTMLE(strategy="oat")` construction.
The default is now `"per_arm"`, which fits a separate treatment mechanism on each arm's outcome
prediction. `CollaborativeTMLEMethod` accepts the same option. Both designs withhold inferential
standard errors, confidence intervals, and p-values.
([#279](https://github.com/esbraun/cleverly-tmle/pull/279))

Treat full-refit bootstrap output as diagnostic for `DRTMLE`, every `CTMLE` strategy, and
cross-fitted point-treatment `TMLE`. Their summaries now report `bootstrap sd` and a
`percentile range`. Their `to_frame()` output uses `bootstrap_sd`, `bootstrap_range_lower`,
and `bootstrap_range_upper` instead of the inferential bootstrap columns.
([#298](https://github.com/esbraun/cleverly-tmle/pull/298))

Reassess DR-TMLE intervals that relied on consistency of only one nuisance estimator.
The corrected contract does not establish interval validity from that condition alone.
The fit retains its `influence_curve` status because it cannot determine which nuisance
conditions hold. Point-estimate double robustness remains unchanged.
([#278](https://github.com/esbraun/cleverly-tmle/pull/278),
[#291](https://github.com/esbraun/cleverly-tmle/pull/291))

Record survival events before censoring. For an event at a node, use censoring 1 and event 1.
For censoring after an event-free interval, use censoring 0 and a missing event.

`LongitudinalData.from_frame()` now refuses event values at or after censoring on an event-free trajectory.
It raises `DataError` where earlier versions discarded those values.
([#297](https://github.com/esbraun/cleverly-tmle/pull/297))

Update simulation error handling if it relied on every exception becoming a failed replication.

`CoverageStudy` now propagates programming errors, unsupported configurations, and errors while
drawing a sample. It retains supported numerical and fitting failures as `ReplicationFailure`
records.
([#285](https://github.com/esbraun/cleverly-tmle/pull/285))

Refit analyses when migrating saved results to 0.1.3. Loading another package version's result
emits `VersionMismatchWarning`; the loader does not migrate the result and loading may fail.
For results containing objects saved by value, use the same Python minor version that saved them.
([#297](https://github.com/esbraun/cleverly-tmle/pull/297))

## Added

### Longitudinal estimators

`cleverly.longitudinal.SDR` and `CausalStudy(...).estimate(method="sdr")` add sequentially
doubly robust estimation. Consistency requires, at each node, a consistent mechanism ratio or
a consistent regression of the transformed outcome. This consistency property alone does not
establish valid confidence intervals; inference requires additional rate and regularity conditions.
([#284](https://github.com/esbraun/cleverly-tmle/pull/284))

SDR supports cross-fitting through a separate recursion within each training fold.
Its registered evidence covers two decision nodes and reports finite-sample coverage shortfalls
in two declared cells.
([#284](https://github.com/esbraun/cleverly-tmle/pull/284),
[#293](https://github.com/esbraun/cleverly-tmle/pull/293))

`cleverly.longitudinal.IPSI` estimates incremental interventions on binary treatments over time.
Use `IncrementalEffect` with `method="ipsi"` through `CausalStudy`.
Each `Incremental(delta)` multiplies treatment odds at the declared nodes.
The estimator supports dropout, survival and competing-risk horizons, and contrasts between plans.
([#286](https://github.com/esbraun/cleverly-tmle/pull/286))

IPSI requires no treatment positivity at a tilt node, but it requires a consistent treatment
mechanism there. It is not doubly robust. Dropout still requires observation positivity, and
fixed-arm nodes retain their positivity requirements. The registered longitudinal incremental
studies have not yet run.
([#286](https://github.com/esbraun/cleverly-tmle/pull/286))

SDR and IPSI report computed estimates even when they leave the outcome range.
Such estimates emit `ParameterSpaceWarning`; derived ratios refuse an undefined scale.
([#284](https://github.com/esbraun/cleverly-tmle/pull/284),
[#286](https://github.com/esbraun/cleverly-tmle/pull/286))

### Long-panel data and reusable rules

`LongitudinalData.from_long_panel()` accepts one row per unit and decision time.
`LongPanel` supplies the matching `CausalStudy` design, and `to_long_panel()` exports an adapted
container. The reader checks temporal ordering, duplicate rows, gaps, late entry, censoring,
and unit-level declarations before fitting learners.
([#297](https://github.com/esbraun/cleverly-tmle/pull/297))

`each_node(rule, n_times)` creates one dynamic rule per decision node.
Each rule receives its node index and the history available at that node.
([#297](https://github.com/esbraun/cleverly-tmle/pull/297))

### Outcome-adaptive C-TMLE

The per-arm outcome-adaptive design accepts baseline strata, missing outcomes, and in-sample
missing treatments. With missing outcomes, it also supports natural-course means and
population attributable risk and fraction. Missing-treatment fits still refuse those
natural-course targets and cross-fitting.
([#283](https://github.com/esbraun/cleverly-tmle/pull/283))

These fits report point estimates and plug-in diagnostics under `generated_design_plugin`.
Their registered studies identify standard-error deficits, so neither per-arm nor shared
outcome-adaptive fits supply inferential intervals.
([#279](https://github.com/esbraun/cleverly-tmle/pull/279),
[#280](https://github.com/esbraun/cleverly-tmle/pull/280),
[#294](https://github.com/esbraun/cleverly-tmle/pull/294))

### Resumable simulation

`CoverageStudy(checkpoint=DirectoryCheckpoint(...))` saves each completed replication and
resumes unfinished work. `ReplicationCheckpoint` defines the interface for other storage.
Use a separate checkpoint directory for each study configuration: stored seeds are checked,
but the data-generating process and estimator configuration are not.
([#285](https://github.com/esbraun/cleverly-tmle/pull/285))

## Changed

### Inference documentation

The DR-TMLE reference now distinguishes conditions that establish an interval from cases where
a first-order remainder remains. A new witness study measures these cases, and the tutorial
uses a declared treatment mechanism. The reference also explains the observed bias with a
gradient-boosting mechanism and an incorrect outcome regression.
([#278](https://github.com/esbraun/cleverly-tmle/pull/278),
[#282](https://github.com/esbraun/cleverly-tmle/pull/282),
[#291](https://github.com/esbraun/cleverly-tmle/pull/291))

Among point-treatment fits, only in-sample ordinary `TMLE` retains an inferential full-refit
bootstrap interval. Its registered evidence uses correctly specified nuisances.
The documentation identifies the remaining evidence gap for data-adaptive learners.
([#298](https://github.com/esbraun/cleverly-tmle/pull/298))

## Fixed

### Saved inline policies

`result.save()` now preserves lambdas, closures, and objects defined in notebooks or `__main__`
by value through `cloudpickle`. Loaded results can evaluate these rules, policy densities,
and maps in a new interpreter. Importable functions and classes still require their modules
on the loading machine.
([#297](https://github.com/esbraun/cleverly-tmle/pull/297))

## Compatibility

cleverly supports Python 3.11, 3.12, and 3.13 with pandas or Polars dataframes.
`cloudpickle>=3.0` is now a core dependency.
([#297](https://github.com/esbraun/cleverly-tmle/pull/297))

This remains an alpha release. The package does not promise compatibility for public APIs,
`CausalStudy` objects, pickled objects, or saved results. Follow the upgrade notes when moving
an analysis to 0.1.3.

**Full changelog:** [v0.1.2...v0.1.3](https://github.com/esbraun/cleverly-tmle/compare/v0.1.2...v0.1.3)
