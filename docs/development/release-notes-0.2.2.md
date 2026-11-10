# cleverly 0.2.2

This release adds a smooth dose-response curve for a continuous treatment, and three survival
analyses: separable effects of a treatment on competing events, collaborative TMLE of a risk curve,
and simulated confounding on survival follow-up.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.

## Added

### Dose-response curves

- `cleverly.estimators.DoseResponseTMLE` estimates a smooth dose-response curve for a continuous
  treatment with cross-validated TMLE (Bibaut and van der Laan, arXiv:1706.07408). The target at
  each dose is the curve averaged by a local-linear kernel over a declared region, set with
  `cleverly.dose_response.DoseResponse` (doses, bandwidths and region). Each dose and bandwidth
  pair gets a Wald interval, and a multiplier band covers all pairs at once. A per-pair support
  report gives the effective sample size, and a pair below `support_floor` (default 50) reads
  `experimental` with the status `low_dose_support_plugin`. Discrete doses, missing data, dose atoms, clusters and a known
  treatment mechanism are refused. `cleverly.plotting.curve` draws the result.
  ([#587](https://github.com/esbraun/cleverly-tmle/pull/587))
- Pointwise intervals read `supported` in the cross-fitted, unweighted configurations the
  `dose-response` study measured: a binary outcome with the logistic fluctuation (with or without
  baseline strata) and a continuous outcome with the linear fluctuation. The band reads
  `supported` only for a binary outcome with the logistic fluctuation and no strata, when every
  pair it covers is supported. Every other configuration reads `experimental`.
  ([#598](https://github.com/esbraun/cleverly-tmle/pull/598))

### Survival and competing risks

- `cleverly.longitudinal.Separable` declares a separable effect for competing events (Stensrud et
  al. 2022): each part of a treatment gets its own level, and `first=` names the cause counted
  first within a time step. Hazard `LTMLE` fits it with `targeting="one_step"`, with or without
  censoring. Fits inside the `separable-effects` study (support rows G49, in sample and two-fold
  cross-fitted, counting the study's cause order) read `supported`.
  ([#585](https://github.com/esbraun/cleverly-tmle/pull/585),
  [#594](https://github.com/esbraun/cleverly-tmle/pull/594))
- `SurvivalCTMLE` fits the collaborative TMLE of a treatment-specific risk curve (Stitelman and van
  der Laan 2010), selecting main-term treatment and pooled censoring models for each arm of a
  binary baseline treatment. `CausalStudy` fits it with `method="collaborative_tmle"` on a
  `TimeToEvent` design. It reports point estimates only: `std_error` and `ci` are refused, and
  `plugin_std_error` and `plugin_interval` remain as diagnostics. Point estimates inside the
  `survival-ctmle` study (support rows G50) read `supported`.
  ([#597](https://github.com/esbraun/cleverly-tmle/pull/597))
- `simulated_confounding()` runs on a survival or competing-risk longitudinal fit whose treatment
  is one baseline decision held over every node, censored or not. The simulated confounder shifts
  each node's event log-odds, censoring is redrawn at the perturbed treatment, and
  `outcome_cause=` moves one cause-specific hazard instead of the all-cause hazard. Each cell
  reports the shares of redrawn censoring, extended follow-up and new events.
  `LongitudinalData.with_replayed_followup` rebuilds the panel. Separable regimens, declared
  censoring, repeated outcomes and treatments that are not held are refused. This surface makes
  no coverage claim. ([#586](https://github.com/esbraun/cleverly-tmle/pull/586))

### Datasets

- `cleverly.datasets.make_heart_failure`, `heart_failure_truth` and `heart_failure_protocol`
  simulate a heart-failure clinic with readmission and death as competing events. The truth is
  exact for any pair of part doses, in both event orders.
  ([#591](https://github.com/esbraun/cleverly-tmle/pull/591))

## Fixed

- A `Separable` fit that counts the other cause first no longer reads `supported`. Support rows
  G49 now key the cause order, and cover only the order the study measured.
  ([#594](https://github.com/esbraun/cleverly-tmle/pull/594))
- `DoseResponseTMLE` estimates in measured configurations no longer read `experimental`.
  ([#598](https://github.com/esbraun/cleverly-tmle/pull/598))

## Documentation

- The user guide has new pages for time-to-event outcomes and for reading the output every
  analysis prints. Statements about intervals, refusals and support labels that contradicted the
  code are corrected. ([#589](https://github.com/esbraun/cleverly-tmle/pull/589),
  [#593](https://github.com/esbraun/cleverly-tmle/pull/593),
  [#596](https://github.com/esbraun/cleverly-tmle/pull/596),
  [#599](https://github.com/esbraun/cleverly-tmle/pull/599))
- The documentation site now deploys with each release rather than with each merge.
  ([#592](https://github.com/esbraun/cleverly-tmle/pull/592))

**Full changelog:** [v0.2.1...v0.2.2](https://github.com/esbraun/cleverly-tmle/compare/v0.2.1...v0.2.2)
