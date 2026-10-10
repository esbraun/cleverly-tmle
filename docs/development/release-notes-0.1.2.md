# cleverly 0.1.2

This alpha release extends missing-data, treatment-policy, and longitudinal analyses, with APIs
for known treatment and censoring mechanisms. It also changes cross-fitting defaults and makes
inference restrictions explicit. These are the changes since 0.1.1.

## Upgrade notes

Review these API changes before refitting an analysis under 0.1.2.

Replace `shifts=` with `policies=` in `TMLE`, `ModifiedTreatmentPolicy`, and
`ModifiedTreatmentPolicyEffect`. Replace `ey_shift` and `ate_shift` with `ey_policy` and
`ate_policy`. `ShiftSet`, `ShiftSupport`, and `check_shift_support` become `PolicySet`,
`PolicySupport`, and `check_policy_support`. Import `Shift` from `cleverly.interventions`.
([#269](https://github.com/esbraun/cleverly-tmle/pull/269))

Cross-fitting now draws unstratified outer folds by default. Treatment-based and outcome-based
fold strata are refused. For continuous outcomes, declare known outcome bounds with
`Targeting(q_bounds=(lower, upper))`. Do not calculate these bounds from the observed sample.
Use `CrossFitting(enabled=False)` for an in-sample fit.
([#218](https://github.com/esbraun/cleverly-tmle/pull/218))

Declare supplied rules, intervention densities, MSM designs, and projection weights as known
when they are fixed independently of the analysis sample. The relevant arguments are
`rule_kind`, `density_kind`, `design_kind`, and `weights_kind`. Undeclared or estimated
functions are refused on these paths.
([#224](https://github.com/esbraun/cleverly-tmle/pull/224),
[#225](https://github.com/esbraun/cleverly-tmle/pull/225),
[#227](https://github.com/esbraun/cleverly-tmle/pull/227),
[#228](https://github.com/esbraun/cleverly-tmle/pull/228))

Check each estimate's `supplies_inference` before reading uncertainty estimates. C-TMLE selection
paths and some estimated-weight DR-TMLE fits withhold standard errors, confidence intervals, and
p-values. Their `plugin_std_error` and `plugin_interval` remain available as diagnostics.
([#223](https://github.com/esbraun/cleverly-tmle/pull/223),
[#224](https://github.com/esbraun/cleverly-tmle/pull/224))

Move `DRTMLEMethod(treatment_probabilities=...)` declarations to
`PointTreatment(treatment_probabilities=...)`. Study designs accept mappings from treatment
levels to probability columns. Pass arrays through the estimator's `fit` method instead.
([#276](https://github.com/esbraun/cleverly-tmle/pull/276))

Replace `diagnostics.stagewise()` with `diagnostics.support()` and
`cv_targeting.fold_evaluated` with `cv_targeting.canonical`. The loader does not migrate saved results.
Loading a result from another version emits `VersionMismatchWarning` and may fail. Refit analyses
under 0.1.2 when migrating saved results.
([#236](https://github.com/esbraun/cleverly-tmle/pull/236))

## Added

### Study protocols and reusable splits

Studies can record their scientific protocol and reuse the outer folds from a previous fit.
`StudyProtocol` records the population, eligibility, treatment strategies, outcome, follow-up,
and assumptions. Identification retains the record, and fitted provenance carries its digest.
([#201](https://github.com/esbraun/cleverly-tmle/pull/201))

`SplitPlan` records assignments across cross-fitting repeats. Pass `result.split_plan` through
`CrossFitting(split_plan=...)` to reuse them. Fits require package-generated assignments and
validate the plan before fitting learners.
([#202](https://github.com/esbraun/cleverly-tmle/pull/202),
[#218](https://github.com/esbraun/cleverly-tmle/pull/218))

### Missing data and baseline strata

Additional estimands support missing outcomes, missing treatments, and estimates within baseline
strata.

`NaturalCourseMean` estimates the population outcome mean with outcomes missing at random.
Ordinary TMLE and the supported stacked cross-fitted configuration are available.
([#207](https://github.com/esbraun/cleverly-tmle/pull/207),
[#210](https://github.com/esbraun/cleverly-tmle/pull/210))

Stacked CV-TMLE supports arm means and contrasts with missing outcomes, subject to its fold,
scale, and composition restrictions.
([#216](https://github.com/esbraun/cleverly-tmle/pull/216))

Missing-outcome DR-TMLE supports randomized treatments with three or more arms. Known
treatment probabilities can be supplied by arm.
([#260](https://github.com/esbraun/cleverly-tmle/pull/260))

TMLE and DR-TMLE support a declared missing treatment, with or without missing outcomes,
through a composite observation indicator. Use `treatment_delta=` on the estimator or
`treatment_missingness=` on `PointTreatment`. Cross-fitting remains unsupported for this
construction. ([#261](https://github.com/esbraun/cleverly-tmle/pull/261))

Ordinary TMLE estimates population attributable risk and fraction with outcomes missing at
random. The stacked cross-fitted route requires a binary outcome and excludes weights and
clusters. ([#262](https://github.com/esbraun/cleverly-tmle/pull/262))

Baseline-stratum targeting extends to incremental interventions, MSMs, natural-course means,
and DR-TMLE. Fold-local targeting and fold evaluation remain unsupported with strata.
([#263](https://github.com/esbraun/cleverly-tmle/pull/263))

### Known treatment and censoring mechanisms

Fits can use treatment probabilities declared by a randomized design instead of estimating them.
`TMLE.fit(treatment_probabilities=...)` and `PointTreatment(treatment_probabilities=...)`
accept known treatment probabilities. TMLE and DR-TMLE use the declaration without fitting a
treatment learner. ([#276](https://github.com/esbraun/cleverly-tmle/pull/276))

LTMLE accepts known treatment and censoring probabilities at individual nodes, including
survival and time-to-event designs. Declare them with `treatment_probabilities=` and
`censoring_probabilities=`; nodes without declarations still estimate their mechanisms. The fit
refuses a regimen that requires an event with declared probability zero. ([#276](https://github.com/esbraun/cleverly-tmle/pull/276))

Known treatment probabilities replace only the treatment mechanism. Missing-outcome and
controlled-direct-effect fits still estimate their additional mechanisms and retain their
inference conditions. ([#276](https://github.com/esbraun/cleverly-tmle/pull/276))

### Treatment policies and learned rules

Policy estimation now covers additional dose transformations and categorical policies.

Modified treatment policies include scaling, piecewise transformations, custom invertible
maps, categorical maps, and randomizers with known probabilities. They run through
point-treatment TMLE and supported longitudinal nodes.
([#269](https://github.com/esbraun/cleverly-tmle/pull/269))

`ratio="classifier"` estimates a policy density ratio by classification. `RiskRatioTilt`
provides a separate binary-treatment policy through the estimator API.
([#269](https://github.com/esbraun/cleverly-tmle/pull/269))

`LearnedRuleValue` estimates the average value of binary treatment rules learned separately
within the training folds. It requires cross-fitting and fold evaluation. Its target is the
fold-average learned-rule value, rather than the value of an optimal rule or one fitted on all
rows. ([#248](https://github.com/esbraun/cleverly-tmle/pull/248),
[#249](https://github.com/esbraun/cleverly-tmle/pull/249))

Longitudinal regimen nodes accept known stochastic categorical policies through
`Stochastic(..., density_kind="known")`.
([#268](https://github.com/esbraun/cleverly-tmle/pull/268))

### Longitudinal and survival analyses

Longitudinal estimation adds baseline-treatment survival input, clustered cross-fitting, and
derived summaries.

`TimeToEvent` and `LongitudinalData.from_time_to_event()` accept one follow-up time and event
code per unit. They support a baseline treatment held over the survival nodes. Censoring
between declared grid times is refused.
([#270](https://github.com/esbraun/cleverly-tmle/pull/270))

Clustered LTMLE cross-fits with whole clusters assigned to folds. Longitudinal MSMs also
cross-fit, with one pooled targeting update per node. `LTMLE(msm=...)` now runs with the
default ten folds. ([#265](https://github.com/esbraun/cleverly-tmle/pull/265),
[#266](https://github.com/esbraun/cleverly-tmle/pull/266))

`ratio()` derives risk ratios or odds ratios from fitted parameters. `wald_test(null=...)`
tests a specified null, and `Transform` supports transformed confidence intervals.
([#258](https://github.com/esbraun/cleverly-tmle/pull/258))

`rmst()` and `rmtl()` report restricted mean survival time and time lost, including
between-regimen contrasts. Gridded time-to-event fits use grid times.
([#258](https://github.com/esbraun/cleverly-tmle/pull/258),
[#270](https://github.com/esbraun/cleverly-tmle/pull/270))

LTMLE accepts `n_bootstrap=` for full refits of resampled trajectories or clusters. Unsupported
inference settings report bootstrap spreads as diagnostics.
([#258](https://github.com/esbraun/cleverly-tmle/pull/258))

### Diagnostics and sensitivity analyses

Reports expose more of the fitted mechanisms and distinguish deferred work from unavailable
analyses.

Support reports include combined mechanism denominators and the concentration of fitted score
weights. These are descriptive measures, without a pass threshold.
([#191](https://github.com/esbraun/cleverly-tmle/pull/191),
[#192](https://github.com/esbraun/cleverly-tmle/pull/192),
[#193](https://github.com/esbraun/cleverly-tmle/pull/193))

Nuisance reports distinguish collaborative working mechanisms and show variation across
repeated splits. Longitudinal reports cover treatment, censoring, outcome, and pseudo-outcome
fits. ([#194](https://github.com/esbraun/cleverly-tmle/pull/194),
[#195](https://github.com/esbraun/cleverly-tmle/pull/195))

Longitudinal truncation curves refit the outcome recursion at each requested mechanism bound.
They report estimate movement and score-cell counts, without selecting a preferred bound.
([#204](https://github.com/esbraun/cleverly-tmle/pull/204))

`assess()` adds a `deferred` status for work awaiting arguments or permission to refit. Its
summary presents returned analyses before inventories of checks and omissions.
([#197](https://github.com/esbraun/cleverly-tmle/pull/197),
[#198](https://github.com/esbraun/cleverly-tmle/pull/198))

## Changed

### Targeting and uncertainty

Clustered intervals account for unequal cluster sizes. Eligible fits below 40 contributing
clusters use a Student t reference. Point-treatment fits below 10 clusters and longitudinal
fits below 20 withhold intervals. Simultaneous bands remain unavailable with a t reference.
([#264](https://github.com/esbraun/cleverly-tmle/pull/264),
[#265](https://github.com/esbraun/cleverly-tmle/pull/265))

Cross-fitted longitudinal targeting uses a pooled fluctuation over held-out predictions. This
changes the fitted construction and can change estimates.
([#220](https://github.com/esbraun/cleverly-tmle/pull/220))

Omitted-variable bounds refuse unsupported estimator compositions. Plug-in point bounds
remain available, but their confidence limits are withheld. Controlled-direct-effect fits no
longer return E-values. ([#223](https://github.com/esbraun/cleverly-tmle/pull/223),
[#229](https://github.com/esbraun/cleverly-tmle/pull/229),
[#230](https://github.com/esbraun/cleverly-tmle/pull/230))

### Density fitting and tutorials

The default density bin count grows with sample size. Density fitting refuses a hazard design
that exceeds available memory.
([#269](https://github.com/esbraun/cleverly-tmle/pull/269))

Method tutorials are executable notebooks with stored outputs.
([#213](https://github.com/esbraun/cleverly-tmle/pull/213))

## Fixed

### Influence curves and bootstrap targets

ATT and ATC omitted-variable bound standard errors now include uncertainty in the
conditioning-arm share. Point bounds are unchanged.
([#230](https://github.com/esbraun/cleverly-tmle/pull/230))

Ratio-scale contrasts and compositions of transformed estimates now apply influence curves
on the correct scale. ([#258](https://github.com/esbraun/cleverly-tmle/pull/258),
[#267](https://github.com/esbraun/cleverly-tmle/pull/267))

Bootstrap replicates retain the point fit's resolved estimands. Missing-outcome fits using
`estimands="all"` no longer fail from requesting unsupported replicate targets.
([#262](https://github.com/esbraun/cleverly-tmle/pull/262))

### Constant longitudinal nodes

Longitudinal nodes with no censoring use a fixed censoring probability of one. Eligible
event-free nodes use hazard zero, avoiding classifier failures on constant targets.
([#270](https://github.com/esbraun/cleverly-tmle/pull/270))

Longitudinal parameters that read a constant node now report `constant_node_plugin` and
withhold confidence intervals and p-values. Derived contrasts and restricted means inherit
the status, and simultaneous bands exclude those parameters.
([#274](https://github.com/esbraun/cleverly-tmle/pull/274))

## Compatibility

cleverly supports Python 3.11, 3.12, and 3.13 with pandas or Polars dataframes.

This remains an alpha release. The package does not promise compatibility for public APIs,
`CausalStudy` objects, pickled objects, or saved results. Follow the upgrade notes when moving
an analysis to 0.1.2.

**Full changelog:** [v0.1.1...v0.1.2](https://github.com/esbraun/cleverly-tmle/compare/v0.1.1...v0.1.2)
