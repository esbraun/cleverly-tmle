# cleverly 0.1.5

This alpha release adds the parametric g-formula and structural nested mean estimation.
It extends longitudinal curves, repeated cross-fitting, categorical incremental policies,
and parametric C-TMLE intervals. These are the changes since 0.1.4.

## Upgrade notes

- Declare `q_bounds` for a continuous outcome in a longitudinal modified treatment policy fit,
  including an in-sample fit. The fit now refuses an undeclared outcome scale before fitting a learner.
  Use `Targeting(q_bounds=(lower, upper))` through the study API, or `q_bounds=` on the engine.
  ([#359](https://github.com/esbraun/cleverly-tmle/pull/359))
- Declare recorded dose atoms with `dose_atoms=`. A continuous-dose policy fit now refuses detected
  undeclared atoms. Declare the dose support when its boundaries have a substantive meaning;
  `Support(lower, upper)` can use history-dependent boundaries.
  ([#359](https://github.com/esbraun/cleverly-tmle/pull/359))
- Pass `simultaneous=False` for a per-arm parametric `CTMLE` fit with `repeats > 1` and multiple estimates.
  The default simultaneous band now raises `CapabilityError` for that composition.
  ([#364](https://github.com/esbraun/cleverly-tmle/pull/364))
- Check `ParametricOutcome(arm_columns=...)` against the treatment design. Point estimators now refuse
  a mismatch before fitting. For K treatment arms, use `arm_columns=K - 1`.
  ([#364](https://github.com/esbraun/cleverly-tmle/pull/364))
- Expect a longitudinal `n_bootstrap=` fit with multiple estimates to also report an experimental
  `result.bootstrap_band` under the default `simultaneous=True`. Pass `simultaneous=False` to omit it.
  ([#358](https://github.com/esbraun/cleverly-tmle/pull/358))
- Update code that reads plug-in omitted-variable refusal keys. `cross_fit` and `penalty_bias` are removed.
  The new keys are `truncation_active`, `correction_unstable`, and `corrected_nonpositive`.
  ([#368](https://github.com/esbraun/cleverly-tmle/pull/368))

Refit analyses when you move saved results to 0.1.5. Loading another version's result emits
`VersionMismatchWarning`. The loader does not migrate the result, and the load can fail.

## Added

### Parametric g-formula

- `cleverly.longitudinal.ParametricGFormula` estimates regimen levels and contrasts by simulating
  covariate histories forward. Use `gformula()` or the study method `"parametric_gformula"`.
  It covers end-of-study, survival, and competing-risk outcomes. Repeated outcomes remain refused.
  ([#352](https://github.com/esbraun/cleverly-tmle/pull/352))
- `ParametricGFormula` admits static, dynamic, known stochastic, incremental, and modified treatment policies.
  It also admits fixed continuous doses and doses set by a known rule, under its declared model reliance.
  ([#352](https://github.com/esbraun/cleverly-tmle/pull/352))
- A g-formula fit with `GLM` models reports a stacked influence-curve interval that includes simulation error.
  Every model that the estimate reads must be correctly specified. A different learner gives
  `adaptive_baseline_plugin`, with the interval withheld. The analytic variance is a natural extension;
  no published result states it for this simulation estimator.
  ([#352](https://github.com/esbraun/cleverly-tmle/pull/352))
- `result.simulation.monte_carlo_share` reports the simulation share of each variance.
  `MonteCarloWarning` fires above 10%; increase `n_draws` to reduce that share.
  ([#352](https://github.com/esbraun/cleverly-tmle/pull/352))

The registered `parametric-gformula` study supplies Python property evidence.
One interval-calibration cell remains red under its declared finite-sample reading.
Four fixtures also agree with `pygformula` 1.2.0: all 63 coefficient rows and 21 estimate rows pass.
These fixtures do not compare repeated-sampling intervals.
([#352](https://github.com/esbraun/cleverly-tmle/pull/352),
[#356](https://github.com/esbraun/cleverly-tmle/pull/356))

### Structural nested mean estimation

- `cleverly.longitudinal.StructuralNestedMean` and `snmm()` estimate linear structural blips,
  the reference mean, and static-sequence levels and contrasts. `blip_features=` declares effect modifiers.
  Every reported level and contrast states its reliance on the declared linear structural nested mean model.
  ([#363](https://github.com/esbraun/cleverly-tmle/pull/363))
- `result.check_model()` compares structural contrasts with `LTMLE` on shared folds.
  It uses five seeded splits by default and a Hotelling F reference with estimated degrees of freedom.
  The check remains experimental. A non-rejection does not confirm the model.
  ([#363](https://github.com/esbraun/cleverly-tmle/pull/363))

The method refuses adaptive policies, censoring, survival, and repeated outcomes.
A blip feature that reads a time-varying covariate also prevents static levels and contrasts.
Registered support rows cover studied cross-fitted `GLM` configurations, including known weights
and at least 40 clusters. Other configurations remain experimental. EconML comparator runs are pending.
([#363](https://github.com/esbraun/cleverly-tmle/pull/363))

### Longitudinal curves and repeated cross-fitting

- `SDR` and `IPSI` now fit `Repeated` outcomes. Each reported horizon estimates the mean
  on the panel truncated after that node. `IncrementalMean` and `IncrementalEffect` accept `horizons=`.
  The IPSI study has Python property evidence; the SDR study has Python evidence with `lmtp` runs pending.
  ([#354](https://github.com/esbraun/cleverly-tmle/pull/354),
  [#370](https://github.com/esbraun/cleverly-tmle/pull/370))
- `LTMLE`, `SDR`, and `IPSI` accept `repeats=R` with cross-fitting.
  Results report median estimates and split-adjusted variances. Contrasts form within each draw before aggregation.
  `result.repeats` retains the draws, and `repeat_spread()` reports their spread.
  Repeated-outcome LTMLE also accepts repeats.
  ([#372](https://github.com/esbraun/cleverly-tmle/pull/372))

Repeated longitudinal fits require more than one fold and refuse `n_bootstrap > 0`.
The registered study measures five folds and three repeats on its declared laws.
Only configurations covered by its support rows read `supported`; other repeated fits remain experimental.
([#372](https://github.com/esbraun/cleverly-tmle/pull/372),
[#374](https://github.com/esbraun/cleverly-tmle/pull/374))

### Categorical incremental policies

- `Incremental({level: factor})` tilts categorical treatment probabilities in point `TMLE` and longitudinal `IPSI`.
  Unnamed levels keep factor 1. An all-categorical vector uses tuple keys.
  A scalar multiplier retains its binary meaning and is refused above two levels.
  ([#367](https://github.com/esbraun/cleverly-tmle/pull/367))

The registered study measures three-level point treatments and two three-level longitudinal nodes,
with and without dropout. It supplies Python property evidence without a comparator.
([#367](https://github.com/esbraun/cleverly-tmle/pull/367))

### Continuous-dose policies

- `cleverly.interventions.Support` and `dose_atoms=` declare continuous-dose boundaries and recorded point masses.
  Policies can send an interval to a declared atom; their ratios include that probability mass.
  The declarations also apply to vector and mixed treatment nodes.
  ([#359](https://github.com/esbraun/cleverly-tmle/pull/359))
- `Shift` and `Scale` accept tuned caps through `DoseQuantile` and `SupportBoundary`.
  Declare `target="population"` or `target="fold_average"` explicitly.
  The population target requires a finite-dimensional tuner, such as `DoseQuantile` or `LinearQuantileRegression`.
  The fold-average target requires cross-fitting.
  ([#359](https://github.com/esbraun/cleverly-tmle/pull/359))

The registered `mtp-boundaries`, `mtp-pricing`, and `mtp-tuning` studies cover declared configurations.
The boundary and pricing studies have R comparisons pending; the tuning study has no comparator.
Four primary pricing rows at five nodes remain experimental because of interval undercoverage.
The learned-density pricing arms report conservative standard errors;
these results do not establish calibration for arbitrary density learners or sample sizes.
([#359](https://github.com/esbraun/cleverly-tmle/pull/359))

### Targeting and fitting controls

- `Targeting` adds `iterate="best"`, `stopping="scaled_var"` or `"sample_size"`,
  `step_bound=`, and `step_search=`. These controls apply to point TMLE targeting and LTMLE node solves.
  Engines without that fluctuation refuse the controls. Defaults preserve the earlier fitting path.
  ([#353](https://github.com/esbraun/cleverly-tmle/pull/353))
- `ModelSpec(outcome_by_arm=True)` fits a separate point-treatment outcome regression for each arm.
  This option carries an experimental support label.
  ([#353](https://github.com/esbraun/cleverly-tmle/pull/353))
- `CTMLE(patience=...)` stops candidate scoring when the declared counter reaches its limit.
  It applies to `greedy`, `ordered`, `discrete`, `lasso`, and `learners` and remains experimental.
  ([#353](https://github.com/esbraun/cleverly-tmle/pull/353))
- `PointTreatment(treatment_adjustment=...)` restricts treatment-mechanism covariates while retaining
  the full outcome adjustment set. The interval requires the declared mechanism restriction and a consistent mechanism fit.
  The result records the restriction and carries an experimental support label.
  ([#353](https://github.com/esbraun/cleverly-tmle/pull/353))

Nondefault stopping rules remain experimental. Best-iterate and bounded-step fits retain their support
under relative stopping when they solve the score equation. R fixtures check the targeting,
arm-specific outcome, patience, and covariate-pool options. The C-TMLE fixtures compare the options,
not parity of the complete fits.
([#353](https://github.com/esbraun/cleverly-tmle/pull/353),
[#361](https://github.com/esbraun/cleverly-tmle/pull/361))

### Sparse-regimen diagnostics and bootstrap

- `LTMLE` and `SDR` accept `bootstrap_interval="bca"` with `n_bootstrap=`.
  BCa intervals and the studentized bootstrap sup-t band remain experimental.
  No committed study establishes their coverage yet. The default bootstrap interval remains percentile.
  ([#358](https://github.com/esbraun/cleverly-tmle/pull/358))
- Longitudinal results add `diagnostics.influence()` with effective counts, skewness, kurtosis,
  and the predicted Edgeworth coverage loss. `SparseInfluenceWarning` flags noisy influence-curve variance estimates
  on eligible fits. The diagnostic is a direct call and is absent from `run_all()`.
  ([#358](https://github.com/esbraun/cleverly-tmle/pull/358))

## Changed

- Per-arm outcome-adaptive `CTMLE` with `ParametricOutcome` and `ParametricMechanism` now reports
  stacked intervals for cross-fitted and repeated fits. Cross-fitted intervals require at most 20 folds
  and at least 30 rows in each held-out fold. Studied binary fits use five folds and one or three repeats.
  ([#357](https://github.com/esbraun/cleverly-tmle/pull/357))
- The same parametric C-TMLE stack now admits multiple arms, observation weights,
  `ey_obs`, `par`, `paf`, and simultaneous bands over estimates that supply inference.
  Estimated weights use `estimated_weight_plugin`. Four-arm fits, weighted in-sample three-arm fits,
  and weighted repeated fits remain experimental. Intervals retain the model conditions of the parametric stack.
  ([#364](https://github.com/esbraun/cleverly-tmle/pull/364))
- `omitted_variable_bounds(..., nu2_estimator="plugin")` now provides confidence limits for qualifying
  cross-fitted logistic mechanisms, including fixed L2 penalties. It subtracts a second-order bias correction
  from the plug-in estimate of `nu2`. Active or near truncation, unstable correction,
  or a nonpositive corrected value can refuse the limits. The default doubly robust estimator remains available.
  ([#368](https://github.com/esbraun/cleverly-tmle/pull/368))

The corrected plug-in study uses correctly specified mechanisms at n = 1,000.
Its cross-fitted three-arm lower bounds include one conservative cell and one refusal-limited cell.
It supplies no cross-implementation evidence or study coverage of weights, clusters, or pipelines.
([#368](https://github.com/esbraun/cleverly-tmle/pull/368))

### Documentation and datasets

- `make_sequential_pricing`, `pricing_protocol`, and `pricing_truth` provide a three-month pricing law
  with exact truths. The sequential-pricing tutorial replaces `longitudinal-tmle.ipynb`.
  It compares price policies with single regressions and reports support as the proposed price rise increases.
  ([#365](https://github.com/esbraun/cleverly-tmle/pull/365))

## Fixed

- Linux memory checks now read reclaimable memory through `MemAvailable` and respect cgroup memory headroom.
  Density preflight checks no longer rely only on free pages or the host's uncapped memory count.
  ([#355](https://github.com/esbraun/cleverly-tmle/pull/355),
  [#369](https://github.com/esbraun/cleverly-tmle/pull/369))
- `LongitudinalResult.summary()` no longer repeats each follower line for every competing cause.
  ([#353](https://github.com/esbraun/cleverly-tmle/pull/353))
- `sensitivity.benchmark()` on a longitudinal fit now raises `CapabilityError` instead of `AttributeError`.
  ([#372](https://github.com/esbraun/cleverly-tmle/pull/372))
- `offers_protocol` now defines eligibility using the subscription term fixed at the first send.
  The earlier wording selected accounts by whether they stayed open after treatment began.
  ([#366](https://github.com/esbraun/cleverly-tmle/pull/366))

## Packaging and compatibility

This remains an alpha release. The package does not promise compatibility for public APIs,
`CausalStudy` objects, pickled objects, or saved results. Follow the upgrade notes when you move
an analysis to 0.1.5.

**Full changelog:** [v0.1.4...v0.1.5](https://github.com/esbraun/cleverly-tmle/compare/v0.1.4...v0.1.5)


