# cleverly 0.2.5

This release adds a continuous-time survival TMLE, moves the default simultaneous band and the
interval of a binary mean to the logit scale, and fixes several targeting and fitting defects.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.
- The default simultaneous band is now `band_scale="logit"`, `multiplier_kind="mammen"` and
  `n_multiplier=10_000`, so band endpoints and critical values change. This change leaves
  pointwise intervals, estimates, standard errors, p-values, `CurveBand`, `effect_curve` and
  `bootstrap_band` as they were. Pass the earlier values explicitly to reproduce an earlier band; such a band reads
  experimental. ([#629](https://github.com/esbraun/cleverly-tmle/pull/629))
- The interval of an unclustered mean of a binary outcome is now built on the logit scale, and
  `wald_test` inverts that interval. Differences, ratios, coefficients, continuous and clustered
  estimates, and estimates with a Student t reference, keep their intervals. ([#637](https://github.com/esbraun/cleverly-tmle/pull/637))
- An estimate that reads an arm, or an arm-by-modifier cell, whose observed outcomes are all equal
  now withholds its interval and p-value with the status `constant_arm_cell_plugin`, and its
  contrasts inherit that status. A binary mean on or outside (0, 1) withholds its interval with
  `out_of_range_plugin`. ([#637](https://github.com/esbraun/cleverly-tmle/pull/637))
- `cv_evaluation=True` on a clustered cross-fitted fit now refuses clusters of unequal mass in the
  estimand's denominator, except for learned-rule targets. Use the default
  `cv_evaluation=False`, which estimates the same effect without the fold-average ratio bias. ([#632](https://github.com/esbraun/cleverly-tmle/pull/632))
- A fit that declares `stopping="scaled_var"` or `stopping="sample_size"` now widens every Wald
  interval by the unsolved score residual, and refuses the bootstrap.
  ([#635](https://github.com/esbraun/cleverly-tmle/pull/635))
- A learner with no `random_state` of its own now takes the fit's seed, so a seeded fit with an
  unseeded stochastic learner (for example a SAGA lasso) gives different numbers than before, and
  the same numbers on every run. ([#639](https://github.com/esbraun/cleverly-tmle/pull/639))

## Added

### Survival

- `ContinuousSurvivalTMLE` (`cleverly.survival`), reached through
  `TimeToEvent(grid="continuous")` with `TMLEMethod`, is the one-step TMLE of Rytgaard and
  van der Laan (2024) for the risk of each competing cause under each arm of a binary baseline
  treatment, with no time grid. The registered `continuous-time-survival` study supports risks
  and risk differences of the cross-fitted default at n ≥ 1,000 with two causes, at times by
  which 40% to 85% of observed events have happened; other configurations read experimental.
  ([#636](https://github.com/esbraun/cleverly-tmle/pull/636))
- `LTMLE(msm=..., survival_model="hazard", targeting="one_step")` targets the coefficients of a
  marginal structural working model over the discrete-time risk curve, for one cause or each cause
  of a competing-risks fit. Support rows G60 license its in-sample Wald intervals and the band on
  one cause. ([#638](https://github.com/esbraun/cleverly-tmle/pull/638))

### Inference

- `band_reference="t"` bands each estimate at a Student t quantile with its Kish effective count
  as degrees of freedom, keeping the multiplier critical value. The default stays `"normal"`. The
  `band-t-reference` study supports it on the rows it measured.
  ([#634](https://github.com/esbraun/cleverly-tmle/pull/634))
- `cleverly.inference` gains `kish_count`, `satterthwaite_df` and `constant_outcome_cells`, and
  `ParameterEstimate` gains `outcome_family`, `small_sample_df`, `effective_df` and
  `interval_scale`. ([#633](https://github.com/esbraun/cleverly-tmle/pull/633))

### Conditional effects

- `assess_score(compare=...)` estimates the paired risk difference of two external CATE scores
  with a Wald interval. The single-score assessment is now a TMLE, by a pooled logistic
  fluctuation. ([#627](https://github.com/esbraun/cleverly-tmle/pull/627))

### Option support

- `stopping=`, `outcome_by_arm=True`, `treatment_adjustment=`, `ratio_trim=` on LTMLE and SDR,
  `markov_order=`, `patience=` on greedy C-TMLE and a binding `isotonic=True` projection each have
  a registered study and support rows for the configurations it measured.
  ([#631](https://github.com/esbraun/cleverly-tmle/pull/631),
  [#635](https://github.com/esbraun/cleverly-tmle/pull/635))
- The fold-wise DR-TMLE default reads supported for three arms (row D12) and for stratified point
  estimates (row D13). Stratified intervals read experimental, because the smallest stratum's
  standard error runs about 12% short. ([#643](https://github.com/esbraun/cleverly-tmle/pull/643))

## Changed

- A clustered fold-evaluated estimate uses the t reference with J − V degrees of freedom at every
  cluster count, also at 40 clusters or more, where it read the normal quantile. Such a fit at 40
  or more clusters carries no simultaneous band.
  ([#632](https://github.com/esbraun/cleverly-tmle/pull/632))
- `heterogeneity_test` with fewer than 40 clusters reads the Hotelling law with J − 2 degrees of
  freedom and refuses when the number of tested modifiers exceeds it. The earlier F reading
  rejected 0.187 at J = 5 under the null.
  ([#627](https://github.com/esbraun/cleverly-tmle/pull/627))
- `targeting_curve` refuses `fold=`, `cv_evaluation=` and `n_bootstrap=` explicitly.
  ([#627](https://github.com/esbraun/cleverly-tmle/pull/627),
  [#628](https://github.com/esbraun/cleverly-tmle/pull/628))
- A three-arm learned rule reads experimental: its conservative coverage cell fails under the logit
  interval. ([#637](https://github.com/esbraun/cleverly-tmle/pull/637))

## Fixed

- `markov_order=` on a design that holds one baseline decision over every node kept the treatment
  in every window. Earlier versions dropped it after node k + 2, so models that read the treatment
  were fitted pooled over the arms and missed the truth.
  ([#642](https://github.com/esbraun/cleverly-tmle/pull/642))
- Cross-fitted separable-effects targeting no longer stalls short of the root (about 2% of fits
  did). A fold-cell whose root lies outside the hazard bound reads `failure="cell_separated"`.
  ([#626](https://github.com/esbraun/cleverly-tmle/pull/626))
- `step_bound=` with `step_search=True` finds the step length as the root of the exact slope of
  the loss. Earlier versions stalled at a relative score of 1e-9 to 1e-8 and reported
  `converged=False` at the default tolerance.
  ([#640](https://github.com/esbraun/cleverly-tmle/pull/640))
- A searched bounded walk stops when it returns to a state it already visited, instead of cycling
  to `max_iter` on some CPUs. ([#641](https://github.com/esbraun/cleverly-tmle/pull/641))
- `SurvivalDRTMLE` accepts the library's band defaults instead of refusing every `n_multiplier`
  but 2,000. ([#630](https://github.com/esbraun/cleverly-tmle/pull/630))

## Documentation

- The technical reference adds pages for the continuous-time survival TMLE, the hazard working
  model, the t-reference band, the small-sample interval study, the option studies and the
  fold-wise DR-TMLE studies, with their validation-grid rows.
  ([#632](https://github.com/esbraun/cleverly-tmle/pull/632),
  [#633](https://github.com/esbraun/cleverly-tmle/pull/633),
  [#634](https://github.com/esbraun/cleverly-tmle/pull/634),
  [#635](https://github.com/esbraun/cleverly-tmle/pull/635),
  [#636](https://github.com/esbraun/cleverly-tmle/pull/636),
  [#638](https://github.com/esbraun/cleverly-tmle/pull/638),
  [#643](https://github.com/esbraun/cleverly-tmle/pull/643))
- Tutorials with band output or binary means are re-executed with the new defaults.
  ([#629](https://github.com/esbraun/cleverly-tmle/pull/629),
  [#637](https://github.com/esbraun/cleverly-tmle/pull/637))

**Full changelog:** [v0.2.4...v0.2.5](https://github.com/esbraun/cleverly-tmle/compare/v0.2.4...v0.2.5)
