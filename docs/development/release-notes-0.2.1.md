# cleverly 0.2.1

This release adds conditional-effect curves and the EP learner, plotting for fitted results,
targeting curves for priority scores, and longitudinal fits that eliminate a competing event.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.

## Added

### Conditional effects

- `ConditionalContrast` asks how a contrast varies with declared effect modifiers. The projection
  targets `cate_projection` (difference scale) and `log_crr_projection` (log risk-ratio scale)
  fit the best approximation of that contrast inside a declared basis from the new
  `cleverly.bases` module. Each fit returns an `EffectCurveResult` with row-level `predict`
  (optionally with pointwise Wald intervals), coefficient intervals, a band on a declared grid,
  `heterogeneity_test()` and held-out curve diagnostics. The projections support weights,
  clusters, missing outcomes and a pair of arms out of several. They read supported inside the
  setting the `conditional-projection` study measured, and the band reads experimental.
  ([#558](https://github.com/esbraun/cleverly-tmle/pull/558))
- `method="ep"` fits the EP learner (arXiv 2402.01972v2), which debiases a sieve fit with an
  offset-logistic step and predicts the conditional effect row by row on the difference or
  log-ratio scale. A fit reads supported for two arms, one outcome model per arm and an estimated
  treatment model, either with a size schedule and `model=None` or with a fixed size and `model=`
  declared on the difference scale. Every other composition reads experimental, and
  `predict(interval=True)` refuses. ([#562](https://github.com/esbraun/cleverly-tmle/pull/562))
- `LearnedRule(loss="ep")` learns a treatment rule from the EP learner's cross-validated risks. A
  fold-evaluated fit with two arms, a binary outcome, unweighted rows, pooled targeting and
  `repeats=1` reads supported. In the study, 95% intervals covered 0.944 to 0.954 at n = 2,000
  and n = 16,000, and covered 0.901 at the exceptional law at n = 2,000, where the standard
  error was 17% too small. ([#565](https://github.com/esbraun/cleverly-tmle/pull/565))

### Plots

`cleverly.plotting` draws results with matplotlib, installed through the `plot` extra
(`pip install "cleverly[plot]"`). `import cleverly` does not import matplotlib. Each plot draws
only the intervals the fit holds, and a fit with withheld inference draws the point alone.
Results, curves and sensitivity objects also expose `plot()`.

- `forest` shows estimates with intervals, one panel per scale. `sensitivity` shows the truncation
  curve, the marginal sensitivity range with its breakdown, and the missingness tilt. `overlap`
  shows the predicted probability of treatment by arm with the probability limits each estimate
  used. ([#563](https://github.com/esbraun/cleverly-tmle/pull/563))
- `curve` draws risk, survival or mean over time per regimen, restricted mean survival, per-level
  means beside a marginal structural model's working model (through the new
  `cleverly.msm.working_model_means`), effect against a policy parameter, and a conditional
  effect over one modifier. ([#564](https://github.com/esbraun/cleverly-tmle/pull/564))
- `calibration`, `learners`, `repeats`, `bootstrap` and `influence` show the calibration of each
  probability model, the Super Learner's candidate risks and weights, the spread of estimates
  across repeated splits, the bootstrap distribution and the spread of influence values.
  ([#566](https://github.com/esbraun/cleverly-tmle/pull/566))
- `weights` shows the distribution of inverse-probability weights, `balance` shows covariate
  balance before and after weighting, and `zipper` shows the coverage of simulated intervals. The
  new `cleverly.sensitivity.covariate_balance` returns the balance table, and
  `cleverly.longitudinal.node_weights` returns per-node weights.
  ([#567](https://github.com/esbraun/cleverly-tmle/pull/567))
- `confounding` draws a contour of how an estimate's bound moves with the strength of an unmeasured
  confounder, with the no-effect contour, the robustness value and covariate benchmarks, and a map
  of the simulated-confounding grid. ([#568](https://github.com/esbraun/cleverly-tmle/pull/568))
- `forest` and `curve` take `truth=` to mark a known true value in simulated data.
  ([#570](https://github.com/esbraun/cleverly-tmle/pull/570))
- `forest`, `repeats` and `curve` take `outcome_label=` to name the whole quantity. `confounding`
  takes `x_label=` and `y_label=`. ([#572](https://github.com/esbraun/cleverly-tmle/pull/572),
  [#575](https://github.com/esbraun/cleverly-tmle/pull/575))
- The tutorial notebook "Plot results" walks through the plots.
  ([#579](https://github.com/esbraun/cleverly-tmle/pull/579))

### Targeting curves

- `cleverly.targeting_curves` computes the targeting operator characteristic, the cumulative
  gain, AUTOC, Qini and a paired comparison of two priority scores. Intervals come from the
  half-sample bootstrap of Yadlowsky et al. (arXiv:2111.07966v2) applied to targeted scores. An
  external score fitted on other rows, or a split route that refits the conditional effect curve
  on training rows, supplies the score.
  ([#576](https://github.com/esbraun/cleverly-tmle/pull/576))
- Targeting curves read supported inside the scope of the
  `targeting-curves` study, which at n = 2,000 with 6,000 replicates measured coverage of 0.949 to
  0.956. Fewer than 40 clusters withhold every interval. `cleverly.plotting.targeting` draws them.
  ([#576](https://github.com/esbraun/cleverly-tmle/pull/576),
  [#578](https://github.com/esbraun/cleverly-tmle/pull/578))

### Longitudinal fits

- `eliminate=["death"]` on `LongitudinalData.from_frame`, `from_long_panel`, `from_time_to_event`
  and `LTMLE.fit` estimates the risk of one cause had a competing cause been prevented, the
  g-formula (23) of Young, Stensrud, Tchetgen Tchetgen and Hernán (2020). The prevented event is
  handled as censoring, and its probability enters the clever covariate, the hazard path, the
  inverse-probability weights and `ratio_trim=` as a second factor, which `competing_learner=`
  fits. The summary reports that factor as its own `competing` row. The E-value, the
  simulated-confounding replay, generated-outcome and plasmode refutations, `with_events`,
  `to_long_panel` and structural nested mean models refuse an eliminated fit. Fits inside the
  scope of the `eliminated-competing-events` study read supported, and every other composition
  reads experimental. ([#580](https://github.com/esbraun/cleverly-tmle/pull/580))

## Changed

- `ParametricGFormula` skips held nodes when it checks a plan's tilts. A held node repeats the
  baseline treatment, so the plan declares node 1 only.
  ([#580](https://github.com/esbraun/cleverly-tmle/pull/580))

## Fixed

- `thread_limit` no longer leaves OpenBLAS unlimited in a block opened from a second thread under
  threadpoolctl 3.8 or later. ([#559](https://github.com/esbraun/cleverly-tmle/pull/559))

## Packaging and compatibility

- The upper bounds on the `docs` and `dev` extras are removed, and the build backend requires
  `hatchling>=1.27`. Runtime dependencies keep floors only.
  ([#559](https://github.com/esbraun/cleverly-tmle/pull/559))

**Full changelog:** [v0.2.0...v0.2.1](https://github.com/esbraun/cleverly-tmle/compare/v0.2.0...v0.2.1)
