# cleverly 0.2.3

This release adds adaptive TMLE for a randomized trial augmented with external data, and a
targeted missingness tilt with its own interval. It also fixes linked marginal structural models,
competing-risk hazards, the SuperLearner, dose-response lookups and transformed repeated
longitudinal contrasts.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.
- Refit marginal structural models under a logit or log link: their coefficients and intervals
  change. The coefficient is now the minimiser of the link's quasi-likelihood (cumulant) loss, as
  in Rosenblum and van der Laan (2010) and R `ltmleMSM`, instead of mean-scale least squares.
  Identity-link MSMs are unchanged. One fluctuation now solves the scores, so
  `Fluctuation.stratified` is removed, stratum coefficients of a linked MSM come from that pooled
  fluctuation, and `res.fluctuations["msm"].projection` is `None` where it held the per-round
  trace. A custom `cleverly.msm.Link` now takes `cumulant` (an antiderivative of `inverse`) in
  place of `curvature`, and `register_link` checks it. ([#607](https://github.com/esbraun/cleverly-tmle/pull/607))
- `meta_learner="auto"`, the `SuperLearner` default, now combines regression candidates with
  `"cc_ls"`; classification still uses `"nnloglik"`. Every default `SuperLearner` fitted as a
  regression goes through it: a continuous outcome, and the pseudo-outcome regressions of
  longitudinal fits even for a binary outcome. Those estimates change. Pass a
  `SuperLearner(meta_learner="nnls")` to keep the earlier combination, which has no oracle
  guarantee. ([#605](https://github.com/esbraun/cleverly-tmle/pull/605))
- On an ordinary `TMLE` fit, `missingness_tilt()` rows at gamma != 0 for the parameters listed
  under Added now come from a targeted estimate of the tilted parameter. Their points move and
  their intervals are new, and `tipping_gamma(use_ci=True)` reads them. Regime means, MSM
  coefficients and excluded fits report no interval off gamma = 0. Their missing-at-random
  spread, centred on the tilted point, appears under the `plugin_` names as a diagnostic.
  ([#603](https://github.com/esbraun/cleverly-tmle/pull/603),
  [#610](https://github.com/esbraun/cleverly-tmle/pull/610))

## Added

### Adaptive TMLE

- Adaptive TMLE (van der Laan, Qiu, Tarp and van der Laan, J. Causal Inference 14(1):20240025)
  estimates the average treatment effect from a randomized trial pooled with external
  real-world rows. Declare the design with `TrialWithExternalData(source=...)` and the estimand
  with `TrialATE(population="pooled" | "trial")`; `CausalStudy.estimate()` picks
  `AdaptiveTMLEMethod` (`"atmle"`) on this design. Lasso-selected working models describe the
  effect and the external bias; `form="cv"` (default) is the paper's adaptive CV-TMLE and
  `form="pooled"` its Algorithm 1. The treatment is binary and the effect is a difference; other
  treatments and ratio scales are refused. The interval covers the effect projected on the
  working models the fit selected. It covers the pooled or trial effect only when those models
  approximate both the treatment effect and the external bias closely enough, and no data check
  confirms that, so every estimate carries the status `working_model_projection` and reads
  `experimental`. Results are `AdaptiveTMLEResult`; the technical reference
  (`technical-reference/adaptive-tmle.md`) lists what is refused.
  ([#604](https://github.com/esbraun/cleverly-tmle/pull/604))

### Missingness tilt

- `missingness_tilt()` targets the tilted parameter at each gamma != 0 for the arm means,
  contrasts, `att`, `atc`, `rr`, `or`, `ey_obs`, `par` and `paf` of an ordinary `TMLE` fit. One
  logit-scale fluctuation of the outcome regression and the missingness mechanism solves the
  tilted parameter's own estimating equation, and the Wald interval comes from its own influence
  curve (`rr` and `or` by the delta method on the log scale). In-sample and cross-fitted fits,
  `cv_evaluation`, observation weights, clusters and `repeats=` are supported. A `DataWarning`
  names an update that moves the missingness probability below `missingness_bound`; the value is
  not clipped, because clipping would break the solved equation. Intervals read the registry under
  `missingness_tilt_interval`. The registered `missingness-tilt-inference` study covers `ey1`,
  `ey0`, `ate`, `att`, `rr`, `ey_obs` and `par` on binary outcomes with two arms, its GLM learners
  and n >= 1,000, fitted in sample, with fixed weights, or at 10 folds. Every other interval reads
  descriptive and is listed in `MissingnessTilt.descriptive`. Regime means, MSM coefficients,
  `DRTMLE` and `CTMLE` fits, composite-missingness and controlled-direct-effect fits, and other
  estimators get no tilted interval; `scope-and-refusals.md` gives the reason for each. ([#610](https://github.com/esbraun/cleverly-tmle/pull/610))

### SuperLearner

- `SuperLearner(meta_learner="cc_ls")` combines candidates by least squares on the simplex (R
  `method.CC_LS`), the combination the oracle inequality covers for squared error with a bounded
  outcome. It falls back to the discrete selector if the solver fails or the ensemble's risk
  exceeds the best single candidate's, and `"nnloglik"` falls back to it.
  ([#605](https://github.com/esbraun/cleverly-tmle/pull/605))

## Fixed

- `missingness_tilt()` no longer publishes, at gamma != 0, a confidence interval built from the
  missing-at-random standard error. The tilted parameter has its own sampling variance: with 5,000
  rows per arm, an outcome mean of 0.1, 30% recorded and gamma = 2, its standard deviation is
  0.0174 against the missing-at-random 0.0078. The targeted tilt above restores intervals where its
  curve is derived. ([#603](https://github.com/esbraun/cleverly-tmle/pull/603),
  [#610](https://github.com/esbraun/cleverly-tmle/pull/610))
- Linked MSM fits no longer stop at a stationary point that is not the declared minimiser. The
  quasi-likelihood loss is convex, a fit with no finite minimiser (separation) reports
  non-convergence, and a point-treatment projection that stops short raises
  `ConvergenceWarning`. ([#607](https://github.com/esbraun/cleverly-tmle/pull/607))
- Hazard survival fits (`survival_model="hazard"`) with competing events keep each cause's share
  in a cell where every follower has an event, and when the largest cause sits at the upper logit
  clip. Earlier versions split such events equally among the causes in a full cell, and distorted
  the cause ratios at the clip.
  Single-cause fits are unchanged; refit competing-risk hazard fits.
  ([#602](https://github.com/esbraun/cleverly-tmle/pull/602))
- Refitting a `SuperLearner` on a binary outcome after a multiclass fit no longer predicts from
  the old classes. ([#605](https://github.com/esbraun/cleverly-tmle/pull/605))
- `DoseResponseResult.estimate` matches a dose by exact value or by parameter name and raises
  `KeyError` otherwise. It used to return a nearby dose. The held-modifier band lookup in the
  curve plot matches exactly, or within 1e-12 when one grid value does.
  ([#605](https://github.com/esbraun/cleverly-tmle/pull/605))
- The missingness and truncation sensitivity plots join their points in order of the x value.
  ([#605](https://github.com/esbraun/cleverly-tmle/pull/605))
- On a repeated longitudinal fit, `contrast(..., transform=...)` keeps its support record, so it
  no longer reports that it was built outside a classified fit.
  ([#611](https://github.com/esbraun/cleverly-tmle/pull/611))
- A repeated fit whose draws differ in a parameter status, for example when one draw has a
  constant node or a failed arm stack, no longer refuses with "carries different support records
  across repeats". Each draw now takes the support record of a draw that holds the reported
  status. ([#611](https://github.com/esbraun/cleverly-tmle/pull/611))

## Documentation

- New tutorial for a trial with external data, with a user-guide page and technical reference.
  ([#604](https://github.com/esbraun/cleverly-tmle/pull/604))
- The discharge-to-readmission tutorial adds a sensitivity replay of the survival analysis.
  ([#606](https://github.com/esbraun/cleverly-tmle/pull/606))
- The dose-response evidence pages add a comparison with R npcausal `ctseff`.
  ([#608](https://github.com/esbraun/cleverly-tmle/pull/608))
- The evidence reading guide says that a verdict grows easier to pass as replications grow only in
  probability; one more replication can turn a realized pass into a fail.
  ([#611](https://github.com/esbraun/cleverly-tmle/pull/611))

**Full changelog:** [v0.2.2...v0.2.3](https://github.com/esbraun/cleverly-tmle/compare/v0.2.2...v0.2.3)
