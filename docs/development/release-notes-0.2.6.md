# cleverly 0.2.6

This release makes every plot and assessment operation either run or refuse with a true reason on
every result type, extends the hazard path of `LTMLE` to several treatment nodes, modified
treatment policies and covariates measured after baseline, and adds restricted means to the
continuous-time survival TMLE.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.
- A fit that passes known treatment probabilities (`treatment_probabilities=`, or longitudinal
  `treatment_probabilities=` or `censoring_probabilities=`) beside non-constant observation
  weights must now name `mechanism_law="target"` or `"sample"`, and refuses without it.
  `"target"` reproduces earlier releases bit for bit and is right when randomization holds in the
  population the weights restore. `"sample"` is right when randomization holds in the rows as
  drawn and the weights depend on treatment or outcome; earlier releases gave such fits a bias
  that did not shrink with n. ([#662](https://github.com/esbraun/cleverly-tmle/pull/662))
- An `LTMLE` interval now adds a variance supplement for a node whose fitted regression rows all
  hold one value, at the exact one-sided binomial bound of its hazard. Intervals of such fits
  widen; others do not move. ([#670](https://github.com/esbraun/cleverly-tmle/pull/670))
- `marginal_sensitivity_bounds` with `estimand=None` no longer bounds the ATE of a fit that does
  not report it. It bounds the ATE when the fit reports it, otherwise the one reported parameter
  the bounds can take, and otherwise raises and names the choices. On a multi-arm fit, name the
  contrast. ([#656](https://github.com/esbraun/cleverly-tmle/pull/656))
- On a clustered fit, `marginal_sensitivity_bounds` now builds its limits on the fit's Student t
  reference instead of a normal quantile, so the limits widen below 40 clusters. On a 12-cluster
  fit the Γ = 1 limits move from ±1.24 to the fit's ±1.41.
  ([#656](https://github.com/esbraun/cleverly-tmle/pull/656))

## Added

### Longitudinal and survival

- `ContinuousSurvivalTMLE(restricted_horizons=...)` targets each arm's restricted mean time lost
  to every cause, and `ContinuousSurvivalResult` gains `curve`, `rmst`, `rmtl`, `contrast`, `ratio`
  and `to_dict`. Restricted means take an interval on the logit scale of (0, τ). The registered
  `continuous-survival-functionals` study supports one restricted horizon at or before the last
  target time, with restricted means at a τ by which 67% to 85% of observed events have happened.
  The E-value of a continuous-time risk ratio now runs.
  ([#664](https://github.com/esbraun/cleverly-tmle/pull/664))
- `LTMLE(survival_model="hazard")` accepts a treatment decided at several nodes when every
  covariate is measured at baseline. The registered `hazard-multinode` study supports regimen
  curves with bands at three and four nodes under `targeting="iterative"`, and the one-step
  working model (support rows G62). ([#669](https://github.com/esbraun/cleverly-tmle/pull/669))
- `LTMLE(survival_model="hazard")` fits a modified treatment policy at a baseline decision held
  over the visits, for a categorical, continuous or mixed treatment. The registered `hazard-mtp`
  study supports a six-level dose, a truncated Gaussian dose on the density route, the classifier
  route in sample and cross-fitted, and a one-visit arm and dose vector (support rows G61).
  ([#646](https://github.com/esbraun/cleverly-tmle/pull/646))
- `LTMLE(survival_model="hazard", targeting="one_step")` accepts covariates measured after
  baseline (`time_varying=` at a node after the first) beside a baseline treatment held over the
  nodes, following Rytgaard, Gerds and van der Laan (2022). The registered `hazard-time-varying`
  study supports the point estimates, Wald intervals and the band of static arms with one or two
  causes, in sample and cross-fitted, and of a policy at node 1 (support rows G63.held).
  ([#659](https://github.com/esbraun/cleverly-tmle/pull/659))
- `LTMLE` and `SDR` take `variance="policy_weighted"` for a modified treatment policy with heavy
  cumulative density ratios. It keeps the estimate and reports a standard error at least as large
  as the default's. The registered `mtp-policy-variance` study supports it as conservative or
  calibrated on the rows it measured; the default interval is unchanged.
  ([#645](https://github.com/esbraun/cleverly-tmle/pull/645))
- `LTMLE` and `SDR` fits of a modified treatment policy on a continuous treatment with censored
  follow-up read supported on the cross-fitted classifier route at three nodes with a learned
  censoring model, from the registered `mtp-pricing-censored` study. No estimator code changed.
  ([#667](https://github.com/esbraun/cleverly-tmle/pull/667))
- `ContinuousSurvivalResult` gains `diagnostics`, `sensitivity`, `validate()`, `assess()` and
  `replayability`. Support, score equations, nuisance models and influence run; simulated
  confounding and the sensitivity bounds refuse with their reasons.
  ([#657](https://github.com/esbraun/cleverly-tmle/pull/657))
- A continuous-time survival fit can be refitted (`estimator.refit(data, random_state=...)`
  reproduces it bit for bit), so `refute()` (placebo, random common cause, subset) and
  `diagnostics.truncation_curve` run on it. Their readings are descriptive: no registered study
  calibrates them yet. ([#671](https://github.com/esbraun/cleverly-tmle/pull/671))

### Point treatment

- `TMLE(conditional_form="known_score")` reports the ATT and ATC of the model in which the
  treatment mechanism is known, on data that declares `treatment_probabilities=`. Its influence
  curve (Chen, Hong and Tarozzi 2008) has a smaller variance than the default `"arm_mean"` form,
  which stays the default. The registered `known-score-att` study supports it (support rows T55).
  ([#653](https://github.com/esbraun/cleverly-tmle/pull/653))
- `mechanism_law="sample"` estimates the treatment mechanism of the weighted law from the
  declared probabilities and the weights, for point TMLE, `LTMLE`, `SDR`, IPSI and the g-formula's natural
  draws. The registered `known-mechanism-sample-weights` study supports it (support rows T56 and G64).
  ([#662](https://github.com/esbraun/cleverly-tmle/pull/662))

### Learned rules

- `OptimalRuleValue` accepts known or estimated weights, strata, known regimes beside the rule
  (with the contrast `ate_optimal_rule`), `cross_fit=False` and an intermediate held at a level.
  The registered `learned-rule-optimal-compositions` study supports each.
  The omitted-variable bound and `simulated_confounding` run on the optimal value.
  ([#665](https://github.com/esbraun/cleverly-tmle/pull/665))

- `LearnedRuleValue(intermediate=z)` and `OptimalRuleValue(intermediate=z)` estimate a learned
  rule's value at a fixed level of a declared intermediate on a `PointTreatment` design.
  ([#654](https://github.com/esbraun/cleverly-tmle/pull/654))
- A learned-rule fit accepts estimated observation weights, read conditionally on the weights as
  ordinary TMLE reads them. Such a fit reads experimental.
  ([#654](https://github.com/esbraun/cleverly-tmle/pull/654))

### Plots

- `overlap()` draws time-to-event fits of a baseline treatment, longitudinal fits per time point,
  structural nested mean fits and adaptive TMLE fits. `learners()` draws a survival C-TMLE fit's
  selection paths and a longitudinal fit's Super Learner weights by time point, and
  `calibration()` draws each longitudinal probability model per time point.
  ([#655](https://github.com/esbraun/cleverly-tmle/pull/655))
- `overlap()`, `weights()` and `learners()` draw a continuous-time survival fit.
  `overlap(censoring=True)` adds the product of the treatment and staying-uncensored
  probabilities before a chosen time. ([#663](https://github.com/esbraun/cleverly-tmle/pull/663))
- `influence()` draws a continuous-time survival fit, and `repeats()` draws a cross-fitted
  longitudinal fit with `repeats=2` or more.
  ([#649](https://github.com/esbraun/cleverly-tmle/pull/649))

## Changed

- Every plot in `cleverly.plotting` either draws a fit or refuses it with a `CapabilityError`
  that gives a true reason. `bootstrap()` on an object that is neither a fit nor replicates raises
  `TypeError`. ([#649](https://github.com/esbraun/cleverly-tmle/pull/649),
  [#650](https://github.com/esbraun/cleverly-tmle/pull/650))
- `overlap()` and `covariate_balance` refuse a g-computation or parametric g-formula fit, which
  estimates no treatment mechanism. ([#650](https://github.com/esbraun/cleverly-tmle/pull/650))
- `result.sensitivity` defers on the same estimand candidates as `marginal_sensitivity_bounds`,
  and the bounds record their `reference_df`.
  ([#656](https://github.com/esbraun/cleverly-tmle/pull/656))
- The `future_treatment` and `negative_control_outcome` refutations read their p-values from the
  control fit's own Wald test, so a clustered fit below 40 clusters uses its t reference and a
  ratio fit its interval scale. ([#668](https://github.com/esbraun/cleverly-tmle/pull/668))
- `marginal_sensitivity_bounds` refuses when the estimate it bounds withholds its own interval.
  ([#658](https://github.com/esbraun/cleverly-tmle/pull/658))
- `SparseInfluenceWarning` on an in-sample modified-treatment-policy fit points to
  `variance="policy_weighted"` when the fit accepts it.
  ([#645](https://github.com/esbraun/cleverly-tmle/pull/645))

## Fixed

- `influence()`, `weights()`, `repeats()`, `balance()`, `learners()`, `calibration()` and
  `curve()` raised `AttributeError` or `TypeError`, or refused with a false reason, on several
  shipped result types (survival, adaptive, cross-fitted longitudinal, dose-response, effect-curve
  and structural nested mean fits). ([#649](https://github.com/esbraun/cleverly-tmle/pull/649))
- The `truncation_curve` capability row of a `SurvivalDRTMLE` or `SurvivalCTMLE` fit, of a hazard
  fit with a modified treatment policy, and of a `LearnedRegimen` fit (also its `refute` row) read
  available while the call refused. Each row now carries the call's refusal.
  ([#652](https://github.com/esbraun/cleverly-tmle/pull/652),
  [#656](https://github.com/esbraun/cleverly-tmle/pull/656),
  [#666](https://github.com/esbraun/cleverly-tmle/pull/666))
- On a held design, a one-node rule, policy or modified treatment policy made `support()` and
  `node_weights()` raise `IndexError`. Later nodes now report `held`.
  ([#652](https://github.com/esbraun/cleverly-tmle/pull/652))
- Refutation called through the result facade with `future_treatment_control` or three other
  arguments failed validation. ([#671](https://github.com/esbraun/cleverly-tmle/pull/671))
- The E-value of a CATE or log-CRR projection fit refuses with its declared record also when the
  estimand is named. ([#666](https://github.com/esbraun/cleverly-tmle/pull/666))
- A binary outcome regression on a fold, arm or DR-TMLE leave-two-out pair with one observed value
  raised scikit-learn's two-class error, which happens at few clusters. It now predicts that value,
  and estimates that read a constant arm withhold their interval.
  ([#658](https://github.com/esbraun/cleverly-tmle/pull/658))
- `markov_order=` beside a node-1 block alone on the hazard path crashed in numpy. It now refuses
  pooled hazards and names `hazard_pooling="node"`.
  ([#659](https://github.com/esbraun/cleverly-tmle/pull/659))

## Documentation

- The technical reference adds an evidence page and validation-grid rows for each new registered
  study. ([#645](https://github.com/esbraun/cleverly-tmle/pull/645),
  [#646](https://github.com/esbraun/cleverly-tmle/pull/646),
  [#653](https://github.com/esbraun/cleverly-tmle/pull/653),
  [#659](https://github.com/esbraun/cleverly-tmle/pull/659),
  [#662](https://github.com/esbraun/cleverly-tmle/pull/662),
  [#664](https://github.com/esbraun/cleverly-tmle/pull/664),
  [#665](https://github.com/esbraun/cleverly-tmle/pull/665),
  [#667](https://github.com/esbraun/cleverly-tmle/pull/667),
  [#669](https://github.com/esbraun/cleverly-tmle/pull/669),
  [#670](https://github.com/esbraun/cleverly-tmle/pull/670))
- The roadmap drops the items that were not TMLE estimators.
  ([#647](https://github.com/esbraun/cleverly-tmle/pull/647))
- The contributor guide asks every new estimator for a capability-sweep fit and a plot-matrix
  entry. ([#648](https://github.com/esbraun/cleverly-tmle/pull/648),
  [#651](https://github.com/esbraun/cleverly-tmle/pull/651))

**Full changelog:** [v0.2.5...v0.2.6](https://github.com/esbraun/cleverly-tmle/compare/v0.2.5...v0.2.6)
