# cleverly 0.1.6

This alpha release gates refutation verdicts behind registered studies and extends the sensitivity
analyses. It adds adaptive-propensity IPTW, four `LTMLE` options, and refutation steps in the
tutorial notebooks. These are the changes since 0.1.5.

## Upgrade notes

- Read `RefutationResult.passed` and `RefutationTest.passed` as `True`, `False` or `None`.
  `None` means that no registered study covers the fit, so the test is descriptive.
  `bool(report)` now raises `TypeError` on a report with a descriptive test.
  Read `report.tests`, and each test's `rung` and `scope_sentence`, instead.
  ([#385](https://github.com/esbraun/cleverly-tmle/pull/385))
- Remove `tolerance=` and `outcome_rule=` from `refute()` calls. Both keywords are removed.
  Pass `outcome_bands=EquivalenceBands(...)` to set the bands of the generated-outcome readings.
  ([#385](https://github.com/esbraun/cleverly-tmle/pull/385))
- Pass `n_replicates=` to keep a smaller `refute()` budget. `placebo`, `subset` and
  `random_common_cause` now default to 20 refits each, and in 0.1.5 they used 5.
  Each generated-outcome test now defaults to 400 draws, and in 0.1.5 it used 100.
  A generated-outcome reading below 400 draws prints at rung `L1` and states no verdict.
  ([#385](https://github.com/esbraun/cleverly-tmle/pull/385),
  [#386](https://github.com/esbraun/cleverly-tmle/pull/386))

Refit analyses when you move saved results to 0.1.6. Loading another version's result emits
`VersionMismatchWarning`. The loader does not migrate the result, and the load can fail.

## Added

### Refutation verdicts behind registered evidence

- `refute()` prints a verdict only where a row of the evidence registry covers the fit.
  Each row states a rung from `L0` to `L3` and a calibration scope over the fit's configuration.
  A descriptive test names the nearest registered row and each axis on which the fit differs.
  ([#385](https://github.com/esbraun/cleverly-tmle/pull/385),
  [#445](https://github.com/esbraun/cleverly-tmle/pull/445))
- The placebo reads a one-sample t statistic on R - 1 degrees of freedom against the exact null.
  The subset test reads (R - 1) rho^2 against a chi-square law, with a reference SD that
  scales with the subset fraction.
  ([#385](https://github.com/esbraun/cleverly-tmle/pull/385))

The registered `refuter-calibration-point` study licenses 12 rows. They cover `TMLE` with `GLM`
and boosted nuisances, greedy `CTMLE`, and `DRTMLE`, at n = 2,000 with five folds.
`refuter-calibration-longitudinal` licenses 20 rows for `LTMLE`, `SDR`, `IPSI`, the sequential
baselines, `ParametricGFormula` and `StructuralNestedMean` at n = 1,000 with `GLM` nuisances.
`refuter-calibration-band` licenses two band-rank placebo rows, scoped to the study's own learners.
A fit outside these scopes prints descriptive readings. No comparator exists for these rules.
([#409](https://github.com/esbraun/cleverly-tmle/pull/409),
[#415](https://github.com/esbraun/cleverly-tmle/pull/415),
[#431](https://github.com/esbraun/cleverly-tmle/pull/431),
[#450](https://github.com/esbraun/cleverly-tmle/pull/450))

### Point-treatment refutation tests

- `tests=("plasmode",)` resamples the covariates, draws treatment from a frozen treatment model,
  and draws the outcome from the declared `simulated_outcome` process. The truth is an exact finite sum.
  A clustered fit resamples whole clusters. Three exact registry rows read bias, SE ratio and coverage.
  ([#389](https://github.com/esbraun/cleverly-tmle/pull/389),
  [#406](https://github.com/esbraun/cleverly-tmle/pull/406))
- `tests=("bootstrap_stability",)` refits bootstrap copies with no added noise.
  Each copy keeps its unit's fold, and each refit reuses the fit's seeds. The report is descriptive.
  ([#388](https://github.com/esbraun/cleverly-tmle/pull/388))
- `BernoulliAdjustmentOutcome`, `BoundedAdjustmentOutcome` and `arm_effects=` extend the generated
  outcomes beyond the Gaussian binary-treatment ATE. The bounded process draws a Beta law on the fit's `q_bounds`.
  Truths cover `ate`, `att` and `atc` for every arm pair, arm levels, `rr` and `or`.
  ([#402](https://github.com/esbraun/cleverly-tmle/pull/402))
- Generated outcomes and the plasmode also run on fits with missing outcomes, a missing treatment,
  an intermediate (`intermediate_effect=`), strata and clusters. They also read `ey_obs`, `par` and `paf`.
  ([#406](https://github.com/esbraun/cleverly-tmle/pull/406))
- Generated truths now cover regime means and contrasts, MSM coefficients, and modified treatment policies.
  Incremental tilts read through the plasmode only. A continuous-dose plasmode is refused.
  ([#411](https://github.com/esbraun/cleverly-tmle/pull/411),
  [#413](https://github.com/esbraun/cleverly-tmle/pull/413))
- The placebo now runs on an identity-link point MSM. A coefficient whose projection row
  vanishes on every unit reads against an exact 0. Other coefficients print descriptively.
  ([#388](https://github.com/esbraun/cleverly-tmle/pull/388))
- The placebo runs on a learned-rule fit against the weighted outcome mean.
  `random_common_cause`, `subset` and `bootstrap_stability` hold each realized fold rule fixed.
  ([#388](https://github.com/esbraun/cleverly-tmle/pull/388),
  [#414](https://github.com/esbraun/cleverly-tmle/pull/414))
- The placebo runs on a dose whose declared support depends on the covariates or the history.
  It permutes each dose's position within its band and maps it into the receiving row's band.
  ([#446](https://github.com/esbraun/cleverly-tmle/pull/446))
- `negative_control_kind="pre_treatment"` declares a control measured before treatment.
  The test refuses a control equal to an adjustment column. It also refuses a control below
  `negative_control_floor` of cross-fitted residual variance.
  ([#397](https://github.com/esbraun/cleverly-tmle/pull/397))

### Longitudinal and structural refutation

- `refute()` runs on every longitudinal fit, including survival, competing-risk, repeated-outcome and MSM fits.
  The placebo draws each treatment trajectory from a first-order Markov kernel (`placebo_mode="markov"`).
  `random_common_cause(noise_nodes=True)` adds one noise column at every node.
  ([#391](https://github.com/esbraun/cleverly-tmle/pull/391))
- `negative_control_outcome` runs on end-of-study and repeated panels.
  The new `future_treatment` test regresses a declared control on a later treatment, given that treatment's history.
  ([#397](https://github.com/esbraun/cleverly-tmle/pull/397))
- Generated outcomes run on Gaussian end-of-study and repeated panels.
  `LogisticHazardOutcome` and `MultinomialHazardOutcome` generate events for survival and competing-risk fits.
  ([#397](https://github.com/esbraun/cleverly-tmle/pull/397))
- The longitudinal plasmode covers end-of-study, survival, competing-risk and repeated-outcome panels.
  Its truth is the g-formula without censoring. `history_effect=` lets the outcome or hazard read the time-varying covariates.
  ([#416](https://github.com/esbraun/cleverly-tmle/pull/416),
  [#419](https://github.com/esbraun/cleverly-tmle/pull/419),
  [#429](https://github.com/esbraun/cleverly-tmle/pull/429))
- `bootstrap_measurement_error` runs on longitudinal results, with a noise scale per node.
  On point fits, strata variables keep their recorded values.
  ([#407](https://github.com/esbraun/cleverly-tmle/pull/407))
- `StructuralNestedMean` results get the `"structural"` assessment family and a refute adapter.
  The placebo, subset, `random_common_cause`, `bootstrap_stability` and negative-control tests
  refit the structural model.
  ([#401](https://github.com/esbraun/cleverly-tmle/pull/401))

### Marginal sensitivity model bounds

- `cleverly.sensitivity.marginal_sensitivity_bounds()` and `result.sensitivity.marginal_bounds()`
  bound arm means and the ATE under Tan's (2006) marginal sensitivity model over a grid of gamma.
  The cross-fitted estimator follows the replication code of Dorn, Guo and Kallus.
  Gamma = 1 returns the cross-fitted AIPW estimate.
  ([#425](https://github.com/esbraun/cleverly-tmle/pull/425),
  [#430](https://github.com/esbraun/cleverly-tmle/pull/430))
- The bounds also cover more than two arms, `att`, `atc`, `rr`, `or`, sampling weights, strata,
  missing outcomes and selecting `CTMLE` fits. Per-arm outcome-adaptive `CTMLE` is refused.
  ([#447](https://github.com/esbraun/cleverly-tmle/pull/447))

Every confidence-limit reading of these bounds is descriptive. The study
`marginal-bound-limit-coverage` is declared but has not run, and no registry row exists.
([#427](https://github.com/esbraun/cleverly-tmle/pull/427),
[#441](https://github.com/esbraun/cleverly-tmle/pull/441),
[#442](https://github.com/esbraun/cleverly-tmle/pull/442))

### Omitted-variable bounds

- `omitted_variable_bounds()` now answers a response mechanism, a declared missing treatment,
  an intermediate variable, and a baseline-stratum parameter. `BenchmarkResult` carries the joint-strength features.
  ([#398](https://github.com/esbraun/cleverly-tmle/pull/398),
  [#400](https://github.com/esbraun/cleverly-tmle/pull/400))
- The bound now answers fits with `repeats > 1`. Bounds, limits and robustness values combine
  the draws by the median rule. `sensitivity_elements(repeat=...)` reads one draw.
  ([#404](https://github.com/esbraun/cleverly-tmle/pull/404))
- The bound covers regime, shift and identity-link MSM parameters of a categorical treatment.
  ([#412](https://github.com/esbraun/cleverly-tmle/pull/412))
- `assume_consistent_mechanism=True` gives a point bound on `DRTMLE` and selecting `CTMLE` fits.
  `mechanism_learner=` declares the mechanism refit, and a per-arm outcome-adaptive fit requires it.
  The confidence limits refuse. The result facade accepts both keywords.
  ([#408](https://github.com/esbraun/cleverly-tmle/pull/408),
  [#453](https://github.com/esbraun/cleverly-tmle/pull/453))

The registered `ovb-limit-coverage` study licenses limit rows for the response, intermediate,
regime, MSM and shift parameters. Its rows need n of at least 2,000, an in-sample fit and the study learners.
Coverage across its 20 rows is 0.9331 to 0.961. No maintained implementation is a comparator.
([#417](https://github.com/esbraun/cleverly-tmle/pull/417),
[#422](https://github.com/esbraun/cleverly-tmle/pull/422),
[#433](https://github.com/esbraun/cleverly-tmle/pull/433),
[#439](https://github.com/esbraun/cleverly-tmle/pull/439))

### E-value

- `evalue()` now answers a controlled direct effect, with one bounding factor per arm
  (Ding and VanderWeele 2016). It also answers stratum contrasts and response-mechanism fits.
  ([#392](https://github.com/esbraun/cleverly-tmle/pull/392))
- `evalue()` derives a risk ratio for `CTMLE`, `DRTMLE` and controlled-direct-effect fits.
  `evalue(result, (numerator, denominator))` reads the risk ratio of two static regimens of a time-to-event fit.
  ([#392](https://github.com/esbraun/cleverly-tmle/pull/392))

The registered `evalue-limit-coverage` study licenses limit rows for stratum risk ratios and the
time-to-event ratio, with coverage 0.9469 to 0.9619. Other new limits print descriptively.
([#417](https://github.com/esbraun/cleverly-tmle/pull/417),
[#422](https://github.com/esbraun/cleverly-tmle/pull/422))

### Simulated confounding

- `simulated_confounding()` admits multi-arm and vector treatments through `arm_pair=`,
  clustered fits through `latent_level=` and `latent_rho=`, categorical benchmarks, and estimated weights.
  ([#393](https://github.com/esbraun/cleverly-tmle/pull/393))
- It admits missing-data fits through `observation_reads=`, controlled direct effects through an
  `intermediate` grid axis, and `q_bounds` fits through `bounded_outcome_law=`.
  ([#396](https://github.com/esbraun/cleverly-tmle/pull/396))
- It runs one time-fixed latent on uncensored end-of-study `LTMLE`, `SDR`, sequential baseline
  and `ParametricGFormula` fits. A time-varying covariate after node 1 is refused.
  ([#399](https://github.com/esbraun/cleverly-tmle/pull/399))
- A point estimator replays when its class declares `simulated_confounding_replay = True`.
  The MSM replay accepts any registered link, and a fixed regime accepts any known-density intervention.
  ([#410](https://github.com/esbraun/cleverly-tmle/pull/410))
- Incremental targets replay on multi-arm fits. A learned-rule fit replays with each realized fold rule held fixed.
  ([#414](https://github.com/esbraun/cleverly-tmle/pull/414))

Every simulated-confounding output is descriptive and makes no interval claim.

### Missingness tilt

- `missingness_tilt()` and `tipping_gamma()` now re-mix `rr`, `or`, `ey_obs`, `par`, `paf`,
  the natural course, regime parameters and MSM coefficients. `tipping_gamma(use_ci=True)` refuses these kinds.
  ([#394](https://github.com/esbraun/cleverly-tmle/pull/394))

### Adaptive-propensity IPTW

- `cleverly.estimators.AdaptiveIPTW` and `AdaptiveIPTWMethod` (study method `"adaptive_iptw"`)
  report Horvitz-Thompson arm means with a propensity targeted along the reduced outcome regression
  (van der Laan 2012, WP302). The estimator fits no outcome regression.
  ([#395](https://github.com/esbraun/cleverly-tmle/pull/395))
- An interval requires a known treatment mechanism. An estimated mechanism gives point estimates
  under the status `estimated_mechanism_plugin`. Cross-fitting requires at least three folds.
  ([#395](https://github.com/esbraun/cleverly-tmle/pull/395))
- `refute()` and `simulated_confounding()` run on `AdaptiveIPTW` fits. A fit without an interval
  keeps only the bias reading of a generated-outcome test.
  ([#403](https://github.com/esbraun/cleverly-tmle/pull/403))

The registered `adaptive-iptw` study supplies Python property evidence. 102 of 122 property cells
pass, and the other 20 are red under the reporting policy, each with an owner.
The `drtmle` 1.1.2 comparator runs are pending.
([#395](https://github.com/esbraun/cleverly-tmle/pull/395))

### LTMLE options

- `LTMLE(ratio_trim=q)` and `SDR(ratio_trim=q)` trim per-node density ratios at a pooled quantile, as `lmtp` does.
  A fit whose trim moves a ratio reports no interval, under the status `trimmed_ratio_plugin`.
  `survival_model="hazard"` refuses the option.
  ([#380](https://github.com/esbraun/cleverly-tmle/pull/380))
- `LongitudinalTreatment(markov_order=k)` restricts every nuisance design to `W` and a window of recent nodes.
  The estimand does not change. `ParametricGFormula` and `StructuralNestedMean` refuse it.
  ([#380](https://github.com/esbraun/cleverly-tmle/pull/380))
- `hazard_pooling="pooled"` fits one censoring model over every node. Under
  `survival_model="hazard"` it also pools each cause's hazard. `ModelSpec` carries the option.
  ([#380](https://github.com/esbraun/cleverly-tmle/pull/380))
- `LTMLE(node_bounds=...)` bounds the iterated mean or the cause-specific hazard at named nodes.
  The fluctuation runs on the bounded logit scale. A bound that excludes the data records `bound_excludes_data`.
  ([#380](https://github.com/esbraun/cleverly-tmle/pull/380))
- The hazard targeting now accepts `stopping=` and `iterate=`, and `step_bound=` under `targeting="iterative"`.
  ([#380](https://github.com/esbraun/cleverly-tmle/pull/380))

The registered `survival-node-bounds` study supplies Python property evidence for bounded mean and
hazard models. All 15 gated cells pass; the `survtmle` comparator runs are pending.
A fit with `ratio_trim=` or `markov_order=` carries the `experimental` support label.
([#380](https://github.com/esbraun/cleverly-tmle/pull/380))

## Changed

- `subset` refits keep the original fold labels, so a fit with `split_plan=` no longer refuses it.
  `random_common_cause` pairs each noise refit with the original fit's seed.
  `seed_spread=True` adds fresh-seed baselines.
  ([#385](https://github.com/esbraun/cleverly-tmle/pull/385))
- `bootstrap_measurement_error` refits keep each bootstrap copy in its original unit's fold.
  ([#386](https://github.com/esbraun/cleverly-tmle/pull/386))
- An unseeded `StructuralNestedMean` fit draws one `fit_seed` and records it in `result.config.fit_seed`.
  A refit with that seed reproduces the fit.
  ([#401](https://github.com/esbraun/cleverly-tmle/pull/401))
- The default assessment defers the E-value of a response-mechanism fit, because that E-value refits a plug-in of sd(Y).
  ([#392](https://github.com/esbraun/cleverly-tmle/pull/392))
- Refusals of the sensitivity and refutation rows that X82 absorbed now cite roadmap row `X82`.
  ([#390](https://github.com/esbraun/cleverly-tmle/pull/390))

### Validation evidence

- The C-TMLE candidate-sequence and C-TMLE0 derivative studies now register R `ctmle` as the comparator.
  ([#376](https://github.com/esbraun/cleverly-tmle/pull/376),
  [#378](https://github.com/esbraun/cleverly-tmle/pull/378))
- The `mtp-boundaries` and `mtp-pricing` studies now register `lmtp` 1.5.4 runs.
  Each study gates every witness estimand at 1e-6 agreement in the estimate and the SE.
  ([#377](https://github.com/esbraun/cleverly-tmle/pull/377))
- The repeated-outcome `SDR` study now registers `lmtp` 1.5.4 `cf_sdr` as the comparator.
  ([#382](https://github.com/esbraun/cleverly-tmle/pull/382))
- The `structural-nested-mean` study now registers EconML 0.17.0 `DynamicDML` as the comparator.
  All 21 paired rows pass. ([#383](https://github.com/esbraun/cleverly-tmle/pull/383),
  [#384](https://github.com/esbraun/cleverly-tmle/pull/384))

### Documentation

- The tutorial notebooks show the refutation and sensitivity checks that fit each design.
  Each notebook states in one sentence why the other checks do not apply.
  ([#423](https://github.com/esbraun/cleverly-tmle/pull/423),
  [#424](https://github.com/esbraun/cleverly-tmle/pull/424),
  [#426](https://github.com/esbraun/cleverly-tmle/pull/426),
  [#428](https://github.com/esbraun/cleverly-tmle/pull/428),
  [#435](https://github.com/esbraun/cleverly-tmle/pull/435),
  [#436](https://github.com/esbraun/cleverly-tmle/pull/436),
  [#437](https://github.com/esbraun/cleverly-tmle/pull/437),
  [#438](https://github.com/esbraun/cleverly-tmle/pull/438),
  [#440](https://github.com/esbraun/cleverly-tmle/pull/440),
  [#444](https://github.com/esbraun/cleverly-tmle/pull/444),
  [#448](https://github.com/esbraun/cleverly-tmle/pull/448),
  [#449](https://github.com/esbraun/cleverly-tmle/pull/449),
  [#454](https://github.com/esbraun/cleverly-tmle/pull/454),
  [#456](https://github.com/esbraun/cleverly-tmle/pull/456))
- The MSM projections tutorial reads the robustness value of the slope.
  ([#412](https://github.com/esbraun/cleverly-tmle/pull/412))
- The point-treatment example is rewritten in plain language for a reader new to causal inference.
  It gives the answer first and moves the extended checks to a closing section.
  ([#455](https://github.com/esbraun/cleverly-tmle/pull/455),
  [#458](https://github.com/esbraun/cleverly-tmle/pull/458))
- The survey-nonresponse, sequential-offers, sequential-pricing and cross-fitting examples
  are rewritten in plain language in the same way.
  ([#459](https://github.com/esbraun/cleverly-tmle/pull/459),
  [#460](https://github.com/esbraun/cleverly-tmle/pull/460),
  [#461](https://github.com/esbraun/cleverly-tmle/pull/461),
  [#463](https://github.com/esbraun/cleverly-tmle/pull/463))
- The TWINS example now estimates the ATT of smoking during pregnancy on the pair-mean birth weight.
  It replaces the first-year mortality outcome and adds the refutation and sensitivity checks.
  It treats gestational age as a mediator and states the selection on two live births.
  ([#457](https://github.com/esbraun/cleverly-tmle/pull/457))
- `docs/development/writing-for-readers.md` names the reader of each documentation area and the writing rules.
  ([#452](https://github.com/esbraun/cleverly-tmle/pull/452))

## Fixed

- `refute(tests=("placebo",))` no longer stops with `DataError` partway through its replicates
  on a dose whose declared support reads the covariates or the history. It uses the band-rank draw.
  ([#428](https://github.com/esbraun/cleverly-tmle/pull/428),
  [#446](https://github.com/esbraun/cleverly-tmle/pull/446))

## Packaging and compatibility

This remains an alpha release. The package does not promise compatibility for public APIs,
`CausalStudy` objects, pickled objects, or saved results. Follow the upgrade notes when you move
an analysis to 0.1.6.

**Full changelog:** [v0.1.5...v0.1.6](https://github.com/esbraun/cleverly-tmle/compare/v0.1.5...v0.1.6)
