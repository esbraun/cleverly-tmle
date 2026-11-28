# cleverly 0.2.9

This release adds the continuous-time longitudinal TMLE, typed learned-rule contrasts, restricted
mean survival and the dose-response curve on the study route, and an outcome-based covariate
screen. It corrects the binned density's tail and narrows several support labels to what their
studies measured. Most of the documentation was rewritten.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.
- `screen_treatment=True` now screens covariates on the outcome, with the treatment held fixed, as
  R's `prescreenW.g` does. The old screen tested correlation with the treatment and, at the default
  `min_retain=None`, kept every covariate. `ModelSpec(screen_threshold=)`, `CorrelationScreener`
  and `screen_by_correlation` are removed; `cleverly.learners.screen_by_outcome` is the new
  function. `min_retain=None` now keeps `min(ceil(n_eff/20), p)` covariates.
  ([#772](https://github.com/esbraun/cleverly-tmle/pull/772))
- `CTMLE(ctmle_estimand=)` is renamed `selection_estimand`.
  ([#764](https://github.com/esbraun/cleverly-tmle/pull/764))
- Settings that a fit used to accept and ignore now raise a typed refusal before any learner
  runs: `pseudo_learner`, `censoring_learner` and `competing_learner` on a point-treatment fit
  ([#774](https://github.com/esbraun/cleverly-tmle/pull/774)); `variance=` on the sequential
  baselines, `max_memory_bytes` on `SurvivalCTMLE` and `SurvivalDRTMLE`, and
  `bootstrap_interval="bca"` on the non-TMLE baselines, which report the percentile interval
  ([#762](https://github.com/esbraun/cleverly-tmle/pull/762)).
- `IPSI(variance="policy_weighted")` is refused. It reported a variance without IPSI's
  estimated-mechanism term. ([#762](https://github.com/esbraun/cleverly-tmle/pull/762))
- `TrialWithExternalData` refuses `treatment_probabilities`, `weights`, `cluster`, `strata`,
  `intermediate` and `treatment_missingness` with typed records. For a known trial probability, fit
  the trial alone with `PointTreatment(treatment_probabilities=...)`.
  ([#764](https://github.com/esbraun/cleverly-tmle/pull/764),
  [#769](https://github.com/esbraun/cleverly-tmle/pull/769))
- A `refute(..., checkpoint=...)` directory written by an earlier release for a continuous-time
  survival fit does not resume, because the checkpoint key now includes the data fingerprint.
  ([#879](https://github.com/esbraun/cleverly-tmle/pull/879))

## Added

### Survival and longitudinal

- `ContinuousLongitudinalTMLE` estimates the risk of each cause under a treatment plan applied at
  baseline and at irregular visits, in continuous time (Rytgaard, Gerds and van der Laan 2022).
  Its data container is `ContinuousLongitudinalData`, and `CoxHazard` gains `entry=` for delayed
  entry. A risk or risk difference reads `supported` for a cross-fitted fit with the default
  libraries and settings, two causes and at least 1,000 units, inside the event shares, follower
  counts and visit-cap shares that the registered `continuous-time-longitudinal` study measured.
  Ratios, other contrasts and the band read `experimental`.
  ([#863](https://github.com/esbraun/cleverly-tmle/pull/863),
  [#871](https://github.com/esbraun/cleverly-tmle/pull/871))
- `RegimeMean` and `RegimeContrast`, and the `LTMLE` and `SDR` engines, take
  `restricted_horizons=` for the restricted mean survival time and the restricted mean time lost.
  On a binned grid each is a fixed linear sum of the targeted risks, with the matching influence
  curve; on the exact-time route the field reaches `ContinuousSurvivalTMLE`. The difference reads
  `supported` on rows G11.nodes4, G12.nodes4 and G14.survival, and the levels on G11.nodes4 and
  G12.nodes4. `SurvivalDRTMLE` refuses it.
  ([#776](https://github.com/esbraun/cleverly-tmle/pull/776))
- `ContinuousSurvivalTMLE` results carry a provenance record, the study route stamps the
  protocol's fingerprint on it, and the engine takes `run_id=`.
  ([#879](https://github.com/esbraun/cleverly-tmle/pull/879))
- In-sample `ContinuousSurvivalTMLE` fits with the default models read `supported` from 1,000
  units for a risk or risk difference with two causes and no restricted horizon, from the
  registered `continuous-survival-variance` study's in-sample cell.
  ([#751](https://github.com/esbraun/cleverly-tmle/pull/751))

### Study route

- `LearnedRuleContrast` and `OptimalRuleContrast` compare a learned or optimal rule against a
  static treatment, a stochastic plan, or `NaturalCourse`. The registered
  `learned-rule-contrast-comparators` study supports the fold-evaluated and stacked fits of each
  comparator. ([#765](https://github.com/esbraun/cleverly-tmle/pull/765))
- `DoseResponseCurve(curve)` fits the smoothed dose-response curve through `CausalStudy`. The
  route fits the same `DoseResponseTMLE` as a direct fit, bit for bit, and refuses by name the
  method fields the curve does not read. The result gains `identified_effect`, `method`,
  `parameter_keys`, provenance and `summary()`.
  ([#763](https://github.com/esbraun/cleverly-tmle/pull/763))
- Engine options reach the method objects: `Inference.bootstrap_interval`, `Inference.variance`,
  `Inference.cluster_variance`, `Runtime.max_memory_bytes`, `ModelSpec.hazard_library`,
  `ModelSpec.censoring_library` and `Targeting.norm`, each with a flat shortcut of the same name.
  An engine that does not take a field refuses it before any learner.
  ([#762](https://github.com/esbraun/cleverly-tmle/pull/762))
- `variable_importance` and `variable_importance_plan` take `protocol=`, which the ranking keeps
  and each candidate fit stamps in its provenance.
  ([#860](https://github.com/esbraun/cleverly-tmle/pull/860))

### Effect modification

- DR-TMLE fits the projection of a conditional contrast on the difference scale, and the per-arm
  outcome-adaptive C-TMLE (`CTMLE(strategy="oat", oat_design="per_arm")`) fits its points. The
  registered `projection-methods` study supports DR-TMLE intervals with a declared mechanism
  (`T49.drtmle.known`), and DR-TMLE and C-TMLE points (`T49.drtmle`, `T49.ctmle_oat`). With an
  estimated mechanism the DR-TMLE interval is not published (`reduced_regression_plugin`).
  ([#757](https://github.com/esbraun/cleverly-tmle/pull/757))
- `refute` runs on a projection fit, the truncation curve reads the producing fit's, and the
  omitted-variable bound covers a difference-scale coefficient.
  ([#757](https://github.com/esbraun/cleverly-tmle/pull/757))
- A learned-rule value whose EP rule comes from a declared sieve or a declared final learner reads
  `supported` (row T51), and the optimal value with a declared sieve reads `supported`
  (T51.optimal), from the registered `learned-rule-ep-declared` study.
  ([#750](https://github.com/esbraun/cleverly-tmle/pull/750))

### Plots

- `cleverly.plotting.confounding(evalue)` and `EValue.plot()` draw the E-value's curve of
  confounder strengths that could explain the estimate away.
  ([#761](https://github.com/esbraun/cleverly-tmle/pull/761))
- `plotting.curve` draws a categorical incremental family that moves one level on a log-odds axis
  instead of a forest. ([#820](https://github.com/esbraun/cleverly-tmle/pull/820))

### Simulated data

- `cleverly.datasets` adds generators with known truths: `make_exact_readmission`,
  `make_home_support`, `make_discharge_practices`, `make_two_decision_rules`, and the truth helpers
  `competing_rule_truth`, `treatment_rule_effect`, `recovery_offer_count` and
  `learned_regimen_truth`.
  ([#846](https://github.com/esbraun/cleverly-tmle/pull/846),
  [#855](https://github.com/esbraun/cleverly-tmle/pull/855),
  [#856](https://github.com/esbraun/cleverly-tmle/pull/856),
  [#861](https://github.com/esbraun/cleverly-tmle/pull/861),
  [#879](https://github.com/esbraun/cleverly-tmle/pull/879))

## Changed

- The binned conditional density uses equal-mass core bins refined in the tails, with gap bins
  over long empty runs. On an unbounded dose the old top bin read the density ratio as 1 and
  widened with n, so the density route was not doubly robust there. `ratio=None` now resolves per
  fit, to the classifier route where it applies and to the density route otherwise, and the
  result records the route. A density fit takes about 3 times as long. The registered
  `mtp-density-tails` study passes, and every study the change could move was regenerated.
  ([#748](https://github.com/esbraun/cleverly-tmle/pull/748))
- A classifier-route fit warns (`DataWarning`) on an isolated cluster of doses beyond a gap bin,
  and `cleverly.learners.isolated_dose_clusters` lists them.
  ([#748](https://github.com/esbraun/cleverly-tmle/pull/748))
- `DoseResponseTMLE` and the binned density peak at about a third of the memory (0.75 GiB against
  2.37 GiB on a 100,000-row fit with 10 pairs), with bitwise identical results at one BLAS thread.
  `DoseResponseTMLE` takes `max_memory_bytes=`.
  ([#794](https://github.com/esbraun/cleverly-tmle/pull/794))
- A study fit's sparse-influence warning names a flagged row the result does not report, and the
  reported contrasts that read it. ([#846](https://github.com/esbraun/cleverly-tmle/pull/846))
- A C-TMLE fit whose strategy does not read a declared `treatment_learner` (`"lasso"`, or
  `"derivative"` without `treatment_candidates=`) warns. Estimates do not change.
  ([#830](https://github.com/esbraun/cleverly-tmle/pull/830))
- The overlap summary's weight-concentration share divides by the rows that carry weight in the
  equation, not by every row of the score mask.
  ([#835](https://github.com/esbraun/cleverly-tmle/pull/835))
- The in-sample calibration note says the slope is 1 by construction only for an unpenalized,
  unweighted two-class logistic regression with an intercept.
  ([#851](https://github.com/esbraun/cleverly-tmle/pull/851))

### Support labels

These labels now read `experimental` because no registered study measured the fit:

- a point modified-treatment-policy fit whose dose density is learned in sample; a cross-fitted
  fit with the measured booster on 40 bins reads `supported` from the registered
  `point-shift-learned-density` study
  ([#853](https://github.com/esbraun/cleverly-tmle/pull/853),
  [#878](https://github.com/esbraun/cleverly-tmle/pull/878));
- a continuous-outcome three-arm fit and a known-mechanism continuous fit, whose rows licensed
  Wald support from a study of the omitted-variable limits
  ([#851](https://github.com/esbraun/cleverly-tmle/pull/851));
- a cross-fitted few-cluster fit with an adaptive learner other than the default library, or with
  fewer clusters than the registered `few-cluster-adaptive-learners` study measured
  ([#749](https://github.com/esbraun/cleverly-tmle/pull/749));
- a dose-response fit with `treatment_adjustment=`
  ([#763](https://github.com/esbraun/cleverly-tmle/pull/763));
- a parametric g-formula fit of a known rule on a law whose study cells hold no rule; rules on the
  G1, G2, G4 and G5 laws read `supported`
  ([#848](https://github.com/esbraun/cleverly-tmle/pull/848),
  [#869](https://github.com/esbraun/cleverly-tmle/pull/869));
- a study effect's bootstrap band narrowed to the rows it reports
  ([#848](https://github.com/esbraun/cleverly-tmle/pull/848)).

A declared treatment or retention mechanism now keys its support row by role and node, so the
label no longer depends on column names
([#804](https://github.com/esbraun/cleverly-tmle/pull/804)).

## Fixed

- `screen_treatment=True` was a no-op at the default `min_retain=None`, and it screened on the
  treatment. ([#772](https://github.com/esbraun/cleverly-tmle/pull/772))
- `MSM.linear` on an all-categorical vector treatment raised an unrelated `DataError`; it now
  refuses at `identify` and names the alternatives. The 20-level refusal on a vector treatment no
  longer suggests `treatment_kind='continuous'`.
  ([#784](https://github.com/esbraun/cleverly-tmle/pull/784))
- A study effect's studentized bootstrap band covered every engine row, not only the reported ones.
  ([#848](https://github.com/esbraun/cleverly-tmle/pull/848))
- `curve.plot()` raised on a DR-TMLE projection with an estimated mechanism; it draws the estimate
  without an interval. `refute()` without `estimand=` on a projection fit named the wrong estimand.
  ([#806](https://github.com/esbraun/cleverly-tmle/pull/806))
- The held-out curve checks raised `DataError` on a modifier stored as text.
  ([#861](https://github.com/esbraun/cleverly-tmle/pull/861))
- A truncation curve of a policy fit refuses where no ratio reads a `g_bounds` bound, since every
  point would equal the fit. ([#764](https://github.com/esbraun/cleverly-tmle/pull/764))
- `IdentifiedEffect.available_methods()` listed methods that refuse a `Separable` plan, and a
  separable plan now prints its two extra identification conditions.
  ([#755](https://github.com/esbraun/cleverly-tmle/pull/755))
- `curve.plot()` called every band experimental; it reads the band's own label. A curve's refute
  row names one of its own coefficients. ([#793](https://github.com/esbraun/cleverly-tmle/pull/793))
- `default_library`'s lasso no longer prints scikit-learn 1.9 `FutureWarning`s, and the mechanism
  description records its L1 ratio. Predictions are unchanged.
  ([#862](https://github.com/esbraun/cleverly-tmle/pull/862))
- G-computation's diagnostic treatment model warning names the cause and remedy on the classifier
  route (a penalized `treatment_learner`). Estimates do not change.
  ([#836](https://github.com/esbraun/cleverly-tmle/pull/836))
- DR-TMLE refuses a `ParametricOutcome` in a reduced-regression slot with a message naming the
  slot, where it raised a bare `ValueError`.
  ([#830](https://github.com/esbraun/cleverly-tmle/pull/830))
- `assess()` names an adaptive TMLE fit's updates and models correctly, and
  `AdaptiveTMLEResult.summary(detail="full")` prints the study description.
  ([#843](https://github.com/esbraun/cleverly-tmle/pull/843))
- The balance notes name declared probabilities as declared.
  ([#827](https://github.com/esbraun/cleverly-tmle/pull/827))

## Documentation

- The user guide is organized by analysis stage and topic, with new pages on settings,
  confidence intervals and bands, diagnostics, sensitivity analysis, refutation, plots, run time
  and memory, ranking candidate exposures, learned rules, dose shifts, plans over time and
  exact-time and competing-event survival. The quickstart and workflow pages were rewritten for a
  reader new to causal inference.
  ([#758](https://github.com/esbraun/cleverly-tmle/pull/758),
  [#767](https://github.com/esbraun/cleverly-tmle/pull/767),
  [#770](https://github.com/esbraun/cleverly-tmle/pull/770),
  [#774](https://github.com/esbraun/cleverly-tmle/pull/774),
  [#780](https://github.com/esbraun/cleverly-tmle/pull/780),
  [#787](https://github.com/esbraun/cleverly-tmle/pull/787),
  [#814](https://github.com/esbraun/cleverly-tmle/pull/814),
  [#817](https://github.com/esbraun/cleverly-tmle/pull/817),
  [#833](https://github.com/esbraun/cleverly-tmle/pull/833),
  [#873](https://github.com/esbraun/cleverly-tmle/pull/873))
- The technical reference adds pages on nuisance learners and the continuous-time longitudinal
  TMLE, groups the method-evidence pages by family, and corrects the method and shared pages
  against the code. ([#795](https://github.com/esbraun/cleverly-tmle/pull/795),
  [#807](https://github.com/esbraun/cleverly-tmle/pull/807),
  [#808](https://github.com/esbraun/cleverly-tmle/pull/808),
  [#809](https://github.com/esbraun/cleverly-tmle/pull/809),
  [#810](https://github.com/esbraun/cleverly-tmle/pull/810),
  [#815](https://github.com/esbraun/cleverly-tmle/pull/815),
  [#824](https://github.com/esbraun/cleverly-tmle/pull/824),
  [#825](https://github.com/esbraun/cleverly-tmle/pull/825))
- New registered studies have evidence pages and validation-grid rows. They are Python simulation
  evidence. ([#748](https://github.com/esbraun/cleverly-tmle/pull/748),
  [#749](https://github.com/esbraun/cleverly-tmle/pull/749),
  [#750](https://github.com/esbraun/cleverly-tmle/pull/750),
  [#757](https://github.com/esbraun/cleverly-tmle/pull/757),
  [#765](https://github.com/esbraun/cleverly-tmle/pull/765),
  [#863](https://github.com/esbraun/cleverly-tmle/pull/863),
  [#869](https://github.com/esbraun/cleverly-tmle/pull/869),
  [#878](https://github.com/esbraun/cleverly-tmle/pull/878))

**Full diff:** [`v0.2.8...v0.2.9`](https://github.com/esbraun/cleverly-tmle/compare/v0.2.8...v0.2.9)
