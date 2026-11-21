# cleverly 0.2.8

This release lets `refute()` count, checkpoint and sample its refits, sets the few-cluster interval
floors per fit family, and adds the projection of a controlled direct effect. It also removes the
optional `glmstar` solver.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.
- The optional `glmstar` solver of the C-TMLE L1 path is removed: `CTMLE(l1_solver=...)`,
  `CollaborativeTMLEMethod(l1_solver=...)`, the `l1_solver` setting of `SurvivalCTMLE`, the
  `cleverly[glmstar]` extra, and the selection-record fields `l1_solver`, `l1_polished`,
  `l1_fallbacks` and `l1_fallback_reasons`. Drop the keyword; the built-in solver is now as fast.
  Results of the default path are bitwise unchanged.
  ([#739](https://github.com/esbraun/cleverly-tmle/pull/739))
- The built-in L1 path (`CTMLE(strategy="lasso")`, `strategy="derivative"` without
  `treatment_candidates=`, and `L1Logistic(penalty > 0)`) now raises `ValueError` on a negative
  row weight, which it used to accept.
  ([#737](https://github.com/esbraun/cleverly-tmle/pull/737))
- Cluster-robust intervals on the t(J−2) reference now use a minimum cluster count for each fit
  family, measured by the registered `few-cluster-floor-tmle` and `few-cluster-floor-ltmle`
  studies. The floors are 6 for in-sample TMLE, 10 for stacked TMLE, 8 for fold-evaluated TMLE and for DR-TMLE
  with a known mechanism, 6 for in-sample LTMLE and 4 for cross-fitted LTMLE. A floor applies only
  to the measured contrast on the measured design (the ATE for point fits; `always` against
  `never` for LTMLE). Other estimands and designs keep the default floors of 10 (point) and 20
  (LTMLE). ([#729](https://github.com/esbraun/cleverly-tmle/pull/729))

## Added

### Refutation

- `cleverly.validation.planned_refits(result, ...)`, also `result.diagnostics.planned_refits(...)`,
  counts the refits and learner fits that a `refute()` call with the same arguments would make,
  without fitting, and refuses what `refute()` would refuse before its first refit.
  ([#740](https://github.com/esbraun/cleverly-tmle/pull/740))
- `refute(..., checkpoint=RefutationCheckpoint(directory), progress=callback)` stores each draw of
  `dummy_outcome`, `simulated_outcome`, `plasmode`, `bootstrap_stability` and
  `bootstrap_measurement_error` as it finishes. A stopped call resumes from the directory, refits
  only the missing draws, and returns the same report as an uninterrupted call. A changed request
  (data, seeds, estimator settings, library versions) refuses and names the part that differs.
  ([#741](https://github.com/esbraun/cleverly-tmle/pull/741))
- `refute(result, sample_limit=m)` refits the fit's estimator on a simple random sample of `m`
  independent units (whole clusters on a clustered fit, whole trajectories on a panel or survival
  fit) and runs every requested test on that sampled fit. Its verdicts are those of an `m`-unit
  fit; a limit of at least `n` is the full call. The registered `refuter-sampled` study measured
  size and power by `m` and `m/n`. `RefutationResult.sample` records the draw.
  ([#743](https://github.com/esbraun/cleverly-tmle/pull/743))

### Clustered longitudinal fits

- `LTMLE(cluster_variance="jackknife")` replaces the cluster influence-curve variance with the
  delete-one-cluster jackknife, keeping the t(J−2) reference. It needs parametric nuisance
  learners (a GLM, a Super Learner of GLMs, a declared-parametric learner or a known mechanism).
  `TMLE` refuses the option.
  ([#729](https://github.com/esbraun/cleverly-tmle/pull/729))

### Effect modification

- The projection TMLE of a conditional contrast fits the projection of a controlled direct effect,
  `ControlledDirectEffect(intermediate=z, contrast=ConditionalContrast(..., model=basis))`, on the
  difference and log-ratio scales, stacked or cross-fitted, with one outcome regression per arm or
  one pooled regression. The score check, the targeting curve and the held-out curve checks refuse
  on such a fit. The registered `projection-evidence` study supports it (support rows
  `T49.intermediate`, `T50.intermediate`, `T49.pooled` and `T50.pooled`).
  ([#738](https://github.com/esbraun/cleverly-tmle/pull/738))
- The simultaneous band of a projection curve (Rademacher, 1,000 draws) reads supported on the
  configurations `projection-evidence` measured.
  ([#738](https://github.com/esbraun/cleverly-tmle/pull/738))
- The EP curve on the log-ratio scale takes a declared `final_learner`. The learner needs
  `predict_proba` and a `fit` that names `sample_weight`.
  ([#738](https://github.com/esbraun/cleverly-tmle/pull/738))
- The external-score check reads supported with weights, clusters (40 or more), missing outcomes,
  the pair (2, 0) of three arms, or an ATE fit in sample, each alone, on the difference-scale binary
  configuration with an estimated mechanism, from the registered `score-assessment-compositions`
  study. A fit with two of these compositions reads experimental.
  ([#738](https://github.com/esbraun/cleverly-tmle/pull/738))

## Changed

- `ContinuousSurvivalTMLE` no longer refuses a fit in which a unit's cause-specific hazard jumps at
  one arm and time sum to one or more. It scales each such cell to 1 − 1e-8, keeping each cause's
  share, raises a `PositivityWarning` naming the cause, the arm and the number of pairs, and marks
  the estimates experimental. `result.record.jump_bound_share` and `jump_bound_at_risk_share`
  report how often it acted. An infinite sum still refuses. Fits where the bound does not act are
  unchanged. ([#732](https://github.com/esbraun/cleverly-tmle/pull/732))
- The built-in L1-penalized logistic and multinomial path is a working-set proximal Newton solver
  with the same KKT residual contract. A default lasso C-TMLE fit at 100,000 rows runs about 1.3
  times as fast, and a derivative fit with `lambdas=None` about 2.7 times; replaying the
  registered C-TMLE studies changed no selection.
  ([#737](https://github.com/esbraun/cleverly-tmle/pull/737))
- `refute()` holds one subset or band-rank placebo draw in memory at a time, with every draw and
  every value unchanged.
  ([#735](https://github.com/esbraun/cleverly-tmle/pull/735))
- `refute()` skips work it never reads: no simultaneous band on the `future_treatment` control or
  on refits of a banded TMLE fit, and the longitudinal and clustered plasmode laws are built once.
  Values are unchanged; `future_treatment` on a 100,000-row fit runs about 12 times as fast. The
  negative-control learner clauses now name the learner each refit uses.
  ([#742](https://github.com/esbraun/cleverly-tmle/pull/742),
  [#740](https://github.com/esbraun/cleverly-tmle/pull/740))

**Full diff:** [`v0.2.7...v0.2.8`](https://github.com/esbraun/cleverly-tmle/compare/v0.2.7...v0.2.8)
