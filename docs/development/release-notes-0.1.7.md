# cleverly 0.1.7

This alpha release adds a readable `summary()` to every result, check and report type. It also
changes the default reduced-regression training of `DRTMLE` on cross-fitted fits. These are the
changes since 0.1.6.

## Upgrade notes

- `result.diagnostics.truncation_curve()` now returns a `TruncationCurve` and no longer returns a
  dataframe. Attribute access, `[]`, `len` and iteration go to the frame, so `curve["psi"]` still
  works. Call `curve.to_frame()` where a dataframe itself is needed, such as `isinstance` or a frame
  comparison. ([#510](https://github.com/esbraun/cleverly-tmle/pull/510))
- A cross-fitted `DRTMLE` fit with a non-empty guard now defaults to `reduced_crossfit="foldwise"`,
  and in 0.1.6 the default was `"pooled"`. Estimates of such fits change. Pass
  `reduced_crossfit="pooled"` to keep the earlier construction. The value `"nested"` is removed.
  `"foldwise"` needs at least three folds and refuses `targeting="one_step"`.
  ([#492](https://github.com/esbraun/cleverly-tmle/pull/492))

Refit analyses when you move saved results to 0.1.7. Loading another version's result emits
`VersionMismatchWarning`. The loader does not migrate the result, and the load can fail.

## Added

### Readable summaries

- `summary(detail="short")` and `summary(detail="full")` print a statsmodels-style report for a
  `TMLEResult`. `"full"` adds the settings, assumptions and study description after the notes.
  Any other `detail` raises `ValueError`.
  ([#477](https://github.com/esbraun/cleverly-tmle/pull/477),
  [#478](https://github.com/esbraun/cleverly-tmle/pull/478))
- The protocol record prints only at `detail="full"`.
  ([#474](https://github.com/esbraun/cleverly-tmle/pull/474))
- The point-estimate objects, the longitudinal results, and the study-setup objects (`CausalStudy`
  and its identified effect) have summaries.
  ([#480](https://github.com/esbraun/cleverly-tmle/pull/480),
  [#481](https://github.com/esbraun/cleverly-tmle/pull/481),
  [#482](https://github.com/esbraun/cleverly-tmle/pull/482),
  [#485](https://github.com/esbraun/cleverly-tmle/pull/485))
- Refutation reports print as readable summaries.
  ([#483](https://github.com/esbraun/cleverly-tmle/pull/483))
- The sensitivity analyses (E-value, omitted-variable bounds, marginal sensitivity bounds,
  missingness tilt, simulated confounding, positivity) have summaries.
  ([#484](https://github.com/esbraun/cleverly-tmle/pull/484),
  [#508](https://github.com/esbraun/cleverly-tmle/pull/508))
- The structural-model, g-formula and overlap reports have summaries.
  ([#486](https://github.com/esbraun/cleverly-tmle/pull/486))
- The validation checks have summaries: the equation check, the DR-TMLE correction check, the
  influence and weight reports, the overlap check, the outcome and treatment model check, the
  longitudinal checks, the combined check reports, the check list, the replayability report and the
  truncation curve.
  ([#488](https://github.com/esbraun/cleverly-tmle/pull/488),
  [#491](https://github.com/esbraun/cleverly-tmle/pull/491),
  [#498](https://github.com/esbraun/cleverly-tmle/pull/498),
  [#501](https://github.com/esbraun/cleverly-tmle/pull/501),
  [#503](https://github.com/esbraun/cleverly-tmle/pull/503),
  [#504](https://github.com/esbraun/cleverly-tmle/pull/504),
  [#506](https://github.com/esbraun/cleverly-tmle/pull/506),
  [#510](https://github.com/esbraun/cleverly-tmle/pull/510))
- `TruncationCurve` and `MethodAvailabilities` are exported from `cleverly`.
  ([#482](https://github.com/esbraun/cleverly-tmle/pull/482),
  [#510](https://github.com/esbraun/cleverly-tmle/pull/510))

### Validation evidence

- The registered long-horizon `LTMLE` and `SDR` studies supply Python property evidence for those
  fits, and their clauses are bound to the committed study rows.
  ([#475](https://github.com/esbraun/cleverly-tmle/pull/475),
  [#476](https://github.com/esbraun/cleverly-tmle/pull/476))
- The study-hypothesis clauses and the omitted-variable standard-error-ratio clause are bound to
  their study rows.
  ([#473](https://github.com/esbraun/cleverly-tmle/pull/473))

## Changed

- Library output (messages, summaries and refusals) uses plainer wording.
  ([#472](https://github.com/esbraun/cleverly-tmle/pull/472))
- Warnings raised inside parallel workers now reach the parent, in task order, so a fit records the
  same warnings at any `n_jobs`.
  ([#477](https://github.com/esbraun/cleverly-tmle/pull/477))

## Fixed

- `DRTMLE` on a cross-fitted fit no longer lets a fold's own data reach that fold's reduced
  regressions through the pooled training rows. Earlier versions showed about +0.23 SD of ATE bias
  at n = 2,000 against the fold-wise construction. The pooled construction stays available as
  `reduced_crossfit="pooled"`, which follows R `drtmle`'s `cvFolds`.
  ([#492](https://github.com/esbraun/cleverly-tmle/pull/492),
  [#495](https://github.com/esbraun/cleverly-tmle/pull/495))

## Packaging and compatibility

This remains an alpha release. The package does not promise compatibility for public APIs,
`CausalStudy` objects, pickled objects, or saved results. Follow the upgrade notes when you move
an analysis to 0.1.7.

**Full changelog:** [v0.1.6...v0.1.7](https://github.com/esbraun/cleverly-tmle/compare/v0.1.6...v0.1.7)
