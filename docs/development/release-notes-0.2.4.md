# cleverly 0.2.4

This release adds a drift-corrected survival TMLE and logit-scale simultaneous bands, and makes
plots cleaner: short titles, shared label keywords, and no note text drawn on the figure.

## Upgrade notes

- Results saved by an earlier release load with a `VersionMismatchWarning`, and the load does not
  migrate them. Fit each analysis again with this release.
- Plots no longer draw notes under their axes, and the figure does not mark an experimental fit
  or band. Read `cleverly.plotting.notes(fig)` before you publish a figure.
  ([#623](https://github.com/esbraun/cleverly-tmle/pull/623))
- `confounding()` takes `xlabel=` and `ylabel=` in place of `x_label=` and `y_label=`.
  ([#624](https://github.com/esbraun/cleverly-tmle/pull/624))
- `multiplier_critical_value()` takes `kind` and `n_replicates` as required keyword arguments.
  ([#617](https://github.com/esbraun/cleverly-tmle/pull/617))
- Bands with `multiplier_kind="mammen"` draw a different multiplier stream, so their critical
  values change at a fixed seed. The draws keep the Mammen moments to within 5.2e-10.
  ([#617](https://github.com/esbraun/cleverly-tmle/pull/617))

## Added

### Survival

- `SurvivalDRTMLE`, reached through `method=DRTMLEMethod()` on a `TimeToEvent` design, is the
  drift-corrected hazard TMLE of a risk curve from Díaz (2019), Section 5. Its interval stays
  valid when only one side is consistent: the event model, or the treatment and censoring models
  together, provided the wrong side keeps the risk ordering its truth gives. It fits a two-level
  treatment with one event type. ([#618](https://github.com/esbraun/cleverly-tmle/pull/618))
- `SurvivalDRTMLE` corrects four defects in eq. (9) of Díaz (2019), recorded in `references.md`.
  It refuses compositions outside the paper, including an estimated treatment model with declared
  censoring. The registered `survival-dr-tmle` study gates
  coverage, robustness and calibration at n = 1,000, 4,000 and 16,000 with ten folds.
  ([#618](https://github.com/esbraun/cleverly-tmle/pull/618))

### Simultaneous bands

- `band_scale="logit"` on `simultaneous_bands()`, `median_bands()` and every estimator with a
  band builds the band of a probability on its logit scale and of a difference of probabilities
  on its atanh scale; the default stays `"identity"`. `ParameterEstimate.bounds` records each
  bounded target's range on a binary outcome. A bounded estimate on or outside its range with a
  positive standard error has no logit band: `simultaneous_bands()` raises `CapabilityError`, and
  a fit skips the band and says why in `summary()`.
  ([#617](https://github.com/esbraun/cleverly-tmle/pull/617))

### Plotting

- `cleverly.plotting.notes(fig)` returns the notes a plot recorded: what an interval is, why one
  is missing, and which fits or bands are experimental. Each plot's docstring says what its notes
  cover. ([#623](https://github.com/esbraun/cleverly-tmle/pull/623))
- All 14 public plots, and the `.plot()` methods that forward to them, take `title=`,
  `subtitle=` (a string or `False`), `xlabel=`, `ylabel=` (a string or one per panel),
  `legend=` (`"top"`, `"right"`, `"bottom"` or `False`) and `figsize=`.
  ([#624](https://github.com/esbraun/cleverly-tmle/pull/624))

## Changed

- Plots use short descriptive titles, such as "Estimates and intervals", in place of questions.
  Subtitles are shorter and name estimands in words, and axis labels show column names with
  spaces. ([#624](https://github.com/esbraun/cleverly-tmle/pull/624))
- Band support labels now come from the band cells each row measured. 26 rows no longer support
  a band, so their band reads experimental while their pointwise intervals keep their labels; a
  band with a non-default `multiplier_kind` or `band_scale` reads experimental. For cross-fitted
  SDR (row G23.cross_fitted), the logit-scale Mammen band with at least 10,000 draws reads
  supported (joint coverage 0.942 for six regimens), and the default band now reads experimental
  (0.931). The [simultaneous bands](../technical-reference/inference.md#simultaneous-bands)
  reference gives the rule.
  ([#617](https://github.com/esbraun/cleverly-tmle/pull/617))

## Fixed

- Diagnostic plots that create their own figure (`calibration`, `learners`, `repeats`,
  `bootstrap`, `influence`) keep their title, subtitle and key in Jupyter's inline display and in
  `savefig(bbox_inches="tight")`. Earlier versions cropped them.
  ([#621](https://github.com/esbraun/cleverly-tmle/pull/621))

## Documentation

- The survival user guide adds a section on keeping the interval when one set of models is wrong.
  ([#618](https://github.com/esbraun/cleverly-tmle/pull/618))
- The plotting API page documents the label keywords and the notes contract, and the tutorials
  are re-executed with the new figures.
  ([#623](https://github.com/esbraun/cleverly-tmle/pull/623),
  [#624](https://github.com/esbraun/cleverly-tmle/pull/624))

**Full changelog:** [v0.2.3...v0.2.4](https://github.com/esbraun/cleverly-tmle/compare/v0.2.3...v0.2.4)
