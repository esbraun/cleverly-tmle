# Results and assessment

Use result objects to read estimates, uncertainty, influence curves, diagnostics, and provenance.
Each result and report prints a readable summary with `summary()`, and `to_frame()` returns its
rows as a dataframe.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.CausalResult
   cleverly.estimators.TMLEResult
   cleverly.estimators.DoseResponseResult
   cleverly.estimators.PairSupport
   cleverly.longitudinal.LongitudinalResult
   cleverly.AdaptiveTMLEResult
   cleverly.estimators.atmle_result.AdaptiveTMLESupport
   cleverly.estimators.atmle_result.AdaptiveTMLEScoreReport
   cleverly.estimators.atmle_result.AdaptiveTMLENuisanceReport
   cleverly.estimators.atmle_result.WorkingModelFold
   cleverly.estimators.atmle_result.AdaptiveTMLECorrections
   cleverly.longitudinal.LongitudinalRepeat
   cleverly.fluctuation.Fluctuation
   cleverly.fluctuation.FoldFluctuation
   cleverly.ParameterEstimate
   cleverly.ParameterKey
   cleverly.AssessmentCapability
   cleverly.AssessmentReport
   cleverly.AssessmentStatus
   cleverly.DiagnosticReport
   cleverly.ValidationReport
   cleverly.Replayability
   cleverly.TruncationCurve
   cleverly.VariableImportanceEntry
   cleverly.VariableImportanceResult
   cleverly.variable_importance
   cleverly.load
   cleverly.effect_curves.EffectCurveResult
   cleverly.effect_curves.CurveBand
   cleverly.effect_curves.HeterogeneityTest
   cleverly.effect_curves.ScoreAssessment
   cleverly.targeting_curves.TargetingCurve
```

`CausalResult` is the protocol every fitted scalar result satisfies. The concrete classes are
`TMLEResult` for a point-treatment fit, `LongitudinalResult` for a sequential one, and
`AdaptiveTMLEResult` for a randomized trial pooled with external data (`TrialWithExternalData`);
`estimate()` returns one of them. The adaptive fit's diagnostics return the four
`cleverly.estimators.atmle_result` reports listed above.

`TMLEResult.fluctuations` returns one `Fluctuation` for each target group. A fold-targeted
cross-validated fit fills that record's `folds` field with one `FoldFluctuation` for each
validation fold.

`TMLEResult.effect_curve()` returns an `EffectCurveResult` for a fit of a `ConditionalContrast`,
and `TMLEResult.heterogeneity_test()` returns a `HeterogeneityTest`.
`IdentifiedEffect.assess_score()` returns a `ScoreAssessment` for a score fitted on other data:
its risk difference against the best constant and its calibration intercept and slope. With
`compare=` it also holds the second score's three parameters in `compared` and the paired risk
difference of the two. Every estimate comes from one TMLE fit and carries a Wald interval and a
support label; `status` is the worst of them, and `span` records the fluctuation's span.
`IdentifiedEffect.targeting_curve()` and `EffectCurveResult.diagnostics.targeting_curve()` return
a `TargetingCurve`: how much treating by a score first gains over treating at random.
