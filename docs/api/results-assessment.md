# Results and assessment

Use result objects to read estimates, uncertainty, influence curves, diagnostics, and provenance.
Reports provide stable summaries and dataframe output for downstream analysis.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.CausalResult
   cleverly.estimators.TMLEResult
   cleverly.longitudinal.LongitudinalResult
   cleverly.ParameterEstimate
   cleverly.ParameterKey
   cleverly.AssessmentCapability
   cleverly.AssessmentStatus
   cleverly.DiagnosticReport
   cleverly.ValidationReport
   cleverly.Replayability
   cleverly.VariableImportanceEntry
   cleverly.VariableImportanceResult
   cleverly.variable_importance
   cleverly.load
```

`CausalResult` is the protocol every fitted scalar result satisfies. The concrete classes are
`TMLEResult` for a point-treatment fit and `LongitudinalResult` for a sequential one; `estimate()`
returns one of those two.
