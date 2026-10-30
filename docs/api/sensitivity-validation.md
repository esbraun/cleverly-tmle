# Sensitivity and validation

Use these objects after estimation. `result.sensitivity` is a `SensitivityFacade`. Its methods vary
assumptions that the data cannot check, such as no unmeasured confounding, and report how the
estimate moves. `result.diagnostics` is a `DiagnosticsFacade`, whose methods check the fit itself.
Each facade's `run_all()` runs the analyses that need no further arguments, and skips refits and
moderate retargets unless you ask for them. Each facade's `capabilities` lists every analysis,
whether the fit supports it, and why not.

The functions listed below take the result as their first argument. [Results and
assessment](../user-guide/results-assessment.md) shows the common calls, and [validation
methods](../technical-reference/validation-methods.md) states the conditions of each.

## Sensitivity

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.assessment.SensitivityFacade
   cleverly.sensitivity.PositivityReport
   cleverly.sensitivity.SensitivityBounds
   cleverly.sensitivity.SensitivityElements
   cleverly.sensitivity.RobustnessValue
   cleverly.sensitivity.SensitivityContour
   cleverly.sensitivity.MissingnessTilt
   cleverly.sensitivity.BenchmarkResult
   cleverly.sensitivity.ConfounderStrengthGrid
   cleverly.sensitivity.ObservedConfounderCalibration
   cleverly.sensitivity.SimulatedConfoundingCell
   cleverly.sensitivity.SimulatedConfoundingResult
   cleverly.sensitivity.EValue
   cleverly.sensitivity.MarginalSensitivityBounds
   cleverly.sensitivity.MarginalSensitivityRow
   cleverly.sensitivity.positivity_report
   cleverly.sensitivity.truncation_curve
   cleverly.sensitivity.omitted_variable_bounds
   cleverly.sensitivity.robustness_value
   cleverly.sensitivity.benchmark
   cleverly.sensitivity.evalue.evalue
   cleverly.sensitivity.missingness_tilt
   cleverly.sensitivity.tipping_gamma
   cleverly.sensitivity.marginal_sensitivity_bounds
   cleverly.sensitivity.simulated_confounding
```

## Validation

`refute()` reads the outcome laws below (`GaussianAdjustmentOutcome` and the rest) for its
generated-outcome tests, and judges each reading against its `EquivalenceBands`. `CoverageStudy`
measures bias and coverage over repeated draws from a law you declare.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.assessment.DiagnosticsFacade
   cleverly.validation.ScoreCheck
   cleverly.validation.ScoreCheckRow
   cleverly.validation.NuisanceDiagnostics
   cleverly.validation.NuisanceModelReport
   cleverly.validation.RepeatSpreadRow
   cleverly.validation.GaussianNoise
   cleverly.validation.GaussianIndependentOutcome
   cleverly.validation.GaussianAdjustmentOutcome
   cleverly.validation.BernoulliAdjustmentOutcome
   cleverly.validation.BoundedAdjustmentOutcome
   cleverly.validation.LogisticHazardOutcome
   cleverly.validation.MultinomialHazardOutcome
   cleverly.validation.RelativeGaussianNoise
   cleverly.validation.BootstrapMeasurementError
   cleverly.validation.BandReading
   cleverly.validation.EmpiricalInclusionRule
   cleverly.validation.EquivalenceBands
   cleverly.validation.EmpiricalRefitRecord
   cleverly.validation.RefutationTest
   cleverly.validation.RefutationResult
   cleverly.validation.CoverageStudy
   cleverly.validation.StudyResult
   cleverly.validation.ReplicationRecord
   cleverly.validation.ReplicationFailure
   cleverly.validation.ReplicationCheckpoint
   cleverly.validation.DirectoryCheckpoint
   cleverly.validation.score_check
   cleverly.validation.nuisance_diagnostics
   cleverly.validation.refute
```

## Inference primitives

These functions compute influence-curve variances and covariances, delta-method influence curves,
simultaneous bands, and full-refit bootstrap replicates.
`Transform` is a monotone map on whose scale a contrast's interval and test are computed.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.inference.BootstrapResult
   cleverly.inference.SimultaneousBands
   cleverly.inference.Transform
   cleverly.inference.delta_method
   cleverly.inference.influence_covariance
   cleverly.inference.influence_variance
   cleverly.inference.multiplier_critical_value
   cleverly.inference.simultaneous_bands
   cleverly.inference.run_bootstrap
```
