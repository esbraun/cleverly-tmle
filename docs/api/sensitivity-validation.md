# Sensitivity and validation

Use these objects after estimation. Diagnostics inspect fitted artifacts, sensitivity methods vary
assumptions, and inference primitives construct derived uncertainty summaries.

## Sensitivity

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.assessment.SensitivityFacade
   cleverly.sensitivity.PositivityReport
   cleverly.sensitivity.SensitivityBounds
   cleverly.sensitivity.SensitivityElements
   cleverly.sensitivity.BenchmarkResult
   cleverly.sensitivity.EValue
   cleverly.sensitivity.positivity_report
   cleverly.sensitivity.truncation_curve
   cleverly.sensitivity.omitted_variable_bounds
   cleverly.sensitivity.robustness_value
   cleverly.sensitivity.benchmark
   cleverly.sensitivity.evalue.evalue
   cleverly.sensitivity.missingness_tilt
   cleverly.sensitivity.tipping_gamma
```

## Validation

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.assessment.DiagnosticsFacade
   cleverly.validation.ScoreCheck
   cleverly.validation.ScoreCheckRow
   cleverly.validation.NuisanceDiagnostics
   cleverly.validation.NuisanceModelReport
   cleverly.validation.GaussianNoise
   cleverly.validation.GaussianIndependentOutcome
   cleverly.validation.GaussianAdjustmentOutcome
   cleverly.validation.EmpiricalInclusionRule
   cleverly.validation.GeneratedOutcomeRecord
   cleverly.validation.RefutationTest
   cleverly.validation.RefutationResult
   cleverly.validation.CoverageStudy
   cleverly.validation.StudyResult
   cleverly.validation.ReplicationRecord
   cleverly.validation.ReplicationFailure
   cleverly.validation.score_check
   cleverly.validation.nuisance_diagnostics
   cleverly.validation.refute
```

## Inference primitives

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.inference.BootstrapResult
   cleverly.inference.SimultaneousBands
   cleverly.inference.delta_method
   cleverly.inference.influence_covariance
   cleverly.inference.influence_variance
   cleverly.inference.multiplier_critical_value
   cleverly.inference.simultaneous_bands
   cleverly.inference.run_bootstrap
```
