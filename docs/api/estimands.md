# Estimands

Choose one estimand for the causal question before you select an estimation method. Each class
defines its target population, contrast scale, intervention axis, and reference value.

```{autosummary}
:nosignatures:

cleverly.ATE
cleverly.ATT
cleverly.ATC
cleverly.BackdoorMeanContrast
cleverly.CounterfactualMean
cleverly.NaturalCourseMean
cleverly.PopulationAttributableRisk
cleverly.PopulationAttributableFraction
cleverly.RiskRatio
cleverly.OddsRatio
cleverly.RegimeMean
cleverly.RegimeContrast
cleverly.ModifiedTreatmentPolicy
cleverly.ModifiedTreatmentPolicyEffect
cleverly.IncrementalMean
cleverly.IncrementalEffect
cleverly.MSMProjection
cleverly.ControlledDirectEffect
```
