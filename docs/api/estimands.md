# Estimands

Choose one estimand for the causal question before you select an estimation method. Each class
defines its target population, contrast scale, intervention axis, and reference value.
An `Interpretation` names one causal reading of an identified functional and the assumptions
that reading adds.

```{eval-rst}
.. autosummary::
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
   cleverly.LearnedRuleValue
   cleverly.OptimalRuleValue
   cleverly.ModifiedTreatmentPolicy
   cleverly.ModifiedTreatmentPolicyEffect
   cleverly.targets.Interpretation
   cleverly.IncrementalMean
   cleverly.IncrementalEffect
   cleverly.MSMProjection
   cleverly.msm.MSM
   cleverly.ControlledDirectEffect
```
