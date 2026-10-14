# Interventions

Use interventions to define static, dynamic, stochastic, incremental, or modified treatment
assignments. Run the matching support check before you interpret an estimate.

A modified treatment policy is a `Shift`, a `Scale`, a `Piecewise` map, a `ModifiedPolicy` with
declared pieces, or a `RiskRatioTilt`. Pass the policy to `TMLE(policies=...)` or as a node of
a longitudinal plan. A `DoseQuantile` or a `SupportBoundary` tunes the cap of a `Shift` or a
`Scale` from the treatment law. `LTMLE` and `SDR` accept a tuned cap, and `TMLE` refuses it.

`DynamicRegimen` defines a plan across the treatment nodes of a longitudinal fit. A node of that
plan can be a rule. It can also be a `Stochastic` node that draws its arm from a known policy.
`LearnedRule` declares a rule that the fit learns inside each training fold, and
`LearnedRuleRecord` records that fit.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.interventions.Intervention
   cleverly.interventions.Static
   cleverly.interventions.Rule
   cleverly.interventions.Stochastic
   cleverly.longitudinal.DynamicRegimen
   cleverly.interventions.LearnedRule
   cleverly.interventions.LearnedRuleRecord
   cleverly.interventions.RegimeSet
   cleverly.interventions.Incremental
   cleverly.interventions.IPSISet
   cleverly.interventions.Shift
   cleverly.interventions.Scale
   cleverly.interventions.DoseQuantile
   cleverly.interventions.SupportBoundary
   cleverly.interventions.LinearQuantileRegression
   cleverly.interventions.Piecewise
   cleverly.interventions.Piece
   cleverly.interventions.ModifiedPolicy
   cleverly.interventions.Randomizer
   cleverly.interventions.RiskRatioTilt
   cleverly.interventions.PolicySet
   cleverly.interventions.SupportReport
   cleverly.interventions.RegimeSupport
   cleverly.interventions.IncrementalSupport
   cleverly.interventions.PolicySupport
   cleverly.interventions.check_support
   cleverly.interventions.check_incremental_support
   cleverly.interventions.check_policy_support
```
