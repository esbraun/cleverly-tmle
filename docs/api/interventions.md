# Interventions

Use interventions to define static, dynamic, stochastic, incremental, or modified treatment
assignments. Run the matching support check before you interpret an estimate.

A modified treatment policy is a `Shift`, a `Scale`, a `Piecewise` map, a `ModifiedPolicy` with
declared pieces, or a `RiskRatioTilt`. Pass the policy to a `ModifiedTreatmentPolicy` or
`ModifiedTreatmentPolicyEffect` estimand, or as a node of a longitudinal plan. Both estimands refuse
a `RiskRatioTilt`, which `cleverly.estimators.TMLE(policies=...)` fits. A vector treatment takes a
`JointPolicy`, which declares one policy for each component.

A `DoseQuantile` or a `SupportBoundary` tunes the cap of a `Shift` or a `Scale` from the treatment
law, and `LinearQuantileRegression` is the linear boundary learner of a `SupportBoundary`. A
`Support` declares the interval of doses that a node can take given its history. The fit refuses,
before any learner, an observed dose outside it and a policy that assigns a dose outside it.

`Incremental` tilts the odds of the treatment. A scalar multiplies the odds of the second of two
levels in sorted order. A mapping from level label to factor tilts a treatment with three or more levels, and a
level that it does not name keeps the factor 1. `Incremental.factors` resolves a declaration
against the levels of one treatment or one node.

`DynamicRegimen` defines a plan across the treatment nodes of a longitudinal fit. A node of that
plan can be a rule. It can also be a `Stochastic` node that draws its arm from a known policy.
`each_node` writes one rule for every node of a long panel, and it returns one `NodeRule` per node.

`LearnedRule` declares a rule that the fit learns inside each training fold. By default the rule
comes from the fit's outcome model, and `loss="blip"` or `loss="weighted_classification"` learns
it with a learner that you pass. `LearnedRuleRecord` records that fit, with one `LearnedRuleSplit`
for each split of the folds.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.interventions.Intervention
   cleverly.interventions.Static
   cleverly.interventions.Rule
   cleverly.interventions.Stochastic
   cleverly.longitudinal.DynamicRegimen
   cleverly.longitudinal.each_node
   cleverly.longitudinal.NodeRule
   cleverly.interventions.LearnedRule
   cleverly.interventions.LearnedRuleRecord
   cleverly.interventions.LearnedRuleSplit
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
   cleverly.interventions.Support
   cleverly.interventions.ModifiedPolicy
   cleverly.interventions.Randomizer
   cleverly.interventions.RiskRatioTilt
   cleverly.interventions.JointPolicy
   cleverly.interventions.PolicySet
   cleverly.interventions.SupportReport
   cleverly.interventions.RegimeSupport
   cleverly.interventions.IncrementalSupport
   cleverly.interventions.PolicySupport
   cleverly.interventions.check_support
   cleverly.interventions.check_incremental_support
   cleverly.interventions.check_policy_support
```
