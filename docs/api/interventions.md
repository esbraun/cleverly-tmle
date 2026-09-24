# Interventions

Use interventions to define static, dynamic, stochastic, incremental, or shifted treatment
assignments. Run the matching support check before you interpret an estimate. `DynamicRegimen`
defines a plan across the treatment nodes of a longitudinal fit. A node of that plan can be a
rule.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.interventions.Intervention
   cleverly.interventions.Static
   cleverly.interventions.Rule
   cleverly.interventions.Stochastic
   cleverly.longitudinal.DynamicRegimen
   cleverly.interventions.RegimeSet
   cleverly.interventions.Incremental
   cleverly.interventions.IPSISet
   cleverly.interventions.Shift
   cleverly.interventions.ShiftSet
   cleverly.interventions.SupportReport
   cleverly.interventions.RegimeSupport
   cleverly.interventions.IncrementalSupport
   cleverly.interventions.ShiftSupport
   cleverly.interventions.check_support
   cleverly.interventions.check_incremental_support
   cleverly.interventions.check_shift_support
```
