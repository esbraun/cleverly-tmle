# Estimation methods

Use a method object to configure nuisance learning, cross-fitting, targeting, inference, and
runtime behavior. Pass the configured object to `IdentifiedEffect.estimate()`.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.EstimationMethod
   cleverly.TMLEMethod
   cleverly.AdaptiveIPTWMethod
   cleverly.CollaborativeTMLEMethod
   cleverly.DRTMLEMethod
   cleverly.SDRMethod
   cleverly.IPSIMethod
   cleverly.GComputationMethod
   cleverly.IPWMethod
   cleverly.ParametricGFormulaMethod
   cleverly.ModelSpec
   cleverly.CrossFitting
   cleverly.SplitPlan
   cleverly.Targeting
   cleverly.Inference
   cleverly.Runtime
   cleverly.MethodAvailability
   cleverly.MethodAvailabilities
```

## Longitudinal estimator classes

A study fit on a longitudinal design builds one of these classes from its method object. You can
also construct one directly with the regimens and learners and call `fit()` on a wide frame with
the column roles, as the [longitudinal guide](../user-guide/longitudinal.md) does.

| method object | class it builds |
| --- | --- |
| `TMLEMethod` | `LTMLE` |
| `SDRMethod` | `SDR` |
| `IPSIMethod` | `IPSI` |
| `GComputationMethod` | `SequentialGComputation` |
| `IPWMethod` | `SequentialIPW` |
| `ParametricGFormulaMethod` | `ParametricGFormula` |

`StructuralNestedMean` has no method object, so it is called only directly. Its `fit()` and the
one-call `snmm()` return a `StructuralNestedMeanResult`, whose `check_model()` returns a
`StructuralModelCheck`.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.longitudinal.LTMLE
   cleverly.longitudinal.SDR
   cleverly.longitudinal.IPSI
   cleverly.longitudinal.SequentialGComputation
   cleverly.longitudinal.SequentialIPW
   cleverly.longitudinal.ParametricGFormula
   cleverly.longitudinal.StructuralNestedMean
   cleverly.longitudinal.StructuralNestedMeanResult
   cleverly.longitudinal.StructuralModelCheck
   cleverly.longitudinal.snmm
```
