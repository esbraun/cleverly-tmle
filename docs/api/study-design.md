# Study design and identification

Use these objects to declare the observed data, identify a causal quantity, and start estimation.
`CausalStudy` is the main entry point. The protocols support advanced identification extensions.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.CausalStudy
   cleverly.PointTreatment
   cleverly.LongitudinalTreatment
   cleverly.Estimand
   cleverly.IdentificationProvider
   cleverly.ExplicitAdjustmentProvider
   cleverly.IdentifiedEffect
   cleverly.Provenance
```

Call `CausalStudy(data, design=...)`, then call `identify()` or `estimate()`. An identification
provider extends this sequence. It does not waive the identification and evidence requirements.
