# Study design and identification

Use these objects to declare the observed data, identify a causal quantity, and start estimation.
`CausalStudy` is the main entry point. The typing protocols support advanced identification
extensions.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.CausalStudy
   cleverly.StudyProtocol
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

Pass `protocol=StudyProtocol(...)` to carry the scientific context through identification,
estimation, summaries, and persistence. Sequence inputs normalize to tuples. The constructor
rejects blank text. It also rejects a strategy list and a version list of different lengths.

[Who owns what](../workflow.md#who-owns-what) states which object owns which decision.
[Study protocol vocabulary](../references.md#study-protocol-vocabulary) names the sources of the
record and states its scope.
