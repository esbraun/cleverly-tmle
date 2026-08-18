# Study design and identification

```{autosummary}
:toctree: generated
:nosignatures:

cleverly.CausalStudy
cleverly.PointTreatment
cleverly.LongitudinalTreatment
cleverly.Estimand
cleverly.IdentificationProvider
cleverly.ExplicitAdjustmentProvider
cleverly.IdentifiedEffect
cleverly.CausalResult
cleverly.ParameterKey
cleverly.Provenance
```

The intended entry point is `CausalStudy(data, design=...)`, followed by `identify()` and
`estimate()`. Provider protocols are documented for extensions; using one does not waive the
identification and evidence requirements.
