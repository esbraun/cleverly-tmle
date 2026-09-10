# Study design and identification

Use these objects to declare the observed data, identify a causal quantity, and start estimation.
`CausalStudy` is the main entry point. The protocols support advanced identification extensions.

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

Pass `protocol=StudyProtocol(...)` to preserve the scientific context across identification,
estimation, summaries, and persistence. Sequence inputs normalize to tuples. The constructor
rejects blank text and unmatched treatment strategy and version counts.

`StudyProtocol.to_dict()` returns the JSON-compatible normalized record. `canonical_json` supplies
its canonical representation, and `fingerprint` supplies its BLAKE2b digest. The digest identifies
the protocol record. It does not replace the separate data and fold fingerprints in `Provenance`.

The public objects divide responsibility as follows:

| object | responsibility |
| --- | --- |
| `StudyProtocol` | study context and study-specific assumption rationale |
| `PointTreatment` or `LongitudinalTreatment` | observed column roles and treatment-time structure |
| typed estimand | the mathematical contrast and intervention metadata |
| `IdentifiedEffect` | the observed-data functional and identification contract |
| estimation method | learner, targeting, inference, and runtime configuration |

The record uses selected
[target-trial components](https://miguelhernan.org/whatifbook),
[ICH E9(R1)](https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf),
and concepts from the literature on
[treatment versions](https://pmc.ncbi.nlm.nih.gov/articles/PMC4219328/). It is not a complete
target-trial protocol or a claim of ICH estimand compliance.

Trusted point and longitudinal artifacts written before `StudyProtocol` still load. Their
summaries state `causal study protocol: absent`, and their protocol fingerprint is `None`.
