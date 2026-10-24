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
   cleverly.Repeated
   cleverly.TimeToEvent
   cleverly.LongPanel
   cleverly.longitudinal.PanelLayout
   cleverly.longitudinal.LongitudinalData.from_long_panel
   cleverly.longitudinal.LongitudinalData.to_long_panel
   cleverly.Estimand
   cleverly.IdentificationProvider
   cleverly.ExplicitAdjustmentProvider
   cleverly.IdentifiedEffect
   cleverly.Provenance
```

`LongPanel` reads a long panel with one row per unit and decision time. It builds the same
trajectory container as `LongitudinalData.from_long_panel`, and `PanelLayout` records the roles
that the container was built from.

Call `CausalStudy(data, design=...)`, then call `identify()` or `estimate()`. An identification
provider extends this sequence. It does not waive the identification and evidence requirements.

Pass `protocol=StudyProtocol(...)` to preserve the scientific context across identification,
estimation, summaries, and persistence. Sequence inputs normalize to tuples. The constructor
rejects blank text and unmatched treatment strategy and version counts.

`StudyProtocol.to_dict()` returns the JSON-compatible normalized record. `canonical_json` supplies
its canonical representation, and `fingerprint` supplies its BLAKE2b digest. The digest identifies
the protocol record. It does not replace the separate data and fold fingerprints in `Provenance`.

An identification or result summary prints the analysis by default. Pass `detail="full"` to
`summary()` to print the record's fields too. No summary prints the fingerprint.
`protocol.summary()` prints the record as tables, and `"\n".join(protocol.summary_lines())` prints
each field on one line.

The fingerprint records the protocol. It does not verify it. The package does not check that
`treatment_strategies` names the regimens or arms a result estimates, because the strategies are
free text. The typed estimand stores the contrast.

The public objects divide responsibility as follows:

| object | responsibility |
| --- | --- |
| `StudyProtocol` | study context and study-specific assumption rationale |
| `PointTreatment`, `LongitudinalTreatment`, `LongPanel` or `TimeToEvent` | observed column roles and treatment-time structure |
| `Repeated` | a non-absorbing outcome column at every node, passed as `outcome=` of `LongitudinalTreatment` |
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
full summaries say that no study protocol was recorded, and their protocol fingerprint is `None`.
