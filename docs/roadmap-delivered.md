# Delivered roadmap items

An item leaves the [roadmap](roadmap.md) when it is delivered. Code comments, tests and evidence
pages still name delivered items by their IDs. The roadmap file at a commit that still held an
item keeps its full contract, so each row below links that version. The newest removals come first.

| items | what they were | the roadmap that holds them |
| --- | --- | --- |
| X76 | registered studies of the shipped targeting and fitting options: stopping rules, per-arm outcome regressions, patience, treatment-mechanism covariates, ratio trimming, Markov order and a binding isotonic projection, each backing support rows | [commit `c9c8037d`](https://github.com/esbraun/cleverly-tmle/blob/c9c8037d2e1d8b6a336ac325f2e1e67542facdb8/docs/roadmap.md#x76-studies-of-the-targeting-and-fitting-options) |
| X32b, X69, X90; the delivered parts of P1 and X92 | repeated outcomes for `SDR` and `IPSI`; the documentation review of the methods added up to 2026-10-30 (#518 to #528); adaptive TMLE for a trial augmented with external data; parts (a) to (c) of the EP learner; plot parts 1 to 6, including the plots in the tutorials. The same commit holds the release milestones and the priority table this page replaced | [commit `f22208d8`](https://github.com/esbraun/cleverly-tmle/blob/f22208d834f0faf3519816ffea76be6e0265ee61/docs/roadmap.md) |
| X59 | long-horizon comparator parity for `LTMLE` and `SDR` | [commit `9de2da30`](https://github.com/esbraun/cleverly-tmle/blob/9de2da305964a68437dc8fc4c4bce8fafccc9d26/docs/roadmap.md#x59-long-horizon-comparator-parity-for-ltmle-and-sdr) |
| X42 | the applied tutorials for offers and pricing | [commit `c1b2ce12`](https://github.com/esbraun/cleverly-tmle/blob/c1b2ce128f1b7ff48977a55345f545c4f588a968/docs/roadmap.md#x42-applied-tutorials-for-offers-and-pricing) |
| RM39 | the failure record and accounting of a study run | [commit `2a8ad955`](https://github.com/esbraun/cleverly-tmle/blob/2a8ad9552e7aeb19f171643da5311eb0c6b03a91/docs/roadmap.md#rm39-keep-a-study-run-alive-through-a-failed-replicate) |
| RM40 | the DR-TMLE inference correction when one nuisance is wrong | [commit `287a1fb5`](https://github.com/esbraun/cleverly-tmle/blob/287a1fb5385ad2cc0faa75e05672a3233012c236/docs/roadmap.md#rm40-correct-the-dr-tmle-inference-claim-when-one-nuisance-is-wrong) |
| X18 | the natural-extension review; the [natural-extension verdicts](technical-reference/natural-extension-verdicts.md) hold its answers | [commit `6a55f038`](https://github.com/esbraun/cleverly-tmle/blob/6a55f0388b703765eab79c78efec82474e67ab8e/docs/roadmap.md#x18-natural-extension-reviews) |
| RM8 to RM33 | the remediation rows delivered before the remediation queue emptied | [commit `4ce96cda`](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md) |

The shipped behavior of each item is described in the [user guide](user-guide/index.md) and the
[technical reference](technical-reference/index.md), and its evidence in the
[implementation validation grid](technical-reference/method-evidence/validation-grid.md).
