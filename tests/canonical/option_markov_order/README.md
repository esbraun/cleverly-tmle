# Markov-order windows on LTMLE (`option-markov-order`)

This study measures `LTMLE` fitted on `LongitudinalData.with_markov_order(k)`, where every
nuisance at node `t` reads the baseline covariates and the last `k` nodes, on laws where the
order is known (Fix-8 plan sections 2.6 and 5.7). It has no comparator: R `ltmle` has no window
option, so every cell is read against its exact truth.

| file | role |
| --- | --- |
| `tests/studies/canonical_option_markov_order.py` | The record and the primary phase. |
| `tests/studies/option_markov_order_properties.py` | The property families and their rules. |
| `tests/studies/_option_markov_law.py` | The laws, learners, fits and exact-law limits. |
| `tests/studies/option_markov_order_pilot.py` | The pre-read pilot and its committed summary. |
| `tests/unit/test_option_markov_order_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_option_markov_order_witness.py` | The fast-tier exact-law witnesses and mutation controls. |

## State of this directory

Registered. The declared run is committed here and the record is in `registered()`. The
support row `G57` covers a windowed LTMLE fit on the key of `G01` with a four-node treatment,
and its limit states the conditional-independence restriction the window imposes. The evidence
page is `docs/technical-reference/method-evidence/option-markov-order.md`.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_markov_order/regenerate.py --replicates 2 --output /tmp/option-markov-order-smoke
```

Run the declared study:

```bash
python tests/canonical/option_markov_order/regenerate.py --output <empty scratch directory>
```
