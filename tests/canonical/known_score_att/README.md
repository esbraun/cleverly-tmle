# Known-score ATT and ATC on point TMLE (`known-score-att`)

This study measures point TMLE fitted with `conditional_form="known_score"` on data that declares
its treatment mechanism: the ATT and the ATC are the plug-ins weighted by the declared
probability of the conditioning arm, with the influence curve of the model in which the
mechanism is known (X35 plan sections 1, 2 and 5). It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/known_score_att.py` | The record and the primary phase. |
| `tests/studies/known_score_att_properties.py` | The property families, their rules and the paired gate. |
| `tests/studies/known_score_att_law.py` | Laws S, S0 and B, the learners, the fits and the exact truths. |
| `tests/studies/known_score_att_pilot.py` | The pre-read pilot and its committed summary. |
| `tests/unit/test_known_score_att_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_known_score_att.py` | The fast-tier exact-law witnesses and mutation controls. |

## State of this directory

Registered. The declared run's artifacts, manifest and log are committed here. Support rows
`T55` (family `P3`, qualifier `known_mechanism`, option `conditional_form="known_score"`) cite
its cells, and [the evidence page](../../../docs/technical-reference/method-evidence/known-score-att.md)
reports them.

## Run

Run a disposable smoke of the primary phase:

```bash
python -m tests.canonical.known_score_att.regenerate --replicates 2 --output /tmp/known-score-att-smoke
```

Run the declared study:

```bash
python -m tests.canonical.known_score_att.regenerate --output <empty scratch directory> --jobs "$(nproc --all)"
```
