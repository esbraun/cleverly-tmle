# Patience on the cross-validated C-TMLE path (`option-patience`)

This study measures `CTMLE(patience=p)`, which stops the cross-validated scoring of a candidate
path after `p` candidates whose risk is not a new minimum (Fix-8 plan sections 2.3 and 5.5). It
reads point estimates, stop shares and paired displacements, because a selector path withholds
the interval (F18). It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/canonical_option_patience.py` | The record and the primary phase. |
| `tests/studies/option_patience_properties.py` | The property families and their rules. |
| `tests/studies/_option_patience_laws.py` | The laws, learners, fits and the LC3 truncation bound. |
| `tests/studies/option_patience_pilot.py` | The pilot and its committed summary. |
| `tests/unit/test_option_patience_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_option_patience_witness.py` | The fast-tier witnesses and mutation controls. |

## State of this directory

Declared, not run. No artifact is committed, so the record is not in `registered()`. The
support row `C12.greedy` (the `C01.greedy` key with `patience` declared, kinds point), the limit
on `C01.greedy`, the deletion of the greedy branch of `patience_conditions` in
`src/cleverly/estimators/_option_support.py` and the evidence page land with the registration,
after the run. The support key already keeps a greedy fit with `patience=` off `C01.greedy`, so
the deletion changes no label until `C12.greedy` exists.

The pre-ordered branch (`"ordered"`, `"discrete"`, `"lasso"`, `"learners"`) is permanent and
names no roadmap item. At registration its text gains the study and the row: "(study
option-patience, ordered_late_confounder) ... on strategy='greedy' each step adds the candidate
that lowers the risk most, and the registered row C12.greedy covers it."

The limit text that `C01.greedy` and `C12.greedy` are to carry, with the measured number of
`reported_estimate/lc3_greedy_whole`: "On a law where the outcome regression omits a confounder
and the selected treatment model is misspecified given the selected set, the greedy point
estimate keeps a bias (study option-patience, greedy whole path on LC3: X SD at n = 2,000)."
The pilot reads 0.42 SD. `C12.greedy` also carries "The selector plug-in interval is withheld."

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_patience/regenerate.py --replicates 2 --output /tmp/option-patience-smoke
```

Run the declared study:

```bash
python tests/canonical/option_patience/regenerate.py --output <empty scratch directory> --jobs 16
```
