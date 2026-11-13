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

Registered from the run at `c4e89da` (`run.log`). Every gated cell passes. The study backs the
support row `C12.greedy` (the `C01.greedy` key with `patience` declared, kinds point) and the
limit on `C01.greedy` and `C12.greedy`: the greedy whole path on LC3 reads 0.42 SD. The greedy
branch of `patience_conditions` in `src/cleverly/estimators/_option_support.py` is deleted, so
the support key alone labels a greedy fit with `patience=`. The pre-ordered branch
(`"ordered"`, `"discrete"`, `"lasso"`, `"learners"`) is permanent, names no roadmap item, and
cites the cell `ordered_late_confounder` and the row `C12.greedy`. The evidence page is
`docs/technical-reference/method-evidence/option-patience.md`.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_patience/regenerate.py --replicates 2 --output /tmp/option-patience-smoke
```

Run the declared study:

```bash
python tests/canonical/option_patience/regenerate.py --output <empty scratch directory> --jobs 16
```
