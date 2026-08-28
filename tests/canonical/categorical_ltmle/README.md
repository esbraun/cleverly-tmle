# Ordinary categorical longitudinal TMLE fixture

This registered study fits ordinary longitudinal TMLE on a two-node law with three labelled
treatment levels. It compares the result with pinned R `lmtp` 1.5.4 on identical samples. Both
implementations receive the exact treatment mechanism and one fold that contains every row.

## One recorded module hash is not the committed one

`manifest.json` records `tests/studies/categorical_longitudinal_properties.py` at
`b8cc6bc6`, and the committed file hashes to `fecaead3`. The cross-fitted row records the
committed hash, so the shared property module was edited between this row's property run and
that one, and this row's cells were not run again afterwards. No test gates a study module
hash, and the manifest is left recording the bytes that ran rather than the bytes that are
committed now.

What that leaves unverified is only the module's own source. Every published verdict on this
row is recomputed in the fast tier from the committed replication rows, and the budgets and
cells those rows carry are the ones the committed module declares. The intermediate revision
is not in the history, so the difference itself cannot be read back. Regenerate this row's
property artifacts to remove the discrepancy.

Run a disposable smoke study first:

```bash
python tests/canonical/categorical_ltmle/regenerate.py --replicates 2 --n 500 --skip-properties --output .tmp/categorical-ltmle-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/categorical_ltmle/regenerate.py
python -m tests.studies.evidence.document --slug canonical-categorical-ltmle
```
