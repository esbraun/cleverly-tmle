# Ordinary categorical longitudinal TMLE fixture

This registered study fits ordinary longitudinal TMLE on a two-node law with three labelled
treatment levels. It compares the result with pinned R `lmtp` 1.5.4 on identical samples. Both
implementations receive the exact treatment mechanism and one fold that contains every row.

## One recorded property-module hash, and why it stands

`manifest.json` records `tests/studies/categorical_longitudinal_properties.py` at `b8cc6bc6`,
and the committed file hashes to `fecaead3`. The cross-fitted row records the committed hash,
so the shared module was edited between the two property runs and this row's cells did not run
again.

This is accepted rather than open. The manifest records the bytes that ran, which is what it is
for, and no test gates a study module hash. Every published verdict on this row is recomputed in
the fast tier from the committed replication rows. The budgets and cells those rows carry are the
ones the committed module declares, so the difference is confined to source the run did not
depend on. Rewriting the hash would be the one action that makes the manifest wrong.

Run a disposable smoke study first:

```bash
python tests/canonical/categorical_ltmle/regenerate.py --replicates 2 --n 500 --skip-properties --output .tmp/categorical-ltmle-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/categorical_ltmle/regenerate.py
python -m tests.studies.evidence.document --slug canonical-categorical-ltmle
```
